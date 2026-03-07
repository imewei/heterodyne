"""JIT-compiled fitting strategy for medium datasets.

Uses JAX's ``jax.jit`` to pre-compile the residual and Jacobian callables
before handing them to ``scipy.optimize.least_squares``.  Pre-compilation
(triggered by a warmup evaluation at the initial parameter point) eliminates
per-iteration trace overhead and yields substantially lower wall-clock time
relative to :class:`ResidualStrategy` for datasets in the 10 k – 250 k
range.

When to use
-----------
- Medium datasets (10 k – 250 k data points).
- GPU/accelerator available: JIT dispatch routes to the active XLA backend.
- Repeated fits with the same array shapes (compilation is cached per shape).

Compilation cache
-----------------
JAX caches compiled XLA executables keyed on input *shape* and *dtype*.
After the first warmup call subsequent calls reuse the cached executable at
negligible overhead.  The cache key is logged at DEBUG level so users can
diagnose unexpected recompilation.

Fallback behaviour
------------------
If JIT compilation or the warmup evaluation raises an exception the strategy
transparently falls back to non-JIT residual evaluation via
:class:`ResidualStrategy`, logging a warning so the user is aware of
degraded performance.

Timing diagnostics
------------------
The :class:`JITStrategy` records separate ``compile_time_s`` and
``execution_time_s`` fields in ``StrategyResult.metadata`` so that profiling
scripts can distinguish compilation overhead from optimisation runtime.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_residuals,
    compute_residuals_jacobian,
)
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)

_BYTES_PER_FLOAT64 = 8


# ---------------------------------------------------------------------------
# Covariance helper
# ---------------------------------------------------------------------------


def _estimate_covariance_from_jac(
    jac: np.ndarray | None,
    residuals: np.ndarray,
    n_params: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate parameter covariance from the final Jacobian.

    Args:
        jac: Jacobian matrix ``(n_residuals, n_params)`` or ``None``.
        residuals: Final residual vector.
        n_params: Number of varying parameters.

    Returns:
        ``(covariance, uncertainties)``; either may be ``None`` on failure.
    """
    if jac is None or jac.size == 0:
        return None, None

    n_residuals = len(residuals)
    n_dof = max(n_residuals - n_params, 1)
    s2 = float(np.dot(residuals, residuals)) / n_dof

    JtJ = jac.T @ jac
    try:
        cov = np.linalg.inv(JtJ) * s2
    except np.linalg.LinAlgError:
        try:
            cov = np.linalg.pinv(JtJ) * s2
            logger.warning(
                "JITStrategy: singular J^T J — used pinv fallback for covariance"
            )
        except np.linalg.LinAlgError:
            logger.warning("JITStrategy: could not compute covariance (pinv failed)")
            return None, None

    diag = np.diag(cov)
    if np.any(diag < 0):
        logger.warning(
            "JITStrategy: negative diagonal in covariance — "
            "uncertainties may be unreliable"
        )
        return cov, None

    return cov, np.sqrt(diag)


# ---------------------------------------------------------------------------
# Compilation cache key
# ---------------------------------------------------------------------------


def _make_cache_key(
    t: jnp.ndarray,
    c2_jax: jnp.ndarray,
    n_params: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, str]:
    """Build a hashable cache key for JIT compilation.

    The key captures the array shapes and element types that determine
    whether the existing compiled executable can be reused.

    Args:
        t: Time array.
        c2_jax: Correlation data array.
        n_params: Number of varying parameters.

    Returns:
        Tuple ``(t_shape, c2_shape, n_params, dtype_str)`` suitable as a
        dict key.
    """
    return (
        tuple(t.shape),
        tuple(c2_jax.shape),
        n_params,
        str(c2_jax.dtype),
    )


# ---------------------------------------------------------------------------
# JITStrategy
# ---------------------------------------------------------------------------


class JITStrategy:
    """JAX JIT-compiled residual and Jacobian for medium datasets.

    Pre-compiles the residual (and optionally Jacobian) callables via
    ``jax.jit`` during a warmup step before passing them to
    ``scipy.optimize.least_squares``.  Compiled executables are cached by
    the instance so that repeated calls with the same array shapes incur
    compilation cost only once.

    Args:
        use_analytic_jac: Supply analytic Jacobian to scipy.  When ``None``
            (default) the value is taken from ``config.use_jac``.
        jit_residual: JIT-compile the residual callable.  Default ``True``.
        jit_jacobian: JIT-compile the Jacobian callable.  Default ``True``.

    Example::

        strategy = JITStrategy()
        sr = strategy.fit(model, c2_data, phi_angle=0.0, config=config)
        print(sr.metadata["compile_time_s"])
    """

    def __init__(
        self,
        use_analytic_jac: bool | None = None,
        jit_residual: bool = True,
        jit_jacobian: bool = True,
    ) -> None:
        self._use_analytic_jac = use_analytic_jac
        self._jit_residual = jit_residual
        self._jit_jacobian = jit_jacobian

        # Compilation cache: cache_key → (jit_residual_fn, jit_jacobian_fn)
        self._compiled: dict[
            tuple[Any, ...],
            tuple[Any, Any],  # (residual_callable, jacobian_callable | None)
        ] = {}

    @property
    def name(self) -> str:
        return "jit"

    # ------------------------------------------------------------------
    # Public fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Fit using JIT-compiled residual and Jacobian callables.

        On the first call for a given array shape the JAX functions are
        traced and compiled (warmup).  Subsequent calls with the same shapes
        reuse the cached executable.

        Falls back to :class:`ResidualStrategy` (non-JIT) on any
        compilation or warmup failure.

        Args:
            model: Configured heterodyne model.
            c2_data: Experimental correlation matrix, shape ``(N, N)``.
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights ``(1/σ²)``.

        Returns:
            :class:`~heterodyne.optimization.nlsq.strategies.base.StrategyResult`.
        """
        wall_start = time.perf_counter()

        pm = model.param_manager
        initial = np.asarray(pm.get_initial_values(), dtype=np.float64)
        lower, upper = pm.get_bounds()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        initial = np.clip(initial, lower, upper)

        n_params = len(initial)
        n_data = int(np.asarray(c2_data).size)

        logger.info(
            "JITStrategy: %d params, %d data points (phi=%.2f°)",
            n_params,
            n_data,
            phi_angle,
        )

        # Pre-convert to JAX once
        c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
        weights_jax: jnp.ndarray | None = (
            jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        )

        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)
        varying_idx_np = np.asarray(varying_idx)

        use_jac = (
            self._use_analytic_jac
            if self._use_analytic_jac is not None
            else config.use_jac
        )

        # ------------------------------------------------------------------
        # Compile (or retrieve from cache)
        # ------------------------------------------------------------------

        cache_key = _make_cache_key(t, c2_jax, n_params)
        compile_time_s = 0.0
        used_jit = False

        try:
            jit_res_fn, jit_jac_fn = self._get_or_compile(
                cache_key=cache_key,
                fixed_values=fixed_values,
                varying_idx=varying_idx,
                t=t,
                q=q,
                dt=dt,
                phi_angle=phi_angle,
                c2_jax=c2_jax,
                weights_jax=weights_jax,
                initial=initial,
                use_jac=use_jac,
            )
            compile_time_s = self._last_compile_time
            used_jit = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JITStrategy: compilation/warmup failed (%s); "
                "falling back to non-JIT ResidualStrategy.",
                exc,
            )
            return self._fallback(
                model=model,
                c2_data=c2_data,
                phi_angle=phi_angle,
                config=config,
                weights=weights,
                fallback_reason=str(exc),
            )

        # ------------------------------------------------------------------
        # Build numpy-returning wrappers for scipy
        # ------------------------------------------------------------------

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            return np.asarray(
                jit_res_fn(jnp.asarray(varying, dtype=jnp.float64)),
                dtype=np.float64,
            )

        def jacobian_fn(varying: np.ndarray) -> np.ndarray:
            if jit_jac_fn is None:
                raise RuntimeError("Jacobian function not compiled")
            J_full = np.asarray(
                jit_jac_fn(jnp.asarray(varying, dtype=jnp.float64)),
                dtype=np.float64,
            )
            # Select columns corresponding to varying parameters
            return J_full[:, varying_idx_np]

        # ------------------------------------------------------------------
        # Run scipy
        # ------------------------------------------------------------------

        method = config.method if config.method != "lm" else "trf"
        max_nfev = config.max_nfev or config.max_iterations * (n_params + 1) * 10

        logger.debug(
            "JITStrategy: method=%s, use_analytic_jac=%s, max_nfev=%d",
            method,
            use_jac and jit_jac_fn is not None,
            max_nfev,
        )

        exec_start = time.perf_counter()
        scipy_result = least_squares(
            residual_fn,
            initial,
            jac=jacobian_fn if (use_jac and jit_jac_fn is not None) else "2-point",
            bounds=(lower, upper),
            method=method,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            max_nfev=max_nfev,
            loss=config.loss,
            verbose=max(0, config.verbose - 1),
        )
        execution_time_s = time.perf_counter() - exec_start
        wall_time = time.perf_counter() - wall_start

        logger.info(
            "JITStrategy: %s | cost=%.4e | nfev=%d | compile=%.2f s | exec=%.2f s",
            "converged" if scipy_result.success else "did not converge",
            scipy_result.cost,
            scipy_result.nfev,
            compile_time_s,
            execution_time_s,
        )

        # ------------------------------------------------------------------
        # Post-fit covariance
        # ------------------------------------------------------------------

        covariance, uncertainties = _estimate_covariance_from_jac(
            scipy_result.jac,
            scipy_result.fun,
            n_params,
        )

        n_dof = max(n_data - n_params, 1)
        reduced_chi2 = 2.0 * float(scipy_result.cost) / n_dof

        # ------------------------------------------------------------------
        # Pack result
        # ------------------------------------------------------------------

        metadata: dict[str, Any] = {
            "strategy": "jit",
            "use_analytic_jac": use_jac and jit_jac_fn is not None,
            "used_jit": used_jit,
            "compile_time_s": compile_time_s,
            "execution_time_s": execution_time_s,
            "phi_angle": phi_angle,
            "cache_key": str(cache_key),
        }

        result = NLSQResult(
            parameters=scipy_result.x,
            parameter_names=pm.varying_names,
            success=scipy_result.success,
            message=scipy_result.message,
            uncertainties=uncertainties,
            covariance=covariance,
            final_cost=float(scipy_result.cost),
            reduced_chi_squared=reduced_chi2,
            n_iterations=0,
            n_function_evals=scipy_result.nfev,
            convergence_reason=scipy_result.message,
            residuals=scipy_result.fun,
            jacobian=scipy_result.jac,
            wall_time_seconds=wall_time,
            metadata=metadata,
        )

        # Compute fitted correlation and update model on success
        if scipy_result.success:
            full_fitted = pm.expand_varying_to_full(result.parameters)
            fitted_c2 = compute_c2_heterodyne(
                jnp.asarray(full_fitted), t, q, dt, phi_angle
            )
            result.fitted_correlation = np.asarray(fitted_c2)
            model.set_params(full_fitted)

        jacobian_mb = n_data * n_params * _BYTES_PER_FLOAT64 / (1024 * 1024)
        peak_memory_mb = jacobian_mb + np.asarray(c2_data).nbytes / (1024 * 1024)

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            n_chunks=1,
            peak_memory_mb=peak_memory_mb,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Compilation helpers
    # ------------------------------------------------------------------

    def _get_or_compile(
        self,
        cache_key: tuple[Any, ...],
        fixed_values: jnp.ndarray,
        varying_idx: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi_angle: float,
        c2_jax: jnp.ndarray,
        weights_jax: jnp.ndarray | None,
        initial: np.ndarray,
        use_jac: bool,
    ) -> tuple[Any, Any]:
        """Return compiled (residual_fn, jacobian_fn) from cache or compile fresh.

        On cache miss the callables are JIT-compiled and warmed up at
        ``initial``.  ``self._last_compile_time`` is set to the warmup
        wall-clock time on cache miss and 0.0 on cache hit.

        Args:
            cache_key: Hashable key for the compilation cache.
            fixed_values: Full parameter vector (JAX).
            varying_idx: Indices of varying parameters (JAX).
            t: Time array.
            q: Scattering wavevector.
            dt: Time step.
            phi_angle: Detector phi angle.
            c2_jax: Correlation data (JAX).
            weights_jax: Optional weights (JAX).
            initial: Initial parameter values (numpy).
            use_jac: Whether to compile a Jacobian function.

        Returns:
            ``(jit_residual_fn, jit_jacobian_fn)`` where the Jacobian
            function may be ``None`` if ``use_jac=False``.
        """
        if cache_key in self._compiled:
            logger.debug("JITStrategy: cache hit for key %s", cache_key)
            self._last_compile_time = 0.0
            return self._compiled[cache_key]

        logger.debug("JITStrategy: cache miss — compiling for key %s", cache_key)
        compile_start = time.perf_counter()

        # Build JAX closures (captures static arrays by reference)
        def _full_params(varying: jnp.ndarray) -> jnp.ndarray:
            return fixed_values.at[varying_idx].set(varying)

        def _residual_jax(varying: jnp.ndarray) -> jnp.ndarray:
            return compute_residuals(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )

        def _jacobian_jax(varying: jnp.ndarray) -> jnp.ndarray:
            return compute_residuals_jacobian(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )

        jit_res = jax.jit(_residual_jax) if self._jit_residual else _residual_jax
        jit_jac: Any = None
        if use_jac:
            jit_jac = jax.jit(_jacobian_jax) if self._jit_jacobian else _jacobian_jax

        # Warmup: trigger XLA compilation by running once at the initial point
        initial_jax = jnp.asarray(initial, dtype=jnp.float64)
        _ = jit_res(initial_jax).block_until_ready()
        if jit_jac is not None:
            _ = jit_jac(initial_jax).block_until_ready()

        self._last_compile_time = time.perf_counter() - compile_start
        logger.info(
            "JITStrategy: compiled and warmed up in %.2f s (cache_key=%s)",
            self._last_compile_time,
            cache_key,
        )

        self._compiled[cache_key] = (jit_res, jit_jac)
        return jit_res, jit_jac

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None,
        fallback_reason: str,
    ) -> StrategyResult:
        """Fall back to :class:`ResidualStrategy` on JIT failure.

        Args:
            model: Heterodyne model.
            c2_data: Correlation data.
            phi_angle: Phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional weights.
            fallback_reason: Exception message from the failed compilation.

        Returns:
            :class:`StrategyResult` with ``metadata["fallback"] = True``.
        """
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        sr = ResidualStrategy(use_analytic_jac=False).fit(
            model, c2_data, phi_angle, config, weights
        )
        sr.metadata["fallback"] = True
        sr.metadata["fallback_reason"] = fallback_reason
        sr.metadata["original_strategy"] = "jit"
        return sr

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the JIT compilation cache.

        Call this when problem dimensions change and you want to force
        recompilation (e.g. between datasets with different time-grid sizes).
        """
        n = len(self._compiled)
        self._compiled.clear()
        logger.info("JITStrategy: cleared %d cached compilation(s)", n)

    @property
    def n_cached(self) -> int:
        """Number of compiled executables currently in cache."""
        return len(self._compiled)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"JITStrategy(use_analytic_jac={self._use_analytic_jac!r}, "
            f"n_cached={self.n_cached})"
        )
