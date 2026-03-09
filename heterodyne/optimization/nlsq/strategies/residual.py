"""Direct residual evaluation strategy for small datasets.

This strategy directly calls ``compute_residuals`` and
``compute_residuals_jacobian`` from the JAX backend and hands the results to
``nlsq.CurveFit``.  It is the simplest and most transparent strategy: no
chunking, no JIT warm-up, no padding.

When to use
-----------
- Small datasets (< 10 k data points).
- Debugging / validation: single-pass execution with full Jacobian.
- Baseline comparison for regression tests.

For larger datasets prefer :class:`JITStrategy` (JAX-compiled Jacobian via
the ``nlsq`` library) or :class:`ChunkedStrategy` (memory-bounded residual
evaluation via ``scipy``).

Weight handling
---------------
Weights are passed as ``1/σ²`` values.  When ``weights`` is ``None`` the
backend treats all residuals as equally weighted.  The strategy accepts both
diagonal (same shape as ``c2_data``) and scalar weight arrays.

Covariance estimation
---------------------
After convergence the covariance is estimated analytically:

    Cov = s² × (J^T J)⁻¹,   s² = ‖r‖² / (n − p)

where ``n`` is the number of residuals and ``p`` the number of free
parameters.  When J^T J is numerically singular ``numpy.linalg.pinv`` is
used as a fallback and a warning is emitted.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from nlsq import CurveFit

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
# Covariance helper (shared with JITStrategy)
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
                "ResidualStrategy: singular J^T J — used pinv fallback for covariance"
            )
        except np.linalg.LinAlgError:
            logger.warning(
                "ResidualStrategy: could not compute covariance (pinv also failed)"
            )
            return None, None

    diag = np.diag(cov)
    if np.any(diag < 0):
        logger.warning(
            "ResidualStrategy: negative diagonal in covariance — "
            "uncertainties may be unreliable"
        )
        return cov, None

    return cov, np.sqrt(diag)


# ---------------------------------------------------------------------------
# ResidualStrategy
# ---------------------------------------------------------------------------


class ResidualStrategy:
    """Direct residual evaluation without chunking or JIT warm-up.

    Calls ``compute_residuals`` once per scipy iteration.  When
    ``config.use_jac`` is ``True`` the analytic Jacobian from
    ``compute_residuals_jacobian`` is supplied to the solver, which
    eliminates finite-difference perturbations and is the recommended
    setting.

    The strategy is intentionally lightweight so that it is suitable both
    as a production path for small datasets and as a transparent reference
    implementation for testing.

    Args:
        use_analytic_jac: Override for analytic-Jacobian usage.  When
            ``None`` (default) the value is taken from ``config.use_jac``.

    Example::

        strategy = ResidualStrategy()
        sr = strategy.fit(model, c2_data, phi_angle=0.0, config=config)
        print(sr.result.reduced_chi_squared)
    """

    def __init__(self, use_analytic_jac: bool | None = None) -> None:
        self._use_analytic_jac = use_analytic_jac

    @property
    def name(self) -> str:
        return "residual"

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
        """Fit using direct residual (and optionally Jacobian) evaluation.

        Builds pure-JAX residual and Jacobian callables, then delegates
        the minimisation to ``nlsq.CurveFit``.

        Args:
            model: Configured heterodyne model.
            c2_data: Experimental correlation matrix, shape ``(N, N)``.
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights ``(1/σ²)`` matching the
                shape of ``c2_data``.

        Returns:
            :class:`~heterodyne.optimization.nlsq.strategies.base.StrategyResult`.
        """
        start_time = time.perf_counter()

        pm = model.param_manager
        initial = np.asarray(pm.get_initial_values(), dtype=np.float64)
        lower, upper = pm.get_bounds()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        initial = np.clip(initial, lower, upper)

        n_params = len(initial)
        n_data = int(np.asarray(c2_data).size)

        logger.info(
            "ResidualStrategy: fitting %d params over %d data points (phi=%.2f°)",
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

        # ------------------------------------------------------------------
        # Callable wrappers
        # ------------------------------------------------------------------

        def _full_params(varying: np.ndarray) -> jnp.ndarray:
            return fixed_values.at[varying_idx].set(
                jnp.asarray(varying, dtype=jnp.float64)
            )

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            r = compute_residuals(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )
            return np.asarray(r, dtype=np.float64)

        def jacobian_fn(varying: np.ndarray) -> np.ndarray:
            J_full = compute_residuals_jacobian(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )
            # J_full has shape (n_residuals, 14); select columns for varying params
            return np.asarray(J_full, dtype=np.float64)[:, np.asarray(varying_idx)]

        # ------------------------------------------------------------------
        # Decide whether to supply analytic Jacobian
        # ------------------------------------------------------------------

        use_jac = (
            self._use_analytic_jac
            if self._use_analytic_jac is not None
            else config.use_jac
        )

        method = config.method if config.method != "lm" else "trf"
        if method == "dogbox":
            logger.warning(
                "ResidualStrategy: 'dogbox' is not supported by CurveFit; "
                "coercing to 'trf'."
            )
            method = "trf"
        max_nfev = config.max_nfev or config.max_iterations * (n_params + 1) * 10

        logger.debug(
            "ResidualStrategy: method=%s, use_analytic_jac=%s, max_nfev=%d",
            method,
            use_jac,
            max_nfev,
        )

        # ------------------------------------------------------------------
        # Run nlsq CurveFit
        # ------------------------------------------------------------------

        # CurveFit expects f(xdata, *params) -> prediction, with
        # residuals computed as ydata - f(xdata, *params).
        # We set ydata=zeros so that: residuals = 0 - f(xdata, *params)
        # Therefore f must return the *negative* of residual_fn.
        # params arrive as individual JAX-traced scalars; jnp.stack reassembles.

        def _wrapped(xdata: np.ndarray, *params: object) -> jnp.ndarray:
            return -residual_fn(np.asarray(jnp.stack(list(params))))  # type: ignore[arg-type]

        _xdata = np.arange(n_data, dtype=np.float64)
        _ydata = np.zeros(n_data, dtype=np.float64)

        jac_callable = None
        if use_jac:
            # CurveFit jac must share the same (xdata, *params) signature as f.
            # Negate Jacobian to match the negated residual convention.
            def _jac_wrapped(xdata: np.ndarray, *params: object) -> np.ndarray:
                return -jacobian_fn(np.asarray(jnp.stack(list(params))))  # type: ignore[arg-type]
            jac_callable = _jac_wrapped

        fitter = CurveFit(flength=n_data)
        nlsq_result = fitter.curve_fit(
            f=_wrapped,
            xdata=_xdata,
            ydata=_ydata,
            p0=initial,
            bounds=(lower, upper),
            method=method,
            jac=jac_callable,
        )

        wall_time = time.perf_counter() - start_time

        logger.info(
            "ResidualStrategy: %s | cost=%.4e | nfev=%d | %.2f s",
            "converged" if nlsq_result.success else "did not converge",
            nlsq_result.cost,
            nlsq_result.nfev,
            wall_time,
        )

        # ------------------------------------------------------------------
        # Post-fit covariance
        # ------------------------------------------------------------------

        # Recompute residuals at the fitted point for covariance estimation.
        # nlsq returns residuals as (ydata - f(xdata, *popt)); negate back to
        # match the residual_fn convention (model - data).
        final_residuals = residual_fn(np.asarray(nlsq_result.x, dtype=np.float64))
        final_jac = np.asarray(nlsq_result.jac, dtype=np.float64) if nlsq_result.jac is not None else None

        covariance, uncertainties = _estimate_covariance_from_jac(
            final_jac,
            final_residuals,
            n_params,
        )

        n_dof = max(n_data - n_params, 1)
        final_cost = 0.5 * float(np.sum(final_residuals**2))
        reduced_chi2 = 2.0 * final_cost / n_dof

        # ------------------------------------------------------------------
        # Pack NLSQResult
        # ------------------------------------------------------------------

        metadata: dict[str, Any] = {
            "strategy": "residual",
            "use_analytic_jac": use_jac,
            "phi_angle": phi_angle,
        }

        result = NLSQResult(
            parameters=np.asarray(nlsq_result.x, dtype=np.float64),
            parameter_names=pm.varying_names,
            success=bool(nlsq_result.success),
            message=str(nlsq_result.message),
            uncertainties=uncertainties,
            covariance=covariance,
            final_cost=final_cost,
            reduced_chi_squared=reduced_chi2,
            n_iterations=0,
            n_function_evals=int(nlsq_result.nfev),
            convergence_reason=str(nlsq_result.message),
            residuals=final_residuals,
            jacobian=final_jac,
            wall_time_seconds=wall_time,
            metadata=metadata,
        )

        # Compute fitted correlation and update model on success
        if nlsq_result.success:
            full_fitted = pm.expand_varying_to_full(result.parameters)
            fitted_c2 = compute_c2_heterodyne(
                jnp.asarray(full_fitted), t, q, dt, phi_angle
            )
            result.fitted_correlation = np.asarray(fitted_c2)
            model.set_params(full_fitted)

        peak_memory_mb = (
            n_data * n_params * _BYTES_PER_FLOAT64 / (1024 * 1024)
            + np.asarray(c2_data).nbytes / (1024 * 1024)
        )

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            n_chunks=1,
            peak_memory_mb=peak_memory_mb,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ResidualStrategy(use_analytic_jac={self._use_analytic_jac!r})"
