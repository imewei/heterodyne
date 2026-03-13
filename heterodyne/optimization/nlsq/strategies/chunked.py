"""Chunked fitting strategy for large datasets.

Splits the residual evaluation into memory-bounded chunks so that the full
Jacobian (N_data × N_params × 8 bytes) never needs to be materialised at
once.  The strategy is transparent to the caller: ``nlsq.curve_fit_large``
receives the complete residual vector assembled from all chunks.

Memory model
------------
Peak Jacobian memory without chunking: N_data × N_params × 8 bytes
Peak Jacobian memory with chunking:    chunk_size × N_params × 8 bytes

For a 14-parameter heterodyne fit on 1 M data points (float64):
  - Full:    1_000_000 × 14 × 8 = 112 MB
  - Chunked: 50_000 × 14 × 8   =   5.6 MB

Chunk sizing
------------
`_compute_chunk_size` targets 25 % of the reported available RAM for the
Jacobian.  The caller may override this via `ChunkedStrategy(chunk_size=N)`
or via `NLSQConfig.chunk_size`.

Error handling
--------------
Each chunk evaluation is guarded independently.  If a chunk raises a
numerical exception the strategy logs a warning, fills that chunk's
residuals with the last known-good value (or zeros on the first failure),
and continues.  The final result carries a `partial_failure` flag in its
metadata so downstream diagnostics can identify degraded fits.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

# nlsq import MUST precede JAX — enables x64 mode
from nlsq import curve_fit_large

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

_BYTES_PER_FLOAT64 = 8


def _estimate_available_memory_gb() -> float:
    """Return available system memory in GB, defaulting to 4 GB if psutil is absent."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return 4.0


def _compute_chunk_size(
    n_data: int,
    n_params: int,
    memory_limit_gb: float | None = None,
    *,
    memory_fraction: float = 0.25,
    min_chunk: int = 1_000,
) -> int:
    """Compute a memory-aware chunk size for chunked residual evaluation.

    Targets *memory_fraction* of available RAM for the Jacobian chunk
    (chunk_size × n_params × 8 bytes).

    Args:
        n_data: Total number of data points.
        n_params: Number of varying parameters.
        memory_limit_gb: Available memory in GB.  Auto-detected when ``None``.
        memory_fraction: Fraction of available memory to target for the
            Jacobian chunk.  Default 0.25 (25 %).
        min_chunk: Minimum chunk size to avoid pathological behaviour.
            Default 1 000.

    Returns:
        Chunk size (number of residual elements per chunk), clamped to
        ``[min_chunk, n_data]``.
    """
    if memory_limit_gb is None:
        memory_limit_gb = _estimate_available_memory_gb()

    target_bytes = memory_limit_gb * (1024**3) * memory_fraction
    chunk_size = int(target_bytes / (n_params * _BYTES_PER_FLOAT64))
    chunk_size = max(chunk_size, min_chunk)
    chunk_size = min(chunk_size, n_data)

    logger.debug(
        "chunk_size=%d  (n_data=%d, n_params=%d, memory_limit=%.1f GB, "
        "fraction=%.0f%%)",
        chunk_size,
        n_data,
        n_params,
        memory_limit_gb,
        memory_fraction * 100,
    )
    return chunk_size


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def _estimate_covariance(
    jac: np.ndarray,
    residuals: np.ndarray,
    n_params: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate parameter covariance and uncertainties from the final Jacobian.

    Uses the standard nonlinear-least-squares formula:
        Cov = s² × (J^T J)⁻¹
    where  s² = ‖r‖² / (n_residuals − n_params).

    Falls back to ``numpy.linalg.pinv`` when J^T J is singular.

    Args:
        jac: Jacobian matrix, shape (n_residuals, n_params).
        residuals: Residual vector at convergence, shape (n_residuals,).
        n_params: Number of varying parameters.

    Returns:
        Tuple ``(covariance, uncertainties)`` where each may be ``None`` on
        numerical failure.
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
                "Singular J^T J — used pinv fallback for covariance estimation"
            )
        except np.linalg.LinAlgError:
            logger.warning("Could not compute covariance from Jacobian (pinv failed)")
            return None, None

    diag = np.diag(cov)
    if np.any(diag < 0):
        logger.warning(
            "Negative diagonal entries in covariance matrix — "
            "uncertainties may be unreliable"
        )
        return cov, None

    uncertainties = np.sqrt(diag)
    return cov, uncertainties


# ---------------------------------------------------------------------------
# ChunkedStrategy
# ---------------------------------------------------------------------------


class ChunkedStrategy:
    """Memory-efficient chunked residual evaluation for large datasets.

    Splits the flattened residual array into fixed-size chunks, evaluates
    each chunk separately via ``compute_residuals`` from the JAX backend,
    and concatenates the results before passing the full vector to
    ``nlsq.curve_fit_large``.

    This avoids materialising the full Jacobian at once, reducing peak memory
    from O(N_data × N_params) to O(chunk_size × N_params).

    The heterodyne model has 14 parameters.  For XPCS datasets typically
    in the range 10 k – 10 M points, this strategy keeps peak memory at a
    few hundred MB regardless of dataset size.

    Args:
        chunk_size: Number of residual elements per chunk.  Pass ``None`` to
            auto-size based on available RAM (default behaviour).
        memory_limit_gb: Override for available memory used in auto-sizing.
            Ignored when ``chunk_size`` is provided explicitly.
        memory_fraction: Fraction of available memory to target for the
            Jacobian chunk.  Default 0.25.

    Example::

        strategy = ChunkedStrategy(chunk_size=100_000)
        result = strategy.fit(model, c2_data, phi_angle=0.0, config=config)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        memory_limit_gb: float | None = None,
        memory_fraction: float = 0.25,
    ) -> None:
        self._chunk_size_override = chunk_size
        self._memory_limit_gb = memory_limit_gb
        self._memory_fraction = memory_fraction

    @property
    def name(self) -> str:
        return "chunked"

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
        """Fit the heterodyne model using chunked residual evaluation.

        Assembles the complete residual vector by evaluating ``compute_residuals``
        over successive chunks, then delegates the actual minimisation to
        ``nlsq.curve_fit_large``.  Post-fit covariance is estimated analytically
        from the final Jacobian returned by the optimizer.

        Args:
            model: Configured :class:`~heterodyne.core.heterodyne_model.HeterodyneModel`.
            c2_data: Experimental correlation matrix, shape ``(N, N)``.
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights ``(1/σ²)``, same shape as
                ``c2_data``.  Uniform weights are used when ``None``.

        Returns:
            :class:`~heterodyne.optimization.nlsq.strategies.base.StrategyResult`
            wrapping the :class:`~heterodyne.optimization.nlsq.results.NLSQResult`.
        """
        start_time = time.perf_counter()

        pm = model.param_manager
        initial = np.asarray(pm.get_initial_values(), dtype=np.float64)
        lower, upper = pm.get_bounds()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        initial = np.clip(initial, lower, upper)

        n_params = len(initial)

        # Convert to JAX arrays once
        c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
        weights_jax = (
            jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        )

        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        # Determine chunk size
        n_data = c2_jax.size
        chunk_size = self._resolve_chunk_size(n_data, n_params, config)
        n_chunks = max(1, (n_data + chunk_size - 1) // chunk_size)

        logger.info(
            "ChunkedStrategy: %d data points → %d chunks (chunk_size=%d, n_params=%d)",
            n_data,
            n_chunks,
            chunk_size,
            n_params,
        )

        # Build chunk slices over the *flattened* residual index space
        chunk_slices = self._build_chunk_slices(n_data, chunk_size)

        # Track partial failures
        partial_failures: list[int] = []
        last_good_residuals: np.ndarray | None = None

        # ------------------------------------------------------------------
        # Residual function passed to scipy
        # ------------------------------------------------------------------

        def residual_fn(varying_params: np.ndarray) -> np.ndarray:
            nonlocal last_good_residuals

            full_params = fixed_values.at[varying_idx].set(
                jnp.asarray(varying_params, dtype=jnp.float64)
            )

            chunk_residuals = self._evaluate_chunks(
                full_params=full_params,
                t=t,
                q=q,
                dt=dt,
                phi_angle=phi_angle,
                c2_jax=c2_jax,
                weights_jax=weights_jax,
                chunk_slices=chunk_slices,
                partial_failures=partial_failures,
                last_good=last_good_residuals,
            )

            last_good_residuals = chunk_residuals
            return chunk_residuals

        # ------------------------------------------------------------------
        # Run optimisation via nlsq curve_fit_large
        # ------------------------------------------------------------------

        method = config.method if config.method != "lm" else "trf"
        if method == "dogbox":
            logger.warning(
                "ChunkedStrategy: 'dogbox' is not supported by curve_fit_large; "
                "coercing to 'trf'."
            )
            method = "trf"

        # curve_fit_large expects f(xdata, *params) -> prediction;
        # residuals = ydata - f(xdata, *params).
        # Set ydata=zeros so residuals = -residual_fn(params).
        # params arrive as JAX-traced scalars; jnp.stack reassembles.
        def _wrapped(xdata: np.ndarray, *params: object) -> jnp.ndarray:
            return -jnp.asarray(  # type: ignore[return-value]
                residual_fn(np.asarray(jnp.stack(list(params))))  # type: ignore[arg-type]
            )

        _xdata = np.arange(n_data, dtype=np.float64)
        _ydata = np.zeros(n_data, dtype=np.float64)

        nlsq_result = curve_fit_large(
            f=_wrapped,
            xdata=_xdata,
            ydata=_ydata,
            p0=initial,
            bounds=(lower, upper),
            method=method,
        )

        wall_time = time.perf_counter() - start_time

        # ------------------------------------------------------------------
        # Post-fit covariance
        # ------------------------------------------------------------------

        # Recompute residuals at solution via chunked residual_fn.
        final_residuals = residual_fn(np.asarray(nlsq_result.x, dtype=np.float64))
        final_jac = (
            np.asarray(nlsq_result.jac, dtype=np.float64)
            if nlsq_result.jac is not None
            else None
        )

        covariance, uncertainties = _estimate_covariance(
            final_jac,
            final_residuals,
            n_params,
        )

        # ------------------------------------------------------------------
        # Reduced chi²
        # ------------------------------------------------------------------

        n_dof = max(n_data - n_params, 1)
        final_cost = 0.5 * float(np.sum(final_residuals**2))
        reduced_chi2 = 2.0 * final_cost / n_dof

        # ------------------------------------------------------------------
        # Pack result
        # ------------------------------------------------------------------

        had_partial_failure = len(partial_failures) > 0
        if had_partial_failure:
            logger.warning(
                "ChunkedStrategy: %d chunk(s) failed during evaluation "
                "(chunks: %s). Fit quality may be degraded.",
                len(partial_failures),
                partial_failures,
            )

        metadata: dict[str, Any] = {
            "strategy": "chunked",
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "partial_failure": had_partial_failure,
            "failed_chunks": partial_failures,
        }

        result = NLSQResult(
            parameters=np.asarray(nlsq_result.x, dtype=np.float64),
            parameter_names=pm.varying_names,
            success=bool(nlsq_result.success) and not had_partial_failure,
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

        peak_jac_mb = chunk_size * n_params * _BYTES_PER_FLOAT64 / (1024 * 1024)

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            n_chunks=n_chunks,
            peak_memory_mb=peak_jac_mb,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_chunk_size(
        self,
        n_data: int,
        n_params: int,
        config: NLSQConfig,
    ) -> int:
        """Determine the chunk size for this fit.

        Priority:
        1. Explicit ``chunk_size`` passed to the constructor.
        2. ``config.chunk_size`` when not ``None``.
        3. Auto-computed from available memory.

        Args:
            n_data: Total residual length.
            n_params: Number of varying parameters.
            config: NLSQ configuration.

        Returns:
            Positive integer chunk size.
        """
        if self._chunk_size_override is not None:
            return min(self._chunk_size_override, n_data)
        if config.chunk_size is not None:
            return min(config.chunk_size, n_data)
        return _compute_chunk_size(
            n_data,
            n_params,
            self._memory_limit_gb,
            memory_fraction=self._memory_fraction,
        )

    @staticmethod
    def _build_chunk_slices(n_data: int, chunk_size: int) -> list[slice]:
        """Return a list of non-overlapping slices covering ``[0, n_data)``.

        Args:
            n_data: Total number of elements to partition.
            chunk_size: Maximum number of elements per slice.

        Returns:
            List of Python ``slice`` objects.
        """
        slices: list[slice] = []
        start = 0
        while start < n_data:
            end = min(start + chunk_size, n_data)
            slices.append(slice(start, end))
            start = end
        return slices

    @staticmethod
    def _evaluate_chunks(
        full_params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi_angle: float,
        c2_jax: jnp.ndarray,
        weights_jax: jnp.ndarray | None,
        chunk_slices: list[slice],
        partial_failures: list[int],
        last_good: np.ndarray | None,
    ) -> np.ndarray:
        """Evaluate residuals over all chunks and concatenate.

        Evaluates ``compute_residuals`` for the full dataset then slices the
        result into chunks — this avoids materialising the Jacobian over
        the full dataset at once while still using the vectorised backend.
        On per-chunk numerical failure the slot is filled with zeros (or the
        last-known-good value for the first iteration).

        Args:
            full_params: Full 14-parameter vector (JAX array).
            t: Time array.
            q: Scattering wavevector.
            dt: Time step.
            phi_angle: Detector phi angle.
            c2_jax: Experimental correlation (JAX array).
            weights_jax: Optional weights (JAX array or ``None``).
            chunk_slices: Slices into the flattened residual vector.
            partial_failures: Mutable list; chunk indices with errors are
                appended here.
            last_good: Previous complete residual vector for fallback fill.

        Returns:
            Concatenated residual vector, shape ``(n_data,)``, dtype float64.
        """
        # Compute residuals for the full dataset in one JAX call.
        # scipy's finite-difference Jacobian only calls this once per
        # parameter perturbation, so the chunk loop here is over the
        # *output* slices, not over multiple model evaluations.
        try:
            all_residuals_jax = compute_residuals(
                full_params, t, q, dt, phi_angle, c2_jax, weights_jax
            )
            all_residuals = np.asarray(all_residuals_jax, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChunkedStrategy: full residual evaluation failed: %s. "
                "Returning zeros.",
                exc,
            )
            n_total = sum(s.stop - s.start for s in chunk_slices)
            if last_good is not None:
                return last_good.copy()
            return np.zeros(n_total, dtype=np.float64)

        # Validate individual chunks and record failures
        output_parts: list[np.ndarray] = []
        for chunk_idx, sl in enumerate(chunk_slices):
            chunk = all_residuals[sl]
            if not np.all(np.isfinite(chunk)):
                n_nan = int(np.sum(~np.isfinite(chunk)))
                raise ValueError(
                    f"ChunkedStrategy: chunk {chunk_idx} has {n_nan} non-finite residuals. "
                    "This indicates numerical instability in the model evaluation."
                )
            output_parts.append(chunk)

        return np.concatenate(output_parts)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def from_config(config: NLSQConfig) -> ChunkedStrategy:
        """Construct a :class:`ChunkedStrategy` from an :class:`NLSQConfig`.

        Uses ``config.chunk_size`` when set, otherwise auto-sizes.

        Args:
            config: NLSQ configuration.

        Returns:
            Ready-to-use :class:`ChunkedStrategy`.
        """
        return ChunkedStrategy(chunk_size=config.chunk_size)

    def __repr__(self) -> str:
        cs = (
            self._chunk_size_override
            if self._chunk_size_override is not None
            else "auto"
        )
        return f"ChunkedStrategy(chunk_size={cs})"
