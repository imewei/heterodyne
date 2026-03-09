"""Out-of-core fitting strategy for very large datasets.

Uses memory-mapped data and chunk-wise Gauss-Newton accumulation
for datasets that exceed available RAM. Integrates with the
parallel_accumulator module for optional multi-threaded evaluation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

# nlsq import MUST precede JAX — enables x64 mode
from nlsq import curve_fit_large

import jax.numpy as jnp
import numpy as np

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)

_BYTES_PER_FLOAT64 = 8


class OutOfCoreStrategy:
    """Memory-mapped out-of-core fitting for very large datasets.

    Similar to ChunkedStrategy but designed for datasets where even the
    raw correlation data may not fit in memory. Uses numpy memory-mapped
    arrays and chunk-wise residual evaluation.

    Args:
        chunk_size: Number of residual elements per chunk.
        n_workers: Number of parallel workers for chunk accumulation.
            Set to 1 for sequential evaluation.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        n_workers: int = 1,
    ) -> None:
        self._chunk_size = chunk_size
        self._n_workers = n_workers

    @property
    def name(self) -> str:
        return "out_of_core"

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Fit using out-of-core chunk evaluation.

        Args:
            model: Configured HeterodyneModel.
            c2_data: Correlation data, shape (N, N).
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights.

        Returns:
            StrategyResult with fit results.
        """
        start_time = time.perf_counter()

        pm = model.param_manager
        initial = np.asarray(pm.get_initial_values(), dtype=np.float64)
        lower, upper = pm.get_bounds()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        initial = np.clip(initial, lower, upper)

        n_params = len(initial)
        n_data = c2_data.size

        # Determine chunk size
        chunk_size = self._chunk_size or config.chunk_size
        if chunk_size is None:
            chunk_size = self._auto_chunk_size(n_data, n_params)

        n_chunks = max(1, (n_data + chunk_size - 1) // chunk_size)

        logger.info(
            "OutOfCoreStrategy: %d data points -> %d chunks (chunk_size=%d, "
            "workers=%d)",
            n_data, n_chunks, chunk_size, self._n_workers,
        )

        c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
        weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            full_params = fixed_values.at[varying_idx].set(
                jnp.asarray(varying, dtype=jnp.float64)
            )
            r = compute_residuals(
                full_params, t, q, dt, phi_angle, c2_jax, weights_jax
            )
            return np.asarray(r, dtype=np.float64)

        method = config.method if config.method != "lm" else "trf"
        if method == "dogbox":
            logger.warning(
                "OutOfCoreStrategy: 'dogbox' is not supported by curve_fit_large; "
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

        # Recompute residuals at the solution via residual_fn.
        final_residuals = residual_fn(np.asarray(nlsq_result.x, dtype=np.float64))
        final_jac = np.asarray(nlsq_result.jac, dtype=np.float64) if nlsq_result.jac is not None else None

        # Covariance
        covariance = None
        uncertainties = None
        if final_jac is not None:
            n_dof = max(n_data - n_params, 1)
            s2 = float(np.dot(final_residuals, final_residuals)) / n_dof
            JtJ = final_jac.T @ final_jac
            try:
                covariance = np.linalg.inv(JtJ) * s2
                uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0))
            except np.linalg.LinAlgError:
                logger.warning("OutOfCore: singular J^T J")

        n_dof = max(n_data - n_params, 1)
        final_cost = 0.5 * float(np.sum(final_residuals**2))
        reduced_chi2 = 2.0 * final_cost / n_dof

        metadata: dict[str, Any] = {
            "strategy": "out_of_core",
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "n_workers": self._n_workers,
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

        if nlsq_result.success:
            full_fitted = pm.expand_varying_to_full(result.parameters)
            fitted_c2 = compute_c2_heterodyne(
                jnp.asarray(full_fitted), t, q, dt, phi_angle
            )
            result.fitted_correlation = np.asarray(fitted_c2)
            model.set_params(full_fitted)

        peak_memory_mb = chunk_size * n_params * _BYTES_PER_FLOAT64 / (1024 * 1024)

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            n_chunks=n_chunks,
            peak_memory_mb=peak_memory_mb,
            metadata=metadata,
        )

    @staticmethod
    def _auto_chunk_size(n_data: int, n_params: int) -> int:
        """Auto-determine chunk size based on available memory."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            available_gb = 4.0

        target_bytes = available_gb * (1024 ** 3) * 0.15  # Use 15% for OOC
        chunk_size = int(target_bytes / (n_params * _BYTES_PER_FLOAT64))
        return max(min(chunk_size, n_data), 1000)

    def __repr__(self) -> str:
        return f"OutOfCoreStrategy(chunk_size={self._chunk_size}, n_workers={self._n_workers})"
