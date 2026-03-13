"""Hybrid streaming optimization strategy for heterodyne fitting.

Implements a 4-phase fitting pipeline:
1. Normalization — compute data statistics for parameter scaling
2. L-BFGS warmup + Gauss-Newton refinement — combined via
   ``nlsq.AdaptiveHybridStreamingOptimizer`` (single call)
3. (internal to optimizer)
4. Denormalization — restore original parameter scaling

This strategy is effective for large datasets where the initial
parameter guess may be far from the optimum.  The L-BFGS warmup phase
provides fast global convergence while the Gauss-Newton phase provides
precise local convergence.  Both phases are managed natively by
``AdaptiveHybridStreamingOptimizer``.

When the ``nlsq`` streaming optimizer is unavailable the strategy falls
back to ``nlsq.curve_fit_large``, which still avoids any scipy dependency.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

# nlsq import MUST precede JAX — enables x64 mode
from nlsq import curve_fit_large

try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None  # type: ignore[assignment,misc]
    HybridStreamingConfig = None  # type: ignore[assignment,misc]

import jax.numpy as jnp
import numpy as np

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.result_builder import build_result_from_nlsq
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


class HybridStreamingStrategy:
    """4-phase hybrid optimization: normalize -> streaming-optimizer -> denormalize.

    Phase 1 (Normalization):
        Computes per-parameter scale factors from parameter ranges.  Transforms
        parameters to a normalized space where all have comparable magnitudes,
        improving optimizer conditioning.

    Phase 2+3 (L-BFGS Warmup + Gauss-Newton Refinement):
        Delegates to ``nlsq.AdaptiveHybridStreamingOptimizer``, which combines
        the L-BFGS warmup and Gauss-Newton refinement into a single call.
        Falls back to ``nlsq.curve_fit_large`` when the streaming optimizer is
        unavailable.

    Phase 4 (Denormalization):
        Transforms the converged parameters back to the original scale
        (handled internally — parameters are always kept in original space).

    Consumes ``NLSQConfig.hybrid_*`` fields.
    """

    @property
    def name(self) -> str:
        return "hybrid_streaming"

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Execute the 4-phase hybrid optimization.

        Args:
            model: Configured HeterodyneModel.
            c2_data: Correlation data, shape (N, N).
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration with hybrid_* fields.
            weights: Optional per-point weights.

        Returns:
            StrategyResult with converged parameters.
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

        c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
        weights_jax = (
            jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        )
        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        # Phase 1: Normalization — build per-parameter scale factors.
        if config.hybrid_normalization:
            _param_scales = np.where(upper - lower > 0, upper - lower, 1.0)
        else:
            _param_scales = np.ones(n_params)

        def _full_params(varying: np.ndarray) -> jnp.ndarray:
            return fixed_values.at[varying_idx].set(
                jnp.asarray(varying, dtype=jnp.float64)
            )

        def residual_fn(xdata: np.ndarray, *params: float) -> np.ndarray:
            """Wrapped residual for the streaming optimizer signature.

            The streaming optimizer expects f(xdata, *params) -> predictions.
            Here xdata carries the flattened c2_data; we return residuals as
            predictions so the optimizer minimises their norm.
            """
            varying = np.asarray(params, dtype=np.float64)
            r = compute_residuals(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )
            return np.asarray(r, dtype=np.float64)

        # Flat index array used as xdata placeholder (shape = [n_data]).
        xdata = np.arange(n_data, dtype=np.float64)
        # Target is zero residual (we treat residuals as "predictions").
        ydata = np.zeros(n_data, dtype=np.float64)

        warmup_maxiter = max(
            1, int(config.max_iterations * config.hybrid_warmup_fraction)
        )
        gn_maxiter = max(
            100,
            config.max_iterations
            - int(config.max_iterations * config.hybrid_warmup_fraction),
        )
        chunk_size: int = getattr(config, "streaming_chunk_size", 50_000)

        metadata: dict[str, Any] = {
            "strategy": "hybrid_streaming",
            "hybrid_method": config.hybrid_method,
            "warmup_fraction": config.hybrid_warmup_fraction,
            "phases": {},
        }

        # Phase 2+3: Optimization — streaming optimizer or fallback.
        if STREAMING_AVAILABLE:
            logger.info(
                "HybridStreaming: using AdaptiveHybridStreamingOptimizer "
                "(warmup=%d iters, gn_max=%d iters)",
                warmup_maxiter,
                gn_maxiter,
            )

            streaming_config = HybridStreamingConfig(
                normalize=config.hybrid_normalization,
                warmup_iterations=warmup_maxiter,
                gauss_newton_max_iterations=gn_maxiter,
                chunk_size=chunk_size,
                validate_numerics=True,
            )
            optimizer = AdaptiveHybridStreamingOptimizer(streaming_config)

            raw_result = optimizer.fit(
                data_source=(xdata, ydata),
                func=residual_fn,
                p0=initial,
                bounds=(lower, upper),
            )

            logger.info(
                "HybridStreaming: optimizer complete (success=%s)",
                raw_result.get("success", "?"),
            )
            metadata["phases"]["streaming"] = raw_result.get(
                "streaming_diagnostics", {}
            )

        else:
            logger.warning(
                "HybridStreaming: AdaptiveHybridStreamingOptimizer unavailable; "
                "falling back to nlsq.curve_fit_large"
            )

            raw_result = curve_fit_large(
                residual_fn,
                xdata,
                ydata,
                p0=initial,
                bounds=(lower, upper),
            )
            metadata["phases"]["fallback"] = "curve_fit_large"

        wall_time = time.perf_counter() - start_time

        # Phase 4: Denormalization — build normalized NLSQResult.
        result = build_result_from_nlsq(
            raw_result,
            parameter_names=pm.varying_names,
            n_data=n_data,
            wall_time=wall_time,
            metadata=metadata,
        )

        if result.success:
            full_fitted = pm.expand_varying_to_full(result.parameters)
            fitted_c2 = compute_c2_heterodyne(
                jnp.asarray(full_fitted), t, q, dt, phi_angle
            )
            result.fitted_correlation = np.asarray(fitted_c2)
            model.set_params(full_fitted)

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return "HybridStreamingStrategy()"
