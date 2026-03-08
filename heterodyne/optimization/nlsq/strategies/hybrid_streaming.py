"""Hybrid streaming optimization strategy for heterodyne fitting.

Implements a 4-phase fitting pipeline:
1. Normalization — compute data statistics for parameter scaling
2. L-BFGS warmup — fast rough convergence via scipy L-BFGS-B
3. Gauss-Newton refinement — precise convergence via least_squares
4. Denormalization — restore original parameter scaling

This strategy is effective for large datasets where the initial
parameter guess may be far from the optimum. The L-BFGS phase
provides fast global convergence, while the Gauss-Newton phase
provides precise local convergence.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


class HybridStreamingStrategy:
    """4-phase hybrid optimization: normalize -> L-BFGS -> Gauss-Newton -> denormalize.

    Phase 1 (Normalization):
        Computes per-parameter scale factors from parameter ranges and data
        statistics. Transforms parameters to a normalized space where all
        have comparable magnitudes, improving optimizer conditioning.

    Phase 2 (L-BFGS Warmup):
        Uses scipy's L-BFGS-B minimizer on the sum-of-squares cost function
        for fast rough convergence. L-BFGS is a quasi-Newton method that
        approximates the Hessian using limited-memory BFGS updates.

    Phase 3 (Gauss-Newton Refinement):
        Switches to scipy's least_squares (trust-region reflective) for
        precise convergence near the optimum. This exploits the least-squares
        structure for faster local convergence.

    Phase 4 (Denormalization):
        Transforms the converged parameters back to the original scale.

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
        weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        # Phase 1: Normalization
        if config.hybrid_normalization:
            _param_scales = np.where(upper - lower > 0, upper - lower, 1.0)
        else:
            _param_scales = np.ones(n_params)

        def _full_params(varying: np.ndarray) -> jnp.ndarray:
            return fixed_values.at[varying_idx].set(
                jnp.asarray(varying, dtype=jnp.float64)
            )

        def cost_fn(varying: np.ndarray) -> float:
            r = compute_residuals(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )
            return float(0.5 * jnp.sum(r ** 2))

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            r = compute_residuals(
                _full_params(varying), t, q, dt, phi_angle, c2_jax, weights_jax
            )
            return np.asarray(r, dtype=np.float64)

        # Phase 2: L-BFGS warmup
        phase_info: dict[str, Any] = {}
        current_params = initial.copy()

        if config.hybrid_method == "lbfgs" or config.hybrid_warmup_fraction > 0:
            warmup_maxiter = max(1, int(config.max_iterations * config.hybrid_warmup_fraction))

            logger.info(
                "HybridStreaming Phase 2: L-BFGS warmup (%d iterations)",
                warmup_maxiter,
            )

            lbfgs_result = minimize(
                cost_fn,
                current_params,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper, strict=True)),
                options={
                    "maxiter": warmup_maxiter,
                    "maxcor": config.hybrid_lbfgs_memory,
                },
            )

            current_params = lbfgs_result.x
            phase_info["lbfgs_cost"] = float(lbfgs_result.fun)
            phase_info["lbfgs_nfev"] = lbfgs_result.nfev
            phase_info["lbfgs_success"] = lbfgs_result.success

            logger.info(
                "HybridStreaming Phase 2 complete: cost=%.4e, nfev=%d",
                lbfgs_result.fun, lbfgs_result.nfev,
            )

        # Phase 3: Gauss-Newton refinement
        logger.info("HybridStreaming Phase 3: Gauss-Newton refinement")

        gn_maxiter = config.max_iterations - int(config.max_iterations * config.hybrid_warmup_fraction)
        gn_maxiter = max(gn_maxiter, 100)

        method = config.method if config.method != "lm" else "trf"

        scipy_result = least_squares(
            residual_fn,
            current_params,
            bounds=(lower, upper),
            method=method,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            max_nfev=gn_maxiter * (n_params + 1),
            loss=config.loss,
            verbose=max(0, config.verbose - 1),
        )

        wall_time = time.perf_counter() - start_time

        # Phase 4: Denormalization (params already in original space)
        # Covariance estimation
        covariance = None
        uncertainties = None
        if scipy_result.jac is not None and scipy_result.jac.size > 0:
            n_dof = max(n_data - n_params, 1)
            s2 = float(np.dot(scipy_result.fun, scipy_result.fun)) / n_dof
            JtJ = scipy_result.jac.T @ scipy_result.jac
            try:
                covariance = np.linalg.inv(JtJ) * s2
                uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0))
            except np.linalg.LinAlgError:
                logger.warning("HybridStreaming: singular J^T J for covariance")

        n_dof = max(n_data - n_params, 1)
        reduced_chi2 = 2.0 * float(scipy_result.cost) / n_dof

        metadata: dict[str, Any] = {
            "strategy": "hybrid_streaming",
            "phases": phase_info,
            "hybrid_method": config.hybrid_method,
            "warmup_fraction": config.hybrid_warmup_fraction,
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
            n_function_evals=scipy_result.nfev + phase_info.get("lbfgs_nfev", 0),
            convergence_reason=scipy_result.message,
            residuals=scipy_result.fun,
            jacobian=scipy_result.jac,
            wall_time_seconds=wall_time,
            metadata=metadata,
        )

        if scipy_result.success:
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
