"""Residual-specialized JIT compilation strategy.

More aggressive JIT compilation focused specifically on the residual
evaluation path. Unlike JITStrategy which JITs the full Jacobian,
this strategy JITs only the residual function and lets scipy compute
the Jacobian via finite differences, which can be faster for very
large parameter spaces or when the analytic Jacobian is expensive.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


class ResidualJITStrategy:
    """JIT-compiled residual evaluation without analytic Jacobian.

    Applies aggressive JIT compilation to the residual function only,
    relying on scipy's finite-difference Jacobian approximation. This
    avoids the cost of compiling and evaluating the analytic Jacobian
    while still benefiting from JIT-compiled forward evaluation.

    This strategy is appropriate when:
    - The analytic Jacobian compilation is slow or fails
    - The parameter count is small relative to data size
    - Quick iteration is preferred over Jacobian accuracy
    """

    @property
    def name(self) -> str:
        return "residual_jit"

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Fit with JIT-compiled residual and finite-difference Jacobian.

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

        c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
        weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        # JIT-compile the residual function
        @jax.jit
        def _jit_residuals(varying_jax: jnp.ndarray) -> jnp.ndarray:
            full_params = fixed_values.at[varying_idx].set(varying_jax)
            return compute_residuals(
                full_params, t, q, dt, phi_angle, c2_jax, weights_jax
            )

        # Warm up JIT
        logger.debug("ResidualJIT: warming up JIT compilation")
        _ = _jit_residuals(jnp.asarray(initial, dtype=jnp.float64))

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            r = _jit_residuals(jnp.asarray(varying, dtype=jnp.float64))
            return np.asarray(r, dtype=np.float64)

        method = config.method if config.method != "lm" else "trf"
        max_nfev = config.max_nfev or config.max_iterations * (n_params + 1) * 10

        logger.info(
            "ResidualJIT: fitting %d params, %d data points "
            "(finite-difference Jacobian)",
            n_params, n_data,
        )

        scipy_result = least_squares(
            residual_fn,
            initial,
            bounds=(lower, upper),
            method=method,
            jac="2-point",  # Always use FD Jacobian
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            max_nfev=max_nfev,
            loss=config.loss,
            verbose=max(0, config.verbose - 1),
        )

        wall_time = time.perf_counter() - start_time

        covariance = None
        uncertainties = None
        if scipy_result.jac is not None:
            n_dof = max(n_data - n_params, 1)
            s2 = float(np.dot(scipy_result.fun, scipy_result.fun)) / n_dof
            JtJ = scipy_result.jac.T @ scipy_result.jac
            try:
                covariance = np.linalg.inv(JtJ) * s2
                uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0))
            except np.linalg.LinAlgError:
                logger.warning("ResidualJIT: singular J^T J")

        n_dof = max(n_data - n_params, 1)
        reduced_chi2 = 2.0 * float(scipy_result.cost) / n_dof

        metadata: dict[str, Any] = {
            "strategy": "residual_jit",
            "jit_compiled": True,
            "jacobian_method": "finite_difference",
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
        return "ResidualJITStrategy()"
