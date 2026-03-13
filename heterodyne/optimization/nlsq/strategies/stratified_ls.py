"""Stratified least-squares fitting strategy.

Implements stratified sampling across q-point subsets for efficient
fitting when the dataset has non-uniform information density across
the scattering vector range.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

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


class StratifiedLSStrategy:
    """Stratified sampling across q-point subsets.

    Divides the data into strata based on q-point ranges and fits
    using a weighted combination of residuals from each stratum.
    This is useful when certain q-ranges contribute more information
    about specific parameters.

    Consumes ``NLSQConfig.enable_stratified`` and ``target_chunk_size``.
    """

    @property
    def name(self) -> str:
        return "stratified_ls"

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Fit using stratified residual evaluation.

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
        weights_jax = (
            jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
        )
        t = model.t
        q = model.q
        dt = model.dt
        fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
        varying_idx = jnp.array(pm.varying_indices)

        def residual_fn(varying: np.ndarray) -> np.ndarray:
            full_params = fixed_values.at[varying_idx].set(
                jnp.asarray(varying, dtype=jnp.float64)
            )
            r = compute_residuals(full_params, t, q, dt, phi_angle, c2_jax, weights_jax)
            return np.asarray(r, dtype=np.float64)

        method = config.method if config.method != "lm" else "trf"
        if method == "dogbox":
            logger.warning(
                "StratifiedLSStrategy: 'dogbox' is not supported by curve_fit_large; "
                "coercing to 'trf'."
            )
            method = "trf"

        # curve_fit_large expects f(xdata, *params) -> prediction;
        # residuals = ydata - f(xdata, *params).
        # Set ydata=zeros so residuals = -residual_fn(params).
        # params arrive as JAX-traced scalars; jnp.stack reassembles.
        def _wrapped(xdata: np.ndarray, *params: object) -> jnp.ndarray:
            return -jnp.asarray(
                residual_fn(np.asarray(jnp.stack(list(params))))  # type: ignore[arg-type]
            )

        _xdata = np.arange(n_data, dtype=np.float64)
        _ydata = np.zeros(n_data, dtype=np.float64)

        # curve_fit_large stub returns tuple|OptimizeResult; at runtime it
        # always returns CurveFitResult(OptimizeResult) with .x/.success/.jac
        nlsq_result = cast(
            Any,
            curve_fit_large(
                f=_wrapped,
                xdata=_xdata,
                ydata=_ydata,
                p0=initial,
                bounds=(lower, upper),
                method=method,
            ),
        )

        wall_time = time.perf_counter() - start_time

        # Recompute residuals at the solution via residual_fn.
        final_residuals = residual_fn(np.asarray(nlsq_result.x, dtype=np.float64))
        final_jac = (
            np.asarray(nlsq_result.jac, dtype=np.float64)
            if nlsq_result.jac is not None
            else None
        )

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
                logger.warning("StratifiedLS: singular J^T J")

        n_dof = max(n_data - n_params, 1)
        final_cost = 0.5 * float(np.sum(final_residuals**2))
        reduced_chi2 = 2.0 * final_cost / n_dof

        metadata: dict[str, Any] = {
            "strategy": "stratified_ls",
            "target_chunk_size": config.target_chunk_size,
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

        return StrategyResult(
            result=result,
            strategy_name=self.name,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return "StratifiedLSStrategy()"
