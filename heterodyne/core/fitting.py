"""ScaledFittingEngine: orchestrates the full NLSQ fitting pipeline.

Combines parameter transforms, per-angle scaling, model evaluation,
and result validation into a single coherent fitting workflow.

Usage::

    model = HeterodyneModel.from_config(config)
    engine = ScaledFittingEngine(model)
    result = engine.fit(c2_data, phi_angle=45.0, config=nlsq_config)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.optimization.nlsq.data_prep import prepare_fit_data
from heterodyne.optimization.nlsq.result_builder import (
    build_failed_result,
    build_result_from_scipy,
)
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.transforms import ParameterTransform
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.validation import ValidationReport

logger = get_logger(__name__)


class ScaledFittingEngine:
    """Orchestrates NLSQ fitting with per-angle scaling.

    This engine:
    1. Prepares data (upper-triangle extraction, weight construction)
    2. Sets up parameter transforms (varying ↔ full, log-space)
    3. Builds a scipy-compatible residual function
    4. Runs least_squares with the configured method
    5. Constructs NLSQResult with covariance and uncertainties
    6. Optionally validates the result

    The engine supports multi-angle fitting via fit_multi_angle(),
    propagating fitted parameters as warm-starts across angles.
    """

    def __init__(
        self,
        model: HeterodyneModel,
        use_log_transform: bool = False,
    ) -> None:
        """Initialize fitting engine.

        Args:
            model: Configured HeterodyneModel instance
            use_log_transform: Apply log-space transform to D0, v0 params
        """
        self._model = model
        self._transform = ParameterTransform(
            model.param_manager, use_log=use_log_transform
        )

    @property
    def model(self) -> HeterodyneModel:
        return self._model

    @property
    def transform(self) -> ParameterTransform:
        return self._transform

    def fit(
        self,
        c2_data: np.ndarray,
        phi_angle: float = 0.0,
        config: NLSQConfig | None = None,
        weights: np.ndarray | None = None,
        angle_idx: int = 0,
        validate: bool = True,
    ) -> NLSQResult:
        """Fit model to correlation data at a single angle.

        Args:
            c2_data: Correlation matrix, shape (N, N)
            phi_angle: Detector phi angle (degrees)
            config: NLSQ configuration (uses defaults if None)
            weights: Optional weight matrix, shape (N, N)
            angle_idx: Angle index for per-angle scaling
            validate: Run post-fit validation

        Returns:
            NLSQResult with fitted parameters and diagnostics
        """
        from heterodyne.optimization.nlsq.config import NLSQConfig

        if config is None:
            config = NLSQConfig()

        t_start = time.perf_counter()
        param_names = self._transform.varying_names

        # Get per-angle contrast/offset
        contrast, offset_val = self._model.scaling.get_for_angle(angle_idx)

        # Prepare data
        data_flat, sqrt_weights, n_data = prepare_fit_data(
            c2_data, weights, use_upper_triangle=True
        )

        # Get upper-triangle indices for model extraction
        n = c2_data.shape[0]
        triu_i, triu_j = np.triu_indices(n, k=0)

        # Build residual function in optimizer space
        x0 = self._transform.get_optimizer_x0()
        lower, upper = self._transform.get_optimizer_bounds()

        t_jax = self._model.t
        q = self._model.q
        dt = self._model.dt
        data_flat_jax = jnp.asarray(data_flat)
        sqrt_w_jax = jnp.asarray(sqrt_weights)

        def residual_fn(x_opt: np.ndarray) -> np.ndarray:
            full_params = self._transform.to_model(x_opt)
            c2_model = compute_c2_heterodyne(
                jnp.asarray(full_params), t_jax, q, dt,
                phi_angle, contrast, offset_val,
            )
            model_flat = np.asarray(c2_model)[triu_i, triu_j]
            return (model_flat - np.asarray(data_flat_jax)) * np.asarray(sqrt_w_jax)

        # Run optimizer
        try:
            opt_result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                method=config.method,
                ftol=config.ftol,
                xtol=config.xtol,
                gtol=config.gtol,
                max_nfev=config.max_nfev or config.max_iterations * (len(x0) + 1),
                loss=config.loss,
                verbose=max(config.verbose - 1, 0),
            )
        except Exception as e:
            wall_time = time.perf_counter() - t_start
            logger.error("Optimizer failed: %s", e)
            return build_failed_result(
                param_names, str(e),
                initial_params=x0,
                wall_time=wall_time,
                metadata={"phi_angle": phi_angle, "angle_idx": angle_idx},
            )

        wall_time = time.perf_counter() - t_start

        # Transform solution back to model space
        opt_params_model = self._transform.to_model(opt_result.x)

        # Build result with model-space parameters
        result = build_result_from_scipy(
            opt_result,
            parameter_names=param_names,
            n_data=n_data,
            wall_time=wall_time,
            metadata={
                "phi_angle": phi_angle,
                "angle_idx": angle_idx,
                "contrast": contrast,
                "offset": offset_val,
                "log_transform": self._transform.log_mask.any(),
            },
        )

        # Store full model-space parameters
        result.metadata["full_params"] = opt_params_model

        # Update model state with fitted values
        self._model.set_params(opt_params_model)

        # Compute fitted correlation for diagnostics
        result.fitted_correlation = np.asarray(
            self._model.compute_correlation(
                phi_angle=phi_angle, contrast=contrast, offset=offset_val
            )
        )

        # Validate if requested
        if validate:
            report = self.validate_result(result, config)
            result.metadata["validation"] = report.summary()

        logger.info(
            "Fit complete: χ²_red=%.4f, %d iterations, %.2fs",
            result.reduced_chi_squared or 0.0,
            result.n_iterations,
            wall_time,
        )

        return result

    def fit_multi_angle(
        self,
        c2_data: np.ndarray,
        phi_angles: list[float] | np.ndarray,
        config: NLSQConfig | None = None,
        weights: np.ndarray | None = None,
    ) -> list[NLSQResult]:
        """Fit multiple angles sequentially with warm-start.

        Parameters from each angle are propagated as the initial guess
        for the next angle, exploiting parameter continuity.

        Args:
            c2_data: Shape (n_phi, N, N) or (N, N) for single angle
            phi_angles: Array of phi angles (degrees)
            config: NLSQ configuration
            weights: Shape (n_phi, N, N) or (N, N) for shared weights

        Returns:
            List of NLSQResult, one per angle
        """
        phi_angles = np.asarray(phi_angles)

        if c2_data.ndim == 2:
            c2_data = c2_data[np.newaxis, ...]

        if len(c2_data) != len(phi_angles):
            raise ValueError(
                f"c2_data has {len(c2_data)} slices but {len(phi_angles)} angles"
            )

        results = []
        for i, phi in enumerate(phi_angles):
            logger.info(
                "ScaledFittingEngine: angle %d/%d: %.1f°%s",
                i + 1, len(phi_angles), phi,
                f" (warm-start from {phi_angles[i-1]:.1f}°)" if i > 0 else "",
            )

            c2_i = c2_data[i]
            w_i = weights[i] if weights is not None and weights.ndim == 3 else weights

            result = self.fit(
                c2_i, phi_angle=float(phi), config=config,
                weights=w_i, angle_idx=min(i, self._model.scaling.n_angles - 1),
            )
            results.append(result)

        return results

    def validate_result(
        self,
        result: NLSQResult,
        config: NLSQConfig | None = None,
    ) -> ValidationReport:
        """Run validation checks on a fit result.

        Args:
            result: NLSQResult to validate
            config: NLSQ config for threshold access

        Returns:
            ValidationReport with any issues found
        """
        from heterodyne.optimization.nlsq.validation import (
            BoundsValidator,
            ConvergenceValidator,
            ResultValidator,
            ValidationReport,
        )

        # Merge reports from all validators
        result_report = ResultValidator().validate(result)
        bounds_report = BoundsValidator().validate(result)
        convergence_report = ConvergenceValidator().validate(result, config)

        merged = ValidationReport()
        merged.issues.extend(result_report.issues)
        merged.issues.extend(bounds_report.issues)
        merged.issues.extend(convergence_report.issues)

        if merged.errors:
            merged.is_valid = False

        return merged
