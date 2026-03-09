"""Unified Heterodyne Engine with JAX-Accelerated Least Squares
==============================================================

Core fitting infrastructure ported from homodyne's ``UnifiedHomodyneEngine``,
adapted for heterodyne's 14-parameter two-component correlation model.

Provides:
- ``ParameterSpace`` — bounds/priors from heterodyne's parameter registry
- ``DatasetSize`` — dataset size categorization for optimization strategy
- ``FitResult`` — structured result container with fit statistics
- ``solve_least_squares_jax`` — batch 2x2 normal-equations solver
- ``solve_least_squares_general_jax`` — N-param solver (Cholesky/SVD)
- ``solve_least_squares_chunked_jax`` — ``jax.lax.scan``-based chunked solver
- ``UnifiedHeterodyneEngine`` — main engine class

This module does NOT run the full NLSQ pipeline — that is handled by
the adapter/strategy system.  It provides the JAX solvers and scaling
estimation infrastructure that the pipeline calls into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeVar

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.utils.logging import get_logger

# Type variable for generic functions
F = TypeVar("F", bound=Callable[..., Any])

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit

    JAX_AVAILABLE = True
    jit = jax_jit
except ImportError:
    JAX_AVAILABLE = False
    import types

    jnp: types.ModuleType = np  # type: ignore[no-redef]

    def jit(f: F) -> F:  # type: ignore[misc]  # noqa: UP047
        return f  # No-op decorator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ParameterSpace
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds and priors.

    Physical parameter bounds/priors are pulled from
    ``heterodyne.config.parameter_registry.DEFAULT_REGISTRY`` so there is a
    single source of truth.

    Scaling parameter defaults follow the task specification:
    - contrast_bounds = (0.0, 10.0), contrast_prior = (1.0, 0.5)
    - offset_bounds  = (-1.0, 1.0), offset_prior  = (0.0, 0.25)
    """

    # Scaling parameters
    contrast_bounds: tuple[float, float] = (0.0, 10.0)
    offset_bounds: tuple[float, float] = (-1.0, 1.0)
    contrast_prior: tuple[float, float] = (1.0, 0.5)
    offset_prior: tuple[float, float] = (0.0, 0.25)

    # Data ranges
    fitted_range: tuple[float, float] = (0.0, 2.0)
    theory_range: tuple[float, float] = (0.0, 1.0)

    # Optional configuration manager for bound override (same pattern as homodyne)
    config_manager: Any | None = None

    # Cached bounds/priors populated from registry
    _physics_bounds: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _physics_priors: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Populate physics bounds/priors from parameter registry."""
        self._physics_bounds = []
        self._physics_priors = []
        for name in ALL_PARAM_NAMES:
            info = DEFAULT_REGISTRY[name]
            self._physics_bounds.append((info.min_bound, info.max_bound))
            prior_mean = info.prior_mean if info.prior_mean is not None else (
                (info.min_bound + info.max_bound) / 2.0
            )
            prior_std = info.prior_std if info.prior_std is not None else (
                (info.max_bound - info.min_bound) / 2.0
            )
            self._physics_priors.append((prior_mean, prior_std))

    def get_param_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for the 14 physics parameters.

        Returns
        -------
        list of tuple
            List of (min, max) bounds tuples, length 14.
        """
        if self.config_manager is not None:
            try:
                if hasattr(self.config_manager, "get_bounds"):
                    bounds = self.config_manager.get_bounds()
                    if bounds and len(bounds) == 14:
                        logger.info(
                            "Loaded %d parameter bounds from config_manager",
                            len(bounds),
                        )
                        return bounds
            except (TypeError, KeyError, AttributeError, ValueError) as e:
                logger.warning(
                    "Failed to use config_manager for bounds: %s, using registry defaults",
                    e,
                )
        return list(self._physics_bounds)

    def get_param_priors(self) -> list[tuple[float, float]]:
        """Get parameter priors for the 14 physics parameters.

        Returns
        -------
        list of tuple
            List of (mean, std) prior tuples, length 14.
        """
        return list(self._physics_priors)


# ---------------------------------------------------------------------------
# DatasetSize
# ---------------------------------------------------------------------------


class DatasetSize:
    """Dataset size categories for optimization."""

    SMALL = "small"    # <1M points
    MEDIUM = "medium"  # 1-10M points
    LARGE = "large"    # >10M points

    @staticmethod
    def categorize(data_size: int) -> str:
        """Categorize dataset size.

        Parameters
        ----------
        data_size : int
            Number of data points.

        Returns
        -------
        str
            One of ``"small"``, ``"medium"``, ``"large"``.
        """
        if data_size < 1_000_000:
            return DatasetSize.SMALL
        elif data_size < 10_000_000:
            return DatasetSize.MEDIUM
        else:
            return DatasetSize.LARGE


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Results from unified heterodyne model fitting.

    Contains both physical and scaling parameters with comprehensive
    fit statistics.
    """

    # Optimized parameters
    params: np.ndarray  # Physical parameters (14-element)
    contrast: float
    offset: float

    # Fit quality metrics
    chi_squared: float
    reduced_chi_squared: float
    degrees_of_freedom: int

    # Parameter uncertainties (if computed)
    param_errors: np.ndarray | None = None
    contrast_error: float | None = None
    offset_error: float | None = None

    # Additional statistics
    residual_std: float = 0.0
    max_residual: float = 0.0
    fit_iterations: int = 0
    converged: bool = True

    # Computational metadata
    computation_time: float = 0.0
    backend: str = "JAX" if JAX_AVAILABLE else "NumPy"
    dataset_size: str = "unknown"

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive fit summary.

        Returns
        -------
        dict
            Nested dictionary with parameters, errors, fit_quality,
            and convergence sections.
        """
        return {
            "parameters": {
                "physical": self.params.tolist(),
                "contrast": self.contrast,
                "offset": self.offset,
            },
            "errors": {
                "physical": (
                    self.param_errors.tolist()
                    if self.param_errors is not None
                    else None
                ),
                "contrast": self.contrast_error,
                "offset": self.offset_error,
            },
            "fit_quality": {
                "chi_squared": self.chi_squared,
                "reduced_chi_squared": self.reduced_chi_squared,
                "degrees_of_freedom": self.degrees_of_freedom,
                "residual_std": self.residual_std,
                "max_residual": self.max_residual,
            },
            "convergence": {
                "converged": self.converged,
                "iterations": self.fit_iterations,
                "computation_time": self.computation_time,
                "backend": self.backend,
                "dataset_size": self.dataset_size,
            },
        }


# ---------------------------------------------------------------------------
# JAX-accelerated least squares solvers
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:

    @jit
    def solve_least_squares_jax(
        theory_batch: jnp.ndarray,
        exp_batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-accelerated batch least squares solver.

        Solves the 2x2 normal equations for each row (angle):

            ``[[sum_t^2, sum_t], [sum_t, N]] * [contrast, offset] = [sum_te, sum_e]``

        Args:
            theory_batch: Theory values, shape ``(n_angles, n_data_points)``.
            exp_batch: Experimental values, shape ``(n_angles, n_data_points)``.

        Returns:
            ``(contrast_batch, offset_batch)``, each shape ``(n_angles,)``.
        """
        n_angles, n_data = theory_batch.shape

        # Vectorized normal equation components
        sum_theory_sq = jnp.sum(theory_batch * theory_batch, axis=1)
        sum_theory = jnp.sum(theory_batch, axis=1)
        sum_exp = jnp.sum(exp_batch, axis=1)
        sum_theory_exp = jnp.sum(theory_batch * exp_batch, axis=1)

        # Determinant of 2x2 system
        det = sum_theory_sq * n_data - sum_theory * sum_theory

        # Handle singular matrix cases
        valid_det = jnp.abs(det) > 1e-12
        safe_det = jnp.where(valid_det, det, 1.0)

        # Solve normal equations
        contrast = (n_data * sum_theory_exp - sum_theory * sum_exp) / safe_det
        offset = (sum_theory_sq * sum_exp - sum_theory * sum_theory_exp) / safe_det

        # Fallback for singular cases
        contrast = jnp.where(valid_det, contrast, 1.0)
        offset = jnp.where(valid_det, offset, 1.0)

        # Ensure contrast > 1e-6 (physical constraint)
        # Use jnp.where for gradient safety (jnp.maximum zeros gradient below floor)
        contrast = jnp.where(contrast > 1e-6, contrast, 1e-6)

        return contrast, offset

    @partial(jit, static_argnums=(2,))
    def solve_least_squares_general_jax(
        design_matrix: jnp.ndarray,
        target_vector: jnp.ndarray,
        regularization: float = 1e-10,
    ) -> jnp.ndarray:
        """General N-parameter least squares solver via normal equations.

        Solves ``min ||A x - b||^2`` via ``(A^T A + lambda I) x = A^T b``.
        Uses Cholesky for well-conditioned systems (cond < 1e10) and
        SVD fallback for ill-conditioned ones, selected via ``jax.lax.cond``.

        Args:
            design_matrix: Design matrix A, shape ``(n_samples, n_params)``.
            target_vector: Target vector b, shape ``(n_samples,)``.
            regularization: Ridge regularization parameter.

        Returns:
            Solution vector, shape ``(n_params,)``.
        """
        # Gram matrix + regularization
        gram_matrix = design_matrix.T @ design_matrix
        n_params = gram_matrix.shape[0]
        gram_matrix_reg = gram_matrix + regularization * jnp.eye(n_params)

        # A^T b
        design_T_target = design_matrix.T @ target_vector

        # Condition number via eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(gram_matrix_reg)
        condition_number = jnp.where(
            eigenvalues[0] > 0,
            eigenvalues[-1] / eigenvalues[0],
            jnp.inf,
        )

        def cholesky_solve() -> Any:
            L = jnp.linalg.cholesky(gram_matrix_reg)
            z = jax.scipy.linalg.solve_triangular(L, design_T_target, lower=True)
            return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

        def svd_solve() -> Any:
            return jnp.linalg.lstsq(gram_matrix_reg, design_T_target, rcond=None)[0]

        params = jax.lax.cond(
            condition_number < 1e10,
            lambda _: cholesky_solve(),
            lambda _: svd_solve(),
            None,
        )

        return params  # type: ignore[no-any-return]

    @jit
    def solve_least_squares_chunked_jax(
        theory_chunks: jnp.ndarray,
        exp_chunks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Memory-efficient chunked solver using ``jax.lax.scan``.

        Accumulates normal equation components across chunks, then
        solves the same 2x2 system as ``solve_least_squares_jax``.

        Args:
            theory_chunks: Theory values, shape ``(n_chunks, chunk_size)``.
            exp_chunks: Experimental values, shape ``(n_chunks, chunk_size)``.

        Returns:
            ``(contrast, offset)`` scalars.
        """

        def process_chunk(
            carry: tuple[Any, Any, Any, Any, Any],
            chunk_data: tuple[Any, Any],
        ) -> tuple[tuple[Any, Any, Any, Any, Any], None]:
            theory_chunk, exp_chunk = chunk_data
            sum_theory_sq, sum_theory, sum_exp, sum_theory_exp, n_data = carry

            chunk_size = theory_chunk.shape[0]
            sum_theory_sq += jnp.sum(theory_chunk * theory_chunk)
            sum_theory += jnp.sum(theory_chunk)
            sum_exp += jnp.sum(exp_chunk)
            sum_theory_exp += jnp.sum(theory_chunk * exp_chunk)
            n_data += chunk_size

            return (sum_theory_sq, sum_theory, sum_exp, sum_theory_exp, n_data), None

        # All carry elements use jnp.array for dtype consistency in scan
        carry_init = (
            jnp.array(0.0, dtype=jnp.float64),  # sum_theory_sq
            jnp.array(0.0, dtype=jnp.float64),  # sum_theory
            jnp.array(0.0, dtype=jnp.float64),  # sum_exp
            jnp.array(0.0, dtype=jnp.float64),  # sum_theory_exp
            jnp.array(0, dtype=jnp.int64),       # n_data
        )

        (
            (
                sum_theory_sq_final,
                sum_theory_final,
                sum_exp_final,
                sum_theory_exp_final,
                n_data_final,
            ),
            _,
        ) = jax.lax.scan(process_chunk, carry_init, (theory_chunks, exp_chunks))  # type: ignore[arg-type]

        # Solve 2x2 system
        det = sum_theory_sq_final * n_data_final - sum_theory_final * sum_theory_final

        valid_det = jnp.abs(det) > 1e-12
        safe_det = jnp.where(valid_det, det, 1.0)

        contrast = (
            n_data_final * sum_theory_exp_final - sum_theory_final * sum_exp_final
        ) / safe_det
        offset = (
            sum_theory_sq_final * sum_exp_final
            - sum_theory_final * sum_theory_exp_final
        ) / safe_det

        contrast = jnp.where(valid_det, contrast, 1.0)
        offset = jnp.where(valid_det, offset, 1.0)
        contrast = jnp.where(contrast > 1e-6, contrast, 1e-6)

        return contrast, offset

else:
    # NumPy fallback versions

    def solve_least_squares_jax(  # type: ignore[misc]
        theory_batch: np.ndarray,
        exp_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for batch least squares when JAX unavailable."""
        n_angles, n_data = theory_batch.shape
        contrast_batch = np.zeros(n_angles)
        offset_batch = np.zeros(n_angles)

        for i in range(n_angles):
            theory = theory_batch[i]
            exp = exp_batch[i]

            sum_theory_sq = np.sum(theory * theory)
            sum_theory = np.sum(theory)
            sum_exp = np.sum(exp)
            sum_theory_exp = np.sum(theory * exp)

            det = sum_theory_sq * n_data - sum_theory * sum_theory
            if abs(det) > 1e-12:
                contrast_batch[i] = (
                    n_data * sum_theory_exp - sum_theory * sum_exp
                ) / det
                offset_batch[i] = (
                    sum_theory_sq * sum_exp - sum_theory * sum_theory_exp
                ) / det
                contrast_batch[i] = max(contrast_batch[i], 1e-6)
            else:
                contrast_batch[i] = 1.0
                offset_batch[i] = 1.0

        return contrast_batch, offset_batch

    def solve_least_squares_general_jax(  # type: ignore[misc]
        design_matrix: np.ndarray,
        target_vector: np.ndarray,
        regularization: float = 1e-10,
    ) -> np.ndarray:
        """NumPy fallback for general least squares."""
        solution, _, _, _ = np.linalg.lstsq(
            design_matrix,
            target_vector,
            rcond=regularization,
        )
        return solution  # type: ignore[no-any-return]

    def solve_least_squares_chunked_jax(  # type: ignore[misc]
        theory_chunks: np.ndarray,
        exp_chunks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for chunked least squares."""
        sum_theory_sq = 0.0
        sum_theory = 0.0
        sum_exp = 0.0
        sum_theory_exp = 0.0
        n_data = 0

        for theory_chunk, exp_chunk in zip(
            theory_chunks, exp_chunks, strict=True
        ):
            sum_theory_sq += np.sum(theory_chunk * theory_chunk)
            sum_theory += np.sum(theory_chunk)
            sum_exp += np.sum(exp_chunk)
            sum_theory_exp += np.sum(theory_chunk * exp_chunk)
            n_data += theory_chunk.shape[0]

        det = sum_theory_sq * n_data - sum_theory * sum_theory

        if abs(det) > 1e-12:
            contrast = (n_data * sum_theory_exp - sum_theory * sum_exp) / det
            offset = (sum_theory_sq * sum_exp - sum_theory * sum_theory_exp) / det
            contrast = max(contrast, 1e-6)
        else:
            contrast = 1.0
            offset = 1.0

        return np.array(contrast), np.array(offset)


# ---------------------------------------------------------------------------
# UnifiedHeterodyneEngine
# ---------------------------------------------------------------------------


class UnifiedHeterodyneEngine:
    """Unified heterodyne fitting engine with JAX acceleration.

    Provides scaling estimation, likelihood computation, input validation,
    and parameter space introspection for heterodyne's 14-parameter
    two-component correlation model.  Unlike homodyne, there is no
    ``analysis_mode`` parameter — heterodyne has a single transport model.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace | None = None,
    ) -> None:
        """Initialize unified heterodyne engine.

        Args:
            parameter_space: Parameter space definition (uses default if None).
        """
        self.parameter_space = parameter_space or ParameterSpace()

        # Cache bounds/priors from parameter space
        self.param_bounds = self.parameter_space.get_param_bounds()
        self.param_priors = self.parameter_space.get_param_priors()

        logger.info("Unified heterodyne engine initialized")
        logger.info(
            "Parameter count: %d physical + 2 scaling", len(self.param_bounds)
        )
        logger.info(
            "JAX acceleration: %s",
            "enabled" if JAX_AVAILABLE else "disabled (NumPy fallback)",
        )

    def estimate_scaling_parameters(
        self,
        data: np.ndarray,
        theory: np.ndarray,
        validate_bounds: bool = True,
    ) -> tuple[float, float]:
        """Estimate contrast and offset via batch least squares.

        Uses ``solve_least_squares_jax`` for the 2x2 normal-equations solve.

        Args:
            data: Experimental correlation data (1-D or 2-D batch).
            theory: Theoretical correlation (1-D or 2-D batch).
            validate_bounds: Clip result to parameter space bounds.

        Returns:
            ``(contrast, offset)`` tuple.
        """
        # Ensure batch dimensions
        if data.ndim == 1 and theory.ndim == 1:
            data_batch = data[np.newaxis, :]
            theory_batch = theory[np.newaxis, :]
        else:
            data_batch = data
            theory_batch = theory

        # Convert to JAX arrays if available
        if JAX_AVAILABLE:
            data_jax: Any = jnp.array(data_batch)
            theory_jax: Any = jnp.array(theory_batch)
        else:
            data_jax = data_batch
            theory_jax = theory_batch

        contrast_batch, offset_batch = solve_least_squares_jax(
            theory_jax, data_jax
        )

        if JAX_AVAILABLE:
            contrast = float(jnp.mean(contrast_batch))
            offset = float(jnp.mean(offset_batch))
        else:
            contrast = float(np.mean(contrast_batch))
            offset = float(np.mean(offset_batch))

        if validate_bounds:
            contrast = float(
                np.clip(contrast, *self.parameter_space.contrast_bounds)
            )
            offset = float(
                np.clip(offset, *self.parameter_space.offset_bounds)
            )

        logger.debug(
            "Scaling parameters: contrast=%.4f, offset=%.4f", contrast, offset
        )

        return contrast, offset

    def compute_likelihood(
        self,
        params: np.ndarray,
        contrast: float,
        offset: float,
        data: np.ndarray,
        sigma: np.ndarray,
        t: np.ndarray,
        phi: float,
        q: float,
        dt: float | None = None,
    ) -> float:
        """Compute negative log-likelihood for heterodyne model.

        Uses ``compute_c2_heterodyne`` (not homodyne's g1-squared Siegert
        relation) to compute the theoretical correlation.

        NLL = 0.5 * sum((data - fitted)^2 / sigma^2)
              + 0.5 * sum(log(2*pi*sigma^2))

        Args:
            params: Physical parameters, shape ``(14,)``.
            contrast: Contrast scaling parameter.
            offset: Offset parameter.
            data: Experimental data.
            sigma: Measurement uncertainties.
            t: Time array.
            phi: Detector phi angle (degrees).
            q: Scattering wavevector magnitude.
            dt: Time step (optional; derived from ``t`` if None).

        Returns:
            Negative log-likelihood value.
        """
        try:
            from heterodyne.core.jax_backend import compute_c2_heterodyne

            if dt is None:
                dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

            params_jax = jnp.asarray(params)
            t_jax = jnp.asarray(t)

            c2_model = compute_c2_heterodyne(
                params_jax, t_jax, q, dt, phi, contrast, offset
            )

            # Flatten for element-wise comparison
            c2_flat = jnp.ravel(c2_model)
            data_flat = jnp.ravel(jnp.asarray(data))
            sigma_flat = jnp.ravel(jnp.asarray(sigma))

            # Validate array lengths match (no silent truncation)
            if len(c2_flat) != len(data_flat):
                raise ValueError(
                    f"Model output length {len(c2_flat)} does not match "
                    f"data length {len(data_flat)}"
                )
            if len(sigma_flat) != len(data_flat):
                raise ValueError(
                    f"Sigma length {len(sigma_flat)} does not match "
                    f"data length {len(data_flat)}"
                )
            residuals = (data_flat - c2_flat) / sigma_flat

            if JAX_AVAILABLE:
                chi_sq = jnp.sum(residuals**2)
                nll = 0.5 * chi_sq + 0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi * sigma_flat ** 2)
                )
                return float(nll)
            else:
                chi_sq = np.sum(np.asarray(residuals) ** 2)
                nll = 0.5 * chi_sq + 0.5 * np.sum(
                    np.log(2 * np.pi * np.asarray(sigma_flat) ** 2)
                )
                return float(nll)

        except (ValueError, ArithmeticError) as e:
            logger.warning("Likelihood computation failed: %s", e)
            return 1e10

    def detect_dataset_size(self, data: np.ndarray) -> str:
        """Detect and categorize dataset size.

        Args:
            data: Data array (any shape).

        Returns:
            One of ``"small"``, ``"medium"``, ``"large"``.
        """
        size = data.size
        category = DatasetSize.categorize(size)

        memory_mb = data.nbytes / (1024 * 1024)
        logger.info("Dataset size: %s points (%s)", f"{size:,}", category)
        logger.info("Estimated memory: %.1f MB", memory_mb)

        return category

    def validate_inputs(
        self,
        data: np.ndarray,
        sigma: np.ndarray | None,
        t: np.ndarray,
        phi: np.ndarray | float,
        q: float,
    ) -> None:
        """Validate fitting inputs.

        Args:
            data: Experimental data.
            sigma: Measurement uncertainties (None to skip sigma checks).
            t: Time array.
            phi: Angle(s).
            q: Scattering wavevector magnitude.

        Raises:
            ValueError: On invalid inputs.
        """
        if data.size == 0:
            raise ValueError("Data array is empty")

        if sigma is not None:
            if data.shape != sigma.shape:
                raise ValueError(
                    f"Data and sigma must have same shape "
                    f"(data={data.shape}, sigma={sigma.shape})"
                )
            if np.any(sigma <= 0):
                raise ValueError("All uncertainties must be positive")
            if not np.all(np.isfinite(sigma)):
                raise ValueError("Sigma contains non-finite values")

        if q <= 0:
            raise ValueError("q must be positive")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_parameter_info(self) -> dict[str, Any]:
        """Get parameter space information.

        Returns
        -------
        dict
            Dictionary with parameter_count, physical_bounds,
            physical_priors, scaling_bounds, scaling_priors, data_ranges.
        """
        return {
            "parameter_count": len(self.param_bounds),
            "physical_bounds": self.param_bounds,
            "physical_priors": self.param_priors,
            "scaling_bounds": {
                "contrast": self.parameter_space.contrast_bounds,
                "offset": self.parameter_space.offset_bounds,
            },
            "scaling_priors": {
                "contrast": self.parameter_space.contrast_prior,
                "offset": self.parameter_space.offset_prior,
            },
            "data_ranges": {
                "fitted": self.parameter_space.fitted_range,
                "theory": self.parameter_space.theory_range,
            },
        }


# Backward compatibility alias
ScaledFittingEngine = UnifiedHeterodyneEngine


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DatasetSize",
    "FitResult",
    "ParameterSpace",
    "ScaledFittingEngine",
    "UnifiedHeterodyneEngine",
    "solve_least_squares_chunked_jax",
    "solve_least_squares_general_jax",
    "solve_least_squares_jax",
]
