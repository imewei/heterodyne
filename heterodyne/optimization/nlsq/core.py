"""Core NLSQ fitting function for heterodyne analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.adapter import NLSQAdapter, ScipyNLSQAdapter
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.multistart import MultiStartOptimizer
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel

logger = get_logger(__name__)


def fit_nlsq_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
    use_nlsq_library: bool = True,
) -> NLSQResult:
    """Fit heterodyne model to correlation data using NLSQ.

    Args:
        model: HeterodyneModel instance with parameters configured
        c2_data: Experimental correlation data, shape (N, N)
        phi_angle: Detector phi angle (degrees)
        config: NLSQ configuration (default if None)
        weights: Optional weights (1/sigma²) for weighted least squares
        use_nlsq_library: Whether to use nlsq library (True) or scipy fallback

    Returns:
        NLSQResult with fitted parameters and diagnostics
    """
    if config is None:
        config = NLSQConfig()

    logger.info(f"Starting NLSQ fit: phi={phi_angle}°, method={config.method}")

    # Get parameter info
    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_varying = param_manager.n_varying

    logger.info(f"Fitting {n_varying} parameters: {varying_names}")

    # Get initial values and bounds for varying parameters
    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()

    # Ensure initial values are within bounds
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    # Convert data to JAX arrays with explicit float64 for consistency
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None

    # Get fixed values for reconstruction (float64 for JAX scatter compatibility)
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices)
    n_data = c2_jax.size  # Total number of residuals

    # Capture constants for JAX residual function
    t = model.t
    q = model.q
    dt = model.dt

    # Create JAX-compatible residual function for nlsq
    def jax_residual_fn(x: jnp.ndarray, *varying_params) -> jnp.ndarray:
        """Pure JAX residual function for nlsq tracing."""
        # Reconstruct full parameter array using JAX ops (explicit float64)
        varying_array = jnp.array(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_array)

        # Compute residuals
        residuals = compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
        )
        return residuals

    # Create numpy residual function for scipy fallback
    def numpy_residual_fn(varying_params: np.ndarray) -> np.ndarray:
        """Numpy residual function for scipy."""
        full_params = param_manager.get_full_values().copy()
        for i, idx in enumerate(param_manager.varying_indices):
            full_params[idx] = varying_params[i]

        residuals = compute_residuals(
            jnp.asarray(full_params),
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
        )
        return np.asarray(residuals)

    # Select adapter and run optimization
    if use_nlsq_library:
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)

            if config.multistart:
                # Use scipy for multistart (more reliable)
                optimizer = MultiStartOptimizer(
                    adapter=adapter,
                    n_starts=config.multistart_n,
                )
                multi_result = optimizer.fit(
                    residual_fn=numpy_residual_fn,
                    initial_params=initial_varying,
                    bounds=(lower_bounds, upper_bounds),
                    config=config,
                )
                result = multi_result.best_result
                result.metadata["multistart"] = {
                    "n_starts": multi_result.n_total,
                    "n_successful": multi_result.n_successful,
                }
            else:
                # Use JAX-traced optimization with nlsq
                result = adapter.fit_jax(
                    jax_residual_fn=jax_residual_fn,
                    initial_params=initial_varying,
                    bounds=(lower_bounds, upper_bounds),
                    config=config,
                    n_data=n_data,
                )
        except ImportError:
            logger.warning("nlsq library not available, falling back to scipy")
            adapter = ScipyNLSQAdapter(parameter_names=varying_names)
            result = adapter.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )
    else:
        adapter = ScipyNLSQAdapter(parameter_names=varying_names)

        if config.multistart:
            optimizer = MultiStartOptimizer(
                adapter=adapter,
                n_starts=config.multistart_n,
            )
            multi_result = optimizer.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )
            result = multi_result.best_result
            result.metadata["multistart"] = {
                "n_starts": multi_result.n_total,
                "n_successful": multi_result.n_successful,
            }
        else:
            result = adapter.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )

    # Compute fitted correlation for output
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            phi_angle,
        )
        result.fitted_correlation = np.asarray(fitted_c2)

        # Update model with fitted values
        model.set_params(full_fitted)

    cost_str = f"{result.final_cost:.4e}" if result.final_cost is not None else "N/A"
    logger.info(f"NLSQ fit complete: success={result.success}, cost={cost_str}")

    return result


def fit_nlsq_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float] | np.ndarray,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> list[NLSQResult]:
    """Fit model to correlation data at multiple phi angles.

    Each angle is fit independently using the same parameter configuration.

    Args:
        model: HeterodyneModel instance
        c2_data: Correlation data, shape (n_phi, N, N) or list of (N, N)
        phi_angles: Array of phi angles
        config: NLSQ configuration
        weights: Optional weights

    Returns:
        List of NLSQResult, one per angle
    """
    phi_angles = np.asarray(phi_angles)

    if c2_data.ndim == 2:
        c2_data = c2_data[np.newaxis, ...]

    if len(c2_data) != len(phi_angles):
        raise ValueError(
            f"Number of c2 matrices ({len(c2_data)}) doesn't match "
            f"number of phi angles ({len(phi_angles)})"
        )

    results = []
    for i, phi in enumerate(phi_angles):
        logger.info(f"Fitting phi angle {i+1}/{len(phi_angles)}: {phi}°")

        c2_i = c2_data[i]
        weights_i = weights[i] if weights is not None and weights.ndim == 3 else weights

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_i,
            phi_angle=float(phi),
            config=config,
            weights=weights_i,
        )
        result.metadata["phi_angle"] = float(phi)
        results.append(result)

    return results
