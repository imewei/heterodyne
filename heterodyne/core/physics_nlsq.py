"""NLSQ-optimized correlation computation for heterodyne model.

Provides functions tailored for the least-squares optimization loop:
- Flat residual computation (vectorized for scipy.optimize.least_squares)
- Efficient Jacobian-vector products
- Upper-triangle-only evaluation to halve computation for symmetric c2

These functions call the core jax_backend but add the adapter layer
needed by the NLSQ strategies.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@partial(jax.jit, static_argnames=("n_times",))
def compute_flat_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data_flat: jnp.ndarray,
    sqrt_weights_flat: jnp.ndarray,
    contrast: float,
    offset: float,
    triu_i: jnp.ndarray,
    triu_j: jnp.ndarray,
    n_times: int,
) -> jnp.ndarray:
    """Compute flat residuals using only upper-triangle elements.

    Instead of computing the full N×N c2 matrix and flattening,
    this extracts only the upper-triangle elements for the residual
    vector. This halves memory for large matrices.

    Args:
        params: Parameter array, shape (14,)
        t: Time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        c2_data_flat: Flattened upper-triangle data
        sqrt_weights_flat: Square-root weights for the same elements
        contrast: Speckle contrast
        offset: Baseline offset
        triu_i: Row indices of upper triangle
        triu_j: Column indices of upper triangle
        n_times: Number of time points (static for JIT)

    Returns:
        Weighted residual vector, shape (n_triu,)
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    c2_model_flat = c2_model[triu_i, triu_j]
    return (c2_model_flat - c2_data_flat) * sqrt_weights_flat


def make_residual_fn(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: np.ndarray,
    weights: np.ndarray | None = None,
    contrast: float = 0.5,
    offset: float = 1.0,
    use_upper_triangle: bool = True,
) -> tuple:
    """Create a JIT-compiled residual function for NLSQ optimization.

    Returns a function mapping 14-param vector → flat residuals,
    plus precomputed indexing arrays.

    Args:
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data, shape (N, N)
        weights: Optional weight matrix, shape (N, N)
        contrast: Speckle contrast
        offset: Baseline offset
        use_upper_triangle: Only use upper triangle (recommended)

    Returns:
        Tuple of (residual_fn, n_residuals) where residual_fn takes
        params (14,) and returns residuals (n_residuals,)
    """
    n = c2_data.shape[0]

    if use_upper_triangle:
        triu_i, triu_j = np.triu_indices(n, k=0)
    else:
        triu_i, triu_j = np.indices((n, n))
        triu_i, triu_j = triu_i.ravel(), triu_j.ravel()

    c2_flat = jnp.asarray(c2_data[triu_i, triu_j])
    n_residuals = len(triu_i)

    if weights is not None:
        w_flat = jnp.asarray(weights[triu_i, triu_j])
    else:
        w_flat = jnp.ones(n_residuals)

    sqrt_w = jnp.sqrt(jnp.maximum(w_flat, 0.0))
    triu_i_jax = jnp.asarray(triu_i)
    triu_j_jax = jnp.asarray(triu_j)
    t_jax = jnp.asarray(t)

    @jax.jit
    def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
        return compute_flat_residuals(
            params,
            t_jax,
            q,
            dt,
            phi_angle,
            c2_flat,
            sqrt_w,
            contrast,
            offset,
            triu_i_jax,
            triu_j_jax,
            n,
        )

    return residual_fn, n_residuals


def make_varying_residual_fn(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: np.ndarray,
    varying_indices: list[int] | np.ndarray,
    fixed_values: np.ndarray,
    weights: np.ndarray | None = None,
    contrast: float = 0.5,
    offset: float = 1.0,
) -> tuple:
    """Create residual function that accepts only varying parameters.

    Combines ParameterTransform-like expand with the residual computation
    in a single JIT-compiled function for maximum efficiency.

    Args:
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental data, shape (N, N)
        varying_indices: Indices of varying params in the 14-param array
        fixed_values: Full 14-param array with fixed values
        weights: Optional weights
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Tuple of (residual_fn, n_residuals) where residual_fn takes
        varying_params (n_varying,) and returns residuals
    """
    full_residual_fn, n_residuals = make_residual_fn(
        t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )

    vary_idx = jnp.asarray(np.asarray(varying_indices))
    fixed_jax = jnp.asarray(fixed_values)

    @jax.jit
    def varying_residual_fn(varying_params: jnp.ndarray) -> jnp.ndarray:
        full_params = fixed_jax.at[vary_idx].set(varying_params)
        return full_residual_fn(full_params)

    return varying_residual_fn, n_residuals


def compute_nlsq_jacobian(
    residual_fn,
    params: jnp.ndarray,
) -> np.ndarray:
    """Compute Jacobian of residual function via JAX autodiff.

    Args:
        residual_fn: JIT-compiled residual function (params → residuals)
        params: Current parameter values

    Returns:
        Jacobian matrix, shape (n_residuals, n_params), as numpy array
    """
    jac_fn = jax.jit(jax.jacobian(residual_fn))
    jac = jac_fn(jnp.asarray(params))
    return np.asarray(jac)
