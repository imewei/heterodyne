"""NLSQ-optimized correlation computation for heterodyne model.

Provides functions tailored for the least-squares optimization loop:
- Flat residual computation (vectorized for scipy.optimize.least_squares)
- Efficient Jacobian-vector products
- Upper-triangle-only evaluation to halve computation for symmetric c2

These functions call the core jax_backend but add the adapter layer
needed by the NLSQ strategies.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

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
    return (c2_model_flat - c2_data_flat) * sqrt_weights_flat  # type: ignore[no-any-return]


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
        return compute_flat_residuals(  # type: ignore[no-any-return]
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
        return full_residual_fn(full_params)  # type: ignore[no-any-return]

    return varying_residual_fn, n_residuals


# Module-level cache: maps residual_fn id → (residual_fn, compiled jacfwd).
# Storing the *original* residual_fn alongside the compiled version keeps
# it alive, preventing Python from recycling its ``id()`` for a different
# object (which would silently return a stale compiled Jacobian).
# Bounded to _JAC_CACHE_MAX entries; least-recently-used entries evicted first.
_JAC_CACHE_MAX = 8
_jac_fn_cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()


def compute_nlsq_jacobian(
    residual_fn: Callable[..., jnp.ndarray],
    params: jnp.ndarray,
) -> np.ndarray:
    """Compute Jacobian of residual function via JAX forward-mode autodiff.

    Uses ``jax.jacfwd`` (forward-mode) rather than ``jax.jacobian``
    (reverse-mode default).  For a residual function with 14 input
    parameters and O(N²) outputs, forward mode runs 14 JVP passes
    instead of O(N²) VJP passes — substantially cheaper for the
    XPCS use-case where N=200--500.

    The compiled ``jacfwd`` closure is cached by the identity of
    ``residual_fn``, so repeated calls during an NLSQ iteration loop
    reuse the already-traced XLA computation without re-tracing.

    Args:
        residual_fn: JIT-compiled residual function (params -> residuals)
        params: Current parameter values

    Returns:
        Jacobian matrix, shape (n_residuals, n_params), as numpy array
    """
    fn_id = id(residual_fn)
    entry = _jac_fn_cache.get(fn_id)
    if entry is not None and entry[0] is residual_fn:
        # Cache hit — move to end for LRU ordering
        _jac_fn_cache.move_to_end(fn_id)
        compiled = entry[1]
    else:
        # jacfwd does n_params JVP passes; far cheaper than n_residuals VJPs.
        compiled = jax.jit(jax.jacfwd(residual_fn))
        # Evict LRU entry if cache is full
        if len(_jac_fn_cache) >= _JAC_CACHE_MAX:
            _jac_fn_cache.popitem(last=False)
        _jac_fn_cache[fn_id] = (residual_fn, compiled)
    jac = compiled(jnp.asarray(params))
    return np.asarray(jac)
