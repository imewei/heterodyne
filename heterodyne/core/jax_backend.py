"""JAX-accelerated computational backend for heterodyne correlation.

This module provides JIT-compiled functions for computing the two-component
heterodyne correlation function using the integral formulation (PNAS Eq. S-95).
All functions are designed to be stateless and compatible with JAX
transformations (jit, vmap, grad).

The correlation is computed as:
    c2 = offset + contrast × [ref + sample + cross] / f²

where transport terms use the integral of the rate J(t):
    half_tr[i,j] = exp(-½q² × |∫_{t_i}^{t_j} J_rate(t') dt'|)

The 14 model parameters in canonical order:
0: D0_ref, 1: alpha_ref, 2: D_offset_ref
3: D0_sample, 4: alpha_sample, 5: D_offset_sample
6: v0, 7: beta, 8: v_offset
9: f0, 10: f1, 11: f2, 12: f3
13: phi0
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    pass


@partial(jax.jit, static_argnames=("n_times",))
def compute_transport_jit(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    n_times: int,
) -> jnp.ndarray:
    """JIT-compiled pointwise transport coefficient computation.

    J(t) = D0 * t^alpha + offset

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses compute_transport_integral_matrix for the
        integral formulation (PNAS Eq. S-95). Retained for test
        compatibility and 1D visualization helpers.

    Args:
        t: Time array
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Constant offset
        n_times: Number of time points (static for JIT)

    Returns:
        Transport coefficient array
    """
    t_safe = jnp.maximum(t, 1e-10)
    t_power = jnp.power(t_safe, alpha)
    t_power = jnp.where(t > 0, t_power, 0.0)
    return D0 * t_power + offset


@jax.jit
def compute_g1_transport(
    J: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """JIT-compiled pointwise g1 correlation from transport coefficient.

    g1(t) = exp(-q² * J(t))

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses the integral formulation via
        compute_transport_integral_matrix. Retained for test
        compatibility and 1D visualization helpers.

    Args:
        J: Transport coefficient array
        q: Scattering wavevector

    Returns:
        g1 correlation array
    """
    return jnp.exp(-q * q * J)


@jax.jit
def compute_fraction_jit(
    t: jnp.ndarray,
    f0: float,
    f1: float,
    f2: float,
    f3: float,
) -> jnp.ndarray:
    """JIT-compiled sample fraction computation.

    f_s(t) = f0 * exp(f1 * (t - f2)) + f3, clipped to [0, 1]

    Args:
        t: Time array
        f0: Amplitude
        f1: Exponential rate
        f2: Time shift
        f3: Baseline

    Returns:
        Fraction array in [0, 1]
    """
    exponent = jnp.clip(f1 * (t - f2), -100, 100)
    fraction = f0 * jnp.exp(exponent) + f3
    return jnp.clip(fraction, 0.0, 1.0)


@jax.jit
def compute_velocity_integral_matrix(
    t: jnp.ndarray,
    v0: float,
    beta: float,
    v_offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled velocity integral matrix.

    Computes M[i,j] = ∫_{t_i}^{t_j} v(t') dt'
    where v(t) = v0 * t^beta + v_offset

    Uses cumsum for O(N) efficiency instead of O(N²) nested loops.

    Args:
        t: Time array, shape (N,)
        v0: Velocity prefactor
        beta: Velocity exponent
        v_offset: Velocity offset
        dt: Time step

    Returns:
        Integral matrix, shape (N, N)
    """
    # Compute velocity at each time point
    t_safe = jnp.maximum(t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, beta), 0.0)
    velocity = v0 * t_power + v_offset

    # Cumulative integral from t=0
    cumsum = jnp.cumsum(velocity) * dt

    # M[i,j] = cumsum[j] - cumsum[i] = integral from t_i to t_j
    # v_integral[i,i] = 0 by construction (outer-subtraction pattern)
    v_integral = cumsum[None, :] - cumsum[:, None]
    return v_integral


@jax.jit
def compute_transport_integral_matrix(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled transport integral matrix.

    Computes M[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|
    where J_rate(t) = D0 * t^alpha + offset

    Uses cumsum for O(N) efficiency instead of O(N²) nested loops.
    The absolute value ensures symmetric decay (direction-independent).

    Args:
        t: Time array, shape (N,)
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Transport rate offset
        dt: Time step

    Returns:
        Transport integral matrix, shape (N, N)
    """
    # Compute transport rate at each time point
    t_safe = jnp.maximum(t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0)
    J_rate = D0 * t_power + offset
    J_rate = jnp.maximum(J_rate, 0.0)

    # Cumulative integral from t=0
    cumsum = jnp.cumsum(J_rate) * dt

    # M[i,j] = |cumsum[j] - cumsum[i]| = |integral from t_i to t_j|
    return jnp.abs(cumsum[None, :] - cumsum[:, None])


@jax.jit
def compute_c2_heterodyne(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """JIT-compiled two-time heterodyne correlation computation.

    Computes c2 = offset + contrast × [ref + sample + cross] / f²

    Uses the integral formulation (PNAS Eq. S-95):
        half_tr[i,j] = exp(-½q² × |∫_{t_i}^{t_j} J_rate(t') dt'|)

    Self-terms use half_tr² to recover exp(-q²∫J), and cross-terms
    multiply half_tr_ref × half_tr_sample.

    Args:
        params: Parameter array of shape (14,) in canonical order:
            [D0_ref, alpha_ref, D_offset_ref,
             D0_sample, alpha_sample, D_offset_sample,
             v0, beta, v_offset,
             f0, f1, f2, f3,
             phi0]
        t: Time array, shape (N,)
        q: Scattering wavevector magnitude
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Correlation matrix c2, shape (N, N)
    """
    # Extract parameters
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    # Transport integral matrices via cumsum
    # J_integral[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|
    J_ref_integral = compute_transport_integral_matrix(
        t, D0_ref, alpha_ref, D_offset_ref, dt
    )
    J_sample_integral = compute_transport_integral_matrix(
        t, D0_sample, alpha_sample, D_offset_sample, dt
    )

    # Half-transport matrices: exp(-½q²∫J)
    q2 = q * q
    half_tr_ref = jnp.exp(-0.5 * q2 * J_ref_integral)
    half_tr_sample = jnp.exp(-0.5 * q2 * J_sample_integral)

    # Sample fraction: f_s(t) = f0 * exp(f1 * (t - f2)) + f3
    t_safe = jnp.maximum(t, 1e-10)
    exponent = jnp.clip(f1 * (t - f2), -100, 100)
    f_sample = jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)
    f_ref = 1.0 - f_sample

    # Velocity integral matrix
    velocity = v0 * jnp.where(t > 0, jnp.power(t_safe, beta), 0.0) + v_offset
    cumsum = jnp.cumsum(velocity) * dt
    v_integral = cumsum[None, :] - cumsum[:, None]

    # Combined phi angle: phi_angle from detector + phi0 from fit
    total_phi = phi_angle + phi0
    phi_rad = jnp.deg2rad(total_phi)

    # Phase factor: q * cos(phi) * velocity_integral
    phase = q * jnp.cos(phi_rad) * v_integral

    # Fraction matrices
    f_ref_matrix = f_ref[:, None] * f_ref[None, :]
    f_sample_matrix = f_sample[:, None] * f_sample[None, :]
    f_cross_vec = f_ref * f_sample
    f_cross_matrix = f_cross_vec[:, None] * f_cross_vec[None, :]

    # Reference term: f_ref_matrix² × half_tr_ref² (½×2 gives full q²∫J)
    ref_term = f_ref_matrix ** 2 * half_tr_ref ** 2

    # Sample term: f_sample_matrix² × half_tr_sample²
    sample_term = f_sample_matrix ** 2 * half_tr_sample ** 2

    # Cross term: 2 × f_cross × half_tr_ref × half_tr_sample × cos(phase)
    cross_term = (
        2.0 * f_cross_matrix * half_tr_ref * half_tr_sample * jnp.cos(phase)
    )

    # Normalization: f² = (f_s² + f_r²)_t1 * (f_s² + f_r²)_t2
    norm_1 = f_sample ** 2 + f_ref ** 2
    normalization = norm_1[:, None] * norm_1[None, :]

    # Full correlation: offset + contrast × [terms] / f²
    c2 = offset + contrast * (ref_term + sample_term + cross_term) / jnp.maximum(
        normalization, 1e-10
    )

    return c2


def compute_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute weighted residuals between model and data.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Flattened residual array
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )


@jax.jit
def _compute_residuals_jit(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """JIT-compiled residuals computation (always receives weights)."""
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    residuals = (c2_model - c2_data) * jnp.sqrt(weights)
    return residuals.ravel()  # type: ignore[no-any-return]


# Gradient of residuals with respect to parameters (for NLSQ)
_compute_residuals_jacobian_jit = jax.jit(
    jax.jacobian(_compute_residuals_jit, argnums=0)
)


def compute_residuals_jacobian(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute Jacobian of residuals with respect to parameters.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Jacobian matrix
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jacobian_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )
