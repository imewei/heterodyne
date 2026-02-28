"""Theory computations for heterodyne correlation model.

Physical model for two-component heterodyne correlation (PNAS Eq. S-95):

c₂(t₁,t₂,φ) = offset + contrast × [ref + sample + cross] / f²

where transport uses the integral of the rate J(t):
    half_tr[i,j] = exp(-½q² × |∫_{t_i}^{t_j} J_rate(t') dt'|)

- ref_term = f_r(t₁)²·f_r(t₂)² × half_tr_ref²
- sample_term = f_s(t₁)²·f_s(t₂)² × half_tr_sample²
- cross_term = 2·f_cross × half_tr_ref × half_tr_sample × cos(phase)
- f² = (f_s(t₁)² + f_r(t₁)²)·(f_s(t₂)² + f_r(t₂)²)

Transport rate: J_rate(t) = D0·t^α + offset (integrated numerically)
Fraction: f_s(t) = f0·exp(f1·(t-f2)) + f3
Velocity integral: ∫v(t)dt from t₁ to t₂
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def compute_transport_coefficient(
    t: jnp.ndarray | np.ndarray,
    D0: float,
    alpha: float,
    offset: float = 0.0,
) -> jnp.ndarray:
    """Compute transport coefficient J(t) = D0 * t^alpha + offset.

    Handles the singularity at t=0 for negative alpha by using
    the physical limit (J(0) = offset for alpha > 0, 0 for alpha <= 0).

    Args:
        t: Time array, shape (N,)
        D0: Diffusion prefactor
        alpha: Diffusion exponent
        offset: Constant offset

    Returns:
        Transport coefficient array, shape (N,)
    """
    t = jnp.asarray(t)

    # Handle t=0 singularity for negative alpha (matches jax_backend pattern)
    t_safe = jnp.maximum(t, 1e-10)

    # Compute t^alpha
    t_power = jnp.power(t_safe, alpha)

    # For t=0: if alpha > 0, t^alpha -> 0; if alpha <= 0, use 0 (physical limit)
    t_power = jnp.where(t > 0, t_power, 0.0)

    return D0 * t_power + offset


def compute_fraction(
    t: jnp.ndarray | np.ndarray,
    f0: float,
    f1: float,
    f2: float,
    f3: float,
) -> jnp.ndarray:
    """Compute sample fraction f_s(t) = f0 * exp(f1 * (t - f2)) + f3.

    Result is clipped to [0, 1] to ensure physical validity.

    Args:
        t: Time array, shape (N,)
        f0: Fraction amplitude
        f1: Exponential rate
        f2: Time shift
        f3: Baseline offset

    Returns:
        Sample fraction array, shape (N,), values in [0, 1]
    """
    t = jnp.asarray(t)
    exponent = jnp.clip(f1 * (t - f2), -100, 100)
    fraction = f0 * jnp.exp(exponent) + f3
    return jnp.clip(fraction, 0.0, 1.0)


def compute_g1_decay(
    J: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """Compute g1 field correlation from transport coefficient.

    g₁(t) = exp(-q² · J(t))

    Args:
        J: Transport coefficient array
        q: Scattering wavevector magnitude

    Returns:
        g1 correlation array
    """
    return jnp.exp(-q * q * J)


def compute_time_integral_matrix(
    values: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Compute cumulative integral matrix for velocity field.

    Returns matrix M where M[i,j] = ∫_{t_i}^{t_j} values(t) dt

    Uses cumsum broadcasting for O(N) computation instead of O(N²) nested loops.

    Args:
        values: Time-dependent values to integrate, shape (N,)
        dt: Time step

    Returns:
        Integral matrix, shape (N, N)
    """
    # Cumulative sum from t=0
    cumsum = jnp.cumsum(values) * dt

    # M[i,j] = cumsum[j] - cumsum[i] = integral from t_i to t_j
    # v_integral[i,i] = 0 by construction (outer-subtraction pattern)
    integral_matrix = cumsum[None, :] - cumsum[:, None]

    return integral_matrix


def compute_transport_integral_matrix(
    t: jnp.ndarray | np.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Compute transport integral matrix using rate values and cumsum.

    M[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|

    Uses compute_transport_coefficient for rate values and
    compute_time_integral_matrix for the cumsum pattern. Applies
    jnp.abs() for symmetric (direction-independent) decay and
    jnp.maximum(..., 0.0) for positivity.

    Args:
        t: Time array, shape (N,)
        D0: Diffusion prefactor
        alpha: Diffusion exponent
        offset: Constant offset for the rate
        dt: Time step

    Returns:
        Transport integral matrix, shape (N, N), non-negative and symmetric
    """
    J_rate = compute_transport_coefficient(t, D0, alpha, offset)
    J_rate = jnp.maximum(J_rate, 0.0)
    integral_matrix = compute_time_integral_matrix(J_rate, dt)
    return jnp.abs(integral_matrix)


def compute_velocity_field(
    t: jnp.ndarray | np.ndarray,
    v0: float,
    beta: float,
    v_offset: float,
) -> jnp.ndarray:
    """Compute velocity field v(t) = v0 * t^beta + v_offset.

    Args:
        t: Time array, shape (N,)
        v0: Velocity prefactor
        beta: Velocity exponent
        v_offset: Constant velocity offset

    Returns:
        Velocity array, shape (N,)
    """
    t = jnp.asarray(t)
    t_safe = jnp.where(t > 0, t, 1e-10)
    t_power = jnp.where(t > 0, jnp.power(t_safe, beta), 0.0)
    return v0 * t_power + v_offset


def compute_cross_term_phase(
    velocity_integral: jnp.ndarray,
    q: float,
    phi: float,
) -> jnp.ndarray:
    """Compute cross-term phase factor.

    phase = q * cos(phi) * ∫v(t)dt

    Args:
        velocity_integral: Integral matrix from compute_time_integral_matrix
        q: Scattering wavevector magnitude
        phi: Flow angle (degrees)

    Returns:
        Phase matrix, shape (N, N)
    """
    phi_rad = jnp.deg2rad(phi)
    return q * jnp.cos(phi_rad) * velocity_integral


def compute_normalization_factor(
    f_s_1: jnp.ndarray,
    f_s_2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute normalization factor f² for correlation.

    f² = (f_s(t₁)² + f_r(t₁)²) · (f_s(t₂)² + f_r(t₂)²)

    where f_r(t) = 1 - f_s(t)

    Args:
        f_s_1: Sample fraction at t1 times, shape (N1,)
        f_s_2: Sample fraction at t2 times, shape (N2,)

    Returns:
        Normalization matrix, shape (N1, N2)
    """
    f_r_1 = 1.0 - f_s_1
    f_r_2 = 1.0 - f_s_2

    # (f_s² + f_r²) at each time
    norm_1 = f_s_1 ** 2 + f_r_1 ** 2  # shape (N1,)
    norm_2 = f_s_2 ** 2 + f_r_2 ** 2  # shape (N2,)

    # Outer product for matrix
    return norm_1[:, None] * norm_2[None, :]
