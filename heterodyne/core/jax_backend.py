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

from heterodyne.core.physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_time_integral_matrix,
    smooth_abs,
    trapezoid_cumsum,
)

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
    """JIT-compiled velocity integral matrix (NLSQ meshgrid path).

    Computes M[i,j] = ∫_{t_i}^{t_j} v(t') dt'
    where v(t) = v0 * t^beta + v_offset

    Uses shared ``trapezoid_cumsum`` → ``create_time_integral_matrix``
    pipeline for O(N) efficiency and O(dt²) accuracy.
    The velocity integral is *signed* (not absolute-valued) because it
    feeds into the phase factor ``cos(q cos(φ) ∫v dt)``.

    Args:
        t: Time array, shape (N,)
        v0: Velocity prefactor
        beta: Velocity exponent
        v_offset: Velocity offset
        dt: Time step

    Returns:
        Signed integral matrix, shape (N, N)
    """
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    cumsum = trapezoid_cumsum(velocity, dt)
    return create_time_integral_matrix(cumsum)


@jax.jit
def compute_transport_integral_matrix(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled transport integral matrix (NLSQ meshgrid path).

    Computes M[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|
    where J_rate(t) = D0 * t^alpha + offset

    Uses shared ``compute_transport_rate`` → ``trapezoid_cumsum`` →
    ``create_time_integral_matrix`` → ``smooth_abs`` pipeline.

    Args:
        t: Time array, shape (N,)
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Transport rate offset
        dt: Time step

    Returns:
        Transport integral matrix, shape (N, N)
    """
    J_rate = compute_transport_rate(t, D0, alpha, offset)
    cumsum = trapezoid_cumsum(J_rate, dt)
    diff = create_time_integral_matrix(cumsum)
    return smooth_abs(diff)


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

    # Transport integral matrices via shared cumsum → meshgrid pipeline
    # J_integral[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|
    J_ref_rate = compute_transport_rate(t, D0_ref, alpha_ref, D_offset_ref)
    J_sample_rate = compute_transport_rate(t, D0_sample, alpha_sample, D_offset_sample)
    ref_cumsum = trapezoid_cumsum(J_ref_rate, dt)
    sample_cumsum = trapezoid_cumsum(J_sample_rate, dt)
    J_ref_integral = smooth_abs(create_time_integral_matrix(ref_cumsum))
    J_sample_integral = smooth_abs(create_time_integral_matrix(sample_cumsum))

    # Half-transport matrices: exp(-½q²∫J) with log-space clipping for
    # numerical safety — prevents underflow for extreme D0 values
    q2 = q * q
    log_half_tr_ref = jnp.clip(-0.5 * q2 * J_ref_integral, -700.0, 0.0)
    log_half_tr_sample = jnp.clip(-0.5 * q2 * J_sample_integral, -700.0, 0.0)
    half_tr_ref = jnp.exp(log_half_tr_ref)
    half_tr_sample = jnp.exp(log_half_tr_sample)

    # Sample fraction: f_s(t) = f0 * exp(f1 * (t - f2)) + f3
    exponent = jnp.clip(f1 * (t - f2), -100, 100)
    f_sample = jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)
    f_ref = 1.0 - f_sample

    # Velocity integral matrix via shared cumsum → meshgrid pipeline
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    v_cumsum = trapezoid_cumsum(velocity, dt)
    v_integral = create_time_integral_matrix(v_cumsum)

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
    ref_term = f_ref_matrix**2 * half_tr_ref**2

    # Sample term: f_sample_matrix² × half_tr_sample²
    sample_term = f_sample_matrix**2 * half_tr_sample**2

    # Cross term: 2 × f_cross × half_tr_ref × half_tr_sample × cos(phase)
    cross_term = 2.0 * f_cross_matrix * half_tr_ref * half_tr_sample * jnp.cos(phase)

    # Normalization: f² = (f_s² + f_r²)_t1 * (f_s² + f_r²)_t2
    norm_1 = f_sample**2 + f_ref**2
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


@jax.jit
def compute_chi_squared(
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
    """JIT-compiled chi-squared computation.

    chi² = sum((c2_model - c2_data)² × weights)

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Weights (1/uncertainty²)
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Chi-squared scalar
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    return jnp.sum((c2_model - c2_data) ** 2 * weights)  # type: ignore[no-any-return]


def batch_chi_squared(
    params_batch: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float = 1.0,
    offset: float = 1.0,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Vectorized chi-squared over a batch of parameter sets.

    Uses ``jax.vmap`` for efficient parallel evaluation.  For large batches
    or large time grids, ``chunk_size`` limits simultaneous N×N allocations
    to prevent XLA memory exhaustion (each vmap'd evaluation allocates
    multiple N×N intermediate matrices).

    Args:
        params_batch: Parameter sets, shape ``(n_sets, 14)``.
        t: Time array, shape ``(N,)``.
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle.
        c2_data: Experimental data.
        weights: Weights.
        contrast: Speckle contrast.
        offset: Baseline offset.
        chunk_size: Max batch elements to vmap simultaneously.  ``None``
            (default) auto-selects based on time-grid size: ``max(1,
            200 // (N // 100))`` to keep peak memory under ~1.6 GB.

    Returns:
        Chi-squared values, shape ``(n_sets,)``.
    """
    n_sets = params_batch.shape[0]
    n_times = t.shape[0]

    def single_chi2(params: jnp.ndarray) -> jnp.ndarray:
        return compute_chi_squared(
            params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
        )

    if chunk_size is None:
        # Auto-select: each evaluation creates ~12 N×N float64 matrices
        # (half_tr, cumsum, integral matrix, ref/sample/cross terms, plus
        # XLA intermediates) → ~96 N² bytes per evaluation.
        # Target peak ≈ 1.6 GB → chunk_size ≈ 1.6e9 / (96 * N²).
        matrix_bytes = 96 * n_times * n_times
        chunk_size = max(1, int(1.6e9 / max(matrix_bytes, 1)))

    if n_sets <= chunk_size:
        return jax.vmap(single_chi2)(params_batch)  # type: ignore[no-any-return]

    # Chunked evaluation to bound peak memory
    chunks = []
    for start in range(0, n_sets, chunk_size):
        chunk = params_batch[start : start + chunk_size]
        chunks.append(jax.vmap(single_chi2)(chunk))
    return jnp.concatenate(chunks)  # type: ignore[no-any-return]


@jax.jit
def compute_multi_angle_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angles: jnp.ndarray,
    c2_data_batch: jnp.ndarray,
    weights_batch: jnp.ndarray,
    contrasts: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled residuals for multiple phi angles simultaneously.

    Args:
        params: Parameter array, shape (14,)
        t: Time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angles: Phi angles, shape (n_phi,)
        c2_data_batch: Experimental data, shape (n_phi, N, N)
        weights_batch: Weights, shape (n_phi, N, N)
        contrasts: Per-angle contrasts, shape (n_phi,)
        offsets: Per-angle offsets, shape (n_phi,)

    Returns:
        Stacked flattened residuals, shape (n_phi × N × N,)
    """

    def single_angle_residual(
        phi: jnp.ndarray,
        c2_exp: jnp.ndarray,
        w: jnp.ndarray,
        c: jnp.ndarray,
        o: jnp.ndarray,
    ) -> jnp.ndarray:
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi, c, o)
        return ((c2_model - c2_exp) * jnp.sqrt(w)).ravel()

    compute_all = jax.vmap(single_angle_residual, in_axes=(0, 0, 0, 0, 0))
    residuals_batch = compute_all(
        phi_angles, c2_data_batch, weights_batch, contrasts, offsets
    )
    return residuals_batch.ravel()  # type: ignore[no-any-return]


# Gradient of chi-squared with respect to parameters
compute_chi_squared_grad = jax.jit(jax.grad(compute_chi_squared, argnums=0))

# Hessian of chi-squared (for uncertainty estimation)
compute_chi_squared_hessian = jax.jit(jax.hessian(compute_chi_squared, argnums=0))


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
