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

from heterodyne.core.physics_utils import (
    create_time_integral_matrix,
    smooth_abs,
    trapezoid_cumsum,
)


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

    # Handle t=0 singularity for negative alpha (matches jax_backend pattern).
    # Use jnp.where instead of jnp.maximum to preserve gradients at the floor
    # (jnp.maximum zeros the gradient when t < 1e-10).
    t_safe = jnp.where(t > 1e-10, t, 1e-10)

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

    Delegates to shared ``trapezoid_cumsum`` → ``create_time_integral_matrix``
    pipeline for O(N) computation and O(dt²) accuracy.

    Args:
        values: Time-dependent values to integrate, shape (N,)
        dt: Time step

    Returns:
        Integral matrix, shape (N, N)
    """
    cumsum = trapezoid_cumsum(values, dt)
    return create_time_integral_matrix(cumsum)


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
    # Physical positivity floor: jnp.maximum is correct here because the
    # subgradient at J_rate=0 is 1 (gradient of offset passes through), while
    # jnp.where(J_rate > 0.0, J_rate, 0.0) would block it at the boundary.
    J_rate = jnp.maximum(J_rate, 0.0)
    integral_matrix = compute_time_integral_matrix(J_rate, dt)
    return smooth_abs(integral_matrix)


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
    norm_1 = f_s_1**2 + f_r_1**2  # shape (N1,)
    norm_2 = f_s_2**2 + f_r_2**2  # shape (N2,)

    # Outer product for matrix
    return norm_1[:, None] * norm_2[None, :]


class TheoryEngine:
    """High-level validated API for heterodyne correlation computation.

    Wraps compute_c2_heterodyne with input validation, error handling,
    and fallback logic. Use this instead of calling jax_backend directly
    for production code.
    """

    def __init__(
        self,
        t: jnp.ndarray,
        q: float,
        dt: float,
        n_params: int = 14,
    ) -> None:
        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.t = jnp.asarray(t)
        self.q = float(q)
        self.dt = float(dt)
        self.n_params = n_params

    def compute_correlation(
        self,
        params: jnp.ndarray,
        phi_angle: float = 0.0,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute c2 with validation."""
        params = jnp.asarray(params)
        if params.shape != (self.n_params,):
            raise ValueError(
                f"Expected {self.n_params} params, got shape {params.shape}"
            )

        from heterodyne.core.jax_backend import compute_c2_heterodyne

        c2 = compute_c2_heterodyne(
            params, self.t, self.q, self.dt, phi_angle, contrast, offset
        )

        if not jnp.isfinite(c2).all():
            raise ValueError("Non-finite values in computed c2 — check parameters")
        return c2  # type: ignore[no-any-return]

    def compute_residuals(
        self,
        params: jnp.ndarray,
        c2_data: jnp.ndarray,
        phi_angle: float = 0.0,
        weights: jnp.ndarray | None = None,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute weighted residuals with validation."""
        from heterodyne.core.jax_backend import compute_residuals

        return compute_residuals(
            jnp.asarray(params),
            self.t,
            self.q,
            self.dt,
            phi_angle,
            jnp.asarray(c2_data),
            weights,
            contrast,
            offset,
        )

    def compute_chi_squared(
        self,
        params: jnp.ndarray,
        c2_data: jnp.ndarray,
        phi_angle: float = 0.0,
        weights: jnp.ndarray | None = None,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> float:
        """Compute chi-squared goodness-of-fit statistic.

        chi² = sum(residuals²) where residuals are weighted.

        Args:
            params: Parameter array, shape (14,)
            c2_data: Experimental correlation data, shape (N, N)
            phi_angle: Detector phi angle (degrees)
            weights: Optional weights (1/uncertainty²)
            contrast: Speckle contrast
            offset: Baseline offset

        Returns:
            Chi-squared value (scalar)
        """
        residuals = self.compute_residuals(
            params, c2_data, phi_angle, weights, contrast, offset
        )
        return float(jnp.sum(residuals**2))

    def compute_batch_chi_squared(
        self,
        params_batch: jnp.ndarray | np.ndarray,
        c2_data: jnp.ndarray,
        phi_angle: float = 0.0,
        weights: jnp.ndarray | None = None,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> np.ndarray:
        """Compute chi-squared for multiple parameter sets.

        Args:
            params_batch: Array of parameter sets, shape (n_sets, 14)
            c2_data: Experimental correlation data, shape (N, N)
            phi_angle: Detector phi angle (degrees)
            weights: Optional weights
            contrast: Speckle contrast
            offset: Baseline offset

        Returns:
            Chi-squared values, shape (n_sets,)
        """
        params_batch = np.asarray(params_batch)
        if params_batch.ndim != 2:
            raise ValueError("params_batch must be 2D (n_sets, n_params)")
        if params_batch.shape[1] != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} params per set, got {params_batch.shape[1]}"
            )

        results = np.empty(params_batch.shape[0])
        for i, params in enumerate(params_batch):
            results[i] = self.compute_chi_squared(
                jnp.array(params), c2_data, phi_angle, weights, contrast, offset
            )
        return results

    def estimate_computation_cost(
        self,
        n_phi: int = 1,
    ) -> dict[str, object]:
        """Estimate computational cost for current data dimensions.

        Args:
            n_phi: Number of phi angles

        Returns:
            Dict with cost estimates
        """
        n_times = len(self.t)
        n_matrix_elements = n_times * n_times
        n_total = n_matrix_elements * n_phi

        # Heterodyne model: ~50 ops/element (transport integrals, fractions, cross term)
        ops_per_element = 50
        total_ops = n_total * ops_per_element

        # Memory: 8 bytes × ~6 intermediate matrices per angle
        memory_mb = (n_matrix_elements * 8 * 6 * n_phi) / (1024**2)

        tier = (
            "light" if total_ops < 1e6 else ("medium" if total_ops < 1e8 else "heavy")
        )

        return {
            "n_times": n_times,
            "n_phi": n_phi,
            "n_matrix_elements": n_matrix_elements,
            "n_total_points": n_total,
            "estimated_operations": total_ops,
            "estimated_memory_mb": memory_mb,
            "performance_tier": tier,
            "n_params": self.n_params,
        }

    def __repr__(self) -> str:
        return (
            f"TheoryEngine(n_times={len(self.t)}, q={self.q:.4e}, "
            f"dt={self.dt:.4e}, n_params={self.n_params})"
        )


# ---------------------------------------------------------------------------
# Convenience module-level functions
# ---------------------------------------------------------------------------


def compute_c2_theory(
    params: np.ndarray,
    t: np.ndarray,
    q: float,
    dt: float,
    phi_angle: float = 0.0,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> np.ndarray:
    """Convenience: compute c2 from parameters in one call.

    Creates a TheoryEngine, computes, returns numpy array.
    For repeated calls, create a TheoryEngine directly.

    Args:
        params: 14-parameter array
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Correlation matrix c2, shape (N, N)
    """
    engine = TheoryEngine(t=jnp.asarray(t), q=q, dt=dt)
    c2 = engine.compute_correlation(jnp.asarray(params), phi_angle, contrast, offset)
    return np.asarray(c2)


def compute_chi2_theory(
    params: np.ndarray,
    t: np.ndarray,
    q: float,
    dt: float,
    c2_data: np.ndarray,
    phi_angle: float = 0.0,
    weights: np.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> float:
    """Convenience: compute chi-squared in one call.

    Args:
        params: 14-parameter array
        t: Time array
        q: Scattering wavevector
        dt: Time step
        c2_data: Experimental data, shape (N, N)
        phi_angle: Detector phi angle (degrees)
        weights: Optional weights
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Chi-squared value
    """
    engine = TheoryEngine(t=jnp.asarray(t), q=q, dt=dt)
    w = jnp.asarray(weights) if weights is not None else None
    return engine.compute_chi_squared(
        jnp.asarray(params),
        jnp.asarray(c2_data),
        phi_angle,
        w,
        contrast,
        offset,
    )


__all__ = [
    "compute_transport_coefficient",
    "compute_fraction",
    "compute_g1_decay",
    "compute_time_integral_matrix",
    "compute_transport_integral_matrix",
    "compute_velocity_field",
    "compute_cross_term_phase",
    "compute_normalization_factor",
    "TheoryEngine",
    "compute_c2_theory",
    "compute_chi2_theory",
]
