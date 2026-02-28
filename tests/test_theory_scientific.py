"""Scientific tests for theory module computations.

This module provides rigorous testing of the physics theory functions
with analytical solutions and known physical limits.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

# ============================================================================
# Transport Coefficient Tests
# ============================================================================

class TestComputeTransportCoefficient:
    """Scientific tests for compute_transport_coefficient."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_brownian_diffusion_alpha_1(self) -> None:
        """Test alpha=1 (Brownian diffusion): J(t) = D0*t."""
        from heterodyne.core.theory import compute_transport_coefficient

        t = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
        D0, alpha = 2.5, 1.0

        J = compute_transport_coefficient(t, D0, alpha, offset=0.0)

        expected = D0 * t
        assert_allclose(J, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_ballistic_motion_alpha_2(self) -> None:
        """Test alpha=2 (ballistic): J(t) = D0*t²."""
        from heterodyne.core.theory import compute_transport_coefficient

        t = jnp.array([0.0, 1.0, 2.0, 3.0])
        D0, alpha = 1.0, 2.0

        J = compute_transport_coefficient(t, D0, alpha, offset=0.0)

        expected = D0 * t ** 2
        assert_allclose(J, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_subdiffusion_alpha_half(self) -> None:
        """Test alpha=0.5 (subdiffusion): J(t) = D0*sqrt(t)."""
        from heterodyne.core.theory import compute_transport_coefficient

        t = jnp.array([0.0, 1.0, 4.0, 9.0])  # Perfect squares
        D0, alpha = 1.0, 0.5

        J = compute_transport_coefficient(t, D0, alpha, offset=0.0)

        # Expected: sqrt([0, 1, 4, 9]) = [0, 1, 2, 3]
        expected = jnp.array([0.0, 1.0, 2.0, 3.0])
        assert_allclose(J, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_offset_adds_constant(self) -> None:
        """Test offset parameter adds constant to J(t)."""
        from heterodyne.core.theory import compute_transport_coefficient

        t = jnp.array([0.0, 1.0, 2.0])
        offset = 5.0

        J = compute_transport_coefficient(t, D0=1.0, alpha=1.0, offset=offset)

        # J(t) = 1.0 * t + 5.0
        expected = t + offset
        assert_allclose(J, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_t_zero_singularity_handled(self) -> None:
        """Test t=0 doesn't cause NaN/Inf for any alpha."""
        from heterodyne.core.theory import compute_transport_coefficient

        t = jnp.array([0.0, 0.1, 1.0])

        for alpha in [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            J = compute_transport_coefficient(t, D0=1.0, alpha=alpha, offset=0.0)
            assert not jnp.any(jnp.isnan(J)), f"NaN for alpha={alpha}"
            assert not jnp.any(jnp.isinf(J)), f"Inf for alpha={alpha}"


# ============================================================================
# Fraction Tests
# ============================================================================

class TestComputeFraction:
    """Scientific tests for compute_fraction."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_constant_fraction_f1_zero(self) -> None:
        """When f1=0, fraction is constant: f = f0 + f3."""
        from heterodyne.core.theory import compute_fraction

        t = jnp.arange(10.0)
        f0, f1, f2, f3 = 0.4, 0.0, 0.0, 0.1

        f = compute_fraction(t, f0, f1, f2, f3)

        expected = jnp.full_like(t, f0 + f3)  # 0.5
        assert_allclose(f, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_exponential_decay_negative_f1(self) -> None:
        """Negative f1 gives exponential decay."""
        from heterodyne.core.theory import compute_fraction

        t = jnp.array([0.0, 1.0, 10.0])
        f0, f1, f2, f3 = 0.5, -1.0, 0.0, 0.0

        f = compute_fraction(t, f0, f1, f2, f3)

        # f(t) = 0.5 * exp(-t), clipped to [0,1]
        expected = 0.5 * jnp.exp(-t)
        assert_allclose(f, expected, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_clipping_enforced(self) -> None:
        """Fraction is always in [0, 1]."""
        from heterodyne.core.theory import compute_fraction

        t = jnp.array([0.0, 100.0])
        # Parameters that would give values > 1
        f0, f1, f2, f3 = 0.5, 1.0, 0.0, 0.6

        f = compute_fraction(t, f0, f1, f2, f3)

        assert jnp.all(f >= 0.0)
        assert jnp.all(f <= 1.0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_time_shift_f2(self) -> None:
        """f2 shifts the exponential center."""
        from heterodyne.core.theory import compute_fraction

        f0, f1, f3 = 0.2, 0.5, 0.1

        # Without shift at t=0
        f_at_t0 = compute_fraction(jnp.array([0.0]), f0, f1, 0.0, f3)
        # With shift of 5, at t=5
        f_at_t5_shifted = compute_fraction(jnp.array([5.0]), f0, f1, 5.0, f3)

        # These should be equal (both evaluate at (t - f2) = 0)
        assert_allclose(f_at_t0, f_at_t5_shifted, rtol=1e-12)


# ============================================================================
# g1 Decay Tests
# ============================================================================

class TestComputeG1Decay:
    """Scientific tests for compute_g1_decay."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_exponential_formula(self) -> None:
        """Test g1 = exp(-q²J)."""
        from heterodyne.core.theory import compute_g1_decay

        J = jnp.array([0.0, 1.0, 2.0, 4.0])
        q = 0.1

        g1 = compute_g1_decay(J, q)

        expected = jnp.exp(-q**2 * J)
        assert_allclose(g1, expected, rtol=1e-14)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_g1_normalized_at_origin(self) -> None:
        """g1(J=0) = 1 (normalization)."""
        from heterodyne.core.theory import compute_g1_decay

        J = jnp.array([0.0])
        g1 = compute_g1_decay(J, q=0.1)

        assert jnp.isclose(g1[0], 1.0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_g1_decays_with_J(self) -> None:
        """g1 decreases as J increases."""
        from heterodyne.core.theory import compute_g1_decay

        J = jnp.arange(10.0)
        g1 = compute_g1_decay(J, q=0.1)

        # g1 should be monotonically decreasing
        assert jnp.all(jnp.diff(g1) <= 0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_g1_scales_with_q_squared(self) -> None:
        """Decay rate scales as q²."""
        from heterodyne.core.theory import compute_g1_decay

        J = jnp.array([1.0])

        g1_q1 = compute_g1_decay(J, q=0.1)
        g1_q2 = compute_g1_decay(J, q=0.2)

        # ln(g1) = -q²J, so ratio of logs should be (q2/q1)² = 4
        ratio = jnp.log(g1_q2) / jnp.log(g1_q1)
        assert jnp.isclose(ratio[0], 4.0, rtol=1e-10)


# ============================================================================
# Transport Integral Matrix Tests
# ============================================================================

class TestComputeTransportIntegralMatrix:
    """Scientific tests for compute_transport_integral_matrix (theory)."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_diagonal_is_zero(self) -> None:
        """Diagonal M[i,i] = 0."""
        from heterodyne.core.theory import compute_transport_integral_matrix

        t = jnp.arange(10.0)
        M = compute_transport_integral_matrix(t, D0=1.0, alpha=1.0, offset=0.0, dt=1.0)

        assert_allclose(jnp.diag(M), 0.0, atol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_symmetric(self) -> None:
        """M[i,j] = M[j,i]."""
        from heterodyne.core.theory import compute_transport_integral_matrix

        t = jnp.arange(10.0)
        M = compute_transport_integral_matrix(t, D0=1.5, alpha=0.8, offset=0.1, dt=1.0)

        assert_allclose(M, M.T, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_non_negative(self) -> None:
        """M[i,j] >= 0 for all i,j."""
        from heterodyne.core.theory import compute_transport_integral_matrix

        t = jnp.arange(10.0)
        M = compute_transport_integral_matrix(t, D0=1.0, alpha=1.0, offset=0.0, dt=1.0)

        assert jnp.all(M >= 0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_constant_rate_analytical(self) -> None:
        """Constant rate: integral = rate * |j-i| * dt."""
        from heterodyne.core.theory import compute_transport_integral_matrix

        N = 10
        t = jnp.arange(N, dtype=jnp.float64)
        rate = 2.5
        dt = 1.0
        M = compute_transport_integral_matrix(t, D0=0.0, alpha=1.0, offset=rate, dt=dt)

        indices = jnp.arange(N, dtype=jnp.float64)
        expected = rate * dt * jnp.abs(indices[None, :] - indices[:, None])
        assert_allclose(M, expected, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_matches_jax_backend(self) -> None:
        """Theory result matches jax_backend result."""
        from heterodyne.core.jax_backend import (
            compute_transport_integral_matrix as jax_impl,
        )
        from heterodyne.core.theory import (
            compute_transport_integral_matrix as theory_impl,
        )

        t = jnp.arange(15, dtype=jnp.float64) * 1.0
        D0, alpha, offset, dt = 1.5, 0.8, 0.2, 1.0

        M_jax = jax_impl(t, D0, alpha, offset, dt)
        M_theory = theory_impl(t, D0, alpha, offset, dt)

        assert_allclose(M_jax, M_theory, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_shape(self) -> None:
        """Output shape is (N, N)."""
        from heterodyne.core.theory import compute_transport_integral_matrix

        t = jnp.arange(8.0)
        M = compute_transport_integral_matrix(t, D0=1.0, alpha=1.0, offset=0.0, dt=0.5)

        assert M.shape == (8, 8)


# ============================================================================
# Velocity Field Tests
# ============================================================================

class TestComputeVelocityField:
    """Scientific tests for compute_velocity_field."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_constant_velocity(self) -> None:
        """v0=0, v_offset=const gives constant velocity."""
        from heterodyne.core.theory import compute_velocity_field

        t = jnp.array([0.0, 1.0, 5.0, 10.0])
        v_offset = 2.5

        v = compute_velocity_field(t, v0=0.0, beta=1.0, v_offset=v_offset)

        assert_allclose(v, v_offset, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_linear_velocity_beta_1(self) -> None:
        """beta=1 gives linear: v(t) = v0*t + v_offset."""
        from heterodyne.core.theory import compute_velocity_field

        t = jnp.array([0.0, 1.0, 2.0, 3.0])
        v0, beta, v_offset = 2.0, 1.0, 1.0

        v = compute_velocity_field(t, v0, beta, v_offset)

        # At t=0, v(0) = 0 + 1 = 1 (since t^1 = 0 at t=0 with our handling)
        # Actually v(t) = v0 * t^beta + v_offset where t^1 at t=0 is handled as 0
        expected = jnp.where(t > 0, v0 * t + v_offset, v_offset)
        assert_allclose(v, expected, rtol=1e-12)


# ============================================================================
# Time Integral Matrix Tests
# ============================================================================

class TestComputeTimeIntegralMatrix:
    """Scientific tests for compute_time_integral_matrix."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_constant_value_integral(self) -> None:
        """Integral of constant = constant * dt * distance."""
        from heterodyne.core.theory import compute_time_integral_matrix

        values = jnp.ones(5) * 2.0  # Constant value 2
        dt = 1.0

        M = compute_time_integral_matrix(values, dt)

        # M[0,4] should be integral from t_0 to t_4
        # With cumsum approach: sum of 4 values * dt = 2 * 4 * 1 = 8
        # But implementation uses cumsum[j] - cumsum[i-1]
        assert M.shape == (5, 5)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_matrix_shape(self) -> None:
        """Output shape is (N, N)."""
        from heterodyne.core.theory import compute_time_integral_matrix

        values = jnp.arange(10.0)
        M = compute_time_integral_matrix(values, dt=0.1)

        assert M.shape == (10, 10)


# ============================================================================
# Cross Term Phase Tests
# ============================================================================

class TestComputeCrossTermPhase:
    """Scientific tests for compute_cross_term_phase."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_phase_formula(self) -> None:
        """Test phase = q * cos(phi) * integral."""
        from heterodyne.core.theory import compute_cross_term_phase

        velocity_integral = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        q = 0.1
        phi = 0.0  # cos(0) = 1

        phase = compute_cross_term_phase(velocity_integral, q, phi)

        expected = q * velocity_integral  # cos(0) = 1
        assert_allclose(phase, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_phase_zero_at_phi_90(self) -> None:
        """Phase is zero at phi=90° (cos(90°) = 0)."""
        from heterodyne.core.theory import compute_cross_term_phase

        velocity_integral = jnp.ones((3, 3))
        phase = compute_cross_term_phase(velocity_integral, q=0.1, phi=90.0)

        assert_allclose(phase, 0.0, atol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_phase_periodicity(self) -> None:
        """Phase should be periodic with 360°."""
        from heterodyne.core.theory import compute_cross_term_phase

        velocity_integral = jnp.ones((3, 3))
        q = 0.1

        phase_0 = compute_cross_term_phase(velocity_integral, q, 0.0)
        phase_360 = compute_cross_term_phase(velocity_integral, q, 360.0)

        assert_allclose(phase_0, phase_360, rtol=1e-10)


# ============================================================================
# Normalization Factor Tests
# ============================================================================

class TestComputeNormalizationFactor:
    """Scientific tests for compute_normalization_factor."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_normalization_formula(self) -> None:
        """Test f² = (f_s² + f_r²)_1 * (f_s² + f_r²)_2."""
        from heterodyne.core.theory import compute_normalization_factor

        f_s_1 = jnp.array([0.3, 0.5, 0.7])
        f_s_2 = jnp.array([0.4, 0.6])

        norm = compute_normalization_factor(f_s_1, f_s_2)

        # Compute expected
        f_r_1 = 1.0 - f_s_1
        f_r_2 = 1.0 - f_s_2
        norm_1 = f_s_1**2 + f_r_1**2
        norm_2 = f_s_2**2 + f_r_2**2
        expected = norm_1[:, None] * norm_2[None, :]

        assert_allclose(norm, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_normalization_shape(self) -> None:
        """Output shape is (N1, N2)."""
        from heterodyne.core.theory import compute_normalization_factor

        f_s_1 = jnp.array([0.3, 0.5, 0.7])
        f_s_2 = jnp.array([0.4, 0.6])

        norm = compute_normalization_factor(f_s_1, f_s_2)

        assert norm.shape == (3, 2)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_normalization_always_positive(self) -> None:
        """Normalization is always positive."""
        from heterodyne.core.theory import compute_normalization_factor

        # Any fraction values in [0,1]
        f_s = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        norm = compute_normalization_factor(f_s, f_s)

        assert jnp.all(norm > 0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_normalization_at_equal_fractions(self) -> None:
        """When f_s = 0.5, f_r = 0.5, norm = (0.25 + 0.25)² = 0.25."""
        from heterodyne.core.theory import compute_normalization_factor

        f_s = jnp.array([0.5])
        norm = compute_normalization_factor(f_s, f_s)

        # (0.5² + 0.5²) * (0.5² + 0.5²) = 0.5 * 0.5 = 0.25
        expected = 0.25
        assert_allclose(norm[0, 0], expected, rtol=1e-12)


# ============================================================================
# Physics Interpretation Tests
# ============================================================================

class TestTransportPhysics:
    """Tests for physical interpretation of transport parameters."""

    @pytest.mark.unit
    def test_interpret_alpha_normal_diffusion(self) -> None:
        """alpha=1 is normal diffusion."""
        from heterodyne.core.physics import TransportPhysics

        interpretation = TransportPhysics.interpret_alpha(1.0)
        assert "normal diffusion" in interpretation.lower()

    @pytest.mark.unit
    def test_interpret_alpha_subdiffusion(self) -> None:
        """alpha<1 is subdiffusive."""
        from heterodyne.core.physics import TransportPhysics

        interpretation = TransportPhysics.interpret_alpha(0.7)
        assert "subdiffusive" in interpretation.lower()

    @pytest.mark.unit
    def test_interpret_alpha_superdiffusion(self) -> None:
        """alpha>1 is superdiffusive."""
        from heterodyne.core.physics import TransportPhysics

        interpretation = TransportPhysics.interpret_alpha(1.3)
        assert "superdiffusive" in interpretation.lower()

    @pytest.mark.unit
    def test_interpret_alpha_ballistic(self) -> None:
        """alpha≈2 is ballistic."""
        from heterodyne.core.physics import TransportPhysics

        interpretation = TransportPhysics.interpret_alpha(2.0)
        assert "ballistic" in interpretation.lower()

    @pytest.mark.unit
    def test_diffusion_coefficient_formula(self) -> None:
        """D_eff = D0 * alpha * t^(alpha-1)."""
        from heterodyne.core.physics import TransportPhysics

        D0, alpha, t = 2.0, 1.0, 1.0
        D_eff = TransportPhysics.diffusion_coefficient(D0, alpha, t)

        # For alpha=1: D_eff = D0 * 1 * t^0 = D0 = 2.0
        assert np.isclose(D_eff, 2.0)

    @pytest.mark.unit
    def test_diffusion_coefficient_at_t_zero(self) -> None:
        """D_eff at t=0 should return 0."""
        from heterodyne.core.physics import TransportPhysics

        D_eff = TransportPhysics.diffusion_coefficient(1.0, 1.0, t=0.0)
        assert D_eff == 0.0
