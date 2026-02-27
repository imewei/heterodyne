"""Property-based tests using Hypothesis.

This module provides property-based testing for the heterodyne package,
verifying mathematical properties and physical constraints hold across
a wide range of inputs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

# ============================================================================
# Strategy Definitions
# ============================================================================

# Physical parameter ranges
positive_float = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
bounded_float = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
fraction_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
angle_float = st.floats(min_value=-360.0, max_value=360.0, allow_nan=False, allow_infinity=False)
alpha_float = st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)

# Time arrays (positive, monotonic)
time_array_strategy = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=50),
    elements=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
).map(lambda x: np.sort(np.unique(np.abs(x))))

# Small time array for faster tests
small_time_array = st.builds(
    lambda n: jnp.arange(n, dtype=jnp.float64),
    n=st.integers(min_value=5, max_value=20)
)


# ============================================================================
# Transport Coefficient Properties
# ============================================================================

class TestTransportCoefficientProperties:
    """Property-based tests for transport coefficient computations."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(D0=positive_float, alpha=alpha_float, t=positive_float)
    @settings(max_examples=100, deadline=None)
    def test_transport_monotonic_in_time(self, D0: float, alpha: float, t: float) -> None:
        """J(t) is monotonically increasing in t for positive D0, alpha."""
        from heterodyne.core.theory import compute_transport_coefficient

        assume(t > 0.1)  # Avoid very small t

        t_arr = jnp.array([t, t + 1.0])
        J = compute_transport_coefficient(t_arr, D0, alpha, offset=0.0)

        # J should increase with t
        assert J[1] >= J[0], f"J not monotonic: J({t})={J[0]}, J({t+1})={J[1]}"

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(D0=positive_float, alpha=alpha_float, offset=bounded_float)
    @settings(max_examples=100, deadline=None)
    def test_transport_at_t_zero(self, D0: float, alpha: float, offset: float) -> None:
        """J(0) = offset (no singularity)."""
        from heterodyne.core.theory import compute_transport_coefficient

        J = compute_transport_coefficient(jnp.array([0.0]), D0, alpha, offset)

        assert jnp.isclose(J[0], offset, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(D0=positive_float, t=positive_float)
    @settings(max_examples=50, deadline=None)
    def test_transport_linearity_alpha_one(self, D0: float, t: float) -> None:
        """For alpha=1, J(t) = D0*t (linear)."""
        from heterodyne.core.theory import compute_transport_coefficient

        assume(t > 0)

        J = compute_transport_coefficient(jnp.array([t]), D0, 1.0, 0.0)
        expected = D0 * t

        assert_allclose(float(J[0]), expected, rtol=1e-10)


# ============================================================================
# g1 Correlation Properties
# ============================================================================

class TestG1CorrelationProperties:
    """Property-based tests for g1 correlation."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(J=positive_float, q=positive_float)
    @settings(max_examples=100, deadline=None)
    def test_g1_bounded_zero_one(self, J: float, q: float) -> None:
        """g1 is always in [0, 1] for J >= 0."""
        from heterodyne.core.jax_backend import compute_g1_transport

        assume(q < 10)  # Avoid extreme q values

        g1 = compute_g1_transport(jnp.array([J]), q)

        assert 0.0 <= float(g1[0]) <= 1.0

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(
        J1=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        q=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_g1_monotonically_decreasing(self, J1: float, delta: float, q: float) -> None:
        """g1 decreases as J increases (for fixed q)."""
        from heterodyne.core.jax_backend import compute_g1_transport

        J2 = J1 + delta  # Guarantees J1 < J2 without filtering

        g1 = compute_g1_transport(jnp.array([J1, J2]), q)

        assert g1[0] >= g1[1], "g1 should decrease with increasing J"

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(
        J=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        q1=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        delta_q=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_g1_decreases_faster_with_larger_q(self, J: float, q1: float, delta_q: float) -> None:
        """Larger q causes faster decay (smaller g1)."""
        from heterodyne.core.jax_backend import compute_g1_transport

        q2 = q1 + delta_q  # Guarantees q1 < q2 without filtering

        g1_q1 = compute_g1_transport(jnp.array([J]), q1)
        g1_q2 = compute_g1_transport(jnp.array([J]), q2)

        assert g1_q1[0] >= g1_q2[0], "Larger q should give faster decay"


# ============================================================================
# Fraction Properties
# ============================================================================

class TestFractionProperties:
    """Property-based tests for fraction computation."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(
        f0=fraction_float,
        f1=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        f2=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        f3=fraction_float,
        t=st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_fraction_always_clipped(self, f0: float, f1: float, f2: float, f3: float, t: float) -> None:
        """Fraction is always in [0, 1] for bounded inputs."""
        from heterodyne.core.jax_backend import compute_fraction_jit

        # Avoid cases where exp(-f1*t) overflows/underflows
        assume(abs(f1 * t) < 50)

        frac = compute_fraction_jit(jnp.array([t]), f0, f1, f2, f3)

        # Check result is finite and bounded
        assert jnp.isfinite(frac[0]), f"Fraction not finite: {frac[0]}"
        assert 0.0 <= float(frac[0]) <= 1.0

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(f0=fraction_float, f3=fraction_float, t=positive_float)
    @settings(max_examples=50, deadline=None)
    def test_fraction_constant_when_f1_zero(self, f0: float, f3: float, t: float) -> None:
        """When f1=0, fraction is constant f0 + f3 (clipped)."""
        from heterodyne.core.jax_backend import compute_fraction_jit

        frac = compute_fraction_jit(jnp.array([t]), f0, 0.0, 0.0, f3)
        expected = np.clip(f0 + f3, 0.0, 1.0)

        # Use atol for near-zero values (subnormals get flushed to zero in JAX)
        assert_allclose(float(frac[0]), expected, rtol=1e-10, atol=1e-300)


# ============================================================================
# C2 Correlation Properties
# ============================================================================

class TestC2CorrelationProperties:
    """Property-based tests for full C2 correlation."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(phi=angle_float)
    @settings(max_examples=50, deadline=None)
    def test_c2_always_positive(self, phi: float) -> None:
        """C2 correlation is always positive."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne

        t = jnp.arange(10, dtype=jnp.float64)
        params = jnp.array([
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
        ], dtype=jnp.float64)

        c2 = compute_c2_heterodyne(params, t, 0.01, 1.0, phi)

        assert jnp.all(c2 >= 0), "C2 should be non-negative"

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(phi=angle_float)
    @settings(max_examples=50, deadline=None)
    def test_c2_phi_periodicity(self, phi: float) -> None:
        """C2 is periodic in phi with period 360°."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne

        t = jnp.arange(10, dtype=jnp.float64)
        params = jnp.array([
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.1, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0  # Non-zero velocity
        ], dtype=jnp.float64)

        c2_phi = compute_c2_heterodyne(params, t, 0.01, 1.0, phi)
        c2_phi_360 = compute_c2_heterodyne(params, t, 0.01, 1.0, phi + 360.0)

        assert_allclose(c2_phi, c2_phi_360, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(phi=angle_float)
    @settings(max_examples=50, deadline=None)
    def test_c2_no_nan_inf(self, phi: float) -> None:
        """C2 should never contain NaN or Inf."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne

        t = jnp.arange(10, dtype=jnp.float64)
        params = jnp.array([
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
        ], dtype=jnp.float64)

        c2 = compute_c2_heterodyne(params, t, 0.01, 1.0, phi)

        assert not jnp.any(jnp.isnan(c2)), "C2 contains NaN"
        assert not jnp.any(jnp.isinf(c2)), "C2 contains Inf"


# ============================================================================
# Residual Properties
# ============================================================================

class TestResidualProperties:
    """Property-based tests for residual computations."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_residual_scales_with_weights(self, scale: float) -> None:
        """Weighted residuals scale as sqrt(weights)."""
        from heterodyne.core.jax_backend import compute_residuals

        t = jnp.arange(5, dtype=jnp.float64)
        params = jnp.array([
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
        ], dtype=jnp.float64)

        c2_data = jnp.ones((5, 5))
        weights = jnp.full((5, 5), scale)

        res_no_weight = compute_residuals(params, t, 0.01, 1.0, 0.0, c2_data, None)
        res_weighted = compute_residuals(params, t, 0.01, 1.0, 0.0, c2_data, weights)

        # Weighted = unweighted * sqrt(scale)
        expected_ratio = np.sqrt(scale)
        actual_ratio = float(jnp.mean(jnp.abs(res_weighted)) / jnp.mean(jnp.abs(res_no_weight)))

        assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)


# ============================================================================
# Normalization Factor Properties
# ============================================================================

class TestNormalizationProperties:
    """Property-based tests for normalization computations."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(f_s=fraction_float)
    @settings(max_examples=100, deadline=None)
    def test_normalization_always_positive(self, f_s: float) -> None:
        """Normalization factor is always positive."""
        from heterodyne.core.theory import compute_normalization_factor

        f_s_arr = jnp.array([f_s])
        norm = compute_normalization_factor(f_s_arr, f_s_arr)

        assert float(norm[0, 0]) > 0

    @pytest.mark.unit
    @pytest.mark.requires_jax
    @given(f_s=fraction_float)
    @settings(max_examples=100, deadline=None)
    def test_normalization_symmetric_in_fractions(self, f_s: float) -> None:
        """Normalization is symmetric: norm(a,b) = norm(b,a)."""
        from heterodyne.core.theory import compute_normalization_factor

        f_s_1 = jnp.array([f_s, 0.5])
        f_s_2 = jnp.array([0.5, f_s])

        norm_12 = compute_normalization_factor(f_s_1, f_s_2)
        norm_21 = compute_normalization_factor(f_s_2, f_s_1)

        # Should be transpose
        assert_allclose(norm_12, norm_21.T, rtol=1e-10)


# ============================================================================
# MCMC Diagnostic Properties
# ============================================================================

class TestMCMCDiagnosticProperties:
    """Property-based tests for MCMC diagnostics."""

    @pytest.mark.unit
    @given(shift=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_rhat_location_invariant(self, shift: float) -> None:
        """R-hat is invariant to location shifts."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((4, 100))

        rhat_original = compute_r_hat(samples)
        rhat_shifted = compute_r_hat(samples + shift)

        # Rank-normalized R-hat: O(1e-4) noise from rank discretization
        assert_allclose(rhat_original, rhat_shifted, rtol=1e-3)

    @pytest.mark.unit
    @given(scale=st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_rhat_scale_invariant(self, scale: float) -> None:
        """R-hat is approximately invariant to scale."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((4, 100))

        rhat_original = compute_r_hat(samples)
        rhat_scaled = compute_r_hat(samples * scale)

        # Rank-normalized R-hat: O(1e-4) noise from rank discretization
        assert_allclose(rhat_original, rhat_scaled, rtol=1e-3)

    @pytest.mark.unit
    @given(shift=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_ess_location_invariant(self, shift: float) -> None:
        """ESS is invariant to location shifts."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        rng = np.random.default_rng(42)
        samples = rng.standard_normal(100)

        ess_original = compute_ess(samples)
        ess_shifted = compute_ess(samples + shift)

        assert_allclose(ess_original, ess_shifted, rtol=0.01)
