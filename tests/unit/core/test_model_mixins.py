"""Tests for heterodyne.core.model_mixins — reusable model behavior mixins.

Tests each mixin in isolation using lightweight stub classes that
satisfy the required interface (t, q, dt attributes).

Covers:
- TransportMixin: compute_transport_rate, compute_transport_integral, compute_half_transport
- FractionMixin: compute_fraction_evolution, compute_fraction_matrices
- VelocityMixin: compute_velocity_field, compute_velocity_integral, compute_phase_factor
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from heterodyne.core.model_mixins import (
    FractionMixin,
    TransportMixin,
    VelocityMixin,
)

# ---------------------------------------------------------------------------
# Stub classes to host the mixins
# ---------------------------------------------------------------------------


class TransportStub(TransportMixin):
    """Minimal class satisfying TransportMixin interface."""

    def __init__(self, t: jnp.ndarray, q: float, dt: float) -> None:
        self.t = t
        self.q = q
        self.dt = dt


class FractionStub(FractionMixin):
    """Minimal class satisfying FractionMixin interface."""

    def __init__(self, t: jnp.ndarray) -> None:
        self.t = t


class VelocityStub(VelocityMixin):
    """Minimal class satisfying VelocityMixin interface."""

    def __init__(self, t: jnp.ndarray, q: float, dt: float) -> None:
        self.t = t
        self.q = q
        self.dt = dt


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_grid() -> jnp.ndarray:
    return jnp.linspace(0.0, 1.0, 21)


@pytest.fixture()
def dt() -> float:
    return 0.05


@pytest.fixture()
def q() -> float:
    return 0.01  # typical q in Angstrom^{-1}


# ---------------------------------------------------------------------------
# TransportMixin
# ---------------------------------------------------------------------------


class TestTransportMixin:
    """Tests for transport computation mixin."""

    def test_compute_transport_rate_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        rate = stub.compute_transport_rate(D0=1e4, alpha=0.5, offset=0.0)
        assert rate.shape == time_grid.shape

    def test_compute_transport_rate_values(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        rate = stub.compute_transport_rate(D0=1.0, alpha=1.0, offset=0.0)
        # At t=1, rate should be D0 * 1^1 + 0 = 1
        npt.assert_allclose(float(rate[-1]), 1.0, rtol=1e-6)

    def test_compute_transport_integral_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        integral = stub.compute_transport_integral(D0=1e4, alpha=0.5, offset=0.0)
        n = len(time_grid)
        assert integral.shape == (n, n)

    def test_transport_integral_non_negative(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        integral = stub.compute_transport_integral(D0=1e4, alpha=0.5, offset=0.0)
        # Uses smooth_abs, so should be >= 0
        assert np.all(np.asarray(integral) >= 0.0)

    def test_transport_integral_diagonal_near_zero(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        integral = stub.compute_transport_integral(D0=1e4, alpha=0.5, offset=0.0)
        diag = np.diag(np.asarray(integral))
        # Diagonal should be close to zero (integral from t_i to t_i)
        npt.assert_allclose(diag, 0.0, atol=1e-4)

    def test_transport_integral_symmetric(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        integral = np.asarray(
            stub.compute_transport_integral(D0=1e4, alpha=0.5, offset=0.0)
        )
        # smooth_abs makes it approximately symmetric
        npt.assert_allclose(integral, integral.T, atol=1e-4)

    def test_compute_half_transport_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        half_tr = stub.compute_half_transport(D0=1e4, alpha=0.5, offset=0.0)
        n = len(time_grid)
        assert half_tr.shape == (n, n)

    def test_half_transport_bounded(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        half_tr = np.asarray(stub.compute_half_transport(D0=1e4, alpha=0.5, offset=0.0))
        # exp(-0.5 * q^2 * integral) should be in (0, 1]
        assert np.all(half_tr > 0.0)
        assert np.all(half_tr <= 1.0 + 1e-10)

    def test_half_transport_diagonal_is_one(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        half_tr = np.asarray(stub.compute_half_transport(D0=1e4, alpha=0.5, offset=0.0))
        # Diagonal: integral = 0 => exp(0) = 1
        npt.assert_allclose(np.diag(half_tr), 1.0, atol=1e-4)

    def test_larger_D0_faster_decay(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = TransportStub(time_grid, q, dt)
        half_tr_small = np.asarray(
            stub.compute_half_transport(D0=1e2, alpha=1.0, offset=0.0)
        )
        half_tr_large = np.asarray(
            stub.compute_half_transport(D0=1e5, alpha=1.0, offset=0.0)
        )
        # Larger D0 => faster decay => smaller off-diagonal values
        # Compare a well-separated off-diagonal element
        assert float(half_tr_large[0, -1]) <= float(half_tr_small[0, -1])


# ---------------------------------------------------------------------------
# FractionMixin
# ---------------------------------------------------------------------------


class TestFractionMixin:
    """Tests for fraction computation mixin."""

    def test_fraction_evolution_shape(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        f = stub.compute_fraction_evolution(f0=0.5, f1=0.0, f2=0.0, f3=0.3)
        assert f.shape == time_grid.shape

    def test_fraction_clipped_to_unit_interval(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        # f0 * exp(f1*(t - f2)) + f3 with large values
        f = stub.compute_fraction_evolution(f0=10.0, f1=0.0, f2=0.0, f3=0.0)
        vals = np.asarray(f)
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_constant_fraction(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        # f0 * exp(0) + f3 = f0 + f3 = 0.5 + 0.2 = 0.7
        f = stub.compute_fraction_evolution(f0=0.5, f1=0.0, f2=0.0, f3=0.2)
        npt.assert_allclose(np.asarray(f), 0.7, rtol=1e-6)

    def test_fraction_matrices_shapes(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        f_sample = jnp.ones(len(time_grid)) * 0.3
        f_ref_mat, f_sample_mat, f_cross_mat, norm = stub.compute_fraction_matrices(
            f_sample
        )
        n = len(time_grid)
        assert f_ref_mat.shape == (n, n)
        assert f_sample_mat.shape == (n, n)
        assert f_cross_mat.shape == (n, n)
        assert norm.shape == (n, n)

    def test_fraction_matrices_non_negative(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        f_sample = jnp.ones(len(time_grid)) * 0.4
        f_ref_mat, f_sample_mat, f_cross_mat, norm = stub.compute_fraction_matrices(
            f_sample
        )
        for mat in [f_ref_mat, f_sample_mat, f_cross_mat, norm]:
            assert np.all(np.asarray(mat) >= 0.0)

    def test_fraction_matrices_symmetric(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        # Use a time-varying fraction
        f_sample = jnp.linspace(0.2, 0.8, len(time_grid))
        f_ref_mat, f_sample_mat, f_cross_mat, norm = stub.compute_fraction_matrices(
            f_sample
        )
        for mat in [f_ref_mat, f_sample_mat, f_cross_mat, norm]:
            arr = np.asarray(mat)
            npt.assert_allclose(arr, arr.T, atol=1e-12)

    def test_fraction_ref_plus_sample_consistent(self, time_grid: jnp.ndarray) -> None:
        stub = FractionStub(time_grid)
        f_s = 0.3
        f_sample = jnp.ones(len(time_grid)) * f_s
        f_ref_mat, f_sample_mat, f_cross_mat, _ = stub.compute_fraction_matrices(
            f_sample
        )
        # f_ref_mat should be (1-f_s)^2 = 0.49
        npt.assert_allclose(np.asarray(f_ref_mat), (1 - f_s) ** 2, rtol=1e-6)
        # f_sample_mat should be f_s^2 = 0.09
        npt.assert_allclose(np.asarray(f_sample_mat), f_s**2, rtol=1e-6)


# ---------------------------------------------------------------------------
# VelocityMixin
# ---------------------------------------------------------------------------


class TestVelocityMixin:
    """Tests for velocity computation mixin."""

    def test_velocity_field_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        v = stub.compute_velocity_field(v0=1e3, beta=1.0, v_offset=0.0)
        assert v.shape == time_grid.shape

    def test_velocity_field_values(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        # v(t) = v0 * t^1 + 0 = v0 * t
        v = stub.compute_velocity_field(v0=100.0, beta=1.0, v_offset=0.0)
        npt.assert_allclose(float(v[-1]), 100.0, rtol=1e-6)  # t[-1] = 1.0

    def test_velocity_integral_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        integral = stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        n = len(time_grid)
        assert integral.shape == (n, n)

    def test_velocity_integral_diagonal_zero(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        integral = np.asarray(
            stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        )
        npt.assert_allclose(np.diag(integral), 0.0, atol=1e-10)

    def test_velocity_integral_antisymmetric(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        integral = np.asarray(
            stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        )
        # Signed integral: M[i,j] = -M[j,i]
        npt.assert_allclose(integral, -integral.T, atol=1e-8)

    def test_phase_factor_shape(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        v_integral = stub.compute_velocity_integral(v0=100.0, beta=1.0, v_offset=0.0)
        phase = stub.compute_phase_factor(v_integral, phi=45.0)
        n = len(time_grid)
        assert phase.shape == (n, n)

    def test_phase_factor_bounded(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        v_integral = stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        phase = np.asarray(stub.compute_phase_factor(v_integral, phi=30.0))
        # cos(...) is bounded in [-1, 1]
        assert np.all(phase >= -1.0 - 1e-10)
        assert np.all(phase <= 1.0 + 1e-10)

    def test_phase_factor_diagonal_is_one(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        v_integral = stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        phase = np.asarray(stub.compute_phase_factor(v_integral, phi=45.0))
        # Diagonal: integral = 0 => cos(0) = 1
        npt.assert_allclose(np.diag(phase), 1.0, atol=1e-10)

    def test_phi_zero_vs_ninety(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        stub = VelocityStub(time_grid, q, dt)
        v_integral = stub.compute_velocity_integral(v0=1e3, beta=1.0, v_offset=0.0)
        phase_0 = np.asarray(stub.compute_phase_factor(v_integral, phi=0.0))
        phase_90 = np.asarray(stub.compute_phase_factor(v_integral, phi=90.0))
        # At phi=90 deg, cos(phi)=0, so phase factor should be cos(0)=1 everywhere
        npt.assert_allclose(phase_90, 1.0, atol=1e-6)
        # At phi=0 deg, cos(phi)=1, so phase factor varies
        # Off-diagonal should differ from 1 (unless integral is very small)
        # Just verify they are different matrices
        assert not np.allclose(phase_0, phase_90, atol=1e-8)

    def test_constant_velocity_integral(
        self, time_grid: jnp.ndarray, q: float, dt: float
    ) -> None:
        """For constant velocity, integral from t_i to t_j should be v*(t_j - t_i)."""
        stub = VelocityStub(time_grid, q, dt)
        v_const = 50.0
        integral = np.asarray(
            stub.compute_velocity_integral(v0=0.0, beta=1.0, v_offset=v_const)
        )
        # M[0, k] should be v_const * t[k]
        for k in range(len(time_grid)):
            expected = v_const * float(time_grid[k])
            npt.assert_allclose(integral[0, k], expected, atol=1e-6)
