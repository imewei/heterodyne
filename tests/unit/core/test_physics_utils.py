"""Tests for heterodyne.core.physics_utils — numerically safe primitives.

Covers all public functions:
- safe_exp, safe_power, safe_divide, safe_log, safe_sqrt
- compute_relative_difference, symmetrize
- smooth_abs, trapezoid_cumsum, create_time_integral_matrix
- compute_transport_rate, compute_velocity_rate, safe_sinc
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from heterodyne.core.physics_utils import (
    compute_relative_difference,
    compute_transport_rate,
    compute_velocity_rate,
    create_time_integral_matrix,
    safe_divide,
    safe_exp,
    safe_log,
    safe_power,
    safe_sinc,
    safe_sqrt,
    smooth_abs,
    symmetrize,
    trapezoid_cumsum,
)


# ---------------------------------------------------------------------------
# safe_exp
# ---------------------------------------------------------------------------


class TestSafeExp:
    """Tests for overflow-protected exponential."""

    def test_normal_values(self) -> None:
        x = jnp.array([0.0, 1.0, -1.0, 2.0])
        result = safe_exp(x)
        npt.assert_allclose(np.asarray(result), np.exp([0.0, 1.0, -1.0, 2.0]), rtol=1e-7)

    def test_overflow_protection(self) -> None:
        x = jnp.array([1000.0, -1000.0])
        result = safe_exp(x)
        assert np.all(np.isfinite(np.asarray(result)))
        # Large positive should be clipped to exp(500)
        npt.assert_allclose(float(result[0]), np.exp(500.0), rtol=1e-7)
        # Large negative should be clipped to exp(-500) ~ 0
        assert float(result[1]) > 0.0
        assert float(result[1]) < 1e-200

    def test_custom_limit(self) -> None:
        x = jnp.array([100.0])
        result = safe_exp(x, limit=50.0)
        npt.assert_allclose(float(result[0]), np.exp(50.0), rtol=1e-7)

    def test_scalar_input(self) -> None:
        result = safe_exp(0.0)
        npt.assert_allclose(float(result), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# safe_power
# ---------------------------------------------------------------------------


class TestSafePower:
    """Tests for safe power function."""

    def test_positive_base(self) -> None:
        base = jnp.array([1.0, 2.0, 4.0])
        result = safe_power(base, 0.5)
        npt.assert_allclose(np.asarray(result), [1.0, np.sqrt(2), 2.0], rtol=1e-6)

    def test_zero_base_returns_zero(self) -> None:
        base = jnp.array([0.0, 0.0])
        result = safe_power(base, 2.0)
        npt.assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-15)

    def test_negative_base_returns_zero(self) -> None:
        base = jnp.array([-1.0, -5.0])
        result = safe_power(base, 2.0)
        npt.assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-15)

    def test_fractional_exponent(self) -> None:
        base = jnp.array([8.0])
        result = safe_power(base, 1.0 / 3.0)
        npt.assert_allclose(float(result[0]), 2.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------


class TestSafeDivide:
    """Tests for safe division."""

    def test_normal_division(self) -> None:
        num = jnp.array([6.0, 10.0])
        den = jnp.array([2.0, 5.0])
        result = safe_divide(num, den)
        npt.assert_allclose(np.asarray(result), [3.0, 2.0], rtol=1e-7)

    def test_zero_denominator_returns_fill(self) -> None:
        num = jnp.array([1.0, 2.0])
        den = jnp.array([0.0, 0.0])
        result = safe_divide(num, den, fill=0.0)
        npt.assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-15)

    def test_custom_fill_value(self) -> None:
        result = safe_divide(jnp.array([1.0]), jnp.array([0.0]), fill=-999.0)
        npt.assert_allclose(float(result[0]), -999.0, atol=1e-10)

    def test_near_zero_denominator(self) -> None:
        # Denominator smaller than min_denom should return fill
        result = safe_divide(jnp.array([1.0]), jnp.array([1e-40]), fill=0.0)
        npt.assert_allclose(float(result[0]), 0.0, atol=1e-10)

    def test_no_nan_or_inf(self) -> None:
        num = jnp.array([1.0, 0.0, -1.0])
        den = jnp.array([0.0, 0.0, 0.0])
        result = safe_divide(num, den)
        assert np.all(np.isfinite(np.asarray(result)))


# ---------------------------------------------------------------------------
# safe_log
# ---------------------------------------------------------------------------


class TestSafeLog:
    """Tests for safe logarithm."""

    def test_positive_values(self) -> None:
        x = jnp.array([1.0, np.e, np.e**2])
        result = safe_log(x)
        npt.assert_allclose(np.asarray(result), [0.0, 1.0, 2.0], rtol=1e-6)

    def test_zero_input_no_nan(self) -> None:
        result = safe_log(jnp.array([0.0]))
        assert np.isfinite(float(result[0]))
        # log(1e-30) ~ -69
        assert float(result[0]) < -60

    def test_negative_input_no_nan(self) -> None:
        result = safe_log(jnp.array([-1.0]))
        assert np.isfinite(float(result[0]))

    def test_custom_floor(self) -> None:
        result = safe_log(jnp.array([0.0]), floor=1.0)
        npt.assert_allclose(float(result[0]), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# safe_sqrt
# ---------------------------------------------------------------------------


class TestSafeSqrt:
    """Tests for safe square root."""

    def test_positive_values(self) -> None:
        x = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = safe_sqrt(x)
        npt.assert_allclose(np.asarray(result), [0.0, 1.0, 2.0, 3.0], rtol=1e-7)

    def test_negative_input_returns_zero(self) -> None:
        result = safe_sqrt(jnp.array([-1.0, -100.0]))
        npt.assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-15)

    def test_no_nan(self) -> None:
        result = safe_sqrt(jnp.array([-5.0]))
        assert np.isfinite(float(result[0]))


# ---------------------------------------------------------------------------
# compute_relative_difference
# ---------------------------------------------------------------------------


class TestComputeRelativeDifference:
    """Tests for relative difference computation."""

    def test_identical_arrays(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        result = compute_relative_difference(a, a)
        npt.assert_allclose(np.asarray(result), 0.0, atol=1e-10)

    def test_known_difference(self) -> None:
        a = jnp.array([1.0])
        b = jnp.array([2.0])
        result = compute_relative_difference(a, b)
        # |1 - 2| / max(1, 2) = 1/2
        npt.assert_allclose(float(result[0]), 0.5, rtol=1e-7)

    def test_near_zero_values(self) -> None:
        # Denominator floor prevents division by zero
        a = jnp.array([1e-15])
        b = jnp.array([0.0])
        result = compute_relative_difference(a, b)
        assert np.isfinite(float(result[0]))

    def test_bounded_range(self) -> None:
        # Relative difference should be in [0, 2]
        a = jnp.array([1.0, -1.0])
        b = jnp.array([-1.0, 1.0])
        result = compute_relative_difference(a, b)
        vals = np.asarray(result)
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 2.0 + 1e-10)


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------


class TestSymmetrize:
    """Tests for matrix symmetrization."""

    def test_already_symmetric(self) -> None:
        m = jnp.array([[1.0, 2.0], [2.0, 3.0]])
        result = symmetrize(m)
        npt.assert_allclose(np.asarray(result), np.asarray(m), atol=1e-15)

    def test_asymmetric_input(self) -> None:
        m = jnp.array([[1.0, 4.0], [0.0, 3.0]])
        result = symmetrize(m)
        expected = np.array([[1.0, 2.0], [2.0, 3.0]])
        npt.assert_allclose(np.asarray(result), expected, atol=1e-15)

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        m = jnp.array(rng.standard_normal((5, 5)))
        result = np.asarray(symmetrize(m))
        npt.assert_allclose(result, result.T, atol=1e-15)

    def test_diagonal_preserved(self) -> None:
        m = jnp.array([[5.0, 1.0], [3.0, 7.0]])
        result = symmetrize(m)
        npt.assert_allclose(float(result[0, 0]), 5.0, atol=1e-15)
        npt.assert_allclose(float(result[1, 1]), 7.0, atol=1e-15)


# ---------------------------------------------------------------------------
# smooth_abs
# ---------------------------------------------------------------------------


class TestSmoothAbs:
    """Tests for gradient-safe absolute value."""

    def test_positive_values(self) -> None:
        x = jnp.array([1.0, 2.0, 10.0])
        result = smooth_abs(x)
        npt.assert_allclose(np.asarray(result), [1.0, 2.0, 10.0], rtol=1e-6)

    def test_negative_values(self) -> None:
        x = jnp.array([-1.0, -3.0])
        result = smooth_abs(x)
        npt.assert_allclose(np.asarray(result), [1.0, 3.0], rtol=1e-6)

    def test_at_zero_small_bias(self) -> None:
        result = smooth_abs(jnp.array([0.0]))
        # sqrt(0 + 1e-12) = 1e-6
        npt.assert_allclose(float(result[0]), 1e-6, rtol=1e-3)

    def test_no_nan_gradient(self) -> None:
        # The whole point: gradient at x=0 should be finite
        import jax
        grad_fn = jax.grad(lambda x: smooth_abs(x.reshape(1))[0])
        g = grad_fn(jnp.array(0.0))
        assert np.isfinite(float(g))

    def test_symmetry(self) -> None:
        x = jnp.array([3.14])
        npt.assert_allclose(
            float(smooth_abs(x)[0]),
            float(smooth_abs(-x)[0]),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# trapezoid_cumsum
# ---------------------------------------------------------------------------


class TestTrapezoidCumsum:
    """Tests for trapezoidal cumulative integration."""

    def test_constant_function(self) -> None:
        # Integral of f=c from 0 to t is c*t
        f = jnp.ones(11)  # 11 points, dt=0.1 => t from 0 to 1.0
        dt = 0.1
        result = trapezoid_cumsum(f, dt)
        # cumsum[k] should be k * dt * 1.0
        expected = np.arange(11) * dt
        npt.assert_allclose(np.asarray(result), expected, atol=1e-10)

    def test_linear_function(self) -> None:
        # Integral of f=t from 0 to T is T^2/2 (O(dt^2) accurate)
        t = jnp.linspace(0.0, 1.0, 101)
        dt = 0.01
        f = t  # f(t) = t
        result = trapezoid_cumsum(f, dt)
        # At the last point: integral from 0 to 1 of t dt = 0.5
        npt.assert_allclose(float(result[-1]), 0.5, rtol=1e-4)

    def test_first_element_is_zero(self) -> None:
        f = jnp.array([5.0, 10.0, 15.0])
        result = trapezoid_cumsum(f, 1.0)
        assert float(result[0]) == 0.0

    def test_output_length(self) -> None:
        f = jnp.ones(20)
        result = trapezoid_cumsum(f, 0.5)
        assert result.shape == (20,)

    def test_quadratic_function_accuracy(self) -> None:
        # Integral of f=t^2 from 0 to 1 is 1/3
        # Trapezoidal rule has O(dt^2) error
        t = jnp.linspace(0.0, 1.0, 1001)
        dt = 0.001
        f = t**2
        result = trapezoid_cumsum(f, dt)
        npt.assert_allclose(float(result[-1]), 1.0 / 3.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# create_time_integral_matrix
# ---------------------------------------------------------------------------


class TestCreateTimeIntegralMatrix:
    """Tests for N x N integral matrix from cumulative sums."""

    def test_shape(self) -> None:
        cumsum = jnp.array([0.0, 1.0, 3.0, 6.0])
        result = create_time_integral_matrix(cumsum)
        assert result.shape == (4, 4)

    def test_diagonal_is_zero(self) -> None:
        cumsum = jnp.array([0.0, 1.0, 3.0, 6.0])
        result = create_time_integral_matrix(cumsum)
        npt.assert_allclose(np.diag(np.asarray(result)), 0.0, atol=1e-15)

    def test_antisymmetric(self) -> None:
        # M[i,j] = cumsum[j] - cumsum[i] => M[i,j] = -M[j,i]
        cumsum = jnp.array([0.0, 2.0, 5.0, 11.0])
        result = np.asarray(create_time_integral_matrix(cumsum))
        npt.assert_allclose(result, -result.T, atol=1e-12)

    def test_known_values(self) -> None:
        cumsum = jnp.array([0.0, 1.0, 3.0])
        result = np.asarray(create_time_integral_matrix(cumsum))
        # M[0,1] = 1 - 0 = 1, M[0,2] = 3 - 0 = 3, M[1,2] = 3 - 1 = 2
        npt.assert_allclose(result[0, 1], 1.0, atol=1e-12)
        npt.assert_allclose(result[0, 2], 3.0, atol=1e-12)
        npt.assert_allclose(result[1, 2], 2.0, atol=1e-12)

    def test_abs_gives_symmetric(self) -> None:
        # When combined with smooth_abs, result should be approximately symmetric
        cumsum = jnp.array([0.0, 1.0, 4.0, 9.0])
        result = np.asarray(smooth_abs(create_time_integral_matrix(cumsum)))
        npt.assert_allclose(result, result.T, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_transport_rate
# ---------------------------------------------------------------------------


class TestComputeTransportRate:
    """Tests for transport rate J(t) = D0 * t^alpha + offset."""

    def test_basic_computation(self) -> None:
        t = jnp.array([0.0, 1.0, 2.0])
        result = compute_transport_rate(t, D0=1.0, alpha=1.0, offset=0.0)
        npt.assert_allclose(np.asarray(result), [0.0, 1.0, 2.0], rtol=1e-6)

    def test_with_offset(self) -> None:
        t = jnp.array([1.0, 2.0])
        result = compute_transport_rate(t, D0=1.0, alpha=0.0, offset=5.0)
        # D0 * t^0 + offset = 1 + 5 = 6
        npt.assert_allclose(np.asarray(result), [6.0, 6.0], rtol=1e-6)

    def test_floored_at_zero(self) -> None:
        # Even if D0*t^alpha + offset < 0, result should be >= 0
        t = jnp.array([1.0])
        result = compute_transport_rate(t, D0=1.0, alpha=1.0, offset=-100.0)
        assert float(result[0]) >= 0.0

    def test_zero_time(self) -> None:
        t = jnp.array([0.0])
        result = compute_transport_rate(t, D0=1e4, alpha=0.5, offset=0.0)
        # At t=0, t^alpha -> 0 (for alpha > 0)
        npt.assert_allclose(float(result[0]), 0.0, atol=1e-5)

    def test_subdiffusive_exponent(self) -> None:
        # alpha < 1: subdiffusive behavior
        t = jnp.array([1.0, 4.0])
        result = compute_transport_rate(t, D0=1.0, alpha=0.5, offset=0.0)
        npt.assert_allclose(np.asarray(result), [1.0, 2.0], rtol=1e-6)

    def test_no_nan_or_inf(self) -> None:
        t = jnp.linspace(0.0, 10.0, 50)
        result = compute_transport_rate(t, D0=1e4, alpha=0.7, offset=10.0)
        assert np.all(np.isfinite(np.asarray(result)))


# ---------------------------------------------------------------------------
# compute_velocity_rate
# ---------------------------------------------------------------------------


class TestComputeVelocityRate:
    """Tests for velocity rate v(t) = v0 * t^beta + v_offset."""

    def test_basic_computation(self) -> None:
        t = jnp.array([0.0, 1.0, 2.0])
        result = compute_velocity_rate(t, v0=3.0, beta=1.0, v_offset=0.0)
        npt.assert_allclose(np.asarray(result), [0.0, 3.0, 6.0], rtol=1e-6)

    def test_with_offset(self) -> None:
        t = jnp.array([1.0])
        result = compute_velocity_rate(t, v0=0.0, beta=1.0, v_offset=5.0)
        npt.assert_allclose(float(result[0]), 5.0, rtol=1e-7)

    def test_allows_negative_unlike_transport(self) -> None:
        # Velocity rate is NOT floored at 0 (unlike transport rate)
        t = jnp.array([1.0])
        result = compute_velocity_rate(t, v0=1.0, beta=1.0, v_offset=-100.0)
        assert float(result[0]) < 0.0

    def test_zero_time(self) -> None:
        t = jnp.array([0.0])
        result = compute_velocity_rate(t, v0=1e3, beta=0.5, v_offset=10.0)
        # At t=0, t^beta -> 0 for beta > 0, so result = v_offset
        npt.assert_allclose(float(result[0]), 10.0, rtol=1e-5)

    def test_no_nan(self) -> None:
        t = jnp.linspace(0.0, 10.0, 50)
        result = compute_velocity_rate(t, v0=1e3, beta=0.8, v_offset=-50.0)
        assert np.all(np.isfinite(np.asarray(result)))


# ---------------------------------------------------------------------------
# safe_sinc
# ---------------------------------------------------------------------------


class TestSafeSinc:
    """Tests for unnormalized sinc: sin(x)/x."""

    def test_at_zero(self) -> None:
        result = safe_sinc(jnp.array([0.0]))
        npt.assert_allclose(float(result[0]), 1.0, atol=1e-8)

    def test_known_values(self) -> None:
        x = jnp.array([np.pi])
        result = safe_sinc(x)
        # sin(pi)/pi ~ 0
        npt.assert_allclose(float(result[0]), 0.0, atol=1e-6)

    def test_small_x(self) -> None:
        # Near x=0, sinc(x) ~ 1 - x^2/6
        x = jnp.array([1e-8])
        result = safe_sinc(x)
        npt.assert_allclose(float(result[0]), 1.0, atol=1e-5)

    def test_array_input(self) -> None:
        x = jnp.array([0.0, np.pi / 2, np.pi])
        result = safe_sinc(x)
        expected = np.array([1.0, np.sin(np.pi / 2) / (np.pi / 2), 0.0])
        npt.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_no_nan(self) -> None:
        x = jnp.linspace(-10.0, 10.0, 201)
        result = safe_sinc(x)
        assert np.all(np.isfinite(np.asarray(result)))

    def test_symmetry(self) -> None:
        x = jnp.array([1.5])
        npt.assert_allclose(
            float(safe_sinc(x)[0]),
            float(safe_sinc(-x)[0]),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Integration: trapezoid_cumsum + create_time_integral_matrix
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    """End-to-end tests combining cumsum and matrix creation."""

    def test_constant_rate_integral(self) -> None:
        """For constant f=c, integral from t_i to t_j is c*(t_j - t_i)."""
        n = 21
        dt = 0.05
        f = jnp.ones(n) * 2.0  # constant rate = 2
        cumsum = trapezoid_cumsum(f, dt)
        matrix = create_time_integral_matrix(cumsum)
        # M[0, k] = c * k * dt = 2 * k * 0.05
        for k in range(n):
            npt.assert_allclose(float(matrix[0, k]), 2.0 * k * dt, atol=1e-10)

    def test_smooth_abs_on_matrix(self) -> None:
        """smooth_abs of integral matrix should be non-negative everywhere."""
        f = jnp.linspace(1.0, 5.0, 30)
        cumsum = trapezoid_cumsum(f, 0.1)
        matrix = create_time_integral_matrix(cumsum)
        abs_matrix = smooth_abs(matrix)
        assert np.all(np.asarray(abs_matrix) >= 0.0)
