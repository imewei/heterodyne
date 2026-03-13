"""Regression tests for numerical stability under edge conditions.

Verifies that core primitives and the correlation kernel remain finite
and well-behaved for extreme (but valid) parameter values, near-zero
inputs, and boundary conditions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.core.fitting import solve_least_squares_jax
from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.core.physics_utils import (
    create_time_integral_matrix,
    smooth_abs,
    trapezoid_cumsum,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N_TIMES = 32
Q = 0.01  # Angstrom^{-1}
PHI_ANGLE = 45.0  # degrees


def _default_params() -> jnp.ndarray:
    """Build the 14-element default parameter vector."""
    return jnp.array(
        [DEFAULT_REGISTRY[name].default for name in ALL_PARAM_NAMES],
        dtype=jnp.float64,
    )


def _time_grid() -> tuple[jnp.ndarray, float]:
    t = jnp.linspace(1e-6, 0.1, N_TIMES)
    dt = float(t[1] - t[0])
    return t, dt


# ---------------------------------------------------------------------------
# smooth_abs
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestSmoothAbs:
    """smooth_abs is finite and gradient-safe near x=0."""

    def test_finite_near_zero(self) -> None:
        """smooth_abs produces finite output for x values near zero."""
        x = jnp.array([0.0, 1e-15, -1e-15, 1e-30, -1e-30])
        result = smooth_abs(x)
        assert jnp.all(jnp.isfinite(result)), (
            "smooth_abs produced non-finite output near zero"
        )
        # All outputs should be >= 0
        assert jnp.all(result >= 0.0)

    def test_gradient_continuous_near_zero(self) -> None:
        """smooth_abs gradient is finite and well-defined near x=0.

        The gradient of smooth_abs = sqrt(x^2 + eps) is x/sqrt(x^2 + eps),
        which transitions smoothly from -1 to +1 through zero.  With eps=1e-12,
        the transition region is ~1e-6 wide, so the gradient changes rapidly
        but continuously.  We verify finiteness and that the gradient at x=0
        is near zero (the smooth midpoint).
        """
        grad_fn = jax.grad(lambda x: smooth_abs(x))
        # Evaluate gradient at individual points near zero
        test_points = jnp.array([0.0, 1e-15, -1e-15, 1e-8, -1e-8, 1e-4, -1e-4])
        grads = jax.vmap(grad_fn)(test_points)

        assert jnp.all(jnp.isfinite(grads)), (
            "smooth_abs gradient contains non-finite values"
        )

        # Gradient at x=0 should be near zero (midpoint of the smooth transition)
        grad_at_zero = float(grads[0])
        assert abs(grad_at_zero) < 1e-3, (
            f"smooth_abs gradient at x=0 is {grad_at_zero:.3e}, expected ~0"
        )

        # Gradient magnitude should be <= 1.0 everywhere (bounded by |x|/|x| = 1)
        assert jnp.all(jnp.abs(grads) <= 1.0 + 1e-10), "smooth_abs gradient exceeds 1.0"

    def test_matches_abs_for_large_values(self) -> None:
        """smooth_abs matches jnp.abs for values far from zero."""
        x = jnp.array([-100.0, -1.0, 1.0, 100.0])
        result = smooth_abs(x)
        expected = jnp.abs(x)
        assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-8)


# ---------------------------------------------------------------------------
# trapezoid_cumsum
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestTrapezoidCumsum:
    """trapezoid_cumsum is monotonically non-decreasing for positive input."""

    def test_monotonic_for_positive_input(self) -> None:
        """Cumulative integral of positive function is non-decreasing."""
        rng = np.random.default_rng(42)
        f = jnp.abs(jnp.array(rng.standard_normal(N_TIMES))) + 0.1
        dt = 0.01
        cumsum = trapezoid_cumsum(f, dt)
        diffs = jnp.diff(cumsum)
        assert jnp.all(diffs >= -1e-15), (
            f"trapezoid_cumsum is not monotonic: min diff = {float(jnp.min(diffs)):.3e}"
        )

    def test_starts_at_zero(self) -> None:
        """First element of cumulative sum is always zero."""
        f = jnp.ones(N_TIMES)
        cumsum = trapezoid_cumsum(f, 0.01)
        assert float(cumsum[0]) == 0.0

    def test_finite_output(self) -> None:
        """Output is finite for typical positive input."""
        f = jnp.linspace(1.0, 100.0, N_TIMES)
        cumsum = trapezoid_cumsum(f, 0.001)
        assert jnp.all(jnp.isfinite(cumsum))


# ---------------------------------------------------------------------------
# create_time_integral_matrix
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestTimeIntegralMatrix:
    """create_time_integral_matrix is antisymmetric; smooth_abs result is symmetric and non-negative."""

    def test_symmetric_after_smooth_abs(self) -> None:
        """smooth_abs(create_time_integral_matrix(cumsum)) is symmetric."""
        cumsum = jnp.linspace(0.0, 1.0, N_TIMES)
        matrix = smooth_abs(create_time_integral_matrix(cumsum))
        diff = jnp.abs(matrix - matrix.T)
        assert jnp.max(diff) < 1e-12, "Matrix is not symmetric after smooth_abs"

    def test_non_negative_after_smooth_abs(self) -> None:
        """smooth_abs(create_time_integral_matrix(cumsum)) is non-negative."""
        cumsum = jnp.linspace(0.0, 5.0, N_TIMES)
        matrix = smooth_abs(create_time_integral_matrix(cumsum))
        assert jnp.all(matrix >= 0.0), "Matrix has negative entries after smooth_abs"

    def test_antisymmetric_raw(self) -> None:
        """Raw create_time_integral_matrix is antisymmetric: M[i,j] = -M[j,i]."""
        cumsum = jnp.linspace(0.0, 2.0, N_TIMES)
        matrix = create_time_integral_matrix(cumsum)
        assert_allclose(
            np.asarray(matrix + matrix.T),
            np.zeros((N_TIMES, N_TIMES)),
            atol=1e-12,
        )

    def test_diagonal_is_zero(self) -> None:
        """Diagonal of raw integral matrix is zero (integral from t_i to t_i)."""
        cumsum = jnp.linspace(0.0, 1.0, N_TIMES)
        matrix = create_time_integral_matrix(cumsum)
        assert_allclose(np.asarray(jnp.diag(matrix)), np.zeros(N_TIMES), atol=1e-15)


# ---------------------------------------------------------------------------
# compute_c2_heterodyne — extreme parameters
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestC2HeterodyneStability:
    """compute_c2_heterodyne is finite for extreme but valid parameter values."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.t, self.dt = _time_grid()
        self.base_params = _default_params()

    def test_finite_for_large_d0(self) -> None:
        """D0=1e6, alpha=0.99 produces finite c2."""
        params = self.base_params.at[0].set(1e6)  # D0_ref
        params = params.at[1].set(0.99)  # alpha_ref
        params = params.at[3].set(1e6)  # D0_sample
        params = params.at[4].set(0.99)  # alpha_sample
        c2 = compute_c2_heterodyne(params, self.t, Q, self.dt, PHI_ANGLE)
        assert jnp.all(jnp.isfinite(c2)), "c2 contains non-finite values for large D0"

    def test_finite_for_minimal_d0(self) -> None:
        """D0=100, alpha=0.01 produces finite c2."""
        params = self.base_params.at[0].set(100.0)  # D0_ref
        params = params.at[1].set(0.01)  # alpha_ref
        params = params.at[3].set(100.0)  # D0_sample
        params = params.at[4].set(0.01)  # alpha_sample
        c2 = compute_c2_heterodyne(params, self.t, Q, self.dt, PHI_ANGLE)
        assert jnp.all(jnp.isfinite(c2)), "c2 contains non-finite values for minimal D0"

    def test_finite_for_large_velocity(self) -> None:
        """v0=1e4, beta=1.5 produces finite c2."""
        params = self.base_params.at[6].set(1e4)  # v0
        params = params.at[7].set(1.5)  # beta
        c2 = compute_c2_heterodyne(params, self.t, Q, self.dt, PHI_ANGLE)
        assert jnp.all(jnp.isfinite(c2)), (
            "c2 contains non-finite values for large velocity"
        )

    def test_finite_for_zero_fraction(self) -> None:
        """f0=0, f3=0 (pure reference) produces finite c2."""
        params = self.base_params.at[9].set(0.0)  # f0
        params = params.at[12].set(0.0)  # f3
        c2 = compute_c2_heterodyne(params, self.t, Q, self.dt, PHI_ANGLE)
        assert jnp.all(jnp.isfinite(c2)), (
            "c2 contains non-finite values for zero fraction"
        )

    def test_finite_for_full_fraction(self) -> None:
        """f0=1, f3=0 (pure sample) produces finite c2."""
        params = self.base_params.at[9].set(1.0)  # f0
        params = params.at[12].set(0.0)  # f3
        c2 = compute_c2_heterodyne(params, self.t, Q, self.dt, PHI_ANGLE)
        assert jnp.all(jnp.isfinite(c2)), (
            "c2 contains non-finite values for full fraction"
        )

    def test_correlation_diagonal_geq_one(self) -> None:
        """Correlation matrix diagonal is >= 1.0 (physical constraint: c2(t,t) >= offset)."""
        c2 = compute_c2_heterodyne(
            self.base_params,
            self.t,
            Q,
            self.dt,
            PHI_ANGLE,
            contrast=0.5,
            offset=1.0,
        )
        diag = jnp.diag(c2)
        # With offset=1.0 and non-negative contrast term, diagonal >= 1.0
        assert jnp.all(diag >= 1.0 - 1e-10), (
            f"Diagonal minimum {float(jnp.min(diag)):.6f} violates physical constraint >= 1.0"
        )

    def test_parameters_at_bounds_no_nan(self) -> None:
        """Parameters set to their registry bounds produce no NaN in c2."""
        # Lower bounds
        params_low = jnp.array(
            [DEFAULT_REGISTRY[name].min_bound for name in ALL_PARAM_NAMES],
            dtype=jnp.float64,
        )
        c2_low = compute_c2_heterodyne(params_low, self.t, Q, self.dt, PHI_ANGLE)
        assert not jnp.any(jnp.isnan(c2_low)), "c2 has NaN at lower bounds"

        # Upper bounds
        params_high = jnp.array(
            [DEFAULT_REGISTRY[name].max_bound for name in ALL_PARAM_NAMES],
            dtype=jnp.float64,
        )
        c2_high = compute_c2_heterodyne(params_high, self.t, Q, self.dt, PHI_ANGLE)
        assert not jnp.any(jnp.isnan(c2_high)), "c2 has NaN at upper bounds"


# ---------------------------------------------------------------------------
# solve_least_squares_jax — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestLeastSquaresStability:
    """solve_least_squares_jax handles degenerate and extreme inputs."""

    def test_near_constant_theory_no_nan(self) -> None:
        """Near-constant theory array does not produce NaN."""
        rng = np.random.default_rng(42)
        n_angles, n_data = 3, 100
        # Theory is nearly constant (small variance)
        theory = jnp.ones((n_angles, n_data)) + 1e-10 * jnp.array(
            rng.standard_normal((n_angles, n_data))
        )
        exp_data = jnp.array(rng.standard_normal((n_angles, n_data))) + 1.0

        contrast, offset = solve_least_squares_jax(theory, exp_data)

        assert jnp.all(jnp.isfinite(contrast)), "contrast has non-finite values"
        assert jnp.all(jnp.isfinite(offset)), "offset has non-finite values"

    def test_large_contrast_ratio(self) -> None:
        """Theory scaled by 1000 + offset still yields finite results."""
        rng = np.random.default_rng(42)
        n_angles, n_data = 2, 50
        base_theory = jnp.array(rng.random((n_angles, n_data))) + 0.5
        theory = base_theory * 1000.0
        # exp_data = contrast * theory + offset  (with known contrast/offset)
        exp_data = 0.3 * theory + 5.0

        contrast, offset = solve_least_squares_jax(theory, exp_data)

        assert jnp.all(jnp.isfinite(contrast)), "contrast has non-finite values"
        assert jnp.all(jnp.isfinite(offset)), "offset has non-finite values"
        # Recovered contrast should be close to 0.3
        assert_allclose(np.asarray(contrast), 0.3 * np.ones(n_angles), rtol=1e-4)

    def test_identical_theory_and_exp(self) -> None:
        """When theory == exp, contrast ~ 1 and offset ~ 0."""
        n_angles, n_data = 2, 64
        rng = np.random.default_rng(42)
        data = jnp.array(rng.random((n_angles, n_data))) + 1.0

        contrast, offset = solve_least_squares_jax(data, data)

        assert_allclose(np.asarray(contrast), np.ones(n_angles), rtol=1e-4)
        assert_allclose(np.asarray(offset), np.zeros(n_angles), atol=1e-4)

    def test_constant_theory_fallback(self) -> None:
        """Exactly constant theory triggers the singular fallback (det ~ 0)."""
        n_angles, n_data = 1, 50
        theory = jnp.ones((n_angles, n_data)) * 5.0
        exp_data = jnp.ones((n_angles, n_data)) * 3.0

        contrast, offset = solve_least_squares_jax(theory, exp_data)

        # Fallback returns contrast=1.0, offset=0.0 (or similar safe defaults)
        assert jnp.all(jnp.isfinite(contrast))
        assert jnp.all(jnp.isfinite(offset))
