"""Regression tests for NLSQ numerical utilities.

All tests use deterministic synthetic inputs — no real data files are
required.  The expected values are derived analytically so they can be
computed by hand and verified independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.cmaes_wrapper import (
    denormalize_from_unit_cube,
    normalize_to_unit_cube,
)
from heterodyne.optimization.nlsq.jacobian import (
    compute_jacobian_condition_number,
    compute_numerical_jacobian,
)
from heterodyne.optimization.nlsq.multistart import generate_lhs_starts


# ---------------------------------------------------------------------------
# compute_numerical_jacobian
# ---------------------------------------------------------------------------


class TestJacobianDeterministic:
    """Central-difference Jacobian against analytically known functions."""

    def test_quadratic_residual_f_x_squared(self) -> None:
        # f(x) = [x[0]^2, x[1]^2]  →  J = diag(2*x[0], 2*x[1])
        def f(x: np.ndarray) -> np.ndarray:
            return np.array([x[0] ** 2, x[1] ** 2])

        x0 = np.array([3.0, 5.0])
        J = compute_numerical_jacobian(f, x0)

        # Analytic Jacobian at x0: [[6, 0], [0, 10]]
        J_analytic = np.diag([2 * x0[0], 2 * x0[1]])
        np.testing.assert_allclose(J, J_analytic, rtol=1e-5, atol=1e-8)

    def test_linear_residual_exact_jacobian(self) -> None:
        # f(x) = A @ x  →  J = A (exact for central differences)
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(x: np.ndarray) -> np.ndarray:
            return A @ x

        x0 = np.array([1.0, 1.0])
        J = compute_numerical_jacobian(f, x0)
        np.testing.assert_allclose(J, A, rtol=1e-8, atol=1e-10)

    def test_output_shape(self) -> None:
        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(x)

        x0 = np.ones(5)
        J = compute_numerical_jacobian(f, x0)
        assert J.shape == (5, 5)

    def test_constant_residual_gives_zero_jacobian(self) -> None:
        def f(x: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        x0 = np.array([1.0, 2.0])
        J = compute_numerical_jacobian(f, x0)
        np.testing.assert_allclose(J, np.zeros((3, 2)), atol=1e-10)

    def test_custom_step_sizes(self) -> None:
        # Verifies that explicit step_sizes parameter is accepted
        def f(x: np.ndarray) -> np.ndarray:
            return x ** 2

        x0 = np.array([2.0, 4.0])
        steps = np.array([1e-6, 1e-6])
        J = compute_numerical_jacobian(f, x0, step_sizes=steps)
        J_expected = np.diag(2 * x0)
        np.testing.assert_allclose(J, J_expected, rtol=1e-5)

    def test_step_size_shape_mismatch_raises(self) -> None:
        def f(x: np.ndarray) -> np.ndarray:
            return x

        x0 = np.array([1.0, 2.0])
        wrong_steps = np.array([1e-6])
        with pytest.raises(ValueError, match="step_sizes shape"):
            compute_numerical_jacobian(f, x0, step_sizes=wrong_steps)


# ---------------------------------------------------------------------------
# compute_jacobian_condition_number
# ---------------------------------------------------------------------------


class TestConditionNumber:
    def test_identity_jacobian_has_condition_one(self) -> None:
        J = np.eye(4)
        cond = compute_jacobian_condition_number(J)
        # cond(I^T I) = cond(I) = 1
        assert cond == pytest.approx(1.0, rel=1e-8)

    def test_scaled_identity_has_condition_one(self) -> None:
        # Scaling all rows/columns uniformly: J = 3*I  →  J^T J = 9*I  →  cond = 1
        J = 3.0 * np.eye(3)
        cond = compute_jacobian_condition_number(J)
        assert cond == pytest.approx(1.0, rel=1e-8)

    def test_ill_conditioned_jacobian(self) -> None:
        # Near-zero column makes J^T J very ill-conditioned
        J = np.array(
            [
                [1.0, 0.0],
                [0.0, 1e-8],
            ],
            dtype=np.float64,
        )
        cond = compute_jacobian_condition_number(J)
        # J^T J = diag(1, 1e-16)  →  cond = 1e16
        assert cond > 1e10

    def test_returns_float(self) -> None:
        J = np.eye(2)
        cond = compute_jacobian_condition_number(J)
        assert isinstance(cond, float)

    def test_2x1_jacobian(self) -> None:
        # J has shape (2, 1); J^T J is a 1x1 matrix
        J = np.array([[3.0], [4.0]])
        cond = compute_jacobian_condition_number(J)
        # J^T J = [[25]], cond = 1
        assert cond == pytest.approx(1.0, rel=1e-8)

    def test_diagonal_jacobian_condition_matches_numpy(self) -> None:
        diag_vals = np.array([1.0, 2.0, 4.0])
        J = np.diag(diag_vals)
        cond = compute_jacobian_condition_number(J)
        jtj = J.T @ J
        expected = float(np.linalg.cond(jtj))
        assert cond == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# generate_lhs_starts — reproducibility
# ---------------------------------------------------------------------------


class TestLHSReproducibility:
    def test_same_seed_gives_identical_results(self) -> None:
        lower = np.zeros(4)
        upper = np.ones(4)
        s1 = generate_lhs_starts(8, lower, upper, seed=42)
        s2 = generate_lhs_starts(8, lower, upper, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_results(self) -> None:
        lower = np.zeros(4)
        upper = np.ones(4)
        s1 = generate_lhs_starts(8, lower, upper, seed=42)
        s2 = generate_lhs_starts(8, lower, upper, seed=99)
        assert not np.array_equal(s1, s2)

    def test_output_shape(self) -> None:
        lower = np.zeros(6)
        upper = np.ones(6)
        starts = generate_lhs_starts(10, lower, upper, seed=0)
        assert starts.shape == (10, 6)

    def test_all_starts_within_bounds(self) -> None:
        lower = np.array([1.0, -2.0, 0.5])
        upper = np.array([3.0, 0.0, 4.5])
        starts = generate_lhs_starts(20, lower, upper, seed=7)
        assert np.all(starts >= lower)
        assert np.all(starts <= upper)

    def test_fixed_dimension_stays_at_bound(self) -> None:
        lower = np.array([0.0, 5.0, 0.0])
        upper = np.array([1.0, 5.0, 1.0])  # dim 1 is fixed
        starts = generate_lhs_starts(5, lower, upper, seed=0)
        np.testing.assert_array_equal(starts[:, 1], 5.0)

    def test_lhs_stratification_property(self) -> None:
        # For n_starts=n, each dimension should have exactly one sample
        # per stratum [k/n, (k+1)/n].
        n = 10
        lower = np.zeros(1)
        upper = np.ones(1)
        starts = generate_lhs_starts(n, lower, upper, seed=42)
        values = np.sort(starts[:, 0])
        strata_indices = (values * n).astype(int).clip(0, n - 1)
        assert len(np.unique(strata_indices)) == n

    def test_n_starts_one_returns_single_row(self) -> None:
        lower = np.zeros(3)
        upper = np.ones(3)
        starts = generate_lhs_starts(1, lower, upper, seed=0)
        assert starts.shape == (1, 3)

    def test_invalid_n_starts_raises(self) -> None:
        with pytest.raises(ValueError, match="n_starts"):
            generate_lhs_starts(0, np.zeros(2), np.ones(2), seed=0)


# ---------------------------------------------------------------------------
# normalize_to_unit_cube / denormalize_from_unit_cube round-trip
# ---------------------------------------------------------------------------


class TestNormalizeDenormalizePrecision:
    def test_round_trip_identity(self) -> None:
        lower = np.array([1.0, -5.0, 100.0])
        upper = np.array([2.0, 5.0, 200.0])
        x = np.array([1.5, 0.0, 150.0])

        x_norm = normalize_to_unit_cube(x, lower, upper)
        x_back = denormalize_from_unit_cube(x_norm, lower, upper)
        np.testing.assert_allclose(x_back, x, atol=1e-12)

    def test_round_trip_at_lower_bound(self) -> None:
        lower = np.array([0.0, -1.0])
        upper = np.array([10.0, 1.0])
        x = lower.copy()
        x_norm = normalize_to_unit_cube(x, lower, upper)
        x_back = denormalize_from_unit_cube(x_norm, lower, upper)
        np.testing.assert_allclose(x_back, x, atol=1e-12)

    def test_round_trip_at_upper_bound(self) -> None:
        lower = np.array([0.0, -1.0])
        upper = np.array([10.0, 1.0])
        x = upper.copy()
        x_norm = normalize_to_unit_cube(x, lower, upper)
        x_back = denormalize_from_unit_cube(x_norm, lower, upper)
        np.testing.assert_allclose(x_back, x, atol=1e-12)

    def test_normalized_values_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        lower = np.array([0.0, -10.0, 100.0])
        upper = np.array([5.0, 10.0, 1000.0])
        # Generate points strictly within bounds
        x = lower + rng.random(3) * (upper - lower)
        x_norm = normalize_to_unit_cube(x, lower, upper)
        assert np.all(x_norm >= 0.0)
        assert np.all(x_norm <= 1.0)

    def test_fixed_dimension_maps_to_zero(self) -> None:
        lower = np.array([0.0, 3.0, 0.0])
        upper = np.array([1.0, 3.0, 1.0])  # dim 1 fixed
        x = np.array([0.5, 3.0, 0.5])
        x_norm = normalize_to_unit_cube(x, lower, upper)
        assert x_norm[1] == pytest.approx(0.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            normalize_to_unit_cube(
                np.array([1.0, 2.0]),
                np.array([0.0]),
                np.array([1.0]),
            )

    def test_round_trip_high_precision_large_values(self) -> None:
        # Physics-relevant scale: diffusion coefficients can be ~1e4
        lower = np.array([1.0, 1e3])
        upper = np.array([1e6, 1e5])
        x = np.array([5000.0, 50000.0])
        x_norm = normalize_to_unit_cube(x, lower, upper)
        x_back = denormalize_from_unit_cube(x_norm, lower, upper)
        np.testing.assert_allclose(x_back, x, atol=1e-12)

    def test_batch_round_trip(self) -> None:
        # Verify all points in a batch survive round-trip
        rng = np.random.default_rng(7)
        lower = np.array([-1.0, 0.0, 10.0])
        upper = np.array([1.0, 100.0, 20.0])
        n = 50
        xs = lower + rng.random((n, 3)) * (upper - lower)

        for i in range(n):
            x_norm = normalize_to_unit_cube(xs[i], lower, upper)
            x_back = denormalize_from_unit_cube(x_norm, lower, upper)
            np.testing.assert_allclose(x_back, xs[i], atol=1e-12)
