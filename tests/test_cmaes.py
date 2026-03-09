"""Unit tests for CMA-ES wrapper utilities.

Focuses on CMAESResult, coordinate transformation, and covariance
adjustment functions — all importable without the optional ``cma`` package.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.cmaes_wrapper import (
    CMAESResult,
    adjust_covariance_for_bounds,
    denormalize_from_unit_cube,
    normalize_to_unit_cube,
)


# ---------------------------------------------------------------------------
# CMAESResult
# ---------------------------------------------------------------------------


class TestCMAESResult:
    """Tests for the CMAESResult dataclass."""

    def _make_result(self, **overrides: object) -> CMAESResult:
        defaults: dict[str, object] = {
            "best_params": {"D0_ref": 1e4, "alpha_ref": 1.8},
            "best_cost": 0.05,
            "n_iterations": 42,
            "n_evaluations": 1000,
            "converged": True,
            "final_sigma": 1e-3,
            "history": [1.0, 0.5, 0.2, 0.05],
        }
        defaults.update(overrides)
        return CMAESResult(**defaults)  # type: ignore[arg-type]

    def test_cmaes_result_creation(self) -> None:
        """CMAESResult can be created with required fields."""
        result = self._make_result()
        assert result.best_cost == pytest.approx(0.05)
        assert result.n_iterations == 42
        assert result.n_evaluations == 1000
        assert result.converged is True
        assert result.final_sigma == pytest.approx(1e-3)

    def test_cmaes_result_best_params(self) -> None:
        """best_params dictionary is accessible and has correct content."""
        result = self._make_result(best_params={"p0": 1.5, "p1": 2.5})
        assert result.best_params["p0"] == pytest.approx(1.5)
        assert result.best_params["p1"] == pytest.approx(2.5)

    def test_cmaes_result_convergence_history(self) -> None:
        """history field stores and returns a list of floats."""
        history = [10.0, 5.0, 2.5, 1.0, 0.1]
        result = self._make_result(history=history)
        assert result.history == history
        assert len(result.history) == 5
        # History should be monotonically decreasing in a typical run
        assert result.history[0] >= result.history[-1]

    def test_cmaes_result_is_frozen(self) -> None:
        """CMAESResult is frozen — attribute mutation raises AttributeError."""
        result = self._make_result()
        with pytest.raises(AttributeError):
            result.best_cost = 999.0  # type: ignore[misc]

    def test_cmaes_result_empty_params(self) -> None:
        """Empty best_params dict is accepted."""
        result = self._make_result(best_params={})
        assert result.best_params == {}


# ---------------------------------------------------------------------------
# normalize_to_unit_cube
# ---------------------------------------------------------------------------


class TestNormalizeToUnitCube:
    """Tests for the normalize_to_unit_cube transform."""

    def test_normalize_known_input(self) -> None:
        """Midpoint of each dimension maps to 0.5."""
        lower = np.array([0.0, -1.0, 10.0])
        upper = np.array([2.0, 1.0, 20.0])
        x = (lower + upper) / 2.0
        result = normalize_to_unit_cube(x, lower, upper)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_normalize_lower_bound_maps_to_zero(self) -> None:
        """Lower bound maps to 0."""
        lower = np.array([1.0, 2.0])
        upper = np.array([3.0, 8.0])
        result = normalize_to_unit_cube(lower, lower, upper)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_normalize_upper_bound_maps_to_one(self) -> None:
        """Upper bound maps to 1."""
        lower = np.array([0.0, 5.0])
        upper = np.array([10.0, 15.0])
        result = normalize_to_unit_cube(upper, lower, upper)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_normalize_fixed_dimension_maps_to_zero(self) -> None:
        """Fixed dimensions (lower == upper) map to 0."""
        lower = np.array([5.0, 0.0])
        upper = np.array([5.0, 1.0])  # dim 0 is fixed
        x = np.array([5.0, 0.5])
        result = normalize_to_unit_cube(x, lower, upper)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)

    def test_normalize_shape_mismatch_raises(self) -> None:
        """Mismatched array shapes raise ValueError."""
        with pytest.raises(ValueError):
            normalize_to_unit_cube(
                np.array([1.0, 2.0]),
                np.array([0.0]),
                np.array([3.0]),
            )


# ---------------------------------------------------------------------------
# denormalize_from_unit_cube
# ---------------------------------------------------------------------------


class TestDenormalizeFromUnitCube:
    """Tests for the denormalize_from_unit_cube transform."""

    def test_denormalize_zero_maps_to_lower(self) -> None:
        """0 maps back to lower bound."""
        lower = np.array([2.0, -3.0])
        upper = np.array([10.0, 3.0])
        result = denormalize_from_unit_cube(np.zeros(2), lower, upper)
        np.testing.assert_allclose(result, lower)

    def test_denormalize_one_maps_to_upper(self) -> None:
        """1 maps back to upper bound."""
        lower = np.array([0.0, 100.0])
        upper = np.array([1.0, 200.0])
        result = denormalize_from_unit_cube(np.ones(2), lower, upper)
        np.testing.assert_allclose(result, upper)

    def test_denormalize_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            denormalize_from_unit_cube(
                np.array([0.5]),
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
            )


# ---------------------------------------------------------------------------
# Roundtrip: normalize → denormalize
# ---------------------------------------------------------------------------


class TestNormalizeDenormalizeRoundtrip:
    """Property tests for the normalize/denormalize pair."""

    def test_normalize_denormalize_roundtrip(self) -> None:
        """x == denormalize(normalize(x)) for arbitrary bounded x."""
        rng = np.random.default_rng(0)
        lower = rng.uniform(-10, 0, size=5)
        upper = lower + rng.uniform(0.1, 20.0, size=5)
        x = lower + rng.random(5) * (upper - lower)

        x_norm = normalize_to_unit_cube(x, lower, upper)
        x_back = denormalize_from_unit_cube(x_norm, lower, upper)
        np.testing.assert_allclose(x_back, x, rtol=1e-12, atol=1e-14)

    def test_roundtrip_at_boundaries(self) -> None:
        """Boundaries survive a normalize → denormalize roundtrip exactly."""
        lower = np.array([0.0, -5.0, 100.0])
        upper = np.array([1.0, 5.0, 200.0])
        for x in (lower, upper):
            x_norm = normalize_to_unit_cube(x, lower, upper)
            x_back = denormalize_from_unit_cube(x_norm, lower, upper)
            np.testing.assert_allclose(x_back, x, atol=1e-14)


# ---------------------------------------------------------------------------
# adjust_covariance_for_bounds
# ---------------------------------------------------------------------------


class TestAdjustCovarianceForBounds:
    """Tests for adjust_covariance_for_bounds."""

    def test_adjust_identity_unit_bounds(self) -> None:
        """Identity covariance with unit-range bounds → identity output."""
        n = 3
        cov = np.eye(n)
        lower = np.zeros(n)
        upper = np.ones(n)  # span = 1 for all dims
        result = adjust_covariance_for_bounds(cov, lower, upper)
        np.testing.assert_allclose(result, np.eye(n))

    def test_adjust_identity_nontrivial_bounds(self) -> None:
        """Identity covariance scaled by outer product of spans.

        cov_adjusted[i,j] = cov[i,j] * span_i * span_j.
        For the identity: diagonal = span_i^2, off-diagonal = 0.
        """
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 4.0])  # spans: [2, 4]
        cov = np.eye(2)
        # identity * outer([2,4],[2,4]) = diag([4, 16]) (off-diag of identity is 0)
        expected = np.array([[4.0, 0.0], [0.0, 16.0]])
        result = adjust_covariance_for_bounds(cov, lower, upper)
        np.testing.assert_allclose(result, expected)

    def test_adjust_ones_matrix_nontrivial_bounds(self) -> None:
        """Ones covariance (all elements = 1) → outer product of spans."""
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 4.0])  # spans: [2, 4]
        cov = np.ones((2, 2))
        # cov_adjusted[i,j] = 1 * span_i * span_j
        expected = np.outer([2.0, 4.0], [2.0, 4.0])
        result = adjust_covariance_for_bounds(cov, lower, upper)
        np.testing.assert_allclose(result, expected)

    def test_adjust_symmetry_preserved(self) -> None:
        """Adjusted covariance matrix remains symmetric."""
        rng = np.random.default_rng(3)
        n = 4
        # Construct a symmetric positive-definite cov
        a = rng.random((n, n))
        cov = a @ a.T + np.eye(n)
        lower = np.zeros(n)
        upper = rng.uniform(1.0, 10.0, size=n)
        result = adjust_covariance_for_bounds(cov, lower, upper)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_adjust_shape_mismatch_raises(self) -> None:
        """Mismatched cov shape raises ValueError."""
        cov = np.eye(3)
        lower = np.zeros(2)
        upper = np.ones(2)
        with pytest.raises(ValueError):
            adjust_covariance_for_bounds(cov, lower, upper)
