"""Tests for heterodyne.optimization.nlsq.data_prep."""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.data_prep import (
    compute_degrees_of_freedom,
    compute_weights,
    flatten_upper_triangle,
    prepare_fit_data,
    unflatten_upper_triangle,
)


# ---------------------------------------------------------------------------
# flatten_upper_triangle
# ---------------------------------------------------------------------------


class TestFlattenUpperTriangle:
    """Tests for flatten_upper_triangle."""

    def test_identity_3x3_with_diagonal(self) -> None:
        mat = np.eye(3)
        flat = flatten_upper_triangle(mat, include_diagonal=True)
        # upper triangle of identity: (0,0)=1, (0,1)=0, (0,2)=0,
        #                             (1,1)=1, (1,2)=0, (2,2)=1
        expected = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(flat, expected)

    def test_identity_3x3_no_diagonal(self) -> None:
        mat = np.eye(3)
        flat = flatten_upper_triangle(mat, include_diagonal=False)
        # strictly upper: (0,1)=0, (0,2)=0, (1,2)=0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(flat, expected)

    def test_full_matrix(self) -> None:
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        flat = flatten_upper_triangle(mat, include_diagonal=True)
        expected = np.array([1, 2, 3, 5, 6, 9])
        np.testing.assert_array_equal(flat, expected)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected square matrix"):
            flatten_upper_triangle(np.zeros((2, 3)))

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected square matrix"):
            flatten_upper_triangle(np.zeros(5))

    def test_1x1_matrix(self) -> None:
        mat = np.array([[42.0]])
        flat = flatten_upper_triangle(mat, include_diagonal=True)
        np.testing.assert_array_equal(flat, np.array([42.0]))

    def test_1x1_no_diagonal(self) -> None:
        mat = np.array([[42.0]])
        flat = flatten_upper_triangle(mat, include_diagonal=False)
        assert len(flat) == 0

    def test_element_count_with_diagonal(self) -> None:
        n = 5
        mat = np.ones((n, n))
        flat = flatten_upper_triangle(mat, include_diagonal=True)
        assert len(flat) == n * (n + 1) // 2

    def test_element_count_without_diagonal(self) -> None:
        n = 5
        mat = np.ones((n, n))
        flat = flatten_upper_triangle(mat, include_diagonal=False)
        assert len(flat) == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# unflatten_upper_triangle
# ---------------------------------------------------------------------------


class TestUnflattenUpperTriangle:
    """Tests for unflatten_upper_triangle."""

    def test_roundtrip_with_diagonal(self) -> None:
        original = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=float)
        flat = flatten_upper_triangle(original, include_diagonal=True)
        recovered = unflatten_upper_triangle(flat, n=3, include_diagonal=True)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_roundtrip_without_diagonal(self) -> None:
        # Symmetric matrix with zero diagonal
        original = np.array([[0, 2, 3], [2, 0, 6], [3, 6, 0]], dtype=float)
        flat = flatten_upper_triangle(original, include_diagonal=False)
        recovered = unflatten_upper_triangle(flat, n=3, include_diagonal=False)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_symmetry(self) -> None:
        flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = unflatten_upper_triangle(flat, n=3, include_diagonal=True)
        np.testing.assert_array_equal(mat, mat.T)

    def test_wrong_length_raises(self) -> None:
        flat = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Expected .* values"):
            unflatten_upper_triangle(flat, n=3, include_diagonal=True)

    def test_diagonal_preserved(self) -> None:
        flat = np.array([10.0, 1.0, 2.0, 20.0, 3.0, 30.0])
        mat = unflatten_upper_triangle(flat, n=3, include_diagonal=True)
        np.testing.assert_array_equal(np.diag(mat), [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------


class TestComputeWeights:
    """Tests for compute_weights."""

    def test_uniform(self) -> None:
        data = np.ones((4, 4))
        w = compute_weights(data, method="uniform")
        np.testing.assert_array_equal(w, np.ones((4, 4)))

    def test_inverse_variance(self) -> None:
        data = np.ones((3, 3))
        sigma = 2.0 * np.ones((3, 3))
        w = compute_weights(data, method="inverse_variance", sigma=sigma)
        expected = 1.0 / 4.0  # 1/sigma^2
        np.testing.assert_allclose(w, expected)

    def test_inverse_variance_no_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma required"):
            compute_weights(np.ones((3, 3)), method="inverse_variance")

    def test_inverse_variance_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma shape"):
            compute_weights(
                np.ones((3, 3)),
                method="inverse_variance",
                sigma=np.ones((2, 2)),
            )

    def test_data_amplitude(self) -> None:
        data = np.full((3, 3), 5.0)
        w = compute_weights(data, method="data_amplitude")
        np.testing.assert_allclose(w, 1.0 / 5.0)

    def test_data_amplitude_zero_safe(self) -> None:
        data = np.zeros((3, 3))
        w = compute_weights(data, method="data_amplitude")
        # Should clamp to 1e-30, so w = 1/1e-30 = 1e30
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight method"):
            compute_weights(np.ones((3, 3)), method="bogus")

    def test_exclude_diagonal(self) -> None:
        data = np.ones((4, 4))
        w = compute_weights(data, method="uniform", exclude_diagonal=True)
        np.testing.assert_array_equal(np.diag(w), np.zeros(4))
        # Off-diagonal should still be 1
        assert w[0, 1] == 1.0

    def test_inverse_variance_near_zero_sigma(self) -> None:
        """Sigma near zero should not produce inf weights."""
        data = np.ones((2, 2))
        sigma = np.full((2, 2), 1e-40)
        w = compute_weights(data, method="inverse_variance", sigma=sigma)
        assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# prepare_fit_data
# ---------------------------------------------------------------------------


class TestPrepareFitData:
    """Tests for prepare_fit_data."""

    def test_upper_triangle_default(self) -> None:
        mat = np.arange(9, dtype=float).reshape(3, 3)
        data_flat, sqrt_w, n_data = prepare_fit_data(mat)
        # 6 upper-triangle elements (with diagonal)
        assert len(data_flat) == 6
        assert len(sqrt_w) == 6
        # All weights are 1 -> sqrt = 1 -> n_data = 6
        assert n_data == 6

    def test_full_matrix(self) -> None:
        mat = np.ones((3, 3))
        data_flat, sqrt_w, n_data = prepare_fit_data(mat, use_upper_triangle=False)
        assert len(data_flat) == 9
        assert n_data == 9

    def test_custom_weights(self) -> None:
        mat = np.ones((3, 3))
        weights = 4.0 * np.ones((3, 3))
        data_flat, sqrt_w, n_data = prepare_fit_data(mat, weights=weights)
        np.testing.assert_allclose(sqrt_w, 2.0)

    def test_exclude_diagonal(self) -> None:
        mat = np.ones((3, 3))
        data_flat, sqrt_w, n_data = prepare_fit_data(mat, exclude_diagonal=True)
        # Diagonal weights are zeroed; 3 diagonal elements -> 3 zero weights
        # Total upper triangle = 6, non-zero = 3
        assert n_data == 3

    def test_negative_weights_clamped(self) -> None:
        mat = np.ones((2, 2))
        weights = -1.0 * np.ones((2, 2))
        _, sqrt_w, n_data = prepare_fit_data(mat, weights=weights)
        # sqrt(max(-1, 0)) = 0 -> all zero
        np.testing.assert_array_equal(sqrt_w, 0.0)
        assert n_data == 0


# ---------------------------------------------------------------------------
# compute_degrees_of_freedom
# ---------------------------------------------------------------------------


class TestComputeDegreesOfFreedom:
    """Tests for compute_degrees_of_freedom."""

    def test_normal_case(self) -> None:
        assert compute_degrees_of_freedom(100, 14) == 86

    def test_minimum_is_one(self) -> None:
        assert compute_degrees_of_freedom(5, 10) == 1

    def test_equal_params_and_data(self) -> None:
        assert compute_degrees_of_freedom(10, 10) == 1

    def test_zero_data(self) -> None:
        assert compute_degrees_of_freedom(0, 5) == 1
