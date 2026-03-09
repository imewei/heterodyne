"""Unit tests for diagonal_correction module.

Covers compute_diagonal_mask, apply_diagonal_correction (all estimators),
estimate_diagonal_excess, compute_weights_excluding_diagonal, and the
batch API.
"""

from __future__ import annotations

import numpy as np
import pytest

# JAX is required for the diagonal_correction functions
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from heterodyne.core.diagonal_correction import (
    apply_diagonal_correction,
    apply_diagonal_correction_batch,
    compute_diagonal_mask,
    compute_weights_excluding_diagonal,
    estimate_diagonal_excess,
    get_available_backends,
    get_diagonal_correction_methods,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_c2(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a symmetric correlation matrix with elevated diagonal."""
    if rng is None:
        rng = np.random.default_rng(0)
    base = rng.uniform(0.9, 1.1, (n, n))
    matrix = (base + base.T) / 2.0
    # Elevate the diagonal to simulate the artifact
    np.fill_diagonal(matrix, matrix.diagonal() + 2.0)
    return matrix.astype(np.float64)


# ---------------------------------------------------------------------------
# compute_diagonal_mask
# ---------------------------------------------------------------------------


class TestComputeDiagonalMask:
    """Tests for compute_diagonal_mask."""

    def test_mask_width_one_is_identity_mask(self) -> None:
        """width=1 masks only the main diagonal."""
        n = 5
        mask = np.asarray(compute_diagonal_mask(n, width=1))
        expected = np.eye(n, dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_mask_width_two_includes_neighbours(self) -> None:
        """width=2 masks main diagonal and immediate neighbours."""
        mask = np.asarray(compute_diagonal_mask(4, width=2))
        # Element (0,1) should be masked; element (0,2) should not
        assert mask[0, 1]
        assert not mask[0, 2]

    def test_mask_shape(self) -> None:
        """Mask has the expected square shape."""
        n = 8
        mask = compute_diagonal_mask(n, width=1)
        assert mask.shape == (n, n)

    def test_mask_invalid_width(self) -> None:
        """width < 1 raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            compute_diagonal_mask(5, width=0)


# ---------------------------------------------------------------------------
# apply_diagonal_correction — basic behaviour
# ---------------------------------------------------------------------------


class TestDiagonalCorrectionBasic:
    """Basic correctness tests for apply_diagonal_correction."""

    @pytest.mark.parametrize("method", ["interpolate", "mask", "mirror"])
    def test_correction_returns_same_shape(self, method: str) -> None:
        """Corrected matrix has the same shape as input."""
        c2 = jnp.array(_make_c2(6))
        result = apply_diagonal_correction(c2, width=1, method=method)
        assert result.shape == c2.shape

    def test_interpolate_modifies_diagonal(self) -> None:
        """Interpolation method changes the diagonal elements."""
        n = 10
        c2 = jnp.array(_make_c2(n))
        result = apply_diagonal_correction(c2, method="interpolate")
        original_diag = np.diag(np.asarray(c2))
        corrected_diag = np.diag(np.asarray(result))
        # The diagonal should have changed
        assert not np.allclose(original_diag, corrected_diag)

    def test_interpolate_preserves_off_diagonal(self) -> None:
        """Interpolation does not modify off-diagonal elements."""
        n = 8
        c2 = jnp.array(_make_c2(n))
        result = apply_diagonal_correction(c2, width=1, method="interpolate")
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(
            np.asarray(result)[mask], np.asarray(c2)[mask]
        )

    def test_mask_method_sets_diagonal_to_nan(self) -> None:
        """Mask method sets diagonal elements to NaN."""
        n = 5
        c2 = jnp.array(_make_c2(n))
        result = apply_diagonal_correction(c2, width=1, method="mask")
        diag = np.diag(np.asarray(result))
        assert np.all(np.isnan(diag))

    def test_invalid_method_raises(self) -> None:
        """Unknown method string raises ValueError."""
        c2 = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="method"):
            apply_diagonal_correction(c2, method="unknown_method")

    def test_invalid_width_raises(self) -> None:
        """width < 1 raises ValueError."""
        c2 = jnp.ones((4, 4))
        with pytest.raises(ValueError, match="width"):
            apply_diagonal_correction(c2, width=0)


# ---------------------------------------------------------------------------
# Correction methods (estimators)
# ---------------------------------------------------------------------------


class TestCorrectionMethods:
    """Test that each correction method produces valid finite output."""

    @pytest.mark.parametrize("method", ["interpolate", "mask", "mirror"])
    def test_standard_methods_valid_output(self, method: str) -> None:
        """Standard method produces output with no unexpected NaN/Inf."""
        n = 12
        c2 = jnp.array(_make_c2(n))
        result = np.asarray(apply_diagonal_correction(c2, width=1, method=method))

        if method == "mask":
            # Diagonal is NaN by design; off-diagonal must be finite
            off_mask = ~np.eye(n, dtype=bool)
            assert np.all(np.isfinite(result[off_mask]))
        else:
            assert np.all(np.isfinite(result))

    def test_mirror_symmetry_off_diagonal(self) -> None:
        """Mirror method applied to a symmetric matrix is a no-op off-diagonal."""
        n = 6
        sym = _make_c2(n)
        # Create a perfectly symmetric matrix (no diagonal artifact for off-diag)
        sym_jax = jnp.array((sym + sym.T) / 2.0)
        result = np.asarray(apply_diagonal_correction(sym_jax, width=1, method="mirror"))
        # Off-diagonal should be unchanged (mirror of symmetric is itself)
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(result[mask], np.asarray(sym_jax)[mask])


# ---------------------------------------------------------------------------
# Batch correction
# ---------------------------------------------------------------------------


class TestBatchCorrection:
    """Tests for apply_diagonal_correction_batch."""

    def test_batch_shape_3d(self) -> None:
        """Batch input (k, N, N) returns same shape."""
        k, n = 4, 8
        c2_batch = jnp.stack([jnp.array(_make_c2(n)) for _ in range(k)])
        result = apply_diagonal_correction_batch(c2_batch, width=1, method="interpolate")
        assert result.shape == (k, n, n)

    def test_batch_2d_delegates_to_single(self) -> None:
        """2D input is treated as a single matrix."""
        n = 6
        c2 = jnp.array(_make_c2(n))
        result = apply_diagonal_correction_batch(c2, width=1, method="interpolate")
        expected = apply_diagonal_correction(c2, width=1, method="interpolate")
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))

    def test_batch_invalid_ndim_raises(self) -> None:
        """1D input raises ValueError."""
        with pytest.raises(ValueError, match="2-D or 3-D"):
            apply_diagonal_correction_batch(jnp.ones(8))

    def test_batch_numpy_path(self) -> None:
        """NumPy batch input is processed correctly."""
        k, n = 3, 6
        c2_batch = np.stack([_make_c2(n) for _ in range(k)])
        result = apply_diagonal_correction_batch(c2_batch, width=1, method="interpolate")
        assert result.shape == (k, n, n)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# estimate_diagonal_excess
# ---------------------------------------------------------------------------


class TestEstimateDiagonalExcess:
    """Tests for estimate_diagonal_excess."""

    def test_excess_keys_present(self) -> None:
        """Output dictionary has all expected keys."""
        expected_keys = {
            "mean_diagonal",
            "mean_off_diagonal",
            "mean_excess",
            "std_diagonal",
            "std_off_diagonal",
            "std_ratio",
            "excess_sigma",
        }
        c2 = jnp.array(_make_c2(10))
        report = estimate_diagonal_excess(c2, width=1)
        assert set(report.keys()) == expected_keys

    def test_excess_positive_for_elevated_diagonal(self) -> None:
        """mean_excess > 0 when diagonal is inflated above off-diagonal."""
        n = 20
        c2 = jnp.array(_make_c2(n))  # diagonal + 2.0 above base
        report = estimate_diagonal_excess(c2, width=1)
        mean_excess = float(report["mean_excess"])
        assert mean_excess > 0.0

    def test_excess_near_zero_for_uniform_matrix(self) -> None:
        """Constant matrix has zero excess."""
        n = 8
        c2 = jnp.ones((n, n))
        report = estimate_diagonal_excess(c2, width=1)
        assert float(report["mean_excess"]) == pytest.approx(0.0, abs=1e-6)

    def test_excess_invalid_width_raises(self) -> None:
        """width < 1 raises ValueError."""
        c2 = jnp.ones((5, 5))
        with pytest.raises(ValueError, match="width"):
            estimate_diagonal_excess(c2, width=0)


# ---------------------------------------------------------------------------
# compute_weights_excluding_diagonal
# ---------------------------------------------------------------------------


class TestComputeWeightsExcludingDiagonal:
    """Tests for compute_weights_excluding_diagonal."""

    def test_weights_zero_on_diagonal(self) -> None:
        """Diagonal elements are zero in the weight array."""
        n = 8
        weights = np.asarray(compute_weights_excluding_diagonal((n, n), width=1))
        diag = np.diag(weights)
        np.testing.assert_array_equal(diag, np.zeros(n))

    def test_weights_one_off_diagonal(self) -> None:
        """Off-diagonal elements are 1.0."""
        n = 6
        weights = np.asarray(compute_weights_excluding_diagonal((n, n), width=1))
        off_mask = ~np.eye(n, dtype=bool)
        np.testing.assert_array_equal(weights[off_mask], np.ones(n * (n - 1)))

    def test_weights_shape(self) -> None:
        """Output shape matches requested shape."""
        shape = (7, 7)
        weights = compute_weights_excluding_diagonal(shape)
        assert weights.shape == shape

    def test_weights_invalid_width_raises(self) -> None:
        """width < 1 raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            compute_weights_excluding_diagonal((5, 5), width=0)


# ---------------------------------------------------------------------------
# Backend/method discovery
# ---------------------------------------------------------------------------


def test_get_diagonal_correction_methods() -> None:
    """All expected correction method strings are reported."""
    methods = get_diagonal_correction_methods()
    for expected in ("interpolate", "mask", "mirror", "statistical"):
        assert expected in methods


def test_get_available_backends() -> None:
    """numpy is always in the available backends list."""
    backends = get_available_backends()
    assert "numpy" in backends
