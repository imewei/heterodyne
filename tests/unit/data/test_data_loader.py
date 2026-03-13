"""Unit tests for the XPCS data loading and validation API.

Tests are self-contained — no real HDF5 files required.  Synthetic
NumPy arrays are used throughout to exercise validation logic.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------


def test_loader_import() -> None:
    """The heterodyne.data package imports without error."""
    import heterodyne.data  # noqa: F401


def test_xpcs_loader_import() -> None:
    """XPCSDataLoader and load_xpcs_data are importable from the package."""
    from heterodyne.data import XPCSDataLoader, load_xpcs_data  # noqa: F401

    assert callable(load_xpcs_data)
    assert XPCSDataLoader is not None


def test_validators_import() -> None:
    """All validator functions are importable."""
    from heterodyne.data import (  # noqa: F401
        validate_correlation_shape,
        validate_no_nan,
        validate_q_range,
        validate_time_arrays,
        validate_weights,
    )


def test_xpcs_data_import() -> None:
    """XPCSData dataclass is importable from the loader module."""
    from heterodyne.data.xpcs_loader import XPCSData  # noqa: F401

    assert XPCSData is not None


# ---------------------------------------------------------------------------
# validate_correlation_shape
# ---------------------------------------------------------------------------


class TestValidateCorrelationShape:
    """Tests for validate_correlation_shape."""

    def test_valid_2d_square(self) -> None:
        """Square 2D array returns no errors."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones((8, 8))
        errors = validate_correlation_shape(c2)
        assert errors == []

    def test_valid_3d_batch(self) -> None:
        """3D array with square trailing dimensions returns no errors."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones((4, 10, 10))
        errors = validate_correlation_shape(c2)
        assert errors == []

    def test_non_square_2d_error(self) -> None:
        """Non-square 2D array returns an error message."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones((5, 7))
        errors = validate_correlation_shape(c2)
        assert len(errors) > 0

    def test_1d_array_error(self) -> None:
        """1D array returns an error (wrong dimensionality)."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones(10)
        errors = validate_correlation_shape(c2)
        assert len(errors) > 0

    def test_expected_shape_mismatch(self) -> None:
        """Correct 2D square but wrong expected_shape returns an error."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones((8, 8))
        errors = validate_correlation_shape(c2, expected_shape=(6, 6))
        assert len(errors) > 0

    def test_expected_shape_match(self) -> None:
        """Matching expected_shape returns no errors."""
        from heterodyne.data import validate_correlation_shape

        c2 = np.ones((6, 6))
        errors = validate_correlation_shape(c2, expected_shape=(6, 6))
        assert errors == []


# ---------------------------------------------------------------------------
# validate_time_arrays
# ---------------------------------------------------------------------------


class TestValidateTimeArrays:
    """Tests for validate_time_arrays."""

    def test_valid_monotonic_arrays(self) -> None:
        """Matching, strictly increasing arrays return no errors."""
        from heterodyne.data import validate_time_arrays

        t = np.linspace(0.0, 10.0, 20)
        errors = validate_time_arrays(t, t.copy())
        assert errors == []

    def test_non_monotonic_t1(self) -> None:
        """Non-monotonic t1 returns an error."""
        from heterodyne.data import validate_time_arrays

        t_bad = np.array([1.0, 0.5, 2.0])
        t_good = np.array([1.0, 2.0, 3.0])
        errors = validate_time_arrays(t_bad, t_good)
        assert any("t1" in e for e in errors)

    def test_length_mismatch(self) -> None:
        """Different-length arrays return an error."""
        from heterodyne.data import validate_time_arrays

        t1 = np.linspace(0, 1, 5)
        t2 = np.linspace(0, 1, 8)
        errors = validate_time_arrays(t1, t2)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# validate_q_range
# ---------------------------------------------------------------------------


class TestValidateQRange:
    """Tests for validate_q_range."""

    def test_valid_q_values(self) -> None:
        """q values within range return no errors."""
        from heterodyne.data import validate_q_range

        q = np.array([0.005, 0.010, 0.015])
        errors = validate_q_range(q, q_min=0.001, q_max=0.1)
        assert errors == []

    def test_q_below_min(self) -> None:
        """q below q_min returns an error."""
        from heterodyne.data import validate_q_range

        q = np.array([0.0001, 0.01])
        errors = validate_q_range(q, q_min=0.001, q_max=0.1)
        assert len(errors) > 0

    def test_q_above_max(self) -> None:
        """q above q_max returns an error."""
        from heterodyne.data import validate_q_range

        q = np.array([0.01, 0.5])
        errors = validate_q_range(q, q_min=0.001, q_max=0.1)
        assert len(errors) > 0

    def test_invalid_range_specification(self) -> None:
        """q_min > q_max is itself an error."""
        from heterodyne.data import validate_q_range

        q = np.array([0.01])
        errors = validate_q_range(q, q_min=1.0, q_max=0.01)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# validate_no_nan
# ---------------------------------------------------------------------------


class TestValidateNoNan:
    """Tests for validate_no_nan."""

    def test_valid_no_nan(self) -> None:
        """Finite array returns no errors."""
        from heterodyne.data import validate_no_nan

        arr = np.ones((5, 5))
        errors = validate_no_nan(arr, name="test_array")
        assert errors == []

    def test_nan_present(self) -> None:
        """Array with NaN returns an error."""
        from heterodyne.data import validate_no_nan

        arr = np.ones((4, 4))
        arr[1, 2] = np.nan
        errors = validate_no_nan(arr, name="c2")
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# validate_weights
# ---------------------------------------------------------------------------


class TestValidateWeights:
    """Tests for validate_weights."""

    def test_valid_weights(self) -> None:
        """Non-negative weights with correct shape return no errors."""
        from heterodyne.data import validate_weights

        weights = np.ones((6, 6))
        errors = validate_weights(weights, data_shape=(6, 6))
        assert errors == []

    def test_wrong_shape(self) -> None:
        """Weights with wrong shape return an error."""
        from heterodyne.data import validate_weights

        weights = np.ones((5, 5))
        errors = validate_weights(weights, data_shape=(6, 6))
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# XPCSData properties
# ---------------------------------------------------------------------------


class TestXPCSDataProperties:
    """Tests for XPCSData dataclass computed properties."""

    def _make_xpcs_data(self, n: int = 10) -> object:
        from heterodyne.data.xpcs_loader import XPCSData

        t = np.linspace(0.0, 1.0, n)
        c2 = np.ones((n, n))
        return XPCSData(c2=c2, t1=t, t2=t)

    def test_shape_property(self) -> None:
        """shape property returns the c2 array shape."""
        data = self._make_xpcs_data(8)
        assert data.shape == (8, 8)  # type: ignore[union-attr]

    def test_n_times_property(self) -> None:
        """n_times equals number of time points for 2D c2."""
        data = self._make_xpcs_data(12)
        assert data.n_times == 12  # type: ignore[union-attr]

    def test_has_multi_phi_false_for_2d(self) -> None:
        """2D c2 does not have multiple phi angles."""
        data = self._make_xpcs_data(6)
        assert data.has_multi_phi is False  # type: ignore[union-attr]

    def test_has_multi_q_false_when_none(self) -> None:
        """has_multi_q is False when q_values is not set."""
        data = self._make_xpcs_data(6)
        assert data.has_multi_q is False  # type: ignore[union-attr]
