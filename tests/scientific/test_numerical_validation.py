"""Tests for optimization numerical validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.exceptions import BoundsError, NumericalError
from heterodyne.optimization.numerical_validation import (
    safe_compute,
    validate_array,
    validate_parameters,
)


class TestValidateArray:
    """Tests for validate_array."""

    @pytest.mark.unit
    def test_clean_array_passes(self) -> None:
        arr = validate_array([1.0, 2.0, 3.0], name="test")
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_nan_raises(self) -> None:
        with pytest.raises(NumericalError, match="NaN"):
            validate_array([1.0, float("nan"), 3.0], name="data")

    @pytest.mark.unit
    def test_inf_raises(self) -> None:
        with pytest.raises(NumericalError, match="Inf"):
            validate_array([1.0, float("inf"), 3.0], name="data")

    @pytest.mark.unit
    def test_negative_inf_raises(self) -> None:
        with pytest.raises(NumericalError, match="Inf"):
            validate_array([float("-inf")], name="x")

    @pytest.mark.unit
    def test_returns_float64(self) -> None:
        result = validate_array([1, 2, 3], name="int_input")
        assert result.dtype == np.float64

    @pytest.mark.unit
    def test_empty_array_passes(self) -> None:
        result = validate_array([], name="empty")
        assert len(result) == 0


class TestValidateParameters:
    """Tests for validate_parameters."""

    @pytest.mark.unit
    def test_valid_params(self) -> None:
        result = validate_parameters(
            [1.0, 0.5],
            names=["D0", "alpha"],
        )
        np.testing.assert_array_equal(result, [1.0, 0.5])

    @pytest.mark.unit
    def test_nan_param_raises_with_name(self) -> None:
        with pytest.raises(NumericalError, match="D0"):
            validate_parameters(
                [float("nan"), 0.5],
                names=["D0", "alpha"],
            )

    @pytest.mark.unit
    def test_bounds_check_passes(self) -> None:
        result = validate_parameters(
            [500.0, 0.5],
            names=["D0", "alpha"],
            bounds=[(100.0, 1e5), (-2.0, 2.0)],
        )
        np.testing.assert_array_equal(result, [500.0, 0.5])

    @pytest.mark.unit
    def test_bounds_check_fails(self) -> None:
        with pytest.raises(BoundsError, match="D0"):
            validate_parameters(
                [50.0, 0.5],
                names=["D0", "alpha"],
                bounds=[(100.0, 1e5), (-2.0, 2.0)],
            )

    @pytest.mark.unit
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            validate_parameters([1.0, 2.0], names=["a"])

    @pytest.mark.unit
    def test_bounds_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="bounds length"):
            validate_parameters(
                [1.0],
                names=["a"],
                bounds=[(0.0, 1.0), (0.0, 1.0)],
            )


class TestSafeCompute:
    """Tests for safe_compute."""

    @pytest.mark.unit
    def test_clean_result_returned(self) -> None:
        result = safe_compute(np.array, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_nan_result_raises(self) -> None:
        def produce_nan() -> np.ndarray:
            return np.array([float("nan")])

        with pytest.raises(NumericalError, match="NaN/Inf"):
            safe_compute(produce_nan)

    @pytest.mark.unit
    def test_nan_result_with_fallback(self) -> None:
        def produce_nan() -> np.ndarray:
            return np.array([float("nan")])

        fallback = np.array([0.0])
        result = safe_compute(produce_nan, fallback=fallback)
        np.testing.assert_array_equal(result, fallback)

    @pytest.mark.unit
    def test_inf_result_with_fallback(self) -> None:
        def produce_inf() -> np.ndarray:
            return np.array([float("inf")])

        fallback = np.array([0.0])
        result = safe_compute(produce_inf, fallback=fallback)
        np.testing.assert_array_equal(result, fallback)
