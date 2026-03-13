"""Tests for NLSQ input validation."""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.validation.input_validator import InputValidator


class TestInputValidator:
    """Tests for InputValidator."""

    def test_valid_inputs_pass(self) -> None:
        """Well-formed inputs pass validation."""
        validator = InputValidator()
        report = validator.validate(
            data=np.ones((10, 10)),
            initial_params=np.array([1.0, 2.0]),
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
        )
        assert report.is_valid

    def test_nan_in_data_fails(self) -> None:
        """NaN in data array fails validation."""
        validator = InputValidator()
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        report = validator.validate(
            data=data,
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
        )
        assert not report.is_valid
        assert any("NaN" in i.message for i in report.errors)

    def test_bounds_inverted_fails(self) -> None:
        """Inverted bounds (lower > upper) fails validation."""
        validator = InputValidator()
        report = validator.validate(
            data=np.ones((5, 5)),
            initial_params=np.array([1.0]),
            bounds=(np.array([10.0]), np.array([0.0])),
        )
        assert not report.is_valid

    def test_params_outside_bounds_fails(self) -> None:
        """Initial params outside bounds fails validation."""
        validator = InputValidator()
        report = validator.validate(
            data=np.ones((5, 5)),
            initial_params=np.array([15.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
        )
        assert not report.is_valid

    def test_empty_data_fails(self) -> None:
        """Empty data array fails validation."""
        validator = InputValidator()
        report = validator.validate(
            data=np.array([]),
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
        )
        assert not report.is_valid

    def test_inf_in_data_fails(self) -> None:
        """Inf in data array fails validation."""
        validator = InputValidator()
        data = np.ones((5, 5))
        data[0, 0] = np.inf
        report = validator.validate(
            data=data,
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
        )
        assert not report.is_valid
