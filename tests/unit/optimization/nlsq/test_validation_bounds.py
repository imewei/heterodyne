"""Tests for heterodyne.optimization.nlsq.validation.bounds."""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.validation.bounds import BoundsValidator
from heterodyne.optimization.nlsq.validation.result import ValidationSeverity


def _make_result(
    names: list[str],
    values: list[float],
) -> NLSQResult:
    """Create a minimal NLSQResult for validation testing."""
    return NLSQResult(
        parameters=np.array(values),
        parameter_names=names,
        success=True,
        message="ok",
    )


class TestBoundsValidator:
    """Tests for BoundsValidator."""

    def test_values_within_bounds_no_issues(self) -> None:
        """Parameters well within bounds should produce no errors or warnings."""
        # Use D0_ref which has bounds [0, 1e6] and default 1e4
        result = _make_result(["D0_ref"], [1e4])
        validator = BoundsValidator()
        report = validator.validate(result)

        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_value_outside_bounds_produces_error(self) -> None:
        """Parameter outside bounds should produce an ERROR."""
        # D0_ref has min_bound=100 — a negative value is outside
        result = _make_result(["D0_ref"], [-1.0])
        validator = BoundsValidator()
        report = validator.validate(result)

        assert report.is_valid is False
        assert len(report.errors) == 1
        assert report.errors[0].severity == ValidationSeverity.ERROR
        assert "outside bounds" in report.errors[0].message

    def test_value_above_upper_bound_produces_error(self) -> None:
        """Parameter above upper bound should produce an ERROR."""
        # D0_ref max_bound = 1e6
        result = _make_result(["D0_ref"], [2e6])
        validator = BoundsValidator()
        report = validator.validate(result)

        assert report.is_valid is False
        assert len(report.errors) == 1

    def test_value_at_lower_edge_produces_warning(self) -> None:
        """Parameter very close to lower bound should produce a WARNING."""
        # D0_ref bounds [100, 1e6], edge_fraction=0.01 means within 1% of range
        # 1% of (1e6-100) ~ 1e4, so value just above 100 is at edge
        result = _make_result(["D0_ref"], [101.0])
        validator = BoundsValidator(edge_fraction=0.01)
        report = validator.validate(result)

        assert report.is_valid is True
        warnings = report.warnings
        assert len(warnings) >= 1
        assert any("lower bound edge" in w.message for w in warnings)

    def test_value_at_upper_edge_produces_warning(self) -> None:
        """Parameter very close to upper bound should produce a WARNING."""
        # D0_ref bounds [0, 1e6], value near max
        result = _make_result(["D0_ref"], [999999.0])
        validator = BoundsValidator(edge_fraction=0.01)
        report = validator.validate(result)

        warnings = report.warnings
        assert len(warnings) >= 1
        assert any("upper bound edge" in w.message for w in warnings)

    def test_unknown_parameter_skipped(self) -> None:
        """Parameters not in registry should be silently skipped."""
        result = _make_result(["nonexistent_param"], [42.0])
        validator = BoundsValidator()
        report = validator.validate(result)

        assert report.is_valid is True
        assert len(report.issues) == 0

    def test_multiple_parameters(self) -> None:
        """Test with multiple parameters, some valid, some not."""
        result = _make_result(
            ["D0_ref", "alpha_ref"],
            [1e4, -999.0],  # D0_ref OK, alpha_ref outside bounds
        )
        validator = BoundsValidator()
        report = validator.validate(result)

        # alpha_ref should be flagged
        assert report.is_valid is False

    def test_custom_edge_fraction(self) -> None:
        """A larger edge fraction should flag more values."""
        # With edge_fraction=0.5, any value in the bottom/top 50% of range
        # is flagged as edge. D0_ref bounds [100, 1e6].
        # (1e4 - 100) / (1e6 - 100) ~ 0.01, well within 50%
        result = _make_result(["D0_ref"], [1e4])
        validator = BoundsValidator(edge_fraction=0.5)
        report = validator.validate(result)

        assert any("lower bound edge" in w.message for w in report.warnings)

    def test_exact_lower_bound_is_not_outside(self) -> None:
        """Value exactly at lower bound should not be an ERROR (just possibly edge)."""
        # D0_ref min_bound = 100
        result = _make_result(["D0_ref"], [100.0])
        validator = BoundsValidator()
        report = validator.validate(result)

        # Should not have ERROR for being outside bounds
        errors = [
            i
            for i in report.issues
            if i.severity == ValidationSeverity.ERROR and "outside" in i.message
        ]
        assert len(errors) == 0
