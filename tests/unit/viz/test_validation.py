"""Unit tests for heterodyne.viz.validation module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from heterodyne.viz.validation import (
    _QUALITY_COLORS,
    _SEVERITY_COLORS,
    plot_bounds_check,
    plot_quality_report,
    plot_validation_report,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers to build lightweight report objects
# ---------------------------------------------------------------------------


def _make_validation_issue(
    severity_value: str = "warning", message: str = "test issue"
) -> object:
    """Create a mock ValidationIssue with .severity.value and .message."""
    from types import SimpleNamespace

    severity = SimpleNamespace(value=severity_value)
    return SimpleNamespace(severity=severity, message=message)


def _make_validation_report(
    *, is_valid: bool = True, issues: list[object] | None = None
) -> object:
    """Create a mock ValidationReport."""
    from types import SimpleNamespace

    return SimpleNamespace(
        is_valid=is_valid,
        issues=issues if issues is not None else [],
    )


def _make_quality_metric(
    name: str = "metric",
    level_value: str = "good",
    message: str = "ok",
) -> object:
    from types import SimpleNamespace

    level = SimpleNamespace(value=level_value)
    return SimpleNamespace(name=name, level=level, message=message)


def _make_quality_report(
    *,
    metrics: list[object] | None = None,
    overall_level_value: str = "good",
) -> object:
    from types import SimpleNamespace

    overall_level = SimpleNamespace(value=overall_level_value)
    return SimpleNamespace(
        metrics=metrics if metrics is not None else [],
        overall_level=overall_level,
    )


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------


class TestColourMaps:
    def test_severity_colors_keys(self) -> None:
        assert set(_SEVERITY_COLORS.keys()) == {"info", "warning", "error"}

    def test_quality_colors_keys(self) -> None:
        assert set(_QUALITY_COLORS.keys()) == {
            "good",
            "acceptable",
            "warning",
            "critical",
        }


# ---------------------------------------------------------------------------
# plot_validation_report
# ---------------------------------------------------------------------------


class TestPlotValidationReport:
    def test_pass_no_issues(self) -> None:
        report = _make_validation_report(is_valid=True, issues=[])
        ax = plot_validation_report(report)  # type: ignore[arg-type]
        assert ax is not None
        assert "PASS" in ax.get_title()

    def test_fail_no_issues(self) -> None:
        report = _make_validation_report(is_valid=False, issues=[])
        ax = plot_validation_report(report)  # type: ignore[arg-type]
        assert "FAIL" in ax.get_title()

    def test_with_issues(self) -> None:
        issues = [
            _make_validation_issue("info", "Info message"),
            _make_validation_issue("warning", "Warning message"),
            _make_validation_issue("error", "Error message"),
        ]
        report = _make_validation_report(is_valid=False, issues=issues)
        ax = plot_validation_report(report)  # type: ignore[arg-type]
        assert "FAIL" in ax.get_title()
        # Should have 3 horizontal bars
        assert len(ax.patches) == 3

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        report = _make_validation_report(is_valid=True, issues=[])
        ax = plot_validation_report(report, ax=provided_ax)  # type: ignore[arg-type]
        assert ax is provided_ax

    def test_unknown_severity_uses_fallback_colour(self) -> None:
        """Severity values not in _SEVERITY_COLORS should get #999999."""
        issues = [_make_validation_issue("unknown_level", "Strange")]
        report = _make_validation_report(is_valid=True, issues=issues)
        ax = plot_validation_report(report)  # type: ignore[arg-type]
        # Should render without error
        assert len(ax.patches) == 1


# ---------------------------------------------------------------------------
# plot_bounds_check
# ---------------------------------------------------------------------------


class TestPlotBoundsCheck:
    def test_all_in_bounds(self) -> None:
        values = np.array([5.0, 10.0])
        names = ["D0_ref", "alpha_ref"]
        bounds = [(0.0, 100.0), (0.0, 20.0)]
        ax = plot_bounds_check(values, names, bounds)
        assert ax is not None
        assert ax.get_title() == "Parameter Bounds Check"

    def test_out_of_bounds(self) -> None:
        values = np.array([150.0])
        names = ["D0_ref"]
        bounds = [(0.0, 100.0)]
        ax = plot_bounds_check(values, names, bounds)
        assert ax is not None

    def test_near_edge(self) -> None:
        # edge_fraction=0.05 => edge_width = 0.05*100 = 5
        # value=3 is within [0, 5) -> amber
        values = np.array([3.0])
        names = ["D0_ref"]
        bounds = [(0.0, 100.0)]
        ax = plot_bounds_check(values, names, bounds, edge_fraction=0.05)
        assert ax is not None

    def test_custom_edge_fraction(self) -> None:
        values = np.array([50.0])
        names = ["x"]
        bounds = [(0.0, 100.0)]
        ax = plot_bounds_check(values, names, bounds, edge_fraction=0.49)
        # 50 is within [0+49, 100-49] = [49, 51] -> green
        assert ax is not None

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        values = np.array([5.0])
        names = ["x"]
        bounds = [(0.0, 10.0)]
        ax = plot_bounds_check(values, names, bounds, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# plot_quality_report
# ---------------------------------------------------------------------------


class TestPlotQualityReport:
    def test_empty_metrics(self) -> None:
        report = _make_quality_report(metrics=[], overall_level_value="good")
        ax = plot_quality_report(report)  # type: ignore[arg-type]
        assert ax is not None

    def test_with_metrics(self) -> None:
        metrics = [
            _make_quality_metric("SNR", "good", "High SNR"),
            _make_quality_metric("NaN fraction", "warning", "Some NaNs"),
            _make_quality_metric("Outlier count", "critical", "Too many"),
        ]
        report = _make_quality_report(metrics=metrics, overall_level_value="critical")
        ax = plot_quality_report(report)  # type: ignore[arg-type]
        assert "CRITICAL" in ax.get_title()
        assert len(ax.patches) == 3

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        report = _make_quality_report()
        ax = plot_quality_report(report, ax=provided_ax)  # type: ignore[arg-type]
        assert ax is provided_ax

    def test_overall_good(self) -> None:
        metrics = [_make_quality_metric("x", "good", "fine")]
        report = _make_quality_report(metrics=metrics, overall_level_value="good")
        ax = plot_quality_report(report)  # type: ignore[arg-type]
        assert "GOOD" in ax.get_title()
