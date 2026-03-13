"""Unit tests for heterodyne.optimization.nlsq.validation.result_validator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from heterodyne.optimization.nlsq.validation.result_validator import (
    ValidationReport,
    check_bound_saturation,
    check_chi_squared,
    check_covariance_health,
    check_uncertainty_ratios,
    validate_result,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for NLSQResult and ParameterRegistry
# ---------------------------------------------------------------------------


def _make_result(
    *,
    parameters: np.ndarray | None = None,
    parameter_names: list[str] | None = None,
    success: bool = True,
    message: str = "converged",
    covariance: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
    reduced_chi_squared: float | None = 1.0,
    residuals: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    final_cost: float | None = None,
) -> Any:
    """Build a minimal NLSQResult-like object via SimpleNamespace."""
    if parameters is None:
        parameters = np.array([1.0, 2.0, 3.0])
    if parameter_names is None:
        parameter_names = [f"p{i}" for i in range(len(parameters))]
    return SimpleNamespace(
        parameters=parameters,
        parameter_names=parameter_names,
        success=success,
        message=message,
        covariance=covariance,
        uncertainties=uncertainties,
        reduced_chi_squared=reduced_chi_squared,
        residuals=residuals,
        jacobian=jacobian,
        final_cost=final_cost,
    )


def _make_registry(entries: dict[str, tuple[float, float]]) -> Any:
    """Build a minimal ParameterRegistry-like mapping.

    ``entries`` maps parameter name -> (min_bound, max_bound).
    """
    store: dict[str, SimpleNamespace] = {}
    for name, (lo, hi) in entries.items():
        store[name] = SimpleNamespace(min_bound=lo, max_bound=hi)

    class _FakeRegistry:
        def __getitem__(self, key: str) -> SimpleNamespace:
            return store[key]

    return _FakeRegistry()


# ===================================================================
# ValidationReport
# ===================================================================


class TestValidationReport:
    def test_defaults(self) -> None:
        report = ValidationReport()
        assert report.passed is True
        assert report.warnings == []
        assert report.errors == []
        assert report.metrics == {}

    def test_summary_pass(self) -> None:
        report = ValidationReport()
        assert "PASS" in report.summary()

    def test_summary_fail(self) -> None:
        report = ValidationReport(passed=False, errors=["bad fit"])
        s = report.summary()
        assert "FAIL" in s
        assert "[X]" in s

    def test_summary_warnings(self) -> None:
        report = ValidationReport(warnings=["mediocre"])
        assert "[!]" in report.summary()

    def test_summary_metrics(self) -> None:
        report = ValidationReport(metrics={"reduced_chi2": 1.234})
        s = report.summary()
        assert "reduced_chi2" in s
        assert "1.234" in s


# ===================================================================
# check_bound_saturation
# ===================================================================


class TestCheckBoundSaturation:
    def test_no_saturation(self) -> None:
        result = _make_result(
            parameters=np.array([5.0, 50.0]),
            parameter_names=["a", "b"],
        )
        registry = _make_registry({"a": (0.0, 10.0), "b": (0.0, 100.0)})
        msgs = check_bound_saturation(result, registry)
        assert msgs == []

    def test_lower_bound_saturation(self) -> None:
        result = _make_result(
            parameters=np.array([0.05]),
            parameter_names=["a"],
        )
        registry = _make_registry({"a": (0.0, 10.0)})
        msgs = check_bound_saturation(result, registry)
        assert len(msgs) == 1
        assert "lower bound" in msgs[0]

    def test_upper_bound_saturation(self) -> None:
        result = _make_result(
            parameters=np.array([9.95]),
            parameter_names=["a"],
        )
        registry = _make_registry({"a": (0.0, 10.0)})
        msgs = check_bound_saturation(result, registry)
        assert len(msgs) == 1
        assert "upper bound" in msgs[0]

    def test_custom_tolerance(self) -> None:
        # Value at 5% from lower bound, with tolerance=0.1 -> saturated
        result = _make_result(
            parameters=np.array([0.5]),
            parameter_names=["a"],
        )
        registry = _make_registry({"a": (0.0, 10.0)})
        assert check_bound_saturation(result, registry, tolerance=0.1) != []
        # Same value with tighter tolerance -> not saturated
        assert check_bound_saturation(result, registry, tolerance=0.01) == []

    def test_unknown_parameter_skipped(self) -> None:
        result = _make_result(
            parameters=np.array([0.0]),
            parameter_names=["unknown"],
        )
        registry = _make_registry({})  # empty
        msgs = check_bound_saturation(result, registry)
        assert msgs == []

    def test_degenerate_bounds_skipped(self) -> None:
        """Parameters with span=0 (fixed) should be silently skipped."""
        result = _make_result(
            parameters=np.array([5.0]),
            parameter_names=["a"],
        )
        registry = _make_registry({"a": (5.0, 5.0)})
        msgs = check_bound_saturation(result, registry)
        assert msgs == []


# ===================================================================
# check_covariance_health
# ===================================================================


class TestCheckCovarianceHealth:
    def test_no_covariance(self) -> None:
        result = _make_result(covariance=None)
        msgs = check_covariance_health(result)
        assert len(msgs) == 1
        assert "unavailable" in msgs[0]

    def test_healthy_covariance(self) -> None:
        cov = np.diag([1.0, 2.0, 3.0])
        result = _make_result(covariance=cov)
        msgs = check_covariance_health(result)
        assert msgs == []

    def test_nan_entries(self) -> None:
        cov = np.diag([1.0, np.nan, 3.0])
        result = _make_result(covariance=cov)
        msgs = check_covariance_health(result)
        assert len(msgs) == 1
        assert "NaN or Inf" in msgs[0]

    def test_inf_entries(self) -> None:
        cov = np.diag([1.0, np.inf, 3.0])
        result = _make_result(covariance=cov)
        msgs = check_covariance_health(result)
        assert any("NaN or Inf" in m for m in msgs)

    def test_negative_variance(self) -> None:
        cov = np.diag([1.0, -0.5, 3.0])
        result = _make_result(
            covariance=cov,
            parameter_names=["a", "b", "c"],
        )
        msgs = check_covariance_health(result)
        assert any("Negative variance" in m for m in msgs)
        assert "b" in msgs[0]

    def test_ill_conditioned(self) -> None:
        cov = np.diag([1e12, 1.0])
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            covariance=cov,
        )
        msgs = check_covariance_health(result)
        assert any("Ill-conditioned" in m for m in msgs)

    def test_singular_matrix(self) -> None:
        # A matrix with a zero singular value
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            covariance=cov,
        )
        msgs = check_covariance_health(result)
        assert any("singular" in m for m in msgs)


# ===================================================================
# check_chi_squared
# ===================================================================


class TestCheckChiSquared:
    def test_none_chi2(self) -> None:
        result = _make_result(reduced_chi_squared=None)
        msgs = check_chi_squared(result)
        assert any("not available" in m for m in msgs)

    def test_non_finite_chi2(self) -> None:
        result = _make_result(reduced_chi_squared=np.inf)
        msgs = check_chi_squared(result)
        assert any("non-finite" in m for m in msgs)

    def test_nan_chi2(self) -> None:
        result = _make_result(reduced_chi_squared=np.nan)
        msgs = check_chi_squared(result)
        assert any("non-finite" in m for m in msgs)

    def test_good_fit(self) -> None:
        """chi2_red near 1 should produce no messages."""
        result = _make_result(reduced_chi_squared=1.05)
        assert check_chi_squared(result) == []

    def test_very_poor_fit(self) -> None:
        result = _make_result(reduced_chi_squared=15.0)
        msgs = check_chi_squared(result)
        assert any("Very poor fit" in m for m in msgs)

    def test_mediocre_fit(self) -> None:
        """chi2_red between 2 and max_threshold triggers mediocre warning."""
        result = _make_result(reduced_chi_squared=5.0)
        msgs = check_chi_squared(result)
        assert any("Mediocre fit" in m for m in msgs)

    def test_chi2_exactly_2_triggers_mediocre(self) -> None:
        """Boundary: chi2 > 2.0 triggers mediocre."""
        result = _make_result(reduced_chi_squared=2.001)
        msgs = check_chi_squared(result)
        assert any("Mediocre" in m for m in msgs)

    def test_suspected_overfit(self) -> None:
        result = _make_result(reduced_chi_squared=0.005)
        msgs = check_chi_squared(result)
        assert any("Suspected over-fit" in m for m in msgs)

    def test_possible_overfit(self) -> None:
        """chi2_red between min_threshold and 0.5 triggers possible over-fit."""
        result = _make_result(reduced_chi_squared=0.3)
        msgs = check_chi_squared(result)
        assert any("Possible over-fit" in m for m in msgs)

    def test_custom_thresholds(self) -> None:
        result = _make_result(reduced_chi_squared=6.0)
        # Default max=10 -> mediocre; lower max=5 -> very poor
        msgs_default = check_chi_squared(result)
        msgs_strict = check_chi_squared(result, max_reduced_chi2=5.0)
        assert any("Mediocre" in m for m in msgs_default)
        assert any("Very poor" in m for m in msgs_strict)

    def test_custom_min_threshold(self) -> None:
        result = _make_result(reduced_chi_squared=0.3)
        # Default min=0.01 -> possible overfit; higher min=0.5 -> suspected overfit
        msgs = check_chi_squared(result, min_reduced_chi2=0.5)
        assert any("Suspected over-fit" in m for m in msgs)


# ===================================================================
# check_uncertainty_ratios
# ===================================================================


class TestCheckUncertaintyRatios:
    def test_no_uncertainties(self) -> None:
        result = _make_result(uncertainties=None)
        assert check_uncertainty_ratios(result) == []

    def test_well_constrained(self) -> None:
        result = _make_result(
            parameters=np.array([10.0, 20.0]),
            parameter_names=["a", "b"],
            uncertainties=np.array([0.5, 1.0]),
        )
        assert check_uncertainty_ratios(result) == []

    def test_poorly_constrained(self) -> None:
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            uncertainties=np.array([5.0, 0.1]),  # a is 500% relative
        )
        msgs = check_uncertainty_ratios(result)
        assert len(msgs) == 1
        assert "'a'" in msgs[0]
        assert "500%" in msgs[0]

    def test_zero_value_skipped(self) -> None:
        result = _make_result(
            parameters=np.array([0.0, 2.0]),
            parameter_names=["a", "b"],
            uncertainties=np.array([1.0, 0.1]),
        )
        assert check_uncertainty_ratios(result) == []

    def test_non_finite_skipped(self) -> None:
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            uncertainties=np.array([np.nan, 0.1]),
        )
        assert check_uncertainty_ratios(result) == []

    def test_custom_threshold(self) -> None:
        result = _make_result(
            parameters=np.array([10.0]),
            parameter_names=["a"],
            uncertainties=np.array([3.0]),  # 30% relative
        )
        # Default threshold 1.0 -> ok
        assert check_uncertainty_ratios(result, max_relative_uncertainty=1.0) == []
        # Stricter threshold 0.2 -> flagged
        msgs = check_uncertainty_ratios(result, max_relative_uncertainty=0.2)
        assert len(msgs) == 1


# ===================================================================
# validate_result (top-level orchestrator)
# ===================================================================


class TestValidateResult:
    def test_successful_clean_result(self) -> None:
        cov = np.diag([0.01, 0.02, 0.03])
        result = _make_result(
            parameters=np.array([5.0, 50.0, 5.0]),
            parameter_names=["D0_ref", "D0_sample", "v0"],
            covariance=cov,
            uncertainties=np.sqrt(np.diag(cov)),
            reduced_chi_squared=1.1,
        )
        report = validate_result(result)
        assert report.passed is True
        assert report.errors == []
        assert "reduced_chi2" in report.metrics

    def test_unconverged_result_fails_fast(self) -> None:
        result = _make_result(success=False, message="did not converge")
        report = validate_result(result)
        assert report.passed is False
        assert any("did not converge" in e for e in report.errors)

    def test_nan_parameters_fails_fast(self) -> None:
        result = _make_result(parameters=np.array([1.0, np.nan, 3.0]))
        report = validate_result(result)
        assert report.passed is False
        assert any("NaN or Inf" in e for e in report.errors)

    def test_very_poor_chi2_is_error(self) -> None:
        result = _make_result(reduced_chi_squared=50.0)
        report = validate_result(result)
        assert report.passed is False
        assert any("Very poor fit" in e for e in report.errors)

    def test_mediocre_chi2_is_warning_not_error(self) -> None:
        result = _make_result(reduced_chi_squared=3.0)
        report = validate_result(result)
        assert report.passed is True
        assert any("Mediocre" in w for w in report.warnings)

    def test_negative_variance_is_error(self) -> None:
        cov = np.diag([1.0, -0.5, 1.0])
        result = _make_result(
            covariance=cov,
            parameter_names=["a", "b", "c"],
        )
        report = validate_result(result)
        assert report.passed is False
        assert any("Negative variance" in e for e in report.errors)

    def test_singular_covariance_is_error(self) -> None:
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            covariance=cov,
        )
        report = validate_result(result)
        assert report.passed is False
        assert any("singular" in e for e in report.errors)

    def test_covariance_condition_number_metric(self) -> None:
        cov = np.diag([1.0, 2.0, 3.0])
        result = _make_result(covariance=cov)
        report = validate_result(result)
        assert "covariance_condition_number" in report.metrics
        assert report.metrics["covariance_condition_number"] == pytest.approx(3.0)

    def test_custom_registry(self) -> None:
        result = _make_result(
            parameters=np.array([0.05]),
            parameter_names=["a"],
        )
        registry = _make_registry({"a": (0.0, 10.0)})
        report = validate_result(result, registry=registry)
        assert any("saturated" in w for w in report.warnings)
        assert report.metrics["n_saturated_params"] == 1.0

    def test_uncertainty_warnings_counted(self) -> None:
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            uncertainties=np.array([5.0, 0.01]),
        )
        report = validate_result(result)
        assert report.metrics["n_poorly_constrained_params"] == 1.0
