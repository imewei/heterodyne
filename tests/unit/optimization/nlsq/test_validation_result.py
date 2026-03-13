"""Tests for heterodyne.optimization.nlsq.validation.result."""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQValidationConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.validation.result import (
    ResultValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

# ---------------------------------------------------------------------------
# ValidationSeverity
# ---------------------------------------------------------------------------


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_values(self) -> None:
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_defaults(self) -> None:
        issue = ValidationIssue(ValidationSeverity.INFO, "test message")
        assert issue.metric_name == ""
        assert issue.metric_value is None

    def test_full_construction(self) -> None:
        issue = ValidationIssue(ValidationSeverity.ERROR, "bad value", "chi2", 99.9)
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "bad value"
        assert issue.metric_name == "chi2"
        assert issue.metric_value == 99.9


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_empty_report(self) -> None:
        report = ValidationReport()
        assert report.is_valid is True
        assert report.issues == []
        assert report.errors == []
        assert report.warnings == []

    def test_errors_property(self) -> None:
        report = ValidationReport(
            issues=[
                ValidationIssue(ValidationSeverity.ERROR, "e1"),
                ValidationIssue(ValidationSeverity.WARNING, "w1"),
                ValidationIssue(ValidationSeverity.ERROR, "e2"),
            ]
        )
        assert len(report.errors) == 2

    def test_warnings_property(self) -> None:
        report = ValidationReport(
            issues=[
                ValidationIssue(ValidationSeverity.WARNING, "w1"),
                ValidationIssue(ValidationSeverity.INFO, "i1"),
                ValidationIssue(ValidationSeverity.WARNING, "w2"),
            ]
        )
        assert len(report.warnings) == 2

    def test_summary_pass(self) -> None:
        report = ValidationReport()
        summary = report.summary()
        assert "PASS" in summary

    def test_summary_fail(self) -> None:
        report = ValidationReport(
            is_valid=False,
            issues=[
                ValidationIssue(ValidationSeverity.ERROR, "something broke"),
            ],
        )
        summary = report.summary()
        assert "FAIL" in summary
        assert "[X]" in summary

    def test_summary_includes_all_severities(self) -> None:
        report = ValidationReport(
            issues=[
                ValidationIssue(ValidationSeverity.INFO, "info msg"),
                ValidationIssue(ValidationSeverity.WARNING, "warn msg"),
                ValidationIssue(ValidationSeverity.ERROR, "err msg"),
            ]
        )
        summary = report.summary()
        assert "[i]" in summary
        assert "[!]" in summary
        assert "[X]" in summary


# ---------------------------------------------------------------------------
# Helper to build NLSQResult
# ---------------------------------------------------------------------------


def _make_result(
    success: bool = True,
    chi2: float | None = 1.0,
    parameters: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    names: list[str] | None = None,
) -> NLSQResult:
    if parameters is None:
        parameters = np.array([1.0, 2.0])
    if names is None:
        names = [f"p{i}" for i in range(len(parameters))]
    return NLSQResult(
        parameters=parameters,
        parameter_names=names,
        success=success,
        message="ok" if success else "failed",
        reduced_chi_squared=chi2,
        uncertainties=uncertainties,
        covariance=covariance,
    )


# ---------------------------------------------------------------------------
# ResultValidator
# ---------------------------------------------------------------------------


class TestResultValidator:
    """Tests for ResultValidator."""

    def test_failed_result_is_invalid(self) -> None:
        result = _make_result(success=False)
        validator = ResultValidator()
        report = validator.validate(result)

        assert report.is_valid is False
        assert len(report.errors) == 1
        assert "failed" in report.errors[0].message.lower()

    def test_failed_returns_early(self) -> None:
        """Failed result should not check chi2 or uncertainties."""
        result = _make_result(success=False, chi2=999.0)
        validator = ResultValidator()
        report = validator.validate(result)

        # Only the failure error, not chi2
        assert len(report.issues) == 1

    def test_good_chi2(self) -> None:
        result = _make_result(chi2=1.0)
        validator = ResultValidator()
        report = validator.validate(result)

        assert report.is_valid is True
        chi2_issues = [i for i in report.issues if i.metric_name == "chi2_red"]
        assert len(chi2_issues) == 1
        assert chi2_issues[0].severity == ValidationSeverity.INFO

    def test_high_chi2_warning(self) -> None:
        config = NLSQValidationConfig(chi2_warn_high=2.0, chi2_fail_high=10.0)
        result = _make_result(chi2=5.0)
        validator = ResultValidator(config)
        report = validator.validate(result)

        chi2_issues = [i for i in report.issues if i.metric_name == "chi2_red"]
        assert any(i.severity == ValidationSeverity.WARNING for i in chi2_issues)

    def test_very_high_chi2_error(self) -> None:
        config = NLSQValidationConfig(chi2_fail_high=10.0)
        result = _make_result(chi2=15.0)
        validator = ResultValidator(config)
        report = validator.validate(result)

        assert report.is_valid is False
        chi2_errors = [i for i in report.errors if i.metric_name == "chi2_red"]
        assert len(chi2_errors) == 1

    def test_low_chi2_overfitting_warning(self) -> None:
        config = NLSQValidationConfig(chi2_warn_low=0.5)
        result = _make_result(chi2=0.1)
        validator = ResultValidator(config)
        report = validator.validate(result)

        chi2_issues = [i for i in report.issues if i.metric_name == "chi2_red"]
        assert any(
            i.severity == ValidationSeverity.WARNING and "overfit" in i.message.lower()
            for i in chi2_issues
        )

    def test_none_chi2_no_issue(self) -> None:
        result = _make_result(chi2=None)
        validator = ResultValidator()
        report = validator.validate(result)

        chi2_issues = [i for i in report.issues if i.metric_name == "chi2_red"]
        assert len(chi2_issues) == 0

    def test_no_uncertainties_warning(self) -> None:
        result = _make_result(uncertainties=None)
        validator = ResultValidator()
        report = validator.validate(result)

        unc_issues = [i for i in report.issues if i.metric_name == "uncertainties"]
        assert len(unc_issues) == 1
        assert unc_issues[0].severity == ValidationSeverity.WARNING

    def test_large_relative_uncertainty_warning(self) -> None:
        config = NLSQValidationConfig(max_relative_uncertainty=0.5)
        params = np.array([1.0, 2.0])
        # uncertainties are 200% of values
        uncertainties = np.array([2.0, 4.0])
        result = _make_result(parameters=params, uncertainties=uncertainties)
        validator = ResultValidator(config)
        report = validator.validate(result)

        unc_warnings = [i for i in report.warnings if "uncertainty" in i.metric_name]
        assert len(unc_warnings) == 2

    def test_zero_param_skips_relative_uncertainty(self) -> None:
        """Parameter value of 0 should skip relative uncertainty check."""
        params = np.array([0.0, 1.0])
        uncertainties = np.array([100.0, 0.01])
        result = _make_result(parameters=params, uncertainties=uncertainties)
        validator = ResultValidator()
        report = validator.validate(result)

        # Only p1 could trigger (but 0.01/1.0 = 1%, which is fine)
        unc_warnings = [i for i in report.warnings if "uncertainty" in i.metric_name]
        assert len(unc_warnings) == 0

    def test_high_correlation_warning(self) -> None:
        config = NLSQValidationConfig(correlation_warn=0.9)
        # Covariance matrix with high correlation
        cov = np.array([[1.0, 0.99], [0.99, 1.0]])
        result = _make_result(covariance=cov)
        validator = ResultValidator(config)
        report = validator.validate(result)

        corr_warnings = [i for i in report.warnings if i.metric_name == "correlation"]
        assert len(corr_warnings) == 1

    def test_low_correlation_no_warning(self) -> None:
        cov = np.array([[1.0, 0.1], [0.1, 1.0]])
        result = _make_result(covariance=cov)
        validator = ResultValidator()
        report = validator.validate(result)

        corr_warnings = [i for i in report.warnings if i.metric_name == "correlation"]
        assert len(corr_warnings) == 0

    def test_no_covariance_skips_correlation(self) -> None:
        result = _make_result(covariance=None)
        validator = ResultValidator()
        report = validator.validate(result)

        corr_issues = [i for i in report.issues if i.metric_name == "correlation"]
        assert len(corr_issues) == 0

    def test_nan_in_parameters_error(self) -> None:
        result = _make_result(parameters=np.array([1.0, float("nan")]))
        validator = ResultValidator()
        report = validator.validate(result)

        assert report.is_valid is False
        nan_errors = [i for i in report.issues if i.metric_name == "nan_params"]
        assert len(nan_errors) == 1

    def test_inf_in_parameters_error(self) -> None:
        result = _make_result(parameters=np.array([1.0, float("inf")]))
        validator = ResultValidator()
        report = validator.validate(result)

        assert report.is_valid is False

    def test_nan_in_uncertainties_warning(self) -> None:
        result = _make_result(
            parameters=np.array([1.0, 2.0]),
            uncertainties=np.array([0.1, float("nan")]),
        )
        validator = ResultValidator()
        report = validator.validate(result)

        nan_warnings = [
            i for i in report.issues if i.metric_name == "nan_uncertainties"
        ]
        assert len(nan_warnings) == 1
        assert nan_warnings[0].severity == ValidationSeverity.WARNING

    def test_default_config_used(self) -> None:
        """Passing None config should use defaults."""
        result = _make_result(chi2=1.0)
        validator = ResultValidator(config=None)
        report = validator.validate(result)
        assert report.is_valid is True
