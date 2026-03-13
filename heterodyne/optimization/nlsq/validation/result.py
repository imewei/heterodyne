"""NLSQ result validation with severity-based reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.config import NLSQValidationConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    message: str
    metric_name: str = ""
    metric_value: float | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for an NLSQ result."""

    issues: list[ValidationIssue] = field(default_factory=list)
    is_valid: bool = True

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Validation: {'PASS' if self.is_valid else 'FAIL'}"]
        for issue in self.issues:
            prefix = {"info": "  [i]", "warning": "  [!]", "error": "  [X]"}
            lines.append(f"{prefix[issue.severity.value]} {issue.message}")
        return "\n".join(lines)


class ResultValidator:
    """Validates NLSQ fit quality against configurable thresholds."""

    def __init__(self, config: NLSQValidationConfig | None = None) -> None:
        from heterodyne.optimization.nlsq.config import NLSQValidationConfig

        self._config = config or NLSQValidationConfig()

    def validate(self, result: NLSQResult) -> ValidationReport:
        """Run all validation checks on an NLSQ result.

        Args:
            result: The NLSQ result to validate

        Returns:
            ValidationReport with all issues found
        """
        report = ValidationReport()

        if not result.success:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Optimization failed: {result.message}",
                    "success",
                )
            )
            report.is_valid = False
            return report

        self._check_chi_squared(result, report)
        self._check_uncertainties(result, report)
        self._check_correlations(result, report)
        self._check_nan_inf(result, report)

        if report.errors:
            report.is_valid = False

        return report

    def _check_chi_squared(self, result: NLSQResult, report: ValidationReport) -> None:
        """Check reduced chi-squared quality."""
        chi2 = result.reduced_chi_squared
        if chi2 is None:
            return

        cfg = self._config
        if chi2 > cfg.chi2_fail_high:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Very poor fit: χ²_red = {chi2:.2f} > {cfg.chi2_fail_high}",
                    "chi2_red",
                    chi2,
                )
            )
        elif chi2 > cfg.chi2_warn_high:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Mediocre fit: χ²_red = {chi2:.2f} > {cfg.chi2_warn_high}",
                    "chi2_red",
                    chi2,
                )
            )
        elif chi2 < cfg.chi2_warn_low:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Possible overfitting: χ²_red = {chi2:.4f} < {cfg.chi2_warn_low}",
                    "chi2_red",
                    chi2,
                )
            )
        else:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    f"Good fit quality: χ²_red = {chi2:.4f}",
                    "chi2_red",
                    chi2,
                )
            )

    def _check_uncertainties(
        self, result: NLSQResult, report: ValidationReport
    ) -> None:
        """Check parameter uncertainty quality."""
        if result.uncertainties is None:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "No uncertainty estimates available",
                    "uncertainties",
                )
            )
            return

        cfg = self._config
        for name, val, unc in zip(
            result.parameter_names, result.parameters, result.uncertainties, strict=True
        ):
            if val != 0 and abs(unc / val) > cfg.max_relative_uncertainty:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Large relative uncertainty: {name} = {val:.3e} ± {unc:.3e} "
                        f"({abs(unc / val) * 100:.0f}%)",
                        f"uncertainty_{name}",
                        abs(unc / val),
                    )
                )

    def _check_correlations(self, result: NLSQResult, report: ValidationReport) -> None:
        """Check for highly correlated parameters."""
        corr = result.get_correlation_matrix()
        if corr is None:
            return

        cfg = self._config
        n = len(result.parameter_names)
        for i in range(n):
            for j in range(i + 1, n):
                r = abs(corr[i, j])
                if r > cfg.correlation_warn:
                    report.issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Highly correlated: {result.parameter_names[i]} and "
                            f"{result.parameter_names[j]} (|r| = {r:.3f})",
                            "correlation",
                            r,
                        )
                    )

    def _check_nan_inf(self, result: NLSQResult, report: ValidationReport) -> None:
        """Check for NaN/Inf in results."""
        if np.any(~np.isfinite(result.parameters)):
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "NaN or Inf in fitted parameters",
                    "nan_params",
                )
            )

        if result.uncertainties is not None and np.any(
            ~np.isfinite(result.uncertainties)
        ):
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "NaN or Inf in uncertainty estimates",
                    "nan_uncertainties",
                )
            )
