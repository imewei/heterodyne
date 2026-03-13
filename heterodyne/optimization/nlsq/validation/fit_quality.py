"""Post-fit quality assessment for NLSQ results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)

# Scaling parameter base names (derived from registry at import time)
_SCALING_NAMES = frozenset(
    info.name for info in DEFAULT_REGISTRY._parameters.values() if info.is_scaling
)


def classify_fit_quality(reduced_chi_squared: float | None) -> str:
    """Classify fit quality into a 3-level flag.

    Thresholds match the homodyne NLSQWrapper convention:

    - ``"good"``     — reduced chi-squared < 1.5
    - ``"marginal"`` — 1.5 <= reduced chi-squared < 3.0
    - ``"poor"``     — reduced chi-squared >= 3.0 or unavailable

    Args:
        reduced_chi_squared: Reduced chi-squared statistic, or ``None``
            if not computed.

    Returns:
        One of ``"good"``, ``"marginal"``, or ``"poor"``.
    """
    if reduced_chi_squared is None:
        return "poor"
    if reduced_chi_squared < 1.5:
        return "good"
    if reduced_chi_squared < 3.0:
        return "marginal"
    return "poor"


def _is_physical_param(name: str) -> bool:
    """Check if parameter is a physics param (not per-angle scaling)."""
    return not any(name.startswith(f"{s}_") or name == s for s in _SCALING_NAMES)


class FitQualityValidator:
    """Assesses overall fit quality combining chi-squared and bounds checks."""

    def __init__(
        self,
        chi2_warn: float = 10.0,
        chi2_fail: float = 100.0,
        edge_fraction: float = 0.005,
    ) -> None:
        self._chi2_warn = chi2_warn
        self._chi2_fail = chi2_fail
        self._edge_fraction = edge_fraction

    def validate(self, result: NLSQResult) -> ValidationReport:
        """Run fit quality checks.

        Args:
            result: Completed NLSQ result.

        Returns:
            ValidationReport with quality assessment.
        """
        report = ValidationReport()
        self._check_chi_squared(result, report)
        self._check_bounds_proximity(result, report)

        if report.errors:
            report.is_valid = False
        return report

    def _check_chi_squared(
        self,
        result: NLSQResult,
        report: ValidationReport,
    ) -> None:
        chi2 = result.reduced_chi_squared
        if chi2 is None:
            return

        if chi2 > self._chi2_fail:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Very poor fit: reduced chi2 = {chi2:.2f} > {self._chi2_fail}",
                    "chi2_red",
                    chi2,
                )
            )
        elif chi2 > self._chi2_warn:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Mediocre fit: reduced chi2 = {chi2:.2f} > {self._chi2_warn}",
                    "chi2_red",
                    chi2,
                )
            )

    def _check_bounds_proximity(
        self,
        result: NLSQResult,
        report: ValidationReport,
    ) -> None:
        for name, value in zip(
            result.parameter_names,
            result.parameters,
            strict=True,
        ):
            if not _is_physical_param(name):
                continue
            try:
                info = DEFAULT_REGISTRY[name]
            except KeyError:
                continue

            span = info.max_bound - info.min_bound
            if span <= 0:
                continue

            frac_lo = (value - info.min_bound) / span
            frac_hi = (info.max_bound - value) / span

            if frac_lo < self._edge_fraction:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"{name} = {value:.4e} near lower bound "
                        f"({frac_lo * 100:.2f}% from min={info.min_bound})",
                        f"bound_edge_{name}",
                        frac_lo,
                    )
                )
            elif frac_hi < self._edge_fraction:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"{name} = {value:.4e} near upper bound "
                        f"({frac_hi * 100:.2f}% from max={info.max_bound})",
                        f"bound_edge_{name}",
                        frac_hi,
                    )
                )
