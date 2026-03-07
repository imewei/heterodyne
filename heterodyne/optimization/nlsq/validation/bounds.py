"""Parameter bounds validation for NLSQ results."""

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


class BoundsValidator:
    """Validates fitted parameters against physical bounds.

    Checks whether fitted parameter values are:
    1. Within registry bounds (hard constraint)
    2. Away from bound edges (soft constraint — parameters at bounds
       suggest the optimizer is constrained)
    """

    def __init__(self, edge_fraction: float = 0.01) -> None:
        """Initialize bounds validator.

        Args:
            edge_fraction: Fraction of range to consider "at edge".
                Parameters within this fraction of a bound are flagged.
        """
        self._edge_fraction = edge_fraction

    def validate(self, result: NLSQResult) -> ValidationReport:
        """Check all parameters against bounds.

        Args:
            result: NLSQ result to validate

        Returns:
            ValidationReport with bounds issues
        """
        report = ValidationReport()

        for name, value in zip(
            result.parameter_names, result.parameters, strict=True
        ):
            try:
                info = DEFAULT_REGISTRY[name]
            except KeyError:
                continue

            lo, hi = info.min_bound, info.max_bound
            span = hi - lo

            if value < lo or value > hi:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"{name} = {value:.4e} outside bounds [{lo}, {hi}]",
                        f"bounds_{name}",
                        value,
                    )
                )
            elif span > 0:
                # Check if at edge of bounds
                frac_from_lo = (value - lo) / span
                frac_from_hi = (hi - value) / span

                if frac_from_lo < self._edge_fraction:
                    report.issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"{name} = {value:.4e} at lower bound edge "
                            f"({frac_from_lo*100:.1f}% from min={lo})",
                            f"edge_{name}",
                            frac_from_lo,
                        )
                    )
                elif frac_from_hi < self._edge_fraction:
                    report.issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"{name} = {value:.4e} at upper bound edge "
                            f"({frac_from_hi*100:.1f}% from max={hi})",
                            f"edge_{name}",
                            frac_from_hi,
                        )
                    )

        if report.errors:
            report.is_valid = False

        return report
