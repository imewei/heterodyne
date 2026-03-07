"""Pre-fit input validation for NLSQ optimization."""
from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validates NLSQ inputs before optimization runs.

    Checks: data shape/finiteness, bounds consistency, initial params
    within bounds.
    """

    def validate(
        self,
        data: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> ValidationReport:
        """Run all input validations.

        Args:
            data: Experimental data array (1D or 2D).
            initial_params: Starting parameter values.
            bounds: (lower, upper) bound arrays.

        Returns:
            ValidationReport with any issues found.
        """
        report = ValidationReport()
        self._check_data(data, report)
        self._check_bounds(bounds, report)
        self._check_initial_params(initial_params, bounds, report)

        if report.errors:
            report.is_valid = False
        return report

    def _check_data(self, data: np.ndarray, report: ValidationReport) -> None:
        if data.size == 0:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "Empty data array",
                "data_empty",
            ))
            return

        n_nan = int(np.sum(np.isnan(data)))
        if n_nan > 0:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"NaN values in data ({n_nan} elements)",
                "data_nan", float(n_nan),
            ))

        n_inf = int(np.sum(np.isinf(data)))
        if n_inf > 0:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Inf values in data ({n_inf} elements)",
                "data_inf", float(n_inf),
            ))

    def _check_bounds(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        report: ValidationReport,
    ) -> None:
        lower, upper = bounds
        if lower.shape != upper.shape:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Bounds shape mismatch: lower={lower.shape}, upper={upper.shape}",
                "bounds_shape",
            ))
            return

        inverted = np.where(lower > upper)[0]
        if len(inverted) > 0:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Inverted bounds at indices {inverted.tolist()}: lower > upper",
                "bounds_inverted", float(len(inverted)),
            ))

    def _check_initial_params(
        self,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        report: ValidationReport,
    ) -> None:
        lower, upper = bounds
        if params.shape != lower.shape:
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Params shape {params.shape} != bounds shape {lower.shape}",
                "params_shape",
            ))
            return

        below = np.where(params < lower)[0]
        above = np.where(params > upper)[0]
        if len(below) > 0 or len(above) > 0:
            out_of_bounds = sorted(set(below.tolist() + above.tolist()))
            report.issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Initial params outside bounds at indices {out_of_bounds}",
                "params_bounds", float(len(out_of_bounds)),
            ))
