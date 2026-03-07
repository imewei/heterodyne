"""Convergence quality assessment for NLSQ results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


class ConvergenceValidator:
    """Assesses convergence quality of NLSQ optimization.

    Checks:
    - Whether the optimizer terminated normally
    - Cost function decrease sufficiency
    - Gradient norm at solution
    - Parameter change rate in final iterations
    """

    def validate(
        self,
        result: NLSQResult,
        config: NLSQConfig | None = None,
    ) -> ValidationReport:
        """Assess convergence quality.

        Args:
            result: NLSQ result to assess
            config: Optional config for threshold access

        Returns:
            ValidationReport with convergence issues
        """
        report = ValidationReport()

        if not result.success:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Optimizer did not converge: {result.convergence_reason}",
                    "convergence",
                )
            )
            report.is_valid = False
            return report

        # Check if hit iteration limit
        if config is not None and result.n_iterations >= config.max_iterations:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Hit iteration limit ({result.n_iterations}/{config.max_iterations}). "
                    f"Solution may not be fully converged.",
                    "max_iterations",
                    float(result.n_iterations),
                )
            )

        # Check residual quality
        if result.residuals is not None:
            residuals = result.residuals
            max_residual = float(np.max(np.abs(residuals)))
            rms_residual = float(np.sqrt(np.mean(residuals**2)))

            if max_residual > 100 * rms_residual:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Outlier residuals detected: max={max_residual:.2e}, "
                        f"RMS={rms_residual:.2e} (ratio={max_residual/rms_residual:.0f}x)",
                        "residual_outliers",
                        max_residual / rms_residual,
                    )
                )

        # Check Jacobian condition number
        if result.jacobian is not None:
            try:
                cond = float(np.linalg.cond(result.jacobian))
                if cond > 1e12:
                    report.issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            f"Ill-conditioned Jacobian (cond={cond:.2e}). "
                            f"Parameter uncertainties may be unreliable.",
                            "jacobian_condition",
                            cond,
                        )
                    )
                elif cond > 1e8:
                    report.issues.append(
                        ValidationIssue(
                            ValidationSeverity.INFO,
                            f"Jacobian condition number: {cond:.2e}",
                            "jacobian_condition",
                            cond,
                        )
                    )
            except np.linalg.LinAlgError:
                report.issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        "Could not compute Jacobian condition number",
                        "jacobian_condition",
                    )
                )

        if report.errors:
            report.is_valid = False

        return report
