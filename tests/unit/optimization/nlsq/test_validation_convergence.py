"""Tests for heterodyne.optimization.nlsq.validation.convergence."""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.validation.convergence import ConvergenceValidator
from heterodyne.optimization.nlsq.validation.result import ValidationSeverity


def _make_result(
    success: bool = True,
    convergence_reason: str = "converged",
    residuals: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    n_iterations: int = 10,
) -> NLSQResult:
    """Create a minimal NLSQResult for convergence validation."""
    return NLSQResult(
        parameters=np.array([1.0, 2.0]),
        parameter_names=["a", "b"],
        success=success,
        message="ok" if success else "failed",
        convergence_reason=convergence_reason,
        residuals=residuals,
        jacobian=jacobian,
        n_iterations=n_iterations,
    )


class TestConvergenceValidator:
    """Tests for ConvergenceValidator."""

    def test_successful_clean_convergence(self) -> None:
        """A converged result with no issues passes cleanly."""
        result = _make_result(success=True, residuals=np.array([0.01, -0.02, 0.01]))
        validator = ConvergenceValidator()
        report = validator.validate(result)

        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_failed_convergence_is_error(self) -> None:
        """A non-converged result should produce an ERROR."""
        result = _make_result(success=False, convergence_reason="max_iter")
        validator = ConvergenceValidator()
        report = validator.validate(result)

        assert report.is_valid is False
        assert len(report.errors) == 1
        assert "did not converge" in report.errors[0].message.lower()

    def test_failed_returns_early(self) -> None:
        """When success=False, should return immediately without checking residuals."""
        result = _make_result(
            success=False,
            residuals=np.array([1e10, -1e10]),  # extreme residuals
            jacobian=np.eye(2),
        )
        validator = ConvergenceValidator()
        report = validator.validate(result)

        # Only the convergence error, not residual/jacobian issues
        assert len(report.issues) == 1
        assert report.issues[0].metric_name == "convergence"

    def test_hit_iteration_limit_warning(self) -> None:
        """Hitting max_iterations should produce a WARNING."""
        config = NLSQConfig(max_iterations=100)
        result = _make_result(success=True, n_iterations=100)
        validator = ConvergenceValidator()
        report = validator.validate(result, config=config)

        warnings = report.warnings
        assert len(warnings) >= 1
        assert any("iteration limit" in w.message.lower() for w in warnings)

    def test_below_iteration_limit_no_warning(self) -> None:
        """Iterations below limit should not produce a warning."""
        config = NLSQConfig(max_iterations=100)
        result = _make_result(success=True, n_iterations=50)
        validator = ConvergenceValidator()
        report = validator.validate(result, config=config)

        iteration_warnings = [
            w for w in report.warnings if "iteration" in w.message.lower()
        ]
        assert len(iteration_warnings) == 0

    def test_outlier_residuals_warning(self) -> None:
        """Residuals with max >> RMS should produce a WARNING."""
        # Need max/rms > 100. Use many small values and one large one.
        small = np.full(10000, 1e-6)
        residuals = np.append(small, [1.0])
        rms = float(np.sqrt(np.mean(residuals**2)))
        max_res = float(np.max(np.abs(residuals)))
        assert max_res / rms > 100

        result = _make_result(success=True, residuals=residuals)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        assert any("outlier" in w.message.lower() for w in report.warnings)

    def test_no_outlier_residuals(self) -> None:
        """Uniform residuals should not trigger outlier warning."""
        residuals = np.array([0.1, 0.11, 0.09, 0.1, 0.12])
        result = _make_result(success=True, residuals=residuals)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        outlier_warnings = [
            w for w in report.warnings if "outlier" in w.message.lower()
        ]
        assert len(outlier_warnings) == 0

    def test_ill_conditioned_jacobian_warning(self) -> None:
        """A Jacobian with condition number > 1e12 should produce a WARNING."""
        # Create ill-conditioned matrix
        jac = np.array([[1.0, 0.0], [0.0, 1e-14]])
        result = _make_result(success=True, jacobian=jac)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        condition_warnings = [
            w for w in report.warnings if "condition" in w.message.lower()
        ]
        assert len(condition_warnings) >= 1

    def test_moderate_condition_number_info(self) -> None:
        """Condition number between 1e8 and 1e12 should produce INFO."""
        jac = np.array([[1.0, 0.0], [0.0, 1e-10]])
        result = _make_result(success=True, jacobian=jac)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        info_issues = [
            i
            for i in report.issues
            if i.severity == ValidationSeverity.INFO
            and "condition" in i.message.lower()
        ]
        assert len(info_issues) >= 1

    def test_well_conditioned_jacobian_no_issue(self) -> None:
        """A well-conditioned Jacobian should not trigger issues."""
        jac = np.eye(2)
        result = _make_result(success=True, jacobian=jac)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        condition_issues = [
            i for i in report.issues if "condition" in i.message.lower()
        ]
        assert len(condition_issues) == 0

    def test_no_residuals_no_warning(self) -> None:
        """When residuals are None, no residual checks are done."""
        result = _make_result(success=True, residuals=None)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        residual_issues = [i for i in report.issues if "residual" in i.message.lower()]
        assert len(residual_issues) == 0

    def test_no_jacobian_no_warning(self) -> None:
        """When jacobian is None, no condition number checks are done."""
        result = _make_result(success=True, jacobian=None)
        validator = ConvergenceValidator()
        report = validator.validate(result)

        condition_issues = [
            i for i in report.issues if "condition" in i.message.lower()
        ]
        assert len(condition_issues) == 0

    def test_no_config_skips_iteration_check(self) -> None:
        """Without config, iteration limit check should be skipped."""
        result = _make_result(success=True, n_iterations=99999)
        validator = ConvergenceValidator()
        report = validator.validate(result, config=None)

        iteration_warnings = [
            w for w in report.warnings if "iteration" in w.message.lower()
        ]
        assert len(iteration_warnings) == 0
