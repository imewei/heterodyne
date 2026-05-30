"""Post-fit result validation for NLSQ optimization results.

Provides functional-style checks for parameter bound saturation, covariance
matrix health, reduced chi-squared ranges, and parameter uncertainty ratios.
Complements the class-based ResultValidator in result.py with a simpler
dataclass-driven API suited for pipeline integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_registry import ParameterRegistry
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Summary of result validation checks.

    Attributes:
        passed: True when no errors were found (warnings do not fail).
        warnings: Non-fatal issues that deserve attention.
        errors: Fatal issues that indicate the result is unreliable.
        metrics: Scalar diagnostic values keyed by metric name.
    """

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the report."""
        lines = [f"Result validation: {'PASS' if self.passed else 'FAIL'}"]
        for msg in self.errors:
            lines.append(f"  [X] {msg}")
        for msg in self.warnings:
            lines.append(f"  [!] {msg}")
        if self.metrics:
            lines.append("  Metrics:")
            for key, val in sorted(self.metrics.items()):
                lines.append(f"    {key}: {val:.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_bound_saturation(
    result: NLSQResult,
    registry: ParameterRegistry,
    tolerance: float = 0.01,
) -> list[str]:
    """Check if any parameters are stuck at their bounds.

    A parameter is considered saturated when its fitted value lies within
    ``tolerance * (max_bound - min_bound)`` of either bound.

    Args:
        result: Fitted NLSQ result.
        registry: Parameter registry providing bound metadata.
        tolerance: Fractional proximity to a bound that triggers a warning.
            Defaults to 0.01 (1% of the parameter range).

    Returns:
        List of warning message strings, one per saturated parameter.
    """
    messages: list[str] = []

    for name, val in zip(result.parameter_names, result.parameters, strict=True):
        try:
            info = registry[name]
        except KeyError:
            continue  # Parameter not in registry; skip silently.

        span = info.max_bound - info.min_bound
        if span <= 0.0:
            continue  # Degenerate bounds; skip.

        margin = tolerance * span
        if val <= info.min_bound + margin:
            messages.append(
                f"Parameter '{name}' saturated at lower bound "
                f"(value={val:.4g}, lower={info.min_bound:.4g})"
            )
        elif val >= info.max_bound - margin:
            messages.append(
                f"Parameter '{name}' saturated at upper bound "
                f"(value={val:.4g}, upper={info.max_bound:.4g})"
            )

    return messages


def check_covariance_health(result: NLSQResult) -> list[str]:
    """Check covariance matrix for ill-conditioning and negative variances.

    Checks performed:
    - Covariance availability.
    - Negative diagonal entries (unphysical variance).
    - NaN or Inf anywhere in the matrix.
    - Condition number exceeding a conservative threshold (1e10).

    Args:
        result: Fitted NLSQ result.

    Returns:
        List of warning/error message strings.
    """
    messages: list[str] = []

    if result.covariance is None:
        messages.append("Covariance matrix unavailable; uncertainty estimates missing.")
        return messages

    cov = result.covariance

    if not np.all(np.isfinite(cov)):
        messages.append("Covariance matrix contains NaN or Inf entries.")
        return messages  # Further checks are meaningless.

    diag = np.diag(cov)
    neg_mask = diag < 0.0
    if np.any(neg_mask):
        bad = [result.parameter_names[i] for i in np.where(neg_mask)[0]]
        messages.append(
            f"Negative variance(s) in covariance diagonal for: {', '.join(bad)}"
        )

    # Condition number via ratio of singular values (robust to near-singular matrices).
    try:
        sv = np.linalg.svd(cov, compute_uv=False)
        sv_max = float(sv[0])
        sv_min = float(sv[-1])
        # Numerical singularity is judged against a relative rank tolerance
        # (sv_max * max(shape) * eps, matching np.linalg.matrix_rank) rather than
        # an exact sv_min > 0 test: a genuinely rank-deficient matrix can return a
        # tiny positive smallest singular value from roundoff, and that roundoff is
        # backend-dependent (e.g. macOS Accelerate vs Linux OpenBLAS).
        eps = float(np.finfo(np.result_type(cov.dtype, np.float64)).eps)
        tol = sv_max * max(cov.shape) * eps
        if sv_min <= tol:
            messages.append(
                "Covariance matrix is singular (zero or negative singular value)."
            )
        else:
            cond = sv_max / sv_min
            if cond > 1e10:
                messages.append(
                    f"Ill-conditioned covariance matrix (condition number={cond:.2e}); "
                    "parameter correlations may be unreliable."
                )
    except np.linalg.LinAlgError:
        messages.append("SVD of covariance matrix failed; matrix may be degenerate.")

    return messages


def check_chi_squared(
    result: NLSQResult,
    max_reduced_chi2: float = 10.0,
    min_reduced_chi2: float = 0.01,
) -> list[str]:
    """Check reduced chi-squared is in a physically reasonable range.

    Values well above 1 indicate a poor fit or under-estimated uncertainties.
    Values far below 1 suggest over-fitting or over-estimated uncertainties.

    Args:
        result: Fitted NLSQ result.
        max_reduced_chi2: Upper threshold; values above this are flagged as errors.
        min_reduced_chi2: Lower threshold; values below this are flagged as warnings.

    Returns:
        List of warning/error message strings.
    """
    messages: list[str] = []

    chi2 = result.reduced_chi_squared
    if chi2 is None:
        messages.append("Reduced chi-squared not available; fit quality unknown.")
        return messages

    if not np.isfinite(chi2):
        messages.append(f"Reduced chi-squared is non-finite: {chi2}.")
        return messages

    if chi2 > max_reduced_chi2:
        messages.append(
            f"Very poor fit: chi2_red = {chi2:.3f} exceeds maximum threshold "
            f"{max_reduced_chi2:.3f}."
        )
    elif chi2 > 2.0:
        messages.append(
            f"Mediocre fit: chi2_red = {chi2:.3f} > 2 suggests systematic residuals "
            "or underestimated data uncertainties."
        )
    elif chi2 < min_reduced_chi2:
        messages.append(
            f"Suspected over-fit: chi2_red = {chi2:.4f} is below minimum threshold "
            f"{min_reduced_chi2:.4f}."
        )
    elif chi2 < 0.5:
        messages.append(
            f"Possible over-fit: chi2_red = {chi2:.4f} < 0.5 suggests over-estimated "
            "data uncertainties or too many free parameters."
        )

    return messages


def check_uncertainty_ratios(
    result: NLSQResult,
    max_relative_uncertainty: float = 1.0,
) -> list[str]:
    """Check that parameter uncertainties are not unreasonably large.

    A relative uncertainty |sigma / value| > ``max_relative_uncertainty``
    indicates a poorly constrained parameter.  Parameters with a best-fit
    value of exactly zero are skipped (ratio undefined).

    Args:
        result: Fitted NLSQ result.
        max_relative_uncertainty: Threshold for |sigma / value|.  Defaults
            to 1.0 (100% relative uncertainty).

    Returns:
        List of warning message strings, one per over-uncertain parameter.
    """
    messages: list[str] = []

    if result.uncertainties is None:
        return messages  # Already flagged by check_covariance_health.

    for name, val, unc in zip(
        result.parameter_names, result.parameters, result.uncertainties, strict=True
    ):
        if not np.isfinite(unc) or not np.isfinite(val):
            continue  # Covered by check_covariance_health / NaN checks.

        if val == 0.0:
            continue

        ratio = abs(float(unc) / float(val))
        if ratio > max_relative_uncertainty:
            messages.append(
                f"Parameter '{name}' poorly constrained: "
                f"value={val:.3e}, sigma={unc:.3e}, "
                f"relative uncertainty={ratio * 100:.0f}%."
            )

    return messages


# ---------------------------------------------------------------------------
# Homodyne-parity: ResultValidator class + functional validators
# ---------------------------------------------------------------------------


class ResultValidator:
    """Homodyne-parity validator for NLSQ optimization results.

    Validates optimized parameters, covariance matrices, and result consistency
    using a simple ``bool`` return API.  Warnings are accessible via
    :attr:`validation_warnings` after each call.

    This is distinct from
    :class:`~heterodyne.optimization.nlsq.validation.result.ResultValidator`,
    which uses a structured :class:`~heterodyne.optimization.nlsq.validation.result.ValidationReport`.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialise ResultValidator.

        Parameters
        ----------
        strict_mode:
            If ``True``, :meth:`validate_all` raises ``ValueError`` on failure.
            If ``False``, logs warnings and returns ``False``.
        """
        self.strict_mode = strict_mode
        self._validation_warnings: list[str] = []

    def validate_all(
        self,
        params: np.ndarray,
        covariance: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        chi_squared: float | None = None,
    ) -> bool:
        """Validate all result components.

        Parameters
        ----------
        params:
            Optimized parameter values.
        covariance:
            Parameter covariance matrix, or ``None``.
        bounds:
            Parameter bounds ``(lower, upper)``, or ``None``.
        chi_squared:
            Chi-squared value for quality check, or ``None``.

        Returns
        -------
        bool
            ``True`` if all validation passes, ``False`` otherwise.
        """
        self._validation_warnings = []

        if not validate_optimized_params(params, bounds):
            self._validation_warnings.append("Optimized parameters outside bounds")

        if covariance is not None:
            if not validate_covariance(covariance, len(params)):
                self._validation_warnings.append("Covariance matrix invalid")

        if chi_squared is not None:
            if not validate_result_consistency(params, chi_squared):
                self._validation_warnings.append("Result consistency check failed")

        if self._validation_warnings:
            if self.strict_mode:
                raise ValueError(
                    f"Result validation failed: {'; '.join(self._validation_warnings)}"
                )
            else:
                for warning in self._validation_warnings:
                    logger.warning("Result validation warning: %s", warning)
                return False

        return True

    @property
    def validation_warnings(self) -> list[str]:
        """Validation warnings from the last :meth:`validate_all` call."""
        return self._validation_warnings.copy()


def validate_optimized_params(
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    tolerance: float = 1e-10,
) -> bool:
    """Validate that optimized parameters are finite and within bounds.

    Parameters
    ----------
    params:
        Optimized parameter values.
    bounds:
        ``(lower, upper)`` bound arrays, or ``None``.
    tolerance:
        Absolute tolerance applied when checking bounds (default ``1e-10``).

    Returns
    -------
    bool
        ``True`` if params are finite and within bounds.
    """
    if not np.all(np.isfinite(params)):
        non_finite = np.where(~np.isfinite(params))[0]
        logger.warning(
            "Non-finite optimized params at indices: %s", non_finite.tolist()
        )
        return False

    if bounds is None:
        return True

    lower, upper = bounds
    below_lower = params < lower - tolerance
    above_upper = params > upper + tolerance

    if np.any(below_lower) or np.any(above_upper):
        violations: list[str] = []
        if np.any(below_lower):
            indices = np.where(below_lower)[0]
            violations.append(f"below lower at {indices.tolist()}")
        if np.any(above_upper):
            indices = np.where(above_upper)[0]
            violations.append(f"above upper at {indices.tolist()}")
        logger.warning("Params outside bounds: %s", ", ".join(violations))
        return False

    return True


def validate_covariance(covariance: np.ndarray, n_params: int) -> bool:
    """Validate covariance matrix shape and basic properties.

    Parameters
    ----------
    covariance:
        Parameter covariance matrix.
    n_params:
        Expected number of parameters.

    Returns
    -------
    bool
        ``True`` if covariance is valid.
    """
    if covariance.shape != (n_params, n_params):
        logger.warning(
            f"Covariance shape {covariance.shape} != ({n_params}, {n_params})"
        )
        return False

    if not np.all(np.isfinite(covariance)):
        nan_count = int(np.sum(np.isnan(covariance)))
        inf_count = int(np.sum(np.isinf(covariance)))
        logger.warning(
            f"Covariance contains non-finite values: {nan_count} NaN, {inf_count} Inf"
        )
        return False

    diag = np.diag(covariance)
    if np.any(diag < 0):
        neg_indices = np.where(diag < 0)[0]
        logger.warning(
            f"Covariance has negative diagonal at indices: {neg_indices.tolist()}"
        )
        return False

    return True


def validate_result_consistency(
    params: np.ndarray,
    chi_squared: float,
) -> bool:
    """Validate basic consistency of the optimization result.

    Parameters
    ----------
    params:
        Optimized parameter values.
    chi_squared:
        Chi-squared value from the fit.

    Returns
    -------
    bool
        ``True`` if result passes consistency checks.
    """
    if not np.all(np.isfinite(params)):
        logger.warning("Optimized params contain non-finite values")
        return False

    if not np.isfinite(chi_squared):
        logger.warning("Chi-squared is non-finite: %s", chi_squared)
        return False

    if chi_squared < 0:
        logger.warning("Chi-squared is negative: %.4e", chi_squared)
        return False

    # Soft warnings (do not fail)
    if chi_squared < 1e-15:
        logger.warning("Chi-squared suspiciously low: %.2e", chi_squared)

    if chi_squared > 1e10:
        logger.warning("Chi-squared very high: %.2e", chi_squared)

    return True


# ---------------------------------------------------------------------------
# Top-level entry point (heterodyne-native)
# ---------------------------------------------------------------------------


def validate_result(
    result: NLSQResult,
    registry: ParameterRegistry | None = None,
) -> ValidationReport:
    """Run all validation checks on an NLSQ result.

    Checks are executed unconditionally so that the report captures the
    complete picture even when earlier checks fail.  Bound-saturation
    checks are skipped when no registry is provided.

    Args:
        result: The fitted NLSQ result to validate.
        registry: Optional parameter registry for bound-saturation checks.
            When ``None``, the default registry is used.

    Returns:
        A :class:`ValidationReport` summarising all issues and scalar metrics.
    """
    from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

    if registry is None:
        registry = DEFAULT_REGISTRY

    report = ValidationReport()

    # --- fatal pre-conditions ---
    if not result.success:
        report.errors.append(f"Optimization did not converge: {result.message}")
        report.passed = False
        logger.warning(
            "validate_result: optimization failure, skipping detailed checks."
        )
        return report

    if not np.all(np.isfinite(result.parameters)):
        report.errors.append("Fitted parameters contain NaN or Inf values.")
        report.passed = False
        return report

    # --- chi-squared ---
    chi2_msgs = check_chi_squared(result)
    # Only the "very poor fit" message (threshold > max_reduced_chi2) is an error.
    for msg in chi2_msgs:
        if msg.startswith("Very poor fit"):
            report.errors.append(msg)
        else:
            report.warnings.append(msg)

    if result.reduced_chi_squared is not None and np.isfinite(
        result.reduced_chi_squared
    ):
        report.metrics["reduced_chi2"] = float(result.reduced_chi_squared)

    # --- covariance health ---
    cov_msgs = check_covariance_health(result)
    for msg in cov_msgs:
        if "Negative variance" in msg or "singular" in msg or "SVD" in msg:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)

    # Condition number as a scalar metric when computable.
    if result.covariance is not None and np.all(np.isfinite(result.covariance)):
        try:
            sv = np.linalg.svd(result.covariance, compute_uv=False)
            if sv[-1] > 0.0:
                report.metrics["covariance_condition_number"] = float(sv[0] / sv[-1])
        except np.linalg.LinAlgError:
            pass

    # --- bound saturation ---
    sat_msgs = check_bound_saturation(result, registry)
    report.warnings.extend(sat_msgs)
    report.metrics["n_saturated_params"] = float(len(sat_msgs))

    # --- uncertainty ratios ---
    unc_msgs = check_uncertainty_ratios(result)
    report.warnings.extend(unc_msgs)
    report.metrics["n_poorly_constrained_params"] = float(len(unc_msgs))

    # --- final verdict ---
    if report.errors:
        report.passed = False

    logger.debug(
        "validate_result: %d error(s), %d warning(s), passed=%s",
        len(report.errors),
        len(report.warnings),
        report.passed,
    )

    return report


__all__ = [
    # Homodyne-parity class + functions
    "ResultValidator",
    "validate_covariance",
    "validate_optimized_params",
    "validate_result_consistency",
    # Heterodyne-native dataclass + functions
    "ValidationReport",
    "check_bound_saturation",
    "check_chi_squared",
    "check_covariance_health",
    "check_uncertainty_ratios",
    "validate_result",
]
