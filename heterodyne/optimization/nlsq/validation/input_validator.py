"""Pre-fit input validation for NLSQ optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass  # Any is imported above for runtime + annotation use

logger = get_logger(__name__)


class InputValidator:
    """Validates NLSQ inputs before optimization runs.

    Checks: data shape/finiteness, bounds consistency, initial params
    within bounds.

    Supports two usage patterns:

    * **Heterodyne-native**: call :meth:`validate` with a raw data array and
      explicit bounds to obtain a :class:`~heterodyne.optimization.nlsq.validation.result.ValidationReport`.
    * **Homodyne-parity**: call :meth:`validate_all` with xdata/ydata/initial_params/bounds
      to get a simple ``bool``, with errors accessible via :attr:`validation_errors`.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        """Initialise InputValidator.

        Parameters
        ----------
        strict_mode:
            If ``True``, :meth:`validate_all` raises ``ValueError`` on failure.
            If ``False``, logs warnings and returns ``False``.
        """
        self.strict_mode = strict_mode
        self._validation_errors: list[str] = []

    # ------------------------------------------------------------------
    # Homodyne-parity interface
    # ------------------------------------------------------------------

    def validate_all(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> bool:
        """Validate all input data.

        Parameters
        ----------
        xdata:
            Independent variable data (t1, t2, phi).
        ydata:
            Dependent variable data (g2 values).
        initial_params:
            Initial parameter guess.
        bounds:
            Parameter bounds ``(lower, upper)`` or ``None``.

        Returns
        -------
        bool
            ``True`` if all validation passes, ``False`` otherwise.
        """
        self._validation_errors = []

        if not validate_array_dimensions(xdata, ydata):
            self._validation_errors.append(
                f"Array dimension mismatch: xdata.shape[0]={len(xdata)}, "
                f"ydata.shape[0]={len(ydata)}"
            )

        if not validate_no_nan_inf(xdata, "xdata"):
            self._validation_errors.append("xdata contains NaN or Inf values")
        if not validate_no_nan_inf(ydata, "ydata"):
            self._validation_errors.append("ydata contains NaN or Inf values")
        if not validate_no_nan_inf(initial_params, "initial_params"):
            self._validation_errors.append("initial_params contains NaN or Inf values")

        if bounds is not None:
            if not validate_bounds_consistency(bounds, initial_params):
                self._validation_errors.append(
                    "Bounds are inconsistent with initial parameters"
                )

        if not validate_initial_params(initial_params, bounds):
            self._validation_errors.append("Initial parameters outside bounds")

        if self._validation_errors:
            if self.strict_mode:
                raise ValueError(
                    f"Input validation failed: {'; '.join(self._validation_errors)}"
                )
            else:
                for error in self._validation_errors:
                    logger.warning("Input validation warning: %s", error)
                return False

        return True

    @property
    def validation_errors(self) -> list[str]:
        """Validation errors from the last :meth:`validate_all` call."""
        return self._validation_errors.copy()

    # ------------------------------------------------------------------
    # Heterodyne-native interface (structured report)
    # ------------------------------------------------------------------

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
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "Empty data array",
                    "data_empty",
                )
            )
            return

        n_nan = int(np.sum(np.isnan(data)))
        if n_nan > 0:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"NaN values in data ({n_nan} elements)",
                    "data_nan",
                    float(n_nan),
                )
            )

        n_inf = int(np.sum(np.isinf(data)))
        if n_inf > 0:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Inf values in data ({n_inf} elements)",
                    "data_inf",
                    float(n_inf),
                )
            )

    def _check_bounds(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        report: ValidationReport,
    ) -> None:
        lower, upper = bounds
        if lower.shape != upper.shape:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Bounds shape mismatch: lower={lower.shape}, upper={upper.shape}",
                    "bounds_shape",
                )
            )
            return

        inverted = np.where(lower > upper)[0]
        if len(inverted) > 0:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Inverted bounds at indices {inverted.tolist()}: lower > upper",
                    "bounds_inverted",
                    float(len(inverted)),
                )
            )

    def _check_initial_params(
        self,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        report: ValidationReport,
    ) -> None:
        lower, upper = bounds
        if params.shape != lower.shape:
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Params shape {params.shape} != bounds shape {lower.shape}",
                    "params_shape",
                )
            )
            return

        below = np.where(params < lower)[0]
        above = np.where(params > upper)[0]
        if len(below) > 0 or len(above) > 0:
            out_of_bounds = sorted(set(below.tolist() + above.tolist()))
            report.issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Initial params outside bounds at indices {out_of_bounds}",
                    "params_bounds",
                    float(len(out_of_bounds)),
                )
            )


# ---------------------------------------------------------------------------
# Module-level functional validators (homodyne parity)
# ---------------------------------------------------------------------------


def validate_array_dimensions(xdata: np.ndarray, ydata: np.ndarray) -> bool:
    """Validate that *xdata* and *ydata* have compatible first dimensions.

    Parameters
    ----------
    xdata:
        Independent variable data.
    ydata:
        Dependent variable data.

    Returns
    -------
    bool
        ``True`` if ``len(xdata) == len(ydata)``, ``False`` otherwise.
    """
    if len(xdata) != len(ydata):
        logger.warning(
            f"Array dimension mismatch: len(xdata)={len(xdata)}, len(ydata)={len(ydata)}"
        )
        return False
    return True


def validate_no_nan_inf(
    arr: np.ndarray,
    name: str,
    iteration: int | None = None,
    context: dict[str, Any] | None = None,
) -> bool:
    """Validate that *arr* contains no NaN or Inf values.

    Parameters
    ----------
    arr:
        Array to validate.
    name:
        Human-readable name used in log messages.
    iteration:
        Optional iteration number, included in log messages when provided.
    context:
        Optional extra key/value pairs for structured log messages.

    Returns
    -------
    bool
        ``True`` if all elements are finite, ``False`` otherwise.
    """
    nan_mask = ~np.isfinite(arr)
    nan_count = int(np.sum(np.isnan(arr)))
    inf_count = int(np.sum(np.isinf(arr)))

    if nan_count == 0 and inf_count == 0:
        return True

    nan_indices = np.where(np.isnan(arr))[0][:5]
    inf_indices = np.where(np.isinf(arr))[0][:5]
    iter_str = f" (iteration {iteration})" if iteration is not None else ""
    ctx_str = f", context={context}" if context else ""
    logger.warning(
        f"Array '{name}'{iter_str} contains non-finite values{ctx_str}:\n"
        f"  NaN count: {nan_count}, first indices: {nan_indices.tolist()}\n"
        f"  Inf count: {inf_count}, first indices: {inf_indices.tolist()}\n"
        f"  Array shape: {arr.shape}, dtype: {arr.dtype}\n"
        f"  Array range: [{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]"
    )
    del nan_mask  # suppress unused-var warning
    return False


def validate_bounds_consistency(
    bounds: tuple[np.ndarray, np.ndarray],
    initial_params: np.ndarray,
) -> bool:
    """Validate that bounds are self-consistent and compatible with *initial_params*.

    Parameters
    ----------
    bounds:
        ``(lower, upper)`` bound arrays.
    initial_params:
        Initial parameter values (used for length check only).

    Returns
    -------
    bool
        ``True`` if bounds are valid, ``False`` otherwise.
    """
    lower, upper = bounds

    if len(lower) != len(initial_params):
        logger.warning(
            f"Lower bound length {len(lower)} != params length {len(initial_params)}"
        )
        return False

    if len(upper) != len(initial_params):
        logger.warning(
            f"Upper bound length {len(upper)} != params length {len(initial_params)}"
        )
        return False

    if not np.all(lower <= upper):
        violations = np.where(lower > upper)[0]
        logger.warning("Lower > upper at indices: %s", violations)
        return False

    return True


def validate_initial_params(
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    """Validate that *initial_params* are within *bounds*.

    Parameters
    ----------
    initial_params:
        Initial parameter values.
    bounds:
        ``(lower, upper)`` bound arrays, or ``None`` (always passes).

    Returns
    -------
    bool
        ``True`` if all parameters are within bounds (or no bounds given).
    """
    if bounds is None:
        return True

    lower, upper = bounds

    if not np.all(np.isfinite(initial_params)):
        non_finite = np.where(~np.isfinite(initial_params))[0]
        logger.warning("Non-finite initial params at indices: %s", non_finite)
        return False

    below_lower = initial_params < lower
    above_upper = initial_params > upper

    if np.any(below_lower):
        indices = np.where(below_lower)[0]
        logger.warning("Params below lower bound at indices: %s", indices)
        return False

    if np.any(above_upper):
        indices = np.where(above_upper)[0]
        logger.warning("Params above upper bound at indices: %s", indices)
        return False

    return True


__all__ = [
    "InputValidator",
    "validate_array_dimensions",
    "validate_bounds_consistency",
    "validate_initial_params",
    "validate_no_nan_inf",
]
