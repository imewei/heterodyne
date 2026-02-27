"""Data validation and quality checks for XPCS data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.data.xpcs_loader import XPCSData

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Report of data quality validation results."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Data Quality Report",
            "=" * 40,
            f"Valid: {self.is_valid}",
        ]

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  ✗ {err}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        if self.statistics:
            lines.append("\nStatistics:")
            for key, val in self.statistics.items():
                lines.append(f"  {key}: {val:.4g}")

        return "\n".join(lines)


def validate_xpcs_data(
    data: XPCSData,
    expected_shape: tuple[int, ...] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    check_symmetry: bool = True,
    check_nans: bool = True,
) -> DataQualityReport:
    """Validate XPCS correlation data.

    Args:
        data: XPCSData to validate
        expected_shape: Expected shape of c2 matrix
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        check_symmetry: Whether to check for c2(t1,t2) ≈ c2(t2,t1)
        check_nans: Whether to check for NaN/Inf values

    Returns:
        DataQualityReport with validation results
    """
    errors: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, float] = {}

    c2 = data.c2

    # Shape validation
    if expected_shape is not None and c2.shape != expected_shape:
        errors.append(f"Shape mismatch: got {c2.shape}, expected {expected_shape}")

    # NaN/Inf checks
    if check_nans:
        nan_count = float(np.sum(np.isnan(c2)))
        inf_count = float(np.sum(np.isinf(c2)))

        if nan_count > 0:
            pct = 100 * nan_count / c2.size
            errors.append(f"Contains {int(nan_count)} NaN values ({pct:.2f}%)")
            statistics["nan_count"] = nan_count

        if inf_count > 0:
            errors.append(f"Contains {int(inf_count)} infinite values")
            statistics["inf_count"] = inf_count

    # Value range checks
    finite_values = c2[np.isfinite(c2)]
    if len(finite_values) > 0:
        statistics["min"] = float(np.min(finite_values))
        statistics["max"] = float(np.max(finite_values))
        statistics["mean"] = float(np.mean(finite_values))
        statistics["std"] = float(np.std(finite_values))

        if min_value is not None and statistics["min"] < min_value:
            warnings.append(f"Values below {min_value}: min = {statistics['min']:.4g}")

        if max_value is not None and statistics["max"] > max_value:
            warnings.append(f"Values above {max_value}: max = {statistics['max']:.4g}")

    # Symmetry check (for 2D square matrices)
    if check_symmetry and c2.ndim == 2 and c2.shape[0] == c2.shape[1]:
        asymmetry = np.abs(c2 - c2.T)
        finite_asymm = asymmetry[np.isfinite(asymmetry)]
        if len(finite_asymm) == 0:
            max_asymmetry = None
        else:
            max_asymmetry = np.max(np.abs(finite_asymm))

        if max_asymmetry is None:
            rel_asymmetry = None
        else:
            rel_asymmetry = max_asymmetry / (np.abs(c2).max() + 1e-10)

        if max_asymmetry is not None:
            statistics["max_asymmetry"] = max_asymmetry
        if rel_asymmetry is not None:
            statistics["relative_asymmetry"] = rel_asymmetry

        if rel_asymmetry is not None and rel_asymmetry > 0.01:
            warnings.append(f"Significant asymmetry: {100*rel_asymmetry:.2f}%")

    # Time array validation
    if data.t1 is not None:
        if not np.all(np.diff(data.t1) > 0):
            errors.append("Time array t1 is not strictly increasing")
        statistics["t1_min"] = float(data.t1.min())
        statistics["t1_max"] = float(data.t1.max())

    if data.t2 is not None:
        if not np.all(np.diff(data.t2) > 0):
            errors.append("Time array t2 is not strictly increasing")
        statistics["t2_min"] = float(data.t2.min())
        statistics["t2_max"] = float(data.t2.max())

    # Diagonal check (should typically be ~1 for normalized correlation)
    if c2.ndim == 2 and c2.shape[0] == c2.shape[1]:
        diag = np.diag(c2)
        diag_mean = np.mean(diag[np.isfinite(diag)])
        statistics["diagonal_mean"] = float(diag_mean)

        if abs(diag_mean - 1.0) > 0.1:
            warnings.append(f"Diagonal mean = {diag_mean:.3f}, expected ~1.0 for normalized c2")

    is_valid = len(errors) == 0

    return DataQualityReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )


def validate_time_consistency(
    t: np.ndarray,
    c2_shape: tuple[int, ...],
    dt_expected: float | None = None,
) -> DataQualityReport:
    """Validate time array consistency with correlation data.

    Args:
        t: Time array
        c2_shape: Shape of correlation matrix
        dt_expected: Expected time step (optional)

    Returns:
        DataQualityReport
    """
    errors: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, float] = {}

    # Length check
    if len(t) < 2:
        errors.append("Time array must have at least 2 elements")
        return DataQualityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )

    if len(t) != c2_shape[0]:
        errors.append(f"Time length {len(t)} doesn't match c2 dimension {c2_shape[0]}")

    # Monotonicity
    diffs = np.diff(t)
    if not np.all(diffs > 0):
        errors.append("Time array is not strictly increasing")
    else:
        dt_actual = np.median(diffs)
        dt_std = np.std(diffs)
        statistics["dt_median"] = float(dt_actual)
        statistics["dt_std"] = float(dt_std)

        # Uniformity check
        if dt_std / dt_actual > 0.01:
            warnings.append(f"Non-uniform time steps: dt_std/dt = {dt_std/dt_actual:.3f}")

        # Expected dt check
        if dt_expected is not None:
            dt_diff = abs(dt_actual - dt_expected) / dt_expected
            if dt_diff > 0.01:
                warnings.append(
                    f"Time step mismatch: measured {dt_actual:.4g}, expected {dt_expected:.4g}"
                )

    return DataQualityReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )
