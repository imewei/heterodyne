"""Data validation and quality checks for XPCS data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
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


# ---------------------------------------------------------------------------
# Incremental validation infrastructure
# ---------------------------------------------------------------------------

class ValidationLevel(Enum):
    """Level of validation to perform."""

    NONE = "none"
    BASIC = "basic"
    FULL = "full"
    CUSTOM = "custom"


@dataclass
class ValidationIssue:
    """A single validation finding with structured metadata."""

    severity: str  # "error", "warning", "info"
    category: str  # "shape", "range", "physics", "statistics"
    message: str
    parameter: str | None = None
    value: float | None = None
    recommendation: str | None = None


class IncrementalValidationCache:
    """Hash-based cache for validation results to avoid redundant checks."""

    def __init__(self, max_cache_size: int = 100) -> None:
        self._cache: dict[str, DataQualityReport] = {}
        self._max_cache_size = max_cache_size
        self._hits: int = 0
        self._misses: int = 0

    def _compute_hash(self, data: XPCSData) -> str:
        """Compute lightweight fingerprint from corner samples, shape, and dtype."""
        h = hashlib.sha256()
        # Shape and dtype metadata
        h.update(str(data.c2.shape).encode())
        h.update(str(data.c2.dtype).encode())
        # Corner samples for collision resistance (O(1) cost)
        flat = data.c2.ravel()
        n = min(512, flat.size)
        sample = np.concatenate([flat[:n], flat[-n:]]) if flat.size > 2 * n else flat
        h.update(np.ascontiguousarray(sample).tobytes())
        # Include time arrays
        if data.t1 is not None:
            h.update(str(data.t1.shape).encode())
            t1_flat = data.t1.ravel()
            n_t = min(64, t1_flat.size)
            h.update(np.ascontiguousarray(t1_flat[:n_t]).tobytes())
        if data.t2 is not None:
            h.update(str(data.t2.shape).encode())
            t2_flat = data.t2.ravel()
            n_t = min(64, t2_flat.size)
            h.update(np.ascontiguousarray(t2_flat[:n_t]).tobytes())
        return h.hexdigest()

    def _check_validation_cache(self, data_hash: str) -> DataQualityReport | None:
        """Look up a cached validation result by hash."""
        result = self._cache.get(data_hash)
        if result is not None:
            self._hits += 1
            logger.debug("Validation cache hit for hash %s", data_hash[:12])
        else:
            self._misses += 1
        return result

    def _cache_validation_result(
        self, data_hash: str, report: DataQualityReport
    ) -> None:
        """Store a validation result, evicting oldest entry if at capacity."""
        if len(self._cache) >= self._max_cache_size:
            # Evict the first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[data_hash] = report

    def clear_validation_cache(self) -> None:
        """Clear all cached validation results and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss/size statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }


# ---------------------------------------------------------------------------
# Component validators
# ---------------------------------------------------------------------------


def _validate_array_component(
    c2: np.ndarray, issues: list[ValidationIssue]
) -> None:
    """Validate array shape, dtype, and size."""
    # Dimensionality check
    if c2.ndim == 2:
        if c2.shape[0] != c2.shape[1]:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="shape",
                    message=(
                        f"2D c2 array is not square: shape {c2.shape}"
                    ),
                    recommendation="Provide a square correlation matrix c2(t1, t2).",
                )
            )
    elif c2.ndim == 3:
        if c2.shape[1] != c2.shape[2]:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="shape",
                    message=(
                        f"3D batch c2 has non-square time dimensions: shape {c2.shape}"
                    ),
                    recommendation="Each batch slice must be square in the last two dims.",
                )
            )
    else:
        issues.append(
            ValidationIssue(
                severity="error",
                category="shape",
                message=f"c2 must be 2D or 3D, got ndim={c2.ndim}",
            )
        )

    # Dtype check
    if not np.issubdtype(c2.dtype, np.floating):
        issues.append(
            ValidationIssue(
                severity="warning",
                category="shape",
                message=f"c2 dtype is {c2.dtype}, expected floating-point",
                recommendation="Cast c2 to float64 before analysis.",
            )
        )

    # Size check
    if c2.size == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="shape",
                message="c2 array is empty (size 0)",
            )
        )
    elif c2.size < 4:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="shape",
                message=f"c2 array is very small (size {c2.size})",
                recommendation="At least a 2x2 correlation matrix is needed.",
            )
        )


def _validate_correlation_component(
    c2: np.ndarray, issues: list[ValidationIssue]
) -> None:
    """Validate correlation-specific properties: NaN/Inf, range, symmetry, diagonal."""
    if c2.size == 0:
        return

    # NaN / Inf
    nan_frac = float(np.mean(np.isnan(c2)))
    inf_frac = float(np.mean(np.isinf(c2)))

    if nan_frac > 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="range",
                message=f"c2 contains NaN values ({100 * nan_frac:.2f}%)",
                value=nan_frac,
            )
        )
    if inf_frac > 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="range",
                message=f"c2 contains Inf values ({100 * inf_frac:.2f}%)",
                value=inf_frac,
            )
        )

    finite_vals = c2[np.isfinite(c2)]
    if len(finite_vals) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="range",
                message="c2 has no finite values",
            )
        )
        return

    # Value range
    c2_min = float(np.min(finite_vals))
    c2_max = float(np.max(finite_vals))
    if c2_min < -10.0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="range",
                message=f"c2 minimum ({c2_min:.4g}) is unusually negative",
                value=c2_min,
                recommendation="Check for baseline subtraction issues.",
            )
        )
    if c2_max > 100.0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="range",
                message=f"c2 maximum ({c2_max:.4g}) is unusually large",
                value=c2_max,
                recommendation="Check normalization.",
            )
        )

    # Symmetry (2D square only)
    if c2.ndim == 2 and c2.shape[0] == c2.shape[1]:
        asym = np.abs(c2 - c2.T)
        finite_asym = asym[np.isfinite(asym)]
        if len(finite_asym) > 0:
            max_asym = float(np.max(finite_asym))
            scale = float(np.abs(c2).max()) + 1e-10
            rel_asym = max_asym / scale
            if rel_asym > 0.05:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="range",
                        message=f"Significant asymmetry: {100 * rel_asym:.2f}%",
                        value=rel_asym,
                    )
                )

    # Diagonal normalization (2D square)
    if c2.ndim == 2 and c2.shape[0] == c2.shape[1] and c2.shape[0] > 0:
        diag = np.diag(c2)
        finite_diag = diag[np.isfinite(diag)]
        if len(finite_diag) > 0:
            diag_mean = float(np.mean(finite_diag))
            if abs(diag_mean - 1.0) > 0.2:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="range",
                        message=(
                            f"Diagonal mean = {diag_mean:.3f}, "
                            "expected ~1.0 for normalized correlation"
                        ),
                        value=diag_mean,
                        recommendation="Verify normalization of c2.",
                    )
                )


def _validate_time_component(
    t1: np.ndarray | None,
    t2: np.ndarray | None,
    c2_shape: tuple[int, ...],
    issues: list[ValidationIssue],
) -> None:
    """Validate time arrays: monotonicity, length match, positivity."""
    for name, t_arr, expected_len in [
        ("t1", t1, c2_shape[-2] if len(c2_shape) >= 2 else None),
        ("t2", t2, c2_shape[-1] if len(c2_shape) >= 2 else None),
    ]:
        if t_arr is None:
            continue

        # Length match
        if expected_len is not None and len(t_arr) != expected_len:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="shape",
                    message=(
                        f"Time array {name} length {len(t_arr)} "
                        f"doesn't match c2 dimension {expected_len}"
                    ),
                    parameter=name,
                )
            )

        if len(t_arr) < 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="shape",
                    message=f"Time array {name} has fewer than 2 elements",
                    parameter=name,
                )
            )
            continue

        # Monotonicity
        diffs = np.diff(t_arr)
        if not np.all(diffs > 0):
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="range",
                    message=f"Time array {name} is not strictly increasing",
                    parameter=name,
                )
            )

        # Positivity
        if np.any(t_arr <= 0):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="range",
                    message=f"Time array {name} contains non-positive values",
                    parameter=name,
                    value=float(np.min(t_arr)),
                    recommendation="XPCS time values should be positive.",
                )
            )


def _validate_physics_parameters(
    data: XPCSData, issues: list[ValidationIssue]
) -> None:
    """Validate physics-related metadata against PhysicsConstants ranges.

    Imports PhysicsConstants locally to avoid circular imports
    (core/ should not be imported at module level in data/).
    """
    from heterodyne.core.physics import PhysicsConstants

    # q-value check
    if data.q is not None:
        q_val = data.q
        if q_val < PhysicsConstants.Q_MIN_TYPICAL:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="physics",
                    message=(
                        f"q = {q_val:.4g} A^-1 is below typical minimum "
                        f"({PhysicsConstants.Q_MIN_TYPICAL:.4g})"
                    ),
                    parameter="q",
                    value=q_val,
                )
            )
        if q_val > PhysicsConstants.Q_MAX_TYPICAL:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="physics",
                    message=(
                        f"q = {q_val:.4g} A^-1 exceeds typical maximum "
                        f"({PhysicsConstants.Q_MAX_TYPICAL:.4g})"
                    ),
                    parameter="q",
                    value=q_val,
                )
            )

    # dt check via metadata (if available)
    dt = data.metadata.get("dt")
    if dt is not None:
        dt_val = float(dt)
        if dt_val < PhysicsConstants.TIME_MIN_XPCS:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="physics",
                    message=(
                        f"dt = {dt_val:.4g} s is below XPCS minimum "
                        f"({PhysicsConstants.TIME_MIN_XPCS:.4g})"
                    ),
                    parameter="dt",
                    value=dt_val,
                )
            )
        if dt_val > PhysicsConstants.TIME_MAX_XPCS:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="physics",
                    message=(
                        f"dt = {dt_val:.4g} s exceeds XPCS maximum "
                        f"({PhysicsConstants.TIME_MAX_XPCS:.4g})"
                    ),
                    parameter="dt",
                    value=dt_val,
                )
            )


def _validate_statistical_properties(
    c2: np.ndarray, issues: list[ValidationIssue]
) -> None:
    """Check kurtosis, stationarity, and variance homogeneity."""
    finite_vals = c2[np.isfinite(c2)]
    if len(finite_vals) < 8:
        return

    # Kurtosis check (excess kurtosis; normal = 0)
    mean = float(np.mean(finite_vals))
    std = float(np.std(finite_vals))
    if std > 0:
        kurtosis = float(np.mean(((finite_vals - mean) / std) ** 4)) - 3.0
        if abs(kurtosis) > 10.0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="statistics",
                    message=f"Excess kurtosis = {kurtosis:.2f} (heavy tails)",
                    value=kurtosis,
                    recommendation=(
                        "Large kurtosis may indicate outliers "
                        "or non-Gaussian noise."
                    ),
                )
            )

    # Stationarity check: compare first and last quarter means
    # Operate along the last axis for 2D/3D
    if c2.ndim >= 2:
        n = c2.shape[-1]
        quarter = max(1, n // 4)
        first_q = c2[..., :quarter]
        last_q = c2[..., -quarter:]
        first_finite = first_q[np.isfinite(first_q)]
        last_finite = last_q[np.isfinite(last_q)]
        if len(first_finite) > 0 and len(last_finite) > 0:
            first_mean = float(np.mean(first_finite))
            last_mean = float(np.mean(last_finite))
            scale = max(abs(first_mean), abs(last_mean), 1e-10)
            drift = abs(first_mean - last_mean) / scale
            if drift > 0.2:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="statistics",
                        message=(
                            f"Non-stationary: first-quarter mean "
                            f"({first_mean:.4g}) vs last-quarter "
                            f"({last_mean:.4g}), drift = {drift:.2f}"
                        ),
                        value=drift,
                        recommendation="Check for beam drift or aging effects.",
                    )
                )

    # Variance homogeneity (first half vs second half)
    if c2.ndim >= 2:
        n = c2.shape[-1]
        half = max(1, n // 2)
        first_h = c2[..., :half]
        last_h = c2[..., half:]
        first_fin = first_h[np.isfinite(first_h)]
        last_fin = last_h[np.isfinite(last_h)]
        if len(first_fin) > 1 and len(last_fin) > 1:
            var1 = float(np.var(first_fin))
            var2 = float(np.var(last_fin))
            var_scale = max(var1, var2, 1e-20)
            var_ratio = abs(var1 - var2) / var_scale
            if var_ratio > 0.5:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        category="statistics",
                        message=(
                            f"Variance heterogeneity: first-half var "
                            f"= {var1:.4g}, second-half var = {var2:.4g}"
                        ),
                        value=var_ratio,
                    )
                )


# ---------------------------------------------------------------------------
# Statistics and quality scoring helpers
# ---------------------------------------------------------------------------


def _compute_data_statistics(c2: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for c2 data."""
    stats: dict[str, float] = {}

    total = float(c2.size) if c2.size > 0 else 1.0
    nan_frac = float(np.sum(np.isnan(c2))) / total
    inf_frac = float(np.sum(np.isinf(c2))) / total
    stats["nan_fraction"] = nan_frac
    stats["inf_fraction"] = inf_frac

    finite_vals = c2[np.isfinite(c2)]
    if len(finite_vals) == 0:
        stats["min"] = float("nan")
        stats["max"] = float("nan")
        stats["mean"] = float("nan")
        stats["std"] = float("nan")
        stats["kurtosis"] = float("nan")
        stats["skewness"] = float("nan")
        return stats

    stats["min"] = float(np.min(finite_vals))
    stats["max"] = float(np.max(finite_vals))
    stats["mean"] = float(np.mean(finite_vals))
    stats["std"] = float(np.std(finite_vals))

    std = stats["std"]
    mean = stats["mean"]
    if std > 0:
        centered = (finite_vals - mean) / std
        stats["kurtosis"] = float(np.mean(centered**4)) - 3.0
        stats["skewness"] = float(np.mean(centered**3))
    else:
        stats["kurtosis"] = 0.0
        stats["skewness"] = 0.0

    return stats


def _compute_quality_score(issues: list[ValidationIssue]) -> float:
    """Compute a 0.0-1.0 quality score from validation issues.

    Starts at 1.0, subtracts 0.3 per error, 0.1 per warning, 0.02 per info.
    Clamped to [0.0, 1.0].
    """
    score = 1.0
    penalties = {"error": 0.3, "warning": 0.1, "info": 0.02}
    for issue in issues:
        score -= penalties.get(issue.severity, 0.0)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Public incremental validation entry point
# ---------------------------------------------------------------------------


def validate_xpcs_data_incremental(
    data: XPCSData,
    level: ValidationLevel = ValidationLevel.FULL,
    cache: IncrementalValidationCache | None = None,
) -> DataQualityReport:
    """Validate XPCS data with incremental, cache-aware checking.

    Args:
        data: XPCSData instance to validate.
        level: Validation depth — NONE skips, BASIC runs array+time,
            FULL and CUSTOM run all validators.
        cache: Optional validation cache for deduplication.

    Returns:
        DataQualityReport with aggregated results.
    """
    # NONE — return trivially valid report
    if level is ValidationLevel.NONE:
        return DataQualityReport(is_valid=True)

    # Cache lookup
    data_hash: str | None = None
    if cache is not None:
        data_hash = cache._compute_hash(data)
        cached = cache._check_validation_cache(data_hash)
        if cached is not None:
            return cached

    issues: list[ValidationIssue] = []

    # --- BASIC level: array + time ---
    _validate_array_component(data.c2, issues)
    _validate_time_component(data.t1, data.t2, data.c2.shape, issues)

    # --- FULL / CUSTOM level: correlation + physics + statistics ---
    if level in (ValidationLevel.FULL, ValidationLevel.CUSTOM):
        _validate_correlation_component(data.c2, issues)
        _validate_physics_parameters(data, issues)
        _validate_statistical_properties(data.c2, issues)

    # Build errors / warnings lists
    errors = [
        issue.message for issue in issues if issue.severity == "error"
    ]
    warnings = [
        issue.message for issue in issues if issue.severity in ("warning", "info")
    ]

    # Statistics
    statistics = _compute_data_statistics(data.c2)
    statistics["quality_score"] = _compute_quality_score(issues)

    report = DataQualityReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )

    # Store in cache
    if cache is not None and data_hash is not None:
        cache._cache_validation_result(data_hash, report)

    return report
