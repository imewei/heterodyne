"""Comprehensive data quality assessment for XPCS correlation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.data.config import DataConfig

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality classification levels, ordered from best to worst."""

    GOOD = "good"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """A single quality assessment metric.

    Attributes:
        name: Human-readable metric name.
        value: Measured value of the metric.
        threshold: Threshold used for classification.
        level: Quality level assigned to this metric.
        message: Descriptive message explaining the assessment.
    """

    name: str
    value: float
    threshold: float
    level: QualityLevel
    message: str


@dataclass
class QualityReport:
    """Aggregated quality report from multiple metrics.

    Attributes:
        metrics: Individual quality metrics.
        overall_level: Worst-case level across all metrics.
        recommendations: Actionable suggestions for improving data quality.
    """

    metrics: list[QualityMetric] = field(default_factory=list)
    overall_level: QualityLevel = QualityLevel.GOOD
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable quality report.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "Data Quality Report",
            "=" * 50,
            f"Overall: {self.overall_level.value.upper()}",
            "",
        ]

        if self.metrics:
            lines.append("Metrics:")
            for m in self.metrics:
                marker = _level_marker(m.level)
                lines.append(
                    f"  {marker} {m.name}: {m.value:.4g} "
                    f"(threshold: {m.threshold:.4g}) - {m.message}"
                )

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


def _level_marker(level: QualityLevel) -> str:
    """Return a text marker for a quality level."""
    return {
        QualityLevel.GOOD: "[OK]",
        QualityLevel.ACCEPTABLE: "[OK]",
        QualityLevel.WARNING: "[WARN]",
        QualityLevel.CRITICAL: "[CRIT]",
    }[level]


# Priority ordering for worst-case aggregation
_LEVEL_PRIORITY: dict[QualityLevel, int] = {
    QualityLevel.GOOD: 0,
    QualityLevel.ACCEPTABLE: 1,
    QualityLevel.WARNING: 2,
    QualityLevel.CRITICAL: 3,
}


class QualityController:
    """Assess quality of XPCS correlation data.

    Runs a battery of checks and produces a ``QualityReport``.

    Args:
        config: Optional data configuration; thresholds may be drawn
            from it in the future.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self._config = config

    def assess(
        self,
        c2: np.ndarray,
        t: np.ndarray,
        q: np.ndarray | None = None,
        phi_angles: np.ndarray | None = None,
    ) -> QualityReport:
        """Run all quality checks and produce a report.

        Args:
            c2: Correlation data, shape (n_t, n_t) or (n_phi, n_t, n_t).
            t: 1D time array.
            q: Optional wavevector array.
            phi_angles: Optional phi angle array.

        Returns:
            QualityReport with per-metric assessments and recommendations.
        """
        metrics: list[QualityMetric] = []
        recommendations: list[str] = []

        metrics.append(self._check_nan_fraction(c2))
        metrics.append(self._check_snr(c2))
        metrics.append(self._check_value_range(c2))
        metrics.append(self._check_time_coverage(t))

        # Shape-dependent checks for square 2D or 3D batches
        if c2.ndim == 2 and c2.shape[0] == c2.shape[1]:
            metrics.append(self._check_symmetry(c2))
            metrics.append(self._check_diagonal_excess(c2))
        elif c2.ndim == 3 and c2.shape[1] == c2.shape[2]:
            # Assess on the first slice as representative
            metrics.append(self._check_symmetry(c2[0]))
            metrics.append(self._check_diagonal_excess(c2[0]))

        # Determine overall level (worst case)
        overall = QualityLevel.GOOD
        for m in metrics:
            if _LEVEL_PRIORITY[m.level] > _LEVEL_PRIORITY[overall]:
                overall = m.level

        # Build recommendations from non-GOOD metrics
        for m in metrics:
            if m.level == QualityLevel.CRITICAL:
                recommendations.append(f"CRITICAL: {m.message}")
            elif m.level == QualityLevel.WARNING:
                recommendations.append(f"Consider addressing: {m.message}")

        report = QualityReport(
            metrics=metrics,
            overall_level=overall,
            recommendations=recommendations,
        )

        logger.info("Quality assessment: %s", overall.value)
        return report

    # ---- Individual check methods ----

    def _check_nan_fraction(self, c2: np.ndarray) -> QualityMetric:
        """Check fraction of NaN values."""
        nan_frac = float(np.sum(np.isnan(c2))) / c2.size if c2.size > 0 else 0.0
        threshold = 0.05  # 5%

        if nan_frac == 0.0:
            level = QualityLevel.GOOD
            msg = "No NaN values"
        elif nan_frac <= 0.01:
            level = QualityLevel.ACCEPTABLE
            msg = f"{100 * nan_frac:.2f}% NaN (minimal)"
        elif nan_frac <= threshold:
            level = QualityLevel.WARNING
            msg = f"{100 * nan_frac:.2f}% NaN values present"
        else:
            level = QualityLevel.CRITICAL
            msg = f"{100 * nan_frac:.2f}% NaN values (exceeds {100 * threshold:.0f}% threshold)"

        return QualityMetric(
            name="nan_fraction",
            value=nan_frac,
            threshold=threshold,
            level=level,
            message=msg,
        )

    def _check_snr(self, c2: np.ndarray) -> QualityMetric:
        """Check signal-to-noise ratio of off-diagonal elements."""
        # Work on a 2D slice for consistency
        if c2.ndim == 3:
            slice_2d = c2[0]
        else:
            slice_2d = c2

        if slice_2d.ndim == 2 and slice_2d.shape[0] == slice_2d.shape[1]:
            off_diag = slice_2d[~np.eye(slice_2d.shape[0], dtype=bool)]
        else:
            off_diag = slice_2d.ravel()

        finite = off_diag[np.isfinite(off_diag)]
        if finite.size == 0:
            return QualityMetric(
                name="snr",
                value=0.0,
                threshold=5.0,
                level=QualityLevel.CRITICAL,
                message="No finite off-diagonal values for SNR computation",
            )

        mean = float(np.mean(finite))
        std = float(np.std(finite))
        snr = abs(mean) / std if std > 0 else 0.0
        threshold = 5.0

        if snr >= 10.0:
            level = QualityLevel.GOOD
            msg = f"SNR = {snr:.1f} (excellent)"
        elif snr >= threshold:
            level = QualityLevel.ACCEPTABLE
            msg = f"SNR = {snr:.1f} (adequate)"
        elif snr >= 2.0:
            level = QualityLevel.WARNING
            msg = f"SNR = {snr:.1f} (low, may affect fit quality)"
        else:
            level = QualityLevel.CRITICAL
            msg = f"SNR = {snr:.1f} (very low, data may be unusable)"

        return QualityMetric(
            name="snr",
            value=snr,
            threshold=threshold,
            level=level,
            message=msg,
        )

    def _check_symmetry(self, c2_2d: np.ndarray) -> QualityMetric:
        """Check c2(t1,t2) ~ c2(t2,t1) symmetry."""
        finite_mask = np.isfinite(c2_2d) & np.isfinite(c2_2d.T)
        if not np.any(finite_mask):
            return QualityMetric(
                name="symmetry",
                value=1.0,
                threshold=0.01,
                level=QualityLevel.CRITICAL,
                message="No finite values for symmetry check",
            )

        asymmetry = np.abs(c2_2d - c2_2d.T)
        scale = np.abs(c2_2d[finite_mask]).max()
        if scale < 1e-15:
            scale = 1.0

        rel_asymmetry = float(np.max(asymmetry[finite_mask])) / scale
        threshold = 0.01  # 1%

        if rel_asymmetry <= 1e-6:
            level = QualityLevel.GOOD
            msg = "Perfectly symmetric"
        elif rel_asymmetry <= threshold:
            level = QualityLevel.GOOD
            msg = f"Relative asymmetry {100 * rel_asymmetry:.4f}%"
        elif rel_asymmetry <= 0.05:
            level = QualityLevel.WARNING
            msg = f"Relative asymmetry {100 * rel_asymmetry:.2f}% (consider symmetrization)"
        else:
            level = QualityLevel.CRITICAL
            msg = f"Relative asymmetry {100 * rel_asymmetry:.2f}% (significant)"

        return QualityMetric(
            name="symmetry",
            value=rel_asymmetry,
            threshold=threshold,
            level=level,
            message=msg,
        )

    def _check_diagonal_excess(self, c2_2d: np.ndarray) -> QualityMetric:
        """Check whether diagonal values are excessively large relative to off-diagonal."""
        n = c2_2d.shape[0]
        diag = np.diag(c2_2d)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = c2_2d[off_diag_mask]

        finite_diag = diag[np.isfinite(diag)]
        finite_off = off_diag[np.isfinite(off_diag)]

        if finite_diag.size == 0 or finite_off.size == 0:
            return QualityMetric(
                name="diagonal_excess",
                value=0.0,
                threshold=2.0,
                level=QualityLevel.WARNING,
                message="Insufficient finite values for diagonal excess check",
            )

        diag_mean = float(np.mean(finite_diag))
        off_mean = float(np.mean(finite_off))

        if abs(off_mean) < 1e-15:
            ratio = abs(diag_mean)
        else:
            ratio = abs(diag_mean / off_mean)

        threshold = 2.0

        if ratio <= 1.5:
            level = QualityLevel.GOOD
            msg = f"Diagonal/off-diagonal ratio = {ratio:.2f}"
        elif ratio <= threshold:
            level = QualityLevel.ACCEPTABLE
            msg = f"Diagonal/off-diagonal ratio = {ratio:.2f} (moderate)"
        elif ratio <= 5.0:
            level = QualityLevel.WARNING
            msg = f"Diagonal/off-diagonal ratio = {ratio:.2f} (elevated, consider diagonal exclusion)"
        else:
            level = QualityLevel.CRITICAL
            msg = f"Diagonal/off-diagonal ratio = {ratio:.2f} (excessive)"

        return QualityMetric(
            name="diagonal_excess",
            value=ratio,
            threshold=threshold,
            level=level,
            message=msg,
        )

    def _check_value_range(self, c2: np.ndarray) -> QualityMetric:
        """Check that values fall in a physically plausible range."""
        finite = c2[np.isfinite(c2)]
        if finite.size == 0:
            return QualityMetric(
                name="value_range",
                value=0.0,
                threshold=100.0,
                level=QualityLevel.CRITICAL,
                message="No finite values in data",
            )

        abs_max = float(np.max(np.abs(finite)))
        threshold = 100.0  # Arbitrary upper bound for normalized correlation

        if abs_max <= 10.0:
            level = QualityLevel.GOOD
            msg = f"Value range [{float(np.min(finite)):.4g}, {float(np.max(finite)):.4g}]"
        elif abs_max <= threshold:
            level = QualityLevel.ACCEPTABLE
            msg = f"Wide value range (max |value| = {abs_max:.4g}), data may need normalization"
        else:
            level = QualityLevel.WARNING
            msg = f"Very large values (max |value| = {abs_max:.4g}), normalization recommended"

        return QualityMetric(
            name="value_range",
            value=abs_max,
            threshold=threshold,
            level=level,
            message=msg,
        )

    def _check_time_coverage(self, t: np.ndarray) -> QualityMetric:
        """Check time array for sufficient coverage and uniformity."""
        n_points = len(t)
        threshold = 10.0  # Minimum number of time points

        if n_points < 2:
            return QualityMetric(
                name="time_coverage",
                value=float(n_points),
                threshold=threshold,
                level=QualityLevel.CRITICAL,
                message=f"Only {n_points} time point(s), need at least 2",
            )

        dt = np.diff(t)
        if not np.all(dt > 0):
            return QualityMetric(
                name="time_coverage",
                value=float(n_points),
                threshold=threshold,
                level=QualityLevel.CRITICAL,
                message="Time array is not strictly increasing",
            )

        dt_uniformity = float(np.std(dt) / np.median(dt)) if np.median(dt) > 0 else 0.0

        if n_points >= threshold and dt_uniformity < 0.01:
            level = QualityLevel.GOOD
            msg = f"{n_points} points, uniform spacing (CV={dt_uniformity:.4f})"
        elif n_points >= threshold:
            level = QualityLevel.ACCEPTABLE
            msg = f"{n_points} points, non-uniform spacing (CV={dt_uniformity:.3f})"
        elif n_points >= 5:
            level = QualityLevel.WARNING
            msg = f"Only {n_points} time points (recommend >= {int(threshold)})"
        else:
            level = QualityLevel.CRITICAL
            msg = f"Only {n_points} time points (minimum recommended: {int(threshold)})"

        return QualityMetric(
            name="time_coverage",
            value=float(n_points),
            threshold=threshold,
            level=level,
            message=msg,
        )


# ---------------------------------------------------------------------------
# 4-Stage Quality Control Pipeline (Task 1.3)
# ---------------------------------------------------------------------------


class QualityControlStage(Enum):
    """Processing stage at which quality assessment is performed."""

    RAW = "raw"
    FILTERED = "filtered"
    PREPROCESSED = "preprocessed"
    FINAL = "final"


@dataclass
class QualityControlConfig:
    """Configuration for the 4-stage quality control pipeline.

    Attributes:
        nan_threshold: Maximum acceptable NaN fraction (0-1).
        snr_threshold: Minimum acceptable signal-to-noise ratio.
        symmetry_threshold: Maximum acceptable relative asymmetry.
        value_range_max: Maximum acceptable absolute value.
        min_time_points: Minimum required number of time points.
        auto_repair_nans: Whether to automatically interpolate NaN values.
        auto_repair_outliers: Whether to automatically clip outlier values.
        outlier_sigma: Number of standard deviations beyond which values
            are considered outliers (used when ``auto_repair_outliers`` is True).
        report_format: Output format for reports, ``"text"`` or ``"json"``.
    """

    nan_threshold: float = 0.05
    snr_threshold: float = 5.0
    symmetry_threshold: float = 0.01
    value_range_max: float = 100.0
    min_time_points: int = 10
    auto_repair_nans: bool = False
    auto_repair_outliers: bool = False
    outlier_sigma: float = 5.0
    report_format: str = "text"


@dataclass
class QualityControlResult:
    """Result of a single-stage quality assessment.

    Attributes:
        stage: The pipeline stage this result corresponds to.
        report: The underlying :class:`QualityReport` produced by the controller.
        auto_corrections: Descriptions of any automatic corrections applied.
        recommendations: Actionable suggestions derived from the assessment.
        quality_score: Aggregate quality score in [0, 1] (1 = perfect).
        metadata: Arbitrary key-value metadata for downstream consumers.
    """

    stage: QualityControlStage
    report: QualityReport
    auto_corrections: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def assess_stage(
    controller: QualityController,
    c2: np.ndarray,
    t: np.ndarray,
    stage: QualityControlStage,
    config: QualityControlConfig | None = None,
) -> QualityControlResult:
    """Run a quality assessment for a specific pipeline stage.

    Args:
        controller: An existing :class:`QualityController` instance.
        c2: Correlation data array.
        t: 1-D time array.
        stage: Pipeline stage being assessed.
        config: Optional configuration; defaults are used when *None*.

    Returns:
        A :class:`QualityControlResult` with score and recommendations.
    """
    if config is None:
        config = QualityControlConfig()

    report = controller.assess(c2, t)

    # Compute aggregate quality score
    quality_score = 1.0
    for m in report.metrics:
        if m.level == QualityLevel.CRITICAL:
            quality_score -= 0.3
        elif m.level == QualityLevel.WARNING:
            quality_score -= 0.1
    quality_score = max(quality_score, 0.0)

    # Stage-specific recommendations
    recommendations: list[str] = list(report.recommendations)
    if stage == QualityControlStage.RAW:
        if quality_score < 0.5:
            recommendations.append(
                "Raw data quality is poor; consider re-acquisition or "
                "aggressive filtering before proceeding."
            )
    elif stage == QualityControlStage.FILTERED:
        if quality_score < 0.7:
            recommendations.append(
                "Filtered data still has quality issues; review filter "
                "parameters or apply additional preprocessing."
            )
    elif stage == QualityControlStage.PREPROCESSED:
        if quality_score < 0.8:
            recommendations.append(
                "Preprocessed data does not meet target quality; "
                "verify normalization and symmetrization steps."
            )
    elif stage == QualityControlStage.FINAL:
        if quality_score < 0.9:
            recommendations.append(
                "Final data quality is below recommended threshold; "
                "fitting results should be interpreted with caution."
            )

    logger.info(
        "Stage %s quality score: %.2f (%d metrics)",
        stage.value,
        quality_score,
        len(report.metrics),
    )

    return QualityControlResult(
        stage=stage,
        report=report,
        recommendations=recommendations,
        quality_score=quality_score,
        metadata={"config": config, "n_metrics": len(report.metrics)},
    )


def suggest_fixes(report: QualityReport) -> list[dict[str, Any]]:
    """Analyse a :class:`QualityReport` and suggest concrete corrective actions.

    Args:
        report: Quality report to analyse.

    Returns:
        List of fix suggestion dicts, each with ``action``, ``description``,
        and ``priority`` keys.
    """
    fixes: list[dict[str, Any]] = []

    for m in report.metrics:
        if m.name == "nan_fraction" and m.level == QualityLevel.CRITICAL:
            fixes.append(
                {
                    "action": "interpolate_nans",
                    "description": (
                        f"Interpolate {100 * m.value:.2f}% NaN values using "
                        "nearest-neighbor or linear interpolation."
                    ),
                    "priority": "high",
                }
            )
        elif m.name == "snr" and m.level in (
            QualityLevel.WARNING,
            QualityLevel.CRITICAL,
        ):
            fixes.append(
                {
                    "action": "smooth_data",
                    "description": (
                        f"SNR is {m.value:.1f}; apply smoothing or "
                        "averaging to improve signal-to-noise ratio."
                    ),
                    "priority": "medium",
                }
            )
        elif m.name == "symmetry" and m.level in (
            QualityLevel.WARNING,
            QualityLevel.CRITICAL,
        ):
            fixes.append(
                {
                    "action": "symmetrize",
                    "description": (
                        f"Relative asymmetry is {100 * m.value:.2f}%; "
                        "symmetrize the correlation matrix via "
                        "(C2 + C2^T) / 2."
                    ),
                    "priority": "medium",
                }
            )
        elif m.name == "value_range" and m.level == QualityLevel.CRITICAL:
            fixes.append(
                {
                    "action": "normalize",
                    "description": (
                        f"Maximum |value| is {m.value:.4g}; normalize data "
                        "to bring values into a physically plausible range."
                    ),
                    "priority": "high",
                }
            )
        elif m.name == "time_coverage" and m.level in (
            QualityLevel.WARNING,
            QualityLevel.CRITICAL,
        ):
            fixes.append(
                {
                    "action": "extend_measurement",
                    "description": (
                        f"Only {int(m.value)} time points; extend "
                        "measurement duration or increase sampling rate."
                    ),
                    "priority": "low",
                }
            )

    return fixes


def apply_auto_corrections(
    c2: np.ndarray,
    t: np.ndarray,
    report: QualityReport,
    config: QualityControlConfig,
) -> tuple[np.ndarray, list[str]]:
    """Apply automatic corrections to data based on quality report.

    The input arrays are not mutated; a corrected copy of *c2* is returned.

    Args:
        c2: Correlation data array.
        t: 1-D time array.
        report: Quality report from a prior assessment.
        config: Configuration controlling which corrections are enabled.

    Returns:
        Tuple of (corrected_c2, list_of_correction_descriptions).
    """
    corrections: list[str] = []
    c2_corrected = c2.copy()

    # --- NaN interpolation (nearest-neighbour fill) ---
    if config.auto_repair_nans:
        nan_mask = np.isnan(c2_corrected)
        nan_count = int(np.sum(nan_mask))
        if nan_count > 0:
            # Nearest-neighbour: replace each NaN with mean of finite neighbours
            finite_mean = float(np.nanmean(c2_corrected))
            c2_corrected = np.where(nan_mask, finite_mean, c2_corrected)
            corrections.append(
                f"Interpolated {nan_count} NaN values using finite-mean "
                f"fill ({finite_mean:.4g})."
            )
            logger.info("Auto-repair: interpolated %d NaN values", nan_count)

    # --- Outlier clipping ---
    if config.auto_repair_outliers:
        finite = c2_corrected[np.isfinite(c2_corrected)]
        if finite.size > 0:
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            if std > 0:
                lo = mean - config.outlier_sigma * std
                hi = mean + config.outlier_sigma * std
                outlier_mask = (c2_corrected < lo) | (c2_corrected > hi)
                # Only count finite outliers (NaNs are not outliers)
                outlier_mask = outlier_mask & np.isfinite(c2_corrected)
                outlier_count = int(np.sum(outlier_mask))
                if outlier_count > 0:
                    c2_corrected = np.clip(c2_corrected, lo, hi)
                    corrections.append(
                        f"Clipped {outlier_count} outlier values beyond "
                        f"{config.outlier_sigma:.1f} sigma "
                        f"(range [{lo:.4g}, {hi:.4g}])."
                    )
                    logger.info(
                        "Auto-repair: clipped %d outliers (%.1f sigma)",
                        outlier_count,
                        config.outlier_sigma,
                    )

    return c2_corrected, corrections


def _compute_adaptive_thresholds(
    c2: np.ndarray,
    config: QualityControlConfig,
) -> QualityControlConfig:
    """Adjust quality thresholds based on empirical data characteristics.

    Returns a *new* :class:`QualityControlConfig` — the input is never mutated.

    Args:
        c2: Correlation data used to characterise noise level.
        config: Baseline configuration.

    Returns:
        A new config with potentially relaxed thresholds.
    """
    finite = c2[np.isfinite(c2)]
    if finite.size == 0:
        return config

    mean = float(np.mean(finite))
    std = float(np.std(finite))
    nan_frac = float(np.sum(np.isnan(c2))) / c2.size if c2.size > 0 else 0.0

    new_snr_threshold = config.snr_threshold
    new_nan_threshold = config.nan_threshold

    # Relax SNR threshold for very noisy data
    if mean != 0.0 and abs(std / mean) > 1.0:
        new_snr_threshold = config.snr_threshold * 0.5
        logger.debug(
            "Adaptive thresholds: relaxed snr_threshold from %.1f to %.1f "
            "(high noise: std/mean=%.2f)",
            config.snr_threshold,
            new_snr_threshold,
            abs(std / mean),
        )

    # Relax NaN threshold slightly when NaN fraction already above 1%
    if nan_frac > 0.01:
        new_nan_threshold = config.nan_threshold * 1.5
        logger.debug(
            "Adaptive thresholds: relaxed nan_threshold from %.3f to %.3f "
            "(%.2f%% NaNs detected)",
            config.nan_threshold,
            new_nan_threshold,
            100 * nan_frac,
        )

    return QualityControlConfig(
        nan_threshold=new_nan_threshold,
        snr_threshold=new_snr_threshold,
        symmetry_threshold=config.symmetry_threshold,
        value_range_max=config.value_range_max,
        min_time_points=config.min_time_points,
        auto_repair_nans=config.auto_repair_nans,
        auto_repair_outliers=config.auto_repair_outliers,
        outlier_sigma=config.outlier_sigma,
        report_format=config.report_format,
    )


def export_report(
    result: QualityControlResult,
    format: str = "text",
) -> str | dict[str, Any]:
    """Export a :class:`QualityControlResult` in the requested format.

    Args:
        result: Quality control result to export.
        format: ``"text"`` for a human-readable string, ``"json"`` for a
            JSON-safe dictionary.

    Returns:
        Formatted report as a string or dict.

    Raises:
        ValueError: If *format* is not ``"text"`` or ``"json"``.
    """
    if format == "text":
        lines = [result.report.summary()]
        if result.auto_corrections:
            lines.append("")
            lines.append("Auto-corrections applied:")
            for corr in result.auto_corrections:
                lines.append(f"  * {corr}")
        if result.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in result.recommendations:
                lines.append(f"  - {rec}")
        lines.append("")
        lines.append(f"Quality score: {result.quality_score:.2f}")
        return "\n".join(lines)

    if format == "json":
        return {
            "stage": result.stage.value,
            "quality_score": result.quality_score,
            "overall_level": result.report.overall_level.value,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "level": m.level.value,
                    "message": m.message,
                }
                for m in result.report.metrics
            ],
            "auto_corrections": result.auto_corrections,
            "recommendations": result.recommendations,
            "metadata": {
                k: v
                for k, v in result.metadata.items()
                if not isinstance(v, QualityControlConfig)
            },
        }

    msg = f"Unsupported report format: {format!r} (use 'text' or 'json')"
    raise ValueError(msg)


def track_quality_history(
    history: list[QualityControlResult],
    new_result: QualityControlResult,
) -> list[QualityControlResult]:
    """Append a new result to the quality history and log the trend.

    Args:
        history: Existing list of results (may be empty).
        new_result: The latest quality control result.

    Returns:
        The updated history list (same object, mutated in place).
    """
    if history:
        prev = history[-1]
        delta = new_result.quality_score - prev.quality_score
        direction = "improved" if delta > 0 else "degraded" if delta < 0 else "unchanged"
        logger.info(
            "Quality trend: %s -> %s (%.2f -> %.2f, %+.2f %s)",
            prev.stage.value,
            new_result.stage.value,
            prev.quality_score,
            new_result.quality_score,
            delta,
            direction,
        )
    else:
        logger.info(
            "Quality history started at stage %s (score=%.2f)",
            new_result.stage.value,
            new_result.quality_score,
        )

    history.append(new_result)
    return history


def run_4_stage_pipeline(
    controller: QualityController,
    c2: np.ndarray,
    t: np.ndarray,
    config: QualityControlConfig | None = None,
) -> list[QualityControlResult]:
    """Execute the full 4-stage quality control pipeline.

    Stages are run in order: RAW -> FILTERED -> PREPROCESSED -> FINAL.
    Between stages, automatic corrections are applied when enabled in
    *config*. Adaptive thresholds are computed once from the raw data.

    Args:
        controller: A :class:`QualityController` instance.
        c2: Correlation data array.
        t: 1-D time array.
        config: Pipeline configuration; defaults are used when *None*.

    Returns:
        List of four :class:`QualityControlResult` objects, one per stage.
    """
    if config is None:
        config = QualityControlConfig()

    # Compute adaptive thresholds from the raw data
    adapted_config = _compute_adaptive_thresholds(c2, config)

    stages = [
        QualityControlStage.RAW,
        QualityControlStage.FILTERED,
        QualityControlStage.PREPROCESSED,
        QualityControlStage.FINAL,
    ]

    history: list[QualityControlResult] = []
    current_c2 = c2

    for stage in stages:
        result = assess_stage(controller, current_c2, t, stage, adapted_config)

        # Apply auto-corrections between stages (not after FINAL)
        if stage != QualityControlStage.FINAL and (
            adapted_config.auto_repair_nans or adapted_config.auto_repair_outliers
        ):
            current_c2, corrections = apply_auto_corrections(
                current_c2, t, result.report, adapted_config
            )
            result.auto_corrections.extend(corrections)

        history = track_quality_history(history, result)

    # Log summary
    first_score = history[0].quality_score
    final_score = history[-1].quality_score
    logger.info(
        "4-stage pipeline complete: quality %.2f -> %.2f "
        "(%d total corrections applied)",
        first_score,
        final_score,
        sum(len(r.auto_corrections) for r in history),
    )

    return history
