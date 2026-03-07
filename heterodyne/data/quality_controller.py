"""Comprehensive data quality assessment for XPCS correlation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

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
