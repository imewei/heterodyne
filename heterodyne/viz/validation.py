"""Validation result visualisation for XPCS analysis.

Provides traffic-light style plots for :class:`ValidationReport` and
:class:`QualityReport`, plus a bounds-check chart for fitted parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from heterodyne.data.quality_controller import QualityReport
    from heterodyne.optimization.nlsq.validation import ValidationReport

logger = get_logger(__name__)

# Traffic-light colour map for severity / quality levels
_SEVERITY_COLORS: dict[str, str] = {
    "info": "#4CAF50",  # green
    "warning": "#FF9800",  # amber
    "error": "#F44336",  # red
}

_QUALITY_COLORS: dict[str, str] = {
    "good": "#4CAF50",
    "acceptable": "#8BC34A",
    "warning": "#FF9800",
    "critical": "#F44336",
}


def plot_validation_report(
    report: ValidationReport,
    ax: Axes | None = None,
) -> Axes:
    """Visual summary of a ValidationReport in traffic-light style.

    Each issue is drawn as a coloured marker on a horizontal strip, with
    the overall pass/fail status shown in the title.

    Args:
        report: ValidationReport from NLSQ validation.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(report.issues) + 1)))

    issues = report.issues
    if not issues:
        ax.text(
            0.5,
            0.5,
            "No issues reported",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Validation: PASS" if report.is_valid else "Validation: FAIL")
        ax.set_axis_off()
        return ax

    y_positions = np.arange(len(issues))
    colors = [_SEVERITY_COLORS.get(i.severity.value, "#999999") for i in issues]
    labels = [i.message for i in issues]

    ax.barh(y_positions, [1] * len(issues), color=colors, height=0.6, left=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.invert_yaxis()

    status = "PASS" if report.is_valid else "FAIL"
    title_color = "#4CAF50" if report.is_valid else "#F44336"
    ax.set_title(
        f"Validation: {status}", fontsize=13, color=title_color, fontweight="bold"
    )

    if hasattr(ax.figure, "tight_layout"):
        ax.figure.tight_layout()

    return ax


def plot_bounds_check(
    param_values: np.ndarray,
    param_names: list[str],
    bounds: list[tuple[float, float]],
    ax: Axes | None = None,
    edge_fraction: float = 0.05,
) -> Axes:
    """Plot parameters versus their bounds, flagging edge proximity.

    Parameters within *edge_fraction* of a bound are highlighted in amber;
    parameters outside bounds are highlighted in red.

    Args:
        param_values: 1-D array of fitted parameter values.
        param_names: List of parameter names.
        bounds: List of (lower, upper) tuples aligned with *param_values*.
        ax: Optional existing Axes.
        edge_fraction: Fraction of bound range that counts as "near edge".

    Returns:
        The matplotlib Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(param_names) + 1)))

    n = len(param_names)
    y = np.arange(n)

    for i, (val, (lo, hi), _name) in enumerate(
        zip(param_values, bounds, param_names, strict=True)
    ):
        span = hi - lo
        edge_width = span * edge_fraction

        # Determine colour
        if val < lo or val > hi:
            colour = "#F44336"  # red — out of bounds
        elif val < lo + edge_width or val > hi - edge_width:
            colour = "#FF9800"  # amber — near edge
        else:
            colour = "#4CAF50"  # green

        # Draw bound range as a grey bar
        ax.barh(i, span, left=lo, height=0.3, color="#E0E0E0", edgecolor="#BDBDBD")
        # Mark the parameter value
        ax.plot(val, i, "o", color=colour, markersize=8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xlabel("Parameter value")
    ax.set_title("Parameter Bounds Check")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    if hasattr(ax.figure, "tight_layout"):
        ax.figure.tight_layout()

    return ax


def plot_quality_report(
    quality_report: QualityReport,
    ax: Axes | None = None,
) -> Axes:
    """Visualise a QualityReport as a traffic-light bar chart.

    Each metric is rendered as a horizontal bar coloured by its quality
    level.

    Args:
        quality_report: QualityReport from the QualityController.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(10, max(3, 0.6 * len(quality_report.metrics) + 1))
        )

    metrics = quality_report.metrics
    if not metrics:
        ax.text(
            0.5,
            0.5,
            "No metrics reported",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        return ax

    y_positions = np.arange(len(metrics))
    colors = [_QUALITY_COLORS.get(m.level.value, "#999999") for m in metrics]
    labels = [f"{m.name}: {m.message}" for m in metrics]

    ax.barh(y_positions, [1] * len(metrics), color=colors, height=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.invert_yaxis()

    overall = quality_report.overall_level.value
    title_color = _QUALITY_COLORS.get(overall, "#999999")
    ax.set_title(
        f"Data Quality: {overall.upper()}",
        fontsize=13,
        color=title_color,
        fontweight="bold",
    )

    if hasattr(ax.figure, "tight_layout"):
        ax.figure.tight_layout()

    return ax
