"""Statistics aggregation across multiple NLSQ fits.

Provides batch-level summaries, outlier detection, and formatted
reporting for collections of :class:`~heterodyne.optimization.nlsq.results.NLSQResult`
objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FitStatistics:
    """Descriptive statistics for a single parameter across many fits.

    Attributes:
        param_name: Parameter name.
        mean: Arithmetic mean of fitted values.
        std: Standard deviation.
        median: Median value.
        q25: 25th percentile.
        q75: 75th percentile.
        n_fits: Number of successful fits contributing.
    """

    param_name: str
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    n_fits: int


@dataclass
class BatchResult:
    """Aggregated result for a batch of NLSQ fits.

    Attributes:
        statistics: Per-parameter statistics.
        overall_success_rate: Fraction of fits that succeeded.
        mean_chi2: Mean reduced chi-squared across successful fits.
        convergence_count: Number of successful fits.
        total_count: Total number of fits attempted.
    """

    statistics: list[FitStatistics] = field(default_factory=list)
    overall_success_rate: float = 0.0
    mean_chi2: float = 0.0
    convergence_count: int = 0
    total_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_batch_statistics(results: list[NLSQResult]) -> BatchResult:
    """Aggregate parameter statistics across multiple NLSQ fits.

    Only *successful* fits (``result.success is True``) contribute to the
    per-parameter statistics and mean chi-squared.

    Args:
        results: List of :class:`NLSQResult` objects.

    Returns:
        A :class:`BatchResult` summarising the batch.

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        msg = "results list must not be empty"
        raise ValueError(msg)

    total_count = len(results)
    successful = [r for r in results if r.success]
    convergence_count = len(successful)
    overall_success_rate = convergence_count / total_count

    if convergence_count == 0:
        logger.warning("No successful fits in batch of %d", total_count)
        return BatchResult(
            statistics=[],
            overall_success_rate=0.0,
            mean_chi2=0.0,
            convergence_count=0,
            total_count=total_count,
        )

    # Collect chi-squared values
    chi2_values = [
        r.reduced_chi_squared
        for r in successful
        if r.reduced_chi_squared is not None
    ]
    mean_chi2 = float(np.mean(chi2_values)) if chi2_values else 0.0

    # Build parameter arrays keyed by name.  All successful results must
    # share the same parameter_names list; use the first as reference.
    ref_names = successful[0].parameter_names
    param_arrays: dict[str, list[float]] = {name: [] for name in ref_names}

    for result in successful:
        for name, value in zip(result.parameter_names, result.parameters, strict=True):
            if name in param_arrays:
                param_arrays[name].append(float(value))

    statistics: list[FitStatistics] = []
    for name in ref_names:
        vals = np.asarray(param_arrays[name])
        statistics.append(
            FitStatistics(
                param_name=name,
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                median=float(np.median(vals)),
                q25=float(np.percentile(vals, 25)),
                q75=float(np.percentile(vals, 75)),
                n_fits=len(vals),
            )
        )

    logger.info(
        "Batch statistics: %d/%d successful, mean chi2=%.4f",
        convergence_count,
        total_count,
        mean_chi2,
    )

    return BatchResult(
        statistics=statistics,
        overall_success_rate=overall_success_rate,
        mean_chi2=mean_chi2,
        convergence_count=convergence_count,
        total_count=total_count,
    )


def identify_outlier_fits(
    results: list[NLSQResult],
    sigma: float = 3.0,
) -> list[int]:
    """Identify fits that are statistical outliers.

    A fit is considered an outlier if **any** of the following are more
    than *sigma* standard deviations from the batch mean (computed over
    successful fits only):

    * Its reduced chi-squared.
    * Any of its parameter values.

    Args:
        results: List of :class:`NLSQResult` objects.
        sigma: Number of standard deviations for the outlier threshold.

    Returns:
        Sorted list of indices (into *results*) flagged as outliers.
    """
    if not results:
        return []

    successful_indices = [i for i, r in enumerate(results) if r.success]
    if len(successful_indices) < 2:
        return []

    outlier_set: set[int] = set()

    # --- Chi-squared outliers ---
    chi2_pairs: list[tuple[int, float]] = []
    for i in successful_indices:
        chi2 = results[i].reduced_chi_squared
        if chi2 is not None:
            chi2_pairs.append((i, chi2))

    if len(chi2_pairs) >= 2:
        chi2_vals = np.array([v for _, v in chi2_pairs])
        mu, sd = float(np.mean(chi2_vals)), float(np.std(chi2_vals))
        if sd > 0:
            for idx, val in chi2_pairs:
                if abs(val - mu) > sigma * sd:
                    outlier_set.add(idx)

    # --- Parameter outliers ---
    ref_names = results[successful_indices[0]].parameter_names
    for p_idx, _name in enumerate(ref_names):
        param_pairs: list[tuple[int, float]] = []
        for i in successful_indices:
            if p_idx < len(results[i].parameters):
                param_pairs.append((i, float(results[i].parameters[p_idx])))

        if len(param_pairs) < 2:
            continue
        vals = np.array([v for _, v in param_pairs])
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        if sd > 0:
            for idx, val in param_pairs:
                if abs(val - mu) > sigma * sd:
                    outlier_set.add(idx)

    outliers = sorted(outlier_set)
    if outliers:
        logger.info("Identified %d outlier fit(s) at indices %s", len(outliers), outliers)
    return outliers


def format_batch_report(batch_result: BatchResult) -> str:
    """Format a :class:`BatchResult` as a human-readable summary table.

    Args:
        batch_result: Aggregated batch result.

    Returns:
        Multi-line formatted string.
    """
    lines: list[str] = [
        "Batch Fit Report",
        "=" * 70,
        f"Total fits:      {batch_result.total_count}",
        f"Successful:      {batch_result.convergence_count}",
        f"Success rate:    {batch_result.overall_success_rate:.1%}",
        f"Mean chi2_red:   {batch_result.mean_chi2:.4f}",
        "",
    ]

    if batch_result.statistics:
        header = (
            f"  {'Parameter':<18s} {'Mean':>12s} {'Std':>12s} "
            f"{'Median':>12s} {'Q25':>12s} {'Q75':>12s} {'N':>5s}"
        )
        lines.append("Parameter Statistics:")
        lines.append("-" * 70)
        lines.append(header)
        lines.append("-" * 70)

        for s in batch_result.statistics:
            lines.append(
                f"  {s.param_name:<18s} {s.mean:>12.4e} {s.std:>12.4e} "
                f"{s.median:>12.4e} {s.q25:>12.4e} {s.q75:>12.4e} {s.n_fits:>5d}"
            )
    else:
        lines.append("No parameter statistics available (no successful fits).")

    return "\n".join(lines)
