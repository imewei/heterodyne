"""General-purpose data filtering utilities for XPCS correlation data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from heterodyne.data.types import FilterResult, QRange
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def apply_time_window(
    c2: np.ndarray,
    t: np.ndarray,
    t_min: float,
    t_max: float,
) -> FilterResult:
    """Mask correlation data outside a time window.

    Retains only rows and columns of ``c2`` whose corresponding time
    values fall within ``[t_min, t_max]``.

    Args:
        c2: Correlation matrix, shape (n_t, n_t) or (n_phi, n_t, n_t).
        t: 1D time array of length n_t.
        t_min: Minimum time (inclusive).
        t_max: Maximum time (inclusive).

    Returns:
        FilterResult with the windowed data and boolean mask over the
        time axis.

    Raises:
        ValueError: If t_min > t_max or no data falls within the window.
    """
    if t_min > t_max:
        raise ValueError(f"t_min ({t_min}) must be <= t_max ({t_max})")

    mask_1d = (t >= t_min) & (t <= t_max)
    n_kept = int(np.sum(mask_1d))
    n_removed = len(t) - n_kept

    if n_kept == 0:
        raise ValueError(
            f"No time points in [{t_min}, {t_max}]. "
            f"Data range: [{float(t.min())}, {float(t.max())}]"
        )

    if c2.ndim == 2:
        filtered = c2[np.ix_(mask_1d, mask_1d)]
    elif c2.ndim == 3:
        filtered = c2[:, np.ix_(mask_1d, mask_1d)[0], np.ix_(mask_1d, mask_1d)[1]]
    else:
        raise ValueError(f"c2 must be 2D or 3D, got {c2.ndim}D")

    logger.debug(
        "Time window [%.4g, %.4g]: kept %d/%d time points",
        t_min,
        t_max,
        n_kept,
        len(t),
    )

    return FilterResult(
        data=filtered,
        mask=mask_1d,
        n_removed=n_removed,
        reason=f"time_window [{t_min}, {t_max}]",
    )


def apply_q_range_filter(
    data: np.ndarray,
    q_values: np.ndarray,
    q_range: QRange,
) -> FilterResult:
    """Mask data outside a wavevector range.

    Assumes the first axis of ``data`` corresponds to the q-values.

    Args:
        data: Array with first axis indexed by q.
        q_values: 1D array of wavevector values.
        q_range: QRange specifying (q_min, q_max) inclusive bounds.

    Returns:
        FilterResult with filtered data along the q axis.

    Raises:
        ValueError: If no q values fall within the range.
    """
    if q_range.q_min > q_range.q_max:
        raise ValueError(f"q_min ({q_range.q_min}) must be <= q_max ({q_range.q_max})")

    mask = (q_values >= q_range.q_min) & (q_values <= q_range.q_max)
    n_kept = int(np.sum(mask))
    n_removed = len(q_values) - n_kept

    if n_kept == 0:
        raise ValueError(
            f"No q values in [{q_range.q_min}, {q_range.q_max}]. "
            f"Data range: [{float(q_values.min())}, {float(q_values.max())}]"
        )

    filtered = data[mask]

    logger.debug(
        "Q range [%.4g, %.4g]: kept %d/%d q values",
        q_range.q_min,
        q_range.q_max,
        n_kept,
        len(q_values),
    )

    return FilterResult(
        data=filtered,
        mask=mask,
        n_removed=n_removed,
        reason=f"q_range [{q_range.q_min}, {q_range.q_max}]",
    )


def apply_sigma_clip(
    c2: np.ndarray,
    sigma: float = 3.0,
) -> FilterResult:
    """Remove outliers from correlation data by sigma clipping.

    Elements more than ``sigma`` standard deviations from the mean
    (computed from finite values only) are replaced with NaN.

    Args:
        c2: Correlation array (any shape).
        sigma: Number of standard deviations for the clipping threshold.

    Returns:
        FilterResult with outliers replaced by NaN. The mask is True
        for elements that were retained (not clipped).
    """
    finite_mask = np.isfinite(c2)
    finite_values = c2[finite_mask]

    if finite_values.size == 0:
        return FilterResult(
            data=c2.copy(),
            mask=np.ones_like(c2, dtype=bool),
            n_removed=0,
            reason=f"sigma_clip (sigma={sigma}): no finite values",
        )

    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values))

    if std == 0.0:
        # All finite values identical; nothing to clip
        return FilterResult(
            data=c2.copy(),
            mask=finite_mask.copy(),
            n_removed=0,
            reason=f"sigma_clip (sigma={sigma}): zero variance",
        )

    inlier_mask = np.abs(c2 - mean) <= sigma * std
    # Keep existing NaN/Inf as-is; only newly-clipped become NaN
    outlier_mask = finite_mask & ~inlier_mask
    n_removed = int(np.sum(outlier_mask))

    result = c2.copy()
    result[outlier_mask] = np.nan

    if n_removed > 0:
        logger.info(
            "Sigma clip (%.1f sigma): removed %d/%d values (%.2f%%)",
            sigma,
            n_removed,
            c2.size,
            100.0 * n_removed / c2.size,
        )

    return FilterResult(
        data=result,
        mask=~outlier_mask,
        n_removed=n_removed,
        reason=f"sigma_clip (sigma={sigma})",
    )


def compute_data_mask(
    c2: np.ndarray,
    conditions: Sequence[np.ndarray],
) -> np.ndarray:
    """Combine multiple boolean filter conditions into a single mask.

    All condition masks are AND-ed together. Each condition array must
    be broadcastable to the shape of ``c2``.

    Args:
        c2: Reference correlation array (used only for shape).
        conditions: Sequence of boolean arrays to combine.

    Returns:
        Combined boolean mask (True = element passes all conditions).

    Raises:
        ValueError: If no conditions are provided.
    """
    if len(conditions) == 0:
        raise ValueError("At least one condition mask is required")

    combined = np.ones(c2.shape, dtype=bool)
    for condition in conditions:
        combined = combined & np.broadcast_to(condition, c2.shape)

    n_pass = int(np.sum(combined))
    logger.debug(
        "Combined %d conditions: %d/%d elements pass (%.1f%%)",
        len(conditions),
        n_pass,
        c2.size,
        100.0 * n_pass / c2.size if c2.size > 0 else 0.0,
    )

    return combined
