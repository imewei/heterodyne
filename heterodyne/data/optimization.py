"""Dataset optimization strategies for XPCS correlation data.

Provides subsampling, time-range estimation, dataset statistics,
and strategy recommendation utilities for pre-processing large
correlation matrices before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubsamplingConfig:
    """Configuration for correlation data subsampling.

    Attributes:
        max_points: Maximum number of points to retain along each time axis.
        method: Subsampling strategy --- one of ``'uniform'``, ``'random'``,
            or ``'adaptive'``.
        seed: Optional RNG seed for reproducibility (used by ``'random'``
            and ``'adaptive'`` methods).
    """

    max_points: int = 10000
    method: str = "uniform"
    seed: int | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def subsample_correlation(
    c2: np.ndarray,
    config: SubsamplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a two-time correlation matrix.

    **Subsampling is off by default and must be explicitly requested.**
    A warning is always emitted when this function is called so that every
    invocation is logged, per project rules.

    Args:
        c2: Two-time correlation matrix, shape ``(n_t, n_t)``.
        config: Subsampling configuration.

    Returns:
        Tuple of ``(subsampled_c2, indices_used)`` where *indices_used*
        are the retained row/column indices into the original matrix.

    Raises:
        ValueError: If *method* is not one of the supported strategies or
            if *c2* is not a 2-D square array.
    """
    if c2.ndim != 2 or c2.shape[0] != c2.shape[1]:
        msg = f"c2 must be a square 2-D array, got shape {c2.shape}"
        raise ValueError(msg)

    n = c2.shape[0]

    # Nothing to subsample if data is already small enough
    if n <= config.max_points:
        logger.warning(
            "Subsampling requested but data size (%d) <= max_points (%d); "
            "returning original data unchanged",
            n,
            config.max_points,
        )
        return c2.copy(), np.arange(n)

    logger.warning(
        "Subsampling ACTIVE: reducing %d points to %d using method='%s' "
        "(seed=%s). This is an explicit opt-in operation.",
        n,
        config.max_points,
        config.method,
        config.seed,
    )

    if config.method == "uniform":
        indices = _subsample_uniform(n, config.max_points)
    elif config.method == "random":
        indices = _subsample_random(n, config.max_points, config.seed)
    elif config.method == "adaptive":
        indices = _subsample_adaptive(n, config.max_points, config.seed)
    else:
        msg = (
            f"Unknown subsampling method '{config.method}'. "
            f"Supported: 'uniform', 'random', 'adaptive'."
        )
        raise ValueError(msg)

    subsampled = c2[np.ix_(indices, indices)]
    return subsampled, indices


def estimate_optimal_time_range(
    c2: np.ndarray,
    t: np.ndarray,
    snr_threshold: float = 2.0,
) -> tuple[float, float]:
    """Estimate the time range where signal-to-noise exceeds a threshold.

    Uses the diagonal decay profile of *c2* to estimate the SNR at each
    lag and returns the range where the SNR stays above *snr_threshold*.

    Args:
        c2: Two-time correlation matrix, shape ``(n_t, n_t)``.
        t: 1-D time array of length *n_t*.
        snr_threshold: Minimum signal-to-noise ratio to consider valid.

    Returns:
        Tuple ``(t_min, t_max)`` giving the recommended time window.

    Raises:
        ValueError: If inputs are incompatible or contain no finite data.
    """
    if c2.ndim != 2 or c2.shape[0] != c2.shape[1]:
        msg = f"c2 must be a square 2-D array, got shape {c2.shape}"
        raise ValueError(msg)

    n = c2.shape[0]
    if t.shape != (n,):
        msg = f"t must have shape ({n},), got {t.shape}"
        raise ValueError(msg)

    # Compute mean and std along each diagonal offset (lag)
    max_lag = n
    snr_profile = np.zeros(max_lag)

    for lag in range(max_lag):
        diag_vals = np.diag(c2, k=lag)
        finite = diag_vals[np.isfinite(diag_vals)]
        if finite.size < 2:
            snr_profile[lag] = 0.0
            continue
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        snr_profile[lag] = abs(mean) / std if std > 0 else 0.0

    # Find contiguous region above threshold starting from lag 0
    above = snr_profile >= snr_threshold
    if not np.any(above):
        logger.warning(
            "No lag region exceeds SNR threshold %.2f; returning full range",
            snr_threshold,
        )
        return float(t[0]), float(t[-1])

    # The useful range maps to time indices where the lag-based SNR is valid.
    # A lag of k corresponds to |t_i - t_j| ~ t[k] - t[0].
    # We find the maximum lag with acceptable SNR.
    last_good_lag = int(np.max(np.nonzero(above)[0]))
    first_good_lag = int(np.min(np.nonzero(above)[0]))

    # Map lag back to time values
    t_min = float(t[first_good_lag]) if first_good_lag < n else float(t[0])
    t_max = float(t[min(last_good_lag, n - 1)])

    logger.info(
        "Optimal time range: [%.4g, %.4g] (lags %d-%d above SNR=%.1f)",
        t_min,
        t_max,
        first_good_lag,
        last_good_lag,
        snr_threshold,
    )
    return t_min, t_max


def compute_dataset_statistics(
    c2: np.ndarray,
    t: np.ndarray,
) -> dict[str, float | int]:
    """Compute summary statistics for a two-time correlation dataset.

    Args:
        c2: Two-time correlation matrix, shape ``(n_t, n_t)``.
        t: 1-D time array.

    Returns:
        Dictionary with keys:
        - ``mean``: Mean of finite values.
        - ``std``: Standard deviation of finite values.
        - ``snr``: Signal-to-noise ratio (|mean| / std).
        - ``n_nan``: Number of NaN elements.
        - ``dynamic_range``: max / min of absolute finite values.
        - ``effective_rank``: Numerical rank estimate via singular value
          threshold (ratio > 1e-6 of largest).
    """
    finite_mask = np.isfinite(c2)
    finite_vals = c2[finite_mask]
    n_nan = int(np.sum(~finite_mask))

    if finite_vals.size == 0:
        logger.warning("Dataset contains no finite values")
        return {
            "mean": 0.0,
            "std": 0.0,
            "snr": 0.0,
            "n_nan": n_nan,
            "dynamic_range": 0.0,
            "effective_rank": 0,
        }

    mean_val = float(np.mean(finite_vals))
    std_val = float(np.std(finite_vals))
    snr = abs(mean_val) / std_val if std_val > 0 else 0.0

    abs_finite = np.abs(finite_vals)
    nonzero = abs_finite[abs_finite > 0]
    if nonzero.size > 0:
        dynamic_range = float(np.max(nonzero) / np.min(nonzero))
    else:
        dynamic_range = 0.0

    # Effective rank via SVD
    c2_clean = np.where(finite_mask, c2, 0.0)
    try:
        sv = np.linalg.svd(c2_clean, compute_uv=False)
        if sv[0] > 0:
            effective_rank = int(np.sum(sv / sv[0] > 1e-6))
        else:
            effective_rank = 0
    except np.linalg.LinAlgError:
        logger.warning("SVD failed during effective rank computation")
        effective_rank = 0

    return {
        "mean": mean_val,
        "std": std_val,
        "snr": snr,
        "n_nan": n_nan,
        "dynamic_range": dynamic_range,
        "effective_rank": effective_rank,
    }


def recommend_strategy(
    c2: np.ndarray,
    t: np.ndarray,
) -> dict[str, object]:
    """Recommend fitting strategy based on dataset characteristics.

    Analyses the size, quality, and structure of *c2* and returns
    recommended settings for the downstream fitting pipeline.

    Args:
        c2: Two-time correlation matrix, shape ``(n_t, n_t)``.
        t: 1-D time array.

    Returns:
        Dictionary with keys:
        - ``use_upper_triangle`` (bool): Whether to fit upper triangle only.
        - ``exclude_diagonal`` (bool): Whether to exclude the diagonal.
        - ``suggested_weight_method`` (str): One of ``'uniform'``,
          ``'variance'``, ``'snr'``.
        - ``suggested_chunk_size`` (int | None): Recommended chunk size
          for batched evaluation, or ``None`` for single-batch.
    """
    n = c2.shape[0] if c2.ndim >= 1 else 0
    stats = compute_dataset_statistics(c2, t)

    # --- use_upper_triangle ---
    # For square matrices, fitting only the upper triangle halves the work
    # and avoids double-counting symmetric data.
    use_upper_triangle: bool = c2.ndim == 2 and c2.shape[0] == c2.shape[1]

    # --- exclude_diagonal ---
    # Diagonal elements often have excess variance from shot noise; exclude
    # when the diagonal/off-diagonal ratio is large.
    exclude_diagonal = False
    if c2.ndim == 2 and c2.shape[0] == c2.shape[1] and n > 1:
        diag_vals = np.diag(c2)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_vals = c2[off_diag_mask]
        finite_diag = diag_vals[np.isfinite(diag_vals)]
        finite_off = off_vals[np.isfinite(off_vals)]
        if finite_diag.size > 0 and finite_off.size > 0:
            ratio = abs(float(np.mean(finite_diag))) / max(
                abs(float(np.mean(finite_off))), 1e-15
            )
            exclude_diagonal = ratio > 2.0

    # --- suggested_weight_method ---
    snr = stats["snr"]
    if isinstance(snr, (int, float)) and snr >= 10.0:
        suggested_weight_method = "uniform"
    elif isinstance(snr, (int, float)) and snr >= 3.0:
        suggested_weight_method = "variance"
    else:
        suggested_weight_method = "snr"

    # --- suggested_chunk_size ---
    n_elements = n * n
    if n_elements > 1_000_000:
        suggested_chunk_size: int | None = max(n // 4, 64)
    elif n_elements > 100_000:
        suggested_chunk_size = max(n // 2, 64)
    else:
        suggested_chunk_size = None

    strategy: dict[str, object] = {
        "use_upper_triangle": use_upper_triangle,
        "exclude_diagonal": exclude_diagonal,
        "suggested_weight_method": suggested_weight_method,
        "suggested_chunk_size": suggested_chunk_size,
    }

    logger.info("Recommended strategy: %s", strategy)
    return strategy


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _subsample_uniform(n: int, max_points: int) -> np.ndarray:
    """Select every N-th index for uniform subsampling."""
    step = max(1, n // max_points)
    return np.arange(0, n, step)[:max_points]


def _subsample_random(n: int, max_points: int, seed: int | None) -> np.ndarray:
    """Randomly select *max_points* indices without replacement."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=min(max_points, n), replace=False)
    indices.sort()
    return indices


def _subsample_adaptive(n: int, max_points: int, seed: int | None) -> np.ndarray:
    """Adaptive subsampling that is denser near the diagonal.

    Points close to the diagonal (small lag) carry more signal for
    typical XPCS decay profiles, so we allocate more samples there.
    We split the budget: 60% for the first quarter of the index range,
    40% for the rest, then merge and deduplicate.
    """
    rng = np.random.default_rng(seed)

    quarter = max(1, n // 4)
    budget_near = int(0.6 * max_points)
    budget_far = max_points - budget_near

    near_indices = rng.choice(quarter, size=min(budget_near, quarter), replace=False)
    far_indices = rng.choice(
        np.arange(quarter, n),
        size=min(budget_far, n - quarter),
        replace=False,
    )

    indices = np.unique(np.concatenate([near_indices, far_indices]))
    indices.sort()

    # Trim to max_points if deduplication didn't reduce enough
    if len(indices) > max_points:
        indices = indices[:max_points]

    return indices
