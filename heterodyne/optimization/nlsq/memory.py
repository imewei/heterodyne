"""Memory-aware strategy selection for NLSQ optimization.

Estimates peak memory usage from Jacobian size and selects between
standard (in-memory), large (chunked J^T J), and streaming (L-BFGS
warmup + streaming Gauss-Newton) strategies.

Strategy decision tree:
    1. Index array alone > threshold  ->  STREAMING  (extreme scale)
    2. Peak Jacobian memory > threshold  ->  LARGE  (out-of-core chunks)
    3. Otherwise  ->  STANDARD  (full in-memory Jacobian)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

from heterodyne.utils.logging import get_logger, log_phase

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_FRACTION: float = 0.75
"""Fraction of system RAM used as the memory threshold."""

FALLBACK_THRESHOLD_GB: float = 16.0
"""Threshold (GB) when system memory cannot be detected."""

MEMORY_FRACTION_ENV_VAR: str = "HETERODYNE_MEMORY_FRACTION"
"""Environment variable that overrides *memory_fraction*."""

_MIN_FRACTION: float = 0.1
_MAX_FRACTION: float = 0.9
_JACOBIAN_OVERHEAD: float = 6.5
"""Overhead factor: base Jacobian + autodiff intermediates + JIT + workspace."""


# ---------------------------------------------------------------------------
# Strategy enum & decision dataclass
# ---------------------------------------------------------------------------


class NLSQStrategy(Enum):
    """NLSQ optimization strategy based on memory constraints."""

    STANDARD = "standard"
    LARGE = "large"
    STREAMING = "streaming"


@dataclass(frozen=True)
class StrategyDecision:
    """Result of memory-based strategy selection.

    Attributes
    ----------
    strategy : NLSQStrategy
        Selected optimization strategy.
    threshold_gb : float
        Memory threshold used for the decision (GB).
    index_memory_gb : float
        Memory required for int64 index array (GB).
    peak_memory_gb : float
        Estimated peak memory for the full Jacobian (GB).
    reason : str
        Human-readable explanation of the decision.
    """

    strategy: NLSQStrategy
    threshold_gb: float
    index_memory_gb: float
    peak_memory_gb: float
    reason: str


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------


def detect_total_system_memory() -> float | None:
    """Detect total system memory in GB.

    Tries ``psutil`` first, then ``os.sysconf`` (Linux/macOS).

    Returns
    -------
    float | None
        Total memory in GB, or ``None`` if detection fails.
    """
    # Method 1: psutil (preferred, cross-platform)
    try:
        import psutil

        total = psutil.virtual_memory().total
        if total > 0:
            return float(total) / (1024**3)
    except ImportError:
        logger.debug("psutil not available, trying os.sysconf fallback")
    except (OSError, ValueError, AttributeError) as exc:
        logger.debug("psutil memory detection failed: %s", exc)

    # Method 2: os.sysconf (Linux/Unix)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return float(page_size * phys_pages) / (1024**3)
    except (ValueError, OSError, AttributeError) as exc:
        logger.debug("os.sysconf memory detection failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_peak_memory_gb(
    n_points: int,
    n_params: int,
    bytes_per_element: int = 8,
    jacobian_overhead: float = _JACOBIAN_OVERHEAD,
) -> float:
    """Estimate peak memory for full-Jacobian NLSQ optimization.

    The dominant cost is the Jacobian matrix ``(n_points x n_params)``
    multiplied by an overhead factor that accounts for autodiff
    intermediates, JIT compilation buffers, and optimizer workspace.

    Parameters
    ----------
    n_points : int
        Residual vector length.
    n_params : int
        Number of varying parameters.
    bytes_per_element : int
        Bytes per array element (default 8 for float64).
    jacobian_overhead : float
        Multiplicative overhead factor (default 6.5).

    Returns
    -------
    float
        Estimated peak memory in GB.
    """
    jacobian_bytes = n_points * n_params * bytes_per_element
    return (jacobian_bytes * jacobian_overhead) / (1024**3)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def get_adaptive_memory_threshold(
    memory_fraction: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute adaptive memory threshold based on system memory.

    The memory threshold determines when NLSQ switches to streaming mode
    for memory-bounded optimization. Instead of a fixed 16 GB threshold,
    this function computes an adaptive threshold as a fraction of total
    system memory.

    Parameters
    ----------
    memory_fraction : float | None, optional
        Fraction of total system memory to use as threshold (0.1 to 0.9).
        If None, uses:
        1. Environment variable HETERODYNE_MEMORY_FRACTION (if set)
        2. Default value of 0.75 (75% of total memory)

    Returns
    -------
    threshold_gb : float
        Memory threshold in gigabytes.
    info : dict
        Diagnostic information with keys:
        - 'total_memory_gb': Detected total system memory (GB)
        - 'memory_fraction': Fraction used
        - 'source': How the fraction was determined ('argument', 'env', 'default')
        - 'detection_method': How memory was detected ('psutil', 'sysconf', 'fallback')

    Notes
    -----
    - If total memory cannot be detected, falls back to 16.0 GB with a warning.
    - Memory fraction is clamped to [0.1, 0.9] for safety.
    - Environment variable HETERODYNE_MEMORY_FRACTION can override the default.

    Examples
    --------
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> print(f"Threshold: {threshold_gb:.1f} GB")
    """
    info: dict[str, Any] = {}

    # Step 1: Determine memory fraction
    fraction_source = "default"
    effective_fraction = DEFAULT_MEMORY_FRACTION

    if memory_fraction is not None:
        effective_fraction = memory_fraction
        fraction_source = "argument"
    else:
        env_value = os.environ.get(MEMORY_FRACTION_ENV_VAR)
        if env_value is not None:
            try:
                effective_fraction = float(env_value)
                fraction_source = "env"
            except ValueError:
                warnings.warn(
                    f"Invalid {MEMORY_FRACTION_ENV_VAR}='{env_value}', "
                    f"using default {DEFAULT_MEMORY_FRACTION}",
                    UserWarning,
                    stacklevel=2,
                )

    # Step 2: Clamp fraction to safe range
    effective_fraction = max(_MIN_FRACTION, min(_MAX_FRACTION, effective_fraction))

    info["memory_fraction"] = effective_fraction
    info["source"] = fraction_source

    # Step 3: Detect system memory and compute threshold
    total_gb = detect_total_system_memory()
    if total_gb is None:
        logger.warning(
            "Could not detect system memory; using fallback threshold %.1f GB",
            FALLBACK_THRESHOLD_GB,
        )
        info["total_memory_gb"] = None
        info["detection_method"] = "fallback"
        threshold_gb = FALLBACK_THRESHOLD_GB
    else:
        info["total_memory_gb"] = total_gb
        info["detection_method"] = "psutil" if _psutil_available() else "sysconf"
        threshold_gb = total_gb * effective_fraction
        logger.debug(
            "Adaptive memory threshold: %.1f GB (%.0f%% of %.1f GB total)",
            threshold_gb,
            effective_fraction * 100,
            total_gb,
        )

    return threshold_gb, info


def _psutil_available() -> bool:
    """Check if psutil is importable."""
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


def _get_memory_threshold(memory_fraction: float) -> float:
    """Compute memory threshold in GB (legacy internal helper).

    Delegates to :func:`get_adaptive_memory_threshold`.
    """
    threshold_gb, _ = get_adaptive_memory_threshold(memory_fraction)
    return threshold_gb


def select_nlsq_strategy(
    n_points: int,
    n_params: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> StrategyDecision:
    """Select NLSQ strategy based on estimated memory usage.

    Decision tree (evaluated top-down):

    1. **STREAMING** — index array alone exceeds threshold (extreme scale).
    2. **LARGE** — peak Jacobian memory exceeds threshold.
    3. **STANDARD** — everything fits in memory.

    Parameters
    ----------
    n_points : int
        Number of data points.
    n_params : int
        Number of varying parameters.
    memory_fraction : float
        Fraction of system memory to use as threshold (default 0.75).

    Returns
    -------
    StrategyDecision
        Decision with selected strategy and rationale.
    """
    with log_phase("memory_strategy_selection", logger=logger):
        threshold_gb, _ = get_adaptive_memory_threshold(memory_fraction)

        # Index array cost (int64 per point)
        index_memory_gb = (n_points * 8) / (1024**3)

        peak_memory_gb = (
            estimate_peak_memory_gb(n_points, n_params) if n_params > 0 else 0.0
        )

        logger.debug(
            "Memory strategy analysis: n_points=%s, n_params=%d, "
            "index=%.2f GB, peak=%.2f GB, threshold=%.2f GB",
            f"{n_points:,}",
            n_params,
            index_memory_gb,
            peak_memory_gb,
            threshold_gb,
        )

        # 1. Extreme scale — even the index array blows memory
        if index_memory_gb > threshold_gb:
            reason = (
                f"Index array ({index_memory_gb:.2f} GB) exceeds "
                f"threshold ({threshold_gb:.2f} GB)"
            )
            logger.info("Auto-switching to STREAMING: %s", reason)
            return StrategyDecision(
                strategy=NLSQStrategy.STREAMING,
                threshold_gb=threshold_gb,
                index_memory_gb=index_memory_gb,
                peak_memory_gb=peak_memory_gb,
                reason=reason,
            )

        # 2. Large scale — Jacobian doesn't fit
        if peak_memory_gb > threshold_gb:
            reason = (
                f"Peak memory ({peak_memory_gb:.2f} GB) exceeds "
                f"threshold ({threshold_gb:.2f} GB)"
            )
            logger.info("Auto-switching to LARGE: %s", reason)
            return StrategyDecision(
                strategy=NLSQStrategy.LARGE,
                threshold_gb=threshold_gb,
                index_memory_gb=index_memory_gb,
                peak_memory_gb=peak_memory_gb,
                reason=reason,
            )

        # 3. Standard — fits in memory
        reason = (
            f"Memory fits: {peak_memory_gb:.2f} GB < {threshold_gb:.2f} GB threshold"
        )
        logger.debug("Selecting STANDARD: %s", reason)
        return StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=threshold_gb,
            index_memory_gb=index_memory_gb,
            peak_memory_gb=peak_memory_gb,
            reason=reason,
        )


__all__ = [
    "DEFAULT_MEMORY_FRACTION",
    "FALLBACK_THRESHOLD_GB",
    "MEMORY_FRACTION_ENV_VAR",
    "NLSQStrategy",
    "StrategyDecision",
    "detect_total_system_memory",
    "estimate_peak_memory_gb",
    "get_adaptive_memory_threshold",
    "select_nlsq_strategy",
]
