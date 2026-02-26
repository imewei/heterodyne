"""Memory-aware strategy selection for NLSQ optimization.

Estimates peak memory usage from Jacobian size and selects between
standard (in-memory) and out-of-core (chunked) strategies.

Adapted from homodyne/optimization/nlsq/memory.py, simplified for
heterodyne's single-phi architecture (no streaming infrastructure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


class NLSQStrategy(Enum):
    """NLSQ optimization strategy based on memory constraints."""

    STANDARD = "standard"
    OUT_OF_CORE = "out_of_core"


@dataclass(frozen=True)
class StrategyDecision:
    """Result of strategy selection.

    Attributes:
        strategy: Selected optimization strategy.
        threshold_gb: Memory threshold used for decision.
        peak_memory_gb: Estimated peak memory usage.
        reason: Human-readable explanation.
    """

    strategy: NLSQStrategy
    threshold_gb: float
    peak_memory_gb: float
    reason: str


def detect_total_system_memory() -> float | None:
    """Detect total system memory in GB.

    Tries psutil first, then os.sysconf, returns None on failure.

    Returns:
        Total memory in GB or None if detection fails.
    """
    # Try psutil
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except (ImportError, AttributeError):
        pass

    # Try os.sysconf (Linux/macOS)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass

    return None


def get_adaptive_memory_threshold(memory_fraction: float = 0.75) -> float:
    """Compute memory threshold in GB for strategy selection.

    Uses environment variable HETERODYNE_MEMORY_FRACTION if set,
    otherwise uses memory_fraction of detected system memory.

    Args:
        memory_fraction: Fraction of system memory to use (0.1-0.9).

    Returns:
        Memory threshold in GB. Defaults to 4.0 if detection fails.
    """
    # Check environment override
    env_fraction = os.environ.get("HETERODYNE_MEMORY_FRACTION")
    if env_fraction is not None:
        try:
            memory_fraction = float(env_fraction)
        except ValueError:
            logger.warning(
                f"Invalid HETERODYNE_MEMORY_FRACTION={env_fraction!r}, "
                f"using default={memory_fraction}"
            )

    # Clamp to valid range
    memory_fraction = max(0.1, min(0.9, memory_fraction))

    total_memory = detect_total_system_memory()
    if total_memory is None:
        logger.debug("Could not detect system memory, using 4.0 GB threshold")
        return 4.0

    threshold = total_memory * memory_fraction
    logger.debug(
        f"System memory: {total_memory:.1f} GB, "
        f"threshold: {threshold:.1f} GB ({memory_fraction:.0%})"
    )
    return threshold


def estimate_peak_memory_gb(n_points: int, n_params: int) -> float:
    """Estimate peak memory usage for NLSQ optimization.

    The dominant cost is the Jacobian matrix (n_points × n_params)
    plus working copies. Heterodyne uses a 4.0× overhead factor
    (lower than homodyne's 6.5× since there's no multi-angle padding).

    Args:
        n_points: Number of data points (residual vector length).
        n_params: Number of varying parameters.

    Returns:
        Estimated peak memory in GB.
    """
    bytes_per_float64 = 8
    jacobian_bytes = n_points * n_params * bytes_per_float64
    overhead_factor = 4.0  # Jacobian + JtJ + workspace + margin
    total_bytes = jacobian_bytes * overhead_factor
    return total_bytes / (1024**3)


def select_nlsq_strategy(
    n_points: int,
    n_params: int,
    memory_fraction: float = 0.75,
) -> StrategyDecision:
    """Select NLSQ strategy based on estimated memory usage.

    Compares estimated peak memory against the adaptive threshold.
    If peak memory exceeds the threshold, recommends chunked
    (out-of-core) processing.

    Args:
        n_points: Number of data points.
        n_params: Number of varying parameters.
        memory_fraction: Fraction of system memory to use.

    Returns:
        StrategyDecision with selected strategy and rationale.
    """
    threshold = get_adaptive_memory_threshold(memory_fraction)
    peak = estimate_peak_memory_gb(n_points, n_params)

    if peak > threshold:
        strategy = NLSQStrategy.OUT_OF_CORE
        reason = (
            f"Peak memory {peak:.2f} GB exceeds threshold {threshold:.2f} GB; "
            f"using chunked processing"
        )
        logger.info(reason)
    else:
        strategy = NLSQStrategy.STANDARD
        reason = (
            f"Peak memory {peak:.2f} GB within threshold {threshold:.2f} GB; "
            f"using standard in-memory processing"
        )
        logger.debug(reason)

    return StrategyDecision(
        strategy=strategy,
        threshold_gb=threshold,
        peak_memory_gb=peak,
        reason=reason,
    )
