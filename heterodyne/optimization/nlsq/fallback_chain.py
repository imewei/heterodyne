"""Fallback chain for NLSQ optimization strategies.

Provides automatic strategy degradation when the primary fitting strategy
fails due to memory constraints, numerical errors, or convergence issues.
The chain tries strategies in order of decreasing performance until one
succeeds or all options are exhausted.

Strategy order:
1. STANDARD (ResidualStrategy) — direct evaluation, fastest for small data
2. LARGE (JITStrategy) — JIT-compiled, good for medium datasets
3. CHUNKED (ChunkedStrategy) — memory-bounded, handles large datasets
4. STREAMING (SequentialStrategy) — per-angle sequential, most robust
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.optimization.nlsq.strategies.base import FittingStrategy, StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies in fallback order."""

    STANDARD = auto()
    LARGE = auto()
    CHUNKED = auto()
    STREAMING = auto()


# Strategy fallback ordering
_FALLBACK_ORDER: list[OptimizationStrategy] = [
    OptimizationStrategy.STANDARD,
    OptimizationStrategy.LARGE,
    OptimizationStrategy.CHUNKED,
    OptimizationStrategy.STREAMING,
]


def _create_strategy(strategy: OptimizationStrategy) -> FittingStrategy:
    """Instantiate a fitting strategy by enum value.

    Args:
        strategy: Strategy to instantiate.

    Returns:
        FittingStrategy instance.
    """
    from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy
    from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy
    from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy
    from heterodyne.optimization.nlsq.strategies.sequential import SequentialStrategy

    strategy_map = {
        OptimizationStrategy.STANDARD: ResidualStrategy,
        OptimizationStrategy.LARGE: JITStrategy,
        OptimizationStrategy.CHUNKED: ChunkedStrategy,
        OptimizationStrategy.STREAMING: SequentialStrategy,
    }

    cls = strategy_map[strategy]
    return cls()  # type: ignore[no-any-return]


def get_fallback_strategy(
    current: OptimizationStrategy,
    error: Exception | None = None,
) -> OptimizationStrategy | None:
    """Determine the next fallback strategy after a failure.

    Args:
        current: The strategy that just failed.
        error: The exception that caused the failure (used for
            heuristic selection).

    Returns:
        Next strategy to try, or None if all options exhausted.
    """
    current_idx = _FALLBACK_ORDER.index(current)

    # Memory errors skip directly to CHUNKED
    if error is not None and _is_memory_error(error):
        for candidate in _FALLBACK_ORDER[current_idx + 1:]:
            if candidate in (OptimizationStrategy.CHUNKED, OptimizationStrategy.STREAMING):
                logger.info(
                    "Memory error detected; skipping to %s strategy",
                    candidate.name,
                )
                return candidate
        return None

    # Default: try the next strategy in order
    next_idx = current_idx + 1
    if next_idx < len(_FALLBACK_ORDER):
        return _FALLBACK_ORDER[next_idx]

    return None


def _is_memory_error(error: Exception) -> bool:
    """Check if an error is memory-related."""
    error_str = str(error).lower()
    memory_keywords = ["memory", "oom", "out of memory", "allocate", "mmap"]
    return (
        isinstance(error, MemoryError)
        or any(kw in error_str for kw in memory_keywords)
    )


def execute_optimization_with_fallback(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    *,
    start_strategy: OptimizationStrategy | None = None,
    weights: np.ndarray | None = None,
) -> StrategyResult:
    """Execute optimization with automatic fallback on failure.

    Tries strategies in order, catching failures and degrading gracefully
    to the next available strategy.

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data, shape (N, N).
        phi_angle: Detector phi angle in degrees.
        config: NLSQ configuration.
        start_strategy: Initial strategy to try. If None, auto-selects
            based on data size.
        weights: Optional per-point weights.

    Returns:
        StrategyResult from the first successful strategy.

    Raises:
        RuntimeError: If all strategies fail.
    """
    if start_strategy is None:
        n_data = c2_data.size
        if n_data < 10_000:
            start_strategy = OptimizationStrategy.STANDARD
        elif n_data < 250_000:
            start_strategy = OptimizationStrategy.LARGE
        else:
            start_strategy = OptimizationStrategy.CHUNKED

    current: OptimizationStrategy | None = start_strategy
    errors: list[tuple[str, str]] = []

    while current is not None:
        strategy_instance = _create_strategy(current)
        logger.info(
            "Fallback chain: attempting %s strategy (%s)",
            current.name,
            strategy_instance.name,
        )

        try:
            result = strategy_instance.fit(
                model, c2_data, phi_angle, config, weights
            )

            if result.result.success:
                logger.info(
                    "Fallback chain: %s succeeded (cost=%.4e)",
                    current.name,
                    result.result.final_cost or 0.0,
                )
                result.metadata["fallback_chain"] = {
                    "strategy_used": current.name,
                    "attempts": len(errors) + 1,
                    "failed_strategies": [e[0] for e in errors],
                }
                return result

            # Strategy ran but didn't converge — current is still non-None
            # because get_fallback_strategy hasn't been called yet.
            assert current is not None  # narrowing for mypy
            logger.warning(
                "Fallback chain: %s did not converge: %s",
                current.name,
                result.result.message,
            )
            errors.append((current.name, result.result.message))
            current = get_fallback_strategy(current)

        except Exception as exc:  # noqa: BLE001
            # Exception occurs before current is reassigned in the try block,
            # so current is guaranteed non-None here.
            assert current is not None  # narrowing for mypy
            logger.warning(
                "Fallback chain: %s raised %s: %s",
                current.name,
                type(exc).__name__,
                exc,
            )
            errors.append((current.name, f"{type(exc).__name__}: {exc}"))
            current = get_fallback_strategy(current, error=exc)

    # All strategies exhausted
    error_summary = "; ".join(f"{name}: {msg}" for name, msg in errors)
    raise RuntimeError(
        f"All optimization strategies failed. Errors: {error_summary}"
    )
