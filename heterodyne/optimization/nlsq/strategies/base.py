"""Base strategy interface and strategy selection logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


@dataclass
class StrategyResult:
    """Result from a fitting strategy execution.

    Wraps the NLSQResult with strategy-specific metadata.
    """

    result: NLSQResult
    strategy_name: str
    n_chunks: int = 1
    peak_memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class FittingStrategy(Protocol):
    """Protocol for NLSQ fitting strategies."""

    @property
    def name(self) -> str:
        """Strategy name for logging."""
        ...

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Execute the fitting strategy.

        Args:
            model: Configured HeterodyneModel
            c2_data: Correlation data, shape (N, N)
            phi_angle: Detector phi angle (degrees)
            config: NLSQ configuration
            weights: Optional weights (1/sigma²)

        Returns:
            StrategyResult wrapping the NLSQResult
        """
        ...


# Memory thresholds for strategy selection (in elements)
_SMALL_DATASET = 100 * 100  # 10K elements
_MEDIUM_DATASET = 500 * 500  # 250K elements
_LARGE_DATASET = 2000 * 2000  # 4M elements


def select_strategy(
    n_data: int,
    n_params: int,
    config: NLSQConfig,
    *,
    available_memory_gb: float | None = None,
) -> FittingStrategy:
    """Select optimal fitting strategy based on data size and resources.

    Strategy selection logic:
    - Small datasets (< 10K): Direct residual evaluation
    - Medium datasets (< 250K): JIT-compiled with analytic Jacobian
    - Large datasets (< 4M): Chunked evaluation
    - Very large datasets (>= 4M): Chunked with explicit chunk_size

    Args:
        n_data: Total number of data points
        n_params: Number of varying parameters
        config: NLSQ configuration (may override strategy)
        available_memory_gb: Available memory in GB (auto-detected if None)

    Returns:
        Selected FittingStrategy
    """
    from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy
    from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy
    from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

    # Explicit chunk_size in config forces chunked strategy
    if config.chunk_size is not None:
        logger.info(f"Using chunked strategy (explicit chunk_size={config.chunk_size})")
        return ChunkedStrategy(chunk_size=config.chunk_size)

    # Estimate Jacobian memory: n_data × n_params × 8 bytes (float64)
    jacobian_bytes = n_data * n_params * 8
    jacobian_mb = jacobian_bytes / (1024 * 1024)

    if n_data < _SMALL_DATASET:
        logger.info("Using residual strategy (small dataset: %d points)", n_data)
        return ResidualStrategy()

    if n_data < _MEDIUM_DATASET:
        logger.info(
            f"Using JIT strategy (medium dataset: {n_data} points, "
            f"Jacobian ~{jacobian_mb:.0f} MB)"
        )
        return JITStrategy()

    # Large dataset: chunk to fit in memory
    if available_memory_gb is None:
        available_memory_gb = _estimate_available_memory()

    # Target: Jacobian chunk fits in 25% of available memory
    target_bytes = available_memory_gb * 1024**3 * 0.25
    chunk_size = max(int(target_bytes / (n_params * 8)), 1000)
    chunk_size = min(chunk_size, n_data)  # Don't exceed data size

    logger.info(
        f"Using chunked strategy (large dataset: {n_data} points, "
        f"chunk_size={chunk_size}, Jacobian ~{jacobian_mb:.0f} MB)"
    )
    return ChunkedStrategy(chunk_size=chunk_size)


def _estimate_available_memory() -> float:
    """Estimate available system memory in GB."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return float(mem.available) / (1024**3)
    except ImportError:
        # Conservative default: 4 GB
        return 4.0
