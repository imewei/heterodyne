"""NLSQ fitting strategies for heterodyne model optimization.

Strategies determine how the residual function is evaluated:
- ResidualStrategy: Direct residual evaluation (default)
- JITStrategy: JAX JIT-compiled residual + Jacobian
- ChunkedStrategy: Memory-efficient chunked evaluation for large datasets
- SequentialStrategy: Per-angle sequential fitting with warm-start
"""

from __future__ import annotations

from heterodyne.optimization.nlsq.strategies.base import (
    FittingStrategy,
    StrategyResult,
    select_strategy,
)
from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy
from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy
from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy
from heterodyne.optimization.nlsq.strategies.sequential import SequentialStrategy

__all__ = [
    "FittingStrategy",
    "StrategyResult",
    "select_strategy",
    "ResidualStrategy",
    "JITStrategy",
    "ChunkedStrategy",
    "SequentialStrategy",
]
