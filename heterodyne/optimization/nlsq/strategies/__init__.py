"""NLSQ fitting strategies for heterodyne model optimization.

Strategies determine how the residual function is evaluated:
- ResidualStrategy: Direct residual evaluation (default)
- JITStrategy: JAX JIT-compiled residual + Jacobian
- ChunkedStrategy: Memory-efficient chunked evaluation for large datasets
- SequentialStrategy: Per-angle sequential fitting with warm-start
- HybridStreamingStrategy: 4-phase hybrid optimizer (L-BFGS + Gauss-Newton)
- OutOfCoreStrategy: Memory-mapped evaluation for very large datasets
- StratifiedLSStrategy: Stratified sampling across q-point subsets
- ResidualJITStrategy: JIT-compiled residual with FD Jacobian
"""

from __future__ import annotations

from heterodyne.optimization.nlsq.strategies.base import (
    FittingStrategy,
    StrategyResult,
    select_strategy,
)
from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy
from heterodyne.optimization.nlsq.strategies.hybrid_streaming import (
    HybridStreamingStrategy,
)
from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy
from heterodyne.optimization.nlsq.strategies.out_of_core import OutOfCoreStrategy
from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy
from heterodyne.optimization.nlsq.strategies.residual_jit import ResidualJITStrategy
from heterodyne.optimization.nlsq.strategies.sequential import SequentialStrategy
from heterodyne.optimization.nlsq.strategies.stratified_ls import StratifiedLSStrategy

__all__ = [
    "FittingStrategy",
    "StrategyResult",
    "select_strategy",
    "ResidualStrategy",
    "JITStrategy",
    "ChunkedStrategy",
    "SequentialStrategy",
    "HybridStreamingStrategy",
    "OutOfCoreStrategy",
    "StratifiedLSStrategy",
    "ResidualJITStrategy",
]
