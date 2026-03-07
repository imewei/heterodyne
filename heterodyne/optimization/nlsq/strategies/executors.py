"""Strategy executor pattern for NLSQ optimization.

Provides executor wrappers that manage pre/post-processing around
fitting strategies, including timing, diagnostics, and error recovery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.strategies.base import (
    FittingStrategy,
    StrategyResult,
    select_strategy,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result from an executor run.

    Wraps StrategyResult with execution-level metadata including
    timing, recovery actions, and convergence status.
    """

    strategy_result: StrategyResult
    wall_time: float
    recovery_actions: list[dict[str, Any]] = field(default_factory=list)
    convergence_status: str = "unknown"
    executor_name: str = ""

    @property
    def result(self) -> NLSQResult:
        """Underlying NLSQResult."""
        return self.strategy_result.result

    @property
    def success(self) -> bool:
        return self.strategy_result.result.success


class OptimizationExecutor:
    """Base executor wrapping a FittingStrategy with lifecycle management.

    Handles:
    - Strategy selection (auto or explicit)
    - Pre-execution validation
    - Timing
    - Post-execution diagnostics
    - Error recovery fallback
    """

    def __init__(
        self,
        strategy: FittingStrategy | None = None,
        *,
        enable_diagnostics: bool = False,
        enable_recovery: bool = True,
        max_recovery_attempts: int = 2,
    ) -> None:
        self._strategy = strategy
        self._enable_diagnostics = enable_diagnostics
        self._enable_recovery = enable_recovery
        self._max_recovery_attempts = max_recovery_attempts

    @property
    def name(self) -> str:
        if self._strategy is not None:
            return f"executor({self._strategy.name})"
        return "executor(auto)"

    def execute(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> ExecutionResult:
        """Execute the fitting strategy with lifecycle management.

        Args:
            model: Configured HeterodyneModel.
            c2_data: Correlation data array.
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights (1/sigma²).

        Returns:
            ExecutionResult containing the StrategyResult and metadata.
        """
        t_start = time.perf_counter()

        # Auto-select strategy if not provided
        strategy = self._strategy
        if strategy is None:
            n_data = np.asarray(c2_data).size
            n_params = model.param_manager.n_varying
            strategy = select_strategy(n_data, n_params, config)

        logger.info("Executor: running strategy '%s'", strategy.name)

        recovery_actions: list[dict[str, Any]] = []
        last_error: Exception | None = None

        for attempt in range(1 + self._max_recovery_attempts):
            try:
                strategy_result = strategy.fit(
                    model=model,
                    c2_data=c2_data,
                    phi_angle=phi_angle,
                    config=config,
                    weights=weights,
                )

                wall_time = time.perf_counter() - t_start
                convergence_status = (
                    "converged" if strategy_result.result.success else "failed"
                )

                if self._enable_diagnostics:
                    self._log_diagnostics(strategy_result, wall_time)

                return ExecutionResult(
                    strategy_result=strategy_result,
                    wall_time=wall_time,
                    recovery_actions=recovery_actions,
                    convergence_status=convergence_status,
                    executor_name=self.name,
                )

            except (RuntimeError, ValueError, MemoryError) as exc:
                last_error = exc
                if not self._enable_recovery or attempt >= self._max_recovery_attempts:
                    break

                logger.warning(
                    "Strategy '%s' failed (attempt %d/%d): %s",
                    strategy.name,
                    attempt + 1,
                    1 + self._max_recovery_attempts,
                    exc,
                )
                recovery_actions.append(
                    {
                        "attempt": attempt,
                        "error": str(exc),
                        "strategy": strategy.name,
                    }
                )

                strategy = self._fallback_strategy(strategy, config, c2_data, model)

        # All attempts exhausted
        wall_time = time.perf_counter() - t_start
        logger.error(
            "All execution attempts failed after %d tries: %s",
            1 + self._max_recovery_attempts,
            last_error,
        )

        from heterodyne.optimization.nlsq.results import NLSQResult

        failed_result = NLSQResult(
            parameters=model.param_manager.get_initial_values(),
            parameter_names=list(model.param_manager.varying_names),
            success=False,
            message=f"All strategies failed: {last_error}",
        )
        return ExecutionResult(
            strategy_result=StrategyResult(
                result=failed_result,
                strategy_name="failed",
            ),
            wall_time=wall_time,
            recovery_actions=recovery_actions,
            convergence_status="failed",
            executor_name=self.name,
        )

    def _fallback_strategy(
        self,
        current: FittingStrategy,
        config: NLSQConfig,
        c2_data: np.ndarray,
        model: HeterodyneModel,
    ) -> FittingStrategy:
        """Select a simpler fallback strategy.

        Always falls back to ResidualStrategy, which has no external
        dependencies and is guaranteed to run on any dataset size.
        """
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        logger.info(
            "Falling back from '%s' to ResidualStrategy",
            current.name,
        )
        return ResidualStrategy()

    def _log_diagnostics(
        self,
        strategy_result: StrategyResult,
        wall_time: float,
    ) -> None:
        """Log execution diagnostics."""
        result = strategy_result.result
        logger.info(
            "Executor diagnostics: strategy=%s success=%s cost=%.4e "
            "time=%.2fs chunks=%d peak_mem=%.1fMB",
            strategy_result.strategy_name,
            result.success,
            result.final_cost if result.final_cost is not None else float("nan"),
            wall_time,
            strategy_result.n_chunks,
            strategy_result.peak_memory_mb,
        )


class StandardExecutor(OptimizationExecutor):
    """Executor for standard (small-to-medium) datasets.

    Uses auto-selected strategy (residual or JIT) with diagnostics
    and recovery enabled.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            enable_diagnostics=True,
            enable_recovery=True,
            **kwargs,
        )


class LargeDatasetExecutor(OptimizationExecutor):
    """Executor for large datasets (>250K points).

    Forces ChunkedStrategy with extended recovery attempts to handle
    transient memory pressure.
    """

    def __init__(self, chunk_size: int = 50_000, **kwargs: Any) -> None:
        from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy

        super().__init__(
            strategy=ChunkedStrategy(chunk_size=chunk_size),
            enable_diagnostics=True,
            enable_recovery=True,
            max_recovery_attempts=3,
            **kwargs,
        )


# Keys consumed by OptimizationExecutor.__init__ — not forwarded to strategies.
_EXECUTOR_KWARGS: frozenset[str] = frozenset(
    {"enable_diagnostics", "enable_recovery", "max_recovery_attempts"}
)


def get_executor(
    strategy_name: str | None = None,
    n_data: int = 0,
    **kwargs: Any,
) -> OptimizationExecutor:
    """Factory function for OptimizationExecutor instances.

    Separates executor-level keyword arguments from strategy-level ones so
    that executor options (``enable_diagnostics``, ``enable_recovery``,
    ``max_recovery_attempts``) are never accidentally forwarded to strategy
    constructors, which do not accept them.

    Args:
        strategy_name: Explicit strategy name or None for auto-selection.
            Recognised names: ``"standard"``, ``"large"``, ``"chunked"``,
            ``"jit"``, ``"residual"``, ``"sequential"``.
        n_data: Dataset size used for auto-selection when
            ``strategy_name`` is None.
        **kwargs: Mixed executor and strategy keyword arguments.
            Executor keys (``enable_diagnostics``, ``enable_recovery``,
            ``max_recovery_attempts``) are routed to the executor; all
            remaining keys are forwarded to the strategy constructor.

    Returns:
        Configured OptimizationExecutor ready to call ``.execute()``.
    """
    executor_kwargs = {k: v for k, v in kwargs.items() if k in _EXECUTOR_KWARGS}
    strategy_kwargs = {k: v for k, v in kwargs.items() if k not in _EXECUTOR_KWARGS}

    # Size-based auto-selection
    if strategy_name is None:
        if n_data > 250_000:
            chunk_size = strategy_kwargs.pop("chunk_size", 50_000)
            return LargeDatasetExecutor(chunk_size=chunk_size, **executor_kwargs)
        return StandardExecutor(**executor_kwargs)

    if strategy_name == "large":
        chunk_size = strategy_kwargs.pop("chunk_size", 50_000)
        return LargeDatasetExecutor(chunk_size=chunk_size, **executor_kwargs)

    if strategy_name == "standard":
        return StandardExecutor(**executor_kwargs)

    if strategy_name == "chunked":
        from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy

        return OptimizationExecutor(
            strategy=ChunkedStrategy(**strategy_kwargs),
            enable_recovery=True,
            **executor_kwargs,
        )

    if strategy_name == "jit":
        from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy

        return OptimizationExecutor(
            strategy=JITStrategy(**strategy_kwargs),
            enable_recovery=True,
            **executor_kwargs,
        )

    if strategy_name == "residual":
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        return OptimizationExecutor(
            strategy=ResidualStrategy(**strategy_kwargs),
            enable_recovery=True,
            **executor_kwargs,
        )

    if strategy_name == "sequential":
        from heterodyne.optimization.nlsq.strategies.sequential import (
            SequentialStrategy,
        )

        return OptimizationExecutor(
            strategy=SequentialStrategy(**strategy_kwargs),
            enable_recovery=True,
            **executor_kwargs,
        )

    logger.warning(
        "Unknown strategy name '%s'; falling back to auto-selection", strategy_name
    )
    return StandardExecutor(**executor_kwargs)
