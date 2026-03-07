"""Optimization progress tracking for NLSQ fitting.

Provides callback-compatible progress recording so that scipy (or other)
optimizers can report per-iteration cost, gradient, and step information.
Supports stall detection and summary generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressRecord:
    """Single iteration progress snapshot.

    Attributes:
        iteration: Iteration number (0-based).
        cost: Cost (sum of squared residuals) at this iteration.
        cost_change: Change in cost from the previous iteration
            (``None`` for the first record).
        gradient_norm: L2 norm of the gradient (``None`` if unavailable).
        step_norm: L2 norm of the parameter step (``None`` if unavailable).
        wall_time: Wall-clock time in seconds since tracking started.
    """

    iteration: int
    cost: float
    cost_change: float | None = None
    gradient_norm: float | None = None
    step_norm: float | None = None
    wall_time: float = 0.0


class ProgressTracker:
    """Track optimization progress across iterations.

    Designed for callback-style integration with scipy optimizers::

        tracker = ProgressTracker()
        def callback(xk):
            cost = float(np.sum(residual_fn(xk) ** 2))
            tracker.record(tracker.n_records, cost, xk)
        scipy.optimize.least_squares(..., callback=callback)
        print(tracker.summary())

    Parameters
    ----------
    parameter_names : list[str] | None
        Optional parameter names for diagnostic output.
    """

    def __init__(self, parameter_names: list[str] | None = None) -> None:
        self._parameter_names = parameter_names
        self._history: list[ProgressRecord] = []
        self._start_time: float | None = None
        self._prev_params: np.ndarray | None = None

    @property
    def n_records(self) -> int:
        """Number of recorded iterations."""
        return len(self._history)

    def record(
        self,
        iteration: int,
        cost: float,
        params: np.ndarray | None = None,
        gradient: np.ndarray | None = None,
    ) -> None:
        """Record one iteration of optimization progress.

        Args:
            iteration: Current iteration number.
            cost: Current cost value (sum of squared residuals).
            params: Current parameter vector (for step-norm calculation).
            gradient: Current gradient vector (for gradient-norm calculation).
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()

        wall_time = time.perf_counter() - self._start_time

        # Cost change
        cost_change: float | None = None
        if self._history:
            cost_change = cost - self._history[-1].cost

        # Gradient norm
        gradient_norm: float | None = None
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=np.float64).ravel()
            gradient_norm = float(np.linalg.norm(gradient))

        # Step norm
        step_norm: float | None = None
        if params is not None:
            params = np.asarray(params, dtype=np.float64).ravel()
            if self._prev_params is not None:
                step = params - self._prev_params
                step_norm = float(np.linalg.norm(step))
            self._prev_params = params.copy()

        record = ProgressRecord(
            iteration=iteration,
            cost=cost,
            cost_change=cost_change,
            gradient_norm=gradient_norm,
            step_norm=step_norm,
            wall_time=wall_time,
        )
        self._history.append(record)

    def is_stalled(
        self,
        patience: int = 10,
        min_improvement: float = 1e-10,
    ) -> bool:
        """Check whether optimization progress has stalled.

        Stall is declared when the last *patience* iterations all have
        absolute cost changes smaller than *min_improvement*.

        Args:
            patience: Number of recent iterations to examine.
            min_improvement: Minimum absolute cost decrease per iteration.

        Returns:
            ``True`` if stalled.
        """
        if len(self._history) < patience + 1:
            return False

        recent = self._history[-patience:]
        for rec in recent:
            if rec.cost_change is not None and abs(rec.cost_change) >= min_improvement:
                return False
        return True

    def get_history(self) -> list[ProgressRecord]:
        """Return a copy of the full progress history.

        Returns:
            List of :class:`ProgressRecord` objects.
        """
        return list(self._history)

    def summary(self) -> str:
        """Generate a human-readable progress summary.

        Returns:
            Multi-line summary string.
        """
        if not self._history:
            return "ProgressTracker: no iterations recorded."

        first = self._history[0]
        last = self._history[-1]

        lines = [
            "Optimization Progress",
            "=" * 50,
            f"  Iterations recorded:  {len(self._history)}",
            f"  Initial cost:         {first.cost:.6e}",
            f"  Final cost:           {last.cost:.6e}",
        ]

        if first.cost > 0:
            reduction = (first.cost - last.cost) / first.cost * 100
            lines.append(f"  Cost reduction:       {reduction:.2f}%")

        lines.append(f"  Total wall time:      {last.wall_time:.3f} s")

        if last.gradient_norm is not None:
            lines.append(f"  Final gradient norm:  {last.gradient_norm:.3e}")

        if last.step_norm is not None:
            lines.append(f"  Final step norm:      {last.step_norm:.3e}")

        # Stall check
        if self.is_stalled():
            lines.append("  WARNING: optimization appears stalled")

        return "\n".join(lines)
