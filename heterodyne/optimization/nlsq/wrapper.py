"""NLSQWrapper: retry-on-failure adapter with progressive recovery.

Wraps either ScipyNLSQAdapter (default) or NLSQAdapter (use_jax=True).
On fit failure, applies progressive parameter perturbation and
regularisation adjustments (HybridRecoveryConfig) before retrying.
Each attempt k uses:

- learning-rate factor : ``lr_decay ** k``
- regularisation factor: ``lambda_growth ** k``
- trust-radius factor  : ``trust_decay ** k``

These scale the tolerance fields (ftol/xtol/gtol) of a per-attempt
NLSQConfig copy so the underlying solver can navigate difficult
loss-function landscapes.

fit_with_jax() mirrors fit() but delegates to NLSQAdapter.fit_jax() for
pure JAX-traced residual functions.

Extended surface
----------------
FunctionEvaluationCounter
    Lightweight counter for function, Jacobian, and gradient evaluations
    with an optional budget cap.

OptimizationDiagnostics
    Per-attempt diagnostic record aggregated across a full fit session.

HybridRecoveryManager
    Stateful manager that orchestrates recovery strategies
    (parameter_perturbation, bound_relaxation, method_switching,
    regularization_adjustment) and tracks success rates.

StreamingOptimizer
    Processes very large residual datasets in sequential chunks,
    accumulating Gauss-Newton normal equations and producing a single
    merged NLSQResult.

NLSQWrapper (extended)
    fit_with_recovery() — full pipeline: initial fit → recovery on
    failure → multi-start fallback.
    fit_streaming()     — streaming fit for large datasets.
    get_optimization_stats() — consolidated metrics dictionary.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from heterodyne.optimization.exceptions import StreamingError
from heterodyne.optimization.nlsq.adapter import NLSQAdapter, ScipyNLSQAdapter
from heterodyne.optimization.nlsq.config import HybridRecoveryConfig, NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional anti-degeneracy support
# ---------------------------------------------------------------------------

try:
    from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
        DegeneracyCheck,
    )

    HAS_ANTI_DEGENERACY = True
except ImportError:
    HAS_ANTI_DEGENERACY = False

# ---------------------------------------------------------------------------
# Recovery strategy literals
# ---------------------------------------------------------------------------

RecoveryStrategy = Literal[
    "parameter_perturbation",
    "bound_relaxation",
    "method_switching",
    "regularization_adjustment",
]

_ALL_RECOVERY_STRATEGIES: tuple[RecoveryStrategy, ...] = (
    "parameter_perturbation",
    "bound_relaxation",
    "method_switching",
    "regularization_adjustment",
)

# ---------------------------------------------------------------------------
# FunctionEvaluationCounter
# ---------------------------------------------------------------------------


@dataclass
class FunctionEvaluationCounter:
    """Track function, Jacobian, and gradient evaluation counts.

    Provides an optional hard budget cap.  When ``budget`` is ``None``
    the counter is unlimited.  Call :meth:`increment_fn`, :meth:`increment_jac`,
    or :meth:`increment_grad` after each evaluation.  The :attr:`budget_exceeded`
    property can be polled by callers to implement early exit.

    Attributes:
        budget: Maximum total evaluations (fn + jac + grad) before
            :attr:`budget_exceeded` becomes ``True``.  ``None`` disables
            the cap.
    """

    budget: int | None = None
    _n_fn: int = field(default=0, init=False, repr=False)
    _n_jac: int = field(default=0, init=False, repr=False)
    _n_grad: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def increment_fn(self, count: int = 1) -> None:
        """Record *count* additional function evaluations.

        Args:
            count: Number of evaluations to add (default: 1).
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        self._n_fn += count

    def increment_jac(self, count: int = 1) -> None:
        """Record *count* additional Jacobian evaluations.

        Args:
            count: Number of evaluations to add (default: 1).
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        self._n_jac += count

    def increment_grad(self, count: int = 1) -> None:
        """Record *count* additional gradient evaluations.

        Args:
            count: Number of evaluations to add (default: 1).
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        self._n_grad += count

    def absorb_result(self, result: NLSQResult) -> None:
        """Absorb evaluation counts from a completed :class:`NLSQResult`.

        Adds :attr:`~NLSQResult.n_function_evals` to the function counter
        and :attr:`~NLSQResult.n_iterations` (used as a Jacobian-eval proxy)
        to the Jacobian counter.

        Args:
            result: Completed fit result.
        """
        self.increment_fn(result.n_function_evals)
        self.increment_jac(result.n_iterations)

    def reset(self) -> None:
        """Reset all counters to zero (budget is preserved)."""
        self._n_fn = 0
        self._n_jac = 0
        self._n_grad = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def n_fn(self) -> int:
        """Total function evaluations recorded."""
        return self._n_fn

    @property
    def n_jac(self) -> int:
        """Total Jacobian evaluations recorded."""
        return self._n_jac

    @property
    def n_grad(self) -> int:
        """Total gradient evaluations recorded."""
        return self._n_grad

    @property
    def total(self) -> int:
        """Sum of all evaluation types."""
        return self._n_fn + self._n_jac + self._n_grad

    @property
    def budget_exceeded(self) -> bool:
        """``True`` when the evaluation budget has been exhausted.

        Always ``False`` when :attr:`budget` is ``None``.
        """
        if self.budget is None:
            return False
        return self.total >= self.budget

    @property
    def budget_remaining(self) -> int | None:
        """Evaluations remaining before budget is exhausted.

        Returns ``None`` when :attr:`budget` is ``None`` (unlimited).
        """
        if self.budget is None:
            return None
        return max(0, self.budget - self.total)

    def __repr__(self) -> str:
        budget_str = (
            f"budget={self.budget} remaining={self.budget_remaining}"
            if self.budget is not None
            else "budget=unlimited"
        )
        return (
            f"FunctionEvaluationCounter("
            f"fn={self._n_fn} jac={self._n_jac} grad={self._n_grad} "
            f"total={self.total} {budget_str})"
        )


# ---------------------------------------------------------------------------
# OptimizationDiagnostics
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    """Diagnostic information for a single optimization attempt.

    Attributes:
        attempt_index: Zero-based attempt number.
        method: Solver method string used (e.g. ``"trf"``).
        n_fn_evals: Function evaluations reported by the solver.
        n_jac_evals: Jacobian evaluations (or iteration count as proxy).
        wall_time: Wall-clock duration of this attempt in seconds.
        final_cost: Final cost reported by the solver (``None`` on hard
            failure before any result was obtained).
        success: Whether the solver declared convergence.
        convergence_reason: Short string from the solver (status code,
            message snippet, etc.).
        recovery_applied: Name of the recovery strategy applied before
            this attempt, or ``None`` for the first attempt.
        degeneracy_detected: Whether a degeneracy check fired after this
            attempt.
    """

    attempt_index: int
    method: str
    n_fn_evals: int
    n_jac_evals: int
    wall_time: float
    final_cost: float | None
    success: bool
    convergence_reason: str
    recovery_applied: RecoveryStrategy | None = None
    degeneracy_detected: bool = False


@dataclass
class OptimizationDiagnostics:
    """Aggregate diagnostics across all attempts in a fit session.

    Attributes:
        parameter_names: Names of the parameters being optimised.
        total_wall_time: Total elapsed time for the entire fit session.
        n_total_fn_evals: Accumulated function evaluations across attempts.
        n_total_jac_evals: Accumulated Jacobian evaluations across attempts.
        attempts: Ordered list of :class:`AttemptRecord` instances.
        final_success: Whether the fit session ended with a successful
            result (either via initial fit or recovery).
        best_cost: Lowest ``final_cost`` seen across all attempts.
        recovery_success_rate: Fraction of recovery attempts (attempts
            after the first) that succeeded.
    """

    parameter_names: list[str]
    total_wall_time: float = 0.0
    n_total_fn_evals: int = 0
    n_total_jac_evals: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_success: bool = False
    best_cost: float | None = None

    def record_attempt(self, record: AttemptRecord) -> None:
        """Append *record* and update aggregate counters.

        Args:
            record: Completed attempt record to incorporate.
        """
        self.attempts.append(record)
        self.n_total_fn_evals += record.n_fn_evals
        self.n_total_jac_evals += record.n_jac_evals

        cost = record.final_cost
        if cost is not None:
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost

    @property
    def n_attempts(self) -> int:
        """Total number of attempts recorded."""
        return len(self.attempts)

    @property
    def n_recovery_attempts(self) -> int:
        """Number of attempts that used a recovery strategy (attempt_index > 0)."""
        return sum(1 for a in self.attempts if a.attempt_index > 0)

    @property
    def recovery_success_rate(self) -> float:
        """Fraction of recovery attempts that succeeded.

        Returns ``0.0`` when there were no recovery attempts.
        """
        n = self.n_recovery_attempts
        if n == 0:
            return 0.0
        succeeded = sum(
            1 for a in self.attempts if a.attempt_index > 0 and a.success
        )
        return succeeded / n

    @property
    def n_degeneracies_detected(self) -> int:
        """Total number of degeneracy flags raised across all attempts."""
        return sum(1 for a in self.attempts if a.degeneracy_detected)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for logging or JSON export.

        Returns:
            Flat dictionary with aggregate metrics and per-attempt detail.
        """
        return {
            "parameter_names": self.parameter_names,
            "total_wall_time": self.total_wall_time,
            "n_total_fn_evals": self.n_total_fn_evals,
            "n_total_jac_evals": self.n_total_jac_evals,
            "n_attempts": self.n_attempts,
            "n_recovery_attempts": self.n_recovery_attempts,
            "recovery_success_rate": self.recovery_success_rate,
            "final_success": self.final_success,
            "best_cost": self.best_cost,
            "n_degeneracies_detected": self.n_degeneracies_detected,
            "attempts": [
                {
                    "attempt_index": a.attempt_index,
                    "method": a.method,
                    "n_fn_evals": a.n_fn_evals,
                    "n_jac_evals": a.n_jac_evals,
                    "wall_time": a.wall_time,
                    "final_cost": a.final_cost,
                    "success": a.success,
                    "convergence_reason": a.convergence_reason,
                    "recovery_applied": a.recovery_applied,
                    "degeneracy_detected": a.degeneracy_detected,
                }
                for a in self.attempts
            ],
        }

    def summary_line(self) -> str:
        """Return a single-line human-readable summary.

        Returns:
            Compact status string suitable for logging.
        """
        cost_str = (
            f"{self.best_cost:.4e}" if self.best_cost is not None else "n/a"
        )
        return (
            f"OptimizationDiagnostics: success={self.final_success} "
            f"attempts={self.n_attempts} best_cost={cost_str} "
            f"fn_evals={self.n_total_fn_evals} "
            f"wall_time={self.total_wall_time:.2f}s "
            f"recovery_rate={self.recovery_success_rate:.0%}"
        )


# ---------------------------------------------------------------------------
# HybridRecoveryManager
# ---------------------------------------------------------------------------


@dataclass
class _RecoveryAttemptStats:
    """Internal per-strategy bookkeeping."""

    strategy: RecoveryStrategy
    n_attempts: int = 0
    n_successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.n_attempts == 0:
            return 0.0
        return self.n_successes / self.n_attempts


class HybridRecoveryManager:
    """Orchestrate recovery strategies for failed NLSQ fits.

    Applies a ranked sequence of recovery strategies.  After each
    attempted fit the manager is notified of the outcome so it can
    update per-strategy success statistics and decide whether to
    escalate to the next strategy.

    Recovery strategies (applied in order):
    1. ``parameter_perturbation``   — add noise to initial parameters.
    2. ``bound_relaxation``         — widen search bounds by a factor.
    3. ``method_switching``         — switch solver from ``trf`` to ``dogbox``
                                     (or vice-versa) or inject ``lm`` for
                                     unconstrained sub-problems.
    4. ``regularization_adjustment``— tighten tolerances and increase
                                     max iterations.

    Args:
        recovery_config: Scaling parameters for retry attempts.
        strategy_order: Ordered list of strategy names to apply.  The
            manager cycles through them in the order given.
        bound_relaxation_factor: Multiplicative factor by which each bound
            is widened when ``bound_relaxation`` is active.  A value of
            ``1.2`` expands ``[lo, hi]`` by 10 % on each side.
        max_bound_relaxation_steps: Maximum number of successive bound
            widenings before the strategy is abandoned.
        method_cycle: Ordered list of solver method strings to try
            when ``method_switching`` is active.
    """

    def __init__(
        self,
        recovery_config: HybridRecoveryConfig | None = None,
        *,
        strategy_order: list[RecoveryStrategy] | None = None,
        bound_relaxation_factor: float = 1.2,
        max_bound_relaxation_steps: int = 3,
        method_cycle: list[str] | None = None,
    ) -> None:
        self._cfg = recovery_config if recovery_config is not None else HybridRecoveryConfig()
        self._strategy_order: list[RecoveryStrategy] = (
            strategy_order
            if strategy_order is not None
            else list(_ALL_RECOVERY_STRATEGIES)
        )
        self._bound_relax_factor = bound_relaxation_factor
        self._max_bound_relax = max_bound_relaxation_steps
        self._method_cycle = method_cycle if method_cycle is not None else ["trf", "dogbox", "lm"]

        # Per-strategy stats
        self._stats: dict[RecoveryStrategy, _RecoveryAttemptStats] = {
            s: _RecoveryAttemptStats(strategy=s) for s in _ALL_RECOVERY_STRATEGIES
        }

        # Mutable session state
        self._current_strategy_idx: int = 0
        self._bound_relax_steps: int = 0
        self._method_idx: int = 0
        self._total_recovery_attempts: int = 0
        self._total_recovery_successes: int = 0

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset per-session state (strategy index, relaxation steps).

        Cumulative statistics are preserved so they remain valid across
        multiple fit sessions.
        """
        self._current_strategy_idx = 0
        self._bound_relax_steps = 0
        self._method_idx = 0
        self._total_recovery_attempts = 0
        self._total_recovery_successes = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def current_strategy(self) -> RecoveryStrategy:
        """Name of the strategy that will be applied on the next attempt."""
        idx = self._current_strategy_idx % len(self._strategy_order)
        return self._strategy_order[idx]

    def notify_outcome(self, *, success: bool) -> None:
        """Update statistics after an attempted recovery.

        Args:
            success: Whether the recovery attempt produced a successful fit.
        """
        strategy = self.current_strategy
        self._stats[strategy].n_attempts += 1
        self._total_recovery_attempts += 1

        if success:
            self._stats[strategy].n_successes += 1
            self._total_recovery_successes += 1
        else:
            # Advance to the next strategy on failure
            self._current_strategy_idx += 1
            if strategy == "bound_relaxation":
                self._bound_relax_steps += 1
            elif strategy == "method_switching":
                self._method_idx += 1

    def apply_parameter_perturbation(
        self,
        params: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        attempt: int,
    ) -> np.ndarray:
        """Perturb *params* using the HybridRecoveryConfig scale.

        The noise scale grows with *attempt* so later retries explore
        progressively wider neighbourhoods::

            noise ~ Uniform(-1, 1) * perturb_scale * attempt * (upper - lower)

        Args:
            params: Current parameter array.
            lower: Lower bound array.
            upper: Upper bound array.
            attempt: One-based attempt index (1 = first recovery attempt).

        Returns:
            Clipped perturbed parameter array.
        """
        if attempt <= 0:
            return np.clip(params, lower, upper)

        rng = np.random.default_rng(seed=42 + attempt)
        width = upper - lower
        noise = rng.uniform(-1.0, 1.0, size=params.shape)
        perturbed = params + self._cfg.perturb_scale * attempt * width * noise
        clipped = np.clip(perturbed, lower, upper)

        logger.debug(
            "HybridRecoveryManager.apply_parameter_perturbation: "
            "attempt=%d perturb_scale=%.4f rms_noise=%.4e",
            attempt,
            self._cfg.perturb_scale,
            float(np.sqrt(np.mean(noise**2))),
        )
        return clipped

    def apply_bound_relaxation(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        original_lower: np.ndarray,
        original_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Widen bounds by :attr:`_bound_relax_factor` relative to originals.

        Each relaxation step multiplies the half-width by the factor, so
        bounds grow geometrically.  Relaxation is capped at
        :attr:`_max_bound_relax` steps.

        Args:
            lower: Current lower bound array.
            upper: Current upper bound array.
            original_lower: Original lower bounds (before any relaxation).
            original_upper: Original upper bounds (before any relaxation).

        Returns:
            ``(new_lower, new_upper)`` after applying one relaxation step.
        """
        step = min(self._bound_relax_steps + 1, self._max_bound_relax)
        factor = self._bound_relax_factor**step

        mid = 0.5 * (original_lower + original_upper)
        half_width = 0.5 * (original_upper - original_lower)
        new_lower = mid - factor * half_width
        new_upper = mid + factor * half_width

        logger.debug(
            "HybridRecoveryManager.apply_bound_relaxation: "
            "step=%d factor=%.3f",
            step,
            factor,
        )
        return new_lower, new_upper

    def apply_method_switching(self, config: NLSQConfig) -> NLSQConfig:
        """Return a copy of *config* with the next solver method in the cycle.

        The cycle list is ``["trf", "dogbox", "lm"]`` by default.  The
        ``lm`` method does not support bound constraints; callers must
        ensure this is acceptable for their problem.

        Args:
            config: Current NLSQ configuration.

        Returns:
            New :class:`NLSQConfig` with :attr:`~NLSQConfig.method` updated.
        """
        idx = self._method_idx % len(self._method_cycle)
        new_method = self._method_cycle[idx]
        adjusted = copy.copy(config)
        adjusted.method = new_method  # type: ignore[assignment]

        logger.debug(
            "HybridRecoveryManager.apply_method_switching: "
            "%s -> %s",
            config.method,
            new_method,
        )
        return adjusted

    def apply_regularization_adjustment(
        self,
        config: NLSQConfig,
        attempt: int,
    ) -> NLSQConfig:
        """Return a copy of *config* with tighter tolerances and more iterations.

        Uses the ``lr_scale`` and ``lambda_scale`` from
        :meth:`HybridRecoveryConfig.get_retry_settings` to adjust
        ``ftol`` and ``xtol``, and scales ``max_iterations`` upward.

        Args:
            config: Current NLSQ configuration.
            attempt: One-based attempt index.

        Returns:
            New :class:`NLSQConfig` with adjusted tolerance and iteration fields.
        """
        settings = self._cfg.get_retry_settings(attempt)
        lr = settings["lr_scale"]

        adjusted = copy.copy(config)
        adjusted.ftol = config.ftol * lr
        adjusted.xtol = config.xtol * lr

        # Increase iteration budget on recovery — cap at a reasonable multiple
        scale = min(2.0**attempt, 8.0)
        adjusted.max_iterations = int(config.max_iterations * scale)

        logger.debug(
            "HybridRecoveryManager.apply_regularization_adjustment: "
            "attempt=%d lr=%.4f ftol=%.2e xtol=%.2e max_iter=%d",
            attempt,
            lr,
            adjusted.ftol,
            adjusted.xtol,
            adjusted.max_iterations,
        )
        return adjusted

    def apply_strategy(
        self,
        strategy: RecoveryStrategy,
        *,
        params: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        original_lower: np.ndarray,
        original_upper: np.ndarray,
        config: NLSQConfig,
        attempt: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, NLSQConfig]:
        """Dispatch to the concrete recovery strategy.

        Returns updated ``(params, lower, upper, config)``.  Only the
        fields that the strategy actually modifies will differ from the
        inputs.

        Args:
            strategy: Which strategy to apply.
            params: Current parameter starting point.
            lower: Current lower bound array.
            upper: Current upper bound array.
            original_lower: Original lower bounds (before any relaxation).
            original_upper: Original upper bounds (before any relaxation).
            config: Current NLSQ configuration.
            attempt: One-based attempt index.

        Returns:
            Tuple ``(new_params, new_lower, new_upper, new_config)``.
        """
        if strategy == "parameter_perturbation":
            new_params = self.apply_parameter_perturbation(params, lower, upper, attempt)
            return new_params, lower, upper, config

        if strategy == "bound_relaxation":
            new_lower, new_upper = self.apply_bound_relaxation(
                lower, upper, original_lower, original_upper
            )
            # Re-clip params to newly relaxed bounds
            new_params = np.clip(params, new_lower, new_upper)
            return new_params, new_lower, new_upper, config

        if strategy == "method_switching":
            new_config = self.apply_method_switching(config)
            return params, lower, upper, new_config

        if strategy == "regularization_adjustment":
            new_config = self.apply_regularization_adjustment(config, attempt)
            return params, lower, upper, new_config

        # Unreachable with valid RecoveryStrategy literal, but mypy needs this
        raise ValueError(f"Unknown recovery strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def overall_success_rate(self) -> float:
        """Fraction of all recovery attempts that produced a successful fit."""
        if self._total_recovery_attempts == 0:
            return 0.0
        return self._total_recovery_successes / self._total_recovery_attempts

    def per_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Return a dict of per-strategy statistics.

        Returns:
            Dictionary keyed by strategy name with ``n_attempts``,
            ``n_successes``, and ``success_rate`` for each.
        """
        return {
            s: {
                "n_attempts": st.n_attempts,
                "n_successes": st.n_successes,
                "success_rate": st.success_rate,
            }
            for s, st in self._stats.items()
        }

    def __repr__(self) -> str:
        return (
            f"HybridRecoveryManager("
            f"current_strategy={self.current_strategy!r} "
            f"total_attempts={self._total_recovery_attempts} "
            f"overall_success_rate={self.overall_success_rate:.0%})"
        )


# ---------------------------------------------------------------------------
# StreamingOptimizer
# ---------------------------------------------------------------------------


def _iter_chunks(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    n_data: int,
    chunk_size: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(start, end, residual_chunk)`` for each chunk of *n_data*.

    The residual function is evaluated at a synthetic index array (same
    convention used by the nlsq CurveFit API: ``x = arange(n_data)``).

    Args:
        residual_fn: Callable accepting a full-length parameter-independent
            index array and returning residuals.  For compatibility with both
            numpy and JAX-traced functions the caller must supply a wrapper
            that evaluates at the current parameter point.
        n_data: Total number of data points.
        chunk_size: Maximum number of points per chunk.

    Yields:
        ``(start, end, chunk_residuals)`` where ``chunk_residuals`` is a
        numpy array of shape ``(end - start,)``.
    """
    start = 0
    while start < n_data:
        end = min(start + chunk_size, n_data)
        chunk_residuals = residual_fn(np.arange(start, end, dtype=np.float64))
        yield start, end, np.asarray(chunk_residuals)
        start = end


@dataclass
class _StreamingAccumulator:
    """Accumulate Gauss-Newton normal-equation components across chunks.

    For the separable nonlinear least-squares problem we accumulate::

        A  = sum_chunks  J_i^T J_i     (n_params x n_params)
        b  = sum_chunks  J_i^T r_i     (n_params,)
        ssr = sum_chunks ||r_i||^2

    from which a one-shot Levenberg-Marquardt step can be computed in
    :meth:`solve`.

    Attributes:
        n_params: Dimension of the parameter space.
    """

    n_params: int
    _A: np.ndarray = field(init=False)
    _b: np.ndarray = field(init=False)
    _ssr: float = field(default=0.0, init=False)
    _n_data: int = field(default=0, init=False)
    _n_chunks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._A = np.zeros((self.n_params, self.n_params), dtype=np.float64)
        self._b = np.zeros(self.n_params, dtype=np.float64)

    def update(self, residuals: np.ndarray, jacobian: np.ndarray) -> None:
        """Accumulate one chunk.

        Args:
            residuals: Residual vector of shape ``(n_chunk,)``.
            jacobian: Jacobian matrix of shape ``(n_chunk, n_params)``.
        """
        self._A += jacobian.T @ jacobian
        self._b += jacobian.T @ residuals
        self._ssr += float(np.dot(residuals, residuals))
        self._n_data += len(residuals)
        self._n_chunks += 1

    def solve(self, lm_damping: float = 1e-6) -> np.ndarray:
        """Solve the damped normal equations for a parameter update.

        Applies Levenberg-Marquardt damping to improve numerical stability
        when ``A`` is near-singular.

        Args:
            lm_damping: Regularisation factor multiplied by ``diag(A)``
                before inversion.

        Returns:
            Parameter update vector ``delta_p`` of shape ``(n_params,)``.
        """
        diag_A = np.diag(self._A)
        damping = lm_damping * np.where(diag_A > 0, diag_A, 1.0)
        A_damp = self._A + np.diag(damping)

        cond = np.linalg.cond(A_damp)
        if cond > 1e14:
            logger.warning(
                "_StreamingAccumulator.solve: ill-conditioned normal matrix "
                "(cond=%.2e), using pseudo-inverse",
                cond,
            )
            delta_p: np.ndarray = -np.linalg.pinv(A_damp) @ self._b
        else:
            delta_p = -np.linalg.solve(A_damp, self._b)

        return delta_p

    @property
    def ssr(self) -> float:
        """Accumulated sum of squared residuals."""
        return self._ssr

    @property
    def n_data(self) -> int:
        """Total number of data points accumulated."""
        return self._n_data

    @property
    def n_chunks(self) -> int:
        """Number of chunks processed."""
        return self._n_chunks

    def covariance_estimate(self, lm_damping: float = 1e-6) -> np.ndarray | None:
        """Estimate parameter covariance from the normal matrix.

        Computes ``cov = s^2 * (A + lm_damping * diag(A))^{-1}`` where
        ``s^2 = ssr / (n_data - n_params)``.

        Returns:
            Covariance matrix of shape ``(n_params, n_params)``, or ``None``
            when degrees of freedom are non-positive.
        """
        n_dof = self._n_data - self.n_params
        if n_dof <= 0:
            return None

        s2 = self._ssr / n_dof
        diag_A = np.diag(self._A)
        damping = lm_damping * np.where(diag_A > 0, diag_A, 1.0)
        A_damp = self._A + np.diag(damping)

        try:
            cov: np.ndarray = np.linalg.inv(A_damp) * s2
            return cov
        except np.linalg.LinAlgError:
            logger.warning("_StreamingAccumulator: covariance inversion failed")
            return None


class StreamingOptimizer:
    """Process large residual datasets in streaming fashion.

    For problems where the full Jacobian would not fit in RAM, the
    streaming optimizer partitions the data into fixed-size chunks,
    evaluates residuals and a finite-difference Jacobian per chunk, and
    accumulates the Gauss-Newton normal equations.  A single linear solve
    then yields the parameter update, which is iterated until convergence.

    This is a simplified Gauss-Newton-over-chunks approach — not a full
    trust-region method.  For high-accuracy fits use :class:`NLSQWrapper`
    directly.  The streaming path is intended as a memory-safe fallback for
    very large datasets (> ``config.streaming_chunk_size`` points).

    Args:
        parameter_names: Names of the parameters being optimised.
        chunk_size: Override the chunk size from the config.  When
            ``None``, the value from :attr:`NLSQConfig.streaming_chunk_size`
            is used.
        max_outer_iterations: Maximum Gauss-Newton outer iterations.
        convergence_rtol: Relative tolerance on parameter update norm for
            outer-loop convergence.
        lm_damping: Levenberg-Marquardt damping for the normal equations.
        fd_step: Finite-difference step for numerical Jacobian estimation.
    """

    def __init__(
        self,
        parameter_names: list[str],
        *,
        chunk_size: int | None = None,
        max_outer_iterations: int = 20,
        convergence_rtol: float = 1e-6,
        lm_damping: float = 1e-6,
        fd_step: float = 1e-7,
    ) -> None:
        self._parameter_names = parameter_names
        self._n_params = len(parameter_names)
        self._chunk_size_override = chunk_size
        self._max_outer = max_outer_iterations
        self._conv_rtol = convergence_rtol
        self._lm_damping = lm_damping
        self._fd_step = fd_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        n_data: int,
    ) -> NLSQResult:
        """Run streaming Gauss-Newton optimisation.

        Args:
            residual_fn: ``(params: np.ndarray) -> np.ndarray`` returning
                the full-length residual vector.  Must accept the current
                parameter array, not an index vector.
            initial_params: Starting parameter values.
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ configuration (chunk size and max iterations
                are read from here unless overridden at construction).
            n_data: Total number of data points.

        Returns:
            :class:`NLSQResult` with the best parameters found.

        Raises:
            StreamingError: When a chunk evaluation produces NaN/Inf
                residuals and recovery is not possible.
        """
        chunk_size = (
            self._chunk_size_override
            if self._chunk_size_override is not None
            else config.streaming_chunk_size
        )
        lower, upper = bounds
        params = np.clip(initial_params, lower, upper)
        t_start = time.perf_counter()

        logger.info(
            "StreamingOptimizer.fit: n_data=%d chunk_size=%d n_params=%d "
            "max_outer=%d",
            n_data,
            chunk_size,
            self._n_params,
            self._max_outer,
        )

        n_total_chunks = math.ceil(n_data / chunk_size)
        best_params = params.copy()
        best_ssr: float = float("inf")
        converged = False
        convergence_reason = "max_iterations"
        n_outer_iter = 0

        for outer_iter in range(self._max_outer):
            n_outer_iter = outer_iter + 1
            accumulator = _StreamingAccumulator(n_params=self._n_params)

            # Build a closure that evaluates residuals at fixed params
            def _full_residuals(p: np.ndarray) -> np.ndarray:
                return np.asarray(residual_fn(p))

            # Evaluate residuals and finite-difference Jacobian chunk by chunk
            for chunk_start in range(0, n_data, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_data)

                # Residuals at current params (slice for the chunk)
                full_res = _full_residuals(params)
                chunk_res = full_res[chunk_start:chunk_end]

                # Validate chunk residuals
                if not np.all(np.isfinite(chunk_res)):
                    raise StreamingError(
                        f"NaN/Inf residuals in chunk [{chunk_start}:{chunk_end}]",
                        chunk_index=chunk_start // chunk_size,
                        total_chunks=n_total_chunks,
                    )

                # Finite-difference Jacobian for this chunk
                chunk_jac = self._fd_jacobian_chunk(
                    residual_fn, params, chunk_start, chunk_end, lower, upper
                )

                accumulator.update(chunk_res, chunk_jac)

            current_ssr = accumulator.ssr
            if current_ssr < best_ssr:
                best_ssr = current_ssr
                best_params = params.copy()

            # Solve normal equations for parameter update
            delta_p = accumulator.solve(lm_damping=self._lm_damping)

            # Clip proposed update to stay in bounds
            new_params = np.clip(params + delta_p, lower, upper)
            step_norm = float(np.linalg.norm(new_params - params))
            param_norm = float(np.linalg.norm(params)) + 1e-30

            logger.debug(
                "StreamingOptimizer outer iter %d/%d: ssr=%.4e step_norm=%.4e",
                n_outer_iter,
                self._max_outer,
                current_ssr,
                step_norm,
            )

            params = new_params

            if step_norm / param_norm < self._conv_rtol:
                converged = True
                convergence_reason = "parameter_convergence"
                logger.info(
                    "StreamingOptimizer converged at outer iter %d "
                    "(step_norm/param_norm=%.2e < rtol=%.2e)",
                    n_outer_iter,
                    step_norm / param_norm,
                    self._conv_rtol,
                )
                break

        wall_time = time.perf_counter() - t_start

        # Re-compute final residuals at best_params for reporting
        try:
            final_residuals = np.asarray(residual_fn(best_params))
        except Exception as exc:  # pragma: no cover
            logger.warning("StreamingOptimizer: final residual evaluation failed: %s", exc)
            final_residuals = None

        final_cost = (
            float(0.5 * np.sum(final_residuals**2))
            if final_residuals is not None
            else best_ssr * 0.5
        )

        n_dof = n_data - self._n_params
        reduced_chi2 = (2.0 * final_cost / n_dof) if n_dof > 0 else None

        # Estimate covariance from final accumulator (last outer iteration)
        covariance = accumulator.covariance_estimate(lm_damping=self._lm_damping)
        uncertainties: np.ndarray | None = None
        if covariance is not None:
            try:
                with np.errstate(invalid="raise"):
                    uncertainties = np.sqrt(np.diag(covariance))
            except (FloatingPointError, ValueError):
                logger.warning("StreamingOptimizer: could not extract uncertainties")

        success = converged and (final_residuals is None or np.all(np.isfinite(best_params)))

        logger.info(
            "StreamingOptimizer.fit done: success=%s cost=%.4e chunks_per_iter=%d "
            "outer_iters=%d wall_time=%.2fs",
            success,
            final_cost,
            n_total_chunks,
            n_outer_iter,
            wall_time,
        )

        return NLSQResult(
            parameters=best_params,
            parameter_names=self._parameter_names,
            success=success,
            message=f"StreamingOptimizer: {convergence_reason}",
            uncertainties=uncertainties,
            covariance=covariance,
            final_cost=final_cost,
            reduced_chi_squared=reduced_chi2,
            n_iterations=n_outer_iter,
            n_function_evals=n_outer_iter * n_total_chunks * (self._n_params + 1),
            convergence_reason=convergence_reason,
            residuals=final_residuals,
            wall_time_seconds=wall_time,
            metadata={
                "streaming": True,
                "chunk_size": chunk_size,
                "n_total_chunks": n_total_chunks,
                "n_outer_iterations": n_outer_iter,
                "converged": converged,
            },
        )

    # ------------------------------------------------------------------
    # Finite-difference Jacobian helper
    # ------------------------------------------------------------------

    def _fd_jacobian_chunk(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        params: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Compute a finite-difference Jacobian for one chunk.

        Uses a forward-difference stencil with step size
        ``max(fd_step * |p_i|, fd_step)`` per parameter.  The full
        residual vector is evaluated for each perturbed parameter; only
        the rows in ``[chunk_start:chunk_end]`` are returned.

        Args:
            residual_fn: Full residual function.
            params: Current parameter array.
            chunk_start: First row index (inclusive).
            chunk_end: Last row index (exclusive).
            lower: Lower bound array (used to clip perturbed params).
            upper: Upper bound array.

        Returns:
            Jacobian sub-matrix of shape ``(chunk_end - chunk_start, n_params)``.
        """
        n_chunk = chunk_end - chunk_start
        jac = np.zeros((n_chunk, self._n_params), dtype=np.float64)

        base_residuals = np.asarray(residual_fn(params))
        base_chunk = base_residuals[chunk_start:chunk_end]

        for i in range(self._n_params):
            h = max(self._fd_step * abs(params[i]), self._fd_step)
            p_fwd = params.copy()
            p_fwd[i] = np.clip(params[i] + h, lower[i], upper[i])
            res_fwd = np.asarray(residual_fn(p_fwd))
            jac[:, i] = (res_fwd[chunk_start:chunk_end] - base_chunk) / h

        return jac


# ---------------------------------------------------------------------------
# NLSQWrapper
# ---------------------------------------------------------------------------


class NLSQWrapper:
    """Retrying wrapper with progressive recovery strategy.

    On each failed attempt (``success=False`` or ``final_cost`` is None)
    the wrapper applies HybridRecoveryConfig scaling to the solver
    tolerances and perturbs the initial parameters before retrying.
    The best result across all attempts (measured by ``final_cost``) is
    always returned, even when every attempt fails.

    Args:
        parameter_names: Names of the parameters being optimised.
        use_jax: If ``True`` use :class:`NLSQAdapter` (JAX/nlsq backend);
            otherwise use :class:`ScipyNLSQAdapter` (default).
        max_retries: Maximum number of additional attempts after the
            first.  Total attempts = max_retries + 1.
        perturb_scale: Base noise scale per unit of parameter-space
            width.  Actual noise on attempt k =
            ``perturb_scale * k * (upper - lower)``.
        enable_recovery: When ``True`` (default) use
            :class:`HybridRecoveryConfig` to adjust tolerances per retry.
            When ``False`` the config is passed through unchanged.
        enable_anti_degeneracy: When ``True`` and the
            ``anti_degeneracy_controller`` module is importable, run a
            degeneracy check after each successful fit.
        enable_diagnostics: When ``True`` log detailed per-attempt
            timing and cost information.
        recovery_config: Custom :class:`HybridRecoveryConfig`.  A default
            instance is used when ``None``.
    """

    def __init__(
        self,
        parameter_names: list[str],
        *,
        use_jax: bool = False,
        max_retries: int = 3,
        perturb_scale: float = 0.05,
        enable_recovery: bool = True,
        enable_anti_degeneracy: bool = False,
        enable_diagnostics: bool = False,
        recovery_config: HybridRecoveryConfig | None = None,
    ) -> None:
        self._parameter_names = parameter_names
        self._max_retries = max_retries
        self._perturb_scale = perturb_scale
        self._enable_recovery = enable_recovery
        self._enable_anti_degeneracy = enable_anti_degeneracy and HAS_ANTI_DEGENERACY
        self._enable_diagnostics = enable_diagnostics
        self._recovery_config: HybridRecoveryConfig = (
            recovery_config if recovery_config is not None else HybridRecoveryConfig()
        )
        self._adapter: NLSQAdapterBase = (
            NLSQAdapter(parameter_names=parameter_names)
            if use_jax
            else ScipyNLSQAdapter(parameter_names=parameter_names)
        )
        self._recovery_actions: list[dict[str, Any]] = []

        # Extended tracking state (populated by fit_with_recovery and streaming)
        self._last_diagnostics: OptimizationDiagnostics | None = None
        self._eval_counter: FunctionEvaluationCounter = FunctionEvaluationCounter()
        self._recovery_manager: HybridRecoveryManager = HybridRecoveryManager(
            recovery_config=self._recovery_config
        )

        if self._enable_anti_degeneracy and not HAS_ANTI_DEGENERACY:
            logger.warning(
                "NLSQWrapper: enable_anti_degeneracy=True but "
                "AntiDegeneracyController could not be imported; disabling."
            )

        logger.debug(
            "NLSQWrapper initialised: backend=%s max_retries=%d "
            "perturb_scale=%s enable_recovery=%s enable_anti_degeneracy=%s",
            self._adapter.name,
            self._max_retries,
            self._perturb_scale,
            self._enable_recovery,
            self._enable_anti_degeneracy,
        )

    # ------------------------------------------------------------------
    # Public API — original (preserved exactly)
    # ------------------------------------------------------------------

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run optimisation with automatic progressive recovery on failure.

        On each attempt the parameters are perturbed by additive noise::

            noise ~ Uniform(-1, 1) * perturb_scale * attempt * (upper - lower)

        The perturbed parameters are clipped to *bounds* before fitting.
        When ``enable_recovery`` is ``True``, solver tolerances are also
        scaled by the HybridRecoveryConfig factors for attempts > 0.

        Args:
            residual_fn: Callable ``(params: np.ndarray) -> np.ndarray``
                returning residuals.
            initial_params: Starting parameter values.
            bounds: ``(lower, upper)`` bound arrays, each of length
                ``len(initial_params)``.
            config: NLSQ optimisation configuration.
            jacobian_fn: Optional analytic Jacobian callable.

        Returns:
            Best :class:`NLSQResult` across all attempts (lowest
            ``final_cost``).  The ``metadata["recovery"]`` key is always
            populated with attempt count, action log, and total wall time.
        """
        lower, upper = bounds
        best_result: NLSQResult | None = None
        self._recovery_actions = []

        total_attempts = self._max_retries + 1
        t_start = time.perf_counter()

        for attempt in range(total_attempts):
            t_attempt = time.perf_counter()

            # Build per-attempt config (adjusted tolerances on retries)
            if self._enable_recovery and attempt > 0:
                retry_settings = self._recovery_config.get_retry_settings(attempt)
                adjusted_config = self._apply_recovery_settings(config, retry_settings)
            else:
                adjusted_config = config
                retry_settings: dict[str, float] = {}

            current_params = self._perturbed_params(
                initial_params, lower, upper, attempt
            )

            logger.info(
                "NLSQWrapper attempt %d/%d (backend=%s%s)",
                attempt + 1,
                total_attempts,
                self._adapter.name,
                f", recovery={retry_settings}" if retry_settings else "",
            )

            result = self._adapter.fit(
                residual_fn=residual_fn,
                initial_params=current_params,
                bounds=bounds,
                config=adjusted_config,
                jacobian_fn=jacobian_fn,
            )

            attempt_time = time.perf_counter() - t_attempt
            self._log_attempt_summary(attempt, result, attempt_time)

            # Track recovery action for attempts beyond the first
            if attempt > 0:
                self._recovery_actions.append({
                    "attempt": attempt,
                    "settings": retry_settings,
                    "success": result.success,
                    "cost": result.final_cost,
                    "time": attempt_time,
                })

            best_result = self._update_best(best_result, result)

            if result.success:
                if self._enable_anti_degeneracy and result.covariance is not None:
                    self._check_degeneracy(result)

                result.metadata["recovery"] = {
                    "attempts": attempt + 1,
                    "actions": self._recovery_actions,
                    "total_time": time.perf_counter() - t_start,
                }
                logger.info(
                    "NLSQWrapper: success on attempt %d (cost=%.4e)",
                    attempt + 1,
                    result.final_cost if result.final_cost is not None else float("nan"),
                )
                return result

            logger.warning(
                "NLSQWrapper: attempt %d failed — %s",
                attempt + 1,
                result.message,
            )

        # All attempts exhausted — return best seen so far.
        assert best_result is not None  # loop always executes at least once
        best_result.metadata["recovery"] = {
            "attempts": total_attempts,
            "actions": self._recovery_actions,
            "total_time": time.perf_counter() - t_start,
            "all_failed": True,
        }
        logger.warning(
            "NLSQWrapper: all %d attempts failed; returning best result "
            "(cost=%.4e, success=%s)",
            total_attempts,
            best_result.final_cost
            if best_result.final_cost is not None
            else float("nan"),
            best_result.success,
        )
        return best_result

    def fit_with_jax(
        self,
        jax_residual_fn: Callable[..., Any],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        n_data: int,
    ) -> NLSQResult:
        """Run optimisation using a JAX-traced residual function.

        Delegates to :meth:`NLSQAdapter.fit_jax` so the nlsq library can
        trace and compile *jax_residual_fn* directly.  The same progressive
        recovery logic as :meth:`fit` applies: on each failed attempt the
        parameters are perturbed and (when ``enable_recovery`` is ``True``)
        the solver tolerances are adjusted.

        Note: this method always uses :class:`NLSQAdapter` regardless of the
        ``use_jax`` flag supplied at construction, because a JAX-traced
        function cannot be passed to the scipy backend.

        Args:
            jax_residual_fn: Pure JAX callable with signature
                ``(x, *params) -> residuals`` where ``x`` is an index array
                of length *n_data*.
            initial_params: Starting parameter values.
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ optimisation configuration.
            n_data: Number of data points (used to size the nlsq fitter).

        Returns:
            Best :class:`NLSQResult` across all attempts, with
            ``metadata["recovery"]`` populated.
        """
        lower, upper = bounds
        best_result: NLSQResult | None = None
        self._recovery_actions = []

        # Always use NLSQAdapter for JAX-traced functions
        jax_adapter = NLSQAdapter(parameter_names=self._parameter_names)

        total_attempts = self._max_retries + 1
        t_start = time.perf_counter()

        for attempt in range(total_attempts):
            t_attempt = time.perf_counter()

            if self._enable_recovery and attempt > 0:
                retry_settings = self._recovery_config.get_retry_settings(attempt)
                adjusted_config = self._apply_recovery_settings(config, retry_settings)
            else:
                adjusted_config = config
                retry_settings = {}

            current_params = self._perturbed_params(
                initial_params, lower, upper, attempt
            )

            logger.info(
                "NLSQWrapper.fit_with_jax attempt %d/%d%s",
                attempt + 1,
                total_attempts,
                f" recovery={retry_settings}" if retry_settings else "",
            )

            result = jax_adapter.fit_jax(
                jax_residual_fn=jax_residual_fn,
                initial_params=current_params,
                bounds=bounds,
                config=adjusted_config,
                n_data=n_data,
            )

            attempt_time = time.perf_counter() - t_attempt
            self._log_attempt_summary(attempt, result, attempt_time)

            if attempt > 0:
                self._recovery_actions.append({
                    "attempt": attempt,
                    "settings": retry_settings,
                    "success": result.success,
                    "cost": result.final_cost,
                    "time": attempt_time,
                })

            best_result = self._update_best(best_result, result)

            if result.success:
                if self._enable_anti_degeneracy and result.covariance is not None:
                    self._check_degeneracy(result)

                result.metadata["recovery"] = {
                    "attempts": attempt + 1,
                    "actions": self._recovery_actions,
                    "total_time": time.perf_counter() - t_start,
                }
                logger.info(
                    "NLSQWrapper.fit_with_jax: success on attempt %d (cost=%.4e)",
                    attempt + 1,
                    result.final_cost if result.final_cost is not None else float("nan"),
                )
                return result

            logger.warning(
                "NLSQWrapper.fit_with_jax: attempt %d failed — %s",
                attempt + 1,
                result.message,
            )

        assert best_result is not None
        best_result.metadata["recovery"] = {
            "attempts": total_attempts,
            "actions": self._recovery_actions,
            "total_time": time.perf_counter() - t_start,
            "all_failed": True,
        }
        logger.warning(
            "NLSQWrapper.fit_with_jax: all %d attempts failed; returning best "
            "(cost=%.4e, success=%s)",
            total_attempts,
            best_result.final_cost
            if best_result.final_cost is not None
            else float("nan"),
            best_result.success,
        )
        return best_result

    # ------------------------------------------------------------------
    # Extended public API
    # ------------------------------------------------------------------

    def fit_with_recovery(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        *,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        multistart_seeds: list[int] | None = None,
        eval_budget: int | None = None,
    ) -> NLSQResult:
        """Full pipeline: initial fit -> per-strategy recovery -> multi-start fallback.

        Extends :meth:`fit` with three escalation tiers:

        1. **Initial fit** — first attempt with unmodified config and params.
        2. **Per-strategy recovery** — apply each recovery strategy in the
           :class:`HybridRecoveryManager` until success or strategies
           are exhausted.
        3. **Multi-start fallback** — run independent fits from randomised
           starting points if all recovery strategies have failed.

        A :class:`FunctionEvaluationCounter` limits the total evaluation
        budget when *eval_budget* is given.  The :class:`OptimizationDiagnostics`
        record is stored in :attr:`last_diagnostics` and embedded in the
        returned result's ``metadata["diagnostics"]`` key.

        Args:
            residual_fn: Callable returning residuals.
            initial_params: Starting parameter values.
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ configuration.
            jacobian_fn: Optional analytic Jacobian.
            multistart_seeds: Optional list of RNG seeds for the multi-start
                fallback.  When ``None`` and multi-start is needed, seeds
                ``[100, 200, 300]`` are used.
            eval_budget: Optional hard cap on total function + Jacobian
                evaluations across all tiers.

        Returns:
            Best :class:`NLSQResult` across all tiers and attempts, with
            ``metadata["diagnostics"]`` populated.
        """
        lower, upper = bounds
        t_start = time.perf_counter()

        self._eval_counter = FunctionEvaluationCounter(budget=eval_budget)
        self._recovery_manager.reset_session()
        diagnostics = OptimizationDiagnostics(parameter_names=list(self._parameter_names))
        best_result: NLSQResult | None = None

        # ------------------------------------------------------------------
        # Tier 1: Initial fit
        # ------------------------------------------------------------------
        logger.info("NLSQWrapper.fit_with_recovery: Tier 1 — initial fit")

        t_attempt = time.perf_counter()
        result = self._adapter.fit(
            residual_fn=residual_fn,
            initial_params=np.clip(initial_params, lower, upper),
            bounds=bounds,
            config=config,
            jacobian_fn=jacobian_fn,
        )
        attempt_time = time.perf_counter() - t_attempt
        self._eval_counter.absorb_result(result)

        degeneracy_flag = False
        if result.success and self._enable_anti_degeneracy and result.covariance is not None:
            dcheck = self._run_degeneracy_check(result)
            degeneracy_flag = dcheck.is_degenerate if dcheck is not None else False

        diagnostics.record_attempt(
            AttemptRecord(
                attempt_index=0,
                method=config.method,
                n_fn_evals=result.n_function_evals,
                n_jac_evals=result.n_iterations,
                wall_time=attempt_time,
                final_cost=result.final_cost,
                success=result.success,
                convergence_reason=result.convergence_reason,
                recovery_applied=None,
                degeneracy_detected=degeneracy_flag,
            )
        )
        best_result = result

        if result.success and not degeneracy_flag:
            diagnostics.final_success = True
            diagnostics.total_wall_time = time.perf_counter() - t_start
            self._last_diagnostics = diagnostics
            result.metadata["diagnostics"] = diagnostics.to_dict()
            logger.info(
                "NLSQWrapper.fit_with_recovery: success on initial fit (cost=%.4e)",
                result.final_cost if result.final_cost is not None else float("nan"),
            )
            return result

        # ------------------------------------------------------------------
        # Tier 2: Per-strategy recovery
        # ------------------------------------------------------------------
        logger.info(
            "NLSQWrapper.fit_with_recovery: Tier 2 — recovery strategies "
            "(initial fit %s)",
            "succeeded but degenerate" if result.success else "failed",
        )

        current_lower = lower.copy()
        current_upper = upper.copy()
        current_params = initial_params.copy()

        n_strategies = len(_ALL_RECOVERY_STRATEGIES)
        for recovery_attempt in range(1, n_strategies + 1):
            if self._eval_counter.budget_exceeded:
                logger.warning(
                    "NLSQWrapper.fit_with_recovery: eval budget exhausted "
                    "before recovery tier completed"
                )
                break

            strategy = self._recovery_manager.current_strategy

            (
                current_params,
                current_lower,
                current_upper,
                adjusted_config,
            ) = self._recovery_manager.apply_strategy(
                strategy,
                params=current_params,
                lower=current_lower,
                upper=current_upper,
                original_lower=lower,
                original_upper=upper,
                config=config,
                attempt=recovery_attempt,
            )

            logger.info(
                "NLSQWrapper.fit_with_recovery: recovery attempt %d/%d "
                "strategy=%r",
                recovery_attempt,
                n_strategies,
                strategy,
            )

            t_attempt = time.perf_counter()
            result = self._adapter.fit(
                residual_fn=residual_fn,
                initial_params=current_params,
                bounds=(current_lower, current_upper),
                config=adjusted_config,
                jacobian_fn=jacobian_fn,
            )
            attempt_time = time.perf_counter() - t_attempt
            self._eval_counter.absorb_result(result)

            degeneracy_flag = False
            if result.success and self._enable_anti_degeneracy and result.covariance is not None:
                dcheck = self._run_degeneracy_check(result)
                degeneracy_flag = dcheck.is_degenerate if dcheck is not None else False

            self._recovery_manager.notify_outcome(success=result.success and not degeneracy_flag)

            diagnostics.record_attempt(
                AttemptRecord(
                    attempt_index=recovery_attempt,
                    method=adjusted_config.method,
                    n_fn_evals=result.n_function_evals,
                    n_jac_evals=result.n_iterations,
                    wall_time=attempt_time,
                    final_cost=result.final_cost,
                    success=result.success,
                    convergence_reason=result.convergence_reason,
                    recovery_applied=strategy,
                    degeneracy_detected=degeneracy_flag,
                )
            )

            best_result = self._update_best(best_result, result)

            if result.success and not degeneracy_flag:
                diagnostics.final_success = True
                diagnostics.total_wall_time = time.perf_counter() - t_start
                self._last_diagnostics = diagnostics
                result.metadata["diagnostics"] = diagnostics.to_dict()
                logger.info(
                    "NLSQWrapper.fit_with_recovery: recovery success on attempt %d "
                    "(strategy=%r cost=%.4e)",
                    recovery_attempt,
                    strategy,
                    result.final_cost if result.final_cost is not None else float("nan"),
                )
                return result

        # ------------------------------------------------------------------
        # Tier 3: Multi-start fallback
        # ------------------------------------------------------------------
        logger.info(
            "NLSQWrapper.fit_with_recovery: Tier 3 — multi-start fallback "
            "(recovery tiers exhausted)"
        )

        seeds = multistart_seeds if multistart_seeds is not None else [100, 200, 300]

        for ms_idx, seed in enumerate(seeds):
            if self._eval_counter.budget_exceeded:
                logger.warning(
                    "NLSQWrapper.fit_with_recovery: eval budget exhausted "
                    "before multi-start fallback completed"
                )
                break

            ms_params = self._random_start(lower, upper, seed=seed)

            logger.info(
                "NLSQWrapper.fit_with_recovery: multi-start %d/%d (seed=%d)",
                ms_idx + 1,
                len(seeds),
                seed,
            )

            t_attempt = time.perf_counter()
            result = self._adapter.fit(
                residual_fn=residual_fn,
                initial_params=ms_params,
                bounds=bounds,
                config=config,
                jacobian_fn=jacobian_fn,
            )
            attempt_time = time.perf_counter() - t_attempt
            self._eval_counter.absorb_result(result)

            degeneracy_flag = False
            if result.success and self._enable_anti_degeneracy and result.covariance is not None:
                dcheck = self._run_degeneracy_check(result)
                degeneracy_flag = dcheck.is_degenerate if dcheck is not None else False

            attempt_global_idx = n_strategies + 1 + ms_idx
            diagnostics.record_attempt(
                AttemptRecord(
                    attempt_index=attempt_global_idx,
                    method=config.method,
                    n_fn_evals=result.n_function_evals,
                    n_jac_evals=result.n_iterations,
                    wall_time=attempt_time,
                    final_cost=result.final_cost,
                    success=result.success,
                    convergence_reason=result.convergence_reason,
                    recovery_applied="parameter_perturbation",
                    degeneracy_detected=degeneracy_flag,
                )
            )

            best_result = self._update_best(best_result, result)

            if result.success and not degeneracy_flag:
                diagnostics.final_success = True
                diagnostics.total_wall_time = time.perf_counter() - t_start
                self._last_diagnostics = diagnostics
                result.metadata["diagnostics"] = diagnostics.to_dict()
                logger.info(
                    "NLSQWrapper.fit_with_recovery: multi-start success "
                    "(seed=%d cost=%.4e)",
                    seed,
                    result.final_cost if result.final_cost is not None else float("nan"),
                )
                return result

        # ------------------------------------------------------------------
        # All tiers exhausted — return best seen
        # ------------------------------------------------------------------
        assert best_result is not None
        diagnostics.final_success = False
        diagnostics.total_wall_time = time.perf_counter() - t_start
        self._last_diagnostics = diagnostics
        best_result.metadata["diagnostics"] = diagnostics.to_dict()
        best_result.metadata["recovery"] = {
            "all_failed": True,
            "total_time": diagnostics.total_wall_time,
            "n_attempts": diagnostics.n_attempts,
        }

        logger.warning(
            "NLSQWrapper.fit_with_recovery: all tiers exhausted; "
            "returning best result (cost=%.4e, success=%s)\n%s",
            best_result.final_cost
            if best_result.final_cost is not None
            else float("nan"),
            best_result.success,
            diagnostics.summary_line(),
        )
        return best_result

    def fit_streaming(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        n_data: int,
        *,
        chunk_size: int | None = None,
        max_outer_iterations: int = 20,
        convergence_rtol: float = 1e-6,
    ) -> NLSQResult:
        """Fit using the streaming Gauss-Newton path for large datasets.

        When ``config.enable_streaming`` is ``False`` this method falls back
        to :meth:`fit` (ignoring *n_data* and *chunk_size*).  This ensures
        callers can unconditionally call :meth:`fit_streaming` and let the
        config gate the actual path.

        The streaming path breaks the data into chunks of
        ``chunk_size`` (or ``config.streaming_chunk_size``) points,
        accumulates the Gauss-Newton normal equations, and solves for
        parameter updates iteratively.  This avoids materialising the
        full ``(n_data × n_params)`` Jacobian in memory.

        Args:
            residual_fn: Callable ``(params: np.ndarray) -> np.ndarray``
                returning the full residual vector of length *n_data*.
            initial_params: Starting parameter values.
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ configuration.  The ``enable_streaming``,
                ``streaming_chunk_size``, and ``max_iterations`` fields
                are honoured.
            n_data: Total number of data points in the residual vector.
            chunk_size: Override chunk size.  When ``None``, uses
                ``config.streaming_chunk_size``.
            max_outer_iterations: Outer Gauss-Newton iterations.  When
                ``None``, uses ``config.max_iterations``.
            convergence_rtol: Relative parameter-change tolerance for
                outer-loop convergence.

        Returns:
            :class:`NLSQResult` from the streaming optimiser (or from
            the standard :meth:`fit` path when streaming is disabled).

        Raises:
            StreamingError: When a chunk evaluation produces NaN/Inf
                residuals that cannot be recovered.
        """
        if not config.enable_streaming:
            logger.info(
                "NLSQWrapper.fit_streaming: streaming disabled; "
                "delegating to NLSQWrapper.fit()"
            )
            return self.fit(
                residual_fn=residual_fn,
                initial_params=initial_params,
                bounds=bounds,
                config=config,
            )

        effective_chunk_size = (
            chunk_size if chunk_size is not None else config.streaming_chunk_size
        )
        effective_max_outer = min(max_outer_iterations, config.max_iterations)

        optimizer = StreamingOptimizer(
            parameter_names=self._parameter_names,
            chunk_size=effective_chunk_size,
            max_outer_iterations=effective_max_outer,
            convergence_rtol=convergence_rtol,
        )

        logger.info(
            "NLSQWrapper.fit_streaming: chunk_size=%d max_outer=%d n_data=%d",
            effective_chunk_size,
            effective_max_outer,
            n_data,
        )

        try:
            result = optimizer.fit(
                residual_fn=residual_fn,
                initial_params=initial_params,
                bounds=bounds,
                config=config,
                n_data=n_data,
            )
        except StreamingError:
            logger.warning(
                "NLSQWrapper.fit_streaming: StreamingError caught; "
                "falling back to standard fit()"
            )
            result = self.fit(
                residual_fn=residual_fn,
                initial_params=initial_params,
                bounds=bounds,
                config=config,
            )

        return result

    def get_optimization_stats(self) -> dict[str, Any]:
        """Return a consolidated metrics dictionary for the last fit session.

        Includes evaluation counter totals, recovery manager statistics,
        and the full :class:`OptimizationDiagnostics` record when available.

        Returns:
            Dictionary with the following top-level keys:

            - ``"eval_counter"``: dict from
              :class:`FunctionEvaluationCounter`.
            - ``"recovery_manager"``: per-strategy stats from
              :class:`HybridRecoveryManager`.
            - ``"diagnostics"``: serialised
              :class:`OptimizationDiagnostics` or ``None``.
            - ``"adapter_name"``: name of the underlying adapter.
            - ``"max_retries"``: configured maximum retry count.
        """
        diag_dict: dict[str, Any] | None = (
            self._last_diagnostics.to_dict()
            if self._last_diagnostics is not None
            else None
        )

        return {
            "eval_counter": {
                "n_fn": self._eval_counter.n_fn,
                "n_jac": self._eval_counter.n_jac,
                "n_grad": self._eval_counter.n_grad,
                "total": self._eval_counter.total,
                "budget": self._eval_counter.budget,
                "budget_exceeded": self._eval_counter.budget_exceeded,
            },
            "recovery_manager": {
                "overall_success_rate": self._recovery_manager.overall_success_rate,
                "per_strategy": self._recovery_manager.per_strategy_stats(),
            },
            "diagnostics": diag_dict,
            "adapter_name": self._adapter.name,
            "max_retries": self._max_retries,
        }

    @property
    def last_diagnostics(self) -> OptimizationDiagnostics | None:
        """Most recent :class:`OptimizationDiagnostics` from :meth:`fit_with_recovery`.

        Returns ``None`` when :meth:`fit_with_recovery` or
        :meth:`fit_streaming` has not yet been called.
        """
        return self._last_diagnostics

    # ------------------------------------------------------------------
    # Internal helpers — convergence quality and logging
    # ------------------------------------------------------------------

    def _apply_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        params: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        original_lower: np.ndarray,
        original_upper: np.ndarray,
        config: NLSQConfig,
        attempt: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, NLSQConfig]:
        """Dispatch a named recovery strategy via :class:`HybridRecoveryManager`.

        This thin wrapper exists so that direct callers of
        :class:`NLSQWrapper` can access recovery actions without needing to
        hold a reference to the manager.

        Args:
            strategy: Strategy to apply.
            params: Current parameter array.
            lower: Current lower bound array.
            upper: Current upper bound array.
            original_lower: Original (pre-relaxation) lower bound array.
            original_upper: Original (pre-relaxation) upper bound array.
            config: Current NLSQ configuration.
            attempt: One-based attempt index for scaling factors.

        Returns:
            ``(new_params, new_lower, new_upper, new_config)`` after applying
            the strategy.
        """
        return self._recovery_manager.apply_strategy(
            strategy,
            params=params,
            lower=lower,
            upper=upper,
            original_lower=original_lower,
            original_upper=original_upper,
            config=config,
            attempt=attempt,
        )

    def _check_convergence_quality(
        self,
        result: NLSQResult,
        config: NLSQConfig,
    ) -> list[str]:
        """Assess post-fit quality against :class:`NLSQValidationConfig` thresholds.

        Checks reduced chi-squared, parameter finiteness, relative
        uncertainties, and correlation coefficients against the thresholds
        in ``config.validation``.

        Args:
            result: Completed fit result.
            config: Configuration whose ``validation`` sub-config provides
                the assessment thresholds.

        Returns:
            List of warning strings.  An empty list indicates a clean fit.
        """
        val = config.validation
        warnings: list[str] = []

        # --- Parameter finiteness ---
        if not np.all(np.isfinite(result.parameters)):
            n_bad = int(np.sum(~np.isfinite(result.parameters)))
            warnings.append(f"{n_bad} non-finite parameter(s) in result")

        # --- Reduced chi-squared ---
        chi2 = result.reduced_chi_squared
        if chi2 is not None:
            if chi2 < val.chi2_warn_low:
                warnings.append(
                    f"Reduced chi-squared ({chi2:.4f}) below warning threshold "
                    f"({val.chi2_warn_low}) — possible over-fit or "
                    f"under-estimated noise"
                )
            elif chi2 > val.chi2_fail_high:
                warnings.append(
                    f"Reduced chi-squared ({chi2:.4f}) above fail threshold "
                    f"({val.chi2_fail_high}) — fit likely invalid"
                )
            elif chi2 > val.chi2_warn_high:
                warnings.append(
                    f"Reduced chi-squared ({chi2:.4f}) above warning threshold "
                    f"({val.chi2_warn_high}) — possible under-fit"
                )

        # --- Relative uncertainties ---
        if result.uncertainties is not None:
            for name, val_p, unc in zip(
                result.parameter_names,
                result.parameters,
                result.uncertainties,
                strict=True,
            ):
                if val_p != 0.0 and not math.isfinite(unc):
                    warnings.append(f"Non-finite uncertainty for parameter {name!r}")
                elif val_p != 0.0 and abs(unc / val_p) > val.max_relative_uncertainty:
                    warnings.append(
                        f"Large relative uncertainty for {name!r}: "
                        f"{abs(unc / val_p):.2%} > {val.max_relative_uncertainty:.0%}"
                    )

        # --- Correlation ---
        corr = result.get_correlation_matrix()
        if corr is not None:
            n = len(result.parameter_names)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr[i, j]) > val.correlation_warn:
                        warnings.append(
                            f"High correlation between "
                            f"{result.parameter_names[i]!r} and "
                            f"{result.parameter_names[j]!r}: "
                            f"r={corr[i, j]:.4f}"
                        )

        return warnings

    def _log_optimization_summary(
        self,
        result: NLSQResult,
        quality_warnings: list[str],
        total_wall_time: float,
        n_attempts: int,
    ) -> None:
        """Emit a structured log summary after a complete fit session.

        Only emitted when ``enable_diagnostics`` is ``True``.

        Args:
            result: Final (best) :class:`NLSQResult`.
            quality_warnings: Warnings from :meth:`_check_convergence_quality`.
            total_wall_time: Total elapsed seconds for the session.
            n_attempts: Total number of optimizer attempts made.
        """
        if not self._enable_diagnostics:
            return

        cost_str = (
            f"{result.final_cost:.4e}"
            if result.final_cost is not None
            else "n/a"
        )
        chi2_str = (
            f"{result.reduced_chi_squared:.4f}"
            if result.reduced_chi_squared is not None
            else "n/a"
        )

        logger.info(
            "=== NLSQWrapper optimization summary ===\n"
            "  success        : %s\n"
            "  final_cost     : %s\n"
            "  reduced_chi2   : %s\n"
            "  n_attempts     : %d\n"
            "  wall_time      : %.2f s\n"
            "  fn_evals       : %d\n"
            "  convergence    : %s\n"
            "  quality_warns  : %d",
            result.success,
            cost_str,
            chi2_str,
            n_attempts,
            total_wall_time,
            result.n_function_evals,
            result.convergence_reason,
            len(quality_warnings),
        )

        for warn in quality_warnings:
            logger.warning("  [quality] %s", warn)

    # ------------------------------------------------------------------
    # Helper methods (original logic preserved exactly)
    # ------------------------------------------------------------------

    def _apply_recovery_settings(
        self,
        config: NLSQConfig,
        settings: dict[str, float],
    ) -> NLSQConfig:
        """Return a shallow copy of *config* with tolerances scaled for recovery.

        The ``lr_scale`` factor tightens ftol and xtol (smaller tolerances
        demand a more precise solution, forcing the solver to work harder).
        ``trust_radius_scale`` is informational here — it is logged but not
        yet forwarded because ``scipy.optimize.least_squares`` does not
        expose a direct trust-radius knob at the Python API level.

        Args:
            config: Original NLSQ configuration.
            settings: Dictionary returned by
                :meth:`HybridRecoveryConfig.get_retry_settings`.

        Returns:
            New :class:`NLSQConfig` instance with adjusted tolerance fields.
        """
        adjusted = copy.copy(config)
        lr_factor = settings.get("lr_scale", 1.0)
        trust_factor = settings.get("trust_radius_scale", 1.0)
        lambda_factor = settings.get("lambda_scale", 1.0)

        adjusted.ftol = config.ftol * lr_factor
        adjusted.xtol = config.xtol * lr_factor
        # gtol is left unchanged — projecting gradient tolerance to the
        # trust scale avoids over-tightening the termination criterion.

        logger.debug(
            "_apply_recovery_settings: lr_scale=%.4f trust_scale=%.4f "
            "lambda_scale=%.4f => ftol=%.2e xtol=%.2e",
            lr_factor,
            trust_factor,
            lambda_factor,
            adjusted.ftol,
            adjusted.xtol,
        )
        return adjusted

    def _check_degeneracy(self, result: NLSQResult) -> None:
        """Check for degenerate (highly correlated) parameter pairs.

        Uses the off-diagonal correlation matrix.  Pairs with |r| > 0.98
        are logged as warnings.  When the full
        :class:`AntiDegeneracyController` is available it is used instead,
        providing richer diagnostics (bound saturation, plateau detection).

        Args:
            result: Completed fit result whose covariance is non-None.
        """
        if HAS_ANTI_DEGENERACY:
            controller = AntiDegeneracyController(correlation_threshold=0.98)
            check = controller.check(result, param_manager=None)
            if check.is_degenerate:
                logger.warning(
                    "AntiDegeneracyController: %s | action: %s",
                    check.message,
                    check.suggested_action,
                )
            return

        # Fallback: manual correlation scan
        corr = result.get_correlation_matrix()
        if corr is None:
            return
        n = len(self._parameter_names)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > 0.98:
                    logger.warning(
                        "Degenerate parameter pair: %s <-> %s (r=%.3f)",
                        self._parameter_names[i],
                        self._parameter_names[j],
                        corr[i, j],
                    )

    def _run_degeneracy_check(self, result: NLSQResult) -> DegeneracyCheck | None:
        """Run degeneracy check and return the result object.

        Unlike :meth:`_check_degeneracy` (which only logs), this helper
        returns the :class:`DegeneracyCheck` so callers can branch on
        ``is_degenerate``.

        Args:
            result: Completed fit result.

        Returns:
            :class:`DegeneracyCheck` when anti-degeneracy support is
            available, otherwise ``None``.
        """
        if not HAS_ANTI_DEGENERACY:
            return None

        controller = AntiDegeneracyController(correlation_threshold=0.98)
        check = controller.check(result, param_manager=None)
        if check.is_degenerate:
            logger.warning(
                "_run_degeneracy_check: %s | action: %s",
                check.message,
                check.suggested_action,
            )
        return check  # type: ignore[return-value]

    def _log_attempt_summary(
        self,
        attempt: int,
        result: NLSQResult,
        wall_time: float,
    ) -> None:
        """Log a one-line summary for a single attempt.

        Only emits output when ``enable_diagnostics`` is ``True``.

        Args:
            attempt: Zero-based attempt index.
            result: Result from the adapter.
            wall_time: Wall-clock duration of this attempt in seconds.
        """
        if self._enable_diagnostics:
            logger.info(
                "  Attempt %d: success=%s cost=%.4e time=%.2fs",
                attempt + 1,
                result.success,
                result.final_cost if result.final_cost is not None else float("nan"),
                wall_time,
            )

    # ------------------------------------------------------------------
    # Internal helpers (original logic preserved exactly)
    # ------------------------------------------------------------------

    def _perturbed_params(
        self,
        initial_params: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        attempt: int,
    ) -> np.ndarray:
        """Return parameters perturbed for *attempt*.

        Attempt 0 returns the original parameters unchanged (clipped to
        bounds).  Subsequent attempts add uniform noise scaled by::

            perturb_scale * attempt * (upper - lower)

        A fixed seed (``42 + attempt``) ensures reproducibility.

        Args:
            initial_params: Unperturbed starting values.
            lower: Lower bound array.
            upper: Upper bound array.
            attempt: Zero-based attempt index.

        Returns:
            Clipped, perturbed parameter array.
        """
        if attempt == 0:
            return np.clip(initial_params, lower, upper)

        rng = np.random.default_rng(seed=42 + attempt)
        width = upper - lower
        noise = rng.uniform(-1.0, 1.0, size=initial_params.shape)
        perturbed = initial_params + self._perturb_scale * attempt * width * noise
        clipped = np.clip(perturbed, lower, upper)
        logger.debug(
            "Perturbed params for attempt %d (scale=%.4f, attempt=%d)",
            attempt + 1,
            self._perturb_scale,
            attempt,
        )
        return clipped

    def _random_start(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Generate a uniformly random starting point within bounds.

        Args:
            lower: Lower bound array.
            upper: Upper bound array.
            seed: RNG seed for reproducibility.

        Returns:
            Parameter array sampled uniformly from ``[lower, upper]``.
        """
        rng = np.random.default_rng(seed=seed)
        return rng.uniform(lower, upper).astype(np.float64)

    @staticmethod
    def _update_best(
        best: NLSQResult | None,
        candidate: NLSQResult,
    ) -> NLSQResult:
        """Return whichever result has the lower final_cost.

        ``None`` cost is treated as infinity so a result with a finite
        cost always wins.

        Args:
            best: Current best result (may be ``None`` on first call).
            candidate: Newly obtained result.

        Returns:
            The result with the lower cost.
        """
        if best is None:
            return candidate

        best_cost = best.final_cost if best.final_cost is not None else float("inf")
        cand_cost = (
            candidate.final_cost if candidate.final_cost is not None else float("inf")
        )
        return candidate if cand_cost < best_cost else best
