"""Fallback chain for NLSQ optimization strategies.

Provides automatic strategy degradation when the primary fitting strategy
fails due to memory constraints, numerical errors, or convergence issues.
The chain tries strategies in descending order (most robust first) until
one succeeds or all options are exhausted.

Strategy order (descending — most robust to least):
1. STREAMING — per-epoch streaming optimizer, most robust
2. LARGE     — chunked J^T J accumulation, handles large datasets
3. STANDARD  — full in-memory Jacobian, fastest for small data

Routing delegates to NLSQWrapper-style functions rather than instantiating
FittingStrategy objects directly.  The initial strategy is selected by
``select_nlsq_strategy()`` from memory.py unless overridden.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.memory import NLSQStrategy, select_nlsq_strategy
from heterodyne.optimization.nlsq.result_builder import build_result_from_nlsq
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class OptimizationStrategy(Enum):
    """Available optimization strategies in descending fallback order.

    Values match :class:`~heterodyne.optimization.nlsq.memory.NLSQStrategy`
    string values for cross-module consistency.
    """

    STANDARD = "standard"
    LARGE = "large"
    STREAMING = "streaming"


# Descending fallback order: most-robust first, fastest last.
_FALLBACK_ORDER: list[OptimizationStrategy] = [
    OptimizationStrategy.STREAMING,
    OptimizationStrategy.LARGE,
    OptimizationStrategy.STANDARD,
]

# Mapping from NLSQStrategy → OptimizationStrategy
_NLSQ_TO_OPT: dict[NLSQStrategy, OptimizationStrategy] = {
    NLSQStrategy.STANDARD: OptimizationStrategy.STANDARD,
    NLSQStrategy.LARGE: OptimizationStrategy.LARGE,
    NLSQStrategy.STREAMING: OptimizationStrategy.STREAMING,
}


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


def handle_nlsq_result(
    raw_result: Any,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Normalize a raw NLSQ return value to ``(popt, pcov, info)``.

    Handles four return formats emitted by the nlsq library:

    - **dict** (``AdaptiveHybridStreamingOptimizer``): keys ``'x'``/``'popt'``
      and optional ``'pcov'``, ``'fun'``, streaming diagnostics.
    - **2-tuple** (``curve_fit``): ``(popt, pcov)``.
    - **3-tuple** (``curve_fit`` with ``full_output=True``): ``(popt, pcov, info)``.
    - **object** with ``.x``/``.popt`` and optional ``.pcov`` attributes.

    This function returns raw arrays. Downstream callers use
    :func:`~heterodyne.optimization.nlsq.result_builder.build_result_from_nlsq`
    to produce a fully populated :class:`~heterodyne.optimization.nlsq.results.NLSQResult`.

    Args:
        raw_result: Raw return value from an nlsq optimization call.

    Returns:
        Tuple ``(popt, pcov, info)`` where *pcov* may be ``None`` and
        *info* is a ``dict`` of supplementary data (possibly empty).

    Raises:
        TypeError: If the result format is unrecognized or required keys
            are absent.
    """
    popt: np.ndarray
    pcov: np.ndarray | None = None
    info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Case 1: dict (StreamingOptimizer)
    # ------------------------------------------------------------------
    if isinstance(raw_result, dict):
        popt_raw = raw_result.get("x", raw_result.get("popt"))
        if popt_raw is None:
            raise TypeError(
                "Dict result has neither 'x' nor 'popt' key. "
                f"Available keys: {list(raw_result.keys())}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)

        pcov_raw = raw_result.get("pcov")
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

        for key in (
            "streaming_diagnostics",
            "success",
            "message",
            "best_loss",
            "final_epoch",
            "fun",
        ):
            val = raw_result.get(key)
            if val is not None:
                info[key] = val

        logger.debug("Normalized StreamingOptimizer dict result")

    # ------------------------------------------------------------------
    # Case 2: tuple (curve_fit / curve_fit with full_output)
    # ------------------------------------------------------------------
    elif isinstance(raw_result, tuple):
        if len(raw_result) == 2:
            popt_raw, pcov_raw = raw_result
            logger.debug("Normalized (popt, pcov) tuple")
        elif len(raw_result) == 3:
            popt_raw, pcov_raw, extra = raw_result
            if isinstance(extra, dict):
                info = dict(extra)
            else:
                logger.warning(
                    "Info object is not a dict (type=%s); wrapping as 'raw_info'",
                    type(extra).__name__,
                )
                info = {"raw_info": extra}
            logger.debug("Normalized (popt, pcov, info) tuple")
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(raw_result)}. "
                "Expected 2 (popt, pcov) or 3 (popt, pcov, info)."
            )
        popt = np.asarray(popt_raw, dtype=np.float64)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

    # ------------------------------------------------------------------
    # Case 3: object with .x / .popt attributes
    # ------------------------------------------------------------------
    elif hasattr(raw_result, "x") or hasattr(raw_result, "popt"):
        popt_raw = getattr(raw_result, "x", getattr(raw_result, "popt", None))
        if popt_raw is None:
            raise TypeError(
                "Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(raw_result)}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)

        pcov_raw = getattr(raw_result, "pcov", None)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None
        if pcov_raw is None:
            logger.debug(
                "No pcov attribute in result object (type=%s)",
                type(raw_result).__name__,
            )

        for attr in ("message", "success", "nfev", "njev", "optimality", "fun"):
            val = getattr(raw_result, attr, None)
            if val is not None:
                info[attr] = val

        if hasattr(raw_result, "info") and isinstance(raw_result.info, dict):
            info.update(raw_result.info)

        logger.debug("Normalized object result (type=%s)", type(raw_result).__name__)

    # ------------------------------------------------------------------
    # Case 4: unrecognized
    # ------------------------------------------------------------------
    else:
        raise TypeError(
            f"Unrecognized NLSQ result format: {type(raw_result)}. "
            "Expected tuple, dict, or object with 'x'/'popt' attributes."
        )

    return popt, pcov, info


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------


def get_fallback_strategy(
    current: OptimizationStrategy,
    error: Exception | None = None,
) -> OptimizationStrategy | None:
    """Determine the next fallback strategy after a failure.

    For memory errors the chain skips ahead to STREAMING (the most
    memory-efficient option).  For all other errors the chain steps one
    position toward STANDARD.

    Args:
        current: The strategy that just failed.
        error: The exception that triggered the fallback; used for
            heuristic selection.

    Returns:
        The next :class:`OptimizationStrategy` to try, or ``None`` when
        the chain is exhausted.
    """
    current_idx = _FALLBACK_ORDER.index(current)

    # Memory errors: skip to STREAMING (most memory-efficient strategy) only
    # when the failure originates from LARGE.  LARGE is the primary candidate
    # for OOM because it accumulates the full Jacobian; STREAMING processes
    # data epoch-by-epoch and avoids that peak.  If current is already
    # STREAMING or STANDARD, there is no memory-safe escalation path.
    if error is not None and _is_memory_error(error):
        if current == OptimizationStrategy.LARGE:
            logger.info("Memory error detected; skipping to STREAMING strategy")
            return OptimizationStrategy.STREAMING
        # STREAMING or STANDARD — chain exhausted
        return None

    next_idx = current_idx + 1
    if next_idx < len(_FALLBACK_ORDER):
        return _FALLBACK_ORDER[next_idx]

    return None


def _is_memory_error(error: Exception) -> bool:
    """Return ``True`` if *error* looks memory-related."""
    if isinstance(error, MemoryError):
        return True
    error_str = str(error).lower()
    memory_keywords = ["memory", "oom", "out of memory", "allocate", "mmap"]
    return any(kw in error_str for kw in memory_keywords)


# ---------------------------------------------------------------------------
# Strategy routing
# ---------------------------------------------------------------------------


def _run_strategy(
    strategy: OptimizationStrategy,
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Dispatch a single strategy attempt and return ``(popt, pcov, info)``.

    Routes to the appropriate nlsq function based on *strategy*:

    - STANDARD  → ``nlsq.curve_fit``
    - LARGE     → ``nlsq.curve_fit_large``
    - STREAMING → ``nlsq.curve_fit_large`` with streaming flag (or fallback)

    Args:
        strategy: The strategy to execute.
        model: Configured HeterodyneModel.
        c2_data: Correlation data, shape ``(N, N)``.
        phi_angle: Detector phi angle in degrees.
        config: NLSQ configuration.
        weights: Optional per-point weights.

    Returns:
        Normalized ``(popt, pcov, info)`` tuple.
    """
    from heterodyne.optimization.nlsq.adapter import NLSQAdapter, NLSQWrapper

    parameter_names = list(model.param_manager.varying_names)

    # Build residual function from model
    def residual_fn(params: np.ndarray) -> np.ndarray:
        model.param_manager.update_values(
            model.param_manager.expand_varying_to_full(params)
        )
        c2_theory = model.compute_correlation(phi_angle=phi_angle)
        residuals = np.asarray(c2_theory).ravel() - c2_data.ravel()
        if weights is not None:
            residuals = residuals * np.sqrt(weights.ravel())
        return residuals  # type: ignore[no-any-return]

    initial_params = np.array(
        model.param_manager.get_initial_values(), dtype=np.float64
    )
    lower, upper = model.param_manager.get_bounds()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    bounds = (lower, upper)

    if strategy == OptimizationStrategy.STANDARD:
        logger.debug("_run_strategy: routing to NLSQAdapter (STANDARD)")
        adapter = NLSQAdapter(parameter_names=parameter_names)
        result = adapter.fit(residual_fn, initial_params, bounds, config)
    elif strategy in (OptimizationStrategy.LARGE, OptimizationStrategy.STREAMING):
        logger.debug("_run_strategy: routing to NLSQWrapper (%s)", strategy.name)
        wrapper = NLSQWrapper(parameter_names=parameter_names)
        result = wrapper.fit(residual_fn, initial_params, bounds, config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")  # pragma: no cover

    # Extract raw arrays from NLSQResult for handle_nlsq_result compatibility
    popt = result.parameters
    pcov = result.covariance
    info: dict[str, Any] = {
        "success": result.success,
        "message": result.message,
    }
    if result.final_cost is not None:
        info["fun"] = result.final_cost
    if result.metadata:
        info.update(result.metadata)

    return popt, pcov, info


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_optimization_with_fallback(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    *,
    start_strategy: OptimizationStrategy | None = None,
    weights: np.ndarray | None = None,
) -> NLSQResult:
    """Execute optimization with automatic fallback on failure.

    Tries strategies in descending robustness order, catching failures and
    degrading gracefully to the next available option.

    If *start_strategy* is ``None``, :func:`select_nlsq_strategy` is called
    with the data size and parameter count to pick the initial strategy
    (replaces the former hardcoded size thresholds).

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data, shape ``(N, N)``.
        phi_angle: Detector phi angle in degrees.
        config: NLSQ configuration.
        start_strategy: Initial strategy. ``None`` triggers memory-aware
            auto-selection via :func:`select_nlsq_strategy`.
        weights: Optional per-point weights.

    Returns:
        :class:`~heterodyne.optimization.nlsq.results.NLSQResult` from
        the first successful strategy.

    Raises:
        RuntimeError: If all strategies fail.
    """
    if start_strategy is None:
        n_params = model.param_manager.n_varying
        decision = select_nlsq_strategy(c2_data.size, n_params)
        start_strategy = _NLSQ_TO_OPT[decision.strategy]
        logger.info(
            "Auto-selected strategy: %s (%s)",
            start_strategy.name,
            decision.reason,
        )

    parameter_names: list[str] = list(model.param_manager.varying_names)
    n_data = int(c2_data.size)

    current: OptimizationStrategy | None = start_strategy
    errors: list[tuple[str, str]] = []

    while current is not None:
        logger.info("Fallback chain: attempting %s strategy", current.name)

        try:
            popt, pcov, info = _run_strategy(
                current, model, c2_data, phi_angle, config, weights
            )

            # Reconstruct a minimal dict/object to pass to build_result_from_nlsq
            raw_for_builder: dict[str, Any] = {"x": popt, **info}
            if pcov is not None:
                raw_for_builder["pcov"] = pcov

            result = build_result_from_nlsq(
                raw_for_builder,
                parameter_names=parameter_names,
                n_data=n_data,
                metadata={
                    "fallback_chain": {
                        "strategy_used": current.name,
                        "attempts": len(errors) + 1,
                        "failed_strategies": [e[0] for e in errors],
                    }
                },
            )

            logger.info(
                "Fallback chain: %s succeeded (cost=%s)",
                current.name,
                f"{result.final_cost:.4e}" if result.final_cost is not None else "n/a",
            )
            return result

        except Exception as exc:  # noqa: BLE001
            # current is non-None here (set at loop entry and not yet mutated)
            assert current is not None  # narrowing for mypy
            logger.warning(
                "Fallback chain: %s raised %s: %s",
                current.name,
                type(exc).__name__,
                exc,
            )
            errors.append((current.name, f"{type(exc).__name__}: {exc}"))
            current = get_fallback_strategy(current, error=exc)

    error_summary = "; ".join(f"{name}: {msg}" for name, msg in errors)
    raise RuntimeError(f"All optimization strategies failed. Errors: {error_summary}")


__all__ = [
    "OptimizationStrategy",
    "execute_optimization_with_fallback",
    "get_fallback_strategy",
    "handle_nlsq_result",
]
