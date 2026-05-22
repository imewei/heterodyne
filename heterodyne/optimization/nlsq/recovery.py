"""Recovery mechanisms for failed NLSQ optimization.

Provides retry logic with progressive parameter perturbation, tolerance
relaxation, and method switching. Complements the strategy-level fallback
chain (which switches *strategies*) by operating at the *attempt* level
within a single strategy.
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass(frozen=True)
class ErrorDiagnosis:
    """Diagnosis of an optimization error.

    Attributes:
        category: Error category for selecting recovery action.
        message: Human-readable description.
        recoverable: Whether recovery is likely to help.
        suggested_action: Recommended recovery action.
    """

    category: str
    message: str
    recoverable: bool
    suggested_action: str


# Error category constants
CATEGORY_OOM = "oom"
CATEGORY_CONVERGENCE = "convergence"
CATEGORY_BOUNDS = "bounds"
CATEGORY_ILL_CONDITIONED = "ill_conditioned"
CATEGORY_NAN = "nan"
CATEGORY_UNKNOWN = "unknown"


def diagnose_error(error: Exception) -> ErrorDiagnosis:
    """Categorize an optimization error for recovery selection.

    Args:
        error: The exception from the failed optimization.

    Returns:
        ErrorDiagnosis with category and recovery suggestion.
    """
    msg = str(error).lower()

    if isinstance(error, MemoryError) or "memory" in msg or "oom" in msg:
        return ErrorDiagnosis(
            category=CATEGORY_OOM,
            message=f"Out of memory: {error}",
            recoverable=True,
            suggested_action="reduce_data_size",
        )

    if "nan" in msg or "inf" in msg or "not finite" in msg:
        return ErrorDiagnosis(
            category=CATEGORY_NAN,
            message=f"Numerical error: {error}",
            recoverable=True,
            suggested_action="perturb_parameters",
        )

    if "bounds" in msg or "constraint" in msg:
        return ErrorDiagnosis(
            category=CATEGORY_BOUNDS,
            message=f"Bounds violation: {error}",
            recoverable=True,
            suggested_action="relax_bounds",
        )

    if "singular" in msg or "ill-conditioned" in msg or "linalg" in msg:
        return ErrorDiagnosis(
            category=CATEGORY_ILL_CONDITIONED,
            message=f"Ill-conditioned problem: {error}",
            recoverable=True,
            suggested_action="add_regularization",
        )

    if "max" in msg and ("iter" in msg or "nfev" in msg or "eval" in msg):
        return ErrorDiagnosis(
            category=CATEGORY_CONVERGENCE,
            message=f"Convergence failure: {error}",
            recoverable=True,
            suggested_action="relax_tolerance",
        )

    return ErrorDiagnosis(
        category=CATEGORY_UNKNOWN,
        message=f"Unknown error: {error}",
        recoverable=False,
        suggested_action="none",
    )


def safe_uncertainties_from_pcov(
    pcov: np.ndarray | None,
    n_params: int = 14,
) -> np.ndarray:
    """Extract parameter uncertainties from a covariance matrix safely.

    Handles singular, near-singular, and negative-diagonal covariance
    matrices gracefully.

    Args:
        pcov: Covariance matrix of shape (n_params, n_params), or None.
        n_params: Expected number of parameters (for fallback shape).

    Returns:
        Array of uncertainties, shape (n_params,). Returns inf for
        parameters with undefined uncertainty.
    """
    if pcov is None:
        logger.warning("No covariance matrix; returning inf uncertainties")
        return np.full(n_params, np.inf)

    pcov = np.asarray(pcov, dtype=np.float64)

    if pcov.shape != (n_params, n_params):
        logger.warning(
            "Covariance shape %s doesn't match n_params=%d; returning inf",
            pcov.shape,
            n_params,
        )
        return np.full(n_params, np.inf)

    diag = np.diag(pcov)

    # Replace negative diagonal entries with inf
    uncertainties = np.where(diag > 0, np.sqrt(diag), np.inf)

    n_inf = int(np.sum(np.isinf(uncertainties)))
    if n_inf > 0:
        logger.warning(
            "%d/%d parameters have undefined uncertainty (negative or zero variance)",
            n_inf,
            n_params,
        )

    return uncertainties


def execute_with_recovery(
    fit_fn: Callable[
        [np.ndarray, tuple[np.ndarray, np.ndarray], NLSQConfig], NLSQResult
    ],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    config: NLSQConfig,
    *,
    max_retries: int = 3,
    perturb_scale: float = 0.1,
    rng_seed: int = 42,
) -> NLSQResult:
    """Execute optimization with progressive recovery on failure.

    Recovery sequence (3 retries):
    1. Perturbation: add Gaussian noise to initial parameters
    2. Tolerance relaxation: increase ftol/xtol/gtol by 10x
    3. Method switching: try different trust-region algorithm

    Args:
        fit_fn: Callable taking (params, bounds, config) and returning NLSQResult.
        initial_params: Initial parameter values.
        bounds: Tuple of (lower, upper) bound arrays.
        config: NLSQ configuration.
        max_retries: Maximum number of recovery attempts.
        perturb_scale: Scale of parameter perturbation as fraction of range.
        rng_seed: Random seed for reproducibility.

    Returns:
        NLSQResult from the first successful attempt.

    Raises:
        RuntimeError: If all attempts fail.
    """
    rng = np.random.default_rng(rng_seed)
    lower, upper = bounds
    param_range = upper - lower

    attempts: list[dict[str, Any]] = []

    for attempt in range(max_retries + 1):
        current_params = initial_params.copy()
        current_config = config

        if attempt == 0:
            action = "initial"
        elif attempt == 1:
            # Perturbation
            action = "perturb"
            noise = (
                rng.normal(0, perturb_scale, size=initial_params.shape) * param_range
            )
            current_params = np.clip(initial_params + noise, lower, upper)
            logger.info("Retrying with perturbed parameters...")
            logger.info(
                "Recovery attempt %d: perturbing parameters (scale=%.3f)",
                attempt,
                perturb_scale,
            )
        elif attempt == 2:
            # Tolerance relaxation — use a shallow copy so the caller's config
            # is never mutated.  copy.copy() works for dataclasses, SimpleNamespace,
            # and any other attribute-bearing object.
            action = "relax_tolerance"
            current_config = copy.copy(config)
            current_config.ftol = config.ftol * 10
            current_config.xtol = config.xtol * 10
            current_config.gtol = config.gtol * 10
            logger.info("Recovery attempt %d: relaxing tolerances by 10x", attempt)
        else:
            # Method switching — same shallow-copy pattern
            action = "switch_method"
            method_cycle: dict[str, Literal["trf", "dogbox", "lm"]] = {
                "trf": "dogbox",
                "dogbox": "trf",
                "lm": "trf",
            }
            fallback_method: Literal["trf", "dogbox", "lm"] = "trf"
            current_config = copy.copy(config)
            current_config.method = method_cycle.get(config.method, fallback_method)
            logger.info(
                "Recovery attempt %d: switching method to %s",
                attempt,
                current_config.method,
            )

        try:
            logger.debug("Using curve_fit_large with NLSQ automatic memory management")
            result = fit_fn(current_params, bounds, current_config)

            attempts.append(
                {
                    "attempt": attempt,
                    "action": action,
                    "success": result.success,
                    "cost": result.final_cost,
                }
            )

            # Log NLSQ result diagnostics (homodyne parity)
            logger.info("NLSQ curve_fit RESULT DIAGNOSTICS")
            # Note: bounds is always a tuple here; log first param for diagnostics
            logger.info(
                "  bounds=None (unbounded)"
                if lower[0] == -np.inf
                else "  bounds provided"
            )
            logger.info(
                "  attempt=%d, action=%s, success=%s",
                attempt + 1,
                action,
                result.success,
            )

            # Warn when parameters appear unchanged
            _params = getattr(result, "parameters", None)
            if _params is not None and np.allclose(
                _params, current_params, rtol=1e-10, atol=1e-14
            ):
                logger.warning(
                    "Optimization returned unchanged parameters on attempt %d.",
                    attempt + 1,
                )
                logger.warning(
                    "   Affected parameters were likely NOT optimized by NLSQ."
                )

            if result.success:
                result.metadata["recovery"] = {
                    "total_attempts": attempt + 1,
                    "successful_action": action,
                    "all_attempts": attempts,
                }
                logger.info(
                    "Recovery succeeded on attempt %d (%s)",
                    attempt + 1,
                    action,
                )
                return result

            logger.warning(
                "Recovery attempt %d (%s) did not converge: %s",
                attempt + 1,
                action,
                result.message,
            )

        except Exception as exc:  # noqa: BLE001
            diagnosis = diagnose_error(exc)
            logger.warning(
                "Recovery attempt %d (%s) failed: %s [%s]",
                attempt + 1,
                action,
                exc,
                diagnosis.category,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "action": action,
                    "error": str(exc),
                    "category": diagnosis.category,
                }
            )

            if not diagnosis.recoverable:
                break

    logger.error(
        "Optimization returned unchanged parameters after all retries. "
        "This may indicate a bug in NLSQ or an intractable problem."
    )
    error_msg = f"All {len(attempts)} recovery attempts failed"
    raise RuntimeError(error_msg)


# ---------------------------------------------------------------------------
# Post-fit RecoveryPlan diagnosis (consolidated from recovery_strategies.py)
# ---------------------------------------------------------------------------



class RecoveryAction(enum.Enum):
    """Possible corrective actions after an optimization failure."""

    RETRY = "retry"
    PERTURB = "perturb"
    REDUCE_STEP = "reduce_step"
    SIMPLIFY = "simplify"
    ABORT = "abort"


@dataclass
class RecoveryPlan:
    """Recommended corrective action with an explanation.

    Attributes:
        action: The recommended :class:`RecoveryAction`.
        message: Human-readable explanation of why this action was chosen.
        modified_config: Optional dictionary of config overrides to apply.
    """

    action: RecoveryAction
    message: str
    modified_config: dict[str, Any] | None = field(default=None)


def diagnose_failure(result: NLSQResult, config: NLSQConfig) -> RecoveryPlan:
    """Inspect a failed (or suspect) NLSQ result and propose a recovery plan.

    Decision logic, evaluated in priority order:

    1. **NaN in parameters** -- the solver diverged numerically.
       Recommend :attr:`RecoveryAction.PERTURB`.
    2. **Singular / ill-conditioned Jacobian** -- recommend
       :attr:`RecoveryAction.REDUCE_STEP`.
    3. **Iteration limit reached** -- recommend :attr:`RecoveryAction.RETRY`
       with doubled iterations.
    4. **Stalled cost** -- recommend :attr:`RecoveryAction.SIMPLIFY`.
    5. **Fallback** -- :attr:`RecoveryAction.ABORT`.

    Args:
        result: The NLSQ result to analyse.
        config: The configuration that produced *result*.

    Returns:
        A :class:`RecoveryPlan` describing the suggested recovery.
    """
    if np.any(np.isnan(result.parameters)):
        logger.warning("NaN detected in fitted parameters")
        return RecoveryPlan(
            action=RecoveryAction.PERTURB,
            message=(
                "Parameters contain NaN -- the solver diverged. "
                "Restarting from a perturbed initial guess."
            ),
            modified_config=None,
        )

    if result.jacobian is not None:
        try:
            singular_values = np.linalg.svd(result.jacobian, compute_uv=False)
            cond = float(singular_values[0] / max(singular_values[-1], 1e-30))
            if cond > 1e14:
                logger.warning("Jacobian condition number %.3e exceeds threshold", cond)
                new_diff_step = (config.diff_step or 1e-8) * 0.1
                return RecoveryPlan(
                    action=RecoveryAction.REDUCE_STEP,
                    message=(
                        f"Jacobian is near-singular (cond={cond:.2e}). "
                        f"Reducing diff_step to {new_diff_step:.1e}."
                    ),
                    modified_config={"diff_step": new_diff_step},
                )
        except np.linalg.LinAlgError:
            return RecoveryPlan(
                action=RecoveryAction.REDUCE_STEP,
                message="SVD of Jacobian failed -- matrix is degenerate.",
                modified_config={"diff_step": 1e-10},
            )

    if result.n_iterations >= config.max_iterations:
        new_max = config.max_iterations * 2
        logger.info(
            "Iteration limit (%d) reached; suggesting retry with %d",
            config.max_iterations,
            new_max,
        )
        return RecoveryPlan(
            action=RecoveryAction.RETRY,
            message=(
                f"Iteration limit ({config.max_iterations}) reached. "
                f"Retrying with max_iterations={new_max}."
            ),
            modified_config={"max_iterations": new_max},
        )

    if result.final_cost is not None and result.final_cost > 1.0:
        fixed = suggest_fixed_parameters(result)
        if fixed:
            logger.info("Suggesting simplification: fix %s", fixed)
            return RecoveryPlan(
                action=RecoveryAction.SIMPLIFY,
                message=(
                    f"Cost ({result.final_cost:.3e}) remains large. "
                    f"Consider fixing poorly-determined parameters: {fixed}."
                ),
                modified_config={"fixed_parameters": fixed},
            )

    return RecoveryPlan(
        action=RecoveryAction.ABORT,
        message=f"No automatic recovery available. Result message: {result.message}",
        modified_config=None,
    )


def apply_recovery(plan: RecoveryPlan, config: NLSQConfig) -> NLSQConfig:
    """Return a new NLSQConfig with the plan's overrides applied.

    Args:
        plan: Recovery plan (typically from :func:`diagnose_failure`).
        config: Original optimisation configuration.

    Returns:
        A (possibly modified) :class:`NLSQConfig`.
    """
    if plan.modified_config is None:
        return config

    overrides = {k: v for k, v in plan.modified_config.items() if hasattr(config, k)}
    if not overrides:
        return config

    cfg_dict = config.to_dict()
    cfg_dict.update(overrides)
    return NLSQConfig.from_dict(cfg_dict)


def suggest_fixed_parameters(result: NLSQResult) -> list[str]:
    """Identify parameters that should be fixed (held constant).

    A parameter is flagged when its relative uncertainty exceeds 100 %,
    or its value sits on a bound recorded in ``result.metadata``.

    Args:
        result: A completed (possibly failed) NLSQ result.

    Returns:
        List of parameter names recommended for fixing.
    """
    candidates: list[str] = []

    if result.uncertainties is not None:
        for name, val, unc in zip(
            result.parameter_names,
            result.parameters,
            result.uncertainties,
            strict=True,
        ):
            if val != 0.0 and abs(unc / val) > 1.0:
                candidates.append(name)

    lower = result.metadata.get("lower_bounds")
    upper = result.metadata.get("upper_bounds")
    if lower is not None and upper is not None:
        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        for name, val, lo, hi in zip(
            result.parameter_names,
            result.parameters,
            lower_arr,
            upper_arr,
            strict=True,
        ):
            at_lower = np.isfinite(lo) and abs(val - lo) < 1e-10 * max(abs(lo), 1.0)
            at_upper = np.isfinite(hi) and abs(val - hi) < 1e-10 * max(abs(hi), 1.0)
            if (at_lower or at_upper) and name not in candidates:
                candidates.append(name)

    return candidates
