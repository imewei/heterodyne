"""Recovery mechanisms for failed NLSQ optimization.

Provides retry logic with progressive parameter perturbation, tolerance
relaxation, and method switching. Complements the strategy-level fallback
chain (which switches *strategies*) by operating at the *attempt* level
within a single strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

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
            pcov.shape, n_params,
        )
        return np.full(n_params, np.inf)

    diag = np.diag(pcov)

    # Replace negative diagonal entries with inf
    uncertainties = np.where(diag > 0, np.sqrt(diag), np.inf)

    n_inf = int(np.sum(np.isinf(uncertainties)))
    if n_inf > 0:
        logger.warning(
            "%d/%d parameters have undefined uncertainty (negative or zero variance)",
            n_inf, n_params,
        )

    return uncertainties


def execute_with_recovery(
    fit_fn: Callable[[np.ndarray, tuple[np.ndarray, np.ndarray], NLSQConfig], NLSQResult],
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
            noise = rng.normal(0, perturb_scale, size=initial_params.shape) * param_range
            current_params = np.clip(initial_params + noise, lower, upper)
            logger.info("Recovery attempt %d: perturbing parameters (scale=%.3f)",
                       attempt, perturb_scale)
        elif attempt == 2:
            # Tolerance relaxation
            action = "relax_tolerance"
            # Create a modified config with relaxed tolerances
            # We can't modify frozen config, so we modify the mutable fields
            current_config = config
            current_config.ftol = config.ftol * 10
            current_config.xtol = config.xtol * 10
            current_config.gtol = config.gtol * 10
            logger.info("Recovery attempt %d: relaxing tolerances by 10x", attempt)
        else:
            # Method switching
            action = "switch_method"
            method_cycle: dict[str, Literal["trf", "dogbox", "lm"]] = {
                "trf": "dogbox", "dogbox": "trf", "lm": "trf",
            }
            current_config = config
            fallback_method: Literal["trf", "dogbox", "lm"] = "trf"
            current_config.method = method_cycle.get(
                config.method, fallback_method,
            )
            logger.info("Recovery attempt %d: switching method to %s",
                       attempt, current_config.method)

        try:
            result = fit_fn(current_params, bounds, current_config)

            attempts.append({
                "attempt": attempt,
                "action": action,
                "success": result.success,
                "cost": result.final_cost,
            })

            if result.success:
                result.metadata["recovery"] = {
                    "total_attempts": attempt + 1,
                    "successful_action": action,
                    "all_attempts": attempts,
                }
                logger.info(
                    "Recovery succeeded on attempt %d (%s)",
                    attempt + 1, action,
                )
                return result

            logger.warning(
                "Recovery attempt %d (%s) did not converge: %s",
                attempt + 1, action, result.message,
            )

        except Exception as exc:  # noqa: BLE001
            diagnosis = diagnose_error(exc)
            logger.warning(
                "Recovery attempt %d (%s) failed: %s [%s]",
                attempt + 1, action, exc, diagnosis.category,
            )
            attempts.append({
                "attempt": attempt,
                "action": action,
                "error": str(exc),
                "category": diagnosis.category,
            })

            if not diagnosis.recoverable:
                break

    error_msg = f"All {len(attempts)} recovery attempts failed"
    raise RuntimeError(error_msg)
