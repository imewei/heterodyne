"""Recovery strategies for optimization failures.

When an NLSQ fit fails or produces suspect results, :func:`diagnose_failure`
inspects the :class:`~heterodyne.optimization.nlsq.results.NLSQResult` and
proposes a :class:`RecoveryPlan` that the caller can apply automatically or
present to the user.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnose_failure(result: NLSQResult, config: NLSQConfig) -> RecoveryPlan:
    """Inspect a failed (or suspect) NLSQ result and propose a recovery plan.

    Decision logic, evaluated in priority order:

    1. **NaN in parameters** -- the solver diverged numerically.
       Recommend :attr:`RecoveryAction.PERTURB` (re-start from a
       perturbed initial guess).
    2. **Singular / ill-conditioned Jacobian** -- the local geometry is
       degenerate.  Recommend :attr:`RecoveryAction.REDUCE_STEP`
       (tighter tolerances / smaller diff step).
    3. **Iteration limit reached** -- the solver ran out of budget.
       Recommend :attr:`RecoveryAction.RETRY` with doubled iterations.
    4. **Stalled cost** -- cost is not decreasing meaningfully.
       Recommend :attr:`RecoveryAction.SIMPLIFY` (fix poorly-determined
       parameters).
    5. **Fallback** -- :attr:`RecoveryAction.ABORT`.

    Args:
        result: The NLSQ result to analyse.
        config: The configuration that produced *result*.

    Returns:
        A :class:`RecoveryPlan` describing the suggested recovery.
    """
    # -- 1. NaN in parameters ------------------------------------------------
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

    # -- 2. Singular / ill-conditioned Jacobian ------------------------------
    if result.jacobian is not None:
        try:
            singular_values = np.linalg.svd(
                result.jacobian, compute_uv=False
            )
            cond = float(singular_values[0] / max(singular_values[-1], 1e-30))
            if cond > 1e14:
                logger.warning(
                    "Jacobian condition number %.3e exceeds threshold", cond
                )
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

    # -- 3. Iteration limit reached ------------------------------------------
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

    # -- 4. Cost not decreasing (stalled) ------------------------------------
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

    # -- 5. Fallback ---------------------------------------------------------
    return RecoveryPlan(
        action=RecoveryAction.ABORT,
        message=f"No automatic recovery available. Result message: {result.message}",
        modified_config=None,
    )


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------


def apply_recovery(plan: RecoveryPlan, config: NLSQConfig) -> NLSQConfig:
    """Return a new :class:`NLSQConfig` with the plan's overrides applied.

    If the plan has no ``modified_config``, the original *config* is
    returned unchanged.

    Args:
        plan: Recovery plan (typically from :func:`diagnose_failure`).
        config: Original optimisation configuration.

    Returns:
        A (possibly modified) :class:`NLSQConfig`.
    """
    if plan.modified_config is None:
        return config

    overrides = {
        k: v
        for k, v in plan.modified_config.items()
        if hasattr(config, k)
    }
    if not overrides:
        return config

    cfg_dict = config.to_dict()
    cfg_dict.update(overrides)
    return NLSQConfig.from_dict(cfg_dict)


# ---------------------------------------------------------------------------
# Parameter triage
# ---------------------------------------------------------------------------


def suggest_fixed_parameters(result: NLSQResult) -> list[str]:
    """Identify parameters that should be fixed (held constant).

    A parameter is flagged when:

    * Its relative uncertainty exceeds 100 % (poorly constrained), **or**
    * Its value sits exactly on a bound (at-bounds heuristic via the
      ``metadata`` dict, if the solver recorded lower/upper bounds).

    Args:
        result: A completed (possibly failed) NLSQ result.

    Returns:
        List of parameter names recommended for fixing.
    """
    candidates: list[str] = []

    # -- Large relative uncertainty ------------------------------------------
    if result.uncertainties is not None:
        for name, val, unc in zip(
            result.parameter_names,
            result.parameters,
            result.uncertainties,
            strict=True,
        ):
            if val != 0.0 and abs(unc / val) > 1.0:
                candidates.append(name)

    # -- At-bounds detection (if bounds stored in metadata) ------------------
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
