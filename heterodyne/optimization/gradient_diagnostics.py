"""Gradient health analysis for NLSQ optimization.

Provides diagnostic utilities that inspect Jacobian matrices and residuals
to detect common pathologies: vanishing or exploding gradients, NaN/Inf
contamination, and severe parameter-sensitivity imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GradientHealth:
    """Summary of gradient health for an optimization snapshot.

    Attributes:
        is_healthy: ``True`` when no issues were detected.
        issues: Human-readable descriptions of each detected problem.
        metrics: Numeric diagnostics keyed by metric name.
    """

    is_healthy: bool
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Thresholds (module-level constants so they are easy to override in tests)
# ---------------------------------------------------------------------------

_VANISHING_THRESHOLD: float = 1e-12
_EXPLODING_THRESHOLD: float = 1e12
_IMBALANCE_THRESHOLD: float = 1e6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_gradients(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    param_names: list[str],
) -> GradientHealth:
    """Run a battery of gradient-health checks on a Jacobian.

    Args:
        jacobian: Jacobian matrix of shape ``(n_residuals, n_params)``.
        residuals: Residual vector of length ``n_residuals``.
        param_names: Parameter names, one per column of *jacobian*.

    Returns:
        A :class:`GradientHealth` report.
    """
    issues: list[str] = []
    metrics: dict[str, float] = {}

    grad_norm = compute_gradient_norm(jacobian)
    metrics["gradient_norm"] = grad_norm

    # -- NaN / Inf -----------------------------------------------------------
    n_nan = int(np.count_nonzero(np.isnan(jacobian)))
    n_inf = int(np.count_nonzero(np.isinf(jacobian)))
    if n_nan > 0:
        issues.append(f"Jacobian contains {n_nan} NaN element(s)")
    if n_inf > 0:
        issues.append(f"Jacobian contains {n_inf} Inf element(s)")

    n_nan_res = int(np.count_nonzero(np.isnan(residuals)))
    n_inf_res = int(np.count_nonzero(np.isinf(residuals)))
    if n_nan_res > 0:
        issues.append(f"Residuals contain {n_nan_res} NaN element(s)")
    if n_inf_res > 0:
        issues.append(f"Residuals contain {n_inf_res} Inf element(s)")

    # -- Vanishing / Exploding -----------------------------------------------
    if grad_norm < _VANISHING_THRESHOLD:
        issues.append(
            f"Vanishing gradient: norm {grad_norm:.3e} < {_VANISHING_THRESHOLD:.0e}"
        )
    if grad_norm > _EXPLODING_THRESHOLD:
        issues.append(
            f"Exploding gradient: norm {grad_norm:.3e} > {_EXPLODING_THRESHOLD:.0e}"
        )

    # -- Per-parameter sensitivity imbalance ----------------------------------
    sensitivities = compute_per_parameter_sensitivity(jacobian, param_names)
    metrics.update({f"sensitivity_{k}": v for k, v in sensitivities.items()})

    sens_values = np.array(list(sensitivities.values()))
    nonzero_sens = sens_values[sens_values > 0.0]
    if len(nonzero_sens) >= 2:
        ratio = float(nonzero_sens.max() / nonzero_sens.min())
        metrics["sensitivity_ratio"] = ratio
        if ratio > _IMBALANCE_THRESHOLD:
            issues.append(
                f"Imbalanced gradients: max/min sensitivity ratio "
                f"{ratio:.3e} > {_IMBALANCE_THRESHOLD:.0e}"
            )

    is_healthy = len(issues) == 0
    if not is_healthy:
        logger.warning("Gradient diagnostics detected %d issue(s)", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)

    return GradientHealth(is_healthy=is_healthy, issues=issues, metrics=metrics)


def compute_gradient_norm(jacobian: np.ndarray) -> float:
    """Frobenius norm of the Jacobian.

    Args:
        jacobian: Jacobian matrix of shape ``(n_residuals, n_params)``.

    Returns:
        Frobenius norm as a Python float.
    """
    return float(np.linalg.norm(jacobian))


def compute_per_parameter_sensitivity(
    jacobian: np.ndarray,
    param_names: list[str],
) -> dict[str, float]:
    """Column-wise L2 norms of the Jacobian.

    Each column norm quantifies how sensitive the residuals are to a
    unit change in the corresponding parameter.

    Args:
        jacobian: Jacobian matrix of shape ``(n_residuals, n_params)``.
        param_names: Parameter names (length must equal ``n_params``).

    Returns:
        Mapping from parameter name to its column norm.
    """
    col_norms = np.linalg.norm(jacobian, axis=0)
    return {
        name: float(norm)
        for name, norm in zip(param_names, col_norms, strict=True)
    }


def suggest_step_sizes(
    jacobian: np.ndarray,
    param_names: list[str],
) -> dict[str, float]:
    """Suggest adaptive finite-difference step sizes for each parameter.

    The heuristic is ``step ~ 1 / column_norm`` so that parameters with
    steep gradients receive smaller perturbations and vice versa.  A
    floor of ``1e-15`` prevents division by zero, and the result is
    clipped to ``[1e-12, 1e-2]`` for numerical safety.

    Args:
        jacobian: Jacobian matrix of shape ``(n_residuals, n_params)``.
        param_names: Parameter names (length must equal ``n_params``).

    Returns:
        Mapping from parameter name to suggested step size.
    """
    col_norms = np.linalg.norm(jacobian, axis=0)
    # Inverse-norm heuristic, clipped for safety
    raw_steps = 1.0 / np.maximum(col_norms, 1e-15)
    clipped = np.clip(raw_steps, 1e-12, 1e-2)
    return {
        name: float(step)
        for name, step in zip(param_names, clipped, strict=True)
    }
