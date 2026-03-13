"""Gradient health analysis for NLSQ optimization.

Provides diagnostic utilities that inspect Jacobian matrices and residuals
to detect common pathologies: vanishing or exploding gradients, NaN/Inf
contamination, and severe parameter-sensitivity imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
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
        name: float(norm) for name, norm in zip(param_names, col_norms, strict=True)
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
    return {name: float(step) for name, step in zip(param_names, clipped, strict=True)}


# ---------------------------------------------------------------------------
# JAX-based gradient diagnostics for parameter scaling
# ---------------------------------------------------------------------------


def compute_gradient_norms(
    residual_fn: Any,
    param_array: jnp.ndarray,
    param_names: list[str],
) -> dict[str, float]:
    """Compute per-parameter gradient norms using JAX autodiff.

    This is model-agnostic: callers build the residual function that maps
    a parameter vector to a residual vector.

    Args:
        residual_fn: Callable ``(params) -> residuals`` where *params* is
            a 1-D JAX array and *residuals* is a 1-D JAX array.
        param_array: Current parameter vector, shape ``(n_params,)``.
        param_names: Parameter names, one per element of *param_array*.

    Returns:
        Dict mapping each parameter name to ``|dL/dp_i|`` where
        ``L = sum(residuals**2)``.
    """
    grad_fn = jax.grad(lambda p: jnp.sum(residual_fn(p) ** 2))
    grads = grad_fn(param_array)
    return {name: float(jnp.abs(g)) for name, g in zip(param_names, grads, strict=True)}


def compute_optimal_x_scale(
    gradient_norms: dict[str, float],
    baseline_params: list[str] | None = None,
    safety_factor: float = 1.0,
    min_scale: float = 1e-8,
    max_scale: float = 1e2,
) -> dict[str, float]:
    """Compute x_scale values inversely proportional to gradient norms.

    Parameters with large gradients receive small scales and vice versa,
    equalising the effective step size across parameters.

    Args:
        gradient_norms: Per-parameter gradient norms from
            :func:`compute_gradient_norms`.
        baseline_params: Parameter names for computing the geometric-mean
            baseline.  Defaults to the first 3 parameters.
        safety_factor: Multiplier applied to the raw scale.
        min_scale: Floor for clipping.
        max_scale: Ceiling for clipping.

    Returns:
        Dict mapping parameter name to recommended x_scale.
    """
    names = list(gradient_norms.keys())
    norms = np.array([gradient_norms[n] for n in names])

    # Baseline: geometric mean of selected parameters
    if baseline_params is None:
        baseline_params = names[:3]
    baseline_norms = np.array(
        [gradient_norms[n] for n in baseline_params if n in gradient_norms]
    )
    if len(baseline_norms) == 0 or np.any(baseline_norms <= 0):
        baseline = float(np.median(norms[norms > 0])) if np.any(norms > 0) else 1.0
    else:
        baseline = float(np.exp(np.mean(np.log(baseline_norms))))

    scales: dict[str, float] = {}
    for name, norm in zip(names, norms, strict=True):
        if norm <= 0:
            scales[name] = 1.0
            continue
        raw = safety_factor * baseline / norm
        clipped = float(np.clip(raw, min_scale, max_scale))
        scales[name] = clipped

        ratio = norm / baseline
        if ratio > 10.0:
            logger.info(
                "Parameter %s gradient %.3e is %.1fx above baseline",
                name,
                norm,
                ratio,
            )
        elif ratio < 0.1:
            logger.info(
                "Parameter %s gradient %.3e is %.1fx below baseline",
                name,
                norm,
                1.0 / ratio,
            )

    return scales


def diagnose_gradient_imbalance(
    gradient_norms: dict[str, float],
    threshold: float = 10.0,
) -> dict[str, Any]:
    """Diagnose gradient imbalance across parameters.

    Args:
        gradient_norms: Per-parameter gradient norms.
        threshold: Ratio above which imbalance is flagged.

    Returns:
        Dict with keys:
        - ``gradient_norms``: The input norms.
        - ``imbalance_detected``: Whether max/min ratio exceeds *threshold*.
        - ``max_ratio``: The max/min gradient ratio.
        - ``recommendations``: Recommended x_scale dict if imbalance
          detected, else ``None``.
    """
    norms = np.array(list(gradient_norms.values()))
    nonzero = norms[norms > 0]

    if len(nonzero) < 2:
        return {
            "gradient_norms": gradient_norms,
            "imbalance_detected": False,
            "max_ratio": 1.0,
            "recommendations": None,
        }

    max_ratio = float(nonzero.max() / nonzero.min())
    imbalanced = max_ratio > threshold

    recommendations = None
    if imbalanced:
        logger.warning(
            "Gradient imbalance detected: max/min ratio = %.1f (threshold %.1f)",
            max_ratio,
            threshold,
        )
        recommendations = compute_optimal_x_scale(gradient_norms)

    return {
        "gradient_norms": gradient_norms,
        "imbalance_detected": imbalanced,
        "max_ratio": max_ratio,
        "recommendations": recommendations,
    }
