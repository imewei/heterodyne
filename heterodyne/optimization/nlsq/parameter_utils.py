"""Parameter manipulation utilities for NLSQ optimization.

Provides helper functions for perturbing, clipping, sensitivity analysis,
and pretty-printing parameter arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def perturb_parameters(
    params: np.ndarray,
    scale: float,
    bounds: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add random perturbation to parameters, staying within bounds.

    Each parameter is perturbed by a Gaussian with standard deviation
    ``scale * (upper - lower)`` for the corresponding bound range, then
    clipped to ``[lower, upper]``.

    Args:
        params: Current parameter values, shape ``(n,)``.
        scale: Perturbation scale as a fraction of the bound range.
            Typical values: 0.01 -- 0.1.
        bounds: ``(lower, upper)`` arrays, each shape ``(n,)``.
        rng: NumPy random generator.  If ``None`` a default is created.

    Returns:
        Perturbed parameter array, shape ``(n,)``.
    """
    params = np.asarray(params, dtype=np.float64)
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)

    if params.shape != lower.shape or params.shape != upper.shape:
        raise ValueError(
            f"Shape mismatch: params {params.shape}, "
            f"lower {lower.shape}, upper {upper.shape}"
        )

    if rng is None:
        rng = np.random.default_rng()

    span = upper - lower
    sigma = scale * span
    perturbation = rng.normal(0.0, np.maximum(sigma, 1e-30), size=params.shape)
    perturbed = params + perturbation
    return np.clip(perturbed, lower, upper)


def clip_to_bounds(
    params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Clip parameter array to specified bounds.

    Args:
        params: Parameter values, shape ``(n,)``.
        lower: Lower bounds, shape ``(n,)``.
        upper: Upper bounds, shape ``(n,)``.

    Returns:
        Clipped parameter array, shape ``(n,)``.
    """
    params = np.asarray(params, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.clip(params, lower, upper)


def compute_parameter_sensitivity(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute parameter sensitivity via finite-difference perturbation.

    For each parameter, computes the ratio of the change in summed squared
    residuals to the parameter perturbation.  This gives a first-order
    approximation of how sensitive the cost function is to each parameter.

    Args:
        residual_fn: Function mapping parameter vector to residual vector.
        params: Current parameter values, shape ``(n_params,)``.
        step_sizes: Per-parameter perturbation sizes, shape ``(n_params,)``.
            Defaults to ``max(1e-8, 1e-4 * |param|)`` for each parameter.

    Returns:
        Sensitivity array of shape ``(n_params,)`` — the magnitude of
        the cost-function gradient with respect to each parameter.
    """
    params = np.asarray(params, dtype=np.float64)
    n = len(params)

    if step_sizes is None:
        step_sizes = np.maximum(1e-8, 1e-4 * np.abs(params))
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    r0 = np.asarray(residual_fn(params), dtype=np.float64)
    cost0 = float(np.sum(r0**2))

    sensitivity = np.zeros(n, dtype=np.float64)
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += step_sizes[i]
        r_plus = np.asarray(residual_fn(params_plus), dtype=np.float64)
        cost_plus = float(np.sum(r_plus**2))
        sensitivity[i] = abs(cost_plus - cost0) / step_sizes[i]

    return sensitivity


def format_parameter_table(
    names: list[str],
    values: np.ndarray,
    uncertainties: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    """Pretty-print a parameter table.

    Produces a human-readable multi-line table showing parameter names,
    fitted values, optional uncertainties, and optional bounds.

    Args:
        names: Parameter names.
        values: Fitted values, shape ``(n,)``.
        uncertainties: Standard errors, shape ``(n,)``.  ``None`` to omit.
        bounds: ``(lower, upper)`` arrays.  ``None`` to omit.

    Returns:
        Formatted multi-line string.
    """
    values = np.asarray(values, dtype=np.float64)

    # Determine columns
    has_unc = uncertainties is not None
    has_bounds = bounds is not None

    header_parts = [f"{'Parameter':<20s}", f"{'Value':>14s}"]
    if has_unc:
        header_parts.append(f"{'Uncertainty':>14s}")
    if has_bounds:
        header_parts.append(f"{'Lower':>12s}")
        header_parts.append(f"{'Upper':>12s}")

    header = "  ".join(header_parts)
    separator = "-" * len(header)

    lines = [header, separator]

    unc_arr = np.asarray(uncertainties, dtype=np.float64) if has_unc else None
    lower_arr = np.asarray(bounds[0], dtype=np.float64) if has_bounds else None
    upper_arr = np.asarray(bounds[1], dtype=np.float64) if has_bounds else None

    for i, name in enumerate(names):
        parts = [f"{name:<20s}", f"{values[i]:>14.6e}"]
        if has_unc and unc_arr is not None:
            parts.append(f"{unc_arr[i]:>14.6e}")
        if has_bounds and lower_arr is not None and upper_arr is not None:
            parts.append(f"{lower_arr[i]:>12.4e}")
            parts.append(f"{upper_arr[i]:>12.4e}")
        lines.append("  ".join(parts))

    return "\n".join(lines)
