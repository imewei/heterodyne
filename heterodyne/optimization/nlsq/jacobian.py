"""Jacobian computation and validation utilities.

Provides central finite-difference Jacobian computation, structural
validation, and analytic-vs-numerical comparison routines for debugging
and verifying NLSQ gradient calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def compute_numerical_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Jacobian via central finite differences.

    For each parameter ``p_i``, evaluates::

        J[:, i] = (r(p + h*e_i) - r(p - h*e_i)) / (2*h)

    where ``h`` is the step size for parameter *i*.

    Args:
        residual_fn: Function mapping parameters to residual vector.
        params: Parameter values at which to evaluate, shape ``(n_params,)``.
        step_sizes: Per-parameter step sizes, shape ``(n_params,)``.
            Defaults to ``max(1e-8, 1e-4 * |param|)`` for each parameter.

    Returns:
        Jacobian matrix of shape ``(n_residuals, n_params)``.
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = len(params)

    if step_sizes is None:
        step_sizes = np.maximum(1e-8, 1e-4 * np.abs(params))
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    if step_sizes.shape != params.shape:
        raise ValueError(
            f"step_sizes shape {step_sizes.shape} does not match "
            f"params shape {params.shape}"
        )

    # Evaluate at the center to determine residual size
    r0 = np.asarray(residual_fn(params), dtype=np.float64)
    n_residuals = len(r0)
    jacobian = np.zeros((n_residuals, n_params), dtype=np.float64)

    for i in range(n_params):
        h = step_sizes[i]

        p_plus = params.copy()
        p_plus[i] += h

        p_minus = params.copy()
        p_minus[i] -= h

        r_plus = np.asarray(residual_fn(p_plus), dtype=np.float64)
        r_minus = np.asarray(residual_fn(p_minus), dtype=np.float64)

        jacobian[:, i] = (r_plus - r_minus) / (2.0 * h)

    return jacobian


def validate_jacobian(
    jac: np.ndarray,
    param_names: list[str] | None = None,
) -> list[str]:
    """Validate a Jacobian matrix for structural issues.

    Checks for:
    - NaN or Inf entries
    - All-zero columns (insensitive parameters)
    - Very large condition number (ill-conditioning)
    - Rank deficiency

    Args:
        jac: Jacobian matrix, shape ``(n_residuals, n_params)``.
        param_names: Optional parameter names for diagnostic messages.

    Returns:
        List of warning strings (empty if no issues found).
    """
    jac = np.asarray(jac, dtype=np.float64)
    warnings: list[str] = []

    n_residuals, n_params = jac.shape
    names = param_names if param_names is not None else [f"p{i}" for i in range(n_params)]

    # Check for NaN/Inf
    nan_mask = np.isnan(jac)
    if np.any(nan_mask):
        nan_cols = np.where(np.any(nan_mask, axis=0))[0]
        nan_names = [names[i] for i in nan_cols if i < len(names)]
        warnings.append(f"NaN in Jacobian columns: {nan_names}")

    inf_mask = np.isinf(jac)
    if np.any(inf_mask):
        inf_cols = np.where(np.any(inf_mask, axis=0))[0]
        inf_names = [names[i] for i in inf_cols if i < len(names)]
        warnings.append(f"Inf in Jacobian columns: {inf_names}")

    # Check for zero columns (insensitive parameters)
    col_norms = np.linalg.norm(jac, axis=0)
    zero_cols = np.where(col_norms < 1e-30)[0]
    if len(zero_cols) > 0:
        zero_names = [names[i] for i in zero_cols if i < len(names)]
        warnings.append(f"Zero Jacobian columns (insensitive parameters): {zero_names}")

    # Condition number of J^T J
    if not np.any(nan_mask) and not np.any(inf_mask):
        try:
            jtj = jac.T @ jac
            cond = float(np.linalg.cond(jtj))
            if cond > 1e14:
                warnings.append(
                    f"J^T J condition number {cond:.2e} is very large "
                    "(possible rank deficiency)"
                )
            elif cond > 1e10:
                warnings.append(
                    f"J^T J condition number {cond:.2e} is moderately large "
                    "(may cause poor uncertainty estimates)"
                )
        except np.linalg.LinAlgError:
            warnings.append("Failed to compute J^T J condition number")

    # Rank check
    if not np.any(nan_mask) and not np.any(inf_mask):
        try:
            rank = int(np.linalg.matrix_rank(jac))
            if rank < n_params:
                warnings.append(
                    f"Jacobian rank {rank} < n_params {n_params} "
                    "(model is locally non-identifiable)"
                )
        except np.linalg.LinAlgError:
            warnings.append("Failed to compute Jacobian rank")

    return warnings


def compare_jacobians(
    analytic: np.ndarray,
    numerical: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> dict[str, object]:
    """Compare analytic and numerical Jacobians element-wise.

    Useful for verifying that an analytic Jacobian implementation is
    correct by comparing against central finite differences.

    Args:
        analytic: Analytic Jacobian, shape ``(m, n)``.
        numerical: Numerical Jacobian, shape ``(m, n)``.
        rtol: Relative tolerance for element-wise comparison.
        atol: Absolute tolerance for element-wise comparison.

    Returns:
        Dictionary with comparison metrics:
        - ``max_abs_diff``: Maximum absolute element-wise difference.
        - ``max_rel_diff``: Maximum relative difference (where numerical != 0).
        - ``mean_abs_diff``: Mean absolute difference.
        - ``all_close``: Whether all elements are within tolerance.
        - ``n_mismatched``: Number of elements exceeding tolerance.
        - ``worst_column``: Column index with the largest absolute difference.
        - ``worst_row``: Row index with the largest absolute difference.
    """
    analytic = np.asarray(analytic, dtype=np.float64)
    numerical = np.asarray(numerical, dtype=np.float64)

    if analytic.shape != numerical.shape:
        raise ValueError(
            f"Shape mismatch: analytic {analytic.shape} vs "
            f"numerical {numerical.shape}"
        )

    abs_diff = np.abs(analytic - numerical)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative difference where numerical is non-zero
    denom = np.abs(numerical)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(denom > atol, abs_diff / denom, 0.0)
    max_rel_diff = float(np.max(rel_diff))

    # Check all_close using numpy's standard
    all_close = bool(np.allclose(analytic, numerical, rtol=rtol, atol=atol))

    # Count mismatched elements
    mismatch_mask = abs_diff > (atol + rtol * np.abs(numerical))
    n_mismatched = int(np.sum(mismatch_mask))

    # Worst element location
    worst_flat = int(np.argmax(abs_diff))
    worst_row, worst_col = np.unravel_index(worst_flat, abs_diff.shape)

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_abs_diff": mean_abs_diff,
        "all_close": all_close,
        "n_mismatched": n_mismatched,
        "worst_column": int(worst_col),
        "worst_row": int(worst_row),
    }
