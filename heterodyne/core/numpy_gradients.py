"""NumPy finite-difference gradient fallback for heterodyne model.

Provides numerical gradient, Jacobian, and Hessian computation using central
finite differences. Intended for validation against JAX autodiff gradients
or as a fallback when JAX is unavailable.

Central difference formulas:
    gradient:  (f(x+h) - f(x-h)) / (2h)
    Hessian:   (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4 h_i h_j)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def _default_step_sizes(params: np.ndarray) -> np.ndarray:
    """Compute default step sizes based on parameter magnitudes.

    Uses max(1e-8, 1e-4 * |param|) to scale step sizes appropriately
    for parameters of different magnitudes.

    Args:
        params: Parameter array, shape (n_params,)

    Returns:
        Step size array, shape (n_params,)
    """
    return np.maximum(1e-8, 1e-4 * np.abs(params))


def compute_gradient_finite_diff(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute gradient of a scalar function via central finite differences.

    For each parameter i:
        grad[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)

    where e_i is the i-th unit vector and h is the step size.

    Args:
        fn: Scalar-valued function f(params) -> float.
        params: Parameter array, shape (n_params,).
        step_sizes: Per-parameter step sizes, shape (n_params,).
            If None, uses max(1e-8, 1e-4 * |param|).

    Returns:
        Gradient array, shape (n_params,).
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]

    if step_sizes is None:
        step_sizes = _default_step_sizes(params)
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    gradient = np.empty(n_params, dtype=np.float64)

    for i in range(n_params):
        h = step_sizes[i]
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h

        f_plus = fn(params_plus)
        f_minus = fn(params_minus)
        gradient[i] = (f_plus - f_minus) / (2.0 * h)

    logger.debug(
        "Computed finite-difference gradient: n_params=%d, max_step=%.2e",
        n_params,
        np.max(step_sizes),
    )

    return gradient


def compute_jacobian_finite_diff(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Jacobian of a vector-valued function via central finite differences.

    For each parameter j:
        J[:, j] = (r(x + h*e_j) - r(x - h*e_j)) / (2*h)

    where r is the residual function and e_j is the j-th unit vector.

    Args:
        residual_fn: Vector-valued function r(params) -> array of shape (n_residuals,).
        params: Parameter array, shape (n_params,).
        step_sizes: Per-parameter step sizes, shape (n_params,).
            If None, uses max(1e-8, 1e-4 * |param|).

    Returns:
        Jacobian matrix, shape (n_residuals, n_params).
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]

    if step_sizes is None:
        step_sizes = _default_step_sizes(params)
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    # Evaluate at baseline to determine output dimension
    r0 = np.asarray(residual_fn(params), dtype=np.float64)
    n_residuals = r0.shape[0]

    jacobian = np.empty((n_residuals, n_params), dtype=np.float64)

    for j in range(n_params):
        h = step_sizes[j]
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += h
        params_minus[j] -= h

        r_plus = np.asarray(residual_fn(params_plus), dtype=np.float64)
        r_minus = np.asarray(residual_fn(params_minus), dtype=np.float64)
        jacobian[:, j] = (r_plus - r_minus) / (2.0 * h)

    logger.debug(
        "Computed finite-difference Jacobian: shape=(%d, %d)",
        n_residuals,
        n_params,
    )

    return jacobian


def compute_hessian_finite_diff(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Hessian of a scalar function via second-order finite differences.

    Diagonal elements:
        H[i,i] = (f(x+h_i) - 2f(x) + f(x-h_i)) / h_i^2

    Off-diagonal elements (using the cross-difference formula):
        H[i,j] = (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4 h_i h_j)

    The result is symmetrized: H = (H + H^T) / 2.

    Args:
        cost_fn: Scalar-valued function f(params) -> float.
        params: Parameter array, shape (n_params,).
        step_sizes: Per-parameter step sizes, shape (n_params,).
            If None, uses max(1e-8, 1e-4 * |param|).

    Returns:
        Hessian matrix, shape (n_params, n_params).
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]

    if step_sizes is None:
        step_sizes = _default_step_sizes(params)
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    f0 = cost_fn(params)
    hessian = np.empty((n_params, n_params), dtype=np.float64)

    # Diagonal elements: second-order central differences
    for i in range(n_params):
        h_i = step_sizes[i]
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h_i
        p_minus[i] -= h_i
        hessian[i, i] = (cost_fn(p_plus) - 2.0 * f0 + cost_fn(p_minus)) / (h_i * h_i)

    # Off-diagonal elements: cross finite differences
    for i in range(n_params):
        h_i = step_sizes[i]
        for j in range(i + 1, n_params):
            h_j = step_sizes[j]

            p_pp = params.copy()
            p_pm = params.copy()
            p_mp = params.copy()
            p_mm = params.copy()

            p_pp[i] += h_i
            p_pp[j] += h_j
            p_pm[i] += h_i
            p_pm[j] -= h_j
            p_mp[i] -= h_i
            p_mp[j] += h_j
            p_mm[i] -= h_i
            p_mm[j] -= h_j

            cross = (
                cost_fn(p_pp) - cost_fn(p_pm) - cost_fn(p_mp) + cost_fn(p_mm)
            ) / (4.0 * h_i * h_j)

            hessian[i, j] = cross
            hessian[j, i] = cross

    logger.debug(
        "Computed finite-difference Hessian: shape=(%d, %d), "
        "n_function_evals=%d",
        n_params,
        n_params,
        1 + 2 * n_params + 4 * n_params * (n_params - 1) // 2,
    )

    return hessian


def validate_gradient(
    analytic_grad: np.ndarray,
    numerical_grad: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Compare analytic (e.g., JAX autodiff) and numerical gradients.

    Computes element-wise absolute differences and checks whether all
    elements satisfy:
        |analytic - numerical| <= atol + rtol * |numerical|

    Args:
        analytic_grad: Gradient from analytic/autodiff computation, shape (n,).
        numerical_grad: Gradient from finite differences, shape (n,).
        rtol: Relative tolerance for the comparison.
        atol: Absolute tolerance for the comparison.

    Returns:
        Dictionary with:
            - max_abs_error: Maximum absolute difference.
            - mean_abs_error: Mean absolute difference.
            - max_rel_error: Maximum relative difference (relative to
              max(|analytic|, |numerical|, 1e-10)).
            - worst_param_idx: Index of the parameter with largest absolute error.
            - all_close: True if all elements are within tolerance.
    """
    analytic = np.asarray(analytic_grad, dtype=np.float64)
    numerical = np.asarray(numerical_grad, dtype=np.float64)

    if analytic.shape != numerical.shape:
        raise ValueError(
            f"Shape mismatch: analytic {analytic.shape} vs numerical {numerical.shape}"
        )

    abs_diff = np.abs(analytic - numerical)
    max_magnitude = np.maximum(
        np.maximum(np.abs(analytic), np.abs(numerical)), 1e-10
    )
    rel_diff = abs_diff / max_magnitude

    worst_idx = int(np.argmax(abs_diff))
    all_close = bool(np.allclose(analytic, numerical, rtol=rtol, atol=atol))

    result: dict[str, object] = {
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "max_rel_error": float(np.max(rel_diff)),
        "worst_param_idx": worst_idx,
        "all_close": all_close,
    }

    if all_close:
        logger.debug(
            "Gradient validation PASSED: max_abs_error=%.2e, max_rel_error=%.2e",
            result["max_abs_error"],
            result["max_rel_error"],
        )
    else:
        logger.warning(
            "Gradient validation FAILED at param %d: max_abs_error=%.2e, "
            "max_rel_error=%.2e (rtol=%.1e, atol=%.1e)",
            worst_idx,
            result["max_abs_error"],
            result["max_rel_error"],
            rtol,
            atol,
        )

    return result
