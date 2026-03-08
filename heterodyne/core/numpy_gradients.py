"""NumPy finite-difference gradient fallback for heterodyne model.

Provides numerical gradient, Jacobian, and Hessian computation using central
finite differences. Intended for validation against JAX autodiff gradients
or as a fallback when JAX is unavailable.

Central difference formulas:
    gradient:  (f(x+h) - f(x-h)) / (2h)
    Hessian:   (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4 h_i h_j)
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
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
    return np.maximum(1e-8, 1e-4 * np.abs(params))  # type: ignore[no-any-return]


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


# ---------------------------------------------------------------------------
# Advanced numerical differentiation (Task 1.6)
# ---------------------------------------------------------------------------

class DifferentiationMethod(Enum):
    """Supported numerical differentiation methods."""

    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"
    COMPLEX_STEP = "complex_step"
    RICHARDSON = "richardson"
    ADAPTIVE = "adaptive"


@dataclass
class DifferentiationConfig:
    """Configuration for numerical differentiation.

    Attributes:
        method: Differentiation method to use.
        step_size: Fixed step size. None means auto-compute from parameter
            magnitudes via ``_default_step_sizes``.
        richardson_terms: Number of extrapolation levels for Richardson method.
        error_tolerance: Tolerance used by adaptive and Richardson methods.
        use_parallel: If True and method is CENTRAL, dispatch to
            ``compute_gradient_parallel``.
        n_workers: Thread-pool size for parallel mode. None auto-detects from
            ``os.cpu_count()``.
    """

    method: DifferentiationMethod = DifferentiationMethod.CENTRAL
    step_size: float | None = None
    richardson_terms: int = 4
    error_tolerance: float = 1e-10
    use_parallel: bool = False
    n_workers: int | None = None


@dataclass
class GradientResult:
    """Container for gradient computation results.

    Attributes:
        gradient: The computed gradient array.
        error_estimate: Estimated numerical error (method-dependent).
        method_used: String label of the method that produced this result.
        n_function_evals: Total number of ``fn`` evaluations performed.
        elapsed_seconds: Wall-clock time for the computation.
    """

    gradient: np.ndarray = field(default_factory=lambda: np.empty(0))
    error_estimate: float | None = None
    method_used: str = ""
    n_function_evals: int = 0
    elapsed_seconds: float = 0.0


# -- Primitive difference schemes -------------------------------------------


def _forward_difference(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_sizes: np.ndarray,
) -> np.ndarray:
    """Compute gradient via forward differences.

    For each parameter *i*::

        grad[i] = (f(x + h*e_i) - f(x)) / h

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        step_sizes: Per-parameter step sizes, shape ``(n_params,)``.

    Returns:
        Gradient array, shape ``(n_params,)``.
    """
    params = np.asarray(params, dtype=np.float64).copy()
    step_sizes = np.asarray(step_sizes, dtype=np.float64)
    n_params = params.shape[0]

    f0 = fn(params)
    gradient = np.empty(n_params, dtype=np.float64)

    for i in range(n_params):
        h = step_sizes[i]
        orig = params[i]
        params[i] = orig + h
        try:
            gradient[i] = (fn(params) - f0) / h
        finally:
            params[i] = orig

    return gradient


def _backward_difference(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_sizes: np.ndarray,
) -> np.ndarray:
    """Compute gradient via backward differences.

    For each parameter *i*::

        grad[i] = (f(x) - f(x - h*e_i)) / h

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        step_sizes: Per-parameter step sizes, shape ``(n_params,)``.

    Returns:
        Gradient array, shape ``(n_params,)``.
    """
    params = np.asarray(params, dtype=np.float64).copy()
    step_sizes = np.asarray(step_sizes, dtype=np.float64)
    n_params = params.shape[0]

    f0 = fn(params)
    gradient = np.empty(n_params, dtype=np.float64)

    for i in range(n_params):
        h = step_sizes[i]
        orig = params[i]
        params[i] = orig - h
        try:
            gradient[i] = (f0 - fn(params)) / h
        finally:
            params[i] = orig

    return gradient


# -- Richardson extrapolation -----------------------------------------------


def _richardson_extrapolation(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    param_idx: int,
    step_size: float,
    n_terms: int = 4,
) -> tuple[float, float]:
    """Richardson extrapolation of a central-difference derivative.

    Implements Neville's algorithm: central differences are computed at
    geometrically decreasing step sizes ``h, h/2, h/4, ...`` and then
    extrapolated to ``h -> 0`` via a triangular table.

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        param_idx: Index of the parameter to differentiate with respect to.
        step_size: Initial step size *h*.
        n_terms: Number of extrapolation levels (default 4).

    Returns:
        ``(derivative_estimate, error_estimate)`` where the error estimate is
        the absolute difference between the last two diagonal entries in the
        Neville table.
    """
    params = np.asarray(params, dtype=np.float64)

    # Build the first column: central differences at h, h/2, h/4, ...
    table: list[list[float]] = []
    h = step_size
    for _ in range(n_terms):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[param_idx] += h
        p_minus[param_idx] -= h
        d = (fn(p_plus) - fn(p_minus)) / (2.0 * h)
        table.append([d])
        h /= 2.0

    # Fill the Neville triangle
    for j in range(1, n_terms):
        factor = 4.0**j
        for i in range(j, n_terms):
            improved = (factor * table[i][j - 1] - table[i - 1][j - 1]) / (
                factor - 1.0
            )
            table[i].append(improved)

    best = table[-1][-1]
    if n_terms >= 2:
        prev = table[-2][-1] if len(table[-2]) == len(table[-1]) else table[-1][-2]
        err = abs(best - prev)
    else:
        err = float("inf")

    return best, err


# -- Complex-step derivative ------------------------------------------------


def _complex_step_derivative(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_size: float = 1e-20,
) -> np.ndarray:
    """Compute gradient via the complex-step method.

    For each parameter *i*::

        grad[i] = Im(f(x + i*h*e_i)) / h

    This yields machine-precision derivatives for real-analytic functions
    because there is no catastrophic cancellation.

    .. note::
       ``fn`` **must** support complex-valued inputs. If it uses operations
       that are not defined for complex numbers (e.g., ``np.real`` inside the
       function), the result will be incorrect.

    Args:
        fn: Scalar-valued function that accepts complex arrays.
        params: Parameter array, shape ``(n_params,)``.
        step_size: Imaginary perturbation magnitude (default ``1e-20``).

    Returns:
        Gradient array, shape ``(n_params,)``.
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]
    gradient = np.empty(n_params, dtype=np.float64)

    params_complex = params.astype(np.complex128)
    for i in range(n_params):
        params_complex[i] += 1j * step_size
        try:
            gradient[i] = np.imag(fn(params_complex)) / step_size
        finally:
            params_complex[i] -= 1j * step_size

    return gradient


# -- Adaptive gradient ------------------------------------------------------


def compute_adaptive_gradient(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    initial_step: float | None = None,
    rtol: float = 1e-8,
) -> GradientResult:
    """Compute gradient with adaptive step-size selection.

    For each parameter, central differences are computed at step sizes *h* and
    *h/2*.  If they agree within *rtol*, the finer estimate is accepted.
    Otherwise the step is halved and the process repeats (up to 10 iterations).

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        initial_step: Starting step size per parameter. If None, uses
            ``_default_step_sizes``.
        rtol: Relative tolerance for convergence between successive estimates.

    Returns:
        :class:`GradientResult` with ``error_estimate`` equal to the maximum
        absolute difference between the last two estimates across all
        parameters.
    """
    t0 = time.monotonic()
    params = np.asarray(params, dtype=np.float64).copy()
    n_params = params.shape[0]

    if initial_step is not None:
        steps = np.full(n_params, initial_step, dtype=np.float64)
    else:
        steps = _default_step_sizes(params)

    gradient = np.empty(n_params, dtype=np.float64)
    max_err = 0.0
    total_evals = 0
    max_iters = 10

    for i in range(n_params):
        h = steps[i]
        orig = params[i]

        # Initial central difference at h
        params[i] = orig + h
        f_plus = fn(params)
        params[i] = orig - h
        f_minus = fn(params)
        params[i] = orig
        prev_est = (f_plus - f_minus) / (2.0 * h)
        total_evals += 2

        converged = False
        for _ in range(max_iters):
            h /= 2.0
            params[i] = orig + h
            f_plus = fn(params)
            params[i] = orig - h
            f_minus = fn(params)
            params[i] = orig
            cur_est = (f_plus - f_minus) / (2.0 * h)
            total_evals += 2

            denom = max(abs(cur_est), abs(prev_est), 1e-15)
            if abs(cur_est - prev_est) <= rtol * denom:
                converged = True
                gradient[i] = cur_est
                max_err = max(max_err, abs(cur_est - prev_est))
                break
            prev_est = cur_est

        if not converged:
            gradient[i] = cur_est
            max_err = max(max_err, abs(cur_est - prev_est))
            logger.debug(
                "Adaptive gradient did not converge for param %d after %d "
                "halvings",
                i,
                max_iters,
            )

    elapsed = time.monotonic() - t0
    logger.debug(
        "Adaptive gradient: n_params=%d, n_evals=%d, max_err=%.2e, "
        "elapsed=%.4fs",
        n_params,
        total_evals,
        max_err,
        elapsed,
    )

    return GradientResult(
        gradient=gradient,
        error_estimate=max_err,
        method_used="adaptive",
        n_function_evals=total_evals,
        elapsed_seconds=elapsed,
    )


# -- Parallel gradient ------------------------------------------------------


def compute_gradient_parallel(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
    n_workers: int | None = None,
) -> GradientResult:
    """Compute central-difference gradient using a thread pool.

    Each parameter's finite difference is computed independently in a separate
    thread, which can speed up wall-clock time when ``fn`` releases the GIL
    (e.g., calls into compiled C/Fortran code).

    .. note::
       Threading only provides a wall-clock speedup when ``fn`` releases the
       GIL.  This is the case for many C/Fortran extensions (NumPy ufuncs,
       SciPy routines, etc.) and I/O-bound workloads.  For pure-Python ``fn``
       the GIL prevents true parallelism and the thread-pool overhead may
       actually *increase* total runtime.

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        step_sizes: Per-parameter step sizes, shape ``(n_params,)``. If None,
            auto-computed.
        n_workers: Number of threads. Defaults to ``min(8, os.cpu_count())``.

    Returns:
        :class:`GradientResult` with method ``"central_parallel"``.
    """
    t0 = time.monotonic()
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]

    if step_sizes is None:
        step_sizes = _default_step_sizes(params)
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 1)

    def _central_diff_i(i: int) -> float:
        h = step_sizes[i]
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        return float((fn(p_plus) - fn(p_minus)) / (2.0 * h))

    gradient = np.empty(n_params, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_central_diff_i, i): i for i in range(n_params)}
        for future in as_completed(futures):
            i = futures[future]
            gradient[i] = future.result()

    elapsed = time.monotonic() - t0
    logger.debug(
        "Parallel gradient: n_params=%d, n_workers=%d, elapsed=%.4fs",
        n_params,
        n_workers,
        elapsed,
    )

    return GradientResult(
        gradient=gradient,
        error_estimate=None,
        method_used="central_parallel",
        n_function_evals=2 * n_params,
        elapsed_seconds=elapsed,
    )


# -- Chunked Jacobian -------------------------------------------------------


def compute_jacobian_chunked(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    step_sizes: np.ndarray | None = None,
    chunk_size: int = 10,
) -> np.ndarray:
    """Compute Jacobian in chunks to limit peak memory usage.

    For large parameter spaces the full set of perturbed residual vectors can
    be prohibitively large.  This function processes ``chunk_size`` parameters
    at a time so that at most ``2 * chunk_size`` perturbed residual vectors
    reside in memory simultaneously.

    Args:
        residual_fn: Vector-valued function ``r(params) -> array(n_residuals,)``.
        params: Parameter array, shape ``(n_params,)``.
        step_sizes: Per-parameter step sizes, shape ``(n_params,)``. If None,
            auto-computed.
        chunk_size: Number of parameters to process per chunk.

    Returns:
        Jacobian matrix, shape ``(n_residuals, n_params)``.
    """
    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]

    if step_sizes is None:
        step_sizes = _default_step_sizes(params)
    else:
        step_sizes = np.asarray(step_sizes, dtype=np.float64)

    # Probe output size
    r0 = np.asarray(residual_fn(params), dtype=np.float64)
    n_residuals = r0.shape[0]

    jacobian = np.empty((n_residuals, n_params), dtype=np.float64)

    for start in range(0, n_params, chunk_size):
        end = min(start + chunk_size, n_params)
        for j in range(start, end):
            h = step_sizes[j]
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[j] += h
            p_minus[j] -= h
            r_plus = np.asarray(residual_fn(p_plus), dtype=np.float64)
            r_minus = np.asarray(residual_fn(p_minus), dtype=np.float64)
            jacobian[:, j] = (r_plus - r_minus) / (2.0 * h)

    logger.debug(
        "Computed chunked Jacobian: shape=(%d, %d), chunk_size=%d",
        n_residuals,
        n_params,
        chunk_size,
    )

    return jacobian


# -- Typed validation wrapper -----------------------------------------------


def validate_gradient_accuracy(
    analytic_grad: np.ndarray,
    numerical_grad: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> GradientResult:
    """Typed wrapper around :func:`validate_gradient`.

    Calls ``validate_gradient()`` and packages the dict result into a
    :class:`GradientResult`.

    Args:
        analytic_grad: Gradient from analytic/autodiff computation.
        numerical_grad: Gradient from finite differences.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        :class:`GradientResult` with ``gradient=analytic_grad``,
        ``error_estimate`` set to the maximum absolute error, and
        ``method_used="validation"``.
    """
    result = validate_gradient(analytic_grad, numerical_grad, rtol=rtol, atol=atol)
    max_abs_err = float(result["max_abs_error"])  # type: ignore[arg-type]

    return GradientResult(
        gradient=np.asarray(analytic_grad, dtype=np.float64),
        error_estimate=max_abs_err,
        method_used="validation",
        n_function_evals=0,
        elapsed_seconds=0.0,
    )


# -- Main dispatch -----------------------------------------------------------


def compute_gradient(
    fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    config: DifferentiationConfig | None = None,
) -> GradientResult:
    """Compute the gradient of *fn* using the method specified by *config*.

    This is the main entry point that dispatches to the appropriate
    differentiation routine:

    ==============================  ====================================
    ``config.method``               Dispatches to
    ==============================  ====================================
    ``FORWARD``                     :func:`_forward_difference`
    ``BACKWARD``                    :func:`_backward_difference`
    ``CENTRAL``                     :func:`compute_gradient_finite_diff`
    ``COMPLEX_STEP``                :func:`_complex_step_derivative`
    ``RICHARDSON``                  :func:`_richardson_extrapolation`
    ``ADAPTIVE``                    :func:`compute_adaptive_gradient`
    ==============================  ====================================

    If ``config.use_parallel`` is True and the method is ``CENTRAL``, the
    parallel variant :func:`compute_gradient_parallel` is used instead.

    Args:
        fn: Scalar-valued function ``f(params) -> float``.
        params: Parameter array, shape ``(n_params,)``.
        config: Differentiation configuration. If None, uses default
            (central differences).

    Returns:
        :class:`GradientResult` containing the gradient, timing, and
        evaluation count.
    """
    if config is None:
        config = DifferentiationConfig()

    params = np.asarray(params, dtype=np.float64)
    n_params = params.shape[0]
    method = config.method

    # Resolve step sizes
    if config.step_size is not None:
        step_sizes = np.full(n_params, config.step_size, dtype=np.float64)
    else:
        step_sizes = _default_step_sizes(params)

    error_estimate: float | None = None
    n_evals = 0
    t0 = time.monotonic()

    if method is DifferentiationMethod.FORWARD:
        grad = _forward_difference(fn, params, step_sizes)
        n_evals = n_params + 1
        method_label = "forward"

    elif method is DifferentiationMethod.BACKWARD:
        grad = _backward_difference(fn, params, step_sizes)
        n_evals = n_params + 1
        method_label = "backward"

    elif method is DifferentiationMethod.CENTRAL:
        if config.use_parallel:
            return compute_gradient_parallel(
                fn, params, step_sizes=step_sizes, n_workers=config.n_workers
            )
        grad = compute_gradient_finite_diff(fn, params, step_sizes=step_sizes)
        n_evals = 2 * n_params
        method_label = "central"

    elif method is DifferentiationMethod.COMPLEX_STEP:
        step = config.step_size if config.step_size is not None else 1e-20
        grad = _complex_step_derivative(fn, params, step_size=step)
        n_evals = n_params
        method_label = "complex_step"

    elif method is DifferentiationMethod.RICHARDSON:
        grad = np.empty(n_params, dtype=np.float64)
        errors = np.empty(n_params, dtype=np.float64)
        for i in range(n_params):
            d, e = _richardson_extrapolation(
                fn,
                params,
                param_idx=i,
                step_size=step_sizes[i],
                n_terms=config.richardson_terms,
            )
            grad[i] = d
            errors[i] = e
        error_estimate = float(np.max(errors))
        # Each Richardson level uses 2 fn evals (central diff),
        # n_terms levels per parameter
        n_evals = 2 * config.richardson_terms * n_params
        method_label = "richardson"

    elif method is DifferentiationMethod.ADAPTIVE:
        initial = config.step_size
        return compute_adaptive_gradient(
            fn, params, initial_step=initial, rtol=config.error_tolerance
        )

    else:  # pragma: no cover
        raise ValueError(f"Unknown differentiation method: {method}")

    elapsed = time.monotonic() - t0

    return GradientResult(
        gradient=grad,
        error_estimate=error_estimate,
        method_used=method_label,
        n_function_evals=n_evals,
        elapsed_seconds=elapsed,
    )
