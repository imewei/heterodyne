"""Build NLSQResult from raw optimizer output.

Centralizes result construction so that every strategy produces
consistent NLSQResult objects with covariance, uncertainties,
reduced chi-squared, and metadata.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.data_prep import compute_degrees_of_freedom
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

logger = get_logger(__name__)


def build_result_from_scipy(
    opt_result: OptimizeResult,
    parameter_names: list[str],
    n_data: int,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct NLSQResult from scipy.optimize.least_squares output.

    Args:
        opt_result: Raw scipy OptimizeResult
        parameter_names: Names for each fitted parameter
        n_data: Number of data points (for reduced chi²)
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata to attach

    Returns:
        Populated NLSQResult
    """
    params = np.asarray(opt_result.x, dtype=np.float64)
    n_params = len(params)

    # Covariance from Jacobian: cov ≈ (J^T J)^{-1} * s²
    covariance = None
    uncertainties = None
    jacobian = getattr(opt_result, "jac", None)

    if jacobian is not None:
        covariance = _compute_covariance(jacobian, opt_result.fun, n_data, n_params)
        if covariance is not None:
            uncertainties = np.sqrt(np.diag(np.abs(covariance)))

    # Reduced chi-squared
    residuals = np.asarray(opt_result.fun, dtype=np.float64)
    cost = float(np.sum(residuals**2))
    dof = compute_degrees_of_freedom(n_data, n_params)
    reduced_chi2 = cost / dof

    # Map scipy status to success
    success = opt_result.status > 0 if hasattr(opt_result, "status") else opt_result.success
    message = getattr(opt_result, "message", str(opt_result.get("message", "")))

    return NLSQResult(
        parameters=params,
        parameter_names=parameter_names,
        success=bool(success),
        message=str(message),
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=getattr(opt_result, "njev", 0),
        n_function_evals=getattr(opt_result, "nfev", 0),
        convergence_reason=_status_to_reason(getattr(opt_result, "status", -1)),
        residuals=residuals,
        jacobian=jacobian,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )


def build_result_from_arrays(
    parameters: np.ndarray,
    parameter_names: list[str],
    residuals: np.ndarray,
    n_data: int,
    success: bool = True,
    message: str = "",
    jacobian: np.ndarray | None = None,
    n_iterations: int = 0,
    n_function_evals: int = 0,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct NLSQResult from raw arrays (for non-scipy backends).

    Args:
        parameters: Fitted parameter values
        parameter_names: Names in order
        residuals: Residual vector
        n_data: Number of data points
        success: Whether optimization converged
        message: Status message
        jacobian: Optional Jacobian at solution
        n_iterations: Number of iterations
        n_function_evals: Number of function evaluations
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata

    Returns:
        Populated NLSQResult
    """
    params = np.asarray(parameters, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    n_params = len(params)

    covariance = None
    uncertainties = None
    if jacobian is not None:
        covariance = _compute_covariance(jacobian, residuals, n_data, n_params)
        if covariance is not None:
            uncertainties = np.sqrt(np.diag(np.abs(covariance)))

    cost = float(np.sum(residuals**2))
    dof = compute_degrees_of_freedom(n_data, n_params)
    reduced_chi2 = cost / dof

    return NLSQResult(
        parameters=params,
        parameter_names=parameter_names,
        success=success,
        message=message,
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=n_iterations,
        n_function_evals=n_function_evals,
        convergence_reason=message,
        residuals=residuals,
        jacobian=jacobian,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )


def build_failed_result(
    parameter_names: list[str],
    message: str,
    initial_params: np.ndarray | None = None,
    wall_time: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Construct a failed NLSQResult.

    Args:
        parameter_names: Names for parameters
        message: Failure description
        initial_params: Initial guess (returned as "best" params)
        wall_time: Wall-clock time before failure
        metadata: Additional metadata

    Returns:
        NLSQResult with success=False
    """
    params = initial_params if initial_params is not None else np.zeros(len(parameter_names))
    return NLSQResult(
        parameters=np.asarray(params, dtype=np.float64),
        parameter_names=parameter_names,
        success=False,
        message=message,
        convergence_reason=message,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )


class TimedContext:
    """Context manager for timing optimizer calls.

    Usage::

        timer = TimedContext()
        with timer:
            result = optimizer.run(...)
        print(f"Took {timer.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> TimedContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def _compute_covariance(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    n_data: int,
    n_params: int,
) -> np.ndarray | None:
    """Compute parameter covariance from Jacobian.

    Uses the Gauss-Newton approximation:
        cov = s² * (J^T J)^{-1}
    where s² = sum(residuals²) / (n_data - n_params).

    Args:
        jacobian: Jacobian matrix at solution, shape (n_residuals, n_params)
        residuals: Residual vector at solution
        n_data: Number of independent data points
        n_params: Number of parameters

    Returns:
        Covariance matrix of shape (n_params, n_params), or None on failure
    """
    try:
        jac = np.asarray(jacobian, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64)

        # J^T J
        jtj = jac.T @ jac

        # Regularize if near-singular
        cond = np.linalg.cond(jtj)
        if cond > 1e14:
            logger.warning(
                "J^T J condition number %.2e; adding Tikhonov regularization", cond
            )
            jtj += 1e-10 * np.eye(n_params)

        jtj_inv = np.linalg.inv(jtj)

        # Variance estimate
        dof = max(n_data - n_params, 1)
        s2 = float(np.sum(res**2)) / dof

        return s2 * jtj_inv

    except np.linalg.LinAlgError:
        logger.warning("Failed to compute covariance: singular J^T J")
        return None


def _status_to_reason(status: int) -> str:
    """Map scipy least_squares status codes to human-readable reasons."""
    reasons = {
        -1: "Improper input parameters",
        0: "Maximum function evaluations reached",
        1: "gtol convergence (gradient sufficiently small)",
        2: "xtol convergence (parameter change sufficiently small)",
        3: "ftol convergence (cost change sufficiently small)",
        4: "Both xtol and ftol convergence",
    }
    return reasons.get(status, f"Unknown status: {status}")
