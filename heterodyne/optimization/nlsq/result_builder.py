"""Build NLSQResult from raw optimizer output.

Centralizes result construction so that every strategy produces
consistent NLSQResult objects with covariance, uncertainties,
reduced chi-squared, and metadata.

Also provides homodyne-parity classes ``QualityMetrics`` and
``ResultBuilder`` (fluent builder interface) plus standalone helpers
``compute_quality_metrics``, ``compute_uncertainties``,
``normalize_nlsq_result``, and ``determine_convergence_status``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.data_prep import compute_degrees_of_freedom
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

logger = get_logger(__name__)


# =============================================================================
# Homodyne-parity: QualityMetrics dataclass
# =============================================================================


@dataclass
class QualityMetrics:
    """Quality metrics for optimization results.

    Attributes:
        chi_squared: Sum of squared residuals
        reduced_chi_squared: chi_squared / degrees of freedom
        quality_flag: 'good', 'marginal', 'poor', or 'unknown'
        n_at_bounds: Number of parameters at bounds
    """

    chi_squared: float
    reduced_chi_squared: float
    quality_flag: str
    n_at_bounds: int = 0


# =============================================================================
# Homodyne-parity: standalone helper functions
# =============================================================================


def compute_quality_metrics(
    residuals: np.ndarray,
    n_data: int,
    n_params: int,
    parameter_status: list[str] | None = None,
) -> QualityMetrics:
    """Compute quality metrics from residuals.

    Args:
        residuals: Array of residuals
        n_data: Number of data points
        n_params: Number of parameters
        parameter_status: List of parameter statuses (optional)

    Returns:
        QualityMetrics with computed values
    """
    chi_squared = float(np.sum(residuals**2))
    dof = max(n_data - n_params, 1)  # Avoid division by zero
    reduced_chi_squared = chi_squared / dof

    # Count parameters at bounds
    n_at_bounds = 0
    if parameter_status:
        n_at_bounds = sum(
            1 for s in parameter_status if s in ("at_lower_bound", "at_upper_bound")
        )

    # Determine quality flag
    if reduced_chi_squared < 2.0 and n_at_bounds == 0:
        quality_flag = "good"
    elif reduced_chi_squared < 5.0 and n_at_bounds <= 2:
        quality_flag = "marginal"
    else:
        quality_flag = "poor"

    return QualityMetrics(
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        quality_flag=quality_flag,
        n_at_bounds=n_at_bounds,
    )


def compute_uncertainties(covariance: np.ndarray | None) -> np.ndarray:
    """Extract parameter uncertainties from covariance matrix.

    Args:
        covariance: Covariance matrix (or None)

    Returns:
        Array of standard deviations (square root of diagonal)
    """
    if covariance is None or np.asarray(covariance).size == 0:
        return np.array([])

    diagonal = np.diag(np.asarray(covariance))
    # Handle negative diagonal elements (numerical issues)
    diagonal = np.maximum(diagonal, 0.0)
    return np.asarray(np.sqrt(diagonal))


def normalize_nlsq_result(
    result: Any,
    strategy_name: str = "unknown",
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Normalize various NLSQ result formats to standard format.

    NLSQ can return results in different formats depending on the function
    and version used. This normalizes them to (popt, pcov, info).

    Args:
        result: NLSQ result in any format
        strategy_name: Name of strategy for logging
        logger: Optional logger

    Returns:
        Tuple of (popt, pcov, info)

    Raises:
        TypeError: If result format is unrecognized
    """
    _log = logger or get_logger(__name__)

    # Case 1: Dict (from StreamingOptimizer or advanced functions)
    if isinstance(result, dict):
        popt_raw = result.get("x", result.get("popt"))
        if popt_raw is None:
            raise KeyError(
                f"Result dict has neither 'x' nor 'popt' key. "
                f"Available keys: {list(result.keys())}"
            )
        popt = np.asarray(popt_raw)
        pcov = np.asarray(result.get("pcov", np.eye(len(popt))))
        info: dict[str, Any] = {
            "streaming_diagnostics": result.get("streaming_diagnostics", {}),
            "success": result.get("success", True),
            "message": result.get("message", ""),
            "best_loss": result.get("best_loss", None),
            "final_epoch": result.get("final_epoch", None),
        }
        _log.debug("Normalized dict result (strategy: %s)", strategy_name)
        return popt, pcov, info

    # Case 2: Tuple with 2 or 3 elements
    if isinstance(result, tuple):
        if len(result) == 2:
            popt, pcov = result
            info = {}
            _log.debug("Normalized (popt, pcov) tuple (strategy: %s)", strategy_name)
        elif len(result) == 3:
            popt, pcov, info = result
            if not isinstance(info, dict):
                # Defensive runtime guard: NLSQ versions occasionally return
                # ``(popt, pcov, scalar_or_namedtuple)`` instead of a dict.
                _log.warning(  # type: ignore[unreachable]
                    "Info object is not a dict: %s. Converting to dict.", type(info)
                )
                info = {"raw_info": info}
            _log.debug(
                "Normalized (popt, pcov, info) tuple (strategy: %s)", strategy_name
            )
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(result)}. "
                "Expected 2 (popt, pcov) or 3 (popt, pcov, info)."
            )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
    if hasattr(result, "x") or hasattr(result, "popt"):
        popt_raw = getattr(result, "x", getattr(result, "popt", None))
        if popt_raw is None:
            raise AttributeError(
                f"Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(result)}"
            )
        popt = np.asarray(popt_raw)

        pcov_raw = getattr(result, "pcov", None)
        if pcov_raw is None:
            _log.warning("No pcov attribute in result object. Using identity matrix.")
            pcov = np.eye(len(popt))
        else:
            pcov = np.asarray(pcov_raw)

        info = {}
        for attr in ["message", "success", "nfev", "njev", "fun", "jac", "optimality"]:
            if hasattr(result, attr):
                info[attr] = getattr(result, attr)

        if hasattr(result, "info") and isinstance(result.info, dict):
            info.update(result.info)

        _log.debug(
            "Normalized object result (type: %s, strategy: %s)",
            type(result).__name__,
            strategy_name,
        )
        return np.asarray(popt), np.asarray(pcov), info

    # Case 4: Unrecognized format
    raise TypeError(
        f"Unrecognized NLSQ result format: {type(result)}. "
        "Expected tuple, dict, or object with 'x'/'popt' attributes."
    )


def determine_convergence_status(
    info: dict[str, Any],
    quality_metrics: QualityMetrics,
) -> str:
    """Determine convergence status from optimization info.

    Args:
        info: Optimization info dict
        quality_metrics: Quality metrics

    Returns:
        Convergence status: 'converged', 'max_iter', or 'failed'
    """
    # Check explicit success flag
    if "success" in info:
        if info["success"]:
            return "converged"
        # Check for max iterations
        message = str(info.get("message", "")).lower()
        if "max" in message and ("iter" in message or "fev" in message):
            return "max_iter"
        return "failed"

    # Infer from quality (no explicit success flag available)
    if quality_metrics.reduced_chi_squared < 5.0:
        logger.warning(
            "Convergence inferred from reduced_chi_squared=%.4f < 5.0 "
            "(no explicit success flag in optimizer info)",
            quality_metrics.reduced_chi_squared,
        )
        return "converged"

    return "failed"


# =============================================================================
# Homodyne-parity: ResultBuilder fluent builder
# =============================================================================


@dataclass
class ResultBuilder:
    """Builder for constructing result dictionaries.

    Provides a fluent interface for building results with proper validation.
    Homodyne parity: mirrors homodyne's ``ResultBuilder`` API while
    producing heterodyne-compatible output dicts.
    """

    parameters: np.ndarray | None = None
    covariance: np.ndarray | None = None
    n_data: int = 0
    start_time: float = field(default_factory=time.time)
    recovery_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    stratification_diagnostics: Any = None
    nlsq_diagnostics: dict[str, Any] | None = None

    def with_parameters(self, params: np.ndarray) -> ResultBuilder:
        """Set optimized parameters."""
        self.parameters = np.asarray(params)
        return self

    def with_covariance(self, cov: np.ndarray) -> ResultBuilder:
        """Set parameter covariance matrix."""
        self.covariance = np.asarray(cov)
        return self

    def with_data_size(self, n_data: int) -> ResultBuilder:
        """Set number of data points."""
        self.n_data = n_data
        return self

    def with_start_time(self, start_time: float) -> ResultBuilder:
        """Set optimization start time."""
        self.start_time = start_time
        return self

    def with_recovery_actions(self, actions: list[str]) -> ResultBuilder:
        """Set recovery actions taken."""
        self.recovery_actions = actions
        return self

    def with_info(self, info: dict[str, Any]) -> ResultBuilder:
        """Set optimization info dict."""
        self.info = info
        return self

    def with_stratification_diagnostics(self, diags: Any) -> ResultBuilder:
        """Set stratification diagnostics."""
        self.stratification_diagnostics = diags
        return self

    def with_nlsq_diagnostics(self, diags: dict[str, Any]) -> ResultBuilder:
        """Set NLSQ solver diagnostics."""
        self.nlsq_diagnostics = diags
        return self

    def with_fourier_covariance_transform(
        self,
        fourier_reparameterizer: Any,
        n_phi: int,
        n_physical: int,
    ) -> ResultBuilder:
        """Transform covariance from Fourier to per-angle space.

        Implements Fourier→per-angle covariance transformation using the
        Jacobian of the Fourier→per-angle mapping::

            Cov_per_angle = J @ Cov_fourier @ J.T

        Physical parameter covariance is preserved (not transformed).

        Parameters
        ----------
        fourier_reparameterizer : FourierReparameterizer
            The Fourier reparameterizer used during optimization.
        n_phi : int
            Number of phi angles.
        n_physical : int
            Number of physical parameters.

        Returns
        -------
        ResultBuilder
            Self for method chaining.

        Notes
        -----
        If covariance is None or fourier_reparameterizer is None,
        this method is a no-op.
        """
        if self.covariance is None or fourier_reparameterizer is None:
            return self

        if not fourier_reparameterizer.use_fourier:
            return self

        try:
            jacobian = fourier_reparameterizer.get_jacobian_transform()
        except AttributeError:
            return self

        n_fourier_coeffs = fourier_reparameterizer.n_coeffs_per_param
        n_fourier_total = 2 * n_fourier_coeffs
        n_per_angle_total = 2 * n_phi

        cov_shape = self.covariance.shape
        expected_fourier_dim = n_fourier_total + n_physical
        if cov_shape[0] != expected_fourier_dim:
            return self

        # Build block-diagonal Jacobian for full parameter vector
        full_jacobian = np.zeros(
            (n_per_angle_total + n_physical, n_fourier_total + n_physical)
        )
        full_jacobian[:n_phi, :n_fourier_coeffs] = jacobian
        full_jacobian[n_phi:n_per_angle_total, n_fourier_coeffs:n_fourier_total] = (
            jacobian
        )
        full_jacobian[n_per_angle_total:, n_fourier_total:] = np.eye(n_physical)

        transformed_cov = full_jacobian @ self.covariance @ full_jacobian.T

        expected_dim = n_per_angle_total + n_physical
        if transformed_cov.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Transformed covariance has wrong dimensions: "
                f"{transformed_cov.shape}, expected ({expected_dim}, {expected_dim})"
            )

        self.covariance = transformed_cov
        return self

    def build(
        self,
        residual_fn: Any = None,
        xdata: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build the result dictionary.

        Args:
            residual_fn: Residual function for computing chi-squared
            xdata: X data for residual computation

        Returns:
            Dictionary with all result fields

        Raises:
            ValueError: If required fields are missing
        """
        if self.parameters is None:
            raise ValueError("Parameters must be set before building result")

        n_params = len(self.parameters)
        execution_time = time.time() - self.start_time

        # Compute uncertainties
        if self.covariance is not None:
            uncertainties = compute_uncertainties(self.covariance)
        else:
            uncertainties = np.zeros(n_params)

        # Compute quality metrics
        if residual_fn is not None and xdata is not None:
            try:
                residuals = residual_fn(xdata, *self.parameters)
                quality = compute_quality_metrics(residuals, self.n_data, n_params)
            except (ValueError, RuntimeError, TypeError):
                # Fallback: NLSQ/scipy stores cost = 0.5*RSS as "fun"
                fun_val = float(self.info.get("fun", 0.0))
                chi_sq_fallback = fun_val * 2.0
                quality = QualityMetrics(
                    chi_squared=chi_sq_fallback,
                    reduced_chi_squared=chi_sq_fallback
                    / max(self.n_data - n_params, 1),
                    quality_flag="unknown",
                )
        else:
            chi_sq = float(self.info.get("fun", 0.0)) * 2.0
            quality = QualityMetrics(
                chi_squared=chi_sq,
                reduced_chi_squared=chi_sq / max(self.n_data - n_params, 1),
                quality_flag="unknown",
            )

        convergence_status = determine_convergence_status(self.info, quality)

        return {
            "parameters": self.parameters,
            "uncertainties": uncertainties,
            "covariance": self.covariance
            if self.covariance is not None
            else np.eye(n_params),
            "chi_squared": quality.chi_squared,
            "reduced_chi_squared": quality.reduced_chi_squared,
            "convergence_status": convergence_status,
            "iterations": int(self.info.get("nfev", 0)),
            "execution_time": execution_time,
            "device_info": {"type": "cpu", "name": "CPU"},
            "recovery_actions": self.recovery_actions,
            "quality_flag": quality.quality_flag,
            "stratification_diagnostics": self.stratification_diagnostics,
            "nlsq_diagnostics": self.nlsq_diagnostics,
        }


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
    success = (
        opt_result.status > 0 if hasattr(opt_result, "status") else opt_result.success
    )
    message = getattr(opt_result, "message", str(opt_result.get("message", "")))

    result = NLSQResult(
        parameters=params,
        parameter_names=parameter_names,
        success=bool(success),
        message=str(message),
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=getattr(opt_result, "nit", 0),
        n_function_evals=getattr(opt_result, "nfev", 0),
        convergence_reason=_status_to_reason(getattr(opt_result, "status", -1)),
        residuals=residuals,
        jacobian=jacobian,
        wall_time_seconds=wall_time,
        metadata=metadata or {},
    )
    logger.debug(
        "Built result: success=%s, n_iter=%d, n_fev=%d, chi2=%.4f, status=%d",
        bool(success),
        getattr(opt_result, "nit", 0),
        getattr(opt_result, "nfev", 0),
        reduced_chi2,
        getattr(opt_result, "status", -1),
    )
    return result


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


def build_result_from_nlsq(
    nlsq_result: Any,
    parameter_names: list[str],
    n_data: int,
    wall_time: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Normalize any NLSQ package return format to NLSQResult.

    Handles 4 return formats:
    - dict with 'x'/'popt', 'pcov' keys (AdaptiveHybridStreamingOptimizer)
    - (popt, pcov) tuple (curve_fit)
    - (popt, pcov, info) tuple (curve_fit with full_output)
    - object with .x/.popt, .pcov attributes (CurveFit result)

    Args:
        nlsq_result: Raw return value from an NLSQ optimization call
        parameter_names: Names for each fitted parameter
        n_data: Number of data points (for reduced chi-squared)
        wall_time: Wall-clock time in seconds
        metadata: Additional metadata to attach

    Returns:
        Populated NLSQResult

    Raises:
        TypeError: If result format is unrecognized
    """
    merged_meta: dict[str, Any] = dict(metadata) if metadata else {}
    popt: np.ndarray
    pcov: np.ndarray | None
    residuals: np.ndarray | None = None

    # Case 1: Dict (from StreamingOptimizer)
    if isinstance(nlsq_result, dict):
        popt_raw = nlsq_result.get("x", nlsq_result.get("popt"))
        if popt_raw is None:
            raise TypeError(
                "Dict result has neither 'x' nor 'popt' key. "
                f"Available keys: {list(nlsq_result.keys())}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)
        pcov_raw = nlsq_result.get("pcov")
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

        # Extract residuals if present
        fun_raw = nlsq_result.get("fun")
        if fun_raw is not None:
            residuals = np.asarray(fun_raw, dtype=np.float64)

        # Merge dict info into metadata.
        # Include nfev/nit/njev so that nlsq CurveFitResult (OptimizeResult
        # subclass, which is a dict) exposes iteration counts correctly.
        for key in (
            "streaming_diagnostics",
            "success",
            "message",
            "best_loss",
            "final_epoch",
            "nfev",
            "nit",
            "njev",
        ):
            val = nlsq_result.get(key)
            if val is not None:
                merged_meta[key] = val

        logger.debug("Normalized StreamingOptimizer dict result")

    # Case 2: Tuple with 2 or 3 elements
    elif isinstance(nlsq_result, tuple):
        if len(nlsq_result) == 2:
            popt_raw, pcov_raw = nlsq_result
            logger.debug("Normalized (popt, pcov) tuple")
        elif len(nlsq_result) == 3:
            popt_raw, pcov_raw, info = nlsq_result
            if isinstance(info, dict):
                merged_meta.update(info)
            else:
                logger.warning(
                    "Info object is not a dict: %s. Wrapping as raw_info.",
                    type(info),
                )
                merged_meta["raw_info"] = info
            logger.debug("Normalized (popt, pcov, info) tuple")
        else:
            raise TypeError(
                f"Unexpected tuple length: {len(nlsq_result)}. "
                "Expected 2 (popt, pcov) or 3 (popt, pcov, info)."
            )
        popt = np.asarray(popt_raw, dtype=np.float64)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None

    # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
    elif hasattr(nlsq_result, "x") or hasattr(nlsq_result, "popt"):
        popt_raw = getattr(nlsq_result, "x", getattr(nlsq_result, "popt", None))
        if popt_raw is None:
            raise TypeError(
                "Result object has neither 'x' nor 'popt' attribute. "
                f"Available attributes: {dir(nlsq_result)}"
            )
        popt = np.asarray(popt_raw, dtype=np.float64)

        pcov_raw = getattr(nlsq_result, "pcov", None)
        pcov = np.asarray(pcov_raw, dtype=np.float64) if pcov_raw is not None else None
        if pcov_raw is None:
            logger.warning("No pcov attribute in result object")

        # Extract residuals if present
        fun_raw = getattr(nlsq_result, "fun", None)
        if fun_raw is not None:
            residuals = np.asarray(fun_raw, dtype=np.float64)

        # Extract common attributes into metadata
        for attr in ("message", "success", "nfev", "nit", "njev", "optimality"):
            if hasattr(nlsq_result, attr):
                merged_meta[attr] = getattr(nlsq_result, attr)

        if hasattr(nlsq_result, "info") and isinstance(nlsq_result.info, dict):
            merged_meta.update(nlsq_result.info)

        logger.debug("Normalized object result (type: %s)", type(nlsq_result).__name__)

    # Case 4: Unrecognized format
    else:
        raise TypeError(
            f"Unrecognized NLSQ result format: {type(nlsq_result)}. "
            "Expected tuple, dict, or object with 'x'/'popt' attributes."
        )

    # --- Build NLSQResult ---
    n_params = len(popt)

    # Uncertainties from covariance diagonal
    uncertainties: np.ndarray | None = None
    if pcov is not None:
        uncertainties = np.sqrt(np.diag(np.abs(pcov)))

    # Cost and reduced chi-squared from residuals (if available)
    final_cost: float | None = None
    reduced_chi2: float | None = None
    if residuals is not None:
        final_cost = float(np.sum(residuals**2))
        dof = compute_degrees_of_freedom(n_data, n_params)
        reduced_chi2 = final_cost / dof

    return NLSQResult(
        parameters=popt,
        parameter_names=parameter_names,
        success=bool(merged_meta.get("success", True)),
        message=str(merged_meta.get("message", "")),
        uncertainties=uncertainties,
        covariance=pcov,
        final_cost=final_cost,
        reduced_chi_squared=reduced_chi2,
        n_iterations=int(merged_meta.get("nit", 0)),
        n_function_evals=int(merged_meta.get("nfev", 0)),
        convergence_reason=str(merged_meta.get("message", "")),
        residuals=residuals,
        wall_time_seconds=wall_time,
        metadata=merged_meta,
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
    logger.warning("NLSQ failed: %s", message)
    params = (
        initial_params if initial_params is not None else np.zeros(len(parameter_names))
    )
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
