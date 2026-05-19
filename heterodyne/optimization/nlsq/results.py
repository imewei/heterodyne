"""Result container for NLSQ optimization.

Provides:
- ``NLSQResult`` — primary heterodyne result dataclass
- ``OptimizationResult`` — homodyne-parity result (dict-builder output)
- ``FallbackInfo`` — adapter-to-wrapper fallback tracking
- ``FunctionEvaluationCounter`` — callable invocation counter
- ``UseSequentialOptimization`` — marker for sequential fallback strategy
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class NLSQResult:
    """Result of NLSQ optimization.

    Contains fitted parameters, uncertainties, and fit quality metrics.
    """

    # Core results
    parameters: np.ndarray  # Fitted parameter values
    parameter_names: list[str]  # Names in order
    success: bool  # Whether optimization succeeded
    message: str  # Status message

    # Uncertainties (from covariance matrix)
    uncertainties: np.ndarray | None = None
    covariance: np.ndarray | None = None

    # Fit quality metrics
    final_cost: float | None = None
    reduced_chi_squared: float | None = None
    n_iterations: int = 0
    n_function_evals: int = 0
    convergence_reason: str = ""

    # Residuals and Jacobian (optional, can be large)
    residuals: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    fitted_correlation: np.ndarray | None = None

    # Timing
    wall_time_seconds: float | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """Number of fitted parameters."""
        return len(self.parameters)

    @property
    def params_dict(self) -> dict[str, float]:
        """Parameters as dictionary."""
        return {
            name: float(self.parameters[i])
            for i, name in enumerate(self.parameter_names)
        }

    def get_param(self, name: str) -> float:
        """Get parameter value by name.

        Args:
            name: Parameter name

        Returns:
            Parameter value

        Raises:
            KeyError: If parameter not found
        """
        try:
            idx = self.parameter_names.index(name)
            return float(self.parameters[idx])
        except ValueError:
            raise KeyError(f"Parameter '{name}' not found") from None

    def get_uncertainty(self, name: str) -> float | None:
        """Get uncertainty for parameter by name.

        Args:
            name: Parameter name

        Returns:
            Uncertainty or None if not available
        """
        if self.uncertainties is None:
            return None
        try:
            idx = self.parameter_names.index(name)
            return float(self.uncertainties[idx])
        except ValueError:
            return None

    def get_correlation_matrix(self) -> np.ndarray | None:
        """Compute correlation matrix from covariance.

        Returns:
            Correlation matrix or None if covariance not available
        """
        if self.covariance is None:
            return None

        std = np.sqrt(np.diag(self.covariance))
        std_outer = np.outer(std, std)
        # Avoid division by zero
        std_outer = np.where(std_outer > 0, std_outer, 1.0)
        return self.covariance / std_outer

    def validate(self) -> list[str]:
        """Validate result quality.

        Returns:
            List of warning/error messages
        """
        warnings = []

        if not self.success:
            warnings.append(f"Optimization failed: {self.message}")

        if self.reduced_chi_squared is not None:
            if self.reduced_chi_squared > 2.0:
                warnings.append(
                    f"Poor fit: χ²_red = {self.reduced_chi_squared:.2f} > 2"
                )
            elif self.reduced_chi_squared < 0.5:
                warnings.append(
                    f"Possible overfit: χ²_red = {self.reduced_chi_squared:.2f} < 0.5"
                )

        if self.uncertainties is not None:
            for name, val, unc in zip(
                self.parameter_names, self.parameters, self.uncertainties, strict=True
            ):
                if val != 0 and abs(unc / val) > 1.0:
                    warnings.append(
                        f"Large uncertainty: {name} = {val:.3e} ± {unc:.3e}"
                    )

        # Check for highly correlated parameters
        corr = self.get_correlation_matrix()
        if corr is not None:
            n = len(self.parameter_names)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr[i, j]) > 0.95:
                        warnings.append(
                            f"Highly correlated: {self.parameter_names[i]} and "
                            f"{self.parameter_names[j]} (r = {corr[i, j]:.3f})"
                        )

        return warnings

    def summary(self) -> str:
        """Generate summary string.

        Returns:
            Multi-line summary
        """
        lines = [
            "NLSQ Fit Result",
            "=" * 50,
            f"Success: {self.success}",
            f"Message: {self.message}",
            "",
            "Parameters:",
            "-" * 50,
        ]

        for i, name in enumerate(self.parameter_names):
            val = self.parameters[i]
            if self.uncertainties is not None:
                unc = self.uncertainties[i]
                lines.append(f"  {name:18s}: {val:12.4e} ± {unc:.2e}")
            else:
                lines.append(f"  {name:18s}: {val:12.4e}")

        lines.append("")
        lines.append("Statistics:")
        lines.append("-" * 50)

        if self.final_cost is not None:
            lines.append(f"  Final cost: {self.final_cost:.6e}")
        if self.reduced_chi_squared is not None:
            lines.append(f"  Reduced χ²: {self.reduced_chi_squared:.4f}")
        lines.append(f"  Iterations: {self.n_iterations}")
        lines.append(f"  Function evals: {self.n_function_evals}")
        if self.wall_time_seconds is not None:
            lines.append(f"  Wall time: {self.wall_time_seconds:.2f} s")

        return "\n".join(lines)


# =============================================================================
# Homodyne-parity: FunctionEvaluationCounter
# =============================================================================


@dataclass
class FunctionEvaluationCounter:
    """Wraps a callable and counts invocations.

    Useful for tracking the number of function evaluations during optimization.

    Attributes:
        fn: The wrapped callable
        count: Number of times the callable has been invoked
    """

    fn: Callable[..., Any]
    count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function and increment count."""
        self.count += 1
        return self.fn(*args, **kwargs)


# =============================================================================
# Homodyne-parity: OptimizationResult
# =============================================================================


@dataclass
class OptimizationResult:
    """Complete optimization result with fit quality metrics and diagnostics.

    Homodyne-parity class. This mirrors homodyne's ``OptimizationResult``
    and is returned by ``ResultBuilder.build()``. Heterodyne's primary result
    type is ``NLSQResult`` (richer API); ``OptimizationResult`` is the
    dict-builder-compatible surface used by wrapper code.

    Attributes
    ----------
    parameters : np.ndarray
        Converged parameter values.
    uncertainties : np.ndarray
        Standard deviations from covariance matrix diagonal.
    covariance : np.ndarray
        Full parameter covariance matrix.
    chi_squared : float
        Sum of squared residuals.
    reduced_chi_squared : float
        chi_squared / (n_data - n_params).
    convergence_status : str
        'converged', 'max_iter', or 'failed'.
    iterations : int
        Number of optimization iterations.
    execution_time : float
        Wall-clock execution time in seconds.
    device_info : dict[str, Any]
        Device used for computation (CPU details).
    recovery_actions : list[str]
        List of error recovery actions taken.
    quality_flag : str
        'good', 'marginal', or 'poor'.
    streaming_diagnostics : dict[str, Any] | None
        Enhanced diagnostics for streaming optimization.
    stratification_diagnostics : Any | None
        Diagnostics for angle-stratified chunking.
    nlsq_diagnostics : dict[str, Any] | None
        Additional NLSQ-specific diagnostics.
    sigma_is_default : bool
        True if sigma weights were defaulted (not user-supplied).
    """

    parameters: np.ndarray
    uncertainties: np.ndarray
    covariance: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    convergence_status: str
    iterations: int
    execution_time: float
    device_info: dict[str, Any]
    recovery_actions: list[str] = field(default_factory=list)
    quality_flag: str = "good"
    streaming_diagnostics: dict[str, Any] | None = None
    stratification_diagnostics: Any | None = None
    nlsq_diagnostics: dict[str, Any] | None = None
    sigma_is_default: bool = False

    @property
    def success(self) -> bool:
        """Return True if optimization converged (backward compatibility)."""
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Return descriptive message about optimization outcome."""
        if self.convergence_status == "converged":
            return f"Optimization converged successfully. chi2={self.chi_squared:.6f}"
        elif self.convergence_status == "max_iter":
            return "Optimization stopped: maximum iterations reached"
        else:
            return f"Optimization failed: {self.convergence_status}"


# =============================================================================
# Homodyne-parity: FallbackInfo
# =============================================================================


@dataclass
class FallbackInfo:
    """Tracks fallback from NLSQAdapter to NLSQWrapper.

    Included in OptimizationResult.device_info when fallback occurs.

    Attributes:
        fallback_occurred: True if fallback was triggered
        adapter_used: "NLSQAdapter" or "NLSQWrapper"
        adapter_error: Error message if adapter failed (None if succeeded)
        wrapper_error: Error message if wrapper also failed (None otherwise)

    States:

    * NLSQAdapter + fallback_occurred=False + adapter_error=None: Adapter succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="...": Fallback succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="..." + wrapper_error="...": Both failed
    """

    fallback_occurred: bool
    adapter_used: str  # "NLSQAdapter" or "NLSQWrapper"
    adapter_error: str | None = None
    wrapper_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for inclusion in device_info."""
        return {
            "fallback_occurred": self.fallback_occurred,
            "adapter_used": self.adapter_used,
            "adapter_error": self.adapter_error,
            "wrapper_error": self.wrapper_error,
        }


# =============================================================================
# Homodyne-parity: UseSequentialOptimization
# =============================================================================


@dataclass
class UseSequentialOptimization:
    """Marker indicating sequential per-angle optimization should be used.

    This is returned by _apply_stratification_if_needed when conditions require
    sequential per-angle optimization as a fallback strategy.

    Attributes
    ----------
    data : Any
        Original XPCS data object.
    reason : str
        Why sequential optimization is needed.
    """

    data: Any
    reason: str
