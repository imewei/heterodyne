"""Physics constraint validators for heterodyne parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES

if TYPE_CHECKING:
    pass


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_parameters(params: np.ndarray | dict[str, float]) -> ValidationResult:
    """Validate heterodyne model parameters against physical constraints.
    
    Args:
        params: Either array of 14 values or dict with parameter names
        
    Returns:
        ValidationResult with errors and warnings
    """
    if isinstance(params, np.ndarray):
        if len(params) != 14:
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected 14 parameters, got {len(params)}"],
                warnings=[],
            )
        param_dict = {name: float(params[i]) for i, name in enumerate(ALL_PARAM_NAMES)}
    else:
        param_dict = dict(params)
    
    errors: list[str] = []
    warnings: list[str] = []
    
    # === Hard constraints (errors) ===
    
    # Diffusion coefficients must be non-negative
    for name in ("D0_ref", "D0_sample"):
        if name in param_dict:
            if param_dict[name] < 0:
                errors.append(f"{name}={param_dict[name]:.3e} must be non-negative")
            elif param_dict[name] < 1e-12:
                warnings.append(
                    f"{name}={param_dict[name]:.3e} is near zero; "
                    "this may cause degenerate diffusion behaviour"
                )
    
    # Fraction amplitude f0 must be in [0, 1]
    if "f0" in param_dict:
        f0 = param_dict["f0"]
        if not (0 <= f0 <= 1):
            errors.append(f"f0={f0:.3f} must be in [0, 1]")
    
    # Fraction baseline f3 must be in [0, 1]
    if "f3" in param_dict:
        f3 = param_dict["f3"]
        if not (0 <= f3 <= 1):
            errors.append(f"f3={f3:.3f} must be in [0, 1]")
    
    # Combined fraction must not exceed 1 (physical impossibility)
    if "f0" in param_dict and "f3" in param_dict:
        if param_dict["f0"] + param_dict["f3"] > 1.0:
            errors.append(
                f"f0 + f3 = {param_dict['f0'] + param_dict['f3']:.3f} > 1; "
                "total fraction exceeds unity, which is physically impossible"
            )
    
    # === Soft constraints (warnings) ===
    
    # Unusual exponent values
    for name in ("alpha_ref", "alpha_sample", "beta"):
        if name in param_dict:
            val = param_dict[name]
            if abs(val) > 2:
                warnings.append(f"{name}={val:.3f} has unusual magnitude (|α| > 2)")
    
    # Very large diffusion coefficients
    for name in ("D0_ref", "D0_sample"):
        if name in param_dict and param_dict[name] > 1e5:
            warnings.append(f"{name}={param_dict[name]:.3e} is unusually large")
    
    # Very large velocities
    if "v0" in param_dict and abs(param_dict["v0"]) > 1e3:
        warnings.append(f"v0={param_dict['v0']:.3e} is unusually large")
    
    # Exponential rate magnitude check
    if "f1" in param_dict and abs(param_dict["f1"]) > 5:
        warnings.append(
            f"f1={param_dict['f1']:.3f} is large, fraction may change rapidly"
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_time_integral_safety(
    alpha: float,
    t_min: float,
    t_max: float,
) -> ValidationResult:
    """Validate that time integral won't have numerical issues.
    
    For J(t) = D0 * t^alpha, the integral from 0 to T needs care when:
    - alpha < 0: singularity at t=0
    - alpha > large: potential overflow
    
    Args:
        alpha: Exponent value
        t_min: Minimum time (should be > 0 if alpha < 0)
        t_max: Maximum time
        
    Returns:
        ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    if alpha < 0 and t_min <= 0:
        errors.append(
            f"alpha={alpha:.3f} < 0 requires t_min > 0, got t_min={t_min}"
        )
    
    if alpha < -1:
        warnings.append(
            f"alpha={alpha:.3f} < -1 may cause numerical instability near t=0"
        )
    
    if alpha > 3:
        # t^alpha can overflow for large t
        if t_max ** alpha > 1e15:
            warnings.append(
                f"t_max^alpha = {t_max}^{alpha} = {t_max**alpha:.2e} may overflow"
            )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_correlation_inputs(
    t1: np.ndarray,
    t2: np.ndarray,
    c2_data: np.ndarray,
) -> ValidationResult:
    """Validate correlation matrix inputs.
    
    Args:
        t1: Time axis 1
        t2: Time axis 2
        c2_data: Correlation data
        
    Returns:
        ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    # Shape checks
    expected_shape = (len(t1), len(t2))
    if c2_data.shape != expected_shape:
        errors.append(
            f"c2_data shape {c2_data.shape} doesn't match time grids "
            f"({len(t1)}, {len(t2)})"
        )
    
    # NaN/Inf checks
    nan_count = np.sum(np.isnan(c2_data))
    if nan_count > 0:
        errors.append(f"c2_data contains {nan_count} NaN values")
    
    inf_count = np.sum(np.isinf(c2_data))
    if inf_count > 0:
        errors.append(f"c2_data contains {inf_count} infinite values")
    
    # Value range checks
    if np.any(c2_data < 0):
        warnings.append("c2_data contains negative values (unusual for correlation)")
    
    if np.any(c2_data > 2):
        warnings.append("c2_data contains values > 2 (unusual for normalized correlation)")
    
    # Monotonicity of time axes
    if not np.all(np.diff(t1) > 0):
        errors.append("t1 must be strictly increasing")
    
    if not np.all(np.diff(t2) > 0):
        errors.append("t2 must be strictly increasing")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
