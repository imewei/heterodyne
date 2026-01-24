"""Writers for NLSQ optimization results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.io.json_utils import json_safe, save_json

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult


def save_nlsq_json_files(
    result: NLSQResult,
    output_dir: Path | str,
    prefix: str = "nlsq",
) -> dict[str, Path]:
    """Save NLSQ results to JSON files.
    
    Creates:
    - {prefix}_parameters.json: Fitted parameter values and uncertainties
    - {prefix}_metadata.json: Fit statistics and convergence info
    - {prefix}_config.json: Configuration used for fitting
    
    Args:
        result: NLSQ fitting result
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict mapping file type to saved path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths: dict[str, Path] = {}
    
    # Parameters file
    params_data = {
        "parameters": json_safe(result.parameters),
        "uncertainties": json_safe(result.uncertainties) if result.uncertainties is not None else None,
        "parameter_names": result.parameter_names,
        "timestamp": datetime.now().isoformat(),
    }
    params_path = output_dir / f"{prefix}_parameters.json"
    save_json(params_data, params_path)
    saved_paths["parameters"] = params_path
    
    # Metadata file
    metadata = {
        "success": result.success,
        "message": result.message,
        "n_iterations": result.n_iterations,
        "n_function_evals": result.n_function_evals,
        "final_cost": float(result.final_cost) if result.final_cost is not None else None,
        "reduced_chi_squared": float(result.reduced_chi_squared) if result.reduced_chi_squared is not None else None,
        "convergence_reason": result.convergence_reason,
        "wall_time_seconds": result.wall_time_seconds,
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    save_json(metadata, metadata_path)
    saved_paths["metadata"] = metadata_path
    
    return saved_paths


def save_nlsq_npz_file(
    result: NLSQResult,
    output_path: Path | str,
    include_residuals: bool = True,
    include_jacobian: bool = False,
) -> Path:
    """Save NLSQ results to compressed NPZ file.
    
    NPZ format is efficient for large arrays (correlation matrices, residuals).
    
    Args:
        result: NLSQ fitting result
        output_path: Output file path
        include_residuals: Whether to include residual array
        include_jacobian: Whether to include Jacobian matrix (large)
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    arrays: dict[str, Any] = {
        "parameters": np.asarray(result.parameters),
        "parameter_names": np.array(result.parameter_names, dtype=object),
        "success": np.array(result.success),
        "final_cost": np.array(result.final_cost if result.final_cost is not None else np.nan),
    }
    
    if result.uncertainties is not None:
        arrays["uncertainties"] = np.asarray(result.uncertainties)
    
    if result.covariance is not None:
        arrays["covariance"] = np.asarray(result.covariance)
    
    if include_residuals and result.residuals is not None:
        arrays["residuals"] = np.asarray(result.residuals)
    
    if include_jacobian and result.jacobian is not None:
        arrays["jacobian"] = np.asarray(result.jacobian)
    
    if result.fitted_correlation is not None:
        arrays["fitted_correlation"] = np.asarray(result.fitted_correlation)
    
    np.savez_compressed(output_path, **arrays)
    return output_path


def format_nlsq_summary(result: NLSQResult) -> str:
    """Format NLSQ result as human-readable summary.
    
    Args:
        result: NLSQ fitting result
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "NLSQ Fitting Results",
        "=" * 60,
        f"Status: {'SUCCESS' if result.success else 'FAILED'}",
        f"Message: {result.message}",
        "",
        "Fitted Parameters:",
        "-" * 40,
    ]
    
    for i, name in enumerate(result.parameter_names):
        value = result.parameters[i]
        if result.uncertainties is not None:
            unc = result.uncertainties[i]
            lines.append(f"  {name:20s}: {value:12.6e} ± {unc:.2e}")
        else:
            lines.append(f"  {name:20s}: {value:12.6e}")
    
    lines.extend([
        "",
        "Fit Statistics:",
        "-" * 40,
        f"  Iterations:          {result.n_iterations}",
        f"  Function evaluations: {result.n_function_evals}",
    ])
    
    if result.final_cost is not None:
        lines.append(f"  Final cost:          {result.final_cost:.6e}")
    
    if result.reduced_chi_squared is not None:
        lines.append(f"  Reduced χ²:          {result.reduced_chi_squared:.4f}")
    
    if result.wall_time_seconds is not None:
        lines.append(f"  Wall time:           {result.wall_time_seconds:.2f} s")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
