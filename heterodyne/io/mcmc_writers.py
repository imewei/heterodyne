"""Writers for MCMC/CMC analysis results."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.io.json_utils import json_safe, save_json

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult


def save_mcmc_results(
    result: CMCResult,
    output_dir: Path | str,
    prefix: str = "mcmc",
) -> dict[str, Path]:
    """Save MCMC/CMC results to files.
    
    Creates:
    - {prefix}_summary.json: Parameter summaries with credible intervals
    - {prefix}_diagnostics.json: Convergence diagnostics (R-hat, ESS)
    - {prefix}_samples.npz: Full posterior samples (compressed)
    
    Args:
        result: CMC result object
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict mapping file type to saved path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    # Stage all files in a temp directory, then move atomically
    with tempfile.TemporaryDirectory(dir=str(output_dir.parent)) as tmp_dir:
        tmp = Path(tmp_dir)

        # Summary file
        summary_data = {
            "parameter_names": result.parameter_names,
            "posterior_mean": json_safe(result.posterior_mean),
            "posterior_std": json_safe(result.posterior_std),
            "credible_intervals": json_safe(result.credible_intervals),
            "map_estimate": json_safe(result.map_estimate) if result.map_estimate is not None else None,
            "timestamp": datetime.now().isoformat(),
            "num_samples": result.num_samples,
            "num_chains": result.num_chains,
        }
        save_json(summary_data, tmp / f"{prefix}_summary.json")

        # Diagnostics file
        save_mcmc_diagnostics(result, tmp / f"{prefix}_diagnostics.json")

        # Samples file (NPZ for efficiency)
        _save_posterior_samples(result, tmp / f"{prefix}_samples.npz")

        # Move all staged files to the real output directory
        for f in tmp.iterdir():
            os.replace(str(f), str(output_dir / f.name))

    saved_paths["summary"] = output_dir / f"{prefix}_summary.json"
    saved_paths["diagnostics"] = output_dir / f"{prefix}_diagnostics.json"
    saved_paths["samples"] = output_dir / f"{prefix}_samples.npz"

    return saved_paths


def save_mcmc_diagnostics(
    result: CMCResult,
    output_path: Path | str,
) -> Path:
    """Save MCMC convergence diagnostics.
    
    Args:
        result: CMC result object
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    diagnostics: dict[str, Any] = {
        "convergence_passed": result.convergence_passed,
        "parameter_diagnostics": {},
    }
    
    for i, name in enumerate(result.parameter_names):
        param_diag: dict[str, Any] = {}
        
        if result.r_hat is not None:
            param_diag["r_hat"] = float(result.r_hat[i])
            param_diag["r_hat_passed"] = result.r_hat[i] < 1.1
        
        if result.ess_bulk is not None:
            param_diag["ess_bulk"] = float(result.ess_bulk[i])
        
        if result.ess_tail is not None:
            param_diag["ess_tail"] = float(result.ess_tail[i])
        
        diagnostics["parameter_diagnostics"][name] = param_diag
    
    # Overall statistics
    if result.r_hat is not None:
        diagnostics["max_r_hat"] = float(np.max(result.r_hat))
        diagnostics["all_r_hat_below_1_1"] = bool(np.all(result.r_hat < 1.1))
    
    if result.ess_bulk is not None:
        diagnostics["min_ess_bulk"] = float(np.min(result.ess_bulk))
    
    if result.bfmi is not None:
        diagnostics["bfmi"] = json_safe(result.bfmi)
        diagnostics["bfmi_passed"] = bool(np.all(np.array(result.bfmi) > 0.3))
    
    diagnostics["sampling_info"] = {
        "num_warmup": result.num_warmup,
        "num_samples": result.num_samples,
        "num_chains": result.num_chains,
        "wall_time_seconds": result.wall_time_seconds,
    }
    
    save_json(diagnostics, output_path)
    return output_path


def _save_posterior_samples(
    result: CMCResult,
    output_path: Path,
) -> Path:
    """Save posterior samples to NPZ file.

    Args:
        result: CMC result object
        output_path: Output file path

    Returns:
        Path to saved file
    """
    arrays: dict[str, Any] = {
        "parameter_names": np.array(result.parameter_names, dtype=object),
    }

    # Save samples for each parameter
    if result.samples is not None:
        for name, samples in result.samples.items():
            arrays[f"samples_{name}"] = np.asarray(samples)

    # Save diagnostics arrays
    if result.r_hat is not None:
        arrays["r_hat"] = np.asarray(result.r_hat)

    if result.ess_bulk is not None:
        arrays["ess_bulk"] = np.asarray(result.ess_bulk)

    if result.ess_tail is not None:
        arrays["ess_tail"] = np.asarray(result.ess_tail)

    np.savez_compressed(output_path, **arrays)
    return output_path


def format_mcmc_summary(result: CMCResult) -> str:
    """Format MCMC result as human-readable summary.
    
    Args:
        result: CMC result object
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "MCMC/CMC Analysis Results",
        "=" * 70,
        f"Convergence: {'PASSED' if result.convergence_passed else 'FAILED'}",
        f"Chains: {result.num_chains} | Samples: {result.num_samples} | Warmup: {result.num_warmup}",
        "",
        "Posterior Summary:",
        "-" * 70,
        f"{'Parameter':20s} {'Mean':>12s} {'Std':>10s} {'2.5%':>12s} {'97.5%':>12s} {'R-hat':>8s}",
        "-" * 70,
    ]
    
    for i, name in enumerate(result.parameter_names):
        mean = result.posterior_mean[i]
        std = result.posterior_std[i]
        
        ci = result.credible_intervals.get(name, {})
        ci_low = ci.get("2.5%", np.nan)
        ci_high = ci.get("97.5%", np.nan)
        
        r_hat = result.r_hat[i] if result.r_hat is not None else np.nan
        r_hat_str = f"{r_hat:.3f}" if not np.isnan(r_hat) else "N/A"
        
        lines.append(
            f"{name:20s} {mean:12.4e} {std:10.2e} {ci_low:12.4e} {ci_high:12.4e} {r_hat_str:>8s}"
        )
    
    lines.extend([
        "-" * 70,
        "",
        "Diagnostics:",
    ])
    
    if result.r_hat is not None:
        max_rhat = np.max(result.r_hat)
        lines.append(f"  Max R-hat: {max_rhat:.4f} {'(PASS)' if max_rhat < 1.1 else '(WARN)'}")
    
    if result.ess_bulk is not None:
        min_ess = np.min(result.ess_bulk)
        lines.append(f"  Min ESS (bulk): {min_ess:.0f}")
    
    if result.bfmi is not None:
        min_bfmi = np.min(result.bfmi)
        lines.append(f"  Min BFMI: {min_bfmi:.3f} {'(PASS)' if min_bfmi > 0.3 else '(WARN)'}")
    
    if result.wall_time_seconds is not None:
        lines.append(f"  Wall time: {result.wall_time_seconds:.1f} s")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
