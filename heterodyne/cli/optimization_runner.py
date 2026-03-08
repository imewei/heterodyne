"""Optimization execution for heterodyne CLI.

Manages NLSQ and CMC fitting runs, including warm-start resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.io.mcmc_writers import format_mcmc_summary, save_mcmc_results
from heterodyne.io.nlsq_writers import (
    format_nlsq_summary,
    save_nlsq_json_files,
    save_nlsq_npz_file,
)
from heterodyne.optimization.cmc import CMCConfig, fit_cmc_jax
from heterodyne.optimization.nlsq import NLSQConfig, fit_nlsq_jax
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def run_nlsq(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[NLSQResult]:
    """Run NLSQ analysis for all phi angles.

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data (2D or 3D).
        phi_angles: Phi angles to analyze.
        config_manager: Configuration manager.
        args: CLI arguments.
        output_dir: Output directory for results.

    Returns:
        List of NLSQResult objects, one per phi angle.
    """
    logger.info("Starting NLSQ analysis")

    nlsq_config_dict = config_manager.nlsq_config
    if getattr(args, "multistart", False):
        nlsq_config_dict["multistart"] = True
        nlsq_config_dict["multistart_n"] = getattr(args, "multistart_n", 10)

    nlsq_config = NLSQConfig.from_dict(nlsq_config_dict)
    nlsq_config.verbose = getattr(args, "verbose", 1)

    results: list[NLSQResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info("Fitting phi=%s\u00b0 (%d/%d)", phi, i + 1, len(phi_angles))

        c2_phi = c2_data[i] if c2_data.ndim == 3 else c2_data

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_phi,
            phi_angle=phi,
            config=nlsq_config,
        )

        result.metadata["phi_angle"] = phi
        results.append(result)

        print(f"\n{'=' * 50}")
        print(f"NLSQ Results for phi={phi}\u00b0")
        print(format_nlsq_summary(result))

        prefix = f"nlsq_phi{int(phi)}" if len(phi_angles) > 1 else "nlsq"
        save_nlsq_json_files(result, output_dir, prefix=prefix)
        save_nlsq_npz_file(result, output_dir / f"{prefix}_data.npz")

    logger.info("NLSQ analysis complete")
    return results


def run_cmc(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    nlsq_results: list[NLSQResult] | None = None,
) -> list[CMCResult]:
    """Run CMC Bayesian analysis for all phi angles.

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data.
        phi_angles: Phi angles to analyze.
        config_manager: Configuration manager.
        args: CLI arguments.
        output_dir: Output directory.
        nlsq_results: Optional NLSQ results for warm-starting.

    Returns:
        List of CMCResult objects, one per phi angle.
    """
    logger.info("Starting CMC analysis")

    cmc_config_dict = config_manager.cmc_config
    if getattr(args, "num_samples", None) is not None:
        cmc_config_dict["num_samples"] = args.num_samples
    if getattr(args, "num_chains", None) is not None:
        cmc_config_dict["num_chains"] = args.num_chains

    cmc_config = CMCConfig.from_dict(cmc_config_dict)

    results: list[CMCResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info("CMC for phi=%s\u00b0 (%d/%d)", phi, i + 1, len(phi_angles))

        c2_phi = c2_data[i] if c2_data.ndim == 3 else c2_data

        nlsq_result_i = nlsq_results[i] if nlsq_results and i < len(nlsq_results) else None
        result = fit_cmc_jax(
            model=model,
            c2_data=c2_phi,
            phi_angle=phi,
            config=cmc_config,
            nlsq_result=nlsq_result_i,
        )

        result.metadata["phi_angle"] = phi
        results.append(result)

        print(f"\n{'=' * 50}")
        print(f"CMC Results for phi={phi}\u00b0")
        print(format_mcmc_summary(result))

        prefix = f"cmc_phi{int(phi)}" if len(phi_angles) > 1 else "cmc"
        save_mcmc_results(result, output_dir, prefix=prefix)

    logger.info("CMC analysis complete")
    return results


def resolve_nlsq_warmstart(
    args: argparse.Namespace,
    output_dir: Path,
) -> NLSQResult | None:
    """Attempt to load previously saved NLSQ results for warm-starting CMC.

    Args:
        args: CLI arguments (may have .warmstart_path).
        output_dir: Default directory to search for NLSQ results.

    Returns:
        NLSQResult if found, None otherwise.
    """
    warmstart_path = getattr(args, "warmstart_path", None)
    if warmstart_path is None:
        # Try default location
        default_path = output_dir / "nlsq_data.npz"
        if default_path.exists():
            warmstart_path = default_path
        else:
            return None

    try:
        from heterodyne.io.nlsq_writers import load_nlsq_npz_file
        result = load_nlsq_npz_file(Path(warmstart_path))
        logger.info("Loaded NLSQ warm-start from %s", warmstart_path)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load NLSQ warm-start from %s: %s", warmstart_path, exc)
        return None
