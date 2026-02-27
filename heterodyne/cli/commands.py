"""Command dispatch for heterodyne CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.config.manager import ConfigManager
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.data.validation import validate_xpcs_data
from heterodyne.data.xpcs_loader import load_xpcs_data
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
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def dispatch_command(args: argparse.Namespace) -> int:
    """Dispatch to appropriate analysis command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Loading configuration from {args.config}")
    config_manager = ConfigManager.from_yaml(args.config)

    # Override output directory if specified
    output_dir = args.output or config_manager.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info(f"Loading data from {config_manager.data_file_path}")
    data = load_xpcs_data(config_manager.data_file_path)

    # Validate data
    validation = validate_xpcs_data(data)
    if not validation.is_valid:
        for err in validation.errors:
            logger.error(f"Data validation error: {err}")
        return 1
    for warn in validation.warnings:
        logger.warning(f"Data validation warning: {warn}")

    # Create model
    model = HeterodyneModel.from_config(config_manager.raw_config)

    # Determine phi angles
    phi_angles = args.phi
    if phi_angles is None:
        phi_angles = config_manager.phi_angles
    if phi_angles is None:
        phi_angles = [0.0]  # Default single angle

    logger.info(f"Analyzing phi angles: {phi_angles}")

    # Initialize result containers to avoid UnboundLocalError
    nlsq_results: list[NLSQResult] = []
    cmc_results: list[CMCResult] = []

    # Run analysis based on method
    if args.method == "nlsq" or args.method == "both":
        nlsq_results = run_nlsq_analysis(
            model=model,
            c2_data=data.c2,
            phi_angles=phi_angles,
            config_manager=config_manager,
            args=args,
            output_dir=output_dir,
        )

    if args.method == "cmc" or args.method == "both":
        nlsq_result = (
            nlsq_results[0] if args.method == "both" and nlsq_results
            else None
        )
        cmc_results = run_cmc_analysis(
            model=model,
            c2_data=data.c2,
            phi_angles=phi_angles,
            config_manager=config_manager,
            args=args,
            output_dir=output_dir,
            nlsq_result=nlsq_result,
        )

    # Generate plots if requested
    if args.plot:
        generate_plots(
            model=model,
            c2_data=data.c2,
            nlsq_results=nlsq_results if args.method in ("nlsq", "both") else None,
            cmc_results=cmc_results if args.method in ("cmc", "both") else None,
            output_dir=output_dir,
        )

    return 0


def run_nlsq_analysis(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[NLSQResult]:
    """Run NLSQ analysis.
    
    Args:
        model: HeterodyneModel instance
        c2_data: Correlation data
        phi_angles: Phi angles to analyze
        config_manager: Configuration manager
        args: CLI arguments
        output_dir: Output directory
        
    Returns:
        List of NLSQResult objects
    """
    logger.info("Starting NLSQ analysis")

    # Build NLSQ config
    nlsq_config_dict = config_manager.nlsq_config
    if args.multistart:
        nlsq_config_dict["multistart"] = True
        nlsq_config_dict["multistart_n"] = args.multistart_n

    nlsq_config = NLSQConfig.from_dict(nlsq_config_dict)
    nlsq_config.verbose = args.verbose

    results: list[NLSQResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info(f"Fitting phi={phi}° ({i+1}/{len(phi_angles)})")

        # Get c2 for this angle (assumes c2_data is 2D or 3D)
        if c2_data.ndim == 3:
            c2_phi = c2_data[i]
        else:
            c2_phi = c2_data

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_phi,
            phi_angle=phi,
            config=nlsq_config,
        )

        result.metadata["phi_angle"] = phi
        results.append(result)

        # Print summary
        print(f"\n{'='*50}")
        print(f"NLSQ Results for phi={phi}°")
        print(format_nlsq_summary(result))

        # Save results
        prefix = f"nlsq_phi{int(phi)}" if len(phi_angles) > 1 else "nlsq"
        save_nlsq_json_files(result, output_dir, prefix=prefix)
        save_nlsq_npz_file(result, output_dir / f"{prefix}_data.npz")

    logger.info("NLSQ analysis complete")
    return results


def run_cmc_analysis(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    nlsq_result: NLSQResult | None = None,
) -> list[CMCResult]:
    """Run CMC Bayesian analysis.
    
    Args:
        model: HeterodyneModel instance
        c2_data: Correlation data
        phi_angles: Phi angles to analyze
        config_manager: Configuration manager
        args: CLI arguments
        output_dir: Output directory
        nlsq_result: Optional NLSQ result for warm-starting
        
    Returns:
        List of CMCResult objects
    """
    logger.info("Starting CMC analysis")

    # Build CMC config
    cmc_config_dict = config_manager.cmc_config
    if args.num_samples is not None:
        cmc_config_dict["num_samples"] = args.num_samples
    if args.num_chains is not None:
        cmc_config_dict["num_chains"] = args.num_chains

    cmc_config = CMCConfig.from_dict(cmc_config_dict)

    results: list[CMCResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info(f"CMC for phi={phi}° ({i+1}/{len(phi_angles)})")

        if c2_data.ndim == 3:
            c2_phi = c2_data[i]
        else:
            c2_phi = c2_data

        result = fit_cmc_jax(
            model=model,
            c2_data=c2_phi,
            phi_angle=phi,
            config=cmc_config,
            nlsq_result=nlsq_result,
        )

        result.metadata["phi_angle"] = phi
        results.append(result)

        # Print summary
        print(f"\n{'='*50}")
        print(f"CMC Results for phi={phi}°")
        print(format_mcmc_summary(result))

        # Save results
        prefix = f"cmc_phi{int(phi)}" if len(phi_angles) > 1 else "cmc"
        save_mcmc_results(result, output_dir, prefix=prefix)

    logger.info("CMC analysis complete")
    return results


def generate_plots(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    nlsq_results: list[NLSQResult] | None,
    cmc_results: list[CMCResult] | None,
    output_dir: Path,
) -> None:
    """Generate diagnostic plots.
    
    Args:
        model: HeterodyneModel
        c2_data: Correlation data
        nlsq_results: NLSQ results (if any)
        cmc_results: CMC results (if any)
        output_dir: Output directory
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend -- must precede viz imports

    from heterodyne.viz.experimental_plots import plot_correlation, plot_g1_components
    from heterodyne.viz.mcmc_plots import plot_posterior, plot_trace
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info(f"Generating plots in {plots_dir}")

    # Raw data plot
    c2_2d = c2_data[0] if c2_data.ndim == 3 else c2_data
    plot_correlation(c2_2d, save_path=plots_dir / "data_correlation.png")

    # NLSQ plots
    if nlsq_results:
        for nlsq_result in nlsq_results:
            phi = nlsq_result.metadata.get("phi_angle", 0)
            suffix = f"_phi{int(phi)}" if len(nlsq_results) > 1 else ""

            plot_nlsq_fit(c2_2d, nlsq_result, save_path=plots_dir / f"nlsq_fit{suffix}.png")
            plot_residual_map(
                nlsq_result, c2_2d,
                save_path=plots_dir / f"nlsq_residuals{suffix}.png",
            )

    # CMC plots
    if cmc_results:
        for cmc_result in cmc_results:
            phi = cmc_result.metadata.get("phi_angle", 0)
            suffix = f"_phi{int(phi)}" if len(cmc_results) > 1 else ""

            plot_posterior(cmc_result, save_path=plots_dir / f"cmc_posterior{suffix}.png")
            plot_trace(cmc_result, save_path=plots_dir / f"cmc_trace{suffix}.png")

    # Model component plots
    plot_g1_components(model, save_path=plots_dir / "g1_components.png")

    logger.info("Plots generated")
