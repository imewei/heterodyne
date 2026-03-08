"""Command dispatch for heterodyne CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from heterodyne.cli.config_handling import load_and_merge_config
from heterodyne.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
from heterodyne.cli.optimization_runner import run_cmc, run_nlsq
from heterodyne.cli.plot_dispatch import dispatch_plots
from heterodyne.core.heterodyne_model import HeterodyneModel
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
    config_manager = load_and_merge_config(args.config, args)

    output_dir = args.output or config_manager.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    data = load_and_validate_data(config_manager)
    model = HeterodyneModel.from_config(config_manager.raw_config)
    phi_angles = resolve_phi_angles(args, config_manager)

    nlsq_results: list[NLSQResult] = []
    cmc_results: list[CMCResult] = []

    if args.method == "nlsq" or args.method == "both":
        nlsq_results = run_nlsq(
            model=model,
            c2_data=data.c2,
            phi_angles=phi_angles,
            config_manager=config_manager,
            args=args,
            output_dir=output_dir,
        )

    if args.method == "cmc" or args.method == "both":
        cmc_results = run_cmc(
            model=model,
            c2_data=data.c2,
            phi_angles=phi_angles,
            config_manager=config_manager,
            args=args,
            output_dir=output_dir,
            nlsq_results=nlsq_results if args.method == "both" else None,
        )

    if args.plot:
        dispatch_plots(
            model=model,
            c2_data=data.c2,
            nlsq_results=nlsq_results if args.method in ("nlsq", "both") else None,
            cmc_results=cmc_results if args.method in ("cmc", "both") else None,
            output_dir=output_dir,
        )

    return 0


# Keep legacy function signatures as thin wrappers for backward compatibility
run_nlsq_analysis = run_nlsq
run_cmc_analysis = run_cmc
generate_plots = dispatch_plots
