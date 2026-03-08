"""Command dispatch for heterodyne CLI."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from heterodyne.cli.config_handling import load_and_merge_config
from heterodyne.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
from heterodyne.cli.optimization_runner import run_cmc, run_nlsq
from heterodyne.cli.plot_dispatch import dispatch_plots
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.utils.logging import AnalysisSummaryLogger, get_logger, log_phase

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
    run_id = f"het_{uuid.uuid4().hex[:8]}"
    method = getattr(args, "method", "nlsq")
    summary = AnalysisSummaryLogger(run_id=run_id, analysis_mode="two_component")
    summary.set_config_summary(optimizer=method)

    try:
        with log_phase("config_loading", logger=logger):
            config_manager = load_and_merge_config(args.config, args)

        output_dir = args.output or config_manager.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        with log_phase("data_loading", logger=logger, track_memory=True) as phase:
            data = load_and_validate_data(config_manager)
            model = HeterodyneModel.from_config(config_manager.raw_config)
            phi_angles = resolve_phi_angles(args, config_manager)

        summary.set_config_summary(n_phi_angles=len(phi_angles))
        summary.start_phase("data_loading")
        summary.end_phase("data_loading", memory_peak_gb=phase.memory_peak_gb)

        nlsq_results: list[NLSQResult] = []
        cmc_results: list[CMCResult] = []

        if method in ("nlsq", "both"):
            with log_phase("nlsq_optimization", logger=logger, track_memory=True) as phase:
                nlsq_results = run_nlsq(
                    model=model,
                    c2_data=data.c2,
                    phi_angles=phi_angles,
                    config_manager=config_manager,
                    args=args,
                    output_dir=output_dir,
                    summary=summary,
                )
            summary.start_phase("nlsq_optimization")
            summary.end_phase("nlsq_optimization", memory_peak_gb=phase.memory_peak_gb)

        if method in ("cmc", "both"):
            with log_phase("cmc_optimization", logger=logger, track_memory=True) as phase:
                cmc_results = run_cmc(
                    model=model,
                    c2_data=data.c2,
                    phi_angles=phi_angles,
                    config_manager=config_manager,
                    args=args,
                    output_dir=output_dir,
                    nlsq_results=nlsq_results if method == "both" else None,
                    summary=summary,
                )
            summary.start_phase("cmc_optimization")
            summary.end_phase("cmc_optimization", memory_peak_gb=phase.memory_peak_gb)

        if args.plot:
            with log_phase("plotting", logger=logger):
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    nlsq_results=nlsq_results if method in ("nlsq", "both") else None,
                    cmc_results=cmc_results if method in ("cmc", "both") else None,
                    output_dir=output_dir,
                )

        summary.set_convergence_status("completed")

    except Exception:
        summary.set_convergence_status("failed")
        summary.log_summary(logger)
        raise

    summary.log_summary(logger)
    return 0


# Keep legacy function signatures as thin wrappers for backward compatibility
run_nlsq_analysis = run_nlsq
run_cmc_analysis = run_cmc
generate_plots = dispatch_plots
