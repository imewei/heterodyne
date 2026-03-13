"""Command dispatch for heterodyne CLI."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heterodyne.cli.config_handling import load_and_merge_config
from heterodyne.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
from heterodyne.cli.optimization_runner import run_cmc, run_nlsq
from heterodyne.cli.plot_dispatch import dispatch_plots
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.utils.logging import AnalysisSummaryLogger, get_logger, log_phase

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_data(
    config_manager: ConfigManager,
    args: argparse.Namespace,
) -> tuple[Any, list[float]]:
    """Load and validate data, resolve phi angles.

    Args:
        config_manager: Merged configuration manager.
        args: Parsed CLI arguments (used for phi-angle resolution).

    Returns:
        Tuple of (XPCSData, phi_angles).
    """
    data = load_and_validate_data(config_manager)
    phi_angles = resolve_phi_angles(args, config_manager)
    logger.debug(
        "Loaded data: c2 shape=%s, %d phi angles",
        data.c2.shape,
        len(phi_angles),
    )
    return data, phi_angles


def _run_optimization(
    method: str,
    model: HeterodyneModel,
    data: Any,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    summary: AnalysisSummaryLogger,
) -> dict[str, Any]:
    """Unified optimization dispatcher.

    Runs NLSQ and/or CMC based on *method* and returns a dict with both
    result lists so the caller doesn't need to track them separately.

    Args:
        method: One of ``"nlsq"``, ``"cmc"``, or ``"both"``.
        model: Configured HeterodyneModel.
        data: XPCSData (needs ``.c2`` attribute).
        phi_angles: Phi angles in degrees.
        config_manager: Merged configuration manager.
        args: Parsed CLI arguments forwarded to runners.
        output_dir: Directory for outputs.
        summary: Logger that tracks analysis phases.

    Returns:
        ``{"nlsq_results": list[NLSQResult], "cmc_results": list[CMCResult]}``
    """
    nlsq_results: list[NLSQResult] = []
    cmc_results: list[CMCResult] = []

    if method in ("nlsq", "both"):
        summary.start_phase("nlsq_optimization")
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
        summary.end_phase("nlsq_optimization", memory_peak_gb=phase.memory_peak_gb)

    if method in ("cmc", "both"):
        summary.start_phase("cmc_optimization")
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
        summary.end_phase("cmc_optimization", memory_peak_gb=phase.memory_peak_gb)

    return {"nlsq_results": nlsq_results, "cmc_results": cmc_results}


def _generate_cmc_diagnostic_plots(
    results: list[CMCResult],
    output_dir: Path,
) -> None:
    """Generate CMC-specific diagnostic plots for each result.

    Imports ``plot_convergence_diagnostics`` and ``plot_kl_divergence_matrix``
    lazily from :mod:`heterodyne.viz` and writes one pair of figures per
    CMC result.  Exceptions are caught and logged so that plotting failures
    never abort the pipeline.

    Args:
        results: List of CMC results to visualise.
        output_dir: Directory to write figures into.
    """
    if not results:
        return

    try:
        from heterodyne.viz.mcmc_diagnostics import (
            plot_convergence_diagnostics,
            plot_kl_divergence_matrix,
        )
    except ImportError:
        logger.warning(
            "Could not import mcmc_diagnostics — skipping CMC diagnostic plots"
        )
        return

    diag_dir = output_dir / "cmc_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results):
        tag = f"angle_{idx}"
        # Convergence diagnostics (ESS, R-hat, BFMI)
        try:
            plot_convergence_diagnostics(
                result, save_path=diag_dir / f"convergence_{tag}.png"
            )
            logger.debug("Saved convergence diagnostics for %s", tag)
        except Exception:
            logger.exception("Failed to generate convergence plot for %s", tag)

        # KL divergence matrix
        try:
            plot_kl_divergence_matrix(
                result, save_path=diag_dir / f"kl_divergence_{tag}.png"
            )
            logger.debug("Saved KL divergence matrix for %s", tag)
        except Exception:
            logger.exception("Failed to generate KL divergence plot for %s", tag)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def dispatch_command(args: argparse.Namespace) -> int:
    """Dispatch to appropriate analysis command.

    Supports ``--plot-only`` (skip optimisation, generate plots from
    existing results) and ``--simulate-only`` (skip optimisation, save
    simulated data from the configured model).

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 on success).
    """
    run_id = f"het_{uuid.uuid4().hex[:8]}"
    method = getattr(args, "method", "nlsq")
    summary = AnalysisSummaryLogger(run_id=run_id, analysis_mode="two_component")
    summary.set_config_summary(optimizer=method)

    try:
        # --- Configuration ---------------------------------------------------
        with log_phase("config_loading", logger=logger):
            config_manager = load_and_merge_config(args.config, args)

        output_dir = args.output or config_manager.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        # --- Data loading ----------------------------------------------------
        with log_phase("data_loading", logger=logger, track_memory=True) as phase:
            data, phi_angles = _load_data(config_manager, args)
            model = HeterodyneModel.from_config(config_manager.raw_config)

        summary.set_config_summary(n_phi_angles=len(phi_angles))
        summary.start_phase("data_loading")
        summary.end_phase("data_loading", memory_peak_gb=phase.memory_peak_gb)

        # --- Simulate-only mode ----------------------------------------------
        if getattr(args, "simulate_only", False):
            logger.info("--simulate-only: saving simulated data and exiting")
            import numpy as np

            sim_path = output_dir / "simulated_data.npz"
            np.savez(
                sim_path,
                c2=np.asarray(data.c2),
                t1=np.asarray(data.t1),
                t2=np.asarray(data.t2),
                phi_angles=np.asarray(phi_angles),
            )
            logger.info("Simulated data saved to %s", sim_path)
            summary.set_convergence_status("completed")
            summary.log_summary(logger)
            return 0

        # --- Plot-only mode --------------------------------------------------
        if getattr(args, "plot_only", False):
            logger.info("--plot-only: generating plots without optimization")
            with log_phase("plotting", logger=logger):
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    nlsq_results=None,
                    cmc_results=None,
                    output_dir=output_dir,
                )
            summary.set_convergence_status("completed")
            summary.log_summary(logger)
            return 0

        # --- Optimization ----------------------------------------------------
        opt = _run_optimization(
            method=method,
            model=model,
            data=data,
            phi_angles=phi_angles,
            config_manager=config_manager,
            args=args,
            output_dir=output_dir,
            summary=summary,
        )
        nlsq_results = opt["nlsq_results"]
        cmc_results = opt["cmc_results"]

        # --- CMC diagnostic plots --------------------------------------------
        if cmc_results:
            with log_phase("cmc_diagnostics", logger=logger):
                _generate_cmc_diagnostic_plots(cmc_results, output_dir)

        # --- User-requested plots --------------------------------------------
        if getattr(args, "plot", False):
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
