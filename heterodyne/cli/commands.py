"""Command dispatch for heterodyne CLI."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heterodyne.cli.config_handling import load_and_merge_config
from heterodyne.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
from heterodyne.cli.optimization_runner import resolve_nlsq_warmstart, run_cmc, run_nlsq
from heterodyne.cli.plot_dispatch import dispatch_plots, handle_plotting

# Re-export from result_saving (homodyne parity)
from heterodyne.cli.result_saving import (  # noqa: F401
    save_cmc_results,
    save_nlsq_results,
    save_results,
    save_summary_manifest,
)
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.utils.logging import (
    AnalysisSummaryLogger,
    get_logger,
    log_exception,
    log_phase,
)

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def _result_converged(result: Any) -> bool:
    """Return True when an optimizer result represents a converged fit.

    NLSQResult exposes ``success`` (least-squares termination).
    CMCResult exposes ``convergence_passed`` (R-hat/ESS gates).

    Read whichever attribute the result type provides; do not silently default
    to ``True`` when neither is present, since that masked the het_a10cf27e
    failure mode (47/47 shards failed every angle, reported as "converged").

    Audit note (2026-05-17): a sweep for vulnerable status-aggregation
    patterns elsewhere in the codebase found no other sites using
    ``getattr(result, "<status>", True)``. All other status reductions use
    direct attribute access on NLSQResult (``r.success``) which is type-safe.
    If a new result class is introduced, route its status check through this
    helper so the guard continues to apply.
    """
    if hasattr(result, "convergence_passed"):
        return bool(result.convergence_passed)
    if hasattr(result, "success"):
        return bool(result.success)
    # Unknown result type — be conservative.
    return False


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
    import numpy as _np_load

    data = load_and_validate_data(config_manager)
    data_phi_angles = (
        _np_load.asarray(data.phi_angles, dtype=float)
        if getattr(data, "phi_angles", None) is not None
        else None
    )
    phi_angles = resolve_phi_angles(
        args, config_manager, data_phi_angles=data_phi_angles
    )
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

    import numpy as _np_opt

    _data_phi_angles = (
        _np_opt.asarray(data.phi_angles, dtype=float)
        if getattr(data, "phi_angles", None) is not None
        else None
    )

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
                data_phi_angles=_data_phi_angles,
            )
        summary.end_phase("nlsq_optimization", memory_peak_gb=phase.memory_peak_gb)

    if method in ("cmc", "both"):
        # In CMC-only mode nlsq_results is empty. Attempt to load a previously
        # saved NLSQ result from disk so NUTS can warm-start near the MAP
        # instead of the prior (which is typically 5-10σ away and causes
        # complete non-mixing: R-hat >> 1, ESS ≈ n_chains).
        if method == "cmc" and not nlsq_results:
            loaded = resolve_nlsq_warmstart(args, output_dir)
            if loaded is not None:
                nlsq_results = [loaded] * len(phi_angles)
                logger.info(
                    "CMC-only mode: loaded NLSQ warm-start from disk "
                    "(chi2=%.4g, success=%s)",
                    loaded.reduced_chi_squared or float("nan"),
                    loaded.success,
                )
            else:
                logger.warning(
                    "CMC-only mode: no NLSQ warm-start found. "
                    "Run NLSQ first (optimizer: nlsq) and re-run CMC, or use "
                    "optimizer: both to run NLSQ→CMC in one pass."
                )

        summary.start_phase("cmc_optimization")
        with log_phase("cmc_optimization", logger=logger, track_memory=True) as phase:
            # run_cmc returns a single joint multi-phi CMCResult (homodyne
            # parity). The local list wrapper keeps downstream consumers
            # (result_saving, plot_dispatch) iterating over results without
            # changes; the single wrapped result still represents ONE joint
            # NUTS inference across all phi angles.
            _cmc_joint_result = run_cmc(
                model=model,
                c2_data=data.c2,
                phi_angles=phi_angles,
                config_manager=config_manager,
                args=args,
                output_dir=output_dir,
                nlsq_results=nlsq_results
                if method == "both"
                else (nlsq_results or None),
                summary=summary,
                data_phi_angles=_data_phi_angles,
            )
            cmc_results = [_cmc_joint_result]
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

        # ArviZ diagnostic suite (forest/energy/autocorr/rank/ess) — homodyne parity
        idata = getattr(result, "inference_data", None)
        if idata is None:
            continue
        try:
            from heterodyne.optimization.cmc.plotting import generate_diagnostic_plots

            arviz_dir = diag_dir / tag
            saved = generate_diagnostic_plots(idata, arviz_dir)
            if saved:
                logger.info("Saved %d ArviZ diagnostic plots for %s", len(saved), tag)
        except Exception:
            logger.exception("Failed to generate ArviZ diagnostic plots for %s", tag)


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

    logger.info("[CLI] Dispatching heterodyne analysis command (run_id=%s)", run_id)
    logger.debug("[CLI] Resolved arguments: %s", vars(args))

    log_file: Path | None = None

    try:
        # --- Configuration ---------------------------------------------------
        summary.start_phase("config_loading")
        with log_phase("config_loading", logger=logger):
            config_manager = load_and_merge_config(args.config, args)
        summary.end_phase("config_loading")

        output_dir = args.output or config_manager.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        # Configure file logging into output directory (homodyne parity)
        # Creates timestamped log file in logs/ subdirectory
        from heterodyne.utils.logging import configure_logging

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"heterodyne_analysis_{run_id}.log"

        # Determine log level from CLI flags (homodyne: -v=INFO, -vv=DEBUG, -vvv=TRACE)
        verbose_level = getattr(args, "verbose", 0)
        quiet = getattr(args, "quiet", False)
        if quiet:
            log_level = "ERROR"
        elif verbose_level >= 2:
            log_level = "DEBUG"
        elif verbose_level >= 1:
            log_level = "INFO"
        else:
            log_level = "INFO"  # File logging always at INFO minimum

        configure_logging(level=log_level, log_file=log_file)  # type: ignore[arg-type]
        logger.info("[CLI] Log file created: %s", log_file)
        summary.add_output_file(log_file)

        # --- Data loading ----------------------------------------------------
        summary.start_phase("data_loading")
        with log_phase("data_loading", logger=logger, track_memory=True) as phase:
            data, phi_angles = _load_data(config_manager, args)
            model = HeterodyneModel.from_config(config_manager.raw_config)

        summary.set_config_summary(n_phi_angles=len(phi_angles))
        summary.end_phase("data_loading", memory_peak_gb=phase.memory_peak_gb)

        # --- Build rich data dict for plotting --------------------------------
        import numpy as _np

        # Convert loader frame indices to elapsed seconds for plotting.
        # The HDF5 cache may store absolute frame indices (e.g. 999..1999);
        # heatmap axes should show elapsed time within the selected window.
        _dt = model.dt
        _t1_sec = None
        if data.t1 is not None:
            _t1_frames = _np.asarray(data.t1, dtype=float)
            if len(_t1_frames) > 0:
                _t1_sec = (_t1_frames - _t1_frames[0]) * _dt

        _data_dict: dict[str, Any] = {
            "c2_exp": _np.asarray(data.c2),
            "t1": _t1_sec,
            "t2": _t1_sec,
            "phi_angles_list": (
                _np.asarray(data.phi_angles)
                if data.phi_angles is not None
                else _np.asarray(phi_angles)
            ),
            "config": config_manager.raw_config,
        }

        # --- Plot-experimental-data mode --------------------------------------
        plot_exp = getattr(args, "plot_experimental_data", False)
        plot_sim = getattr(args, "plot_simulated_data", False)

        if plot_exp and not plot_sim:
            logger.info("--plot-experimental-data: plotting data and exiting")
            summary.start_phase("plotting")
            with log_phase("plotting", logger=logger, track_memory=True) as phase:
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    output_dir=output_dir,
                    mode="experimental",
                    phi_angles=phi_angles,
                    data_dict=_data_dict,
                )
            summary.end_phase("plotting", memory_peak_gb=phase.memory_peak_gb)
            summary.set_convergence_status("completed")
            summary.log_summary(logger)
            return 0

        if plot_sim and not plot_exp:
            logger.info("--plot-simulated-data: plotting simulated data and exiting")
            summary.start_phase("plotting")
            with log_phase("plotting", logger=logger, track_memory=True) as phase:
                from heterodyne.cli.plot_dispatch import _plot_simulated_data

                plots_dir = output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Use configured scaling values; CLI --contrast/--offset-sim override
                _sim_contrast, _sim_offset = model.scaling.get_for_angle(0)
                _cli_contrast = getattr(args, "contrast", None)
                _cli_offset = getattr(args, "offset_sim", None)
                if _cli_contrast is not None:
                    _sim_contrast = float(_cli_contrast)
                if _cli_offset is not None:
                    _sim_offset = float(_cli_offset)
                # Use the filtered phi_angles so hsim shows the same angles as
                # ht-nlsq fitted simulations, making the two plots directly comparable.
                # Without this, hsim uses all 23 HDF5 angles (e.g. phi=0°) while
                # ht-nlsq shows only the 2 fitted angles (phi=-5.79°, +4.88°).
                _sim_data = {
                    **_data_dict,
                    "phi_angles_list": _np.asarray(phi_angles),
                }
                _plot_simulated_data(
                    config=config_manager.raw_config,
                    contrast=_sim_contrast,
                    offset=_sim_offset,
                    phi_angles_str=getattr(args, "phi_angles", None),
                    plots_dir=plots_dir,
                    data=_sim_data,
                )
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    output_dir=output_dir,
                    mode="simulated",
                    phi_angles=phi_angles,
                    data_dict=_data_dict,
                )
            summary.end_phase("plotting", memory_peak_gb=phase.memory_peak_gb)
            summary.set_convergence_status("completed")
            summary.log_summary(logger)
            return 0

        if plot_exp and plot_sim:
            logger.info("Plotting both experimental and simulated data and exiting")
            summary.start_phase("plotting")
            with log_phase("plotting", logger=logger, track_memory=True) as phase:
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    output_dir=output_dir,
                    mode="both",
                    phi_angles=phi_angles,
                    data_dict=_data_dict,
                )
            summary.end_phase("plotting", memory_peak_gb=phase.memory_peak_gb)
            summary.set_convergence_status("completed")
            summary.log_summary(logger)
            return 0

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
            summary.start_phase("plotting")
            with log_phase("plotting", logger=logger, track_memory=True) as phase:
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    nlsq_results=None,
                    cmc_results=None,
                    output_dir=output_dir,
                )
            summary.end_phase("plotting", memory_peak_gb=phase.memory_peak_gb)
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

        # Record convergence status and metrics from optimization results.
        # NLSQResult exposes ``success`` (least-squares convergence); CMCResult
        # exposes ``convergence_passed`` (R-hat/ESS gates). Using a single
        # ``getattr(r, "success", True)`` silently treated every CMCResult as
        # successful because the attribute does not exist on CMCResult — even
        # when 100% of MCMC shards failed convergence (run het_a10cf27e: all
        # 47/47 shards failed across all 3 angles, yet status was "converged").
        active_results = (
            cmc_results if method in ("cmc", "both") and cmc_results else nlsq_results
        )
        if active_results:
            converged = all(_result_converged(r) for r in active_results)
            summary.set_convergence_status(
                "converged" if converged else "not_converged"
            )
            chi2_vals = [
                r.reduced_chi_squared
                for r in active_results
                if getattr(r, "reduced_chi_squared", None) is not None
            ]
            if chi2_vals:
                summary.record_metric(
                    "chi_squared", float(sum(chi2_vals) / len(chi2_vals))
                )
        else:
            summary.set_convergence_status("completed")

        # --- CMC diagnostic plots --------------------------------------------
        if cmc_results:
            with log_phase("cmc_diagnostics", logger=logger):
                _generate_cmc_diagnostic_plots(cmc_results, output_dir)

        # --- Unified result manifest (homodyne parity) -----------------------
        if nlsq_results or cmc_results:
            summary.start_phase("result_saving")
            with log_phase("result_saving", logger=logger):
                try:
                    saved_paths = save_results(
                        method=method,
                        nlsq_results=nlsq_results or None,
                        cmc_results=cmc_results or None,
                        output_dir=output_dir,
                        phi_angles=list(phi_angles),
                        model=model,
                    )
                    n_files = sum(len(paths) for paths in saved_paths.values())
                    logger.info("Unified result manifest: %d files written", n_files)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning(
                        "Unified result manifest skipped (%s); inline saves "
                        "from optimization_runner remain available",
                        exc,
                    )
            summary.end_phase("result_saving")

        # --- User-requested plots --------------------------------------------
        if getattr(args, "plot", False):
            summary.start_phase("plotting")
            with log_phase("plotting", logger=logger, track_memory=True) as phase:
                dispatch_plots(
                    model=model,
                    c2_data=data.c2,
                    nlsq_results=nlsq_results if method in ("nlsq", "both") else None,
                    cmc_results=cmc_results if method in ("cmc", "both") else None,
                    output_dir=output_dir,
                    data_dict=_data_dict,
                )
            summary.end_phase("plotting", memory_peak_gb=phase.memory_peak_gb)

        # --- Save-plots: fit comparison + fitted simulations ------------------
        if getattr(args, "save_plots", False):
            summary.start_phase("save_plots")
            with log_phase("save_plots", logger=logger, track_memory=True) as phase:
                for _res in nlsq_results if nlsq_results else [None]:
                    handle_plotting(
                        args=args,
                        result=_res,
                        data=_data_dict,
                        config=config_manager.raw_config,
                    )
            summary.end_phase("save_plots", memory_peak_gb=phase.memory_peak_gb)

    except KeyboardInterrupt:
        summary.set_convergence_status("failed")
        logger.info("[CLI] Analysis interrupted by user")
        summary.log_summary(logger)
        raise

    except Exception as exc:
        summary.set_convergence_status("failed")
        log_exception(logger, exc, context={"run_id": run_id, "phase": "dispatch"})
        summary.log_summary(logger)
        raise

    logger.info("[CLI] Analysis completed successfully")
    summary.log_summary(logger)
    if log_file is not None:
        logger.info("[CLI] Analysis log saved to: %s", log_file)
    # Non-zero exit when convergence failed so CI / downstream tooling can
    # detect the het_a10cf27e-class regression even if the human reader
    # missed the "not_converged" line in the analysis summary.
    if summary.is_failure():
        logger.warning(
            "[CLI] Exiting with code 2 — optimization did not converge "
            "(status=%s; see analysis summary above).",
            summary.convergence_status,
        )
        return 2
    return 0


# Keep legacy function signatures as thin wrappers for backward compatibility
run_nlsq_analysis = run_nlsq
run_cmc_analysis = run_cmc
generate_plots = dispatch_plots
