"""Plot generation dispatch for heterodyne CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Angle filtering wrapper
# ---------------------------------------------------------------------------


def _apply_angle_filtering_for_plot(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering for plot generation.

    Delegates to :func:`heterodyne.data.angle_filtering.apply_angle_filtering_for_plot`.
    """
    from heterodyne.data.angle_filtering import apply_angle_filtering_for_plot

    return apply_angle_filtering_for_plot(phi_angles, c2_exp, data)


# ---------------------------------------------------------------------------
# Experimental data plots
# ---------------------------------------------------------------------------


def _plot_experimental_data(data: dict[str, Any], plots_dir: Path) -> None:
    """Plot experimental data with optional angle filtering.

    Args:
        data: Data dictionary with ``c2_exp``, ``t1``, ``t2``, ``phi_angles_list``.
        plots_dir: Output directory for plots.
    """
    from heterodyne.viz.experimental_plots import plot_experimental_data

    plot_experimental_data(
        data, plots_dir, angle_filter_func=_apply_angle_filtering_for_plot
    )


# ---------------------------------------------------------------------------
# Fit comparison plots
# ---------------------------------------------------------------------------


def _plot_fit_comparison(
    result: Any, data: dict[str, Any], plots_dir: Path
) -> None:
    """Plot fit-vs-experiment comparison.

    Args:
        result: Optimization result.
        data: Data dictionary.
        plots_dir: Output directory.
    """
    from heterodyne.viz.experimental_plots import plot_fit_comparison

    plot_fit_comparison(result, data, plots_dir)


# ---------------------------------------------------------------------------
# Simulated data plots
# ---------------------------------------------------------------------------


def _plot_simulated_data(
    config: dict[str, Any],
    contrast: float,
    offset: float,
    phi_angles_str: str | None,
    plots_dir: Path,
    data: dict[str, Any] | None = None,
) -> None:
    """Plot theoretical/simulated C2 heatmaps from config parameters.

    Args:
        config: Configuration dictionary with model parameters.
        contrast: Scaling contrast for simulated c2.
        offset: Baseline offset for simulated c2.
        phi_angles_str: Comma-separated phi angles, or None for defaults.
        plots_dir: Output directory for plots.
        data: Optional experimental data for extracting phi angles.
    """
    from heterodyne.viz.nlsq_plots import plot_simulated_data

    plot_simulated_data(
        config=config,
        contrast=contrast,
        offset=offset,
        phi_angles_str=phi_angles_str,
        plots_dir=plots_dir,
        data=data,
    )


# ---------------------------------------------------------------------------
# NLSQ 3-panel heatmaps
# ---------------------------------------------------------------------------


def generate_nlsq_plots(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_theoretical_scaled: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    config: Any = None,
    use_datashader: bool = True,
    parallel: bool = True,
    *,
    c2_solver_scaled: np.ndarray | None = None,
) -> None:
    """Generate 3-panel heatmaps (experimental | fit | residuals).

    Args:
        phi_angles: Scattering angles in degrees.
        c2_exp: Experimental correlation data.
        c2_theoretical_scaled: Scaled theoretical fits.
        residuals: Fit residuals.
        t1: Time array 1.
        t2: Time array 2.
        output_dir: Output directory.
        config: Configuration for color scaling.
        use_datashader: Whether to prefer Datashader backend.
        parallel: Generate per-angle plots in parallel.
        c2_solver_scaled: Optional solver-computed C2.
    """
    from heterodyne.viz.nlsq_plots import generate_nlsq_plots as _viz_gen

    _viz_gen(
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        c2_theoretical_scaled=c2_theoretical_scaled,
        residuals=residuals,
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        config=config,
        use_datashader=use_datashader,
        parallel=parallel,
        c2_solver_scaled=c2_solver_scaled,
    )


# ---------------------------------------------------------------------------
# Per-mode dispatchers (with per-operation error isolation)
# ---------------------------------------------------------------------------


def _dispatch_experimental_plots(
    c2_data: np.ndarray,
    plots_dir: Path,
    phi_angles: list[float] | None = None,
    data_dict: dict[str, Any] | None = None,
) -> None:
    """Plot raw correlation data with homodyne-parity entry point.

    If a data dictionary is available, delegates to :func:`_plot_experimental_data`
    for per-angle heatmaps with stats overlay. Otherwise falls back to
    simple per-slice plotting.

    Args:
        c2_data: Correlation data (2D or 3D).
        plots_dir: Directory for saving plots.
        phi_angles: Phi angles for labeling per-slice plots.
        data_dict: Full data dictionary for the rich plotting path.
    """
    if data_dict is not None:
        _plot_experimental_data(data_dict, plots_dir)
        return

    # Fallback: simple per-slice plots using plot_correlation
    from heterodyne.viz.experimental_plots import plot_correlation

    if phi_angles is not None and c2_data.ndim == 3:
        for i, phi in enumerate(phi_angles):
            if i < c2_data.shape[0]:
                plot_correlation(
                    c2_data[i],
                    save_path=plots_dir / f"data_correlation_phi{int(phi)}.png",
                )
    else:
        c2_2d = c2_data[0] if c2_data.ndim == 3 else c2_data
        plot_correlation(c2_2d, save_path=plots_dir / "data_correlation.png")


def _dispatch_simulated_plots(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    nlsq_results: list[NLSQResult] | None,
    plots_dir: Path,
    data_phi_angles: np.ndarray | None = None,
) -> None:
    """Plot model/fit comparisons and component decompositions.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data (2D or 3D).
        nlsq_results: NLSQ results (if any).
        plots_dir: Directory for saving plots.
        data_phi_angles: Phi angles of each c2_data slice (for correct slice selection).
    """
    from heterodyne.viz.experimental_plots import plot_g1_components
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    # Model component plots
    try:
        plot_g1_components(model, save_path=plots_dir / "g1_components.png")
    except Exception:
        logger.exception("Failed to generate g1 component plots")

    # NLSQ plots
    if nlsq_results:
        for nlsq_result in nlsq_results:
            phi = nlsq_result.metadata.get("phi_angle", 0)
            # Select the data slice closest to the fitted phi angle
            if c2_data.ndim == 3:
                if data_phi_angles is not None and len(data_phi_angles) == c2_data.shape[0]:
                    idx = int(np.argmin(np.abs(data_phi_angles - float(phi))))
                    c2_2d = c2_data[idx]
                else:
                    c2_2d = c2_data[0]
            else:
                c2_2d = c2_data
            suffix = f"_phi{int(phi)}" if len(nlsq_results) > 1 else ""

            try:
                plot_nlsq_fit(
                    c2_2d,
                    nlsq_result,
                    save_path=plots_dir / f"nlsq_fit{suffix}.png",
                )
            except Exception:
                logger.exception("Failed to generate NLSQ fit plot for phi=%s", phi)

            try:
                plot_residual_map(
                    nlsq_result,
                    c2_2d,
                    save_path=plots_dir / f"nlsq_residuals{suffix}.png",
                )
            except Exception:
                logger.exception("Failed to generate residual map for phi=%s", phi)


# ---------------------------------------------------------------------------
# Top-level plot dispatch entry point
# ---------------------------------------------------------------------------


def handle_plotting(
    args: Any,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Handle all plotting options from CLI arguments.

    Reads ``args.plot_experimental_data``, ``args.plot_simulated_data``,
    and ``args.save_plots`` flags and dispatches accordingly. Each plot
    operation is wrapped in ``try/except`` so a failure in one does not
    abort the others.

    Args:
        args: Parsed CLI namespace.
        result: Optimization result (may be None for plot-only modes).
        data: Data dictionary.
        config: Configuration dictionary (needed for simulated plots).
    """
    output_dir = getattr(args, "output", None) or Path(".")
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_exp = getattr(args, "plot_experimental_data", False)
    plot_sim = getattr(args, "plot_simulated_data", False)
    save_plots = getattr(args, "save_plots", False)

    if plot_exp:
        try:
            _plot_experimental_data(data, plots_dir)
        except Exception:
            logger.exception("Failed to generate experimental data plots")

    if plot_sim and config is not None:
        try:
            contrast = getattr(args, "contrast", 0.3)
            offset = getattr(args, "offset_sim", 1.0)
            phi_angles_str = getattr(args, "phi_angles", None)
            _plot_simulated_data(config, contrast, offset, phi_angles_str, plots_dir, data)
        except Exception:
            logger.exception("Failed to generate simulated data plots")

    if save_plots and result is not None:
        try:
            _plot_fit_comparison(result, data, plots_dir)
        except Exception:
            logger.exception("Failed to generate fit comparison plot")


def dispatch_plots(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    nlsq_results: list[NLSQResult] | None = None,
    cmc_results: list[CMCResult] | None = None,
    output_dir: Path | None = None,
    mode: str = "both",
    phi_angles: list[float] | None = None,
    data_dict: dict[str, Any] | None = None,
) -> None:
    """Generate all diagnostic plots for the analysis results.

    Dispatches to the appropriate plotting functions based on which
    results are available and the selected mode. Each operation is
    wrapped in ``try/except`` for error isolation.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data.
        nlsq_results: NLSQ results (if any).
        cmc_results: CMC results (if any).
        output_dir: Output directory for plots.
        mode: Plot dispatch mode - "experimental", "simulated", or "both".
        phi_angles: Phi angles for filtering/labeling per-angle plots.
        data_dict: Full data dictionary for rich experimental plots.
    """
    import matplotlib

    matplotlib.use("Agg")

    if output_dir is None:
        logger.warning("No output directory specified; skipping plot generation")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info("Generating plots in %s (mode=%s)", plots_dir, mode)

    # Filter results by phi_angles if specified
    filtered_nlsq: list[NLSQResult] | None = _filter_by_phi(nlsq_results, phi_angles)  # type: ignore[assignment]
    filtered_cmc: list[CMCResult] | None = _filter_by_phi(cmc_results, phi_angles)  # type: ignore[assignment]

    # Dispatch based on mode
    if mode in ("experimental", "both"):
        try:
            _dispatch_experimental_plots(c2_data, plots_dir, phi_angles, data_dict)
        except Exception:
            logger.exception("Failed to generate experimental plots")

    if mode in ("simulated", "both"):
        try:
            _data_phi = (
                np.asarray(data_dict["phi_angles_list"], dtype=float)
                if data_dict is not None and "phi_angles_list" in data_dict
                else None
            )
            _dispatch_simulated_plots(model, c2_data, filtered_nlsq, plots_dir, _data_phi)
            
            # Generate per-angle fitted simulations from NLSQ results.
            # Each result is scoped to its own phi angle so that calls do not
            # overwrite each other and each uses the correct fitted parameters.
            if filtered_nlsq and data_dict is not None and output_dir is not None:
                from heterodyne.viz.nlsq_plots import generate_and_plot_fitted_simulations

                config: dict[str, Any] = data_dict.get("config") or {}
                for i, nlsq_result in enumerate(filtered_nlsq):
                    try:
                        phi = nlsq_result.metadata.get("phi_angle", 0)
                        logger.debug(
                            "Generating fitted simulations for phi=%s (result %d/%d)",
                            phi, i + 1, len(filtered_nlsq),
                        )
                        # Narrow to this angle only — prevents overwriting other
                        # angles' outputs and ensures correct per-angle parameters.
                        phi_data = {
                            **data_dict,
                            "phi_angles_list": np.array([float(phi)]),
                        }
                        generate_and_plot_fitted_simulations(
                            result=nlsq_result,
                            data=phi_data,
                            config=config,
                            output_dir=output_dir,
                            angle_filter_func=None,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to generate fitted simulations for phi=%s",
                            nlsq_result.metadata.get("phi_angle", 0),
                        )
        except Exception:
            logger.exception("Failed to generate simulated plots")

    # CMC plots (generated in "simulated" or "both" modes)
    if mode in ("simulated", "both") and filtered_cmc:
        try:
            _dispatch_cmc_plots(filtered_cmc, plots_dir)
        except Exception:
            logger.exception("Failed to generate CMC plots")

    logger.info("Plots generated")


def _filter_by_phi(
    results: list[NLSQResult] | list[CMCResult] | None,
    phi_angles: list[float] | None,
) -> list[NLSQResult] | list[CMCResult] | None:
    """Filter results list to only those matching specified phi angles.

    Args:
        results: List of results to filter.
        phi_angles: Angles to keep. If None, return all results.

    Returns:
        Filtered list or original list if no filtering needed.
    """
    if results is None or phi_angles is None:
        return results

    filtered = [r for r in results if r.metadata.get("phi_angle", 0) in phi_angles]
    return filtered if filtered else results  # type: ignore[return-value]


def _dispatch_cmc_plots(
    cmc_results: list[CMCResult],
    plots_dir: Path,
) -> None:
    """Generate CMC posterior, trace, and diagnostic plots.

    Args:
        cmc_results: CMC results to plot.
        plots_dir: Directory for saving plots.
    """
    from heterodyne.viz.mcmc_diagnostics import (
        plot_convergence_diagnostics,
        plot_kl_divergence_matrix,
    )
    from heterodyne.viz.mcmc_plots import plot_posterior, plot_trace

    for cmc_result in cmc_results:
        phi = cmc_result.metadata.get("phi_angle", 0)
        suffix = f"_phi{int(phi)}" if len(cmc_results) > 1 else ""

        try:
            plot_posterior(
                cmc_result,
                save_path=plots_dir / f"cmc_posterior{suffix}.png",
            )
        except Exception:
            logger.exception("Failed to generate CMC posterior for phi=%s", phi)

        try:
            plot_trace(
                cmc_result,
                save_path=plots_dir / f"cmc_trace{suffix}.png",
            )
        except Exception:
            logger.exception("Failed to generate CMC trace for phi=%s", phi)

        try:
            plot_convergence_diagnostics(
                cmc_result,
                save_path=plots_dir / f"cmc_convergence{suffix}.png",
            )
        except Exception:
            logger.exception("Failed to generate convergence plot for phi=%s", phi)

        try:
            plot_kl_divergence_matrix(
                cmc_result,
                save_path=plots_dir / f"cmc_kl_divergence{suffix}.png",
            )
        except Exception:
            logger.exception("Failed to generate KL divergence plot for phi=%s", phi)
