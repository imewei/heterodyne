"""Plot generation dispatch for heterodyne CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def _dispatch_experimental_plots(
    c2_data: np.ndarray,
    plots_dir: Path,
    phi_angles: list[float] | None = None,
) -> None:
    """Plot raw correlation data, optionally per angle slice.

    Args:
        c2_data: Correlation data (2D or 3D).
        plots_dir: Directory for saving plots.
        phi_angles: Phi angles for labeling per-slice plots.
    """
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
) -> None:
    """Plot model/fit comparisons and component decompositions.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data (2D or 3D).
        nlsq_results: NLSQ results (if any).
        plots_dir: Directory for saving plots.
    """
    from heterodyne.viz.experimental_plots import plot_g1_components
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    # Model component plots
    plot_g1_components(model, save_path=plots_dir / "g1_components.png")

    # NLSQ plots
    if nlsq_results:
        c2_2d = c2_data[0] if c2_data.ndim == 3 else c2_data
        for nlsq_result in nlsq_results:
            phi = nlsq_result.metadata.get("phi_angle", 0)
            suffix = f"_phi{int(phi)}" if len(nlsq_results) > 1 else ""

            plot_nlsq_fit(
                c2_2d,
                nlsq_result,
                save_path=plots_dir / f"nlsq_fit{suffix}.png",
            )
            plot_residual_map(
                nlsq_result,
                c2_2d,
                save_path=plots_dir / f"nlsq_residuals{suffix}.png",
            )


def dispatch_plots(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    nlsq_results: list[NLSQResult] | None = None,
    cmc_results: list[CMCResult] | None = None,
    output_dir: Path | None = None,
    mode: str = "both",
    phi_angles: list[float] | None = None,
) -> None:
    """Generate all diagnostic plots for the analysis results.

    Dispatches to the appropriate plotting functions based on which
    results are available and the selected mode.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data.
        nlsq_results: NLSQ results (if any).
        cmc_results: CMC results (if any).
        output_dir: Output directory for plots.
        mode: Plot dispatch mode - "experimental", "simulated", or "both".
        phi_angles: Phi angles for filtering/labeling per-angle plots.
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
        _dispatch_experimental_plots(c2_data, plots_dir, phi_angles)

    if mode in ("simulated", "both"):
        _dispatch_simulated_plots(model, c2_data, filtered_nlsq, plots_dir)

    # CMC plots (generated in "simulated" or "both" modes)
    if mode in ("simulated", "both") and filtered_cmc:
        _dispatch_cmc_plots(filtered_cmc, plots_dir)

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

        plot_posterior(
            cmc_result,
            save_path=plots_dir / f"cmc_posterior{suffix}.png",
        )
        plot_trace(
            cmc_result,
            save_path=plots_dir / f"cmc_trace{suffix}.png",
        )
        plot_convergence_diagnostics(
            cmc_result,
            save_path=plots_dir / f"cmc_convergence{suffix}.png",
        )
        plot_kl_divergence_matrix(
            cmc_result,
            save_path=plots_dir / f"cmc_kl_divergence{suffix}.png",
        )
