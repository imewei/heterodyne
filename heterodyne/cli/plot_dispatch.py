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


def dispatch_plots(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    nlsq_results: list[NLSQResult] | None = None,
    cmc_results: list[CMCResult] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate all diagnostic plots for the analysis results.

    Dispatches to the appropriate plotting functions based on which
    results are available.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data.
        nlsq_results: NLSQ results (if any).
        cmc_results: CMC results (if any).
        output_dir: Output directory for plots.
    """
    import matplotlib
    matplotlib.use("Agg")

    from heterodyne.viz.experimental_plots import plot_correlation, plot_g1_components
    from heterodyne.viz.mcmc_plots import plot_posterior, plot_trace
    from heterodyne.viz.nlsq_plots import plot_nlsq_fit, plot_residual_map

    if output_dir is None:
        logger.warning("No output directory specified; skipping plot generation")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info("Generating plots in %s", plots_dir)

    # Raw data plot
    c2_2d = c2_data[0] if c2_data.ndim == 3 else c2_data
    plot_correlation(c2_2d, save_path=plots_dir / "data_correlation.png")

    # NLSQ plots
    if nlsq_results:
        for nlsq_result in nlsq_results:
            phi = nlsq_result.metadata.get("phi_angle", 0)
            suffix = f"_phi{int(phi)}" if len(nlsq_results) > 1 else ""

            plot_nlsq_fit(
                c2_2d, nlsq_result,
                save_path=plots_dir / f"nlsq_fit{suffix}.png",
            )
            plot_residual_map(
                nlsq_result, c2_2d,
                save_path=plots_dir / f"nlsq_residuals{suffix}.png",
            )

    # CMC plots
    if cmc_results:
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

    # Model component plots
    plot_g1_components(model, save_path=plots_dir / "g1_components.png")

    logger.info("Plots generated")
