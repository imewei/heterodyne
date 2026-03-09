"""ArviZ-based MCMC visualization for heterodyne analysis.

Provides publication-quality trace, posterior, and pair plots using ArviZ,
with graceful fallback to the basic mcmc_plots module when ArviZ is not
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)


def _has_arviz() -> bool:
    """Check if ArviZ is available."""
    try:
        import arviz  # noqa: F401
        return True
    except ImportError:
        return False


def to_inference_data(cmc_result: CMCResult) -> Any:
    """Convert a CMCResult to an ArviZ InferenceData object.

    Delegates to ``cmc_result_to_arviz`` in the results module.

    Args:
        cmc_result: Completed CMC analysis result with posterior samples.

    Returns:
        ``arviz.InferenceData`` with posterior group.

    Raises:
        ImportError: If ArviZ is not installed.
        ValueError: If samples are empty.
    """
    from heterodyne.optimization.cmc.results import cmc_result_to_arviz
    return cmc_result_to_arviz(cmc_result)


def plot_arviz_trace(
    result: CMCResult,
    var_names: list[str] | None = None,
    save_path: Path | str | None = None,
) -> Figure | None:
    """Plot MCMC trace using ArviZ, with fallback to basic plots.

    Args:
        result: CMC result with posterior samples.
        var_names: Parameter names to plot. None plots all.
        save_path: Path to save the figure. None displays interactively.

    Returns:
        Matplotlib Figure, or None if ArviZ not available and fallback used.
    """
    if not _has_arviz():
        logger.warning("ArviZ not installed; falling back to basic trace plot")
        from heterodyne.viz.mcmc_plots import plot_trace
        return plot_trace(result, save_path=save_path)

    import arviz as az
    import matplotlib.pyplot as plt

    idata = to_inference_data(result)

    axes = az.plot_trace(idata, var_names=var_names, compact=True)
    fig = axes.ravel()[0].get_figure()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved ArviZ trace plot to %s", save_path)
        plt.close(fig)

    return fig  # type: ignore[no-any-return]


def plot_arviz_posterior(
    result: CMCResult,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.95,
    save_path: Path | str | None = None,
) -> Figure | None:
    """Plot posterior distributions using ArviZ.

    Args:
        result: CMC result with posterior samples.
        var_names: Parameter names to plot. None plots all.
        hdi_prob: Highest density interval probability. Default 0.95.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure, or None on fallback.
    """
    if not _has_arviz():
        logger.warning("ArviZ not installed; falling back to basic posterior plot")
        from heterodyne.viz.mcmc_plots import plot_posterior
        return plot_posterior(result, save_path=save_path)

    import arviz as az
    import matplotlib.pyplot as plt

    idata = to_inference_data(result)

    axes = az.plot_posterior(idata, var_names=var_names, hdi_prob=hdi_prob)
    if hasattr(axes, "ravel"):
        fig = axes.ravel()[0].get_figure()
    else:
        fig = axes.get_figure()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved ArviZ posterior plot to %s", save_path)
        plt.close(fig)

    return fig  # type: ignore[no-any-return]


def plot_arviz_pair(
    result: CMCResult,
    var_names: list[str] | None = None,
    save_path: Path | str | None = None,
) -> Figure | None:
    """Plot pairwise posterior relationships using ArviZ.

    Args:
        result: CMC result with posterior samples.
        var_names: Parameter names to include. None uses all.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    if not _has_arviz():
        logger.warning("ArviZ not installed; skipping pair plot")
        return None

    import arviz as az
    import matplotlib.pyplot as plt

    idata = to_inference_data(result)

    axes = az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        marginals=True,
    )
    fig = axes.ravel()[0].get_figure() if hasattr(axes, "ravel") else axes.get_figure()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved ArviZ pair plot to %s", save_path)
        plt.close(fig)

    return fig  # type: ignore[no-any-return]


def _create_empty_figure(title: str = "No data available") -> Figure:
    """Create a minimal placeholder figure with a centered message.

    Used as a safe fallback when a plotting function cannot produce a
    meaningful output (e.g., empty posterior samples, ArviZ conversion
    failure).

    Args:
        title: Text to display in the centre of the figure.

    Returns:
        Matplotlib Figure containing only the message text.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14, color="gray")
    ax.set_axis_off()
    return fig
