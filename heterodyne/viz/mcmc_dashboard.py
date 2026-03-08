"""MCMC Summary Dashboard Visualization.

Provides a comprehensive multi-panel CMC summary dashboard combining
convergence diagnostics, trace plots, and posterior histograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from heterodyne.utils.logging import get_logger
from heterodyne.utils.path_validation import PathValidationError, validate_output_path
from heterodyne.viz.mcmc_diagnostics import ESS_THRESHOLD, RHAT_THRESHOLD

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)

# Maximum parameters to show in trace/posterior panels (2 columns available)
_MAX_PANEL_PARAMS = 2


def plot_cmc_summary_dashboard(
    result: CMCResult,
    figsize: tuple[float, float] = (16, 12),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Create comprehensive multi-panel CMC summary dashboard.

    Combines key diagnostic plots into a single figure:
    - Panel 1: R-hat per parameter (convergence quality)
    - Panel 2: ESS per parameter (sampling efficiency)
    - Panel 3: Trace plots (selected parameters)
    - Panel 4: Posterior histograms (selected parameters)

    Parameters
    ----------
    result : CMCResult
        CMC result object with posterior samples and diagnostics.
    figsize : tuple, default=(16, 12)
        Figure size (width, height).
    save_path : str or Path, optional
        If provided, save figure to this path.
    dpi : int, default=150
        DPI for saved figure.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> plot_cmc_summary_dashboard(cmc_result, save_path='cmc_summary.png')
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    param_names = result.parameter_names
    n_params = len(param_names)

    # Panel 1: R-hat bar chart (top left)
    ax_rhat = fig.add_subplot(gs[0, 0])
    _plot_rhat_panel(ax_rhat, result, param_names, n_params)

    # Panel 2: ESS bar chart (top right)
    ax_ess = fig.add_subplot(gs[0, 1])
    _plot_ess_panel(ax_ess, result, param_names, n_params)

    # Panel 3: Trace plots (middle row, up to 2 parameters)
    num_trace = min(_MAX_PANEL_PARAMS, n_params)
    for i in range(num_trace):
        ax_trace = fig.add_subplot(gs[1, i])
        _plot_trace_panel(ax_trace, result, param_names, i)

    # Panel 4: Posterior histograms (bottom row, up to 2 parameters)
    num_hist = min(_MAX_PANEL_PARAMS, n_params)
    for i in range(num_hist):
        ax_hist = fig.add_subplot(gs[2, i])
        _plot_posterior_panel(ax_hist, result, param_names, i)

    # Overall title
    convergence_str = "PASSED" if result.convergence_passed else "FAILED"
    fig.suptitle(
        f"CMC Summary Dashboard ({result.num_chains} chains, "
        f"convergence: {convergence_str})",
        fontsize=14,
        fontweight="bold",
    )

    if save_path is not None:
        try:
            validated_path = validate_output_path(save_path)
            fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
            logger.info("CMC summary dashboard saved to %s", validated_path.name)
        except (PathValidationError, ValueError) as e:
            logger.warning("Could not save CMC summary dashboard: %s", e)
        plt.close(fig)

    return fig


def _plot_rhat_panel(
    ax: Any,
    result: CMCResult,
    param_names: list[str],
    n_params: int,
) -> None:
    """Plot R-hat bar chart with threshold line."""
    try:
        if result.r_hat is not None:
            r_hat = np.asarray(result.r_hat)
            positions = np.arange(n_params)

            colors = [
                "forestgreen" if v <= RHAT_THRESHOLD else "firebrick"
                for v in r_hat
            ]
            ax.bar(positions, r_hat, color=colors, alpha=0.8, edgecolor="gray")
            ax.axhline(
                y=RHAT_THRESHOLD,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({RHAT_THRESHOLD})",
            )

            ax.set_xticks(positions)
            ax.set_xticklabels(param_names, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("R-hat", fontsize=9)
            ax.set_title("R-hat Convergence", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            _placeholder(ax, "R-hat diagnostics not available")
    except (ValueError, TypeError, IndexError) as e:
        _placeholder(ax, f"Error plotting R-hat:\n{e}")


def _plot_ess_panel(
    ax: Any,
    result: CMCResult,
    param_names: list[str],
    n_params: int,
) -> None:
    """Plot ESS bar chart with threshold line."""
    try:
        if result.ess_bulk is not None:
            ess = np.asarray(result.ess_bulk)
            positions = np.arange(n_params)

            colors = [
                "steelblue" if v >= ESS_THRESHOLD else "orange"
                for v in ess
            ]
            ax.bar(positions, ess, color=colors, alpha=0.8, edgecolor="gray")
            ax.axhline(
                y=ESS_THRESHOLD,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({ESS_THRESHOLD})",
            )

            ax.set_xticks(positions)
            ax.set_xticklabels(param_names, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Effective Sample Size", fontsize=9)
            ax.set_title("ESS (bulk)", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            _placeholder(ax, "ESS diagnostics not available")
    except (ValueError, TypeError, IndexError) as e:
        _placeholder(ax, f"Error plotting ESS:\n{e}")


def _plot_trace_panel(
    ax: Any,
    result: CMCResult,
    param_names: list[str],
    param_idx: int,
) -> None:
    """Plot trace for a single parameter."""
    name = param_names[param_idx]
    try:
        if result.samples is not None and name in result.samples:
            samples = np.asarray(result.samples[name])

            if samples.ndim == 2:
                # (chains, draws) — plot each chain
                n_chains = samples.shape[0]
                colors = matplotlib.colormaps["tab10"](
                    np.linspace(0, 1, max(n_chains, 1))
                )
                for chain_idx in range(n_chains):
                    ax.plot(
                        samples[chain_idx],
                        color=colors[chain_idx],
                        alpha=0.6,
                        linewidth=0.5,
                    )
            else:
                # 1D — single trace
                ax.plot(samples, color="steelblue", alpha=0.7, linewidth=0.5)

            ax.set_xlabel("Sample Index", fontsize=9)
            ax.set_ylabel(name, fontsize=9)
            ax.set_title(f"{name} Trace", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
        else:
            _placeholder(ax, f"No samples for {name}")
    except (ValueError, TypeError, KeyError, IndexError) as e:
        _placeholder(ax, f"Error:\n{e}")


def _plot_posterior_panel(
    ax: Any,
    result: CMCResult,
    param_names: list[str],
    param_idx: int,
) -> None:
    """Plot posterior histogram for a single parameter."""
    name = param_names[param_idx]
    try:
        if result.samples is not None and name in result.samples:
            samples = np.asarray(result.samples[name]).ravel()

            ax.hist(
                samples,
                bins=30,
                alpha=0.7,
                color="steelblue",
                density=True,
            )

            # Vertical line at posterior mean
            idx = result.parameter_names.index(name)
            mean_val = float(result.posterior_mean[idx])
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2e}",
            )

            ax.set_xlabel(name, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"{name} Posterior", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
        else:
            _placeholder(ax, f"No samples for {name}")
    except (ValueError, TypeError, KeyError, IndexError) as e:
        _placeholder(ax, f"Error:\n{e}")


def _placeholder(ax: Any, message: str) -> None:
    """Show centered placeholder text on an axes."""
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )


