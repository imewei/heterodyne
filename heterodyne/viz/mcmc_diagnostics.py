"""MCMC convergence diagnostic plots for heterodyne analysis.

Provides visualization of convergence metrics including ESS evolution,
adaptation summaries, and divergence analysis for NUTS sampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)


def plot_ess_evolution(
    result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot effective sample size (ESS) across parameters.

    Shows bulk and tail ESS as a grouped bar chart, with a horizontal
    reference line at the minimum recommended ESS (400).

    Args:
        result: CMC result with ESS diagnostics.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(result.parameter_names) * 0.8), 5))

    x = np.arange(len(result.parameter_names))
    width = 0.35

    if result.ess_bulk is not None:
        ax.bar(x - width / 2, result.ess_bulk, width, label="ESS bulk", color="C0", alpha=0.8)

    if result.ess_tail is not None:
        ax.bar(x + width / 2, result.ess_tail, width, label="ESS tail", color="C1", alpha=0.8)

    ax.axhline(y=400, color="red", linestyle="--", alpha=0.5, label="Min recommended (400)")
    ax.set_xticks(x)
    ax.set_xticklabels(result.parameter_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Effective Sample Size")
    ax.set_title("ESS by Parameter")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved ESS evolution plot to %s", save_path)
        plt.close(fig)

    return fig


def plot_adaptation_summary(
    result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot R-hat convergence diagnostic across parameters.

    Shows R-hat values as a bar chart with a horizontal reference line
    at the convergence threshold (1.1).

    Args:
        result: CMC result with R-hat diagnostics.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: R-hat
    ax_rhat = axes[0]
    if result.r_hat is not None:
        x = np.arange(len(result.parameter_names))
        colors = ["red" if rh > 1.1 else "C0" for rh in result.r_hat]
        ax_rhat.bar(x, result.r_hat, color=colors, alpha=0.8)
        ax_rhat.axhline(y=1.1, color="red", linestyle="--", alpha=0.5, label="Threshold (1.1)")
        ax_rhat.set_xticks(x)
        ax_rhat.set_xticklabels(result.parameter_names, rotation=45, ha="right", fontsize=8)
        ax_rhat.set_ylabel("R-hat")
        ax_rhat.set_title("R-hat by Parameter")
        ax_rhat.legend(fontsize=8)
    else:
        ax_rhat.text(0.5, 0.5, "R-hat not available", ha="center", va="center",
                     transform=ax_rhat.transAxes)

    # Panel 2: BFMI
    ax_bfmi = axes[1]
    if result.bfmi is not None:
        chain_idx = np.arange(len(result.bfmi))
        colors = ["red" if b < 0.3 else "C0" for b in result.bfmi]
        ax_bfmi.bar(chain_idx, result.bfmi, color=colors, alpha=0.8)
        ax_bfmi.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Min threshold (0.3)")
        ax_bfmi.set_xlabel("Chain")
        ax_bfmi.set_ylabel("BFMI")
        ax_bfmi.set_title("Bayesian Fraction of Missing Information")
        ax_bfmi.legend(fontsize=8)
    else:
        ax_bfmi.text(0.5, 0.5, "BFMI not available", ha="center", va="center",
                     transform=ax_bfmi.transAxes)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved adaptation summary to %s", save_path)
        plt.close(fig)

    return fig


def plot_divergence_scatter(
    result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot divergence analysis scatter plot.

    If divergence information is available in result metadata, shows a
    scatter plot of parameter values for divergent vs non-divergent
    transitions. Falls back to a posterior density summary if no
    divergence info is available.

    Args:
        result: CMC result.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    divergent = result.metadata.get("divergent_transitions")

    if divergent is not None and result.samples is not None and len(result.parameter_names) >= 2:
        divergent = np.asarray(divergent, dtype=bool)
        p1_name = result.parameter_names[0]
        p2_name = result.parameter_names[1]
        s1 = np.asarray(result.samples[p1_name])
        s2 = np.asarray(result.samples[p2_name])

        # Trim to match if needed
        n = min(len(s1), len(s2), len(divergent))
        s1, s2, divergent = s1[:n], s2[:n], divergent[:n]

        non_div = ~divergent
        ax.scatter(s1[non_div], s2[non_div], alpha=0.1, s=1, color="C0", label="Non-divergent")
        if np.any(divergent):
            ax.scatter(s1[divergent], s2[divergent], alpha=0.8, s=10, color="red",
                       marker="x", label=f"Divergent ({np.sum(divergent)})")

        ax.set_xlabel(p1_name)
        ax.set_ylabel(p2_name)
        ax.legend(fontsize=8)
        ax.set_title("Divergence Analysis")
    else:
        # Fallback: show posterior means with R-hat coloring
        if result.r_hat is not None:
            x = np.arange(len(result.parameter_names))
            scatter = ax.scatter(
                x, result.posterior_mean,
                c=result.r_hat, cmap="RdYlGn_r", s=80,
                edgecolors="black", linewidths=0.5,
                vmin=0.99, vmax=1.2,
            )
            ax.errorbar(x, result.posterior_mean, yerr=result.posterior_std,
                        fmt="none", ecolor="gray", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(result.parameter_names, rotation=45, ha="right", fontsize=8)
            plt.colorbar(scatter, ax=ax, label="R-hat")
            ax.set_ylabel("Posterior Mean")
            ax.set_title("Posterior Summary (colored by R-hat)")
        else:
            ax.text(0.5, 0.5, "No divergence or diagnostic data available",
                    ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved divergence scatter to %s", save_path)
        plt.close(fig)

    return fig
