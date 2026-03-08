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

# Diagnostic threshold constants
ESS_THRESHOLD = 400
RHAT_THRESHOLD = 1.1
BFMI_THRESHOLD = 0.3


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

    ax.axhline(y=ESS_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Min recommended ({ESS_THRESHOLD})")
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
        colors = ["red" if rh > RHAT_THRESHOLD else "C0" for rh in result.r_hat]
        ax_rhat.bar(x, result.r_hat, color=colors, alpha=0.8)
        ax_rhat.axhline(y=RHAT_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({RHAT_THRESHOLD})")
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
        colors = ["red" if b < BFMI_THRESHOLD else "C0" for b in result.bfmi]
        ax_bfmi.bar(chain_idx, result.bfmi, color=colors, alpha=0.8)
        ax_bfmi.axhline(y=BFMI_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Min threshold ({BFMI_THRESHOLD})")
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


def plot_kl_divergence_matrix(
    result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot pairwise KL divergence heatmap between parameter posteriors.

    Computes histogram-based KL divergence for each pair of parameters
    using 50-bin histograms with epsilon smoothing.

    Args:
        result: CMC result with posterior samples.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    names = result.parameter_names
    n_params = len(names)
    kl_matrix = np.zeros((n_params, n_params))
    eps = 1e-10
    n_bins = 50

    if result.samples is not None:
        for i in range(n_params):
            samples_i = np.asarray(result.samples[names[i]]).ravel()
            for j in range(n_params):
                if i == j:
                    continue
                samples_j = np.asarray(result.samples[names[j]]).ravel()
                # Shared range for both histograms
                lo = min(float(np.min(samples_i)), float(np.min(samples_j)))
                hi = max(float(np.max(samples_i)), float(np.max(samples_j)))
                if lo == hi:
                    continue
                bins = np.linspace(lo, hi, n_bins + 1)
                p, _ = np.histogram(samples_i, bins=bins, density=True)
                q, _ = np.histogram(samples_j, bins=bins, density=True)
                # Normalize to probability distributions
                p = p / (np.sum(p) + eps)
                q = q / (np.sum(q) + eps)
                # KL(p || q)
                p_safe = p + eps
                q_safe = q + eps
                kl_matrix[i, j] = float(np.sum(p_safe * np.log(p_safe / q_safe)))

    fig, ax = plt.subplots(figsize=(max(6, n_params * 0.6), max(5, n_params * 0.5)))
    im = ax.imshow(kl_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise KL Divergence")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved KL divergence matrix to %s", save_path)
        plt.close(fig)

    return fig


def plot_convergence_diagnostics(
    result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot a 2x2 MCMC convergence summary figure.

    Panels:
        (0,0) ESS bulk bar chart
        (0,1) R-hat bar chart with threshold
        (1,0) BFMI bar chart per chain with threshold
        (1,1) Text summary of key convergence statistics

    Args:
        result: CMC result with convergence diagnostics.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("MCMC Convergence Summary", fontsize=14)

    names = result.parameter_names
    x = np.arange(len(names))

    # Panel (0,0): ESS bulk
    ax_ess = axes[0, 0]
    if result.ess_bulk is not None:
        ax_ess.bar(x, result.ess_bulk, color="C0", alpha=0.8)
        ax_ess.axhline(y=ESS_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Min recommended ({ESS_THRESHOLD})")
        ax_ess.set_xticks(x)
        ax_ess.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax_ess.set_ylabel("ESS (bulk)")
        ax_ess.set_title("Effective Sample Size (bulk)")
        ax_ess.legend(fontsize=7)
    else:
        ax_ess.text(0.5, 0.5, "ESS not available", ha="center", va="center",
                    transform=ax_ess.transAxes)
        ax_ess.set_title("Effective Sample Size (bulk)")

    # Panel (0,1): R-hat
    ax_rhat = axes[0, 1]
    if result.r_hat is not None:
        colors = ["red" if rh > RHAT_THRESHOLD else "C0" for rh in result.r_hat]
        ax_rhat.bar(x, result.r_hat, color=colors, alpha=0.8)
        ax_rhat.axhline(y=RHAT_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({RHAT_THRESHOLD})")
        ax_rhat.set_xticks(x)
        ax_rhat.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax_rhat.set_ylabel("R-hat")
        ax_rhat.set_title("R-hat by Parameter")
        ax_rhat.legend(fontsize=7)
    else:
        ax_rhat.text(0.5, 0.5, "R-hat not available", ha="center", va="center",
                     transform=ax_rhat.transAxes)
        ax_rhat.set_title("R-hat by Parameter")

    # Panel (1,0): BFMI
    ax_bfmi = axes[1, 0]
    if result.bfmi is not None:
        chain_idx = np.arange(len(result.bfmi))
        colors_bfmi = ["red" if b < 0.3 else "C0" for b in result.bfmi]
        ax_bfmi.bar(chain_idx, result.bfmi, color=colors_bfmi, alpha=0.8)
        ax_bfmi.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Min threshold (0.3)")
        ax_bfmi.set_xlabel("Chain")
        ax_bfmi.set_ylabel("BFMI")
        ax_bfmi.set_title("Bayesian Fraction of Missing Information")
        ax_bfmi.legend(fontsize=7)
    else:
        ax_bfmi.text(0.5, 0.5, "BFMI not available", ha="center", va="center",
                     transform=ax_bfmi.transAxes)
        ax_bfmi.set_title("Bayesian Fraction of Missing Information")

    # Panel (1,1): Text summary
    ax_text = axes[1, 1]
    ax_text.axis("off")
    ax_text.set_title("Convergence Summary")

    divergences = result.metadata.get("divergent_transitions")
    total_div = int(np.sum(divergences)) if divergences is not None else None

    min_ess = float(np.min(result.ess_bulk)) if result.ess_bulk is not None else None
    max_rhat = float(np.max(result.r_hat)) if result.r_hat is not None else None
    min_bfmi = float(np.min(result.bfmi)) if result.bfmi is not None else None

    # Convergence assessment
    if max_rhat is not None and min_ess is not None:
        converged = max_rhat < RHAT_THRESHOLD and min_ess > ESS_THRESHOLD
        assessment = "Converged" if converged else "Not converged"
        assess_color = "green" if converged else "red"
    else:
        assessment = "N/A"
        assess_color = "gray"

    summary_lines = [
        f"Total divergences:  {total_div if total_div is not None else 'N/A'}",
        f"Min ESS (bulk):     {min_ess:.1f}" if min_ess is not None else "Min ESS (bulk):     N/A",
        f"Max R-hat:          {max_rhat:.4f}" if max_rhat is not None else "Max R-hat:          N/A",
        f"Min BFMI:           {min_bfmi:.4f}" if min_bfmi is not None else "Min BFMI:           N/A",
    ]

    y_pos = 0.75
    for line in summary_lines:
        ax_text.text(0.1, y_pos, line, transform=ax_text.transAxes,
                     fontsize=11, fontfamily="monospace", verticalalignment="top")
        y_pos -= 0.12

    ax_text.text(0.1, y_pos, f"Assessment:         {assessment}",
                 transform=ax_text.transAxes, fontsize=11, fontfamily="monospace",
                 verticalalignment="top", color=assess_color, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved convergence diagnostics to %s", save_path)
        plt.close(fig)

    return fig
