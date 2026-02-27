"""Visualization for MCMC/CMC results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult


def plot_posterior(
    result: CMCResult,
    params: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot posterior distributions.

    Args:
        result: CMC result with samples
        params: Parameters to plot (None for all)
        save_path: Optional save path
        figsize: Optional figure size

    Returns:
        Matplotlib figure
    """
    if result.samples is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No samples available", ha='center', va='center')
        return fig

    if params is None:
        params = result.parameter_names

    n_params = len(params)
    if n_params == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
        return fig

    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, name in enumerate(params):
        ax = axes[i]

        if name not in result.samples:
            ax.text(0.5, 0.5, f"{name}: No samples", ha='center', va='center')
            continue

        samples = result.samples[name].ravel()

        # Histogram
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')

        # Add mean and credible interval lines
        idx = result.parameter_names.index(name)
        mean = result.posterior_mean[idx]
        std = result.posterior_std[idx]

        ax.axvline(mean, color='red', linestyle='-', lw=2, label=f'Mean: {mean:.3e}')
        ax.axvline(mean - std, color='red', linestyle='--', alpha=0.5)
        ax.axvline(mean + std, color='red', linestyle='--', alpha=0.5)

        # Credible interval
        if name in result.credible_intervals:
            ci = result.credible_intervals[name]
            if "2.5%" in ci:
                ax.axvline(ci["2.5%"], color='green', linestyle=':', alpha=0.7)
            if "97.5%" in ci:
                ax.axvline(ci["97.5%"], color='green', linestyle=':', alpha=0.7)

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        if result.r_hat is not None and idx < len(result.r_hat):
            rhat_str = f" (R-hat={result.r_hat[idx]:.3f})"
        else:
            rhat_str = ""
        ax.set_title(f"{name}{rhat_str}")

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_trace(
    result: CMCResult,
    params: list[str] | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot trace plots for MCMC chains.

    Args:
        result: CMC result with samples
        params: Parameters to plot
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if result.samples is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No samples available", ha='center', va='center')
        return fig

    if params is None:
        params = result.parameter_names

    n_params = len(params)
    if n_params == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(params):
        if name not in result.samples:
            continue

        samples = result.samples[name]

        # Trace plot
        ax_trace = axes[i, 0]
        if samples.ndim == 2:
            # Multiple chains
            for chain in range(samples.shape[0]):
                ax_trace.plot(samples[chain], alpha=0.7, lw=0.5)
        else:
            ax_trace.plot(samples, alpha=0.7, lw=0.5)

        ax_trace.set_ylabel(name)
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_title(f"{name} - Trace")

        # Posterior histogram
        ax_hist = axes[i, 1]
        ax_hist.hist(samples.ravel(), bins=50, density=True, alpha=0.7)

        # Add statistics
        idx = result.parameter_names.index(name)
        mean = result.posterior_mean[idx]
        ax_hist.axvline(mean, color='red', linestyle='-', lw=2)

        ax_hist.set_xlabel(name)
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{name} - Posterior")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_corner(
    result: CMCResult,
    params: list[str] | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot corner plot showing parameter correlations.

    Args:
        result: CMC result
        params: Parameters to include
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if result.samples is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No samples available", ha='center', va='center')
        return fig

    if params is None:
        params = result.parameter_names

    n_params = len(params)
    if n_params == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(n_params, n_params, figsize=(2 * n_params, 2 * n_params))
    if n_params == 1:
        axes = np.array([[axes]])

    for i, name_i in enumerate(params):
        for j, name_j in enumerate(params):
            ax = axes[i, j]

            if i < j:
                # Upper triangle: hide
                ax.set_visible(False)
            elif i == j:
                # Diagonal: histogram
                if name_i in result.samples:
                    ax.hist(result.samples[name_i].ravel(), bins=30, density=True, alpha=0.7)
                ax.set_yticks([])
            else:
                # Lower triangle: scatter/contour
                if name_i in result.samples and name_j in result.samples:
                    ax.scatter(
                        result.samples[name_j].ravel(),
                        result.samples[name_i].ravel(),
                        alpha=0.1,
                        s=1,
                    )

            # Labels
            if i == n_params - 1:
                ax.set_xlabel(name_j, fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(name_i, fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
