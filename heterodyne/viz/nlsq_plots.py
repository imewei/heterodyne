"""Visualization for NLSQ fitting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult


def plot_nlsq_fit(
    c2_data: np.ndarray,
    result: NLSQResult,
    t: np.ndarray | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (15, 5),
) -> plt.Figure:
    """Plot NLSQ fit comparison.

    Creates three-panel figure:
    1. Experimental data
    2. Fitted model
    3. Residuals

    Args:
        c2_data: Experimental correlation data
        result: NLSQ fitting result
        t: Optional time array
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if c2_data.size == 0:
        fig.suptitle("No data available")
        return fig

    if t is None:
        t = np.arange(c2_data.shape[0])

    if t.size == 0:
        fig.suptitle("No data available")
        return fig

    # Data plot
    im0 = axes[0].imshow(
        c2_data,
        extent=[t[0], t[-1], t[-1], t[0]],
        aspect='auto',
        cmap='viridis',
    )
    axes[0].set_title("Experimental Data")
    axes[0].set_xlabel("t₂")
    axes[0].set_ylabel("t₁")
    plt.colorbar(im0, ax=axes[0], label="c₂")

    # Model plot
    if result.fitted_correlation is not None:
        im1 = axes[1].imshow(
            result.fitted_correlation,
            extent=[t[0], t[-1], t[-1], t[0]],
            aspect='auto',
            cmap='viridis',
        )
        axes[1].set_title("Fitted Model")
        axes[1].set_xlabel("t₂")
        axes[1].set_ylabel("t₁")
        plt.colorbar(im1, ax=axes[1], label="c₂")
    else:
        axes[1].text(0.5, 0.5, "No fitted correlation", ha='center', va='center')
        axes[1].set_title("Fitted Model")

    # Residual plot
    if result.fitted_correlation is not None:
        if result.residuals is not None:
            residual_2d = result.residuals
        else:
            residual_2d = c2_data - result.fitted_correlation
        vmax = np.nanpercentile(np.abs(residual_2d), 99)
        im2 = axes[2].imshow(
            residual_2d,
            extent=[t[0], t[-1], t[-1], t[0]],
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
        )
        axes[2].set_title("Residuals")
        axes[2].set_xlabel("t₂")
        axes[2].set_ylabel("t₁")
        plt.colorbar(im2, ax=axes[2], label="Residual")
    else:
        axes[2].text(0.5, 0.5, "No residuals", ha='center', va='center')
        axes[2].set_title("Residuals")

    # Add fit statistics
    chi2 = result.reduced_chi_squared
    stats_text = f"χ²_red = {chi2:.3f}" if chi2 is not None else ""
    fig.suptitle(f"NLSQ Fit Results  {stats_text}", fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_residual_map(
    result: NLSQResult,
    c2_data: np.ndarray,
    t: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot detailed residual analysis.

    Args:
        result: NLSQ result
        c2_data: Original data
        t: Time array
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    if result.fitted_correlation is None:
        fig.suptitle("No fitted correlation available")
        return fig

    if result.residuals is not None:
        residuals = result.residuals
    else:
        residuals = c2_data - result.fitted_correlation

    if t is None:
        t = np.arange(c2_data.shape[0])

    # 2D residual map
    vmax = np.nanpercentile(np.abs(residuals), 99)
    im = axes[0, 0].imshow(
        residuals,
        extent=[t[0], t[-1], t[-1], t[0]],
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 0].set_title("Residual Map")
    axes[0, 0].set_xlabel("t₂")
    axes[0, 0].set_ylabel("t₁")
    plt.colorbar(im, ax=axes[0, 0])

    # Histogram of residuals
    axes[0, 1].hist(residuals.ravel(), bins=50, density=True, alpha=0.7)
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Residual Distribution")

    # Add normal distribution overlay
    mu, sigma = np.nanmean(residuals), np.nanstd(residuals)
    if sigma > 0:
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        axes[0, 1].plot(x, np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)),
                        'r-', lw=2, label=f'Normal(μ={mu:.2e}, σ={sigma:.2e})')
    axes[0, 1].legend()

    # Residual along diagonal
    diag_residuals = np.diag(residuals)
    axes[1, 0].plot(t, diag_residuals, 'b-', lw=1)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Residual")
    axes[1, 0].set_title("Diagonal Residuals")

    # Residual vs fitted value
    axes[1, 1].scatter(
        result.fitted_correlation.ravel(),
        residuals.ravel(),
        alpha=0.1,
        s=1,
    )
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel("Fitted Value")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Residuals vs Fitted")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_parameter_uncertainties(
    result: NLSQResult,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot parameter values with uncertainties.

    Args:
        result: NLSQ result
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_params = len(result.parameter_names)
    x = np.arange(n_params)

    # Normalize parameters for visualization
    params = result.parameters
    if result.uncertainties is not None:
        errors = np.maximum(result.uncertainties, 0.0)
    else:
        errors = np.zeros(n_params)

    # Plot as errorbar
    ax.errorbar(x, params, yerr=errors, fmt='o', capsize=5, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(result.parameter_names, rotation=45, ha='right')
    ax.set_ylabel("Parameter Value")
    ax.set_title("Fitted Parameters with Uncertainties")
    ax.grid(True, alpha=0.3)

    # Use log scale if values span many orders of magnitude
    nonzero_params = params[params != 0]
    if len(nonzero_params) > 0 and np.max(np.abs(params)) / np.min(np.abs(nonzero_params)) > 100:
        ax.set_yscale('symlog')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
