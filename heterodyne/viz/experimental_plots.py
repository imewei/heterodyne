"""Visualization for experimental XPCS data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel


def plot_correlation(
    c2: np.ndarray,
    t: np.ndarray | None = None,
    title: str = "Two-Time Correlation",
    save_path: Path | str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot two-time correlation matrix.
    
    Args:
        c2: Correlation matrix, shape (N, N)
        t: Time array
        title: Plot title
        save_path: Optional save path
        cmap: Colormap name
        vmin, vmax: Color scale limits
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if t is None:
        t = np.arange(c2.shape[0])
    
    extent = [t[0], t[-1], t[-1], t[0]]
    
    im = ax.imshow(
        c2,
        extent=extent,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='upper',
    )
    
    ax.set_xlabel("t₂", fontsize=12)
    ax.set_ylabel("t₁", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("c₂(t₁, t₂)", fontsize=12)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_g1_components(
    model: HeterodyneModel,
    params: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot g1 correlation components (reference and sample).
    
    Args:
        model: HeterodyneModel instance
        params: Parameter array (uses model's current params if None)
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t = np.asarray(model.t)
    
    # g1 correlations
    g1_ref = np.asarray(model.compute_g1_reference(params))
    g1_sample = np.asarray(model.compute_g1_sample(params))
    fraction = np.asarray(model.compute_fraction(params))
    
    # Top left: g1 reference
    axes[0, 0].semilogy(t, g1_ref, 'b-', lw=2, label='g₁ reference')
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("g₁")
    axes[0, 0].set_title("Reference Component")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Top right: g1 sample
    axes[0, 1].semilogy(t, g1_sample, 'r-', lw=2, label='g₁ sample')
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("g₁")
    axes[0, 1].set_title("Sample Component")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Bottom left: both g1 together
    axes[1, 0].semilogy(t, g1_ref, 'b-', lw=2, label='Reference')
    axes[1, 0].semilogy(t, g1_sample, 'r-', lw=2, label='Sample')
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("g₁")
    axes[1, 0].set_title("g₁ Comparison")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Bottom right: fraction
    axes[1, 1].plot(t, fraction, 'g-', lw=2, label='f_sample')
    axes[1, 1].plot(t, 1 - fraction, 'm--', lw=2, label='f_reference')
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("Component Fractions")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_diagonal_decay(
    c2: np.ndarray,
    t: np.ndarray | None = None,
    fitted_c2: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot diagonal of correlation matrix (autocorrelation decay).
    
    Args:
        c2: Correlation matrix
        t: Time array
        fitted_c2: Optional fitted correlation matrix
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if t is None:
        t = np.arange(c2.shape[0])
    
    diag = np.diag(c2)
    ax.plot(t, diag, 'bo-', markersize=3, label='Data', alpha=0.7)
    
    if fitted_c2 is not None:
        fitted_diag = np.diag(fitted_c2)
        ax.plot(t, fitted_diag, 'r-', lw=2, label='Fit')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("c₂(t, t)")
    ax.set_title("Diagonal (Autocorrelation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_phi_dependence(
    c2_multi_phi: np.ndarray,
    phi_angles: np.ndarray,
    t_slice: int | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot phi angle dependence of correlation.
    
    Args:
        c2_multi_phi: Correlation data, shape (n_phi, N, N)
        phi_angles: Array of phi angles
        t_slice: Time index for slice (None for t=0)
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    if t_slice is None:
        t_slice = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_phi = len(phi_angles)
    
    # Left: c2 slices at fixed t1
    ax1 = axes[0]
    for i, phi in enumerate(phi_angles):
        c2_slice = c2_multi_phi[i, t_slice, :]
        ax1.plot(c2_slice, label=f'φ={phi:.0f}°')
    ax1.set_xlabel("t₂ index")
    ax1.set_ylabel(f"c₂(t₁={t_slice}, t₂)")
    ax1.set_title("Correlation vs φ angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: diagonal mean vs phi
    ax2 = axes[1]
    diag_means = [np.mean(np.diag(c2_multi_phi[i])) for i in range(n_phi)]
    ax2.plot(phi_angles, diag_means, 'o-', markersize=8)
    ax2.set_xlabel("φ (degrees)")
    ax2.set_ylabel("Mean diagonal c₂")
    ax2.set_title("φ Dependence of Diagonal")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
