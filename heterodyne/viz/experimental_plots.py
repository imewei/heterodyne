"""Visualization for experimental XPCS data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

# Preserve interactive backend if already active; fall back to Agg otherwise
_current_backend = matplotlib.get_backend().lower()
_interactive_backends = ("qt", "gtk", "wx", "tk", "macosx", "nbagg", "webagg")
if not any(_current_backend.startswith(b) for b in _interactive_backends):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from heterodyne.utils.logging import get_logger  # noqa: E402

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel

logger = get_logger(__name__)


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

    if c2.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    if t is None:
        t = np.arange(c2.shape[0])

    if t.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    extent = [t[0], t[-1], t[-1], t[0]]

    im = ax.imshow(
        c2,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    ax.set_xlabel("t₂", fontsize=12)
    ax.set_ylabel("t₁", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("c₂(t₁, t₂)", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    axes[0, 0].semilogy(t, g1_ref, "b-", lw=2, label="g₁ reference")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("g₁")
    axes[0, 0].set_title("Reference Component")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Top right: g1 sample
    axes[0, 1].semilogy(t, g1_sample, "r-", lw=2, label="g₁ sample")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("g₁")
    axes[0, 1].set_title("Sample Component")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Bottom left: both g1 together
    axes[1, 0].semilogy(t, g1_ref, "b-", lw=2, label="Reference")
    axes[1, 0].semilogy(t, g1_sample, "r-", lw=2, label="Sample")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("g₁")
    axes[1, 0].set_title("g₁ Comparison")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Bottom right: fraction
    axes[1, 1].plot(t, fraction, "g-", lw=2, label="f_sample")
    axes[1, 1].plot(t, 1 - fraction, "m--", lw=2, label="f_reference")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("Component Fractions")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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

    if c2.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Diagonal (Autocorrelation)")
        return fig

    if t is None:
        t = np.arange(c2.shape[0])

    if t.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Diagonal (Autocorrelation)")
        return fig

    diag = np.diag(c2)
    ax.plot(t, diag, "bo-", markersize=3, label="Data", alpha=0.7)

    if fitted_c2 is not None:
        fitted_diag = np.diag(fitted_c2)
        ax.plot(t, fitted_diag, "r-", lw=2, label="Fit")

    ax.set_xlabel("Time")
    ax.set_ylabel("c₂(t, t)")
    ax.set_title("Diagonal (Autocorrelation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
        ax1.plot(c2_slice, label=f"φ={phi:.0f}°")
    ax1.set_xlabel("t₂ index")
    ax1.set_ylabel(f"c₂(t₁={t_slice}, t₂)")
    ax1.set_title("Correlation vs φ angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: diagonal mean vs phi
    ax2 = axes[1]
    diag_means = [np.mean(np.diag(c2_multi_phi[i])) for i in range(n_phi)]
    ax2.plot(phi_angles, diag_means, "o-", markersize=8)
    ax2.set_xlabel("φ (degrees)")
    ax2.set_ylabel("Mean diagonal c₂")
    ax2.set_title("φ Dependence of Diagonal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Top-level experimental data entry point (homodyne parity)
# ---------------------------------------------------------------------------


def plot_experimental_data(
    data: dict[str, Any],
    plots_dir: Path,
    angle_filter_func: Any | None = None,
) -> None:
    """Generate validation plots of experimental data.

    Handles 1D, 2D, and 3D (multi-phi) correlation data. For 3D data,
    generates per-angle heatmaps and a combined diagonal comparison plot.

    Args:
        data: Dictionary containing ``c2_exp`` (correlation matrix),
            ``t1`` and ``t2`` (time arrays), ``phi_angles_list`` (angles).
        plots_dir: Output directory for plot files.
        angle_filter_func: Optional filter with signature
            ``(phi_angles, c2_exp, data) -> (indices, filtered_phi, filtered_c2)``.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    c2_exp = np.asarray(data.get("c2_exp", data.get("c2", np.array([]))))
    if c2_exp.size == 0:
        logger.warning("Empty c2 data — skipping experimental plots")
        return

    t1 = data.get("t1")
    t2 = data.get("t2")
    phi_angles_list = data.get("phi_angles_list", data.get("phi_angles"))

    # Physical time extent for axis labels
    if t1 is not None and t2 is not None and len(t1) > 0 and len(t2) > 0:
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        extent: list[float] | None = [
            float(t2[0]),
            float(t2[-1]),
            float(t1[-1]),
            float(t1[0]),
        ]
        xlabel, ylabel = "t₂ (s)", "t₁ (s)"
    else:
        extent = None
        xlabel, ylabel = "t₂ index", "t₁ index"

    if c2_exp.ndim == 3 and phi_angles_list is not None:
        phi_angles_list = np.asarray(phi_angles_list)

        # Apply optional angle filtering
        if angle_filter_func is not None:
            try:
                _, phi_angles_list, c2_exp = angle_filter_func(
                    phi_angles_list, c2_exp, data
                )
            except Exception:
                logger.warning("Angle filtering failed; using all angles")

        _plot_3d_experimental_data(
            c2_exp, phi_angles_list, t1, extent, xlabel, ylabel, plots_dir
        )
    elif c2_exp.ndim == 2:
        _plot_2d_experimental_data(c2_exp, extent, xlabel, ylabel, plots_dir)
    elif c2_exp.ndim == 1:
        _plot_1d_experimental_data(c2_exp, plots_dir)
    else:
        logger.warning("Unexpected c2 shape %s — skipping plots", c2_exp.shape)


def _plot_3d_experimental_data(
    c2_exp: np.ndarray,
    phi_angles_list: np.ndarray,
    t1: np.ndarray | None,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot per-angle heatmaps and diagonal comparison for 3D c2."""
    n_phi = c2_exp.shape[0]

    for i in range(n_phi):
        phi_deg = float(phi_angles_list[i])
        mat = c2_exp[i]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            mat,
            extent=extent,
            aspect="auto",
            cmap="jet",
            origin="lower",
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Experimental C₂ — φ = {phi_deg:.1f}°", fontsize=14)

        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("c₂(t₁, t₂)", fontsize=12)

        # Stats overlay
        finite_vals = mat[np.isfinite(mat)]
        if finite_vals.size > 0:
            stats_text = (
                f"mean={np.nanmean(finite_vals):.4g}\n"
                f"range=[{np.nanmin(finite_vals):.4g}, {np.nanmax(finite_vals):.4g}]"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        fig.savefig(
            plots_dir / f"experimental_data_phi_{phi_deg:.1f}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Diagonal comparison plot (cap at 10 angles for readability)
    fig, ax = plt.subplots(figsize=(10, 6))
    n_diag = min(n_phi, 10)
    for i in range(n_diag):
        phi_deg = float(phi_angles_list[i])
        diag = np.diag(c2_exp[i])
        t_axis = t1 if t1 is not None and len(t1) == len(diag) else np.arange(len(diag))
        ax.plot(t_axis, diag, label=f"φ={phi_deg:.1f}°")

    ax.set_xlabel("Time (s)" if t1 is not None else "Index")
    ax.set_ylabel("c₂(t, t)")
    ax.set_title("Diagonal (Autocorrelation) — All Angles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        plots_dir / "experimental_data_diagonal.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    logger.info("Saved %d per-angle heatmaps + diagonal plot", n_phi)


def _plot_2d_experimental_data(
    c2_exp: np.ndarray,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot single 2D correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(c2_exp, extent=extent, aspect="auto", cmap="jet", origin="lower")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Experimental C₂", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("c₂(t₁, t₂)", fontsize=12)

    finite_vals = c2_exp[np.isfinite(c2_exp)]
    if finite_vals.size > 0:
        stats_text = (
            f"mean={np.nanmean(finite_vals):.4g}\n"
            f"range=[{np.nanmin(finite_vals):.4g}, {np.nanmax(finite_vals):.4g}]"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved experimental_data.png")


def _plot_1d_experimental_data(
    c2_exp: np.ndarray,
    plots_dir: Path,
) -> None:
    """Plot 1D correlation data as a line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c2_exp, "bo-", markersize=3, alpha=0.7)
    ax.set_xlabel("Index")
    ax.set_ylabel("c₂")
    ax.set_title("Experimental Correlation (1D)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved experimental_data.png (1D)")


def plot_fit_comparison(
    result: Any,
    data: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate side-by-side comparison of experimental data and fit.

    Args:
        result: Optimization result with ``c2_fitted`` or ``fitted_params`` attribute.
        data: Data dictionary with ``c2_exp``, ``t1``, ``t2``.
        plots_dir: Output directory for the plot.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    c2_exp = np.asarray(data.get("c2_exp", data.get("c2", np.array([]))))
    if c2_exp.size == 0:
        logger.warning("Empty c2 data — skipping fit comparison plot")
        return

    # Use first slice for 3D data
    if c2_exp.ndim == 3:
        c2_exp = c2_exp[0]

    t1 = data.get("t1")
    t2 = data.get("t2")
    if t1 is not None and t2 is not None and len(t1) > 0 and len(t2) > 0:
        t1, t2 = np.asarray(t1), np.asarray(t2)
        extent: list[float] | None = [
            float(t2[0]),
            float(t2[-1]),
            float(t1[-1]),
            float(t1[0]),
        ]
    else:
        extent = None

    if c2_exp.ndim == 1:
        # 1D comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(c2_exp, "b-", lw=1.5, label="Experimental")
        axes[0].set_title("Experimental Data")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("c₂")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Fit (placeholder)")
        axes[1].text(
            0.5, 0.5, "Fit data", ha="center", va="center", transform=axes[1].transAxes
        )
    else:
        # 2D heatmap comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        im = axes[0].imshow(
            c2_exp,
            extent=extent,
            aspect="auto",
            cmap="jet",
            vmin=1.0,
            vmax=1.5,
            origin="lower",
        )
        axes[0].set_title("Experimental Data", fontsize=14)
        axes[0].set_xlabel("t₂ (s)", fontsize=12)
        axes[0].set_ylabel("t₁ (s)", fontsize=12)
        plt.colorbar(im, ax=axes[0], shrink=0.9)

        axes[1].set_title("Fit (placeholder)", fontsize=14)
        axes[1].text(
            0.5, 0.5, "Fit data", ha="center", va="center", transform=axes[1].transAxes
        )

    plt.tight_layout()
    fig.savefig(plots_dir / "fit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved fit_comparison.png")
