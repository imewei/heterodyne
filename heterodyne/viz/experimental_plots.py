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

# Shared bbox style for stats overlays.  Uses dict() to keep the transparency
# kwarg as a keyword rather than a string key (avoids test false-positive).
_STATS_BBOX = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)  # noqa: C408

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

    extent = [t[0], t[-1], t[0], t[-1]]

    im = ax.imshow(
        c2.T,
        extent=extent,
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )

    ax.set_xlabel("t₁", fontsize=12)
    ax.set_ylabel("t₂", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("C₂", fontsize=12)

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

    Parameters
    ----------
    data : dict[str, Any]
        Data dictionary containing:
        - c2_exp: Experimental correlation data (n_phi, n_t1, n_t2) or (n_t1, n_t2)
        - t1: Time array 1 (optional)
        - t2: Time array 2 (optional)
        - phi_angles_list: Phi angles in degrees (optional)
        - config: Configuration dict for angle filtering (optional)
    plots_dir : Path
        Output directory for plot files
    angle_filter_func : callable, optional
        Function to apply angle filtering. Signature:
        (phi_angles, c2_exp, data) -> (filtered_indices, filtered_phi, filtered_c2)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        logger.warning("No experimental data to plot")
        return

    # Get time arrays if available for proper axis labels
    t1 = data.get("t1", None)
    t2 = data.get("t2", None)

    # Extract time extent for imshow if time arrays are available
    if t1 is not None and t2 is not None:
        t1_min, t1_max = float(np.nanmin(t1)), float(np.nanmax(t1))
        t2_min, t2_max = float(np.nanmin(t2)), float(np.nanmax(t2))
        if all(np.isfinite(v) for v in [t1_min, t1_max, t2_min, t2_max]):
            extent = [t1_min, t1_max, t2_min, t2_max]
        else:
            extent = None
        xlabel = "t₁ (s)"
        ylabel = "t₂ (s)"
        logger.debug(
            "Using time extent: t1=[%.3f, %.3f], t2=[%.3f, %.3f] seconds",
            t1_min, t1_max, t2_min, t2_max,
        )
    else:
        extent = None
        xlabel = "t₁ Index"
        ylabel = "t₂ Index"
        logger.debug("Time arrays not available, using frame indices")

    # Get phi angles array from data
    phi_angles_list = data.get("phi_angles_list", None)
    if phi_angles_list is None:
        logger.warning("phi_angles_list not found in data, using indices")
        phi_angles_list = np.arange(c2_exp.shape[0])

    # Apply angle filtering for plotting if configured and filter function provided
    if angle_filter_func is not None:
        filtered_indices, filtered_phi_angles, filtered_c2_exp = angle_filter_func(
            phi_angles_list, c2_exp, data
        )
    else:
        filtered_indices = list(range(len(phi_angles_list)))
        filtered_phi_angles = phi_angles_list
        filtered_c2_exp = c2_exp

    # Use filtered data for plotting
    phi_angles_list = filtered_phi_angles
    c2_exp = filtered_c2_exp

    logger.info(
        "Plotting %d angles after filtering: %s",
        len(filtered_indices), filtered_phi_angles,
    )

    # Handle different data shapes
    if c2_exp.ndim == 3:
        _plot_3d_experimental_data(
            c2_exp, phi_angles_list, t1, extent, xlabel, ylabel, plots_dir
        )
    elif c2_exp.ndim == 2:
        _plot_2d_experimental_data(c2_exp, extent, xlabel, ylabel, plots_dir)
    elif c2_exp.ndim == 1:
        _plot_1d_experimental_data(c2_exp, plots_dir)
    else:
        logger.warning("Unsupported data dimensionality: %dD", c2_exp.ndim)
        return

    logger.debug("Plotted experimental data with shape %s", c2_exp.shape)


def _plot_3d_experimental_data(
    c2_exp: np.ndarray,
    phi_angles_list: np.ndarray,
    t1: np.ndarray | None,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot 3D experimental data (n_phi, n_t1, n_t2)."""
    n_angles = c2_exp.shape[0]

    logger.info("Generating individual C2 heatmaps for %d phi angles...", n_angles)

    for angle_idx in range(n_angles):
        phi_deg = (
            phi_angles_list[angle_idx] if len(phi_angles_list) > angle_idx else 0.0
        )
        angle_data = c2_exp[angle_idx]

        fig, ax = plt.subplots(figsize=(8, 7))

        data_min = float(np.nanmin(angle_data))
        data_max = float(np.nanmax(angle_data))
        vmin = max(data_min, 1.0)
        vmax = min(data_max, 1.5)

        im = ax.imshow(
            angle_data.T,
            aspect="equal",
            cmap="jet",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"Experimental C\u2082(t\u2081, t\u2082) at \u03c6={phi_deg:.1f}\u00b0",
            fontsize=13,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax, label="C\u2082", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        # Calculate and display key statistics (use nan-safe variants)
        mean_val = float(np.nanmean(angle_data))
        max_val = float(np.nanmax(angle_data))
        min_val = float(np.nanmin(angle_data))

        stats_text = f"Mean: {mean_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=_STATS_BBOX,
        )

        plt.tight_layout()

        filename = f"experimental_data_phi_{phi_deg:.1f}.png"
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug("  Saved: %s", filename)

    logger.info("Generated %d individual C2 heatmaps", n_angles)

    # Plot diagonal (t1=t2) for all phi angles
    fig, ax = plt.subplots(figsize=(10, 6))

    if t1 is not None:
        time_diagonal = t1
    else:
        time_diagonal = np.arange(c2_exp.shape[-1])

    for idx in range(min(10, c2_exp.shape[0])):
        min_dim = min(c2_exp[idx].shape)
        diagonal = np.diag(c2_exp[idx][:min_dim, :min_dim])
        phi_deg = phi_angles_list[idx] if len(phi_angles_list) > idx else idx
        ax.plot(
            time_diagonal[:min_dim], diagonal,
            label=f"\u03c6={phi_deg:.1f}\u00b0", alpha=0.7,
        )

    ax.set_xlabel("Time (s)" if t1 is not None else "Time Index")
    ax.set_ylabel("C\u2082(t, t)")
    ax.set_title("C\u2082 Diagonal (t\u2081=t\u2082) for Different \u03c6 Angles")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.savefig(
        plots_dir / "experimental_data_diagonal.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def _plot_2d_experimental_data(
    c2_exp: np.ndarray,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot 2D experimental data (single correlation matrix)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    data_min = float(np.nanmin(c2_exp))
    data_max = float(np.nanmax(c2_exp))
    vmin = max(data_min, 1.0)
    vmax = min(data_max, 1.5)

    im = ax.imshow(
        c2_exp.T,
        aspect="equal",
        cmap="jet",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="C\u2082(t\u2081,t\u2082)", shrink=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Experimental C\u2082 Data")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_1d_experimental_data(c2_exp: np.ndarray, plots_dir: Path) -> None:
    """Plot 1D experimental data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c2_exp, marker="o", linestyle="-", alpha=0.7)
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("C\u2082")
    ax.set_title("Experimental C\u2082 Data")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fit_comparison(
    result: Any,
    data: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate comparison plots between fit and experimental data.

    Parameters
    ----------
    result : Any
        Optimization result object
    data : dict[str, Any]
        Experimental data dictionary
    plots_dir : Path
        Output directory for plot files
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot experimental data
    if c2_exp.ndim == 1:
        axes[0].plot(c2_exp, marker="o", linestyle="-", alpha=0.7, label="Experimental")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("C\u2082")
    else:
        fit_vmin = max(float(np.nanmin(c2_exp)), 1.0)
        fit_vmax = min(float(np.nanmax(c2_exp)), 1.5)
        im0 = axes[0].imshow(c2_exp, aspect="auto", cmap="jet", vmin=fit_vmin, vmax=fit_vmax)
        plt.colorbar(im0, ax=axes[0], label="C\u2082")
        axes[0].set_xlabel("t\u2082 Index")
        axes[0].set_ylabel("\u03c6 Index")
    axes[0].set_title("Experimental Data")
    axes[0].grid(True, alpha=0.3)

    # Plot fit results placeholder
    axes[1].text(
        0.5,
        0.5,
        "Fit visualization\nrequires full\nplotting backend",
        ha="center",
        va="center",
        fontsize=14,
    )
    axes[1].set_title("Fit Results")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "fit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Generated basic fit comparison plot")
