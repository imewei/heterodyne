"""Diagnostic overlay plots for XPCS analysis.

Provides visualisations that aid interactive assessment of correlation data,
residuals, fitting weights, and parameter sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = get_logger(__name__)


def plot_diagonal_overlay(
    c2: np.ndarray,
    corrected_c2: np.ndarray,
    times: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Show before/after diagonal correction on c2.

    Overlays the diagonal of the original and corrected correlation
    matrices so the user can verify that diagonal artefacts have been
    removed.

    Args:
        c2: Original correlation matrix, shape (N, N).
        corrected_c2: Corrected correlation matrix, shape (N, N).
        times: 1-D time array of length N.
        ax: Optional existing Axes; one is created if ``None``.

    Returns:
        The matplotlib Axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    diag_original = np.diag(c2)
    diag_corrected = np.diag(corrected_c2)

    ax.plot(times, diag_original, "o-", markersize=3, alpha=0.7, label="Original")
    ax.plot(times, diag_corrected, "s-", markersize=3, alpha=0.7, label="Corrected")

    ax.set_xlabel("Time")
    ax.set_ylabel("c₂(t, t)")
    ax.set_title("Diagonal Correction Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_residual_map(
    residuals: np.ndarray,
    times: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """2-D heatmap of fit residuals.

    Args:
        residuals: Residual matrix, shape (N, N).
        times: 1-D time array of length N.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes containing the heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    vmax = float(np.nanpercentile(np.abs(residuals), 99))
    extent: tuple[float, float, float, float] = (
        float(times[0]),
        float(times[-1]),
        float(times[-1]),
        float(times[0]),
    )

    im = ax.imshow(
        residuals,
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )
    ax.set_xlabel("t₂")
    ax.set_ylabel("t₁")
    ax.set_title("Residual Map")
    plt.colorbar(im, ax=ax, label="Residual")

    return ax


def plot_weight_map(
    weights: np.ndarray,
    times: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Visualise the fitting weight matrix.

    Args:
        weights: Weight matrix, shape (N, N).
        times: 1-D time array of length N.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes containing the heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    extent2: tuple[float, float, float, float] = (
        float(times[0]),
        float(times[-1]),
        float(times[-1]),
        float(times[0]),
    )

    im = ax.imshow(
        weights,
        extent=extent2,
        aspect="auto",
        cmap="viridis",
        origin="upper",
    )
    ax.set_xlabel("t₂")
    ax.set_ylabel("t₁")
    ax.set_title("Weight Map")
    plt.colorbar(im, ax=ax, label="Weight")

    return ax


def plot_convergence_trace(
    losses: np.ndarray,
    ax: Axes | None = None,
    log_scale: bool = True,
) -> Axes:
    """Plot optimization convergence trace.

    Args:
        losses: Loss values per iteration, shape (n_iter,).
        ax: Optional existing Axes.
        log_scale: Use log scale for y-axis.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    iterations = np.arange(len(losses))
    ax.plot(iterations, losses, "-", linewidth=1.5, color="steelblue")

    if log_scale and np.all(losses > 0):
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence Trace")
    ax.grid(True, alpha=0.3)

    return ax


def plot_trace_posterior(
    samples: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Trace + posterior density plots for MCMC samples.

    Creates a two-column layout: left column shows trace plots,
    right column shows marginal posterior histograms.

    Args:
        samples: Dict mapping parameter names to sample arrays,
            each of shape (n_samples,) or (n_chains, n_samples).
        param_names: Subset of parameter names to plot (default: all).
        figsize: Optional figure size.

    Returns:
        The matplotlib Figure.
    """
    if param_names is None:
        param_names = list(samples.keys())

    n_params = len(param_names)
    if figsize is None:
        figsize = (12, 2.5 * n_params)

    fig, axes = plt.subplots(n_params, 2, figsize=figsize, squeeze=False)

    for i, name in enumerate(param_names):
        vals = samples[name]
        ax_trace = axes[i, 0]
        ax_hist = axes[i, 1]

        # Trace plot
        if vals.ndim == 2:
            # Multiple chains
            for chain_idx in range(vals.shape[0]):
                ax_trace.plot(vals[chain_idx], alpha=0.7, linewidth=0.5)
        else:
            ax_trace.plot(vals, alpha=0.7, linewidth=0.5, color="steelblue")

        ax_trace.set_ylabel(name)
        ax_trace.set_title(f"Trace: {name}" if i == 0 else "")
        ax_trace.grid(True, alpha=0.2)

        # Posterior histogram
        flat_vals = vals.ravel()
        ax_hist.hist(flat_vals, bins=50, density=True, alpha=0.7, color="steelblue")
        ax_hist.axvline(
            np.median(flat_vals),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"median={np.median(flat_vals):.4g}",
        )
        ax_hist.legend(fontsize=8)
        ax_hist.set_title(f"Posterior: {name}" if i == 0 else "")
        ax_hist.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Sample index")
    axes[-1, 1].set_xlabel("Value")

    fig.tight_layout()
    return fig


def plot_pair_correlation(
    samples: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Parameter correlation matrix heatmap.

    Args:
        samples: Dict mapping parameter names to 1-D sample arrays.
        param_names: Subset of names to include (default: all).
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes.
    """
    if param_names is None:
        param_names = list(samples.keys())

    n = len(param_names)
    corr_matrix = np.zeros((n, n))

    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names):
            vals_i = samples[name_i].ravel()
            vals_j = samples[name_j].ravel()
            min_len = min(len(vals_i), len(vals_j))
            if min_len > 1:
                corr_matrix[i, j] = np.corrcoef(vals_i[:min_len], vals_j[:min_len])[
                    0, 1
                ]
            else:
                corr_matrix[i, j] = 0.0

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.8 * n), max(5, 0.7 * n)))

    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(param_names, fontsize=8)
    ax.set_title("Parameter Correlation")
    plt.colorbar(im, ax=ax, label="Correlation")

    return ax


def plot_residual_histogram(
    residuals: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Histogram of residuals with Gaussian overlay.

    Args:
        residuals: Residual array (any shape, will be flattened).
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    flat = residuals.ravel()
    flat = flat[np.isfinite(flat)]

    ax.hist(
        flat, bins=80, density=True, alpha=0.7, color="steelblue", label="Residuals"
    )

    # Gaussian overlay
    mu, sigma = np.mean(flat), np.std(flat)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x, gaussian, "r-", linewidth=2, label=f"N({mu:.3g}, {sigma:.3g}²)")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_parameter_sensitivity(
    sensitivity_dict: dict[str, float],
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of per-parameter sensitivity values.

    Args:
        sensitivity_dict: Mapping of parameter name to sensitivity value.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes containing the bar chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    names = list(sensitivity_dict.keys())
    values = [sensitivity_dict[n] for n in names]

    x = np.arange(len(names))
    ax.bar(x, values, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Parameter Sensitivity")
    ax.grid(True, axis="y", alpha=0.3)

    if hasattr(ax.figure, "tight_layout"):
        ax.figure.tight_layout()

    return ax


# ---------------------------------------------------------------------------
# Diagonal overlay statistics
# ---------------------------------------------------------------------------


@dataclass
class DiagonalOverlayResult:
    """Statistics from comparing diagonals of experimental vs fitted C2 surfaces.

    Attributes:
        phi_index: Angle index used for extraction.
        raw_diagonal: Diagonal of the experimental C2.
        solver_diagonal: Diagonal of the solver-fitted C2.
        posthoc_diagonal: Diagonal of the post-hoc corrected C2.
        raw_variance: Variance of the raw diagonal.
        solver_variance: Variance of the solver diagonal.
        posthoc_variance: Variance of the post-hoc diagonal.
        solver_rmse: RMSE between raw and solver diagonals.
        posthoc_rmse: RMSE between raw and post-hoc diagonals.
    """

    phi_index: int
    raw_diagonal: np.ndarray
    solver_diagonal: np.ndarray
    posthoc_diagonal: np.ndarray
    raw_variance: float
    solver_variance: float
    posthoc_variance: float
    solver_rmse: float
    posthoc_rmse: float


def compute_diagonal_overlay_stats(
    c2_exp: np.ndarray,
    c2_solver: np.ndarray | None,
    c2_posthoc: np.ndarray,
    *,
    phi_index: int = 0,
) -> DiagonalOverlayResult:
    """Compute diagonal overlay statistics for visual validation.

    Extracts the diagonal from each C2 matrix at the given angle index
    and computes variance and RMSE metrics.

    Args:
        c2_exp: Experimental C2, shape ``(n_phi, N, N)``.
        c2_solver: Solver-fitted C2, shape ``(n_phi, N, N)``.
        c2_posthoc: Post-hoc corrected C2, shape ``(n_phi, N, N)``.
        phi_index: Angle index to extract.

    Returns:
        :class:`DiagonalOverlayResult` with diagonal arrays and metrics.

    Raises:
        ValueError: If *c2_solver* is ``None``.
    """
    if c2_solver is None:
        msg = "c2_solver must not be None"
        raise ValueError(msg)

    raw_diag = np.diag(c2_exp[phi_index])
    solver_diag = np.diag(c2_solver[phi_index])
    posthoc_diag = np.diag(c2_posthoc[phi_index])

    return DiagonalOverlayResult(
        phi_index=phi_index,
        raw_diagonal=raw_diag,
        solver_diagonal=solver_diag,
        posthoc_diagonal=posthoc_diag,
        raw_variance=float(np.nanvar(raw_diag)),
        solver_variance=float(np.nanvar(solver_diag)),
        posthoc_variance=float(np.nanvar(posthoc_diag)),
        solver_rmse=float(np.sqrt(np.nanmean((raw_diag - solver_diag) ** 2))),
        posthoc_rmse=float(np.sqrt(np.nanmean((raw_diag - posthoc_diag) ** 2))),
    )
