"""Visualization for NLSQ fitting results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_color_limits(
    matrix: np.ndarray,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> tuple[float, float]:
    """Percentile-based colour limits with fallback for empty/flat data."""
    if matrix.size == 0 or not np.any(np.isfinite(matrix)):
        return 1.0, 1.5
    vmin = float(np.nanpercentile(matrix, percentile_min))
    vmax = float(np.nanpercentile(matrix, percentile_max))
    if not np.isfinite(vmin):
        vmin = 1.0
    if not np.isfinite(vmax):
        vmax = 1.5
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_fig(
    fig: Figure,
    save_path: Path | str | None,
    dpi: int = 150,
) -> None:
    """Save and close a figure when a path is provided."""
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Existing plots (kept intact)
# ---------------------------------------------------------------------------


def plot_nlsq_fit(
    c2_data: np.ndarray,
    result: NLSQResult,
    t: np.ndarray | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (15, 5),
) -> Figure:
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
        aspect="auto",
        cmap="viridis",
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
            aspect="auto",
            cmap="viridis",
        )
        axes[1].set_title("Fitted Model")
        axes[1].set_xlabel("t₂")
        axes[1].set_ylabel("t₁")
        plt.colorbar(im1, ax=axes[1], label="c₂")
    else:
        axes[1].text(0.5, 0.5, "No fitted correlation", ha="center", va="center")
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
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[2].set_title("Residuals")
        axes[2].set_xlabel("t₂")
        axes[2].set_ylabel("t₁")
        plt.colorbar(im2, ax=axes[2], label="Residual")
    else:
        axes[2].text(0.5, 0.5, "No residuals", ha="center", va="center")
        axes[2].set_title("Residuals")

    # Add fit statistics
    chi2 = result.reduced_chi_squared
    stats_text = f"χ²_red = {chi2:.3f}" if chi2 is not None else ""
    fig.suptitle(f"NLSQ Fit Results  {stats_text}", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_residual_map(
    result: NLSQResult,
    c2_data: np.ndarray,
    t: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> Figure:
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
        aspect="auto",
        cmap="RdBu_r",
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
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        axes[0, 1].plot(
            x,
            np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)),
            "r-",
            lw=2,
            label=f"Normal(μ={mu:.2e}, σ={sigma:.2e})",
        )
    axes[0, 1].legend()

    # Residual along diagonal
    diag_residuals = np.diag(residuals)
    axes[1, 0].plot(t, diag_residuals, "b-", lw=1)
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
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
    axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("Fitted Value")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Residuals vs Fitted")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_parameter_uncertainties(
    result: NLSQResult,
    save_path: Path | str | None = None,
) -> Figure:
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
    ax.errorbar(x, params, yerr=errors, fmt="o", capsize=5, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(result.parameter_names, rotation=45, ha="right")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Fitted Parameters with Uncertainties")
    ax.grid(True, alpha=0.3)

    # Use log scale if values span many orders of magnitude
    nonzero_params = params[params != 0]
    if (
        len(nonzero_params) > 0
        and np.max(np.abs(params)) / np.min(np.abs(nonzero_params)) > 100
    ):
        ax.set_yscale("symlog")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# New plots
# ---------------------------------------------------------------------------


def plot_fit_surface(
    c2_exp: np.ndarray,
    c2_fit: np.ndarray,
    times: np.ndarray,
    phi_angle: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 5),
    dpi: int = 150,
) -> Figure:
    """Side-by-side 2D heatmaps: experimental data, fit, and residuals.

    Three panels are rendered with per-panel percentile-based colour
    scaling so that structure is visible in both the experimental and
    fitted matrices even when their absolute ranges differ.

    Args:
        c2_exp: Experimental correlation matrix, shape ``(n_t, n_t)``.
        c2_fit: Fitted model correlation matrix, shape ``(n_t, n_t)``.
        times: 1-D time axis of length ``n_t`` (seconds).
        phi_angle: Optional azimuthal angle in degrees for the title.
        save_path: Optional path to save the figure.
        figsize: Figure size ``(width, height)``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``c2_exp`` and ``c2_fit`` have different shapes.
    """
    c2_exp = np.asarray(c2_exp)
    c2_fit = np.asarray(c2_fit)

    if c2_exp.shape != c2_fit.shape:
        raise ValueError(
            f"c2_exp shape {c2_exp.shape} does not match c2_fit shape {c2_fit.shape}"
        )

    residual = c2_exp - c2_fit
    t_extent = [times[0], times[-1], times[-1], times[0]]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    vmin_exp, vmax_exp = _resolve_color_limits(c2_exp)
    vmin_fit, vmax_fit = _resolve_color_limits(c2_fit)
    vmax_res = float(np.nanpercentile(np.abs(residual), 99))
    if not np.isfinite(vmax_res) or vmax_res == 0:
        vmax_res = 0.01

    im0 = axes[0].imshow(
        c2_exp.T,
        origin="lower",
        aspect="equal",
        cmap="jet",
        extent=t_extent,
        vmin=vmin_exp,
        vmax=vmax_exp,
    )
    phi_str = f"  φ={phi_angle:.1f}°" if phi_angle is not None else ""
    axes[0].set_title(f"Experimental C₂{phi_str}", fontsize=11)
    axes[0].set_xlabel("t₁ (s)")
    axes[0].set_ylabel("t₂ (s)")
    plt.colorbar(im0, ax=axes[0], label="C₂", shrink=0.85)

    im1 = axes[1].imshow(
        c2_fit.T,
        origin="lower",
        aspect="equal",
        cmap="jet",
        extent=t_extent,
        vmin=vmin_fit,
        vmax=vmax_fit,
    )
    axes[1].set_title(f"Fitted C₂{phi_str}", fontsize=11)
    axes[1].set_xlabel("t₁ (s)")
    axes[1].set_ylabel("t₂ (s)")
    plt.colorbar(im1, ax=axes[1], label="C₂", shrink=0.85)

    im2 = axes[2].imshow(
        residual.T,
        origin="lower",
        aspect="equal",
        cmap="RdBu_r",
        extent=t_extent,
        vmin=-vmax_res,
        vmax=vmax_res,
    )
    axes[2].set_title(f"Residuals (exp - fit){phi_str}", fontsize=11)
    axes[2].set_xlabel("t₁ (s)")
    axes[2].set_ylabel("t₂ (s)")
    plt.colorbar(im2, ax=axes[2], label="ΔC₂", shrink=0.85)

    rms = float(np.sqrt(np.nanmean(residual**2)))
    fig.suptitle(
        f"NLSQ Fit Surface  [RMS residual = {rms:.4f}]",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_per_angle_residuals(
    residuals: np.ndarray,
    phi_angles: np.ndarray,
    times: np.ndarray,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Per-angle 2D residual heatmap.

    Renders one heatmap panel per phi angle, showing the signed residual
    ``(experimental - fitted)`` on a symmetric diverging colour scale.

    Args:
        residuals: 3-D array of shape ``(n_angles, n_t, n_t)``.
        phi_angles: 1-D array of azimuthal angles in degrees, length
            ``n_angles``.
        times: 1-D time axis of length ``n_t`` (seconds).
        save_path: Optional path to save the figure.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``residuals.shape[0]`` differs from
            ``len(phi_angles)``.
    """
    residuals = np.asarray(residuals)
    phi_angles = np.asarray(phi_angles)

    if residuals.ndim != 3:
        raise ValueError(
            f"residuals must be 3-D (n_angles, n_t, n_t), got shape {residuals.shape}"
        )
    if residuals.shape[0] != len(phi_angles):
        raise ValueError(
            f"residuals first dimension {residuals.shape[0]} "
            f"does not match phi_angles length {len(phi_angles)}"
        )

    n_angles = len(phi_angles)
    ncols = min(4, n_angles)
    nrows = (n_angles + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    # Shared colour scale across all angles
    global_vmax = float(np.nanpercentile(np.abs(residuals), 99))
    if not np.isfinite(global_vmax) or global_vmax == 0:
        global_vmax = 0.01
    t_extent = [times[0], times[-1], times[-1], times[0]]

    for angle_idx in range(n_angles):
        ax: Axes = axes_flat[angle_idx]
        im = ax.imshow(
            residuals[angle_idx].T,
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            extent=t_extent,
            vmin=-global_vmax,
            vmax=global_vmax,
        )
        ax.set_title(f"φ = {phi_angles[angle_idx]:.1f}°", fontsize=10)
        ax.set_xlabel("t₁ (s)", fontsize=8)
        ax.set_ylabel("t₂ (s)", fontsize=8)
        plt.colorbar(im, ax=ax, label="ΔC₂", shrink=0.8)

    for idx in range(n_angles, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Per-angle residuals  (experimental − fitted)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_parameter_evolution(
    history: list[dict[str, Any]],
    param_names: list[str],
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Parameter values across multistart optimisation attempts.

    Displays the trajectory of each parameter's best-so-far value and
    the individual starting-point results as scatter dots, making it
    easy to see whether multistart has converged on a consistent
    solution.

    Args:
        history: List of result records, one per multistart attempt
            (ordered by attempt index).  Each record must contain:

            - ``"params"`` – 1-D array of parameter values, same length
              as ``param_names``.
            - ``"loss"`` – scalar loss value for this attempt.
            - ``"converged"`` – bool flag (optional, default ``True``).

        param_names: Parameter names corresponding to the ``"params"``
            arrays.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``history`` is empty or ``param_names`` is empty.
    """
    if not history:
        raise ValueError("history must be non-empty")
    if not param_names:
        raise ValueError("param_names must be non-empty")

    n_params = len(param_names)
    n_attempts = len(history)
    attempt_idx = np.arange(n_attempts)

    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    losses = np.array([float(r.get("loss", np.nan)) for r in history])
    converged_flags = np.array([bool(r.get("converged", True)) for r in history])

    for p_idx, pname in enumerate(param_names):
        ax: Axes = axes_flat[p_idx]

        param_vals = np.array(
            [
                float(np.asarray(r["params"])[p_idx]) if "params" in r else np.nan
                for r in history
            ]
        )

        # Best-so-far trajectory
        best_so_far = np.full(n_attempts, np.nan)
        best_loss = np.inf
        best_val = np.nan
        for i in range(n_attempts):
            if np.isfinite(losses[i]) and losses[i] < best_loss:
                best_loss = losses[i]
                best_val = param_vals[i]
            best_so_far[i] = best_val

        ax.plot(attempt_idx, best_so_far, "k-", lw=1.5, label="Best so far", zorder=3)

        # Individual attempts
        conv_mask = converged_flags
        div_mask = ~converged_flags
        if np.any(conv_mask):
            ax.scatter(
                attempt_idx[conv_mask],
                param_vals[conv_mask],
                s=20,
                alpha=0.6,
                color="steelblue",
                zorder=4,
                label="Converged",
            )
        if np.any(div_mask):
            ax.scatter(
                attempt_idx[div_mask],
                param_vals[div_mask],
                s=20,
                alpha=0.6,
                color="tomato",
                marker="x",
                zorder=4,
                label="Not converged",
            )

        ax.set_xlabel("Attempt index")
        ax.set_ylabel(pname)
        ax.set_title(f"{pname}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        if p_idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Parameter evolution across {n_attempts} multistart attempts",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_scaling_comparison(
    solver_scaling: np.ndarray,
    lstsq_scaling: np.ndarray,
    phi_angles: np.ndarray,
    param_labels: tuple[str, str] = ("contrast", "offset"),
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 150,
) -> Figure:
    """Contrast and offset comparison: solver vs least-squares post-hoc.

    Side-by-side bar charts compare the contrast and offset values
    produced by the nonlinear solver with those derived from the
    post-hoc linear least-squares scaling step.  Significant
    disagreement may indicate a poorly conditioned solver or
    overfitting.

    Args:
        solver_scaling: Array of shape ``(n_angles, 2)`` containing
            ``[contrast, offset]`` from the nonlinear solver for each
            phi angle.
        lstsq_scaling: Array of shape ``(n_angles, 2)`` containing
            ``[contrast, offset]`` from the least-squares scaling step.
        phi_angles: 1-D array of phi angles in degrees, length
            ``n_angles``.
        param_labels: Labels for the two scaling parameters.  Default
            is ``("contrast", "offset")``.
        save_path: Optional save path.
        figsize: Figure size.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If array shapes are inconsistent.
    """
    solver_scaling = np.asarray(solver_scaling)
    lstsq_scaling = np.asarray(lstsq_scaling)
    phi_angles = np.asarray(phi_angles)

    n_angles = len(phi_angles)
    for arr, name in (
        (solver_scaling, "solver_scaling"),
        (lstsq_scaling, "lstsq_scaling"),
    ):
        if arr.shape != (n_angles, 2):
            raise ValueError(f"{name} must have shape ({n_angles}, 2), got {arr.shape}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(n_angles)
    width = 0.35

    for col_idx, label in enumerate(param_labels):
        ax: Axes = axes[col_idx]
        solver_vals = solver_scaling[:, col_idx]
        lstsq_vals = lstsq_scaling[:, col_idx]

        ax.bar(
            x - width / 2,
            solver_vals,
            width,
            label="Solver",
            alpha=0.75,
            color="steelblue",
        )
        ax.bar(
            x + width / 2,
            lstsq_vals,
            width,
            label="Least-squares",
            alpha=0.75,
            color="orange",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{p:.1f}°" for p in phi_angles], rotation=45, ha="right", fontsize=8
        )
        ax.set_xlabel("phi angle")
        ax.set_ylabel(label)
        ax.set_title(f"{label} — solver vs least-squares", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Scaling parameter comparison per angle", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_chi_squared_landscape(
    chi2_values: np.ndarray,
    param_values: np.ndarray,
    param_name: str,
    best_value: float | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> Figure:
    """1-D chi-squared profile plot for a single parameter.

    Displays how the reduced chi-squared changes as one parameter is
    swept through a range of values (all other parameters fixed at
    their best-fit values).  The minimum of the profile locates the
    best-fit value; the width at ``Δχ² = 1`` gives the 68% confidence
    interval for that parameter.

    Args:
        chi2_values: 1-D array of reduced chi-squared values, length
            ``n_sweep``.
        param_values: 1-D array of parameter values swept, length
            ``n_sweep``.
        param_name: Human-readable parameter name for axis labels.
        best_value: Optional known best-fit value; if provided, a
            vertical dashed line is drawn at this position.
        save_path: Optional save path.
        figsize: Figure size.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``chi2_values`` and ``param_values`` have
            different lengths.
    """
    chi2_values = np.asarray(chi2_values, dtype=float)
    param_values = np.asarray(param_values, dtype=float)

    if chi2_values.shape != param_values.shape:
        raise ValueError(
            f"chi2_values length {chi2_values.shape} does not match "
            f"param_values length {param_values.shape}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(param_values, chi2_values, "o-", color="steelblue", ms=4, lw=1.5)

    # Mark the minimum
    min_idx = int(np.nanargmin(chi2_values))
    ax.scatter(
        param_values[min_idx],
        chi2_values[min_idx],
        color="red",
        zorder=5,
        s=80,
        label=f"Min χ² = {chi2_values[min_idx]:.3f} at {param_values[min_idx]:.4g}",
    )

    # Delta chi2 = 1 band
    chi2_min = float(np.nanmin(chi2_values))
    ax.axhline(
        chi2_min + 1, color="green", linestyle="--", lw=1.5, label="χ²_min + 1 (68% CI)"
    )

    if best_value is not None:
        ax.axvline(
            best_value,
            color="orange",
            linestyle=":",
            lw=2,
            label=f"Best fit = {best_value:.4g}",
        )

    ax.set_xlabel(param_name)
    ax.set_ylabel("Reduced χ²")
    ax.set_title(f"χ² landscape — {param_name}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_multistart_summary(
    results: list[dict[str, Any]],
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 150,
) -> Figure:
    """Loss distribution and convergence summary across multistart attempts.

    Three panels:

    1. **Loss histogram** — distribution of final loss values across all
       starts, with the best loss marked.
    2. **Loss rank plot** — loss values sorted in ascending order to
       reveal the shape of the basin landscape.
    3. **Convergence fraction** — pie chart of converged vs non-converged
       starts.

    Args:
        results: List of result dictionaries, one per multistart
            attempt.  Each record must contain:

            - ``"loss"`` – scalar final loss.
            - ``"converged"`` – bool convergence flag.

        save_path: Optional save path.
        figsize: Figure size.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must be non-empty")

    losses = np.array([float(r.get("loss", np.nan)) for r in results])
    converged = np.array([bool(r.get("converged", True)) for r in results])

    finite_losses = losses[np.isfinite(losses)]
    n_total = len(results)
    n_converged = int(np.sum(converged))
    n_not_converged = n_total - n_converged

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: loss histogram
    ax0: Axes = axes[0]
    if len(finite_losses) > 0:
        bins = min(30, max(5, len(finite_losses) // 2))
        ax0.hist(
            finite_losses, bins=bins, color="steelblue", alpha=0.75, edgecolor="white"
        )
        best_loss = float(np.nanmin(finite_losses))
        ax0.axvline(
            best_loss,
            color="red",
            lw=2,
            linestyle="--",
            label=f"Best = {best_loss:.4g}",
        )
        ax0.legend(fontsize=8)
    ax0.set_xlabel("Final loss")
    ax0.set_ylabel("Count")
    ax0.set_title("Loss distribution", fontweight="bold")
    ax0.grid(True, alpha=0.3)

    # Panel 2: sorted loss rank plot
    ax1: Axes = axes[1]
    if len(finite_losses) > 0:
        sorted_losses = np.sort(finite_losses)
        ax1.plot(
            np.arange(1, len(sorted_losses) + 1),
            sorted_losses,
            "o-",
            ms=4,
            color="steelblue",
        )
        ax1.set_xlabel("Rank (1 = best)")
        ax1.set_ylabel("Loss")
        ax1.set_title("Sorted loss (rank plot)", fontweight="bold")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No finite losses",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    # Panel 3: convergence pie
    ax2: Axes = axes[2]
    if n_total > 0:
        pie_vals = [n_converged, n_not_converged]
        pie_labels = [
            f"Converged\n({n_converged})",
            f"Not converged\n({n_not_converged})",
        ]
        pie_colors = ["green", "tomato"]
        non_zero = [
            (v, lab, c)
            for v, lab, c in zip(pie_vals, pie_labels, pie_colors, strict=True)
            if v > 0
        ]
        if non_zero:
            nz_vals, nz_labels, nz_colors = zip(*non_zero, strict=True)
            ax2.pie(
                nz_vals,
                labels=nz_labels,
                colors=nz_colors,
                autopct="%1.0f%%",
                startangle=90,
            )
    ax2.set_title(
        f"Convergence ({n_converged}/{n_total})",
        fontweight="bold",
    )

    fig.suptitle(
        f"Multistart summary  [{n_total} attempts]",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig
