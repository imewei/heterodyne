"""Visualization for MCMC/CMC results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_figure(message: str) -> Figure:
    """Return a blank figure containing only a centred text message."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=13,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


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


def plot_posterior(
    result: CMCResult,
    params: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
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
) -> Figure:
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
) -> Figure:
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


# ---------------------------------------------------------------------------
# New diagnostic plots
# ---------------------------------------------------------------------------


def plot_forest(
    samples: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    credible_interval: float = 0.94,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Forest plot with highest-density interval (HDI) bars.

    Each parameter is rendered as a horizontal bar spanning its HDI,
    with a dot at the posterior mean.  Multiple chains/shards are
    overlaid with slight vertical offsets when the sample array is 2-D
    ``(n_chains, n_samples)``.

    Args:
        samples: Dictionary mapping parameter names to arrays of shape
            ``(n_samples,)`` or ``(n_chains, n_samples)``.
        param_names: Ordered list of parameter names to include.  When
            ``None``, all keys of ``samples`` are used (sorted).
        credible_interval: Probability mass for the HDI bars.  Must be
            in ``(0, 1)``.  Default is 0.94 (94% HDI).
        save_path: Optional path to save the figure.
        figsize: Figure size ``(width, height)``.  Auto-computed when
            ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``credible_interval`` is not in ``(0, 1)``.
    """
    if not 0 < credible_interval < 1:
        raise ValueError(
            f"credible_interval must be in (0, 1), got {credible_interval}"
        )

    if param_names is None:
        param_names = sorted(samples.keys())

    n_params = len(param_names)
    if n_params == 0:
        return _empty_figure("No parameters to plot")

    alpha_tail = (1.0 - credible_interval) / 2.0

    if figsize is None:
        figsize = (9, max(3, 0.6 * n_params + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n_params, dtype=float)

    for yi, name in zip(y_positions, param_names, strict=True):
        if name not in samples:
            ax.text(0, yi, f"{name}: missing", va="center", fontsize=8, color="gray")
            continue

        arr = np.asarray(samples[name])

        # Support (n_chains, n_samples) by flattening
        if arr.ndim == 2:
            chains = [arr[c].ravel() for c in range(arr.shape[0])]
        else:
            chains = [arr.ravel()]

        colors = plt.colormaps["tab10"](np.linspace(0, 0.9, len(chains)))
        offsets = np.linspace(-0.15 * (len(chains) - 1), 0.15 * (len(chains) - 1), len(chains))

        for chain_idx, (chain_samples, color, offset) in enumerate(
            zip(chains, colors, offsets, strict=True)
        ):
            lo = float(np.percentile(chain_samples, 100 * alpha_tail))
            hi = float(np.percentile(chain_samples, 100 * (1 - alpha_tail)))
            mean = float(np.mean(chain_samples))

            ax.plot([lo, hi], [yi + offset, yi + offset], color=color, lw=3, alpha=0.7)
            ax.plot(
                mean, yi + offset, "o", color=color, ms=6, zorder=5,
                label=f"Chain {chain_idx}" if yi == 0 else "",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xlabel("Parameter value")
    ax.set_title(
        f"Forest plot — {int(100 * credible_interval)}% HDI",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    if len(chains) > 1:
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_energy(
    samples: dict[str, np.ndarray],
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> Figure:
    """Energy transition vs marginal energy diagnostic plot (NUTS).

    Compares the distribution of the Hamiltonian energy at each
    transition (``energy``) with the marginal energy distribution
    (``energy_diff = energy[1:] - energy[:-1]``).  Good mixing
    produces overlapping distributions; a separated pair indicates
    poor exploration of the posterior.

    Args:
        samples: Sample dictionary.  Must contain an ``"energy"`` key
            with a 1-D array of Hamiltonian energies recorded by NUTS.
        save_path: Optional save path.
        figsize: Figure size ``(width, height)``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.
    """
    if "energy" not in samples:
        return _empty_figure("'energy' key not found in samples.\nRun NUTS with energy tracking enabled.")

    energy = np.asarray(samples["energy"]).ravel()
    energy_diff = np.diff(energy)

    fig, ax = plt.subplots(figsize=figsize)

    # Normalise both distributions to the same range for visual comparison
    bins = min(60, max(10, len(energy) // 10))
    ax.hist(
        energy,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Marginal energy H(θ, r)",
    )
    ax.hist(
        energy_diff,
        bins=bins,
        density=True,
        alpha=0.6,
        color="tomato",
        label="Energy transition ΔH",
    )

    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title("Energy diagnostic (NUTS)\nOverlap indicates good mixing", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate with BFMI approximation
    bfmi = float(np.var(energy_diff) / np.var(energy)) if np.var(energy) > 0 else float("nan")
    bfmi_color = "green" if bfmi >= 0.3 else "red"
    ax.text(
        0.97,
        0.97,
        f"BFMI ≈ {bfmi:.3f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color=bfmi_color,
        fontweight="bold",
    )

    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_autocorrelation(
    samples: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    max_lag: int = 50,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Per-parameter autocorrelation function plot.

    Displays the sample autocorrelation function (ACF) up to
    ``max_lag`` for each requested parameter.  Rapid decay to zero
    indicates efficient mixing; slow decay indicates high
    autocorrelation and low effective sample size.

    Args:
        samples: Dictionary mapping parameter names to sample arrays of
            shape ``(n_samples,)`` or ``(n_chains, n_samples)``.
        param_names: Parameters to include.  Defaults to all keys.
        max_lag: Maximum lag to compute.  Clamped to ``n_samples - 1``.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.
    """
    if param_names is None:
        param_names = sorted(samples.keys())

    # Drop non-numeric / missing params
    valid = [p for p in param_names if p in samples]
    if not valid:
        return _empty_figure("No valid parameters for autocorrelation plot")

    n_params = len(valid)
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, name in enumerate(valid):
        ax: Axes = axes_flat[ax_idx]
        arr = np.asarray(samples[name])
        if arr.ndim == 2:
            arr = arr[0]  # Use first chain
        arr = arr.ravel()

        n = len(arr)
        effective_max_lag = min(max_lag, n - 1)
        arr_centered = arr - arr.mean()
        var = float(np.var(arr_centered))

        if var == 0:
            ax.axhline(0, color="gray")
            ax.set_title(f"{name}\n(zero variance)")
            continue

        lags = np.arange(effective_max_lag + 1)
        acf = np.array([
            float(np.mean(arr_centered[: n - lag] * arr_centered[lag:])) / var
            for lag in lags
        ])

        ax.bar(lags, acf, color="steelblue", alpha=0.7, width=0.8)
        # 95% significance bands: ±1.96 / sqrt(n)
        band = 1.96 / np.sqrt(n)
        ax.axhline(band, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.axhline(-band, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlim(-0.5, effective_max_lag + 0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Sample Autocorrelation Functions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_rank_histogram(
    samples: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Rank histogram (rank plot) for assessing between-chain mixing.

    For each parameter, the combined samples from all chains are ranked.
    Each chain's samples are assigned their global ranks, and the rank
    distribution for each chain is plotted as a histogram.  Uniform
    rank histograms indicate well-mixed chains; U-shaped or spike-tailed
    histograms indicate convergence problems.

    Requires sample arrays of shape ``(n_chains, n_samples)``; 1-D
    arrays are treated as single-chain and produce a trivially uniform
    plot (logged as a warning).

    Args:
        samples: Dictionary mapping parameter names to arrays of shape
            ``(n_chains, n_samples)`` or ``(n_samples,)``.
        param_names: Parameters to include.  Defaults to all keys.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.
    """
    if param_names is None:
        param_names = sorted(samples.keys())

    valid = [p for p in param_names if p in samples]
    if not valid:
        return _empty_figure("No valid parameters for rank histogram")

    n_params = len(valid)
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, name in enumerate(valid):
        ax: Axes = axes_flat[ax_idx]
        arr = np.asarray(samples[name])

        if arr.ndim == 1:
            logger.warning(
                "plot_rank_histogram: '%s' is 1-D; rank plot requires "
                "multiple chains (n_chains, n_samples)",
                name,
            )
            ax.text(
                0.5,
                0.5,
                f"{name}\nSingle chain — no rank plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            continue

        n_chains, n_samples = arr.shape
        # Compute global ranks across all chains
        all_samples = arr.ravel()
        # scipy.stats.rankdata equivalent using argsort
        order = np.argsort(np.argsort(all_samples))
        ranks_per_chain = order.reshape(n_chains, n_samples)

        n_bins = min(n_samples // 2, 20)
        colors = plt.colormaps["tab10"](np.linspace(0, 0.9, n_chains))

        for chain_idx in range(n_chains):
            ax.hist(
                ranks_per_chain[chain_idx],
                bins=n_bins,
                alpha=max(0.3, 0.8 / n_chains),
                color=colors[chain_idx],
                density=True,
                label=f"Chain {chain_idx}",
            )

        # Expected uniform level
        expected = n_chains / (n_chains * n_samples)
        ax.axhline(
            expected,
            color="black",
            linestyle="--",
            lw=1.5,
            label="Uniform",
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}", fontweight="bold")
        if n_chains <= 5:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Rank Histograms (chain mixing diagnostic)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_posterior_predictive(
    c2_observed: np.ndarray,
    c2_predicted: np.ndarray,
    times: np.ndarray,
    phi_angle: float | None = None,
    n_samples_overlay: int = 50,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] = (14, 5),
    dpi: int = 150,
) -> Figure:
    """Posterior predictive check — model vs observed data overlay.

    Displays three panels:

    1. Observed ``c2`` two-time matrix.
    2. Posterior predictive mean (average over ``c2_predicted`` samples).
    3. Residual ``observed - mean_predicted`` with symmetric colour scale.

    Args:
        c2_observed: Observed correlation matrix, shape ``(n_t, n_t)``.
        c2_predicted: Posterior predictive draws, shape
            ``(n_posterior, n_t, n_t)`` or ``(n_t, n_t)`` for a single
            prediction.
        times: 1-D time axis of length ``n_t``.
        phi_angle: Optional azimuthal angle in degrees for the title.
        n_samples_overlay: Number of random diagonal slices to overlay on
            a fourth panel (0 disables the panel).
        save_path: Optional save path.
        figsize: Figure size.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If array shapes are inconsistent.
    """
    c2_obs = np.asarray(c2_observed)
    c2_pred = np.asarray(c2_predicted)

    if c2_pred.ndim == 2:
        c2_pred = c2_pred[np.newaxis]

    if c2_obs.shape != c2_pred.shape[1:]:
        raise ValueError(
            f"c2_observed shape {c2_obs.shape} does not match "
            f"c2_predicted shape {c2_pred.shape[1:]}"
        )

    c2_mean = np.mean(c2_pred, axis=0)
    residual = c2_obs - c2_mean

    add_diagonal = n_samples_overlay > 0 and c2_pred.shape[0] > 1
    ncols = 4 if add_diagonal else 3
    fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols / 3, figsize[1]))

    t_extent = [times[0], times[-1], times[-1], times[0]]
    cmap_data = "viridis"

    vmin = float(np.nanpercentile(c2_obs, 1))
    vmax = float(np.nanpercentile(c2_obs, 99))

    im0 = axes[0].imshow(c2_obs, extent=t_extent, aspect="auto", cmap=cmap_data, vmin=vmin, vmax=vmax)
    axes[0].set_title("Observed c₂")
    axes[0].set_xlabel("t₂")
    axes[0].set_ylabel("t₁")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(c2_mean, extent=t_extent, aspect="auto", cmap=cmap_data, vmin=vmin, vmax=vmax)
    axes[1].set_title("Posterior predictive mean")
    axes[1].set_xlabel("t₂")
    axes[1].set_ylabel("t₁")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    vmax_res = float(np.nanpercentile(np.abs(residual), 99))
    im2 = axes[2].imshow(
        residual,
        extent=t_extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_res,
        vmax=vmax_res,
    )
    axes[2].set_title("Residual (obs - mean)")
    axes[2].set_xlabel("t₂")
    axes[2].set_ylabel("t₁")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    if add_diagonal:
        ax_diag: Axes = axes[3]
        rng = np.random.default_rng(0)
        n_draw = min(n_samples_overlay, c2_pred.shape[0])
        chosen = rng.choice(c2_pred.shape[0], size=n_draw, replace=False)
        for draw_idx in chosen:
            diag = np.diag(c2_pred[draw_idx])
            ax_diag.plot(times, diag, color="steelblue", alpha=0.15, lw=0.8)
        ax_diag.plot(times, np.diag(c2_obs), color="black", lw=2, label="Observed")
        ax_diag.plot(times, np.diag(c2_mean), color="red", lw=2, linestyle="--", label="Pred. mean")
        ax_diag.set_xlabel("Time")
        ax_diag.set_ylabel("c₂(t, t)")
        ax_diag.set_title(f"Diagonal slices ({n_draw} draws)")
        ax_diag.legend(fontsize=8)
        ax_diag.grid(True, alpha=0.3)

    title = "Posterior Predictive Check"
    if phi_angle is not None:
        title += f"  [φ = {phi_angle:.1f}°]"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_shard_comparison(
    shard_results: list[dict[str, np.ndarray]],
    param_names: list[str] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Cross-shard posterior comparison for CMC diagnostics.

    Overlays the marginal posterior histograms of each shard for every
    requested parameter.  When shards produce similar posteriors, the
    histograms overlap; when they diverge, the plot reveals
    multi-modality or data heterogeneity.

    Args:
        shard_results: List of sample dictionaries, one per shard.
            Each dictionary maps parameter names to 1-D sample arrays.
        param_names: Parameters to compare.  Defaults to keys found in
            the first shard.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``shard_results`` is empty.
    """
    if not shard_results:
        raise ValueError("shard_results must be non-empty")

    if param_names is None:
        param_names = sorted(shard_results[0].keys())

    n_params = len(param_names)
    if n_params == 0:
        return _empty_figure("No parameters to compare across shards")

    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    n_shards = len(shard_results)
    colors = plt.colormaps["tab10"](np.linspace(0, 0.9, n_shards))

    for ax_idx, name in enumerate(param_names):
        ax: Axes = axes_flat[ax_idx]

        for shard_idx, shard in enumerate(shard_results):
            if name not in shard:
                continue
            arr = np.asarray(shard[name]).ravel()
            ax.hist(
                arr,
                bins=30,
                density=True,
                alpha=0.4,
                color=colors[shard_idx],
                label=f"Shard {shard_idx}" if n_shards <= 10 else "",
            )

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"{name}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0 and n_shards <= 10:
            ax.legend(fontsize=7)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Cross-shard posterior comparison ({n_shards} shards)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_divergence_scatter(
    samples: dict[str, np.ndarray],
    divergent_mask: np.ndarray,
    param_pairs: list[tuple[str, str]] | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Scatter plot highlighting divergent transitions.

    Plots pairs of parameters as scatter plots, colouring divergent
    transitions in red and non-divergent transitions in grey.
    Divergent transitions that cluster in parameter space indicate
    problematic posterior geometry (e.g., funnel-shaped posteriors).

    Args:
        samples: Dictionary mapping parameter names to 1-D sample
            arrays of the same length.
        divergent_mask: Boolean array of length ``n_samples`` where
            ``True`` marks a divergent transition.
        param_pairs: List of ``(param_x, param_y)`` tuples to plot.
            Defaults to all adjacent pairs of sorted parameter names.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``divergent_mask`` length does not match sample
            length, or if a requested parameter is missing from
            ``samples``.
    """
    div_mask = np.asarray(divergent_mask, dtype=bool).ravel()

    # Build default param pairs from sorted keys
    all_params = sorted(samples.keys())
    if param_pairs is None:
        param_pairs = [
            (all_params[i], all_params[i + 1])
            for i in range(min(len(all_params) - 1, 5))
        ]

    if not param_pairs:
        return _empty_figure("No parameter pairs for divergence scatter")

    ncols = min(3, len(param_pairs))
    nrows = (len(param_pairs) + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, (px, py) in enumerate(param_pairs):
        ax: Axes = axes_flat[ax_idx]

        if px not in samples:
            ax.text(0.5, 0.5, f"Missing: {px}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        if py not in samples:
            ax.text(0.5, 0.5, f"Missing: {py}", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        x = np.asarray(samples[px]).ravel()
        y = np.asarray(samples[py]).ravel()

        if len(div_mask) != len(x):
            raise ValueError(
                f"divergent_mask length {len(div_mask)} does not match "
                f"sample length {len(x)}"
            )

        non_div = ~div_mask
        ax.scatter(x[non_div], y[non_div], s=2, alpha=0.3, color="lightgrey", label="Non-divergent")
        n_div = int(div_mask.sum())
        if n_div > 0:
            ax.scatter(
                x[div_mask], y[div_mask],
                s=20, alpha=0.8, color="red", zorder=5,
                label=f"Divergent ({n_div})",
            )

        ax.set_xlabel(px, fontsize=9)
        ax.set_ylabel(py, fontsize=9)
        ax.set_title(f"{px} vs {py}", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(len(param_pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    n_div_total = int(div_mask.sum())
    n_total = len(div_mask)
    fig.suptitle(
        f"Divergent transitions  [{n_div_total}/{n_total} = {100*n_div_total/max(n_total,1):.1f}%]",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig


def plot_rhat_summary(
    rhat_dict: dict[str, float],
    threshold: float = 1.01,
    save_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """R-hat bar chart with convergence threshold line.

    Renders one bar per parameter coloured green (converged,
    R-hat < ``threshold``) or red (not converged).  A dashed horizontal
    line marks ``threshold``.

    Args:
        rhat_dict: Dictionary mapping parameter names to their R-hat
            scalar values.
        threshold: Convergence threshold.  The standard criterion is
            R-hat < 1.01 (strict) or < 1.1 (relaxed).  Default is 1.01.
        save_path: Optional save path.
        figsize: Figure size.  Auto-computed when ``None``.
        dpi: Resolution for saved figures.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If ``rhat_dict`` is empty.
    """
    if not rhat_dict:
        raise ValueError("rhat_dict must be non-empty")

    names = list(rhat_dict.keys())
    values = [float(rhat_dict[n]) for n in names]
    n = len(names)

    if figsize is None:
        figsize = (max(6, 0.6 * n + 2), 5)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green" if v < threshold else "tomato" for v in values]
    x = np.arange(n)
    ax.bar(x, values, color=colors, alpha=0.8, edgecolor="white")

    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        lw=2,
        label=f"Threshold = {threshold}",
    )

    # Annotate bars with numeric values
    for xi, val in zip(x, values, strict=True):
        ax.text(
            xi,
            val + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R-hat")
    ax.set_title("R-hat convergence summary", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    n_converged = sum(1 for v in values if v < threshold)
    ax.text(
        0.02,
        0.97,
        f"Converged: {n_converged}/{n}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        color="green" if n_converged == n else "red",
    )

    plt.tight_layout()
    _save_fig(fig, save_path, dpi=dpi)
    return fig
