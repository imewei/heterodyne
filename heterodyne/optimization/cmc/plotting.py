"""ArviZ-based MCMC diagnostic plots for CMC results.

All functions accept ArviZ ``InferenceData`` objects and return
matplotlib figures.  ``arviz`` is imported with a try/except guard
so that the rest of the package does not hard-depend on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = get_logger(__name__)

try:
    import arviz as az

    _HAS_ARVIZ = True
except ImportError:  # pragma: no cover
    _HAS_ARVIZ = False


def _require_arviz() -> None:
    """Raise ImportError if ArviZ is not installed."""
    if not _HAS_ARVIZ:
        raise ImportError(
            "arviz is required for CMC diagnostic plots.  "
            "Install it with: uv add arviz"
        )


# ------------------------------------------------------------------
# Public plotting functions
# ------------------------------------------------------------------


def plot_trace_summary(
    idata: object,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """ArviZ trace plot with marginal posteriors.

    Args:
        idata: ArviZ InferenceData object.
        var_names: Subset of variable names to plot (``None`` for all).
        figsize: Optional figure size override.

    Returns:
        Matplotlib Figure.
    """
    _require_arviz()

    axes = az.plot_trace(
        idata,
        var_names=var_names,
        figsize=figsize,
        compact=True,
        combined=False,
    )
    fig = axes.ravel()[0].figure
    fig.suptitle("Trace + Posterior Summary", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_pair_plot(
    idata: object,
    var_names: list[str] | None = None,
    divergences: bool = True,
) -> Figure:
    """ArviZ pair plot with optional divergence markers.

    Args:
        idata: ArviZ InferenceData object.
        var_names: Subset of variable names.
        divergences: Whether to overlay divergence markers.

    Returns:
        Matplotlib Figure.
    """
    _require_arviz()

    axes = az.plot_pair(
        idata,
        var_names=var_names,
        divergences=divergences,
        kind="scatter",
        marginals=True,
    )
    # az.plot_pair returns a 2-D ndarray of Axes
    fig = axes.ravel()[0].figure
    fig.suptitle("Pair Plot", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_posterior_predictive(
    idata: object,
    c2_data: np.ndarray,
    times: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Overlay posterior-predictive draws on experimental data.

    If *idata* contains a ``posterior_predictive`` group with a variable
    named ``"c2_pred"``, the 5th/95th percentile envelope is drawn.
    Otherwise a message is displayed.

    Args:
        idata: ArviZ InferenceData object.
        c2_data: 2-D experimental correlation matrix, shape (N, N).
        times: 1-D time array.
        ax: Optional existing Axes.

    Returns:
        The matplotlib Axes.
    """
    _require_arviz()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Plot observed diagonal
    diag_obs = np.diag(c2_data)
    ax.plot(times, diag_obs, "ko", markersize=3, alpha=0.7, label="Observed", zorder=3)

    # Attempt to extract posterior predictive
    pp = getattr(idata, "posterior_predictive", None)
    if pp is not None and "c2_pred" in pp:
        c2_pred = pp["c2_pred"].values  # (chain, draw, N, N) or (draw, N, N)
        # Flatten chains
        if c2_pred.ndim == 4:
            n_chain, n_draw, n_t, _ = c2_pred.shape
            c2_pred = c2_pred.reshape(n_chain * n_draw, n_t, n_t)

        # Extract diagonals for each draw
        diag_draws = np.array([np.diag(c2_pred[d]) for d in range(c2_pred.shape[0])])
        lo = np.percentile(diag_draws, 5, axis=0)
        hi = np.percentile(diag_draws, 95, axis=0)
        median = np.percentile(diag_draws, 50, axis=0)

        ax.fill_between(times, lo, hi, alpha=0.3, color="steelblue", label="90% CI")
        ax.plot(times, median, "-", color="steelblue", lw=1.5, label="Median")
    else:
        ax.text(
            0.5, 0.95,
            "No posterior_predictive['c2_pred'] in InferenceData",
            ha="center", va="top", transform=ax.transAxes, fontsize=9, color="gray",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("c₂(t, t)")
    ax.set_title("Posterior Predictive Check (diagonal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_diagnostics_summary(idata: object) -> Figure:
    """Combined R-hat, ESS, and BFMI diagnostic panels.

    Creates a three-panel figure:
    1. R-hat per parameter (bar chart).
    2. Bulk ESS per parameter (bar chart).
    3. BFMI per chain (bar chart).

    Args:
        idata: ArviZ InferenceData object.

    Returns:
        Matplotlib Figure.
    """
    _require_arviz()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # -- R-hat panel --
    ax_rhat = axes[0]
    rhat_data = az.rhat(idata)
    if hasattr(rhat_data, "to_dataframe"):
        rhat_df = rhat_data.to_dataframe().iloc[0]
        names = list(rhat_df.index)
        rhat_vals = rhat_df.values.astype(float)
    else:
        # Fallback: treat as dict-like
        names = list(rhat_data.data_vars)
        rhat_vals = np.array([float(rhat_data[n].values) for n in names])

    x = np.arange(len(names))
    colors_rhat = ["#F44336" if v > 1.1 else "#4CAF50" for v in rhat_vals]
    ax_rhat.bar(x, rhat_vals, color=colors_rhat, alpha=0.8)
    ax_rhat.axhline(1.1, color="red", linestyle="--", lw=1, label="Threshold (1.1)")
    ax_rhat.set_xticks(x)
    ax_rhat.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax_rhat.set_ylabel("R-hat")
    ax_rhat.set_title("R-hat")
    ax_rhat.legend(fontsize=8)
    ax_rhat.grid(True, axis="y", alpha=0.3)

    # -- ESS panel --
    ax_ess = axes[1]
    ess_data = az.ess(idata)
    if hasattr(ess_data, "to_dataframe"):
        ess_df = ess_data.to_dataframe().iloc[0]
        ess_vals = ess_df.values.astype(float)
    else:
        ess_vals = np.array([float(ess_data[n].values) for n in names])

    colors_ess = ["#F44336" if v < 100 else "#4CAF50" for v in ess_vals]
    ax_ess.bar(x, ess_vals, color=colors_ess, alpha=0.8)
    ax_ess.axhline(100, color="red", linestyle="--", lw=1, label="Minimum (100)")
    ax_ess.set_xticks(x)
    ax_ess.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax_ess.set_ylabel("ESS (bulk)")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.legend(fontsize=8)
    ax_ess.grid(True, axis="y", alpha=0.3)

    # -- BFMI panel --
    ax_bfmi = axes[2]
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "energy" in sample_stats:
        bfmi_vals = az.bfmi(idata)
        bfmi_x = np.arange(len(bfmi_vals))
        colors_bfmi = ["#F44336" if v < 0.3 else "#4CAF50" for v in bfmi_vals]
        ax_bfmi.bar(bfmi_x, bfmi_vals, color=colors_bfmi, alpha=0.8)
        ax_bfmi.axhline(0.3, color="red", linestyle="--", lw=1, label="Minimum (0.3)")
        ax_bfmi.set_xticks(bfmi_x)
        ax_bfmi.set_xticklabels([f"Chain {i}" for i in bfmi_x], fontsize=8)
        ax_bfmi.set_ylabel("BFMI")
        ax_bfmi.set_title("Bayesian Fraction of Missing Information")
        ax_bfmi.legend(fontsize=8)
        ax_bfmi.grid(True, axis="y", alpha=0.3)
    else:
        ax_bfmi.text(
            0.5, 0.5, "No energy data available",
            ha="center", va="center", transform=ax_bfmi.transAxes, fontsize=11,
        )
        ax_bfmi.set_title("BFMI")

    fig.suptitle("MCMC Diagnostics Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig
