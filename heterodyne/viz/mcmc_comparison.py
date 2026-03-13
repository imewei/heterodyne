"""NLSQ vs CMC comparison plots for heterodyne analysis.

Provides side-by-side visualization of point estimates (NLSQ) against
posterior distributions (CMC) for assessing consistency between
optimization and Bayesian inference results.
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
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def plot_nlsq_vs_cmc(
    nlsq_result: NLSQResult,
    cmc_result: CMCResult,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot NLSQ point estimates against CMC posterior distributions.

    For each shared parameter, shows the posterior distribution as a
    violin/box and overlays the NLSQ point estimate with error bars.

    Args:
        nlsq_result: NLSQ optimization result.
        cmc_result: CMC Bayesian result with posterior samples.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    # Find common parameters
    common_params = [
        name
        for name in cmc_result.parameter_names
        if name in nlsq_result.parameter_names
    ]

    if not common_params:
        logger.warning("No common parameters between NLSQ and CMC results")
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No common parameters",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    n_params = len(common_params)
    fig, axes = plt.subplots(1, n_params, figsize=(3 * n_params, 5), squeeze=False)
    axes_flat = axes[0]

    for i, name in enumerate(common_params):
        ax = axes_flat[i]

        # CMC posterior
        cmc_idx = cmc_result.parameter_names.index(name)
        cmc_mean = float(cmc_result.posterior_mean[cmc_idx])
        cmc_std = float(cmc_result.posterior_std[cmc_idx])

        samples = cmc_result.get_samples(name)
        if samples is not None:
            ax.violinplot(samples, positions=[0], showmeans=True, showmedians=True)
        else:
            # No samples: draw a Gaussian approximation
            x = np.linspace(cmc_mean - 3 * cmc_std, cmc_mean + 3 * cmc_std, 100)
            pdf = np.exp(-0.5 * ((x - cmc_mean) / cmc_std) ** 2) / (
                cmc_std * np.sqrt(2 * np.pi)
            )
            ax.fill_betweenx(x, 0, pdf, alpha=0.3, color="C0", label="CMC posterior")

        # NLSQ point estimate
        nlsq_val = nlsq_result.get_param(name)
        nlsq_unc = nlsq_result.get_uncertainty(name)
        yerr = nlsq_unc if nlsq_unc is not None else 0
        ax.errorbar(
            0,
            nlsq_val,
            yerr=yerr,
            fmt="ro",
            markersize=8,
            capsize=5,
            label="NLSQ",
            zorder=10,
        )

        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("NLSQ vs CMC Comparison", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved NLSQ vs CMC comparison to %s", save_path)
        plt.close(fig)

    return fig


def plot_multi_angle_comparison(
    results_by_phi: dict[float, CMCResult],
    save_path: Path | str | None = None,
) -> Figure:
    """Plot parameter posteriors across multiple phi angles.

    Creates a grid showing how posterior distributions vary with
    detector angle, useful for assessing angle-dependent systematics.

    Args:
        results_by_phi: Dict mapping phi angle (degrees) to CMCResult.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    if not results_by_phi:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No results provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    phi_angles = sorted(results_by_phi.keys())
    first_result = results_by_phi[phi_angles[0]]
    param_names = first_result.parameter_names

    n_params = len(param_names)
    n_angles = len(phi_angles)

    fig, axes = plt.subplots(
        n_params,
        1,
        figsize=(max(6, 1.5 * n_angles), 2.5 * n_params),
        squeeze=False,
    )

    for i, name in enumerate(param_names):
        ax = axes[i, 0]

        means = []
        stds = []
        for phi in phi_angles:
            result = results_by_phi[phi]
            idx = result.parameter_names.index(name)
            means.append(float(result.posterior_mean[idx]))
            stds.append(float(result.posterior_std[idx]))

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.errorbar(
            phi_angles,
            means_arr,
            yerr=stds_arr,
            fmt="o-",
            capsize=4,
            markersize=5,
        )
        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(labelsize=8)

        if i == n_params - 1:
            ax.set_xlabel("phi angle (deg)")
        else:
            ax.set_xticklabels([])

    fig.suptitle("Parameter Posteriors vs Phi Angle", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved multi-angle comparison to %s", save_path)
        plt.close(fig)

    return fig
