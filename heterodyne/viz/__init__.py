"""Visualization utilities for heterodyne analysis."""

from heterodyne.viz.experimental_plots import (
    plot_correlation,
    plot_diagonal_decay,
    plot_g1_components,
    plot_phi_dependence,
)
from heterodyne.viz.mcmc_arviz import (
    plot_arviz_pair,
    plot_arviz_posterior,
    plot_arviz_trace,
    to_inference_data,
)
from heterodyne.viz.mcmc_comparison import (
    plot_multi_angle_comparison,
    plot_nlsq_vs_cmc,
)
from heterodyne.viz.mcmc_dashboard import plot_cmc_summary_dashboard
from heterodyne.viz.mcmc_diagnostics import (
    plot_adaptation_summary,
    plot_convergence_diagnostics,
    plot_divergence_scatter,
    plot_ess_evolution,
    plot_kl_divergence_matrix,
)
from heterodyne.viz.mcmc_plots import plot_corner, plot_posterior, plot_trace
from heterodyne.viz.mcmc_report import (
    ReportConfig,
    generate_report,
)
from heterodyne.viz.nlsq_plots import (
    plot_nlsq_fit,
    plot_parameter_uncertainties,
    plot_residual_map,
)

__all__ = [
    "ReportConfig",
    "generate_report",
    "plot_adaptation_summary",
    "plot_cmc_summary_dashboard",
    "plot_arviz_pair",
    "plot_arviz_posterior",
    "plot_arviz_trace",
    "plot_convergence_diagnostics",
    "plot_corner",
    "plot_correlation",
    "plot_diagonal_decay",
    "plot_divergence_scatter",
    "plot_ess_evolution",
    "plot_g1_components",
    "plot_kl_divergence_matrix",
    "plot_multi_angle_comparison",
    "plot_nlsq_fit",
    "plot_nlsq_vs_cmc",
    "plot_parameter_uncertainties",
    "plot_phi_dependence",
    "plot_posterior",
    "plot_residual_map",
    "plot_trace",
    "to_inference_data",
]
