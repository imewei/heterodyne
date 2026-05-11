"""Consensus Monte Carlo (CMC) Bayesian analysis for heterodyne fitting."""

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.core import fit_cmc_jax, fit_cmc_sharded
from heterodyne.optimization.cmc.diagnostics import (
    BimodalConsensusResult,
    BimodalResult,
    ModeCluster,
    check_shard_bimodality,
    cluster_shard_modes,
    compute_nlsq_comparison_metrics,
    compute_precision_analysis,
    detect_bimodal,
    summarize_cross_shard_bimodality,
    summarize_diagnostics,
    validate_convergence,
)
from heterodyne.optimization.cmc.model import (
    get_heterodyne_model,
    get_model_param_count,
    validate_model_output,
)
from heterodyne.optimization.cmc.priors import (
    build_init_values_dict,
    estimate_per_angle_scaling,
    extract_nlsq_values_for_cmc,
    get_param_names_in_order,
    validate_initial_value_bounds,
)
from heterodyne.optimization.cmc.reparameterization import ReparamConfig, compute_t_ref
from heterodyne.optimization.cmc.results import CMCResult, ParameterStats
from heterodyne.optimization.cmc.sampler import (
    DIVERGENCE_RATE_CRITICAL,
    DIVERGENCE_RATE_HIGH,
    DIVERGENCE_RATE_TARGET,
    SamplingStats,
    run_nuts_with_retry,
)
from heterodyne.optimization.cmc.scaling import ParameterScaling

# Homodyne-parity alias.  Homodyne's public entry point is named
# ``fit_mcmc_jax``; heterodyne renamed it to ``fit_cmc_jax``.  Expose the
# legacy name so cross-package CLI / driver code keeps working.
fit_mcmc_jax = fit_cmc_jax

__all__ = [
    "fit_cmc_jax",
    "fit_cmc_sharded",
    "fit_mcmc_jax",  # homodyne-parity alias
    "CMCConfig",
    "CMCResult",
    "ParameterStats",
    "get_heterodyne_model",
    "get_model_param_count",
    "validate_model_output",
    "validate_convergence",
    "ReparamConfig",
    "compute_t_ref",
    "ParameterScaling",
    # Diagnostics (parity)
    "BimodalConsensusResult",
    "BimodalResult",
    "ModeCluster",
    "check_shard_bimodality",
    "cluster_shard_modes",
    "compute_nlsq_comparison_metrics",
    "compute_precision_analysis",
    "detect_bimodal",
    "summarize_cross_shard_bimodality",
    "summarize_diagnostics",
    # Priors (parity)
    "get_param_names_in_order",
    "validate_initial_value_bounds",
    "build_init_values_dict",
    "estimate_per_angle_scaling",
    "extract_nlsq_values_for_cmc",
    # Sampler (parity)
    "SamplingStats",
    "run_nuts_with_retry",
    "DIVERGENCE_RATE_TARGET",
    "DIVERGENCE_RATE_HIGH",
    "DIVERGENCE_RATE_CRITICAL",
]
