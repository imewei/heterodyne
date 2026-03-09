"""Consensus Monte Carlo (CMC) Bayesian analysis for heterodyne fitting."""

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.core import fit_cmc_jax
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
from heterodyne.optimization.cmc.results import CMCResult
from heterodyne.optimization.cmc.sampler import (
    DIVERGENCE_RATE_CRITICAL,
    DIVERGENCE_RATE_HIGH,
    DIVERGENCE_RATE_TARGET,
    SamplingStats,
    run_nuts_with_retry,
)
from heterodyne.optimization.cmc.scaling import ParameterScaling

__all__ = [
    "fit_cmc_jax",
    "CMCConfig",
    "CMCResult",
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
