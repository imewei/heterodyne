"""TypedDict definitions for configuration structures."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ParameterConfig(TypedDict):
    """Configuration for a single parameter."""

    value: float
    min: float
    max: float
    vary: bool
    prior: NotRequired[str]  # "uniform", "normal", "lognormal"
    prior_params: NotRequired[dict[str, float]]


class ParameterGroupConfig(TypedDict):
    """Configuration for a parameter group.

    Keys use full parameter names matching parameter_space.from_config().
    Group 'reference': D0_ref, alpha_ref, D_offset_ref
    Group 'sample': D0_sample, alpha_sample, D_offset_sample
    Group 'velocity': v0, beta, v_offset
    Group 'fraction': f0, f1, f2, f3
    Group 'angle': phi0
    """

    # Reference transport
    D0_ref: NotRequired[ParameterConfig]
    alpha_ref: NotRequired[ParameterConfig]
    D_offset_ref: NotRequired[ParameterConfig]
    # Sample transport
    D0_sample: NotRequired[ParameterConfig]
    alpha_sample: NotRequired[ParameterConfig]
    D_offset_sample: NotRequired[ParameterConfig]
    # Velocity
    v0: NotRequired[ParameterConfig]
    beta: NotRequired[ParameterConfig]
    v_offset: NotRequired[ParameterConfig]
    # Fraction
    f0: NotRequired[ParameterConfig]
    f1: NotRequired[ParameterConfig]
    f2: NotRequired[ParameterConfig]
    f3: NotRequired[ParameterConfig]
    # Angle
    phi0: NotRequired[ParameterConfig]


class NLSQOptimizationConfig(TypedDict):
    """NLSQ optimization configuration."""

    max_iterations: NotRequired[int]
    tolerance: NotRequired[float]
    method: NotRequired[str]
    multistart: NotRequired[bool]
    multistart_n: NotRequired[int]


class CMCOptimizationConfig(TypedDict):
    """CMC (Consensus Monte Carlo) configuration."""

    enable: NotRequired[Literal["auto", "always", "never"]]
    num_warmup: NotRequired[int]
    num_samples: NotRequired[int]
    num_chains: NotRequired[int]
    target_accept_prob: NotRequired[float]


class OptimizationConfig(TypedDict):
    """Full optimization configuration."""

    method: str  # "nlsq" or "cmc"
    nlsq: NotRequired[NLSQOptimizationConfig]
    cmc: NotRequired[CMCOptimizationConfig]


class ExperimentalDataConfig(TypedDict):
    """Experimental data file configuration."""

    file_path: str
    data_folder_path: NotRequired[str]
    file_format: NotRequired[str]  # "hdf5", "npz", "mat"


class TemporalConfig(TypedDict):
    """Temporal/timing configuration."""

    dt: float
    time_length: int
    t_start: NotRequired[int]


class ScatteringConfig(TypedDict):
    """Scattering geometry configuration."""

    wavevector_q: float
    phi_angles: NotRequired[list[float]]


class OutputConfig(TypedDict):
    """Output configuration."""

    output_dir: NotRequired[str]
    save_correlation: NotRequired[bool]
    save_residuals: NotRequired[bool]
    format: NotRequired[str]  # "json", "npz", "both"


class HeterodyneConfig(TypedDict):
    """Complete heterodyne analysis configuration."""

    experimental_data: ExperimentalDataConfig
    temporal: TemporalConfig
    scattering: ScatteringConfig
    parameters: dict[str, ParameterGroupConfig]
    optimization: OptimizationConfig
    output: NotRequired[OutputConfig]


# ---------------------------------------------------------------------------
# CMC Types
# ---------------------------------------------------------------------------


class CMCShardingConfig(TypedDict):
    """Sharding configuration for Consensus Monte Carlo."""

    n_shards: NotRequired[int]
    shard_strategy: NotRequired[str]  # "equal", "adaptive"
    min_pairs_per_shard: NotRequired[int]


class CMCInitializationConfig(TypedDict):
    """Initialization strategy for CMC chains."""

    method: NotRequired[str]  # "nlsq_warmstart", "prior_sample", "manual"
    nlsq_chi2_threshold: NotRequired[float]
    jitter_scale: NotRequired[float]


class CMCBackendConfig(TypedDict):
    """Backend configuration for CMC sampling."""

    sampler: NotRequired[str]  # "nuts", "hmc", "sa"
    jit_compile: NotRequired[bool]
    progress_bar: NotRequired[bool]


class CMCValidationConfig(TypedDict):
    """Convergence validation criteria for CMC."""

    max_r_hat: NotRequired[float]
    min_ess: NotRequired[int]
    min_bfmi: NotRequired[float]
    max_divergences: NotRequired[int]


class CMCCombinationConfig(TypedDict):
    """Shard combination strategy for CMC."""

    method: NotRequired[str]  # "consensus", "weighted", "median"
    weights: NotRequired[list[float]]


class CMCPerShardMCMCConfig(TypedDict):
    """Per-shard MCMC sampler settings."""

    num_warmup: NotRequired[int]
    num_samples: NotRequired[int]
    num_chains: NotRequired[int]
    target_accept_prob: NotRequired[float]
    max_tree_depth: NotRequired[int]


# ---------------------------------------------------------------------------
# Infrastructure Types
# ---------------------------------------------------------------------------


class StreamingConfig(TypedDict):
    """Streaming data ingestion configuration."""

    enable: NotRequired[bool]
    chunk_size: NotRequired[int]
    buffer_size: NotRequired[int]


class StratificationConfig(TypedDict):
    """Data stratification configuration."""

    enable: NotRequired[bool]
    n_strata: NotRequired[int]
    strategy: NotRequired[str]  # "time", "value", "adaptive"


class SequentialConfig(TypedDict):
    """Sequential optimization configuration."""

    enable: NotRequired[bool]
    max_iterations: NotRequired[int]
    convergence_threshold: NotRequired[float]


class HardwareConfig(TypedDict):
    """Hardware and resource configuration."""

    device: NotRequired[str]  # "cpu" (heterodyne is CPU-only)
    n_threads: NotRequired[int]
    memory_limit_gb: NotRequired[float]
    numa_aware: NotRequired[bool]


class LoggingConfig(TypedDict):
    """Logging configuration."""

    level: NotRequired[str]  # "DEBUG", "INFO", "WARNING", "ERROR"
    file: NotRequired[str]
    structured: NotRequired[bool]
    log_convergence: NotRequired[bool]


class MetadataConfig(TypedDict):
    """Experiment metadata configuration."""

    experiment_id: NotRequired[str]
    operator: NotRequired[str]
    beamline: NotRequired[str]
    sample_name: NotRequired[str]
    notes: NotRequired[str]


# ---------------------------------------------------------------------------
# Analysis Types
# ---------------------------------------------------------------------------


class AnalysisConfig(TypedDict):
    """Top-level analysis configuration."""

    method: str  # "nlsq", "cmc", "both"
    phi_angles: NotRequired[list[float]]
    q: NotRequired[float]
    dt: NotRequired[float]
    scaling_mode: NotRequired[
        str
    ]  # "constant", "individual", "auto", "constant_averaged"


class AnalyzerParametersConfig(TypedDict):
    """Grouped parameter configuration for the analyzer."""

    reference: NotRequired[ParameterGroupConfig]
    sample: NotRequired[ParameterGroupConfig]
    velocity: NotRequired[ParameterGroupConfig]
    fraction: NotRequired[ParameterGroupConfig]
    angle: NotRequired[ParameterGroupConfig]
    scaling: NotRequired[ParameterGroupConfig]


class HmcConfig(TypedDict):
    """HMC-specific sampler configuration."""

    step_size: NotRequired[float]
    num_leapfrog_steps: NotRequired[int]
    adapt_step_size: NotRequired[bool]
    adapt_mass_matrix: NotRequired[bool]


class PhiFilteringConfig(TypedDict):
    """Phi-angle filtering configuration."""

    include_angles: NotRequired[list[float]]
    exclude_angles: NotRequired[list[float]]
    tolerance: NotRequired[float]


# ---------------------------------------------------------------------------
# Name Mapping
# ---------------------------------------------------------------------------

PARAMETER_NAME_MAPPING: dict[str, str] = {
    "D0_reference": "D0_ref",
    "D0_ref_value": "D0_ref",
    "alpha_reference": "alpha_ref",
    "D_offset_reference": "D_offset_ref",
    "velocity": "v0",
    "v_0": "v0",
    "velocity_offset": "v_offset",
    "fraction_0": "f0",
    "fraction_1": "f1",
    "fraction_2": "f2",
    "fraction_3": "f3",
    "angle_phi0": "phi0",
    "target_accept": "target_accept_prob",
    "r_hat_threshold": "max_r_hat",
    "prior_width_factor": "nlsq_prior_width_factor",
}
