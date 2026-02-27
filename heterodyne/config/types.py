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
    target_accept: NotRequired[float]


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
