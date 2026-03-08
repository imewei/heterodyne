"""Configuration management for heterodyne analysis."""

from heterodyne.config.manager import ConfigManager, load_xpcs_config
from heterodyne.config.parameter_manager import ParameterManager
from heterodyne.config.parameter_names import (
    ALL_PARAM_NAMES,
    ANGLE_PARAMS,
    FRACTION_PARAMS,
    PARAM_GROUPS,
    REFERENCE_PARAMS,
    SAMPLE_PARAMS,
    VELOCITY_PARAMS,
)
from heterodyne.config.parameter_registry import ParameterInfo, ParameterRegistry
from heterodyne.config.parameter_space import ParameterSpace, PriorDistribution
from heterodyne.config.types import (
    PARAMETER_NAME_MAPPING,
    AnalysisConfig,
    AnalyzerParametersConfig,
    CMCBackendConfig,
    CMCCombinationConfig,
    CMCInitializationConfig,
    CMCOptimizationConfig,
    CMCPerShardMCMCConfig,
    CMCShardingConfig,
    CMCValidationConfig,
    ExperimentalDataConfig,
    HardwareConfig,
    HeterodyneConfig,
    HmcConfig,
    LoggingConfig,
    MetadataConfig,
    NLSQOptimizationConfig,
    OptimizationConfig,
    OutputConfig,
    ParameterConfig,
    ParameterGroupConfig,
    PhiFilteringConfig,
    ScatteringConfig,
    SequentialConfig,
    StratificationConfig,
    StreamingConfig,
    TemporalConfig,
)

__all__ = [
    # Manager
    "ConfigManager",
    "load_xpcs_config",
    # Parameter management
    "ParameterManager",
    "ParameterInfo",
    "ParameterRegistry",
    "ParameterSpace",
    "PriorDistribution",
    # Parameter names
    "REFERENCE_PARAMS",
    "SAMPLE_PARAMS",
    "VELOCITY_PARAMS",
    "FRACTION_PARAMS",
    "ANGLE_PARAMS",
    "ALL_PARAM_NAMES",
    "PARAM_GROUPS",
    # Types — existing
    "ParameterConfig",
    "ParameterGroupConfig",
    "NLSQOptimizationConfig",
    "CMCOptimizationConfig",
    "OptimizationConfig",
    "ExperimentalDataConfig",
    "TemporalConfig",
    "ScatteringConfig",
    "OutputConfig",
    "HeterodyneConfig",
    # Types — CMC
    "CMCShardingConfig",
    "CMCInitializationConfig",
    "CMCBackendConfig",
    "CMCValidationConfig",
    "CMCCombinationConfig",
    "CMCPerShardMCMCConfig",
    # Types — infrastructure
    "StreamingConfig",
    "StratificationConfig",
    "SequentialConfig",
    "HardwareConfig",
    "LoggingConfig",
    "MetadataConfig",
    # Types — analysis
    "AnalysisConfig",
    "AnalyzerParametersConfig",
    "HmcConfig",
    "PhiFilteringConfig",
    # Name mapping
    "PARAMETER_NAME_MAPPING",
]
