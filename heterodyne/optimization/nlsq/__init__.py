"""Non-linear least squares optimization for heterodyne fitting."""

from heterodyne.optimization.nlsq.adapter import NLSQAdapter
from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
    DegeneracyCheck,
    GradientCollapseDetector,
    compute_effective_lambda,
    detect_hierarchical_trigger,
    suggest_regularization,
)
from heterodyne.optimization.nlsq.cmaes_wrapper import (
    CMAES_AVAILABLE,
    CMAESResult,
    adjust_covariance_for_bounds,
    compute_adaptive_cmaes_params,
    denormalize_from_unit_cube,
    fit_with_cmaes,
    normalize_to_unit_cube,
)
from heterodyne.optimization.nlsq.config import NLSQConfig, NLSQValidationConfig
from heterodyne.optimization.nlsq.core import fit_nlsq_jax, fit_nlsq_multi_phi
from heterodyne.optimization.nlsq.data_prep import (
    ExpandedParameters,
    PreparedData,
    build_parameter_labels,
    classify_parameter_status,
    compute_degrees_of_freedom,
    compute_weights,
    convert_bounds_to_nlsq_format,
    expand_per_angle_parameters,
    flatten_upper_triangle,
    prepare_fit_data,
    unflatten_upper_triangle,
    validate_bounds,
    validate_initial_params,
)
from heterodyne.optimization.nlsq.hierarchical import HierarchicalResult
from heterodyne.optimization.nlsq.jacobian import (
    analyze_parameter_sensitivity,
    compare_jacobians,
    compute_jacobian_condition_number,
    compute_jacobian_stats,
    compute_numerical_jacobian,
    estimate_gradient_noise,
    validate_jacobian,
)
from heterodyne.optimization.nlsq.memory import (
    NLSQStrategy,
    StrategyDecision,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    get_adaptive_memory_threshold,
    select_nlsq_strategy,
)
from heterodyne.optimization.nlsq.multistart import (
    MultiStartOptimizer,
    check_zero_volume_bounds,
    generate_lhs_starts,
)
from heterodyne.optimization.nlsq.result_builder import (
    TimedContext,
    build_failed_result,
    build_result_from_arrays,
    build_result_from_scipy,
)
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.strategies import (
    ChunkedStrategy,
    FittingStrategy,
    JITStrategy,
    ResidualStrategy,
    SequentialStrategy,
    StrategyResult,
    select_strategy,
)
from heterodyne.optimization.nlsq.transforms import ParameterTransform
from heterodyne.optimization.nlsq.validation import (
    BoundsValidator,
    ConvergenceValidator,
    FitQualityConfig,
    FitQualityReport,
    FitQualityValidator,
    InputValidator,
    ResultValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    classify_fit_quality,
    validate_fit_quality,
)
from heterodyne.optimization.nlsq.wrapper import NLSQWrapper

__all__ = [
    # Core
    "fit_nlsq_jax",
    "fit_nlsq_multi_phi",
    "NLSQConfig",
    "NLSQValidationConfig",
    "NLSQResult",
    "NLSQAdapter",
    "NLSQAdapterBase",
    "NLSQWrapper",
    "MultiStartOptimizer",
    "NLSQStrategy",
    "select_nlsq_strategy",
    # Strategies
    "FittingStrategy",
    "StrategyResult",
    "ResidualStrategy",
    "JITStrategy",
    "ChunkedStrategy",
    "SequentialStrategy",
    "select_strategy",
    # Data prep
    "ExpandedParameters",
    "PreparedData",
    "build_parameter_labels",
    "classify_parameter_status",
    "convert_bounds_to_nlsq_format",
    "expand_per_angle_parameters",
    "flatten_upper_triangle",
    "unflatten_upper_triangle",
    "compute_weights",
    "prepare_fit_data",
    "compute_degrees_of_freedom",
    "validate_bounds",
    "validate_initial_params",
    # Result building
    "build_result_from_scipy",
    "build_result_from_arrays",
    "build_failed_result",
    "TimedContext",
    # Transforms
    "ParameterTransform",
    # Validation
    "BoundsValidator",
    "ConvergenceValidator",
    "FitQualityConfig",
    "FitQualityReport",
    "FitQualityValidator",
    "InputValidator",
    "ResultValidator",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "classify_fit_quality",
    "validate_fit_quality",
    # Anti-degeneracy (active orchestrator — parity with homodyne)
    "AntiDegeneracyConfig",
    "AntiDegeneracyController",
    "DegeneracyCheck",
    "GradientCollapseDetector",
    "suggest_regularization",
    "compute_effective_lambda",
    "detect_hierarchical_trigger",
    # CMA-ES (parity)
    "CMAES_AVAILABLE",
    "CMAESResult",
    "compute_adaptive_cmaes_params",
    "fit_with_cmaes",
    "normalize_to_unit_cube",
    "denormalize_from_unit_cube",
    "adjust_covariance_for_bounds",
    # Hierarchical (parity)
    "HierarchicalResult",
    # Jacobian (parity)
    "compute_jacobian_stats",
    "compute_jacobian_condition_number",
    "analyze_parameter_sensitivity",
    "estimate_gradient_noise",
    "compare_jacobians",
    "compute_numerical_jacobian",
    "validate_jacobian",
    # Memory (parity)
    "StrategyDecision",
    "detect_total_system_memory",
    "estimate_peak_memory_gb",
    "get_adaptive_memory_threshold",
    # Multi-start (parity)
    "check_zero_volume_bounds",
    "generate_lhs_starts",
]
