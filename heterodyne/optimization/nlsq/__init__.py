"""Non-linear least squares optimization for heterodyne fitting."""

from heterodyne.optimization.nlsq.adapter import NLSQAdapter
from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    GradientCollapseDetector,
    compute_effective_lambda,
    detect_hierarchical_trigger,
    suggest_regularization,
)
from heterodyne.optimization.nlsq.cmaes_wrapper import (
    CMAESResult,
    adjust_covariance_for_bounds,
    denormalize_from_unit_cube,
    normalize_to_unit_cube,
)
from heterodyne.optimization.nlsq.config import NLSQConfig, NLSQValidationConfig
from heterodyne.optimization.nlsq.core import fit_nlsq_jax
from heterodyne.optimization.nlsq.data_prep import (
    compute_degrees_of_freedom,
    compute_weights,
    flatten_upper_triangle,
    prepare_fit_data,
    unflatten_upper_triangle,
)
from heterodyne.optimization.nlsq.hierarchical import HierarchicalResult
from heterodyne.optimization.nlsq.jacobian import (
    analyze_parameter_sensitivity,
    compute_jacobian_condition_number,
    estimate_gradient_noise,
)
from heterodyne.optimization.nlsq.memory import NLSQStrategy, select_nlsq_strategy
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
    ResultValidator,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    # Core
    "fit_nlsq_jax",
    "NLSQConfig",
    "NLSQValidationConfig",
    "NLSQResult",
    "NLSQAdapter",
    "NLSQAdapterBase",
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
    "flatten_upper_triangle",
    "unflatten_upper_triangle",
    "compute_weights",
    "prepare_fit_data",
    "compute_degrees_of_freedom",
    # Result building
    "build_result_from_scipy",
    "build_result_from_arrays",
    "build_failed_result",
    "TimedContext",
    # Transforms
    "ParameterTransform",
    # Validation
    "ResultValidator",
    "BoundsValidator",
    "ConvergenceValidator",
    "ValidationReport",
    "ValidationSeverity",
    # Anti-degeneracy (parity)
    "GradientCollapseDetector",
    "suggest_regularization",
    "compute_effective_lambda",
    "detect_hierarchical_trigger",
    # CMA-ES (parity)
    "CMAESResult",
    "normalize_to_unit_cube",
    "denormalize_from_unit_cube",
    "adjust_covariance_for_bounds",
    # Hierarchical (parity)
    "HierarchicalResult",
    # Jacobian (parity)
    "compute_jacobian_condition_number",
    "analyze_parameter_sensitivity",
    "estimate_gradient_noise",
    # Multi-start (parity)
    "check_zero_volume_bounds",
    "generate_lhs_starts",
]
