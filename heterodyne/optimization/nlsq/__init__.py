"""Non-linear least squares optimization for heterodyne fitting."""

from heterodyne.optimization.nlsq.adapter import NLSQAdapter
from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
from heterodyne.optimization.nlsq.config import NLSQConfig, NLSQValidationConfig
from heterodyne.optimization.nlsq.core import fit_nlsq_jax
from heterodyne.optimization.nlsq.data_prep import (
    compute_degrees_of_freedom,
    compute_weights,
    flatten_upper_triangle,
    prepare_fit_data,
    unflatten_upper_triangle,
)
from heterodyne.optimization.nlsq.memory import NLSQStrategy, select_nlsq_strategy
from heterodyne.optimization.nlsq.multistart import MultiStartOptimizer
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
]
