"""Optimization modules for heterodyne analysis."""

from heterodyne.optimization.cmc import (
    CMCConfig,
    CMCResult,
    fit_cmc_jax,
)
from heterodyne.optimization.exceptions import (
    BoundsError,
    ConvergenceError,
    DegeneracyError,
    NumericalError,
    OptimizationError,
    ValidationError,
)
from heterodyne.optimization.nlsq import (
    NLSQAdapter,
    NLSQConfig,
    NLSQResult,
    fit_nlsq_jax,
)

__all__ = [
    # Exceptions
    "OptimizationError",
    "ConvergenceError",
    "NumericalError",
    "BoundsError",
    "DegeneracyError",
    "ValidationError",
    # NLSQ
    "fit_nlsq_jax",
    "NLSQConfig",
    "NLSQResult",
    "NLSQAdapter",
    # CMC
    "fit_cmc_jax",
    "CMCConfig",
    "CMCResult",
]
