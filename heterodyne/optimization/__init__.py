"""Optimization modules for heterodyne analysis."""

from heterodyne.optimization.cmc import (
    CMCConfig,
    CMCResult,
    fit_cmc_jax,
)
from heterodyne.optimization.nlsq import (
    NLSQAdapter,
    NLSQConfig,
    NLSQResult,
    fit_nlsq_jax,
)

__all__ = [
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
