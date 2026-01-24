"""Consensus Monte Carlo (CMC) Bayesian analysis for heterodyne fitting."""

from heterodyne.optimization.cmc.core import fit_cmc_jax
from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.results import CMCResult
from heterodyne.optimization.cmc.model import get_heterodyne_model
from heterodyne.optimization.cmc.diagnostics import validate_convergence

__all__ = [
    "fit_cmc_jax",
    "CMCConfig",
    "CMCResult",
    "get_heterodyne_model",
    "validate_convergence",
]
