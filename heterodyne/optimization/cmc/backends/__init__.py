"""MCMC execution backends for heterodyne CMC analysis.

Provides CPU-optimized (sequential), GPU-optimized (parallel), and
multi-device (pjit) backends for running NUTS chains via NumPyro.
"""

from heterodyne.optimization.cmc.backends.base import (
    MCMCBackend,
    select_backend,
)
from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
from heterodyne.optimization.cmc.backends.gpu_backend import GPUBackend
from heterodyne.optimization.cmc.backends.pjit_backend import PjitBackend

__all__ = [
    "CPUBackend",
    "GPUBackend",
    "MCMCBackend",
    "PjitBackend",
    "select_backend",
]
