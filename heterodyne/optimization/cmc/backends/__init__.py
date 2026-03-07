"""MCMC execution backends for heterodyne CMC analysis.

Provides CPU-optimized (sequential) and GPU-optimized (parallel) backends
for running NUTS chains via NumPyro.
"""

from heterodyne.optimization.cmc.backends.base import (
    MCMCBackend,
    select_backend,
)
from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
from heterodyne.optimization.cmc.backends.gpu_backend import GPUBackend

__all__ = [
    "CPUBackend",
    "GPUBackend",
    "MCMCBackend",
    "select_backend",
]
