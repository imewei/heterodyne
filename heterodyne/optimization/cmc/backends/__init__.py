"""MCMC execution backends for heterodyne CMC analysis.

Provides CPU-optimized (sequential), multiprocessing (parallel shards),
and multi-device (pjit) backends for running NUTS chains via NumPyro.
Heterodyne is CPU-only.
"""

from heterodyne.optimization.cmc.backends.base import (
    MCMCBackend,
    select_backend,
)
from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
from heterodyne.optimization.cmc.backends.multiprocessing_backend import (
    MultiprocessingBackend,
)
from heterodyne.optimization.cmc.backends.pjit_backend import PjitBackend

__all__ = [
    "CPUBackend",
    "MCMCBackend",
    "MultiprocessingBackend",
    "PjitBackend",
    "select_backend",
]
