"""MCMC execution backends for heterodyne CMC analysis.

Provides CPU-optimized (sequential), multiprocessing (parallel shards),
and multi-device (pjit) backends for running NUTS chains via NumPyro.
Heterodyne is CPU-only.
"""

from heterodyne.optimization.cmc.backends.base import (
    MCMCBackend,
    ShardPosterior,
    combine_shard_samples,
    combine_shard_samples_bimodal,
    consensus_mc,
    robust_consensus_mc,
    select_backend,
)
from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
from heterodyne.optimization.cmc.backends.multiprocessing import (
    MultiprocessingBackend,
)
from heterodyne.optimization.cmc.backends.pbs import PBSBackend, PBSConfig
from heterodyne.optimization.cmc.backends.pjit import PjitBackend
from heterodyne.optimization.cmc.backends.worker_pool import (
    PersistentWorkerPool,
    WorkerPoolBackend,
    should_use_persistent_pool,
)

__all__ = [
    "CPUBackend",
    "MCMCBackend",
    "MultiprocessingBackend",
    "PBSBackend",
    "PBSConfig",
    "PersistentWorkerPool",
    "PjitBackend",
    "ShardPosterior",
    "WorkerPoolBackend",
    "combine_shard_samples",
    "combine_shard_samples_bimodal",
    "consensus_mc",
    "robust_consensus_mc",
    "select_backend",
    "should_use_persistent_pool",
]
