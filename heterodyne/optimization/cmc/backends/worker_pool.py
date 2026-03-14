"""Worker pool backend for multi-shard CMC execution.

Spawns persistent workers that each run MCMC on assigned shards.
Amortizes JAX/NumPyro initialization overhead across tasks.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)


def _estimate_physical_workers() -> int:
    """Estimate optimal worker count from physical core topology.

    Uses ``detect_cpu_info()`` for accurate physical core detection
    (lscpu on Linux, sysctl on macOS), reserving one core for the
    main process. Falls back to ``os.cpu_count() // 2`` if detection
    fails or returns suspicious values.

    Returns:
        Number of worker processes (>= 1).
    """
    try:
        from heterodyne.device.cpu import detect_cpu_info

        info = detect_cpu_info()
        physical = info.physical_cores
        # Sanity: physical should be <= logical and >= 1
        if 1 <= physical <= info.logical_cores:
            return max(1, physical - 1)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: assume hyperthreading (2 threads/core)
    logical = os.cpu_count() or 2
    physical_estimate = max(1, logical // 2)
    return max(1, physical_estimate - 1)


def _run_shard_worker(
    model_fn: Callable[..., Any],
    config_dict: dict[str, Any],
    shard_data: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Per-shard MCMC worker function.

    Runs in a subprocess. Imports NumPyro locally to avoid
    fork-safety issues with JAX.

    Args:
        model_fn: NumPyro model factory (must be picklable).
        config_dict: Serialized CMCConfig.
        shard_data: Data for this shard.
        seed: Random seed for this shard.

    Returns:
        Dictionary of posterior samples.
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    # Ensure CPU backend in spawned worker process
    jax.config.update("jax_platform_name", "cpu")

    rng_key = jax.random.PRNGKey(seed)

    kernel = NUTS(
        model_fn,
        target_accept_prob=config_dict["target_accept_prob"],
        max_tree_depth=config_dict.get("max_tree_depth", 10),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=config_dict["num_warmup"],
        num_samples=config_dict["num_samples"],
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )

    mcmc.run(rng_key)
    return dict(mcmc.get_samples())


class WorkerPoolBackend:
    """Persistent worker pool for multi-shard CMC execution.

    Distributes MCMC shards across a pool of worker processes.
    Each worker runs one chain per shard, and results are combined.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        """Initialize with optional worker count.

        Args:
            n_workers: Number of workers. Defaults to physical_cores - 1
                (reserving one core for the main process). Uses
                ``detect_cpu_info()`` for accurate physical core detection;
                falls back to ``os.cpu_count() // 2`` if detection fails.
        """
        if n_workers is None:
            n_workers = _estimate_physical_workers()
        self._n_workers = max(1, n_workers)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def get_name(self) -> str:
        return "worker_pool"

    @staticmethod
    def should_use_pool(n_shards: int, n_workers: int) -> bool:
        """Check if pool execution is beneficial.

        Pool overhead is only amortized when there are enough shards
        to distribute across workers (at least 2 per worker, minimum 3).

        Args:
            n_shards: Number of data shards.
            n_workers: Available workers.

        Returns:
            True if parallelism would be beneficial.
        """
        return n_shards >= max(3, n_workers)

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run MCMC via worker pool.

        Args:
            model: NumPyro model function.
            config: CMC configuration.
            rng_key: JAX PRNG key (used to generate per-shard seeds).
            init_params: Optional initial values (not used in pool mode).

        Returns:
            Combined posterior samples from all workers.
        """
        import jax
        import jax.numpy as jnp

        n_chains = config.num_chains

        logger.info(
            "WorkerPoolBackend: dispatching %d chains across %d workers",
            n_chains,
            self._n_workers,
        )

        config_dict = {
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "target_accept_prob": config.target_accept_prob,
            "max_tree_depth": config.max_tree_depth,
        }

        # Generate deterministic seeds per chain
        seeds = [
            int(
                jax.random.randint(
                    jax.random.fold_in(rng_key, i),
                    (),
                    0,
                    2**31 - 1,
                )
            )
            for i in range(n_chains)
        ]

        with ProcessPoolExecutor(
            max_workers=self._n_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(_run_shard_worker, model, config_dict, {}, seed)
                for seed in seeds
            ]

            all_samples: dict[str, list[Any]] = {}
            for future in futures:
                samples = future.result()
                for name, arr in samples.items():
                    all_samples.setdefault(name, []).append(arr)

        combined = {name: jnp.concatenate(arrs) for name, arrs in all_samples.items()}

        logger.info("WorkerPoolBackend: combined %d chains", n_chains)
        return combined
