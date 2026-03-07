"""CPU-optimized MCMC execution backend.

Runs NUTS chains sequentially (one at a time) to avoid memory pressure
on CPU-only machines where all chains share the same memory pool.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization as numpyro_init

from heterodyne.optimization.cmc.backends.base import BackendCapabilities, CMCBackend
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)

# Map config string names to NumPyro initialization factories
_INIT_STRATEGY_MAP: dict[str, Callable[..., Any]] = {
    "init_to_median": numpyro_init.init_to_median,
    "init_to_sample": numpyro_init.init_to_sample,
    "init_to_value": numpyro_init.init_to_value,
}

# Memory constants for CPU estimation
# Bytes per float64 scalar
_BYTES_PER_FLOAT64: int = 8
# Heuristic multiplier: JAX overhead, gradient buffers, NumPyro state
_CPU_MEMORY_OVERHEAD_FACTOR: float = 6.0
_BYTES_PER_GB: float = 1024.0 ** 3


class CPUBackend(CMCBackend):
    """CPU-optimized MCMC backend using sequential chain execution.

    Runs each MCMC chain one at a time via NumPyro's ``chain_method="sequential"``
    to keep peak memory usage proportional to a single chain. This is the
    recommended backend for CPU-only machines and HPC nodes without GPUs.
    """

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run NUTS sampling with sequential chain execution.

        Args:
            model: NumPyro model function.
            config: CMC configuration.
            rng_key: JAX PRNG key.
            init_params: Optional per-chain initial values.

        Returns:
            Dictionary of posterior samples from all chains.

        Raises:
            RuntimeError: If MCMC sampling fails.
        """
        logger.info(
            f"CPUBackend: running {config.num_chains} chains sequentially "
            f"({config.num_warmup} warmup, {config.num_samples} samples each)"
        )

        init_fn = _INIT_STRATEGY_MAP.get(
            config.init_strategy, numpyro_init.init_to_median
        )

        kernel = NUTS(
            model,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            init_strategy=init_fn(),
        )

        mcmc = MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            chain_method="sequential",
            progress_bar=True,
        )

        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))

        samples = mcmc.get_samples()
        logger.info("CPUBackend: sampling complete")

        return dict(samples)

    def get_capabilities(self) -> BackendCapabilities:
        """Return CPU backend capabilities.

        The CPU backend runs chains sequentially (one at a time), does not
        use GPU acceleration, and does not support cross-device sharding.

        Returns:
            ``BackendCapabilities`` reflecting sequential CPU execution.
        """
        return BackendCapabilities(
            supports_sharding=False,
            supports_parallel_chains=False,
            max_parallel_shards=1,
            supports_gpu=False,
        )

    def validate_resources(self) -> None:
        """Verify that CPU resources are available for sampling.

        Checks that at least one JAX CPU device is accessible.

        Raises:
            RuntimeError: If no CPU device is found via ``jax.devices``.
        """
        import jax

        devices = jax.devices("cpu")
        if not devices:
            raise RuntimeError(
                "CPUBackend: no JAX CPU devices found. "
                "Ensure JAX is installed correctly."
            )
        logger.debug("CPUBackend.validate_resources: %d CPU device(s) found", len(devices))

    def estimate_memory(
        self,
        n_data: int,
        n_params: int,
        n_chains: int,
    ) -> float:
        """Estimate peak CPU memory for a single sequential chain.

        Because chains run sequentially only one chain's state is live
        at any moment, so ``n_chains`` does not multiply peak usage.

        The formula accounts for:

        - Flat parameter storage per draw: ``n_params`` float64 scalars.
        - Gradient / momentum buffers: same size as parameters.
        - Sample storage for the completed chain: ``num_samples * n_params``.
        - Data residual buffer: ``n_data`` float64 scalars.
        - A conservative overhead multiplier (``_CPU_MEMORY_OVERHEAD_FACTOR``)
          for JAX tracing buffers and NumPyro auxiliary state.

        Args:
            n_data: Number of data points per shard.
            n_params: Number of model parameters.
            n_chains: Number of chains (not used for sequential backend;
                included for API uniformity).

        Returns:
            Estimated peak memory in gigabytes.
        """
        # Storage for one chain's live state (params + momentum + grad)
        state_bytes = 3 * n_params * _BYTES_PER_FLOAT64
        # Data buffer (residuals, weights)
        data_bytes = n_data * _BYTES_PER_FLOAT64
        # Raw bytes before overhead
        raw_bytes = state_bytes + data_bytes
        total_bytes = raw_bytes * _CPU_MEMORY_OVERHEAD_FACTOR
        return total_bytes / _BYTES_PER_GB

    def _configure_threading(self) -> None:
        """Set XLA / OpenMP threading flags for optimal CPU throughput.

        Reads ``OMP_NUM_THREADS`` from the environment.  When not set,
        defaults to the physical CPU count reported by ``os.cpu_count()``.
        Sets the ``XLA_FLAGS`` environment variable to pin XLA's inter-op
        and intra-op thread counts, preventing over-subscription on NUMA
        nodes.

        This method is idempotent: calling it multiple times has no
        additional effect beyond the first call.
        """
        cpu_count = os.cpu_count() or 1
        n_threads = int(os.environ.get("OMP_NUM_THREADS", cpu_count))

        existing_flags = os.environ.get("XLA_FLAGS", "")

        # Only inject our flags if they have not already been set to avoid
        # overriding deliberate user configuration.
        injected: list[str] = []
        if "--xla_cpu_multi_thread_eigen" not in existing_flags:
            injected.append("--xla_cpu_multi_thread_eigen=true")
        if "--intra_op_parallelism_threads" not in existing_flags:
            injected.append(f"--intra_op_parallelism_threads={n_threads}")
        if "--inter_op_parallelism_threads" not in existing_flags:
            # Sequential chain execution needs minimal inter-op threads
            injected.append("--inter_op_parallelism_threads=1")

        if injected:
            separator = " " if existing_flags else ""
            os.environ["XLA_FLAGS"] = existing_flags + separator + " ".join(injected)
            logger.debug(
                "CPUBackend._configure_threading: set XLA_FLAGS += %s "
                "(n_threads=%d)",
                " ".join(injected),
                n_threads,
            )
        else:
            logger.debug(
                "CPUBackend._configure_threading: XLA_FLAGS already configured, skipping"
            )

    def cleanup(self) -> None:
        """Release CPU backend resources.

        The CPU backend holds no persistent state beyond what JAX and
        NumPyro manage internally, so this is a no-op.  Included for
        API parity with GPU/worker-pool backends.
        """
        logger.debug("CPUBackend.cleanup: nothing to release")
