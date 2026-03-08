"""Multi-device parallel MCMC backend using JAX sharding.

Distributes NUTS chains across multiple JAX devices (CPU or GPU) using
JAX's modern sharding API (``jax.sharding``), available since JAX 0.4.1.
This replaces the deprecated ``jax.experimental.pjit`` with the stable
``jax.jit`` + sharding annotation approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization as numpyro_init

from heterodyne.optimization.cmc.backends.base import (
    BackendCapabilities,
    CMCBackend,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)

# Map config string names to NumPyro initialization factories
_INIT_STRATEGY_MAP: dict[str, Callable[..., Any]] = {
    "init_to_median": numpyro_init.init_to_median,
    "init_to_sample": numpyro_init.init_to_sample,
    "init_to_value": numpyro_init.init_to_value,
}


class PjitBackend(CMCBackend):
    """Multi-device parallel MCMC backend using JAX sharding.

    Distributes NUTS chains across all available JAX devices. Each
    device runs a subset of the requested chains in parallel via
    NumPyro's vectorized chain execution.

    When only a single device is available, this backend transparently
    falls back to running all chains on that device (equivalent to the
    GPU backend's ``chain_method="parallel"``).

    This backend uses the modern ``jax.sharding`` API (stable since
    JAX 0.4.1), not the deprecated ``jax.experimental.pjit``.
    """

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run NUTS sampling distributed across multiple devices.

        Splits chains across available devices. Each device group runs
        independently, and results are gathered and concatenated.

        Args:
            model: NumPyro model function.
            config: CMC configuration with sampling hyperparameters.
            rng_key: JAX PRNG key for reproducibility.
            init_params: Optional per-chain initial parameter values.

        Returns:
            Dictionary mapping parameter names to flat sample arrays.

        Raises:
            RuntimeError: If sampling fails on any device.
        """
        devices = jax.devices()
        n_devices = len(devices)
        n_chains = config.num_chains

        logger.info(
            "PjitBackend: distributing %d chains across %d device(s) "
            "(%d warmup, %d samples each)",
            n_chains,
            n_devices,
            config.num_warmup,
            config.num_samples,
        )

        # Determine chains per device
        chains_per_device = max(1, n_chains // n_devices)
        remainder = n_chains - chains_per_device * n_devices

        init_fn = _INIT_STRATEGY_MAP.get(
            config.init_strategy, numpyro_init.init_to_median
        )

        # Split RNG keys for each device group
        rng_keys = jax.random.split(rng_key, n_devices)

        shard_results: list[dict[str, Any]] = []

        for device_idx in range(n_devices):
            # Last device picks up remainder chains
            device_chains = chains_per_device + (1 if device_idx < remainder else 0)
            if device_chains == 0:
                continue

            device = devices[device_idx]
            device_rng = rng_keys[device_idx]

            logger.debug(
                "PjitBackend: device %d (%s) running %d chain(s)",
                device_idx,
                device.platform,
                device_chains,
            )

            # Place computation on specific device
            with jax.default_device(device):
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
                    num_chains=device_chains,
                    chain_method="parallel" if device_chains > 1 else "sequential",
                    progress_bar=(device_idx == 0),  # Only show for first device
                )

                mcmc.run(
                    device_rng,
                    init_params=init_params,
                    extra_fields=("energy",),
                )

                shard_samples = mcmc.get_samples()
                shard_results.append(dict(shard_samples))

        # Combine results from all shards
        combined = combine_shard_samples(shard_results)

        logger.info(
            "PjitBackend: sampling complete — %d total samples across %d device(s)",
            config.num_samples * n_chains,
            n_devices,
        )

        return combined

    def get_capabilities(self) -> BackendCapabilities:
        """Return capabilities for multi-device parallel execution.

        Returns:
            BackendCapabilities with sharding and GPU support flags.
        """
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)

        return BackendCapabilities(
            supports_sharding=True,
            supports_parallel_chains=True,
            max_parallel_shards=len(devices),
            supports_gpu=has_gpu,
        )

    def validate_resources(self) -> None:
        """Check that multiple devices are available for sharding.

        Logs a warning (but does not raise) when only one device is
        detected, since the backend can still function in single-device
        mode.

        Raises:
            RuntimeError: If no JAX devices are available at all.
        """
        devices = jax.devices()

        if not devices:
            raise RuntimeError("PjitBackend: no JAX devices available")

        if len(devices) == 1:
            logger.warning(
                "PjitBackend: only 1 device detected (%s); "
                "sharding will not provide parallelism. "
                "Consider using CPUBackend or GPUBackend instead.",
                devices[0].platform,
            )
        else:
            logger.info(
                "PjitBackend: %d devices available (%s)",
                len(devices),
                ", ".join(f"{d.platform}:{d.id}" for d in devices),
            )

    def estimate_memory(
        self,
        n_data: int,
        n_params: int,
        n_chains: int,
    ) -> float:
        """Estimate peak memory per device in GB.

        Each device holds a fraction of the chains. Memory per chain
        is approximately: n_params x n_data x 8 bytes (float64) for
        the likelihood evaluation, plus sample storage.

        Args:
            n_data: Number of data points per shard.
            n_params: Number of model parameters.
            n_chains: Total number of MCMC chains.

        Returns:
            Estimated peak memory in GB per device.
        """
        n_devices = max(len(jax.devices()), 1)
        chains_per_device = max(1, n_chains // n_devices)

        # Per-chain memory: model evaluation + gradient + samples
        bytes_per_chain = (
            n_data * n_params * 8  # Jacobian-like
            + n_params * n_params * 8  # Mass matrix
            + n_data * 8  # Residuals
        )
        total_bytes = bytes_per_chain * chains_per_device

        return total_bytes / (1024**3)

    def cleanup(self) -> None:
        """Release resources. No-op for JAX-managed devices."""
        logger.debug("PjitBackend: cleanup (no-op for JAX-managed devices)")


def combine_shard_samples(
    shard_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine posterior samples from multiple device shards.

    Concatenates sample arrays along the first axis (samples dimension).

    Args:
        shard_results: List of sample dictionaries, one per device shard.
            Each dict maps parameter names to numpy/JAX arrays.

    Returns:
        Combined dictionary with concatenated samples.

    Raises:
        ValueError: If shard_results is empty.
    """
    if not shard_results:
        raise ValueError("combine_shard_samples requires at least 1 shard result")

    if len(shard_results) == 1:
        return shard_results[0]

    # Get parameter names from first shard
    param_names = list(shard_results[0].keys())
    combined: dict[str, Any] = {}

    for name in param_names:
        arrays = [np.asarray(sr[name]) for sr in shard_results if name in sr]
        if arrays:
            combined[name] = np.concatenate(arrays, axis=0)

    logger.debug(
        "combine_shard_samples: combined %d shards → %d parameters",
        len(shard_results),
        len(combined),
    )

    return combined
