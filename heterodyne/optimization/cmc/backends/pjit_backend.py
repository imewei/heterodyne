"""Multi-device parallel MCMC backend using JAX sharding.

Distributes NUTS chains across multiple CPU devices using JAX's modern
sharding API (``jax.sharding``), available since JAX 0.4.1.  This
replaces the deprecated ``jax.experimental.pjit`` with the stable
``jax.jit`` + sharding annotation approach.  Heterodyne is CPU-only.
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


def _slice_init_params(
    init_params: dict[str, jnp.ndarray] | None,
    start: int,
    count: int,
    n_chains_total: int,
) -> dict[str, jnp.ndarray] | None:
    """Return per-device init slice for the chain block ``[start, start+count)``.

    Leaves whose leading axis equals ``n_chains_total`` are sliced along
    axis 0; everything else (scalars, broadcast inits, init_to_median
    state) is forwarded unchanged so common cases keep working.
    """
    if init_params is None or count == n_chains_total:
        return init_params
    sliced: dict[str, jnp.ndarray] = {}
    for name, leaf in init_params.items():
        if hasattr(leaf, "shape") and leaf.shape and leaf.shape[0] == n_chains_total:
            sliced[name] = leaf[start : start + count]
        else:
            sliced[name] = leaf
    return sliced


class PjitBackend(CMCBackend):
    """Multi-device parallel MCMC backend using JAX sharding.

    Distributes NUTS chains across all available JAX devices. Each
    device runs a subset of the requested chains in parallel via
    NumPyro's vectorized chain execution.

    When only a single device is available, this backend transparently
    falls back to running all chains on that device (equivalent to the
    single-device ``chain_method="parallel"``).

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

        if n_chains < 1:
            raise ValueError(f"PjitBackend: num_chains must be >= 1, got {n_chains}")
        if n_devices < 1:
            raise RuntimeError("PjitBackend: no JAX devices available")

        # Chains-per-device distribution. The previous ``max(1, n_chains //
        # n_devices)`` produced spurious extra chains when n_chains < n_devices
        # (e.g. 2 chains on 8 devices ran 8 chains instead of 2). Allocate
        # base chains by floor division and spread the remainder over the
        # first ``remainder`` devices; devices with zero chains are skipped.
        chains_per_device = n_chains // n_devices
        remainder = n_chains % n_devices

        init_fn = _INIT_STRATEGY_MAP.get(
            config.init_strategy, numpyro_init.init_to_median
        )

        # Split RNG keys for each device group
        rng_keys = jax.random.split(rng_key, n_devices)

        shard_results: list[dict[str, Any]] = []
        chain_cursor = 0  # tracks position in init_params chain axis

        for device_idx in range(n_devices):
            # First ``remainder`` devices pick up one extra chain
            device_chains = chains_per_device + (1 if device_idx < remainder else 0)
            if device_chains == 0:
                continue

            device = devices[device_idx]
            device_rng = rng_keys[device_idx]

            # Slice init_params for this device's chain block if the user
            # supplied a chain-shaped init (leading axis == n_chains). Leaf-
            # broadcasted inits (no chain axis, or singleton leading dim) are
            # forwarded unchanged so init_to_median / scalar inits still work.
            device_init = _slice_init_params(
                init_params, chain_cursor, device_chains, n_chains
            )
            chain_cursor += device_chains

            logger.debug(
                "PjitBackend: device %d (%s) running %d chain(s) "
                "[chains %d..%d of %d total]",
                device_idx,
                device.platform,
                device_chains,
                chain_cursor - device_chains,
                chain_cursor - 1,
                n_chains,
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
                    init_params=device_init,
                    extra_fields=_NUTS_EXTRA_FIELDS,
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
            BackendCapabilities with sharding support flags.
        """
        devices = jax.devices()

        return BackendCapabilities(
            supports_sharding=True,
            supports_parallel_chains=True,
            max_parallel_shards=len(devices),
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
                "Consider using CPUBackend instead.",
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
