"""GPU-optimized MCMC execution backend.

Uses NumPyro's ``chain_method="parallel"`` to run all NUTS chains
simultaneously across GPU devices via ``jax.pmap``.  Falls back to
the CPU backend when no GPU is available at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization as numpyro_init

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


class GPUBackend:
    """GPU-optimized MCMC backend using parallel chain execution.

    Runs all MCMC chains in parallel via ``jax.pmap``, distributing
    one chain per GPU device (or per XLA device on multi-device CPU
    configurations).  If no GPU is detected at ``run()`` time, falls
    back to the CPU backend transparently.
    """

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run NUTS sampling with parallel chain execution on GPU.

        If no GPU device is found, delegates to
        :class:`~heterodyne.optimization.cmc.backends.cpu_backend.CPUBackend`.

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
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)

        if not has_gpu:
            logger.warning("GPUBackend: no GPU detected, falling back to CPUBackend")
            from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend

            return CPUBackend().run(model, config, rng_key, init_params)

        num_gpu = sum(1 for d in devices if d.platform == "gpu")
        logger.info(
            f"GPUBackend: running {config.num_chains} chains in parallel "
            f"across {num_gpu} GPU(s) "
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
            chain_method="parallel",
            progress_bar=True,
        )

        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))

        samples = mcmc.get_samples()
        logger.info("GPUBackend: sampling complete")

        return dict(samples)
