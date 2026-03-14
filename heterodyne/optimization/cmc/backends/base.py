"""Abstract base and factory for MCMC execution backends.

Includes Consensus Monte Carlo utilities for combining posteriors from
independent MCMC shards via inverse-variance (precision) weighting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackendCapabilities:
    """Static description of what an MCMC backend can do.

    Used by the backend selection logic and resource estimation code to
    choose the best available backend at runtime without instantiating
    every candidate.

    Attributes:
        supports_sharding: True if the backend can distribute data shards
            across workers or devices.
        supports_parallel_chains: True if chains can run concurrently
            (e.g. via ``pmap`` or a worker pool).
        max_parallel_shards: Maximum number of shards the backend can
            handle simultaneously.  1 means strictly sequential.
    """

    supports_sharding: bool = False
    supports_parallel_chains: bool = True
    max_parallel_shards: int = 1


@runtime_checkable
class MCMCBackend(Protocol):
    """Protocol for MCMC execution backends.

    Each backend wraps NumPyro's MCMC machinery with a CPU execution
    strategy (sequential single-device or parallel multi-device).
    """

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run MCMC sampling and return posterior samples.

        Args:
            model: NumPyro model function (callable with no required args).
            config: CMC configuration with sampling hyperparameters.
            rng_key: JAX PRNG key for reproducibility.
            init_params: Optional initial parameter values per chain.
                Keys are parameter names; values have shape ``(num_chains,)``.

        Returns:
            Dictionary mapping parameter names to sample arrays.
            Each array has shape ``(num_samples * num_chains,)`` for
            ungrouped samples, matching NumPyro's default ``get_samples()``
            behavior.
        """
        ...


class CMCBackend(ABC):
    """Abstract base class for CMC execution backends.

    Concrete subclasses implement CPU MCMC execution strategies
    (sequential, multi-device parallel, worker-pool, etc.).
    Subclasses must override ``run``, ``get_capabilities``,
    ``validate_resources``, ``estimate_memory``, and ``cleanup``.
    """

    @abstractmethod
    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run MCMC sampling and return posterior samples.

        Args:
            model: NumPyro model function.
            config: CMC configuration with sampling hyperparameters.
            rng_key: JAX PRNG key for reproducibility.
            init_params: Optional per-chain initial parameter values.

        Returns:
            Dictionary mapping parameter names to flat sample arrays.
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Return a static description of this backend's capabilities.

        Returns:
            Frozen ``BackendCapabilities`` dataclass.
        """
        ...

    @abstractmethod
    def validate_resources(self) -> None:
        """Check that required hardware and software resources are available.

        Raises:
            RuntimeError: If a required resource (device, library, memory)
                is unavailable.
        """
        ...

    @abstractmethod
    def estimate_memory(
        self,
        n_data: int,
        n_params: int,
        n_chains: int,
    ) -> float:
        """Estimate peak memory consumption for a sampling run.

        The estimate is intentionally conservative (upper-bound) to help
        callers decide whether to proceed or reduce chain count / shard size.

        Args:
            n_data: Number of data points per shard.
            n_params: Number of model parameters.
            n_chains: Number of MCMC chains to run.

        Returns:
            Estimated peak memory in gigabytes.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources held by this backend.

        Called after sampling is complete.  Implementations should be
        idempotent (safe to call more than once).
        """
        ...


def select_backend(config: CMCConfig) -> MCMCBackend:
    """Select the appropriate MCMC backend based on available CPU devices.

    Heterodyne is CPU-only.  Selection logic:
    - Multiple CPU devices -> PjitBackend (multi-device parallel)
    - Single CPU device   -> CPUBackend (sequential chains)

    Args:
        config: CMC configuration (reserved for future backend-selection
            heuristics such as ``config.num_chains``).

    Returns:
        An instantiated backend ready for ``run()``.
    """
    from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
    from heterodyne.optimization.cmc.backends.pjit_backend import PjitBackend

    devices = jax.devices()

    if len(devices) > 1:
        logger.info(
            "Multiple CPU devices detected (%d), selecting PjitBackend for "
            "multi-device parallel execution",
            len(devices),
        )
        return PjitBackend()

    logger.info("Single CPU device, selecting CPUBackend for sequential chain execution")
    return CPUBackend()


# ---------------------------------------------------------------------------
# Consensus Monte Carlo — multi-shard posterior combination
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardPosterior:
    """Posterior summary from a single MCMC shard.

    Attributes:
        mean: Parameter means, shape ``(n_params,)``.
        covariance: Covariance matrix, shape ``(n_params, n_params)``.
        n_samples: Number of effective posterior samples in this shard.
        shard_id: Optional identifier for logging / diagnostics.
    """

    mean: np.ndarray
    covariance: np.ndarray
    n_samples: int = 0
    shard_id: int = 0


def consensus_mc(
    shard_posteriors: list[ShardPosterior],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine shard posteriors using Consensus Monte Carlo.

    Each shard's posterior is summarised by its mean and covariance.
    The combined posterior is the precision-weighted average::

        Λ_combined = Σ_k Λ_k          (sum of precisions)
        μ_combined = Λ_combined⁻¹ Σ_k Λ_k μ_k

    This is exact when the sub-posteriors are Gaussian and the prior
    factorises across shards (the "embarrassingly parallel" regime).

    Args:
        shard_posteriors: List of :class:`ShardPosterior`, one per shard.
            All must have the same dimensionality.

    Returns:
        Tuple of ``(combined_mean, combined_covariance)`` where
        ``combined_mean`` has shape ``(n_params,)`` and
        ``combined_covariance`` has shape ``(n_params, n_params)``.

    Raises:
        ValueError: If fewer than 1 shard is provided or shapes are
            inconsistent.
    """
    if len(shard_posteriors) == 0:
        raise ValueError("consensus_mc requires at least 1 shard posterior")

    n_params = shard_posteriors[0].mean.shape[0]

    # Accumulate precision-weighted mean and total precision
    precision_sum = np.zeros((n_params, n_params), dtype=np.float64)
    precision_mean_sum = np.zeros(n_params, dtype=np.float64)

    for sp in shard_posteriors:
        if sp.mean.shape[0] != n_params:
            raise ValueError(
                f"Shard {sp.shard_id} has {sp.mean.shape[0]} params, "
                f"expected {n_params}"
            )
        try:
            precision_k = np.linalg.inv(sp.covariance)
        except np.linalg.LinAlgError:
            logger.warning(
                "Shard %d has singular covariance, using pseudo-inverse",
                sp.shard_id,
            )
            precision_k = np.linalg.pinv(sp.covariance)

        precision_sum += precision_k
        precision_mean_sum += precision_k @ sp.mean

    # Invert accumulated precision to get combined covariance
    try:
        combined_cov = np.linalg.inv(precision_sum)
    except np.linalg.LinAlgError:
        logger.warning("Combined precision matrix is singular, using pseudo-inverse")
        combined_cov = np.linalg.pinv(precision_sum)

    combined_mean = combined_cov @ precision_mean_sum

    logger.info(
        "consensus_mc: combined %d shards → %d params",
        len(shard_posteriors),
        n_params,
    )
    return combined_mean, combined_cov


def robust_consensus_mc(
    shard_posteriors: list[ShardPosterior],
    *,
    outlier_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine shard posteriors with outlier-resistant weighting.

    Like :func:`consensus_mc` but first identifies and downweights
    outlier shards whose means deviate from the cross-shard median
    by more than ``outlier_sigma`` standard deviations.

    Outlier detection uses the *median absolute deviation* (MAD) of
    per-shard means for each parameter.  Shards flagged as outliers
    on *any* parameter have their precision scaled by ``1 / n_shards``
    (i.e. they contribute but don't dominate).

    Args:
        shard_posteriors: List of :class:`ShardPosterior`.
        outlier_sigma: Number of MAD-scaled deviations beyond which a
            shard is considered an outlier.  Default ``3.0``.

    Returns:
        Tuple of ``(combined_mean, combined_covariance)``.

    Raises:
        ValueError: If fewer than 2 shards are provided (need ≥ 2 for
            robust statistics) or shapes are inconsistent.
    """
    if len(shard_posteriors) < 2:
        raise ValueError(
            "robust_consensus_mc requires at least 2 shards; "
            f"got {len(shard_posteriors)}"
        )

    n_shards = len(shard_posteriors)
    n_params = shard_posteriors[0].mean.shape[0]

    # Stack shard means: shape (n_shards, n_params)
    means = np.array([sp.mean for sp in shard_posteriors])

    # Per-parameter median and MAD
    medians = np.median(means, axis=0)
    mad = np.median(np.abs(means - medians), axis=0)
    # Normalise MAD to standard-deviation scale (for Gaussian: MAD ≈ 0.6745 σ)
    mad_std = mad / 0.6745
    # Floor to avoid division by zero for perfectly agreeing shards
    mad_std = np.maximum(mad_std, 1e-12)

    # Identify outlier shards: any parameter > outlier_sigma MAD-stds from median
    deviations = np.abs(means - medians) / mad_std  # (n_shards, n_params)
    is_outlier = np.any(deviations > outlier_sigma, axis=1)  # (n_shards,)

    n_outliers = int(np.sum(is_outlier))
    if n_outliers > 0:
        outlier_ids = [
            shard_posteriors[i].shard_id for i in range(n_shards) if is_outlier[i]
        ]
        logger.warning(
            "robust_consensus_mc: %d/%d shards flagged as outliers (ids=%s)",
            n_outliers,
            n_shards,
            outlier_ids,
        )

    # Accumulate with downweighting
    precision_sum = np.zeros((n_params, n_params), dtype=np.float64)
    precision_mean_sum = np.zeros(n_params, dtype=np.float64)

    for i, sp in enumerate(shard_posteriors):
        if sp.mean.shape[0] != n_params:
            raise ValueError(
                f"Shard {sp.shard_id} has {sp.mean.shape[0]} params, "
                f"expected {n_params}"
            )
        try:
            precision_k = np.linalg.inv(sp.covariance)
        except np.linalg.LinAlgError:
            precision_k = np.linalg.pinv(sp.covariance)

        # Downweight outlier shards
        weight = 1.0 / n_shards if is_outlier[i] else 1.0

        precision_sum += weight * precision_k
        precision_mean_sum += weight * precision_k @ sp.mean

    try:
        combined_cov = np.linalg.inv(precision_sum)
    except np.linalg.LinAlgError:
        combined_cov = np.linalg.pinv(precision_sum)

    combined_mean = combined_cov @ precision_mean_sum

    logger.info(
        "robust_consensus_mc: combined %d shards (%d outliers downweighted) → %d params",
        n_shards,
        n_outliers,
        n_params,
    )
    return combined_mean, combined_cov
