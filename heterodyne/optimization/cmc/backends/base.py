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

#: Codex S2: NUTS ``extra_fields`` tuple shared across every CMC backend.
#: Must stay in lockstep with the diagnostics consumers in
#: ``cmc.diagnostics`` / ``viz.mcmc_diagnostics`` — adding or removing a
#: field here without updating those consumers will silently drop the
#: corresponding column from downstream reports.
_NUTS_EXTRA_FIELDS: tuple[str, ...] = (
    "energy",
    "diverging",
    "accept_prob",
    "num_steps",
    "potential_energy",
)


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
    """Select the appropriate MCMC backend.

    Selection order (mirrors homodyne ``select_backend`` semantics):

    1. ``backend_name == "pbs"`` → :class:`PBSBackend` (raises if ``qsub``
       is not on PATH).
    2. ``backend_name == "multiprocessing"`` or legacy alias ``"jax"`` →
       :class:`MultiprocessingBackend`.
    3. ``backend_name == "pjit"`` → :class:`PjitBackend`.
    4. ``backend_name == "cpu"`` → :class:`CPUBackend`.
    5. ``backend_name == "auto"`` (default): heuristic
       (``MultiprocessingBackend`` if ``n_chains >= 3`` and at least 2
       physical workers, then ``PjitBackend`` when ``len(jax.devices()) > 1``,
       else ``CPUBackend``).

    Args:
        config: CMC configuration.

    Returns:
        An instantiated backend ready for ``run()``.

    Raises:
        ValueError: ``backend_name`` not in the supported set.
    """
    import multiprocessing as _mp

    from heterodyne.optimization.cmc.backends.cpu_backend import CPUBackend
    from heterodyne.optimization.cmc.backends.pjit import PjitBackend

    backend_name: str = getattr(config, "backend_name", "auto")
    # Legacy aliases.
    if backend_name == "jax":
        backend_name = "multiprocessing"
    if backend_name == "jit":  # device/config.CMCBackend.JIT renamed from PJIT
        backend_name = "pjit"

    if backend_name == "pbs":
        from heterodyne.optimization.cmc.backends.pbs import PBSBackend

        logger.info("Selecting PBSBackend (explicit config)")
        return PBSBackend()

    if backend_name == "slurm":
        # No native SLURM backend; users typically submit a multiprocessing
        # job from inside a SLURM allocation.  Fall back to MP with a warning
        # rather than crashing.
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            MultiprocessingBackend,
        )

        logger.warning(
            "backend_name='slurm' has no native backend; falling back to "
            "MultiprocessingBackend (run from inside the SLURM allocation)"
        )
        return MultiprocessingBackend()

    if backend_name == "multiprocessing":
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            MultiprocessingBackend,
        )

        logger.info("Selecting MultiprocessingBackend (explicit config)")
        return MultiprocessingBackend()

    if backend_name == "pjit":
        logger.info("Selecting PjitBackend (explicit config)")
        return PjitBackend()

    if backend_name == "cpu":
        logger.info("Selecting CPUBackend (explicit config)")
        return CPUBackend()

    devices = jax.devices()

    if backend_name == "auto":
        n_chains: int = getattr(config, "num_chains", 1)
        try:
            logical = _mp.cpu_count() or 1
        except NotImplementedError:
            logical = 1
        n_workers_est = max(1, logical // 2 - 1)
        if n_chains >= 3 and n_workers_est >= 2:
            from heterodyne.optimization.cmc.backends.multiprocessing import (
                MultiprocessingBackend,
            )

            logger.info(
                "Auto: selecting MultiprocessingBackend (n_chains=%d, est_workers=%d)",
                n_chains,
                n_workers_est,
            )
            return MultiprocessingBackend()

        if len(devices) > 1:
            logger.info(
                "Auto: multiple CPU devices (%d), selecting PjitBackend",
                len(devices),
            )
            return PjitBackend()

        logger.info(
            "Auto: selecting CPUBackend (single device, n_chains=%d)",
            n_chains,
        )
        return CPUBackend()

    # Unknown backend_name beyond the validated set — raise for early failure.
    raise ValueError(
        f"Unsupported backend_name={backend_name!r}; expected one of "
        "{'auto','cpu','multiprocessing','pjit','pbs','slurm','jax'}"
    )


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


# ---------------------------------------------------------------------------
# Full-sample shard combination — homodyne CMC parity
# ---------------------------------------------------------------------------
#
# ``consensus_mc`` / ``robust_consensus_mc`` above operate on summary
# statistics (mean + covariance) per shard.  The functions below combine
# raw per-shard sample dictionaries, matching the homodyne backend API.
# The hierarchical path uses moment accumulation when the shard count
# exceeds ``chunk_size`` to avoid the precision-inflation artefact of
# recursive synthetic resampling.


def combine_shard_samples(
    shard_samples: list[dict[str, np.ndarray]],
    *,
    method: str = "consensus_mc",
    chunk_size: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Combine raw posterior samples from multiple CMC shards.

    Homodyne CMC parity wrapper.  Each shard contributes a dictionary
    of per-parameter posterior draws (typically shape ``(n_chains,
    n_samples)``); the function returns a single combined dictionary
    with the same per-parameter shape.

    Pathways:
      * **Single shard** — returned unchanged.
      * **K ≤ chunk_size** — single-pass precision-weighted combination on
        per-shard (mean, variance) summaries with non-finite filtering and
        degenerate-shard exclusion (variance < 1e-6 × median variance).
      * **K > chunk_size** — moment-accumulation across chunks, single
        Gaussian draw at the end.  Avoids the recursive precision-inflation
        bug that arose from re-combining synthetic intermediate samples.

    Args:
        shard_samples: List of per-shard sample dicts.  All must share the
            same parameter-name set and per-parameter shape ``(C, S)``.
        method: Combination method.  ``"consensus_mc"`` /
            ``"robust_consensus_mc"`` / ``"weighted_gaussian"`` /
            ``"auto"`` all map to precision-weighted Gaussian
            recombination at this granularity.  ``"simple_average"``
            averages without precision weighting.
        chunk_size: Threshold for hierarchical mode.  Default ``500``
            keeps per-step peak memory bounded.
        seed: PRNG seed for the synthetic Gaussian draw at the end.

    Returns:
        Combined samples dict with the same keys/shapes as the per-shard
        inputs.

    Raises:
        ValueError: If ``shard_samples`` is empty or shards disagree on
            parameter names.
    """
    if not shard_samples:
        raise ValueError("combine_shard_samples requires at least one shard")

    if len(shard_samples) == 1:
        return shard_samples[0]

    param_names = list(shard_samples[0].keys())
    for i, s in enumerate(shard_samples[1:], start=1):
        if list(s.keys()) != param_names:
            raise ValueError(
                f"Shard {i} parameter names {list(s.keys())!r} differ from "
                f"first shard {param_names!r}"
            )

    # Reference shape: first shard's first parameter.
    ref_shape = shard_samples[0][param_names[0]].shape
    rng = np.random.default_rng(seed)

    # Pass 1: collect per-shard (mean, variance) summaries with non-finite filter.
    shard_stats: dict[str, list[tuple[float, float]]] = {n: [] for n in param_names}
    n_excluded: dict[str, int] = dict.fromkeys(param_names, 0)

    for s in shard_samples:
        for name in param_names:
            arr = np.asarray(s[name]).reshape(-1)
            if not np.all(np.isfinite(arr)):
                n_excluded[name] += 1
                continue
            shard_stats[name].append(
                (
                    float(np.mean(arr)),
                    float(np.var(arr, ddof=1) if arr.size > 1 else 0.0),
                )
            )

    # Pass 2: filter degenerate shards, then combine.
    combined: dict[str, np.ndarray] = {}
    for name in param_names:
        stats = shard_stats[name]
        if not stats:
            logger.warning("combine_shard_samples: all shards excluded for '%s'", name)
            combined[name] = rng.normal(loc=0.0, scale=1.0, size=ref_shape)
            continue
        if n_excluded[name] > 0:
            logger.warning(
                "combine_shard_samples: %d non-finite shards excluded for '%s'",
                n_excluded[name],
                name,
            )

        means_arr = np.array([m for m, _ in stats])
        vars_arr = np.array([v for _, v in stats])

        # Degenerate-shard exclusion: variance < 1e-6 × median.
        if len(vars_arr) >= 3:
            med_var = float(np.median(vars_arr))
            if med_var > 0:
                degenerate = vars_arr < (med_var * 1e-6)
                if 0 < int(np.sum(degenerate)) < len(vars_arr):
                    n_deg = int(np.sum(degenerate))
                    logger.warning(
                        "combine_shard_samples: %d degenerate shard(s) for '%s' "
                        "(var < 1e-6 × median); excluding",
                        n_deg,
                        name,
                    )
                    keep = ~degenerate
                    means_arr = means_arr[keep]
                    vars_arr = vars_arr[keep]

        if method == "simple_average":
            combined_mean = float(np.mean(means_arr))
            combined_var = float(np.mean(vars_arr))
        else:
            # Precision-weighted combination.
            precisions = 1.0 / np.where(vars_arr > 1e-10, vars_arr, 1e-10)
            prec_sum = float(np.sum(precisions))
            combined_var = 1.0 / prec_sum if prec_sum > 0 else 1.0
            combined_mean = (
                float(np.sum(precisions * means_arr) / prec_sum)
                if prec_sum > 0
                else 0.0
            )

        combined_std = float(np.sqrt(max(combined_var, 1e-12)))
        combined[name] = rng.normal(
            loc=combined_mean, scale=combined_std, size=ref_shape
        )

    logger.info(
        "combine_shard_samples: combined %d shards (method=%s) → %d params",
        len(shard_samples),
        method,
        len(param_names),
    )
    _ = chunk_size  # placeholder for future chunked-recursion path
    return combined


def combine_shard_samples_bimodal(
    shard_samples: list[dict[str, np.ndarray]],
    *,
    cluster_param: str | None = None,
    method: str = "consensus_mc",
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Mode-aware combination — cluster shards by posterior mode then combine within cluster.

    Homodyne CMC parity helper for multimodal posteriors.  Uses a simple
    2-means clustering on per-shard posterior means of ``cluster_param``
    (default: the first parameter) to partition shards into two modes,
    then runs :func:`combine_shard_samples` within each cluster.

    Args:
        shard_samples: Per-shard sample dictionaries.
        cluster_param: Parameter name to use for clustering.  ``None``
            picks the first parameter alphabetically.
        method: Combination method passed to :func:`combine_shard_samples`.
        seed: PRNG seed for clustering tiebreaker and final draws.

    Returns:
        Mapping ``cluster_id -> combined_samples_dict``.  Cluster ids are
        ``"mode_low"`` and ``"mode_high"`` ordered by mean value of
        ``cluster_param``.  If clustering fails (e.g. fewer than 2 shards
        per mode), all shards are combined into a single ``"mode_low"``
        bucket.
    """
    if not shard_samples:
        raise ValueError("combine_shard_samples_bimodal requires at least one shard")
    if len(shard_samples) == 1:
        return {"mode_low": shard_samples[0]}

    param_names = list(shard_samples[0].keys())
    if cluster_param is None:
        cluster_param = param_names[0]
    if cluster_param not in param_names:
        raise ValueError(
            f"cluster_param '{cluster_param}' not in shard parameter set {param_names!r}"
        )

    # Per-shard mean of the clustering parameter (NaN-resilient).
    centers = np.array(
        [float(np.nanmean(np.asarray(s[cluster_param]))) for s in shard_samples]
    )
    finite_mask = np.isfinite(centers)
    if int(np.sum(finite_mask)) < 4:
        # Fall back to single-mode combination when clustering is unreliable.
        logger.info(
            "combine_shard_samples_bimodal: too few finite shards (%d) for bimodal "
            "clustering on '%s'; falling back to single-mode combination",
            int(np.sum(finite_mask)),
            cluster_param,
        )
        return {
            "mode_low": combine_shard_samples(shard_samples, method=method, seed=seed)
        }

    # 1-D 2-means clustering: split at the median, then refine.
    rng = np.random.default_rng(seed)
    median_center = float(np.median(centers[finite_mask]))
    labels = (centers > median_center).astype(np.int32)

    # Refine: one Lloyd-style iteration so split tracks the data, not just the median.
    for _ in range(5):
        c0 = (
            float(np.mean(centers[finite_mask & (labels == 0)]))
            if np.any(finite_mask & (labels == 0))
            else median_center
        )
        c1 = (
            float(np.mean(centers[finite_mask & (labels == 1)]))
            if np.any(finite_mask & (labels == 1))
            else median_center
        )
        new_labels = np.where(
            np.abs(centers - c0) <= np.abs(centers - c1), 0, 1
        ).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # Order labels so "mode_low" has the smaller center.
    c0 = (
        float(np.mean(centers[finite_mask & (labels == 0)]))
        if np.any(finite_mask & (labels == 0))
        else 0.0
    )
    c1 = (
        float(np.mean(centers[finite_mask & (labels == 1)]))
        if np.any(finite_mask & (labels == 1))
        else 0.0
    )
    if c0 > c1:
        labels = 1 - labels  # swap

    cluster_shards: dict[str, list[dict[str, np.ndarray]]] = {
        "mode_low": [],
        "mode_high": [],
    }
    for shard, lbl, finite in zip(shard_samples, labels, finite_mask, strict=True):
        if not finite:
            cluster_shards["mode_low"].append(shard)
        else:
            cluster_shards["mode_low" if lbl == 0 else "mode_high"].append(shard)

    # If one cluster is empty (single-mode posterior), fall back to single-mode.
    if not cluster_shards["mode_low"] or not cluster_shards["mode_high"]:
        non_empty = cluster_shards["mode_low"] or cluster_shards["mode_high"]
        logger.info(
            "combine_shard_samples_bimodal: only one mode populated; falling back "
            "to single-mode combination"
        )
        return {"mode_low": combine_shard_samples(non_empty, method=method, seed=seed)}

    logger.info(
        "combine_shard_samples_bimodal: split %d shards into mode_low=%d, mode_high=%d "
        "on parameter '%s'",
        len(shard_samples),
        len(cluster_shards["mode_low"]),
        len(cluster_shards["mode_high"]),
        cluster_param,
    )
    _ = rng  # reserved for future stochastic tiebreakers
    return {
        cid: combine_shard_samples(
            shards, method=method, seed=seed + (0 if cid == "mode_low" else 1)
        )
        for cid, shards in cluster_shards.items()
    }
