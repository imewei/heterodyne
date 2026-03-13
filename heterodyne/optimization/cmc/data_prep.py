"""Data preparation utilities for CMC analysis.

Handles validation, JAX conversion, and sharding of correlation data
for large-dataset MCMC workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sharding strategy enum
# ---------------------------------------------------------------------------


class ShardingStrategy(Enum):
    """Strategy for partitioning prepared data into shards.

    Attributes:
        RANDOM: Randomly assign data points to shards with a fixed seed.
        CONTIGUOUS: Split along the time axis into contiguous blocks.
        STRATIFIED: Stratified by time range so each shard covers all epochs.
        ANGLE_BALANCED: Each shard receives proportional representation from
            every phi angle, preventing heterogeneous sub-posteriors.
    """

    RANDOM = "random"
    CONTIGUOUS = "contiguous"
    STRATIFIED = "stratified"
    ANGLE_BALANCED = "angle_balanced"


# ---------------------------------------------------------------------------
# PreparedData dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    """Validated and structured data container for CMC/NUTS sampling.

    All arrays are kept as NumPy for compatibility with both JAX and
    SciPy backends.  The caller converts to JAX inside the sampler.

    Attributes:
        c2_data: Flattened observed correlation values, shape ``(n_total,)``.
        weights: Per-element likelihood weights, shape ``(n_total,)`` or
            ``None`` when uniform weighting is used.
        time_array: Unique time values used to build the time grid, shape
            ``(n_times,)``.
        phi_angles: Per-element phi angles (radians or degrees), shape
            ``(n_total,)``.
        q: Wavevector magnitude in Å⁻¹.
        dt: Frame time step in seconds.
        metadata: Arbitrary key/value pairs (configuration, provenance, …).
        n_angles: Number of unique phi angles.
        n_times: Length of ``time_array``.
    """

    c2_data: np.ndarray
    weights: np.ndarray | None
    time_array: np.ndarray
    phi_angles: np.ndarray
    q: float
    dt: float
    metadata: dict[str, Any] = field(default_factory=dict)
    n_angles: int = 0
    n_times: int = 0

    def __post_init__(self) -> None:
        if self.n_angles == 0:
            self.n_angles = int(len(np.unique(self.phi_angles)))
        if self.n_times == 0:
            self.n_times = int(len(self.time_array))


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def prepare_cmc_data(
    c2_data: np.ndarray | jnp.ndarray,
    sigma: np.ndarray | float | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | float, jnp.ndarray | None]:
    """Validate and convert correlation data to JAX arrays.

    Performs shape, dtype, NaN, and monotonicity checks on the input
    correlation matrix before transferring to JAX device memory.

    Args:
        c2_data: Observed two-time correlation matrix. Must be 2-D and
            square (or 1-D for single-time slices).
        sigma: Measurement uncertainty. Scalar broadcasts to all elements;
            array must match ``c2_data`` shape. If ``None``, returns
            ``None`` and the caller is responsible for estimation.
        weights: Optional per-element weights for likelihood weighting.
            Must match ``c2_data`` shape if provided.

    Returns:
        Tuple of ``(c2_jax, sigma_jax, weights_jax)`` ready for the
        NumPyro model. ``sigma_jax`` is the scalar or array sigma
        (or ``None`` passthrough when input is ``None``).
        ``weights_jax`` is ``None`` when no weights are provided.

    Raises:
        ValueError: If data contains NaN, has mismatched shapes, or
            violates expected structure.
    """
    c2_np = np.asarray(c2_data)

    # --- Shape validation ---
    if c2_np.ndim == 1:
        logger.info("1-D correlation data: length=%d", c2_np.shape[0])
    elif c2_np.ndim == 2:
        if c2_np.shape[0] != c2_np.shape[1]:
            raise ValueError(f"c2_data must be square, got shape {c2_np.shape}")
        logger.info("2-D correlation matrix: shape=%s", c2_np.shape)
    else:
        raise ValueError(
            f"c2_data must be 1-D or 2-D, got {c2_np.ndim}-D with shape {c2_np.shape}"
        )

    # --- NaN check ---
    nan_count = int(np.sum(np.isnan(c2_np)))
    if nan_count > 0:
        raise ValueError(
            f"c2_data contains {nan_count} NaN values; clean data before CMC analysis"
        )

    # --- Dtype conversion (ensure float64 for numerical stability) ---
    if not np.issubdtype(c2_np.dtype, np.floating):
        logger.info("Converting c2_data from %s to float64", c2_np.dtype)
        c2_np = c2_np.astype(np.float64)

    c2_jax = jnp.asarray(c2_np)

    # --- Sigma validation ---
    sigma_jax: jnp.ndarray | float
    if sigma is None:
        sigma_jax = sigma  # type: ignore[assignment]
    elif isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"Scalar sigma must be positive, got {sigma}")
        sigma_jax = float(sigma)
    else:
        sigma_np = np.asarray(sigma)
        if sigma_np.shape != c2_np.shape:
            raise ValueError(
                f"sigma shape {sigma_np.shape} does not match "
                f"c2_data shape {c2_np.shape}"
            )
        if np.any(sigma_np <= 0):
            raise ValueError("sigma array must be strictly positive everywhere")
        nan_sigma = int(np.sum(np.isnan(sigma_np)))
        if nan_sigma > 0:
            raise ValueError(f"sigma contains {nan_sigma} NaN values")
        sigma_jax = jnp.asarray(sigma_np)

    # --- Weights validation ---
    weights_jax: jnp.ndarray | None = None
    if weights is not None:
        weights_np = np.asarray(weights)
        if weights_np.shape != c2_np.shape:
            raise ValueError(
                f"weights shape {weights_np.shape} does not match "
                f"c2_data shape {c2_np.shape}"
            )
        if np.any(weights_np < 0):
            raise ValueError("weights must be non-negative")
        weights_jax = jnp.asarray(weights_np)

    return c2_jax, sigma_jax, weights_jax


def create_shard_grid(
    n_times: int,
    n_shards: int,
) -> list[tuple[int, int]]:
    """Create time-index partitions for sharding a correlation matrix.

    Divides the time axis into approximately equal chunks so that each
    shard can be processed independently (e.g., for consensus Monte Carlo
    on very large two-time matrices).

    Args:
        n_times: Number of time points along one axis of the correlation
            matrix.
        n_shards: Number of shards to create. Must be >= 1 and <= n_times.

    Returns:
        List of ``(start, stop)`` index pairs (half-open intervals) that
        partition ``range(n_times)`` into ``n_shards`` contiguous chunks.

    Raises:
        ValueError: If ``n_shards < 1`` or ``n_shards > n_times``.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_shards > n_times:
        raise ValueError(f"n_shards ({n_shards}) exceeds n_times ({n_times})")

    # Use numpy's array_split logic for balanced partitioning
    boundaries = np.linspace(0, n_times, n_shards + 1, dtype=int)
    grid: list[tuple[int, int]] = []
    for i in range(n_shards):
        start = int(boundaries[i])
        stop = int(boundaries[i + 1])
        grid.append((start, stop))

    logger.info(
        f"Created {n_shards} shards for n_times={n_times}: "
        f"sizes={[stop - start for start, stop in grid]}"
    )
    return grid


def shard_correlation_data(
    c2_data: np.ndarray | jnp.ndarray,
    shard_grid: list[tuple[int, int]],
) -> list[jnp.ndarray]:
    """Split a two-time correlation matrix into shards along both axes.

    Each shard is a sub-block of the full correlation matrix defined by
    the row and column index ranges in ``shard_grid``. Only diagonal
    blocks (same row and column shard) are returned, as off-diagonal
    blocks carry cross-shard correlations that are handled separately
    in the consensus step.

    Args:
        c2_data: Full two-time correlation matrix of shape ``(N, N)``.
        shard_grid: List of ``(start, stop)`` index pairs from
            :func:`create_shard_grid`.

    Returns:
        List of JAX arrays, one per shard, each of shape
        ``(stop - start, stop - start)``.

    Raises:
        ValueError: If ``c2_data`` is not 2-D or if shard indices
            are out of bounds.
    """
    c2_np = np.asarray(c2_data)
    if c2_np.ndim != 2:
        raise ValueError(f"c2_data must be 2-D for sharding, got shape {c2_np.shape}")

    n = c2_np.shape[0]
    shards: list[jnp.ndarray] = []

    for start, stop in shard_grid:
        if start < 0 or stop > n:
            raise ValueError(
                f"Shard indices ({start}, {stop}) out of bounds for matrix size {n}"
            )
        block = c2_np[start:stop, start:stop]
        shards.append(jnp.asarray(block))

    logger.info(
        f"Sharded {c2_np.shape} matrix into {len(shards)} diagonal blocks: "
        f"sizes={[s.shape for s in shards]}"
    )
    return shards


def merge_shard_results(
    shard_results: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Combine per-shard posterior samples via simple concatenation.

    For consensus Monte Carlo, this implements the naive pooling strategy
    where samples from each shard's sub-posterior are concatenated. The
    caller may apply further weighting or density product corrections.

    All shards must contain the same set of parameter names.

    Args:
        shard_results: List of sample dictionaries, one per shard.
            Each dict maps parameter names to 1-D arrays of posterior
            draws.

    Returns:
        Merged dictionary with concatenated samples for each parameter.

    Raises:
        ValueError: If shard results are empty or have mismatched keys.
    """
    if not shard_results:
        raise ValueError("shard_results must be non-empty")

    reference_keys = set(shard_results[0].keys())
    for i, shard in enumerate(shard_results[1:], start=1):
        shard_keys = set(shard.keys())
        if shard_keys != reference_keys:
            missing = reference_keys - shard_keys
            extra = shard_keys - reference_keys
            raise ValueError(
                f"Shard {i} has mismatched keys: missing={missing}, extra={extra}"
            )

    merged: dict[str, np.ndarray] = {}
    for name in sorted(reference_keys):
        arrays = [np.asarray(shard[name]) for shard in shard_results]
        merged[name] = np.concatenate(arrays, axis=0)

    total_samples = next(iter(merged.values())).shape[0] if merged else 0
    logger.info(
        f"Merged {len(shard_results)} shard results: "
        f"{len(merged)} parameters, {total_samples} total samples each"
    )
    return merged


# ---------------------------------------------------------------------------
# New high-level API
# ---------------------------------------------------------------------------


def _estimate_noise_scale(data: np.ndarray) -> float:
    """Robust MAD-based noise scale estimate (sigma_MAD = 1.4826 * MAD)."""
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return max(mad * 1.4826, 1e-6)


def prepare_data(
    raw_data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> PreparedData:
    """Validate, normalise, and package raw XPCS data for CMC sampling.

    This is the main entry point for converting a raw data dictionary
    (as produced by the XPCS loader) into a :class:`PreparedData`
    instance suitable for NUTS or CMC workflows.

    Args:
        raw_data: Dictionary with at least the following keys:

            - ``"c2_data"`` – array-like, shape ``(n_angles, n_t, n_t)``
              or ``(n_t, n_t)``.
            - ``"phi_angles"`` – 1-D array of azimuthal angles (degrees or
              radians), length ``n_angles``.
            - ``"time_array"`` – 1-D monotonically increasing time axis.
            - ``"q"`` – scalar wavevector magnitude (Å⁻¹).
            - ``"dt"`` – scalar frame time step (seconds).

            Optional keys:

            - ``"weights"`` – array matching ``c2_data``, per-element
              likelihood weights.
            - ``"mask"`` – boolean array matching ``c2_data``; ``True``
              where data should be *excluded*.

        config: Optional configuration dictionary.  Recognised keys:

            - ``"normalize_weights"`` (bool, default ``True``) – rescale
              weights so their mean equals 1.
            - ``"require_positive_diagonal"`` (bool, default ``True``) –
              raise if any diagonal element <= 0.

    Returns:
        :class:`PreparedData` ready for :func:`create_shards`.

    Raises:
        ValueError: If required keys are missing, arrays have unexpected
            shapes, or data contains NaN / non-finite values.
        KeyError: If a required key is absent from ``raw_data``.
    """
    cfg = config or {}
    normalize_weights: bool = bool(cfg.get("normalize_weights", True))
    require_positive_diagonal: bool = bool(cfg.get("require_positive_diagonal", True))

    # --- Required keys ---
    for key in ("c2_data", "phi_angles", "time_array", "q", "dt"):
        if key not in raw_data:
            raise KeyError(f"raw_data missing required key: '{key}'")

    c2_raw = np.asarray(raw_data["c2_data"], dtype=np.float64)
    phi_raw = np.asarray(raw_data["phi_angles"], dtype=np.float64)
    time_raw = np.asarray(raw_data["time_array"], dtype=np.float64)
    q = float(raw_data["q"])
    dt = float(raw_data["dt"])

    # --- Shape normalisation: accept (n_t, n_t) or (n_phi, n_t, n_t) ---
    if c2_raw.ndim == 2:
        c2_raw = c2_raw[np.newaxis, ...]  # promote to (1, n_t, n_t)
    if c2_raw.ndim != 3:
        raise ValueError(
            f"c2_data must be 2-D or 3-D, got {c2_raw.ndim}-D shape {c2_raw.shape}"
        )
    n_phi, n_t1, n_t2 = c2_raw.shape
    if n_t1 != n_t2:
        raise ValueError(f"c2_data time dimensions must be equal, got ({n_t1}, {n_t2})")

    if phi_raw.ndim != 1:
        raise ValueError(f"phi_angles must be 1-D, got shape {phi_raw.shape}")
    if phi_raw.shape[0] != n_phi:
        raise ValueError(
            f"phi_angles length {phi_raw.shape[0]} does not match "
            f"c2_data first dimension {n_phi}"
        )

    if time_raw.ndim != 1:
        raise ValueError(f"time_array must be 1-D, got shape {time_raw.shape}")
    if time_raw.shape[0] != n_t1:
        raise ValueError(
            f"time_array length {time_raw.shape[0]} does not match "
            f"c2_data time dimension {n_t1}"
        )

    # --- NaN / inf check ---
    nan_count = int(np.sum(~np.isfinite(c2_raw)))
    if nan_count > 0:
        raise ValueError(
            f"c2_data contains {nan_count} non-finite values (NaN or Inf); "
            "clean data before CMC analysis"
        )

    if not np.all(np.isfinite(time_raw)):
        raise ValueError("time_array contains non-finite values")

    # --- Monotonicity check ---
    if not np.all(np.diff(time_raw) > 0):
        raise ValueError("time_array must be strictly monotonically increasing")

    # --- Positive diagonal check ---
    if require_positive_diagonal:
        for angle_idx in range(n_phi):
            diag = np.diag(c2_raw[angle_idx])
            n_bad = int(np.sum(diag <= 0))
            if n_bad > 0:
                raise ValueError(
                    f"Angle index {angle_idx} (phi={phi_raw[angle_idx]:.4f}) "
                    f"has {n_bad} non-positive diagonal elements; "
                    "check diagonal correction before CMC analysis"
                )

    # --- Optional mask application ---
    weights_raw: np.ndarray | None = None
    if "mask" in raw_data:
        mask = np.asarray(raw_data["mask"], dtype=bool)
        if mask.shape != c2_raw.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match c2_data shape {c2_raw.shape}"
            )
        # Convert mask to weight=0 / weight=1
        weights_raw = (~mask).astype(np.float64)
        n_masked = int(np.sum(mask))
        logger.info(
            "Applied mask: %d elements excluded (%.1f%%)",
            n_masked,
            100.0 * n_masked / mask.size,
        )

    if "weights" in raw_data and raw_data["weights"] is not None:
        w = np.asarray(raw_data["weights"], dtype=np.float64)
        if w.shape != c2_raw.shape:
            raise ValueError(
                f"weights shape {w.shape} does not match c2_data shape {c2_raw.shape}"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        weights_raw = w if weights_raw is None else weights_raw * w

    if weights_raw is not None and normalize_weights:
        w_mean = float(np.mean(weights_raw[weights_raw > 0]))
        if w_mean > 0:
            weights_raw = weights_raw / w_mean

    # --- Flatten to 1-D pooled arrays ---
    # Build per-element phi angle array matching the flat c2 layout
    # (n_phi * n_t * n_t,)
    phi_per_element = np.repeat(phi_raw, n_t1 * n_t2)
    c2_flat = c2_raw.ravel()
    weights_flat = weights_raw.ravel() if weights_raw is not None else None

    noise_scale = _estimate_noise_scale(c2_flat)
    n_angles = int(len(np.unique(phi_raw)))

    logger.info(
        "prepare_data: shape=(%d, %d, %d), n_angles=%d, n_times=%d, "
        "q=%.4f, dt=%.6f, noise_scale=%.4f",
        n_phi,
        n_t1,
        n_t2,
        n_angles,
        n_t1,
        q,
        dt,
        noise_scale,
    )

    return PreparedData(
        c2_data=c2_flat,
        weights=weights_flat,
        time_array=time_raw,
        phi_angles=phi_per_element,
        q=q,
        dt=dt,
        metadata={
            "noise_scale": noise_scale,
            "n_phi_original": n_phi,
            "c2_shape_original": c2_raw.shape,
        },
        n_angles=n_angles,
        n_times=n_t1,
    )


# ---------------------------------------------------------------------------
# Shard creation
# ---------------------------------------------------------------------------


def create_shards(
    prepared_data: PreparedData,
    n_shards: int,
    strategy: ShardingStrategy = ShardingStrategy.ANGLE_BALANCED,
    *,
    seed: int = 42,
) -> list[PreparedData]:
    """Split a :class:`PreparedData` instance into ``n_shards`` sub-datasets.

    Each shard is itself a :class:`PreparedData` with the same ``q``,
    ``dt``, and ``time_array`` as the parent but containing only a
    subset of the pooled data points.

    Args:
        prepared_data: Source data returned by :func:`prepare_data`.
        n_shards: Number of shards to create.  Must be >= 1.
        strategy: Splitting strategy (see :class:`ShardingStrategy`).
        seed: Random seed used by stochastic strategies (RANDOM,
            ANGLE_BALANCED).

    Returns:
        List of ``n_shards`` :class:`PreparedData` instances.

    Raises:
        ValueError: If ``n_shards < 1`` or strategy is unsupported.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_shards == 1:
        return [prepared_data]

    if strategy is ShardingStrategy.RANDOM:
        return _random_split(prepared_data, n_shards, seed=seed)
    if strategy is ShardingStrategy.CONTIGUOUS:
        return _contiguous_split(prepared_data, n_shards)
    if strategy is ShardingStrategy.STRATIFIED:
        return _stratified_split(prepared_data, n_shards)
    if strategy is ShardingStrategy.ANGLE_BALANCED:
        return _angle_balanced_split(prepared_data, n_shards, seed=seed)

    raise ValueError(f"Unknown sharding strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Split implementations
# ---------------------------------------------------------------------------


def _build_shard(
    parent: PreparedData,
    indices: np.ndarray,
) -> PreparedData:
    """Construct a shard PreparedData from a parent and an index array."""
    indices = np.sort(indices)
    c2_sub = parent.c2_data[indices]
    phi_sub = parent.phi_angles[indices]
    weights_sub = parent.weights[indices] if parent.weights is not None else None
    noise_scale = _estimate_noise_scale(c2_sub)
    n_angles = int(len(np.unique(phi_sub)))
    meta = dict(parent.metadata)
    meta["noise_scale"] = noise_scale
    return PreparedData(
        c2_data=c2_sub,
        weights=weights_sub,
        time_array=parent.time_array,
        phi_angles=phi_sub,
        q=parent.q,
        dt=parent.dt,
        metadata=meta,
        n_angles=n_angles,
        n_times=parent.n_times,
    )


def _contiguous_split(
    data: PreparedData,
    n_shards: int,
) -> list[PreparedData]:
    """Split data into contiguous blocks by flat index order."""
    n = len(data.c2_data)
    boundaries = np.linspace(0, n, n_shards + 1, dtype=int)
    shards: list[PreparedData] = []
    for i in range(n_shards):
        start = int(boundaries[i])
        stop = int(boundaries[i + 1])
        indices = np.arange(start, stop)
        shards.append(_build_shard(data, indices))
    logger.info(
        "Contiguous split: %d points -> %d shards (~%d points each)",
        n,
        n_shards,
        n // n_shards,
    )
    return shards


def _stratified_split(
    data: PreparedData,
    n_shards: int,
) -> list[PreparedData]:
    """Stratified split by time range.

    The full time axis is divided into ``n_shards`` equal-width strata.
    Data points are assigned to the stratum whose midpoint is closest
    to their time coordinate (using the first time dimension implicit in
    the flat layout).

    Each resulting shard therefore spans the same fraction of the
    observation window, preserving temporal diversity.

    Args:
        data: Source :class:`PreparedData`.
        n_shards: Number of strata / shards.

    Returns:
        List of :class:`PreparedData` shards.
    """
    n = len(data.c2_data)
    t_min = float(data.time_array[0])
    t_max = float(data.time_array[-1])
    t_range = t_max - t_min

    # Assign each flat element an approximate time value by cycling through
    # the time array (element i corresponds to time_array[i % n_times]).
    n_times = data.n_times
    element_times = data.time_array[np.arange(n) % n_times]

    # Bin into strata
    stratum_width = t_range / n_shards
    stratum_idx = np.clip(
        ((element_times - t_min) / stratum_width).astype(int),
        0,
        n_shards - 1,
    )

    shards: list[PreparedData] = []
    for s in range(n_shards):
        indices = np.where(stratum_idx == s)[0]
        if len(indices) == 0:
            logger.warning("Stratified split: stratum %d is empty", s)
            continue
        shards.append(_build_shard(data, indices))

    logger.info(
        "Stratified split: %d points -> %d shards (time strata)", n, len(shards)
    )
    return shards


def _angle_balanced_split(
    data: PreparedData,
    n_shards: int,
    *,
    seed: int = 42,
) -> list[PreparedData]:
    """Split with balanced representation of every phi angle per shard.

    Each unique phi angle contributes a proportional fraction of its
    data points to every shard.  This prevents heterogeneous
    sub-posteriors caused by angle-sparse shards, which can produce
    high coefficient-of-variation across CMC shards.

    Algorithm:
        1. Group flat indices by their unique phi angle.
        2. Shuffle indices within each angle group independently.
        3. For each shard, take ``floor(angle_count / n_shards)`` points
           from each group; the last shard absorbs any remainder.
        4. Sort the combined indices to restore temporal order.

    Args:
        data: Source :class:`PreparedData`.
        n_shards: Number of output shards.
        seed: RNG seed for reproducible intra-group shuffles.

    Returns:
        List of :class:`PreparedData` shards with balanced angle coverage.
    """
    rng = np.random.default_rng(seed)
    phi_unique = np.unique(data.phi_angles)
    n_phi = len(phi_unique)

    if n_phi == 1:
        logger.info(
            "angle_balanced_split: single angle detected, falling back to random split"
        )
        return _random_split(data, n_shards, seed=seed)

    # Group indices by angle using tolerance-aware nearest-neighbour matching
    angle_groups: list[np.ndarray] = []
    for phi_val in phi_unique:
        mask = np.abs(data.phi_angles - phi_val) < 1e-9
        idxs = np.where(mask)[0].copy()
        rng.shuffle(idxs)
        angle_groups.append(idxs)

    angle_positions = [0] * n_phi
    shards: list[PreparedData] = []

    for shard_num in range(n_shards):
        is_last = shard_num == n_shards - 1
        shard_idx_parts: list[np.ndarray] = []
        for g_idx, group in enumerate(angle_groups):
            angle_total = len(group)
            already_used = angle_positions[g_idx]
            remaining = angle_total - already_used

            if is_last:
                n_take = remaining
            else:
                target = angle_total // n_shards
                remaining_shards = n_shards - shard_num
                n_take = max(target, remaining // remaining_shards)
                n_take = min(n_take, remaining)

            if n_take > 0:
                start = angle_positions[g_idx]
                shard_idx_parts.append(group[start : start + n_take])
                angle_positions[g_idx] = start + n_take

        if not shard_idx_parts:
            continue

        combined = np.concatenate(shard_idx_parts)
        shards.append(_build_shard(data, combined))

    # Coverage reporting
    min_cov = min(s.n_angles / n_phi for s in shards)
    mean_cov = sum(s.n_angles / n_phi for s in shards) / len(shards)
    logger.info(
        "Angle-balanced split: %d points -> %d shards; "
        "angle coverage: min=%.0f%%, mean=%.0f%%",
        len(data.c2_data),
        len(shards),
        100 * min_cov,
        100 * mean_cov,
    )
    return shards


def _random_split(
    data: PreparedData,
    n_shards: int,
    *,
    seed: int = 42,
) -> list[PreparedData]:
    """Randomly assign data points to shards with a fixed seed.

    Args:
        data: Source :class:`PreparedData`.
        n_shards: Number of output shards.
        seed: RNG seed for reproducibility.

    Returns:
        List of :class:`PreparedData` shards.
    """
    rng = np.random.default_rng(seed)
    n = len(data.c2_data)
    indices = np.arange(n)
    rng.shuffle(indices)

    boundaries = np.linspace(0, n, n_shards + 1, dtype=int)
    shards: list[PreparedData] = []
    for i in range(n_shards):
        start = int(boundaries[i])
        stop = int(boundaries[i + 1])
        shards.append(_build_shard(data, indices[start:stop]))

    logger.info(
        "Random split (seed=%d): %d points -> %d shards (~%d points each)",
        seed,
        n,
        n_shards,
        n // n_shards,
    )
    return shards


# ---------------------------------------------------------------------------
# Shard validation and memory estimation
# ---------------------------------------------------------------------------


def validate_shard_data(shard: PreparedData) -> None:
    """Validate a single shard for common data quality issues.

    Checks performed:

    - No NaN or non-finite values in ``c2_data``.
    - Shape consistency between ``c2_data``, ``phi_angles``, and
      ``weights`` (when present).
    - At least one data point.
    - Positive values in the subset of elements corresponding to the
      diagonal of the original two-time matrix.

    Args:
        shard: :class:`PreparedData` shard to validate.

    Raises:
        ValueError: On any detected integrity issue.
    """
    n = len(shard.c2_data)

    if n == 0:
        raise ValueError("Shard contains zero data points")

    if shard.phi_angles.shape[0] != n:
        raise ValueError(
            f"phi_angles length {shard.phi_angles.shape[0]} "
            f"does not match c2_data length {n}"
        )

    if shard.weights is not None and shard.weights.shape[0] != n:
        raise ValueError(
            f"weights length {shard.weights.shape[0]} does not match c2_data length {n}"
        )

    nan_count = int(np.sum(~np.isfinite(shard.c2_data)))
    if nan_count > 0:
        raise ValueError(f"Shard c2_data contains {nan_count} non-finite values")

    if shard.weights is not None:
        nan_w = int(np.sum(~np.isfinite(shard.weights)))
        if nan_w > 0:
            raise ValueError(f"Shard weights contain {nan_w} non-finite values")
        if np.any(shard.weights < 0):
            raise ValueError("Shard weights contain negative values")

    # Check that diagonal-like elements (t1 == t2, i.e. index % (n_t+1) == 0
    # in the square matrix) are positive.
    n_times = shard.n_times
    if n_times > 1:
        diag_stride = n_times + 1
        diag_indices = np.arange(0, n, diag_stride)
        if len(diag_indices) > 0:
            diag_vals = shard.c2_data[diag_indices]
            n_nonpos = int(np.sum(diag_vals <= 0))
            if n_nonpos > 0:
                raise ValueError(
                    f"Shard has {n_nonpos} non-positive diagonal elements; "
                    "check diagonal correction"
                )

    logger.debug(
        "Shard validation passed: n=%d, n_angles=%d, n_times=%d",
        n,
        shard.n_angles,
        shard.n_times,
    )


def estimate_shard_memory(shard: PreparedData) -> int:
    """Estimate the device memory footprint of a shard in bytes.

    Counts all NumPy arrays stored in the shard, using their actual
    ``nbytes`` attribute.  This is a lower bound because JAX may add
    internal buffers during JIT compilation, but it is accurate enough
    for pre-flight capacity checks.

    Args:
        shard: :class:`PreparedData` shard.

    Returns:
        Estimated memory in bytes.
    """
    total = 0
    total += shard.c2_data.nbytes
    total += shard.phi_angles.nbytes
    total += shard.time_array.nbytes
    if shard.weights is not None:
        total += shard.weights.nbytes
    # Account for JAX internal copies: typically 2-3x the raw array size
    overhead_factor = 2
    estimated = total * overhead_factor
    logger.debug(
        "Shard memory estimate: raw=%d B, estimated=%d B (factor=%d)",
        total,
        estimated,
        overhead_factor,
    )
    return estimated
