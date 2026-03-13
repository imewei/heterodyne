"""CMC-optimized correlation computation for heterodyne MCMC sampling.

Provides two evaluation modes:

**Element-wise path** (primary CMC path):
    Pre-computes ``searchsorted`` indices via ``ShardGrid`` ONCE per shard,
    then evaluates c2 at paired (t1[k], t2[k]) points via O(n_pairs) cumsum
    lookups — no N×N matrix allocation inside the NUTS hot loop.  This
    prevents the 80GB+ OOM that occurs when building full matrices at every
    leapfrog step.

**Meshgrid path** (fallback / legacy):
    Calls ``compute_c2_heterodyne`` from ``jax_backend`` to build the full
    N×N matrix.  Used for posterior predictive checks and diagnostics where
    memory is not the bottleneck.

Both paths use the shared ``trapezoid_cumsum`` and rate functions from
``physics_utils`` to guarantee identical physics.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.core.physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    smooth_abs,
    trapezoid_cumsum,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ShardGrid: pre-computed index structure for element-wise evaluation
# ---------------------------------------------------------------------------


class ShardGrid(NamedTuple):
    """Pre-computed shard structure for element-wise CMC evaluation.

    Created ONCE per shard (outside the NUTS hot path) via
    ``precompute_shard_grid``.  Inside NUTS, element-wise kernels
    use ``idx1``/``idx2`` to look up cumsum values at O(1) per pair.

    Attributes:
        time_grid: Sorted time array used for cumsum, shape (N_grid,).
        idx1: Index into ``time_grid`` for each pair's first time,
            shape (n_pairs,).
        idx2: Index into ``time_grid`` for each pair's second time,
            shape (n_pairs,).
        n_pairs: Number of (t1, t2) pairs in this shard.
    """

    time_grid: jnp.ndarray
    idx1: jnp.ndarray
    idx2: jnp.ndarray
    n_pairs: int


def precompute_shard_grid(
    time_grid: jnp.ndarray,
    t1_flat: jnp.ndarray,
    t2_flat: jnp.ndarray,
) -> ShardGrid:
    """Pre-compute searchsorted indices for a shard's (t1, t2) pairs.

    Maps each t1[k] and t2[k] to the nearest index in ``time_grid``
    via ``jnp.searchsorted``.  This is called ONCE per shard before
    MCMC starts — the returned ``ShardGrid`` is then passed as static
    data into the NUTS kernel.

    Args:
        time_grid: Full sorted time array, shape (N,).
        t1_flat: First time coordinate for each data point, shape (n_pairs,).
        t2_flat: Second time coordinate for each data point, shape (n_pairs,).

    Returns:
        ShardGrid with pre-computed indices.
    """
    time_grid = jnp.asarray(time_grid)
    t1_flat = jnp.asarray(t1_flat)
    t2_flat = jnp.asarray(t2_flat)

    # searchsorted gives insertion point — clip to valid range
    n_grid = time_grid.shape[0]
    idx1 = jnp.clip(jnp.searchsorted(time_grid, t1_flat), 0, n_grid - 1)
    idx2 = jnp.clip(jnp.searchsorted(time_grid, t2_flat), 0, n_grid - 1)

    return ShardGrid(
        time_grid=time_grid,
        idx1=idx1,
        idx2=idx2,
        n_pairs=int(t1_flat.shape[0]),
    )


def precompute_shard_grid_from_matrix(
    time_grid: jnp.ndarray,
    shard_start: int,
    shard_end: int,
) -> ShardGrid:
    """Create ShardGrid for a diagonal block c2[start:end, start:end].

    Generates (t1, t2) pairs for the upper triangle of the shard block,
    which is the standard layout for symmetric correlation matrices.

    Args:
        time_grid: Full sorted time array, shape (N,).
        shard_start: Start index in full time array.
        shard_end: End index in full time array.

    Returns:
        ShardGrid for the upper triangle of the shard block.
    """
    import numpy as np

    shard_size = shard_end - shard_start
    triu_i, triu_j = np.triu_indices(shard_size, k=0)
    # Map local shard indices to global time indices
    global_i = triu_i + shard_start
    global_j = triu_j + shard_start

    t1_flat = jnp.asarray(time_grid[global_i])
    t2_flat = jnp.asarray(time_grid[global_j])

    return precompute_shard_grid(time_grid, t1_flat, t2_flat)


# ---------------------------------------------------------------------------
# Element-wise correlation kernels (CMC hot path)
# ---------------------------------------------------------------------------


@jax.jit
def compute_transport_elementwise(
    shard_grid: ShardGrid,
    D0: float,
    alpha: float,
    offset: float,
    q: float,
    dt: float,
) -> jnp.ndarray:
    """Element-wise half-transport: exp(-½q²|∫J dt|) at paired indices.

    Computes the transport rate on the full ``time_grid``, builds the
    trapezoidal cumsum, then looks up integrals at pre-computed
    (idx1, idx2) pairs.  No N×N matrix is allocated.

    Args:
        shard_grid: Pre-computed ShardGrid.
        D0: Transport prefactor.
        alpha: Transport exponent.
        offset: Transport rate offset.
        q: Scattering wavevector.
        dt: Time step.

    Returns:
        Half-transport values, shape (n_pairs,).
    """
    J_rate = compute_transport_rate(shard_grid.time_grid, D0, alpha, offset)
    cumsum = trapezoid_cumsum(J_rate, dt)

    # Element-wise integral lookup: |cumsum[idx2] - cumsum[idx1]|
    integral = smooth_abs(cumsum[shard_grid.idx2] - cumsum[shard_grid.idx1])

    # half-transport: exp(-½q²∫J)
    q2_half = 0.5 * q * q
    log_ht = -q2_half * integral
    return jnp.exp(jnp.clip(log_ht, -700.0, 0.0))


@jax.jit
def compute_velocity_elementwise(
    shard_grid: ShardGrid,
    v0: float,
    beta: float,
    v_offset: float,
    dt: float,
) -> jnp.ndarray:
    """Element-wise velocity integral at paired indices.

    Returns the *signed* integral ∫v dt (no absolute value) because
    it feeds into the phase factor ``cos(q cos(φ) ∫v dt)``.

    Args:
        shard_grid: Pre-computed ShardGrid.
        v0: Velocity prefactor.
        beta: Velocity exponent.
        v_offset: Velocity offset.
        dt: Time step.

    Returns:
        Signed velocity integral values, shape (n_pairs,).
    """
    velocity = compute_velocity_rate(shard_grid.time_grid, v0, beta, v_offset)
    cumsum = trapezoid_cumsum(velocity, dt)
    return cumsum[shard_grid.idx2] - cumsum[shard_grid.idx1]


@jax.jit
def compute_c2_elementwise(
    params: jnp.ndarray,
    shard_grid: ShardGrid,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Element-wise c2 computation for CMC (no N×N matrix allocation).

    This is the heterodyne equivalent of homodyne's
    ``_compute_g1_total_with_precomputed``.  It evaluates:

        c2[k] = offset + contrast × [ref + sample + cross][k] / f²[k]

    at each pre-computed (t1[k], t2[k]) pair from the ShardGrid.

    Args:
        params: 14-parameter array in canonical order.
        shard_grid: Pre-computed ShardGrid with paired time indices.
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle (degrees).
        contrast: Speckle contrast (beta).
        offset: Baseline offset.

    Returns:
        Correlation values, shape (n_pairs,).
    """
    # Extract parameters
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset_param = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    # Half-transport at paired indices (no N×N)
    half_tr_ref = compute_transport_elementwise(
        shard_grid,
        D0_ref,
        alpha_ref,
        D_offset_ref,
        q,
        dt,
    )
    half_tr_sample = compute_transport_elementwise(
        shard_grid,
        D0_sample,
        alpha_sample,
        D_offset_sample,
        q,
        dt,
    )

    # Velocity integral at paired indices (signed, no N×N)
    v_integral = compute_velocity_elementwise(
        shard_grid,
        v0,
        beta,
        v_offset_param,
        dt,
    )

    # Phase factor
    total_phi = phi_angle + phi0
    phi_rad = jnp.deg2rad(total_phi)
    phase = q * jnp.cos(phi_rad) * v_integral

    # Sample fraction at paired time points
    t1_vals = shard_grid.time_grid[shard_grid.idx1]
    t2_vals = shard_grid.time_grid[shard_grid.idx2]

    def _fraction(t_vals: jnp.ndarray) -> jnp.ndarray:
        exponent = jnp.clip(f1 * (t_vals - f2), -100, 100)
        f_s = jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)
        return f_s

    f_sample_1 = _fraction(t1_vals)
    f_sample_2 = _fraction(t2_vals)
    f_ref_1 = 1.0 - f_sample_1
    f_ref_2 = 1.0 - f_sample_2

    # Fraction products (element-wise)
    f_ref_prod = f_ref_1 * f_ref_2
    f_sample_prod = f_sample_1 * f_sample_2
    f_cross_1 = f_ref_1 * f_sample_1
    f_cross_2 = f_ref_2 * f_sample_2
    f_cross_prod = f_cross_1 * f_cross_2

    # Correlation terms
    ref_term = f_ref_prod**2 * half_tr_ref**2
    sample_term = f_sample_prod**2 * half_tr_sample**2
    cross_term = 2.0 * f_cross_prod * half_tr_ref * half_tr_sample * jnp.cos(phase)

    # Normalization: (f_s² + f_r²)_t1 × (f_s² + f_r²)_t2
    norm_1 = f_sample_1**2 + f_ref_1**2
    norm_2 = f_sample_2**2 + f_ref_2**2
    normalization = jnp.maximum(norm_1 * norm_2, 1e-10)

    c2 = offset + contrast * (ref_term + sample_term + cross_term) / normalization
    return c2


# ---------------------------------------------------------------------------
# Log-likelihood functions (element-wise and meshgrid variants)
# ---------------------------------------------------------------------------


@jax.jit
def compute_log_likelihood(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    contrast: float = 0.5,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute Gaussian log-likelihood via meshgrid path (full N×N).

    Suitable for small matrices or diagnostics.  For large matrices
    inside NUTS, use ``compute_log_likelihood_elementwise`` instead.

    Args:
        params: Parameter array, shape (14,)
        t: Time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        c2_data: Observed correlation matrix, shape (N, N)
        sigma: Measurement uncertainty (scalar or shape (N, N))
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Scalar log-likelihood
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    residuals = (c2_model - c2_data) / sigma
    return -0.5 * jnp.sum(residuals**2)  # type: ignore[no-any-return]


@jax.jit
def compute_log_likelihood_elementwise(
    params: jnp.ndarray,
    shard_grid: ShardGrid,
    c2_data_flat: jnp.ndarray,
    sigma_flat: jnp.ndarray | float,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute Gaussian log-likelihood via element-wise path (CMC primary).

    Uses pre-computed ``ShardGrid`` to evaluate c2 at paired indices
    without allocating an N×N matrix.

    Args:
        params: Parameter array, shape (14,).
        shard_grid: Pre-computed ShardGrid.
        c2_data_flat: Observed c2 values at shard pairs, shape (n_pairs,).
        sigma_flat: Uncertainty at shard pairs (scalar or shape (n_pairs,)).
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle (degrees).
        contrast: Speckle contrast.
        offset: Baseline offset.

    Returns:
        Scalar log-likelihood.
    """
    c2_model = compute_c2_elementwise(
        params,
        shard_grid,
        q,
        dt,
        phi_angle,
        contrast,
        offset,
    )
    residuals = (c2_model - c2_data_flat) / sigma_flat
    return -0.5 * jnp.sum(residuals**2)  # type: ignore[no-any-return]


@partial(jax.jit, static_argnames=("shard_start", "shard_end"))
def compute_shard_log_likelihood(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_shard: jnp.ndarray,
    sigma_shard: jnp.ndarray | float,
    contrast: float,
    offset: float,
    shard_start: int,
    shard_end: int,
) -> jnp.ndarray:
    """Compute log-likelihood for a diagonal shard (meshgrid path, legacy).

    A shard is a sub-block c2[start:end, start:end] — a square slice
    along the diagonal. This enables memory-efficient evaluation by
    processing shards independently.

    .. deprecated::
        Use ``compute_log_likelihood_elementwise`` with a ``ShardGrid``
        for production CMC.  This meshgrid variant builds the **full N×N**
        correlation matrix and then slices, defeating the purpose of
        sharding.  For N > 2000, this can cause OOM (e.g. N=10000 → 800 MB
        per evaluation).  Use the element-wise path instead.

    Args:
        params: Parameter array, shape (14,)
        t: Full time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_shard: Observed shard, shape (shard_size, shard_size)
        sigma_shard: Uncertainty for shard
        contrast: Speckle contrast
        offset: Baseline offset
        shard_start: Start index in full time array (static)
        shard_end: End index in full time array (static)

    Returns:
        Scalar log-likelihood for this shard
    """
    import warnings

    warnings.warn(
        "compute_shard_log_likelihood builds the full N×N matrix and is "
        "deprecated. Use compute_log_likelihood_elementwise with a "
        "ShardGrid instead to avoid OOM on large datasets.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_times = t.shape[0]
    peak_mb = n_times * n_times * 8 / (1024 * 1024)
    if n_times > 2000:
        logger.warning(
            "compute_shard_log_likelihood: N=%d will allocate ~%.0f MB "
            "for the full correlation matrix. Consider switching to "
            "compute_log_likelihood_elementwise.",
            n_times,
            peak_mb,
        )
    c2_full = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    c2_model_shard = jax.lax.dynamic_slice(
        c2_full,
        (shard_start, shard_start),
        (shard_end - shard_start, shard_end - shard_start),
    )
    residuals = (c2_model_shard - c2_shard) / sigma_shard
    return -0.5 * jnp.sum(residuals**2)  # type: ignore[no-any-return]


def compute_sharded_log_likelihood(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    shards: list[tuple[int, int, jnp.ndarray, jnp.ndarray | float]],
    contrast: float = 0.5,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Sum log-likelihoods across all shards (meshgrid path, legacy).

    .. deprecated::
        Use ``compute_sharded_log_likelihood_elementwise`` for
        production CMC to avoid N×N allocation per shard.

    Args:
        params: Parameter array, shape (14,)
        t: Full time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        shards: List of (start, end, c2_shard, sigma_shard) tuples
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Total scalar log-likelihood
    """
    import warnings

    warnings.warn(
        "compute_sharded_log_likelihood is deprecated. Use "
        "compute_sharded_log_likelihood_elementwise instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    total = jnp.float64(0.0)
    for start, end, c2_shard, sigma_shard in shards:
        ll = compute_shard_log_likelihood(
            params,
            t,
            q,
            dt,
            phi_angle,
            c2_shard,
            sigma_shard,
            contrast,
            offset,
            shard_start=start,
            shard_end=end,
        )
        total = total + ll
    return total


def compute_sharded_log_likelihood_elementwise(
    params: jnp.ndarray,
    shard_grids: list[ShardGrid],
    c2_data_flats: list[jnp.ndarray],
    sigma_flats: list[jnp.ndarray | float],
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Sum log-likelihoods across shards via element-wise path (production).

    Each shard has a pre-computed ``ShardGrid`` and flattened data/sigma
    arrays.  No N×N matrix is allocated at any point.

    Args:
        params: Parameter array, shape (14,).
        shard_grids: List of pre-computed ShardGrids, one per shard.
        c2_data_flats: List of c2 data at shard pairs, each (n_pairs_k,).
        sigma_flats: List of sigma at shard pairs.
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle (degrees).
        contrast: Speckle contrast.
        offset: Baseline offset.

    Returns:
        Total scalar log-likelihood.
    """
    total = jnp.float64(0.0)
    for sg, c2_flat, sigma_flat in zip(
        shard_grids,
        c2_data_flats,
        sigma_flats,
        strict=True,
    ):
        ll = compute_log_likelihood_elementwise(
            params,
            sg,
            c2_flat,
            sigma_flat,
            q,
            dt,
            phi_angle,
            contrast,
            offset,
        )
        total = total + ll
    return total


# ---------------------------------------------------------------------------
# Posterior predictive and diagnostics (meshgrid path is fine here)
# ---------------------------------------------------------------------------


@jax.jit
def compute_posterior_predictive(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    sigma: jnp.ndarray | float,
    contrast: float = 0.5,
    offset: float = 1.0,
    rng_key: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Generate a posterior predictive sample (meshgrid path).

    Computes the model prediction and adds Gaussian noise with the
    given sigma, suitable for posterior predictive checks.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        sigma: Noise level
        contrast: Speckle contrast
        offset: Baseline offset
        rng_key: JAX random key (if None, returns noiseless prediction)

    Returns:
        Predicted correlation matrix, shape (N, N)
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    if rng_key is not None:
        noise = jax.random.normal(rng_key, shape=c2_model.shape) * sigma
        return c2_model + noise
    return c2_model


# ---------------------------------------------------------------------------
# Shard grid utilities
# ---------------------------------------------------------------------------


def create_shard_grid(
    n_times: int,
    n_shards: int,
) -> list[tuple[int, int]]:
    """Partition the time axis into balanced shard intervals.

    Args:
        n_times: Total number of time points
        n_shards: Number of shards to create

    Returns:
        List of (start, end) index tuples
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if n_shards > n_times:
        n_shards = n_times

    boundaries = [int(round(i * n_times / n_shards)) for i in range(n_shards + 1)]
    grid = []
    for i in range(n_shards):
        if boundaries[i] < boundaries[i + 1]:
            grid.append((boundaries[i], boundaries[i + 1]))
    return grid


def prepare_shards(
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    shard_grid: list[tuple[int, int]],
) -> list[tuple[int, int, jnp.ndarray, jnp.ndarray | float]]:
    """Extract diagonal shards from full correlation data (meshgrid path).

    Args:
        c2_data: Full correlation matrix, shape (N, N)
        sigma: Uncertainty (scalar or shape (N, N))
        shard_grid: List of (start, end) intervals

    Returns:
        List of (start, end, c2_shard, sigma_shard) tuples
    """
    shards = []
    for start, end in shard_grid:
        c2_shard = c2_data[start:end, start:end]
        if isinstance(sigma, (int, float)):
            sigma_shard: jnp.ndarray | float = sigma
        else:
            sigma_shard = sigma[start:end, start:end]
        shards.append((start, end, c2_shard, sigma_shard))
    return shards


def prepare_shards_elementwise(
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    time_grid: jnp.ndarray,
    shard_intervals: list[tuple[int, int]],
) -> tuple[list[ShardGrid], list[jnp.ndarray], list[jnp.ndarray | float]]:
    """Prepare element-wise shards with pre-computed ShardGrids.

    Extracts upper-triangle elements from each diagonal shard block
    and creates the corresponding ShardGrid for element-wise evaluation.

    Args:
        c2_data: Full correlation matrix, shape (N, N).
        sigma: Uncertainty (scalar or shape (N, N)).
        time_grid: Full sorted time array, shape (N,).
        shard_intervals: List of (start, end) index tuples.

    Returns:
        Tuple of (shard_grids, c2_data_flats, sigma_flats).
    """
    import numpy as np

    shard_grids = []
    c2_flats = []
    sigma_flats: list[jnp.ndarray | float] = []

    for start, end in shard_intervals:
        sg = precompute_shard_grid_from_matrix(time_grid, start, end)
        shard_grids.append(sg)

        # Extract upper-triangle data matching the ShardGrid pairs
        shard_size = end - start
        triu_i, triu_j = np.triu_indices(shard_size, k=0)
        global_i = triu_i + start
        global_j = triu_j + start
        c2_flats.append(jnp.asarray(c2_data[global_i, global_j]))

        if isinstance(sigma, (int, float)):
            sigma_flats.append(sigma)
        else:
            sigma_flats.append(jnp.asarray(sigma[global_i, global_j]))

    return shard_grids, c2_flats, sigma_flats
