"""NumPyro model definition for heterodyne Bayesian inference."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_INDICES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_c2_heterodyne_pooled,
)
from heterodyne.core.physics_cmc import ShardGrid, compute_c2_elementwise
from heterodyne.optimization.cmc.reparameterization import (
    ReparamConfig,
    reparam_to_physics_jax,
)
from heterodyne.optimization.cmc.scaling import ParameterScaling, smooth_bound
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.config.parameter_space import ParameterSpace
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def _heterodyne_sample_shared_physics(space: ParameterSpace) -> jnp.ndarray:
    """Sample the 14 heterodyne physics parameters (shared across angles).

    Returns the full ``(14,)`` parameter vector with sampled values for
    ``space.varying_names`` and fixed values from ``space`` for the rest.
    ``contrast`` and ``offset`` are skipped here — caller supplies them
    via ``contrast_arr`` / ``offset_arr`` (per-angle).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    params = jnp.asarray(fixed_values)
    for i, name in enumerate(ALL_PARAM_NAMES):
        if name in ("contrast", "offset"):
            continue
        if name in varying_names:
            prior = space.priors[name]
            param = numpyro.sample(name, prior.to_numpyro(name))
            params = params.at[i].set(param)
    return params


def _heterodyne_pooled_likelihood(
    params: jnp.ndarray,
    contrast_arr: jnp.ndarray,
    offset_arr: jnp.ndarray,
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    num_shards: int,
) -> None:
    """Shared physics → boundary mask → likelihood (joint multi-phi).

    Phase 4 of the joint multi-phi refactor: calls
    :func:`compute_c2_heterodyne_pooled` directly to obtain the
    ``(n_total,)`` c2 vector at the pooled ``(phi, t1, t2)`` points without
    ever materializing the ``(n_phi, N, N)`` stack that the older
    vmap+gather path required. All 4 joint variants (scaled, constant,
    averaged, constant_averaged) delegate the last stages to this helper.
    """
    c2_per_point = compute_c2_heterodyne_pooled(
        params,
        t,
        q,
        dt,
        i1_indices,
        i2_indices,
        phi_indices,
        phi_unique,
        contrast_arr,
        offset_arr,
    )

    n_nan = jnp.sum(~jnp.isfinite(c2_per_point))
    numpyro.deterministic("n_numerical_issues", n_nan)

    sigma_scale = float(noise_scale) * 1.5 * math.sqrt(num_shards)
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    boundary_mask = (i1_indices > 0) & (i2_indices > 0)
    with numpyro.handlers.mask(mask=boundary_mask):
        numpyro.sample("obs", dist.Normal(c2_per_point, sigma), obs=data)


def xpcs_model_heterodyne_scaled(
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    num_shards: int = 1,
) -> None:
    """Joint multi-phi heterodyne CMC model (homodyne parity).

    Mirrors ``homodyne.optimization.cmc.model.xpcs_model_scaled``: ONE NUTS
    pass over pooled multi-phi data with shared 14 physics parameters and
    per-angle sampled contrast / offset. The likelihood site evaluates the
    Normal log-prob at every pooled point in a single ``numpyro.sample``
    call, gather-by-phi-index.

    Parameters
    ----------
    data:
        Pooled C2 values, shape ``(n_total,)`` after diagonal filtering.
    t:
        Unique time grid, shape ``(N,)``. Used by ``compute_c2_heterodyne``.
    q, dt:
        Physics scalars.
    phi_unique:
        Sorted unique phi angles, shape ``(n_phi,)``.
    phi_indices:
        Per-point index into ``phi_unique``, shape ``(n_total,)``.
    i1_indices, i2_indices:
        Per-point indices into ``t`` for the two time coordinates, shape
        ``(n_total,)`` each. Pre-computed via ``np.searchsorted(t, t1)`` /
        ``np.searchsorted(t, t2)``.
    noise_scale:
        Data-driven sigma prior centre (homodyne-parity ``HalfNormal``
        scale = ``noise_scale * 1.5 * sqrt(num_shards)``).
    space:
        Parameter space holding priors and initial values for the 14
        physics parameters + 2 scaling.
    num_shards:
        Shard count for CMC sigma-prior tempering (Scott et al. 2016).
        Default ``1`` (no tempering).
    """
    n_phi = int(phi_unique.shape[0])
    contrast_prior = space.priors["contrast"]
    offset_prior = space.priors["offset"]

    contrast_list = [
        numpyro.sample(f"contrast_{i}", contrast_prior.to_numpyro(f"contrast_{i}"))
        for i in range(n_phi)
    ]
    offset_list = [
        numpyro.sample(f"offset_{i}", offset_prior.to_numpyro(f"offset_{i}"))
        for i in range(n_phi)
    ]
    contrast_arr = jnp.stack(contrast_list)
    offset_arr = jnp.stack(offset_list)

    params = _heterodyne_sample_shared_physics(space)
    _heterodyne_pooled_likelihood(
        params,
        contrast_arr,
        offset_arr,
        data,
        t,
        q,
        dt,
        phi_unique,
        phi_indices,
        i1_indices,
        i2_indices,
        noise_scale,
        num_shards,
    )


def xpcs_model_heterodyne_constant(
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    fixed_contrast: jnp.ndarray,
    fixed_offset: jnp.ndarray,
    num_shards: int = 1,
) -> None:
    """Joint multi-phi CMC model with FIXED per-angle scaling.

    Mirrors ``homodyne.optimization.cmc.model.xpcs_model_constant``. Per-angle
    ``contrast`` and ``offset`` are passed in as arrays (length ``n_phi``,
    typically derived from quantile estimation on the raw data) and NOT
    sampled — only the 14 physics params + sigma are sampled.
    """
    contrast_arr = jnp.asarray(fixed_contrast, dtype=jnp.float64)
    offset_arr = jnp.asarray(fixed_offset, dtype=jnp.float64)
    params = _heterodyne_sample_shared_physics(space)
    _heterodyne_pooled_likelihood(
        params,
        contrast_arr,
        offset_arr,
        data,
        t,
        q,
        dt,
        phi_unique,
        phi_indices,
        i1_indices,
        i2_indices,
        noise_scale,
        num_shards,
    )


def xpcs_model_heterodyne_averaged(
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    num_shards: int = 1,
) -> None:
    """Joint multi-phi CMC model with SAMPLED averaged (single) scaling.

    Mirrors ``homodyne.optimization.cmc.model.xpcs_model_averaged``. A single
    ``contrast`` and a single ``offset`` are sampled and broadcast across
    all ``n_phi`` angles (cf. heterodyne ``per_angle_mode="auto"`` when the
    auto-resolver promotes to averaged scaling).
    """
    n_phi = int(phi_unique.shape[0])
    contrast = numpyro.sample(
        "contrast", space.priors["contrast"].to_numpyro("contrast")
    )
    offset = numpyro.sample("offset", space.priors["offset"].to_numpyro("offset"))
    contrast_arr = jnp.full((n_phi,), contrast)
    offset_arr = jnp.full((n_phi,), offset)
    params = _heterodyne_sample_shared_physics(space)
    _heterodyne_pooled_likelihood(
        params,
        contrast_arr,
        offset_arr,
        data,
        t,
        q,
        dt,
        phi_unique,
        phi_indices,
        i1_indices,
        i2_indices,
        noise_scale,
        num_shards,
    )


def xpcs_model_heterodyne_constant_averaged(
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    fixed_contrast: float,
    fixed_offset: float,
    num_shards: int = 1,
) -> None:
    """Joint multi-phi CMC model with FIXED averaged (single) scaling.

    Mirrors ``homodyne.optimization.cmc.model.xpcs_model_constant_averaged``.
    A single ``contrast`` and ``offset`` (typically the mean of the NLSQ
    per-angle estimates) are broadcast across all ``n_phi`` angles. No
    scaling parameters are sampled — only the 14 physics params + sigma.
    """
    n_phi = int(phi_unique.shape[0])
    contrast_arr = jnp.full((n_phi,), float(fixed_contrast))
    offset_arr = jnp.full((n_phi,), float(fixed_offset))
    params = _heterodyne_sample_shared_physics(space)
    _heterodyne_pooled_likelihood(
        params,
        contrast_arr,
        offset_arr,
        data,
        t,
        q,
        dt,
        phi_unique,
        phi_indices,
        i1_indices,
        i2_indices,
        noise_scale,
        num_shards,
    )


def get_heterodyne_pooled_model_for_mode(
    per_angle_mode: str,
    *,
    data: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    i1_indices: jnp.ndarray,
    i2_indices: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    fixed_contrast: np.ndarray | jnp.ndarray | float | None = None,
    fixed_offset: np.ndarray | jnp.ndarray | float | None = None,
    num_shards: int = 1,
) -> Callable[[], None]:
    """Dispatch to the joint multi-phi model variant for ``per_angle_mode``.

    Mirrors ``homodyne.optimization.cmc.model.get_xpcs_model`` at the
    pooled-data layer. Returns a zero-arg callable suitable for passing to
    ``NUTS``.

    Modes:

    - ``"individual"`` / ``"scaled"`` → :func:`xpcs_model_heterodyne_scaled`
      (per-angle sampled contrast/offset).
    - ``"constant"`` → :func:`xpcs_model_heterodyne_constant`. Requires
      ``fixed_contrast`` and ``fixed_offset`` as length-``n_phi`` arrays.
    - ``"auto"`` / ``"averaged"`` → :func:`xpcs_model_heterodyne_averaged`
      (single sampled averaged contrast/offset).
    - ``"constant_averaged"`` → :func:`xpcs_model_heterodyne_constant_averaged`.
      Requires scalar ``fixed_contrast`` and ``fixed_offset``.
    """
    if per_angle_mode in ("individual", "scaled"):
        return lambda: xpcs_model_heterodyne_scaled(
            data=data,
            t=t,
            q=q,
            dt=dt,
            phi_unique=phi_unique,
            phi_indices=phi_indices,
            i1_indices=i1_indices,
            i2_indices=i2_indices,
            noise_scale=noise_scale,
            space=space,
            num_shards=num_shards,
        )
    if per_angle_mode == "constant":
        if fixed_contrast is None or fixed_offset is None:
            raise ValueError(
                "per_angle_mode='constant' requires fixed_contrast and "
                "fixed_offset arrays of length n_phi."
            )
        fc = jnp.asarray(fixed_contrast, dtype=jnp.float64)
        fo = jnp.asarray(fixed_offset, dtype=jnp.float64)
        return lambda: xpcs_model_heterodyne_constant(
            data=data,
            t=t,
            q=q,
            dt=dt,
            phi_unique=phi_unique,
            phi_indices=phi_indices,
            i1_indices=i1_indices,
            i2_indices=i2_indices,
            noise_scale=noise_scale,
            space=space,
            fixed_contrast=fc,
            fixed_offset=fo,
            num_shards=num_shards,
        )
    if per_angle_mode in ("auto", "averaged"):
        return lambda: xpcs_model_heterodyne_averaged(
            data=data,
            t=t,
            q=q,
            dt=dt,
            phi_unique=phi_unique,
            phi_indices=phi_indices,
            i1_indices=i1_indices,
            i2_indices=i2_indices,
            noise_scale=noise_scale,
            space=space,
            num_shards=num_shards,
        )
    if per_angle_mode == "constant_averaged":
        if fixed_contrast is None or fixed_offset is None:
            raise ValueError(
                "per_angle_mode='constant_averaged' requires scalar "
                "fixed_contrast and fixed_offset."
            )
        return lambda: xpcs_model_heterodyne_constant_averaged(
            data=data,
            t=t,
            q=q,
            dt=dt,
            phi_unique=phi_unique,
            phi_indices=phi_indices,
            i1_indices=i1_indices,
            i2_indices=i2_indices,
            noise_scale=noise_scale,
            space=space,
            fixed_contrast=float(np.asarray(fixed_contrast).mean()),
            fixed_offset=float(np.asarray(fixed_offset).mean()),
            num_shards=num_shards,
        )
    raise ValueError(
        f"Unknown per_angle_mode {per_angle_mode!r}; expected one of "
        "{'individual','scaled','constant','auto','averaged',"
        "'constant_averaged'}"
    )


def _likelihood_boundary_mask(
    c2_data: jnp.ndarray, shard_grid: ShardGrid | None
) -> jnp.ndarray:
    """Boolean mask: True where (t1, t2) is NOT on the t=0 row or column.

    Mirrors the NLSQ-side mask in ``heterodyne.core.jax_backend`` so the
    Bayesian likelihood also honors the t=0 boundary contract — t=0 is
    loaded and plotted but excluded from the likelihood. The mask works
    for both the meshgrid path (``c2_data`` shape ``(N, N)``) and the
    element-wise sharded path (``c2_data`` shape ``(n_pairs,)``).
    """
    if shard_grid is not None:
        return (shard_grid.idx1 > 0) & (shard_grid.idx2 > 0)
    n_time = c2_data.shape[-1]
    indices = jnp.arange(n_time)
    return (indices[:, None] > 0) & (indices[None, :] > 0)


def get_heterodyne_model(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    contrast: float = 1.0,
    offset: float = 1.0,
    shard_grid: ShardGrid | None = None,
    priors_override: dict | None = None,
    num_shards: int = 1,
):
    """Create NumPyro model for heterodyne correlation fitting.

    Sigma is sampled as a posterior variable via ``HalfNormal(noise_scale *
    1.5 * sqrt(num_shards))``, matching the homodyne parity convention so the
    posterior captures noise uncertainty.

    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data — shape ``(N, N)`` for meshgrid
            path, or ``(n_pairs,)`` for element-wise path.
        noise_scale: Data-driven prior center for the measurement-uncertainty
            ``sigma`` posterior.  Typically the mean / RMS of an external
            estimate from :func:`estimate_sigma`.
        space: Parameter space with priors
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0
        shard_grid: Optional pre-computed ShardGrid.  When provided, uses
            the memory-efficient element-wise path (no N×N allocation).
            ``c2_data`` must then be flattened to match the shard grid's
            paired indices.
        priors_override: Optional dictionary mapping parameter names to
            NumPyro distributions.  When provided, overrides the default
            ``space.priors[name]`` for any matching parameter name.  Used
            by ``fit_cmc_sharded`` to inject tempered priors.
        num_shards: Number of CMC shards for sigma prior tempering.  Widens
            the ``HalfNormal`` scale by ``sqrt(num_shards)`` so that the
            product across shards stays equivalent to the unsharded prior.
            Defaults to ``1`` (no tempering).

    Returns:
        NumPyro model function
    """
    # Pre-compute indices and masks
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    prior_scale = math.sqrt(num_shards)
    sigma_scale = float(noise_scale) * 1.5 * prior_scale

    def model():
        """NumPyro model for heterodyne correlation."""
        # Sample varying parameters and scatter into fixed array
        # Using .at[].set() instead of jnp.array([...]) to avoid
        # tracing issues with mixed tracer/concrete values.
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                if priors_override is not None and name in priors_override:
                    param = numpyro.sample(name, priors_override[name])
                else:
                    prior = space.priors[name]
                    param = numpyro.sample(name, prior.to_numpyro(name))
                params = params.at[i].set(param)

        # Compute model prediction — dispatch to appropriate path
        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params,
                shard_grid,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params,
                t,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )

        # Track NaN/inf so callers can flag pathological shards.
        n_nan = jnp.sum(~jnp.isfinite(c2_model))
        numpyro.deterministic("n_numerical_issues", n_nan)

        # Sample sigma with prior tempered for CMC sharding (parity with homodyne).
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

        # Likelihood (t=0 boundary excluded via mask, per the heterodyne
        # contract: load and plot full N×N, exclude t=0 row/col from
        # fitting only).
        with numpyro.handlers.mask(mask=_likelihood_boundary_mask(c2_data, shard_grid)):
            numpyro.sample(
                "obs",
                dist.Normal(c2_model, sigma),
                obs=c2_data,
            )

    return model


def get_heterodyne_model_reparam(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    nlsq_params: jnp.ndarray | None = None,
    reparam_config: ReparamConfig | None = None,
    scalings: dict[str, ParameterScaling] | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
    shard_grid: ShardGrid | None = None,
    num_shards: int = 1,
):
    """Create NumPyro model with reparameterization for better sampling.

    When NLSQ result and reparameterization config are provided, uses:
    1. Reference-time reparameterization for power-law pairs
    2. Smooth bounded transforms (tanh) instead of jnp.clip()
    3. NLSQ-informed priors with delta-method uncertainty propagation

    Falls back to the original clip-based behavior when the new
    infrastructure is not provided (backward compatibility).

    Sigma is sampled internally via ``HalfNormal(noise_scale * 1.5 *
    sqrt(num_shards))`` to match homodyne CMC parity.

    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data
        noise_scale: Data-driven prior center for sampled ``sigma``.
        space: Parameter space
        nlsq_params: Optional NLSQ fitted values for centering (legacy path)
        reparam_config: Reparameterization config (enables new path)
        scalings: Pre-computed ParameterScaling per reparam-space param
        num_shards: CMC shard count for sigma prior tempering. Default ``1``.

    Returns:
        NumPyro model function
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    prior_scale = math.sqrt(num_shards)
    sigma_scale = float(noise_scale) * 1.5 * prior_scale

    # --- New reparameterized path ---
    if reparam_config is not None and scalings is not None:
        return _build_reparam_model(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            sigma_scale=sigma_scale,
            space=space,
            fixed_values=jnp.asarray(fixed_values),
            varying_names=varying_names,
            reparam_config=reparam_config,
            scalings=scalings,
            contrast=contrast,
            offset=offset,
            shard_grid=shard_grid,
        )

    # --- Legacy clip-based path (backward compatibility) ---
    if nlsq_params is not None:
        prior_centers = {
            name: float(nlsq_params[ALL_PARAM_NAMES.index(name)])
            for name in varying_names
        }
    else:
        prior_centers = {name: space.values[name] for name in varying_names}

    def model():
        """NumPyro model with centered parameterization (legacy)."""
        # Using .at[].set() instead of jnp.array([...]) to avoid
        # tracing issues with mixed tracer/concrete values.
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                center = prior_centers[name]
                bounds = space.bounds[name]
                scale = (bounds[1] - bounds[0]) / 6.0

                raw = numpyro.sample(f"{name}_raw", dist.Normal(center, scale))
                # NOTE: jnp.clip has discontinuous gradient at bounds.
                # The reparameterized path uses smooth_bound() instead.
                param = jnp.clip(raw, bounds[0], bounds[1])
                numpyro.deterministic(name, param)
                params = params.at[i].set(param)
        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params,
                shard_grid,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params,
                t,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        n_nan = jnp.sum(~jnp.isfinite(c2_model))
        numpyro.deterministic("n_numerical_issues", n_nan)
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))
        with numpyro.handlers.mask(mask=_likelihood_boundary_mask(c2_data, shard_grid)):
            numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def _build_reparam_model(
    *,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma_scale: float,
    space: ParameterSpace,
    fixed_values: jnp.ndarray,
    varying_names: list[str],
    reparam_config: ReparamConfig,
    scalings: dict[str, ParameterScaling],
    contrast: float = 1.0,
    offset: float = 1.0,
    shard_grid: ShardGrid | None = None,
):
    """Build NumPyro model using reference-time reparameterization + smooth bounds."""
    # Pre-compute which sampling-space names map to which physics params
    # Build lookup: sampling_name -> (scaling, is_reparam_log, pair_info)
    enabled_pairs = reparam_config.enabled_pairs
    t_ref = reparam_config.t_ref

    # Map prefactor names to their reparam log-space names
    prefactor_to_log: dict[str, str] = {}
    log_to_prefactor: dict[str, str] = {}
    log_to_exponent: dict[str, str] = {}
    for prefactor, exponent in enabled_pairs:
        if prefactor in varying_names and exponent in varying_names:
            log_name = reparam_config.get_reparam_name(prefactor)
            prefactor_to_log[prefactor] = log_name
            log_to_prefactor[log_name] = prefactor
            log_to_exponent[log_name] = exponent

    # Determine sampling-space parameter names (in order for the model)
    sampling_names: list[str] = []
    for name in varying_names:
        if name in prefactor_to_log:
            sampling_names.append(prefactor_to_log[name])
        else:
            sampling_names.append(name)

    def model():
        """NumPyro model with reference-time reparam + smooth bounds."""
        # Sample in z-space, then transform
        sampled_values: dict[str, jnp.ndarray] = {}

        for sname in sampling_names:
            if sname not in scalings:
                continue
            sc = scalings[sname]

            # Sample z ~ N(0, 1)
            z = numpyro.sample(f"{sname}_z", dist.Normal(0.0, 1.0))

            # Transform: raw = center + scale * z, then smooth bound
            bounded = sc.to_original(z)
            sampled_values[sname] = bounded

        # Back-transform reparameterized pairs to physics space
        physics_values: dict[str, jnp.ndarray] = {}
        for sname, value in sampled_values.items():
            if sname in log_to_prefactor:
                # This is a log_X_at_tref — back-transform to prefactor
                prefactor = log_to_prefactor[sname]
                exponent = log_to_exponent[sname]
                alpha = sampled_values[exponent]
                a0 = reparam_to_physics_jax(value, alpha, t_ref)
                physics_values[prefactor] = a0
                # Register physics-space prefactor as deterministic
                numpyro.deterministic(prefactor, a0)
                # Register the log value too for diagnostics
                numpyro.deterministic(sname, value)
            elif sname not in physics_values:
                # Direct parameter (exponent or non-reparameterized)
                physics_values[sname] = value
                numpyro.deterministic(sname, value)

        # Assemble full parameter array using scatter (handles MCMC batch dims).
        # squeeze() removes any singleton batch dimensions from chain vectorization
        # so that values match the scalar elements of fixed_values.
        params = jnp.asarray(fixed_values)
        for name, value in physics_values.items():
            params = params.at[PARAM_INDICES[name]].set(jnp.squeeze(value))
        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params,
                shard_grid,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params,
                t,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        n_nan = jnp.sum(~jnp.isfinite(c2_model))
        numpyro.deterministic("n_numerical_issues", n_nan)
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))
        with numpyro.handlers.mask(mask=_likelihood_boundary_mask(c2_data, shard_grid)):
            numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


# ---------------------------------------------------------------------------
# Per-angle mode models
# ---------------------------------------------------------------------------


def get_heterodyne_model_constant(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    fixed_contrast: jnp.ndarray,
    fixed_offset: jnp.ndarray,
    shard_grid: ShardGrid | None = None,
    num_shards: int = 1,
):
    """Create NumPyro model with FIXED (pre-computed) per-angle scaling.

    Contrast and offset are not sampled — they are provided as fixed arrays
    from a preceding NLSQ or preprocessing step.  Suitable for
    ``per_angle_mode="constant"``, where each angle has its own fixed scaling
    but the physical parameters are shared.

    Sigma is sampled internally via ``HalfNormal(noise_scale * 1.5 *
    sqrt(num_shards))`` for homodyne CMC parity.

    Args:
        t: Time array, shape ``(n_t,)``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angle: Detector phi angle for this shard (degrees).
        c2_data: Observed correlation data, shape ``(n_t,)`` or ``(n_phi, n_t)``.
        noise_scale: Data-driven prior center for sampled ``sigma``.
        space: Parameter space carrying priors and fixed values.
        fixed_contrast: Speckle contrast per angle, shape ``(n_phi,)`` or scalar.
        fixed_offset: Baseline offset per angle, shape ``(n_phi,)`` or scalar.
        num_shards: CMC shard count for sigma prior tempering. Default ``1``.

    Returns:
        NumPyro model callable (no required arguments).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    prior_scale = math.sqrt(num_shards)
    sigma_scale = float(noise_scale) * 1.5 * prior_scale

    # Materialise fixed arrays outside the model closure so they are not
    # traced as model parameters.
    contrast_arr = jnp.asarray(fixed_contrast)
    offset_arr = jnp.asarray(fixed_offset)

    def model():
        """NumPyro model with fixed per-angle contrast and offset."""
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                prior = space.priors[name]
                param = numpyro.sample(name, prior.to_numpyro(name))
                params = params.at[i].set(param)

        # contrast/offset are fixed — use scalar mean if 1-D array is passed
        # so that compute_c2_heterodyne receives a scalar-compatible value.
        contrast_val = jnp.mean(contrast_arr)
        offset_val = jnp.mean(offset_arr)

        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params,
                shard_grid,
                q,
                dt,
                phi_angle,
                contrast_val,
                offset_val,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params,
                t,
                q,
                dt,
                phi_angle,
                contrast_val,
                offset_val,
            )
        n_nan = jnp.sum(~jnp.isfinite(c2_model))
        numpyro.deterministic("n_numerical_issues", n_nan)
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))
        with numpyro.handlers.mask(mask=_likelihood_boundary_mask(c2_data, shard_grid)):
            numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def get_heterodyne_model_constant_averaged(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    mean_contrast: float,
    mean_offset: float,
    shard_grid: ShardGrid | None = None,
    num_shards: int = 1,
):
    """Create NumPyro model with a single averaged scaling broadcast to all angles.

    Both ``mean_contrast`` and ``mean_offset`` are scalars computed from the
    average over all phi angles.  They are treated as fixed (not sampled) and
    broadcast uniformly.  Suitable for ``per_angle_mode="constant_averaged"``.

    Sigma is sampled internally via ``HalfNormal(noise_scale * 1.5 *
    sqrt(num_shards))`` for homodyne CMC parity.

    Args:
        t: Time array, shape ``(n_t,)``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angle: Detector phi angle for this shard (degrees).
        c2_data: Observed correlation data.
        noise_scale: Data-driven prior center for sampled ``sigma``.
        space: Parameter space carrying priors and fixed values.
        mean_contrast: Scalar speckle contrast averaged over all phi angles.
        mean_offset: Scalar baseline offset averaged over all phi angles.
        num_shards: CMC shard count for sigma prior tempering. Default ``1``.

    Returns:
        NumPyro model callable (no required arguments).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    prior_scale = math.sqrt(num_shards)
    sigma_scale = float(noise_scale) * 1.5 * prior_scale

    # Ensure Python floats to avoid accidental JAX tracing at closure time.
    _contrast = float(mean_contrast)
    _offset = float(mean_offset)

    def model():
        """NumPyro model with angle-averaged fixed contrast and offset."""
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                prior = space.priors[name]
                param = numpyro.sample(name, prior.to_numpyro(name))
                params = params.at[i].set(param)

        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params,
                shard_grid,
                q,
                dt,
                phi_angle,
                _contrast,
                _offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params,
                t,
                q,
                dt,
                phi_angle,
                _contrast,
                _offset,
            )
        n_nan = jnp.sum(~jnp.isfinite(c2_model))
        numpyro.deterministic("n_numerical_issues", n_nan)
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))
        with numpyro.handlers.mask(mask=_likelihood_boundary_mask(c2_data, shard_grid)):
            numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def get_heterodyne_model_individual(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angles: jnp.ndarray,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    contrast_prior_loc: jnp.ndarray | float = 0.5,
    contrast_prior_scale: float = 0.25,
    offset_prior_loc: jnp.ndarray | float = 1.0,
    offset_prior_scale: float = 0.25,
    shard_grids: list[ShardGrid] | None = None,
    num_shards: int = 1,
):
    """Create NumPyro model with per-angle sampled contrast and offset.

    The most general per-angle model: independently samples ``contrast_i``
    and ``offset_i`` for each phi angle using weakly informative Gaussian
    priors.  Suitable for ``per_angle_mode="individual"``.

    Physical parameters are shared across all angles; the per-angle scaling
    lives in a ``numpyro.plate`` over the angle dimension.

    Args:
        t: Time array, shape ``(n_t,)``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angles: Detector phi angles, shape ``(n_phi,)``.
        c2_data: Observed correlation data, shape ``(n_phi, n_t)``.
        sigma: Measurement uncertainty — scalar or shape ``(n_phi, n_t)``.
        space: Parameter space carrying priors and fixed values.
        contrast_prior_loc: Prior centre(s) for contrast.  Scalar or
            ``(n_phi,)`` array.  Default ``0.5``.
        contrast_prior_scale: Prior width for contrast.  Default ``0.25``.
        offset_prior_loc: Prior centre(s) for offset.  Scalar or
            ``(n_phi,)`` array.  Default ``1.0``.
        offset_prior_scale: Prior width for offset.  Default ``0.25``.
        shard_grids: Optional list of pre-computed ShardGrids, one per phi
            angle.  When provided, uses the memory-efficient element-wise
            path (no N×N allocation per angle).  ``c2_data[ai]`` and
            ``sigma[ai]`` must then be flattened to match each shard grid's
            paired indices.  Without this, the model builds n_phi N×N
            matrices per NUTS step which can cause OOM for large datasets.

    Returns:
        NumPyro model callable (no required arguments).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()

    phi_arr = jnp.asarray(phi_angles)
    n_phi = phi_arr.shape[0]

    if shard_grids is not None and len(shard_grids) != n_phi:
        raise ValueError(
            f"shard_grids length {len(shard_grids)} must match n_phi {n_phi}"
        )

    contrast_loc = jnp.broadcast_to(jnp.asarray(contrast_prior_loc), (n_phi,))
    offset_loc = jnp.broadcast_to(jnp.asarray(offset_prior_loc), (n_phi,))

    prior_scale = math.sqrt(num_shards)
    sigma_scale = float(noise_scale) * 1.5 * prior_scale

    def model():
        """NumPyro model with per-angle sampled contrast and offset."""
        # --- Shared physical parameters ---
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                prior = space.priors[name]
                param = numpyro.sample(name, prior.to_numpyro(name))
                params = params.at[i].set(param)

        # --- Per-angle scaling sampled in z-space + smooth_bound ---
        # Homodyne parity: sample in unconstrained z-space, then
        # transform via smooth_bound (tanh) for NUTS-safe gradients.
        with numpyro.plate("angles", n_phi):
            contrast_z = numpyro.sample(
                "contrast_z",
                dist.Normal(0.0, 1.0),
            )
            offset_z = numpyro.sample(
                "offset_z",
                dist.Normal(0.0, 1.0),
            )

        # Transform: raw = loc + scale * z, then smooth bound to physics range
        contrast_raw = contrast_loc + contrast_prior_scale * contrast_z
        contrast_i = smooth_bound(contrast_raw, 0.0, 1.0)
        numpyro.deterministic("contrast", contrast_i)

        offset_raw = offset_loc + offset_prior_scale * offset_z
        offset_i = smooth_bound(offset_raw, 0.5, 1.5)
        numpyro.deterministic("offset", offset_i)

        # --- Sigma sampled once and shared across angles (homodyne parity) ---
        sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

        # --- Likelihood over all angles ---
        # contrast_i / offset_i have shape (n_phi,); iterate to build
        # predictions per angle.  A vmap would require static phi_arr indexing
        # which is safe here, but a Python loop keeps tracing simple and avoids
        # shape-inference issues with dynamic plate sizes.
        n_total_nan: jnp.ndarray | int = 0
        for ai in range(n_phi):
            if shard_grids is not None:
                c2_model_i = compute_c2_elementwise(
                    params,
                    shard_grids[ai],
                    q,
                    dt,
                    float(phi_arr[ai]),
                    contrast_i[ai],
                    offset_i[ai],
                )
            else:
                c2_model_i = compute_c2_heterodyne(
                    params,
                    t,
                    q,
                    dt,
                    float(phi_arr[ai]),
                    contrast_i[ai],
                    offset_i[ai],
                )
            n_total_nan = n_total_nan + jnp.sum(~jnp.isfinite(c2_model_i))
            sg_i = shard_grids[ai] if shard_grids is not None else None
            with numpyro.handlers.mask(
                mask=_likelihood_boundary_mask(c2_data[ai], sg_i)
            ):
                numpyro.sample(
                    f"obs_{ai}",
                    dist.Normal(c2_model_i, sigma),
                    obs=c2_data[ai],
                )
        numpyro.deterministic("n_numerical_issues", n_total_nan)

    return model


def get_model_for_mode(
    per_angle_mode: str,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    noise_scale: float,
    space: ParameterSpace,
    nlsq_result: NLSQResult | None = None,
    reparam_config: ReparamConfig | None = None,
    num_shards: int = 1,
    **kwargs: object,
) -> Callable[[], None]:
    """Select and build the appropriate NumPyro model based on per-angle mode.

    Factory that maps ``per_angle_mode`` strings to concrete model constructors.
    Extra keyword arguments are forwarded to the selected constructor, allowing
    callers to pass mode-specific parameters (e.g. ``fixed_contrast``,
    ``mean_contrast``, ``phi_angles``) without branching at the call site.

    Mapping
    -------
    ``"auto"``
        Delegates to :func:`get_heterodyne_model` (sampled contrast/offset from
        the parameter space) or :func:`get_heterodyne_model_reparam` when
        ``reparam_config`` is supplied.
    ``"constant"``
        Delegates to :func:`get_heterodyne_model_constant`.
        Requires ``fixed_contrast`` and ``fixed_offset`` in ``kwargs``.
    ``"constant_averaged"``
        Delegates to :func:`get_heterodyne_model_constant_averaged`.
        Requires ``mean_contrast`` and ``mean_offset`` in ``kwargs``.
    ``"individual"``
        Delegates to :func:`get_heterodyne_model_individual`.
        Requires ``phi_angles`` and ``c2_data`` shaped ``(n_phi, n_t)`` in
        ``kwargs``.

    Args:
        per_angle_mode: One of ``"auto"``, ``"constant"``,
            ``"constant_averaged"``, ``"individual"``.
        t: Time array.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angle: Scalar phi angle (used by non-individual modes).
        c2_data: Observed correlation data.
        noise_scale: Data-driven prior centre for the sampled ``sigma`` site.
        space: Parameter space.
        nlsq_result: Optional NLSQ result for warm-starting (used by
            ``"auto"`` mode when ``reparam_config`` is supplied).
        reparam_config: Optional reparameterization config.  When provided
            alongside ``"auto"`` mode, activates the reparam model path.
        num_shards: CMC shard count for sigma prior tempering. Default ``1``.
        **kwargs: Mode-specific keyword arguments forwarded verbatim.

    Returns:
        NumPyro model callable (no required arguments).

    Raises:
        ValueError: If ``per_angle_mode`` is not a recognised string.
    """
    _VALID_MODES = frozenset({"auto", "constant", "constant_averaged", "individual"})
    if per_angle_mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown per_angle_mode '{per_angle_mode}'. "
            f"Valid options: {sorted(_VALID_MODES)}"
        )

    if per_angle_mode == "auto":
        scalings: dict[str, ParameterScaling] | None = kwargs.pop(  # type: ignore[assignment]
            "scalings", None
        )
        contrast: float = float(kwargs.pop("contrast", 1.0))  # type: ignore[arg-type]
        offset: float = float(kwargs.pop("offset", 1.0))  # type: ignore[arg-type]
        sg: ShardGrid | None = kwargs.pop("shard_grid", None)  # type: ignore[assignment]
        if reparam_config is not None:
            return get_heterodyne_model_reparam(
                t=t,
                q=q,
                dt=dt,
                phi_angle=phi_angle,
                c2_data=c2_data,
                noise_scale=noise_scale,
                space=space,
                reparam_config=reparam_config,
                scalings=scalings,
                contrast=contrast,
                offset=offset,
                shard_grid=sg,
                num_shards=num_shards,
            )
        return get_heterodyne_model(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            noise_scale=noise_scale,
            space=space,
            contrast=contrast,
            offset=offset,
            shard_grid=sg,
            num_shards=num_shards,
        )

    if per_angle_mode == "constant":
        fixed_contrast = kwargs.pop("fixed_contrast")
        fixed_offset = kwargs.pop("fixed_offset")
        sg_const: ShardGrid | None = kwargs.pop("shard_grid", None)  # type: ignore[assignment]
        return get_heterodyne_model_constant(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            noise_scale=noise_scale,
            space=space,
            fixed_contrast=fixed_contrast,  # type: ignore[arg-type]
            fixed_offset=fixed_offset,  # type: ignore[arg-type]
            shard_grid=sg_const,
            num_shards=num_shards,
        )

    if per_angle_mode == "constant_averaged":
        mean_contrast = float(kwargs.pop("mean_contrast", 1.0))  # type: ignore[arg-type]
        mean_offset = float(kwargs.pop("mean_offset", 1.0))  # type: ignore[arg-type]
        sg_avg: ShardGrid | None = kwargs.pop("shard_grid", None)  # type: ignore[assignment]
        return get_heterodyne_model_constant_averaged(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            noise_scale=noise_scale,
            space=space,
            mean_contrast=mean_contrast,
            mean_offset=mean_offset,
            shard_grid=sg_avg,
            num_shards=num_shards,
        )

    # per_angle_mode == "individual"
    phi_angles = kwargs.pop("phi_angles")
    sg_individual: list[ShardGrid] | None = kwargs.pop("shard_grids", None)  # type: ignore[assignment]
    return get_heterodyne_model_individual(
        t=t,
        q=q,
        dt=dt,
        phi_angles=phi_angles,  # type: ignore[arg-type]
        c2_data=c2_data,
        noise_scale=noise_scale,
        space=space,
        shard_grids=sg_individual,
        num_shards=num_shards,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Sigma estimation
# ---------------------------------------------------------------------------


def estimate_sigma(
    c2_data: jnp.ndarray,
    method: str = "diagonal",
    nlsq_result: NLSQResult | None = None,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 0,
) -> jnp.ndarray:
    """Estimate measurement uncertainty from data.

    Supported methods:

    - ``"diagonal"`` -- Uses the standard deviation of the diagonal of
      ``c2_data`` relative to its mean, floored at 1 % of the data's
      overall scale.  Fast and requires no additional information.
    - ``"constant"`` -- Returns the overall standard deviation of
      ``c2_data`` as a scalar.
    - ``"local"`` -- Computes a spatially smoothed local variance via
      ``scipy.ndimage.uniform_filter``.  Requires SciPy.
    - ``"residual"`` -- Estimates sigma from the RMS of NLSQ residuals.
      Requires ``nlsq_result`` with a non-``None`` ``residuals`` field.
      Falls back to ``"diagonal"`` if residuals are unavailable.
    - ``"bootstrap"`` -- Draws ``n_bootstrap`` bootstrap replicates of the
      diagonal and returns the standard deviation of per-replicate means as
      the noise estimate.  Useful when the diagonal has enough points to
      bootstrap.

    Args:
        c2_data: Correlation data, shape ``(n_t,)`` or ``(n_phi, n_t)``.
        method: Estimation method — one of ``"diagonal"``, ``"constant"``,
            ``"local"``, ``"residual"``, ``"bootstrap"``.
        nlsq_result: NLSQ result object.  Required (and used) only for
            ``method="residual"``.
        n_bootstrap: Number of bootstrap replicates for ``method="bootstrap"``.
            Default ``200``.
        bootstrap_seed: JAX PRNG seed for ``method="bootstrap"``.  Default ``0``.

    Returns:
        Estimated sigma — same shape as ``c2_data`` for ``"local"``, scalar
        or ``(n_t,)`` array for all other methods.

    Raises:
        ValueError: If ``method`` is not a recognised string.
    """
    import jax

    if method == "diagonal":
        # Use deviation from diagonal as proxy for noise
        diag = jnp.diag(c2_data)
        expected_diag = jnp.mean(diag)
        sigma = jnp.std(diag - expected_diag)
        # Floor at 1% of data scale to avoid near-zero sigma for
        # normalized data where diagonal values are very uniform.
        # Rule 7: jnp.where preserves gradients below the floor; jnp.maximum
        # zeros them, which stalls downstream Jacobians and NUTS leapfrog.
        _std = jnp.std(c2_data)
        data_scale = jnp.where(_std > 1e-6, _std, 1e-6)
        _floor = 0.01 * data_scale
        return jnp.where(sigma > _floor, sigma, _floor)

    elif method == "constant":
        # Use overall standard deviation
        return jnp.std(c2_data)

    elif method == "local":
        # Local variance estimation
        import numpy as np
        from scipy.ndimage import uniform_filter

        c2_np = np.asarray(c2_data)
        mean_local = uniform_filter(c2_np, size=5, mode="reflect")
        var_local = uniform_filter(c2_np**2, size=5, mode="reflect") - mean_local**2
        sigma_np = np.sqrt(np.maximum(var_local, 1e-12))
        return jnp.asarray(sigma_np)

    elif method == "residual":
        # Estimate sigma from NLSQ residuals when available.
        if nlsq_result is not None and nlsq_result.residuals is not None:
            residuals = jnp.asarray(nlsq_result.residuals)
            rms = jnp.sqrt(jnp.mean(residuals**2))
            # Floor at 1 % of data scale for robustness. Rule 7: gradient-safe.
            _std = jnp.std(c2_data)
            data_scale = jnp.where(_std > 1e-6, _std, 1e-6)
            _floor = 0.01 * data_scale
            return jnp.where(rms > _floor, rms, _floor)
        # Fall back gracefully so callers don't need to guard against None.
        return estimate_sigma(c2_data, method="diagonal")

    elif method == "bootstrap":
        # Bootstrap estimate of sigma from repeated diagonal measurements.
        # Draws n_bootstrap replicates of the diagonal with replacement and
        # uses the standard deviation of replicate means as the noise level.
        diag = jnp.diag(c2_data)
        n = diag.shape[0]
        key = jax.random.PRNGKey(bootstrap_seed)

        # Draw indices: shape (n_bootstrap, n)
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, shape=(n_bootstrap, n), minval=0, maxval=n)
        # Replicate means: shape (n_bootstrap,)
        replicate_means = jnp.mean(diag[indices], axis=1)
        sigma_boot = jnp.std(replicate_means)

        # Floor at 0.1 % of data scale (bootstrap can give very small values
        # when the diagonal is extremely uniform). Rule 7: gradient-safe.
        _std = jnp.std(c2_data)
        data_scale = jnp.where(_std > 1e-6, _std, 1e-6)
        _floor = 0.001 * data_scale
        return jnp.where(sigma_boot > _floor, sigma_boot, _floor)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: "
            "'diagonal', 'constant', 'local', 'residual', 'bootstrap'."
        )


# ---------------------------------------------------------------------------
# Model output validation and parameter counting
# ---------------------------------------------------------------------------


def validate_model_output(
    c2_theory: jnp.ndarray,
    params: jnp.ndarray,
) -> bool:
    """Validate that theoretical C2 values are physically reasonable.

    Checks for NaN/inf values and enforces the heterodyne C2 range
    constraint ``[-1.0, 10.0]``.  Heterodyne C2 can go negative due to
    the velocity phase term, unlike homodyne where C2 >= 0.

    Args:
        c2_theory: Theoretical C2 array from model evaluation.
        params: Parameter array used to produce ``c2_theory`` (logged
            on failure for diagnostics).

    Returns:
        ``True`` if the output passes all checks, ``False`` otherwise.
    """
    # Check for NaN values
    if bool(jnp.any(jnp.isnan(c2_theory))):
        logger.warning(
            "validate_model_output: NaN detected in C2 theory (params=%s)",
            params,
        )
        return False

    # Check for inf values
    if bool(jnp.any(jnp.isinf(c2_theory))):
        logger.warning(
            "validate_model_output: inf detected in C2 theory (params=%s)",
            params,
        )
        return False

    # Enforce heterodyne C2 range: [-1.0, 10.0]
    c2_min = float(jnp.min(c2_theory))
    c2_max = float(jnp.max(c2_theory))
    if c2_min < -1.0 or c2_max > 10.0:
        logger.warning(
            "validate_model_output: C2 range [%.4e, %.4e] exceeds "
            "physical bounds [-1.0, 10.0] (params=%s)",
            c2_min,
            c2_max,
            params,
        )
        return False

    return True


def get_model_param_count(
    n_phi: int,
    per_angle_mode: str = "individual",
) -> int:
    """Return total number of sampled parameters for the model.

    Accounts for per-angle mode semantics when counting contrast/offset
    parameters that are sampled in addition to the shared physics
    parameters.

    Per-angle mode contributions:

    * ``"constant"`` — 0 per-angle params (fixed contrast/offset).
    * ``"constant_averaged"`` — 0 per-angle params (fixed averaged
      contrast/offset).
    * ``"auto"`` — physics params only (contrast/offset live in the
      parameter space, already counted).
    * ``"individual"`` — ``2 * n_phi`` per-angle params
      (``contrast_z`` + ``offset_z`` per angle).

    Args:
        n_phi: Number of scattering angles.
        per_angle_mode: One of ``"constant"``, ``"constant_averaged"``,
            ``"auto"``, ``"individual"``.

    Returns:
        Total number of sampled parameters (int).

    Raises:
        ValueError: If ``per_angle_mode`` is not recognised.
    """
    _VALID_MODES = frozenset({"auto", "constant", "constant_averaged", "individual"})
    if per_angle_mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown per_angle_mode '{per_angle_mode}'. "
            f"Valid options: {sorted(_VALID_MODES)}"
        )

    # Base: count physics params that vary by default in the registry
    n_physics = sum(
        1 for name in ALL_PARAM_NAMES if DEFAULT_REGISTRY[name].vary_default
    )

    # Per-angle contributions
    if per_angle_mode == "individual":
        n_per_angle = 2 * n_phi  # contrast_z + offset_z per angle
    else:
        # "constant", "constant_averaged", "auto" — no additional sampled params
        n_per_angle = 0

    total = n_physics + n_per_angle
    logger.debug(
        "get_model_param_count: n_physics=%d, n_per_angle=%d (mode=%s, n_phi=%d) -> %d",
        n_physics,
        n_per_angle,
        per_angle_mode,
        n_phi,
        total,
    )
    return total
