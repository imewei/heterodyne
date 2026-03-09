"""NumPyro model definition for heterodyne Bayesian inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_INDICES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.core.jax_backend import compute_c2_heterodyne
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


def get_heterodyne_model(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    contrast: float = 1.0,
    offset: float = 1.0,
    shard_grid: ShardGrid | None = None,
):
    """Create NumPyro model for heterodyne correlation fitting.

    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data — shape ``(N, N)`` for meshgrid
            path, or ``(n_pairs,)`` for element-wise path.
        sigma: Measurement uncertainty (scalar or array matching c2_data)
        space: Parameter space with priors
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0
        shard_grid: Optional pre-computed ShardGrid.  When provided, uses
            the memory-efficient element-wise path (no N×N allocation).
            ``c2_data`` and ``sigma`` must then be flattened to match
            the shard grid's paired indices.

    Returns:
        NumPyro model function
    """
    # Pre-compute indices and masks
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()

    def model():
        """NumPyro model for heterodyne correlation."""
        # Sample varying parameters and scatter into fixed array
        # Using .at[].set() instead of jnp.array([...]) to avoid
        # tracing issues with mixed tracer/concrete values.
        params = jnp.asarray(fixed_values)

        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                prior = space.priors[name]
                param = numpyro.sample(name, prior.to_numpyro(name))
                params = params.at[i].set(param)

        # Compute model prediction — dispatch to appropriate path
        if shard_grid is not None:
            c2_model = compute_c2_elementwise(
                params, shard_grid, q, dt, phi_angle, contrast, offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params, t, q, dt, phi_angle, contrast, offset,
            )

        # Likelihood
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
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    nlsq_params: jnp.ndarray | None = None,
    nlsq_result: NLSQResult | None = None,
    reparam_config: ReparamConfig | None = None,
    scalings: dict[str, ParameterScaling] | None = None,
    prior_width_factor: float = 2.0,
    contrast: float = 1.0,
    offset: float = 1.0,
    shard_grid: ShardGrid | None = None,
):
    """Create NumPyro model with reparameterization for better sampling.

    When NLSQ result and reparameterization config are provided, uses:
    1. Reference-time reparameterization for power-law pairs
    2. Smooth bounded transforms (tanh) instead of jnp.clip()
    3. NLSQ-informed priors with delta-method uncertainty propagation

    Falls back to the original clip-based behavior when the new
    infrastructure is not provided (backward compatibility).

    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data
        sigma: Measurement uncertainty
        space: Parameter space
        nlsq_params: Optional NLSQ fitted values for centering (legacy)
        nlsq_result: Optional NLSQ result for reparameterization
        reparam_config: Reparameterization config (enables new path)
        scalings: Pre-computed ParameterScaling per reparam-space param
        prior_width_factor: Multiplier on NLSQ uncertainty for prior width

    Returns:
        NumPyro model function
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()

    # --- New reparameterized path ---
    if reparam_config is not None and scalings is not None:
        return _build_reparam_model(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            sigma=sigma,
            space=space,
            fixed_values=fixed_values,
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
                params, shard_grid, q, dt, phi_angle, contrast, offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params, t, q, dt, phi_angle, contrast, offset,
            )
        numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def _build_reparam_model(
    *,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
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
                params, shard_grid, q, dt, phi_angle, contrast, offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params, t, q, dt, phi_angle, contrast, offset,
            )
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
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    fixed_contrast: jnp.ndarray,
    fixed_offset: jnp.ndarray,
    shard_grid: ShardGrid | None = None,
):
    """Create NumPyro model with FIXED (pre-computed) per-angle scaling.

    Contrast and offset are not sampled — they are provided as fixed arrays
    from a preceding NLSQ or preprocessing step.  Suitable for
    ``per_angle_mode="constant"``, where each angle has its own fixed scaling
    but the physical parameters are shared.

    Args:
        t: Time array, shape ``(n_t,)``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angle: Detector phi angle for this shard (degrees).
        c2_data: Observed correlation data, shape ``(n_t,)`` or ``(n_phi, n_t)``.
        sigma: Measurement uncertainty — scalar or matching shape of ``c2_data``.
        space: Parameter space carrying priors and fixed values.
        fixed_contrast: Speckle contrast per angle, shape ``(n_phi,)`` or scalar.
        fixed_offset: Baseline offset per angle, shape ``(n_phi,)`` or scalar.

    Returns:
        NumPyro model callable (no required arguments).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()

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
                params, shard_grid, q, dt, phi_angle, contrast_val, offset_val,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params, t, q, dt, phi_angle, contrast_val, offset_val,
            )
        numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def get_heterodyne_model_constant_averaged(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    mean_contrast: float,
    mean_offset: float,
    shard_grid: ShardGrid | None = None,
):
    """Create NumPyro model with a single averaged scaling broadcast to all angles.

    Both ``mean_contrast`` and ``mean_offset`` are scalars computed from the
    average over all phi angles.  They are treated as fixed (not sampled) and
    broadcast uniformly.  Suitable for ``per_angle_mode="constant_averaged"``.

    Args:
        t: Time array, shape ``(n_t,)``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        phi_angle: Detector phi angle for this shard (degrees).
        c2_data: Observed correlation data.
        sigma: Measurement uncertainty.
        space: Parameter space carrying priors and fixed values.
        mean_contrast: Scalar speckle contrast averaged over all phi angles.
        mean_offset: Scalar baseline offset averaged over all phi angles.

    Returns:
        NumPyro model callable (no required arguments).
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()

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
                params, shard_grid, q, dt, phi_angle, _contrast, _offset,
            )
        else:
            c2_model = compute_c2_heterodyne(
                params, t, q, dt, phi_angle, _contrast, _offset,
            )
        numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def get_heterodyne_model_individual(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angles: jnp.ndarray,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    contrast_prior_loc: jnp.ndarray | float = 0.5,
    contrast_prior_scale: float = 0.25,
    offset_prior_loc: jnp.ndarray | float = 1.0,
    offset_prior_scale: float = 0.25,
    shard_grids: list[ShardGrid] | None = None,
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
                "contrast_z", dist.Normal(0.0, 1.0),
            )
            offset_z = numpyro.sample(
                "offset_z", dist.Normal(0.0, 1.0),
            )

        # Transform: raw = loc + scale * z, then smooth bound to physics range
        contrast_raw = contrast_loc + contrast_prior_scale * contrast_z
        contrast_i = smooth_bound(contrast_raw, 0.0, 1.0)
        numpyro.deterministic("contrast", contrast_i)

        offset_raw = offset_loc + offset_prior_scale * offset_z
        offset_i = smooth_bound(offset_raw, 0.5, 1.5)
        numpyro.deterministic("offset", offset_i)

        # --- Likelihood over all angles ---
        # contrast_i / offset_i have shape (n_phi,); iterate to build
        # predictions per angle.  A vmap would require static phi_arr indexing
        # which is safe here, but a Python loop keeps tracing simple and avoids
        # shape-inference issues with dynamic plate sizes.
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
            sigma_i = sigma[ai] if hasattr(sigma, "__len__") else sigma  # type: ignore[index]
            numpyro.sample(
                f"obs_{ai}",
                dist.Normal(c2_model_i, sigma_i),
                obs=c2_data[ai],
            )

    return model


def get_model_for_mode(
    per_angle_mode: str,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
    nlsq_result: NLSQResult | None = None,
    reparam_config: ReparamConfig | None = None,
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
        sigma: Measurement uncertainty.
        space: Parameter space.
        nlsq_result: Optional NLSQ result for warm-starting (used by
            ``"auto"`` mode when ``reparam_config`` is supplied).
        reparam_config: Optional reparameterization config.  When provided
            alongside ``"auto"`` mode, activates the reparam model path.
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
                sigma=sigma,
                space=space,
                nlsq_result=nlsq_result,
                reparam_config=reparam_config,
                scalings=scalings,
                contrast=contrast,
                offset=offset,
                shard_grid=sg,
            )
        return get_heterodyne_model(
            t=t,
            q=q,
            dt=dt,
            phi_angle=phi_angle,
            c2_data=c2_data,
            sigma=sigma,
            space=space,
            contrast=contrast,
            offset=offset,
            shard_grid=sg,
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
            sigma=sigma,
            space=space,
            fixed_contrast=fixed_contrast,  # type: ignore[arg-type]
            fixed_offset=fixed_offset,  # type: ignore[arg-type]
            shard_grid=sg_const,
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
            sigma=sigma,
            space=space,
            mean_contrast=mean_contrast,
            mean_offset=mean_offset,
            shard_grid=sg_avg,
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
        sigma=sigma,
        space=space,
        shard_grids=sg_individual,
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

    Methods
    -------
    ``"diagonal"``
        Uses the standard deviation of the diagonal of ``c2_data`` relative
        to its mean, floored at 1 % of the data's overall scale.  Fast and
        requires no additional information.
    ``"constant"``
        Returns the overall standard deviation of ``c2_data`` as a scalar.
    ``"local"``
        Computes a spatially smoothed local variance via
        ``scipy.ndimage.uniform_filter``.  Requires SciPy.
    ``"residual"``
        Estimates sigma from the RMS of NLSQ residuals.  Requires
        ``nlsq_result`` with a non-``None`` ``residuals`` field.  Falls back
        to ``"diagonal"`` if residuals are unavailable.
    ``"bootstrap"``
        Draws ``n_bootstrap`` bootstrap replicates of the diagonal and
        returns the standard deviation of per-replicate means as the noise
        estimate.  Useful when the diagonal has enough points to bootstrap.

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
        # normalized data where diagonal values are very uniform
        data_scale = jnp.maximum(jnp.std(c2_data), 1e-6)
        return jnp.maximum(sigma, 0.01 * data_scale)

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
            # Floor at 1 % of data scale for robustness.
            data_scale = jnp.maximum(jnp.std(c2_data), 1e-6)
            return jnp.maximum(rms, 0.01 * data_scale)
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
        # when the diagonal is extremely uniform).
        data_scale = jnp.maximum(jnp.std(c2_data), 1e-6)
        return jnp.maximum(sigma_boot, 0.001 * data_scale)

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
            "validate_model_output: NaN detected in C2 theory "
            "(params=%s)",
            params,
        )
        return False

    # Check for inf values
    if bool(jnp.any(jnp.isinf(c2_theory))):
        logger.warning(
            "validate_model_output: inf detected in C2 theory "
            "(params=%s)",
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
        1 for name in ALL_PARAM_NAMES
        if DEFAULT_REGISTRY[name].vary_default
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
