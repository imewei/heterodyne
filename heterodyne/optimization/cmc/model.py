"""NumPyro model definition for heterodyne Bayesian inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_INDICES
from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.optimization.cmc.reparameterization import (
    ReparamConfig,
    reparam_to_physics_jax,
)
from heterodyne.optimization.cmc.scaling import ParameterScaling

if TYPE_CHECKING:
    from heterodyne.config.parameter_space import ParameterSpace
    from heterodyne.optimization.nlsq.results import NLSQResult


def get_heterodyne_model(
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    sigma: jnp.ndarray | float,
    space: ParameterSpace,
):
    """Create NumPyro model for heterodyne correlation fitting.

    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data
        sigma: Measurement uncertainty (scalar or array)
        space: Parameter space with priors

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

        # Compute model prediction
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle)

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
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle)
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
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle)
        numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)

    return model


def estimate_sigma(c2_data: jnp.ndarray, method: str = "diagonal") -> jnp.ndarray:
    """Estimate measurement uncertainty from data.

    Args:
        c2_data: Correlation data
        method: Estimation method ('diagonal', 'constant', 'local')

    Returns:
        Estimated sigma (same shape as c2_data or scalar)
    """
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
        mean_local = uniform_filter(c2_np, size=5, mode='reflect')
        var_local = uniform_filter(c2_np**2, size=5, mode='reflect') - mean_local**2
        sigma = np.sqrt(np.maximum(var_local, 1e-12))
        return jnp.asarray(sigma)

    else:
        raise ValueError(f"Unknown method: {method}")
