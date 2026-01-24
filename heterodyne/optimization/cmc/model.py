"""NumPyro model definition for heterodyne Bayesian inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.core.jax_backend import compute_c2_heterodyne

if TYPE_CHECKING:
    from heterodyne.config.parameter_space import ParameterSpace


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
    varying_indices = [ALL_PARAM_NAMES.index(name) for name in varying_names]
    fixed_values = space.get_initial_array()
    
    def model():
        """NumPyro model for heterodyne correlation."""
        # Sample varying parameters
        params_list = []
        
        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                prior = space.priors[name]
                param = numpyro.sample(name, prior.to_numpyro(name))
                params_list.append(param)
            else:
                params_list.append(fixed_values[i])
        
        # Stack into parameter array
        params = jnp.array(params_list)
        
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
):
    """Create NumPyro model with reparameterization for better sampling.
    
    Uses NLSQ results to center the posterior if available.
    
    Args:
        t: Time array
        q: Wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Observed correlation data
        sigma: Measurement uncertainty
        space: Parameter space
        nlsq_params: Optional NLSQ fitted values for centering
        
    Returns:
        NumPyro model function
    """
    varying_names = space.varying_names
    fixed_values = space.get_initial_array()
    
    # Use NLSQ params as prior centers if available
    if nlsq_params is not None:
        prior_centers = {
            name: float(nlsq_params[ALL_PARAM_NAMES.index(name)])
            for name in varying_names
        }
    else:
        prior_centers = {name: space.values[name] for name in varying_names}
    
    def model():
        """NumPyro model with centered parameterization."""
        params_list = []
        
        for i, name in enumerate(ALL_PARAM_NAMES):
            if name in varying_names:
                # Use normal prior centered on NLSQ estimate
                center = prior_centers[name]
                bounds = space.bounds[name]
                
                # Scale based on bounds
                scale = (bounds[1] - bounds[0]) / 6.0  # 6 sigma covers 99.7%
                
                # Sample from normal, then clip to bounds
                raw = numpyro.sample(f"{name}_raw", dist.Normal(center, scale))
                param = jnp.clip(raw, bounds[0], bounds[1])
                numpyro.deterministic(name, param)
                params_list.append(param)
            else:
                params_list.append(fixed_values[i])
        
        params = jnp.array(params_list)
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
        n = c2_data.shape[0]
        diag = jnp.diag(c2_data)
        expected_diag = jnp.mean(diag)
        sigma = jnp.std(diag - expected_diag)
        return jnp.maximum(sigma, 1e-6)
    
    elif method == "constant":
        # Use overall standard deviation
        return jnp.std(c2_data)
    
    elif method == "local":
        # Local variance estimation
        from scipy.ndimage import uniform_filter
        import numpy as np
        
        c2_np = np.asarray(c2_data)
        mean_local = uniform_filter(c2_np, size=5, mode='reflect')
        var_local = uniform_filter(c2_np**2, size=5, mode='reflect') - mean_local**2
        sigma = np.sqrt(np.maximum(var_local, 1e-12))
        return jnp.asarray(sigma)
    
    else:
        raise ValueError(f"Unknown method: {method}")
