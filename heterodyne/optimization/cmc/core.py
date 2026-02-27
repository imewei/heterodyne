"""Core CMC fitting function for heterodyne Bayesian analysis."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.diagnostics import validate_convergence
from heterodyne.optimization.cmc.model import (
    estimate_sigma,
    get_heterodyne_model,
    get_heterodyne_model_reparam,
)
from heterodyne.optimization.cmc.reparameterization import (
    ReparamConfig,
    compute_t_ref,
    transform_nlsq_to_reparam_space,
    transform_to_physics_space,
)
from heterodyne.optimization.cmc.results import CMCResult
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel

logger = get_logger(__name__)


def fit_cmc_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: CMCConfig | None = None,
    sigma: np.ndarray | float | None = None,
    nlsq_result: NLSQResult | None = None,
) -> CMCResult:
    """Fit heterodyne model using Consensus Monte Carlo.

    Uses NumPyro's NUTS sampler for Bayesian posterior inference.

    Args:
        model: HeterodyneModel with configured parameters
        c2_data: Observed correlation data
        phi_angle: Detector phi angle (degrees)
        config: CMC configuration (default if None)
        sigma: Measurement uncertainty (estimated if None)
        nlsq_result: Optional NLSQ result for warm-starting

    Returns:
        CMCResult with posterior samples and diagnostics
    """
    if config is None:
        config = CMCConfig()

    logger.info(f"Starting CMC analysis: chains={config.num_chains}, samples={config.num_samples}")

    start_time = time.perf_counter()

    # Prepare data
    c2_jax = jnp.asarray(c2_data)

    # Estimate sigma if not provided
    if sigma is None:
        sigma = estimate_sigma(c2_jax, method="diagonal")
        logger.info(f"Estimated sigma = {float(jnp.mean(sigma)):.4e}")

    sigma_jax = jnp.asarray(sigma) if isinstance(sigma, np.ndarray) else sigma

    # Get parameter space
    space = model.param_manager.space
    varying_names = model.param_manager.varying_names

    logger.info(f"Sampling {len(varying_names)} parameters: {varying_names}")

    # Decide whether to use reparameterized model
    use_reparam = (
        config.use_reparam
        and nlsq_result is not None
        and nlsq_result.success
    )

    reparam_config = None
    scalings = None
    prior_std_dict: dict[str, float] = {}

    if use_reparam:
        # Compute reference time
        t_array = np.asarray(model.t)
        dt_val = float(t_array[1] - t_array[0]) if len(t_array) > 1 else float(t_array[0])
        t_max_val = float(t_array[-1])
        t_ref = compute_t_ref(dt_val, t_max_val, fallback_value=1.0)

        reparam_config = ReparamConfig(t_ref=t_ref)
        logger.info(f"Using reference-time reparameterization: t_ref={t_ref:.4e}")

        # Get NLSQ values and uncertainties
        nlsq_values = {
            name: float(nlsq_result.get_param(name))
            for name in varying_names
            if name in nlsq_result.parameter_names
        }
        nlsq_uncertainties = {
            name: float(nlsq_result.get_uncertainty(name))
            for name in varying_names
            if name in nlsq_result.parameter_names
            and nlsq_result.get_uncertainty(name) is not None
        }

        # Transform to reparam space with delta-method UQ
        reparam_values, reparam_uncertainties = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref, reparam_config,
        )

        # Build scaling factors in reparam (sampling) space.
        # compute_scaling_factors uses space.varying_names (physics-space keys),
        # but we need scalings keyed by sampling-space names. Build manually.
        from heterodyne.optimization.cmc.scaling import ParameterScaling

        scalings: dict[str, ParameterScaling] = {}
        prefactor_to_log = {}
        for prefactor, exponent in reparam_config.enabled_pairs:
            if prefactor in varying_names and exponent in varying_names:
                prefactor_to_log[prefactor] = reparam_config.get_reparam_name(prefactor)

        for name in varying_names:
            if name in prefactor_to_log:
                # This prefactor is replaced by its log-at-tref name
                sname = prefactor_to_log[name]
            else:
                sname = name

            center = reparam_values.get(sname, space.values.get(name, 0.0))
            unc = reparam_uncertainties.get(sname, 0.0)
            scale = unc * config.prior_width_factor if unc > 0 else 1.0
            scale = max(scale, 1e-10)

            # Bounds: for log-space params use wide range, for others use physics bounds
            if sname.startswith("log_"):
                low = center - 10.0 * scale
                high = center + 10.0 * scale
            else:
                low, high = space.bounds[name]

            scalings[sname] = ParameterScaling(
                name=sname, center=center, scale=scale, low=low, high=high,
            )

        # Track prior_std for PCR diagnostics, keyed by physics-space names.
        # For reparameterized prefactors, convert log-space scale to physics-space
        # via delta method: sigma_A0 ≈ A0_at_tref * sigma_log.
        prior_std_dict: dict[str, float] = {}
        for name in varying_names:
            if name in prefactor_to_log:
                sname = prefactor_to_log[name]
                sc = scalings[sname]
                center_physics = float(np.exp(sc.center))
                prior_std_dict[name] = center_physics * sc.scale
            else:
                sname = name
                if sname in scalings:
                    prior_std_dict[name] = scalings[sname].scale

        # Create reparameterized model
        numpyro_model = get_heterodyne_model_reparam(
            t=model.t,
            q=model.q,
            dt=model.dt,
            phi_angle=phi_angle,
            c2_data=c2_jax,
            sigma=sigma_jax,
            space=space,
            nlsq_result=nlsq_result,
            reparam_config=reparam_config,
            scalings=scalings,
        )
    else:
        # Use standard model (no reparameterization)
        numpyro_model = get_heterodyne_model(
            t=model.t,
            q=model.q,
            dt=model.dt,
            phi_angle=phi_angle,
            c2_data=c2_jax,
            sigma=sigma_jax,
            space=space,
        )

    # Set up NUTS sampler
    kernel = NUTS(
        numpyro_model,
        target_accept_prob=config.target_accept,
        max_tree_depth=config.max_tree_depth,
        dense_mass=config.dense_mass,
    )

    # Initialize from NLSQ if available
    init_params = None
    if config.use_nlsq_warmstart and nlsq_result is not None and nlsq_result.success:
        logger.info("Using NLSQ result for initialization")

        if use_reparam and scalings is not None:
            # Initialize in z-space: z = (value - center) / scale
            # Add small random perturbation per chain to break symmetry
            init_params = {}
            perturb_key = jax.random.PRNGKey(
                config.seed + 1 if config.seed is not None else 1
            )
            for sname, sc in scalings.items():
                perturb_key, subkey = jax.random.split(perturb_key)
                reparam_val = reparam_values.get(sname, sc.center)
                z_init = sc.to_normalized(reparam_val)
                base = jnp.full((config.num_chains,), jnp.float64(z_init))
                perturbation = 0.01 * jax.random.normal(subkey, shape=(config.num_chains,))
                init_params[f"{sname}_z"] = base + perturbation
        else:
            # Legacy: replicate NLSQ values for each chain
            # Add small random perturbation per chain to break symmetry
            init_params = {}
            perturb_key = jax.random.PRNGKey(
                config.seed + 1 if config.seed is not None else 1
            )
            for name in varying_names:
                if name in nlsq_result.parameter_names:
                    perturb_key, subkey = jax.random.split(perturb_key)
                    base = jnp.full(
                        (config.num_chains,),
                        jnp.float64(nlsq_result.get_param(name)),
                    )
                    perturbation = 0.01 * jax.random.normal(
                        subkey, shape=(config.num_chains,)
                    )
                    init_params[name] = base + perturbation

    # Set up MCMC
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=True,
    )

    # Run sampling
    rng_key = jax.random.PRNGKey(config.seed if config.seed is not None else 0)

    try:
        mcmc.run(rng_key, init_params=init_params)
    except Exception as e:
        logger.error(f"MCMC sampling failed: {e}")
        return _create_failed_result(varying_names, str(e))

    # Get samples
    samples = mcmc.get_samples()

    # Convert to ArviZ for diagnostics
    idata = az.from_numpyro(mcmc)

    # Determine which names to report in physics space
    if use_reparam and reparam_config is not None:
        # Back-transform reparameterized samples to physics space
        raw_samples = {k: np.asarray(v) for k, v in samples.items()}
        physics_samples = transform_to_physics_space(raw_samples, reparam_config)

        # Use physics-space names for output (the varying_names are already physics-space)
        output_names = varying_names

        # Compute diagnostics from deterministic sites (physics space)
        # ArviZ summary uses the deterministic sites which are in physics space
        available_names = [n for n in output_names if n in idata.posterior]
        summary = az.summary(idata, var_names=available_names, hdi_prob=0.95) if available_names else None
    else:
        physics_samples = {k: np.asarray(v) for k, v in samples.items()}
        output_names = varying_names
        summary = az.summary(idata, var_names=output_names, hdi_prob=0.95)

    # Extract posterior statistics
    if summary is not None and len(summary) > 0:
        posterior_mean = np.array([
            float(summary.loc[name, "mean"]) if name in summary.index else 0.0
            for name in output_names
        ])
        posterior_std = np.array([
            float(summary.loc[name, "sd"]) if name in summary.index else 0.0
            for name in output_names
        ])
        r_hat = np.array([
            float(summary.loc[name, "r_hat"]) if name in summary.index else np.nan
            for name in output_names
        ])
        ess_bulk = np.array([
            float(summary.loc[name, "ess_bulk"]) if name in summary.index else 0.0
            for name in output_names
        ])
        ess_tail = np.array([
            float(summary.loc[name, "ess_tail"]) if name in summary.index else 0.0
            for name in output_names
        ])
    else:
        # Compute from physics_samples directly
        posterior_mean = np.array([
            float(np.mean(physics_samples[name])) if name in physics_samples else 0.0
            for name in output_names
        ])
        posterior_std = np.array([
            float(np.std(physics_samples[name])) if name in physics_samples else 0.0
            for name in output_names
        ])
        r_hat = np.full(len(output_names), np.nan)
        ess_bulk = np.full(len(output_names), 0.0)
        ess_tail = np.full(len(output_names), 0.0)

    # Credible intervals
    credible_intervals = {}
    for name in output_names:
        if summary is not None and name in summary.index:
            credible_intervals[name] = {
                "2.5%": float(summary.loc[name, "hdi_2.5%"]),
                "97.5%": float(summary.loc[name, "hdi_97.5%"]),
            }
        elif name in physics_samples:
            s = physics_samples[name]
            credible_intervals[name] = {
                "2.5%": float(np.percentile(s, 2.5)),
                "97.5%": float(np.percentile(s, 97.5)),
            }

    # BFMI
    bfmi = None
    bfmi_compute_failed = False
    try:
        bfmi_result = az.bfmi(idata)
        if hasattr(bfmi_result, 'values'):
            bfmi = list(bfmi_result.values)
        elif isinstance(bfmi_result, (list, np.ndarray)):
            bfmi = list(bfmi_result)
    except Exception:
        logger.warning("Could not compute BFMI")
        bfmi_compute_failed = True

    # Store physics-space samples
    samples_dict = {
        name: physics_samples[name]
        for name in output_names
        if name in physics_samples
    }

    # MAP estimate (mode of posterior)
    map_estimate = posterior_mean.copy()

    wall_time = time.perf_counter() - start_time

    # Check convergence — NaN R-hat or zero ESS means failure, not skip
    r_hat_finite = r_hat[~np.isnan(r_hat)]
    ess_finite = ess_bulk[~np.isnan(ess_bulk)]
    convergence_passed = bool(
        len(r_hat_finite) > 0
        and np.all(r_hat_finite < config.r_hat_threshold)
        and len(ess_finite) > 0
        and np.all(ess_finite > config.min_ess)
    )
    if bfmi is not None:
        convergence_passed = convergence_passed and min(bfmi) > config.min_bfmi
    if bfmi_compute_failed:
        convergence_passed = False

    # Build metadata
    metadata: dict = {}
    if use_reparam and reparam_config is not None:
        metadata["t_ref"] = reparam_config.t_ref
        metadata["prior_std"] = prior_std_dict

    result = CMCResult(
        parameter_names=output_names,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        credible_intervals=credible_intervals,
        convergence_passed=convergence_passed,
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        bfmi=bfmi,
        samples=samples_dict,
        map_estimate=map_estimate,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        wall_time_seconds=wall_time,
        metadata=metadata,
    )

    # Log convergence status
    conv_report = validate_convergence(result, config.r_hat_threshold, config.min_ess, config.min_bfmi)
    for msg in conv_report.messages:
        logger.info(msg)

    logger.info(f"CMC complete in {wall_time:.1f}s, convergence: {'PASSED' if convergence_passed else 'FAILED'}")

    return result


def _create_failed_result(parameter_names: list[str], message: str) -> CMCResult:
    """Create a failed CMC result."""
    n_params = len(parameter_names)
    return CMCResult(
        parameter_names=parameter_names,
        posterior_mean=np.zeros(n_params),
        posterior_std=np.zeros(n_params),
        credible_intervals={},
        convergence_passed=False,
        metadata={"error": message},
    )
