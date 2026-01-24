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
from heterodyne.optimization.cmc.model import estimate_sigma, get_heterodyne_model
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

    # Create NumPyro model
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
    # Note: When using multiple chains, NumPyro expects init_params to have shape (num_chains,)
    # for each parameter. We replicate the NLSQ solution for all chains.
    init_params = None
    if config.use_nlsq_warmstart and nlsq_result is not None and nlsq_result.success:
        logger.info("Using NLSQ result for initialization")
        # Replicate NLSQ values for each chain
        init_params = {
            name: jnp.full((config.num_chains,), nlsq_result.get_param(name))
            for name in varying_names
            if name in nlsq_result.parameter_names
        }

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

    # Compute diagnostics
    summary = az.summary(idata, var_names=varying_names)

    # Extract posterior statistics
    posterior_mean = np.array([float(summary.loc[name, "mean"]) for name in varying_names])
    posterior_std = np.array([float(summary.loc[name, "sd"]) for name in varying_names])

    # R-hat and ESS
    r_hat = np.array([float(summary.loc[name, "r_hat"]) for name in varying_names])
    ess_bulk = np.array([float(summary.loc[name, "ess_bulk"]) for name in varying_names])
    ess_tail = np.array([float(summary.loc[name, "ess_tail"]) for name in varying_names])

    # Credible intervals
    credible_intervals = {}
    for name in varying_names:
        credible_intervals[name] = {
            "2.5%": float(summary.loc[name, "hdi_3%"]),
            "97.5%": float(summary.loc[name, "hdi_97%"]),
        }

    # BFMI
    bfmi = None
    try:
        bfmi_result = az.bfmi(idata)
        if hasattr(bfmi_result, 'values'):
            bfmi = list(bfmi_result.values)
        elif isinstance(bfmi_result, (list, np.ndarray)):
            bfmi = list(bfmi_result)
    except Exception:
        logger.warning("Could not compute BFMI")

    # Store samples
    samples_dict = {name: np.asarray(samples[name]) for name in varying_names if name in samples}

    # MAP estimate (mode of posterior)
    map_estimate = posterior_mean.copy()  # Use mean as approximation

    wall_time = time.perf_counter() - start_time

    # Check convergence
    convergence_passed = (
        np.all(r_hat < config.r_hat_threshold) and
        np.all(ess_bulk > config.min_ess)
    )
    if bfmi is not None:
        convergence_passed = convergence_passed and min(bfmi) > config.min_bfmi

    result = CMCResult(
        parameter_names=varying_names,
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
