"""Core CMC fitting functions for heterodyne Bayesian analysis.

Includes the original single-run ``fit_cmc_jax`` and the new sharded
Consensus Monte Carlo entry point ``fit_cmc_sharded``, plus all supporting
helpers for shard creation, prior tempering, and posterior combination.
"""

from __future__ import annotations

import math
import secrets
import time
from typing import TYPE_CHECKING, Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.diagnostics import (
    analyze_divergences,
    log_analysis_summary,
    validate_convergence,
)
from heterodyne.optimization.cmc.model import (
    estimate_sigma,
    get_heterodyne_model,
    get_heterodyne_model_reparam,
)
from heterodyne.optimization.cmc.priors import (
    build_default_priors,
    build_nlsq_informed_priors,
    temper_priors,
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


def _block_until_ready_pytree(tree: Any) -> Any:
    """Block every JAX array leaf in a pytree and return the original object."""
    for leaf in jax.tree_util.tree_leaves(tree):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()
    return tree


# ---------------------------------------------------------------------------
# Public: original single-run entry point (signature preserved exactly)
# ---------------------------------------------------------------------------


def fit_cmc_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: CMCConfig | None = None,
    sigma: np.ndarray | float | None = None,
    nlsq_result: NLSQResult | None = None,
    t_override: np.ndarray | None = None,
    priors_override: dict | None = None,
    prior_width_multiplier: float = 1.0,
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
        t_override: Optional time array replacing ``model.t`` for model
            construction. Used by ``fit_cmc_sharded`` to pass shard time
            slices. If ``None``, falls back to ``model.t``.
        priors_override: Optional dict of pre-built NumPyro distributions
            keyed by parameter name.  When provided, these distributions
            replace the default ``space.priors`` for matching parameters.
            Used by ``fit_cmc_sharded`` to inject tempered shard priors
            into the non-reparam model path.
        prior_width_multiplier: Scalar multiplier applied to the ``scale``
            of each reparam-path prior AFTER ``nlsq_prior_width_factor``
            scaling.  Default 1.0 (no change).  Used by
            ``fit_cmc_sharded`` to widen reparam priors by ``sqrt(K)``.

    Returns:
        CMCResult with posterior samples and diagnostics
    """
    if config is None:
        config = CMCConfig()

    t_for_model = jnp.asarray(t_override) if t_override is not None else model.t

    logger.info(
        "[CMC] Starting analysis: chains=%d, samples=%d, warmup=%d",
        config.num_chains,
        config.num_samples,
        config.num_warmup,
    )

    start_time = time.perf_counter()

    # --- Phase 1: data preparation ---
    logger.info("[CMC] Phase 1/4: data preparation")
    c2_jax = jnp.asarray(c2_data)

    _n_total = int(c2_jax.size)
    _MAX_SINGLE_SHARD = 100_000
    if _n_total > _MAX_SINGLE_SHARD:
        logger.warning(
            "[CMC] Single-shard data has %d points (> %d). "
            "NUTS is O(n) per leapfrog step — consider fit_cmc_sharded for large datasets.",
            _n_total,
            _MAX_SINGLE_SHARD,
        )

    if sigma is None:
        sigma = estimate_sigma(c2_jax, method="diagonal")
        logger.info("[CMC] Estimated sigma = %.4e", float(jnp.mean(sigma)))

    sigma_jax = jnp.asarray(sigma) if isinstance(sigma, np.ndarray) else sigma
    # Scalar prior centre for the sampled sigma site (homodyne parity).
    noise_scale = float(jnp.mean(jnp.asarray(sigma_jax)))

    # --- Phase 2: model construction ---
    logger.info("[CMC] Phase 2/4: model construction")
    space = model.param_manager.space
    varying_names = model.param_manager.varying_names

    # Read fitted contrast/offset from model scaling (angle_idx=0 for per-angle CMC).
    # After NLSQ, model.scaling holds the fitted values; passing 1.0 defaults would
    # silently use the wrong scaling and bias the entire posterior.
    contrast, offset = model.scaling.get_for_angle(0)
    logger.info(
        "[CMC] Using contrast=%.4f, offset=%.4f from model scaling",
        contrast,
        offset,
    )

    logger.info("[CMC] Sampling %d parameters: %s", len(varying_names), varying_names)

    # Validate NLSQ warm-start
    use_reparam = config.use_reparam and nlsq_result is not None and nlsq_result.success
    # priors_override is a hard contract: caller wants these exact distributions.
    # The reparam path samples in z-space and ignores override → silent data loss.
    # Force the non-reparam path so the override is actually used.
    if priors_override is not None and use_reparam:
        logger.info(
            "[CMC] priors_override provided; disabling reparameterization so the "
            "caller-supplied distributions are sampled directly."
        )
        use_reparam = False
    if config.use_nlsq_warmstart and nlsq_result is None:
        logger.warning(
            "[CMC] NLSQ warm-start is enabled (use_nlsq_warmstart=True) but no "
            "nlsq_result was provided. Chains will initialize at the prior, which "
            "is typically 5-10σ from the true posterior for the 14-parameter "
            "heterodyne model. This produces R-hat >> 1 and ESS ≈ n_chains — "
            "effectively a failed run after hours of sampling. "
            "Pass nlsq_result= or use optimizer: both in the CLI config."
        )
    elif (
        config.use_nlsq_warmstart
        and nlsq_result is not None
        and not nlsq_result.success
    ):
        logger.warning(
            "[CMC] NLSQ warm-start requested but result is not converged "
            "(success=False); falling back to default initialization"
        )

    reparam_config = None
    scalings = None
    prior_std_dict: dict[str, float] = {}

    if use_reparam:
        t_array = (
            np.asarray(t_override) if t_override is not None else np.asarray(model.t)
        )
        dt_val = (
            float(t_array[1] - t_array[0]) if len(t_array) > 1 else float(t_array[0])
        )
        t_max_val = float(t_array[-1])
        t_ref = compute_t_ref(dt_val, t_max_val, fallback_value=1.0)

        # reparameterization_d_total controls both D0_ref/alpha_ref and
        # D0_sample/alpha_sample pairs; reparameterization_log_gamma controls v0/beta.
        reparam_config = ReparamConfig(
            t_ref=t_ref,
            enable_d_ref=config.reparameterization_d_total,
            enable_d_sample=config.reparameterization_d_total,
            enable_v_ref=config.reparameterization_log_gamma,
        )
        logger.info(
            "[CMC] Reference-time reparameterization: t_ref=%.4e "
            "(d_ref=%s, d_sample=%s, v_ref=%s)",
            t_ref,
            config.reparameterization_d_total,
            config.reparameterization_d_total,
            config.reparameterization_log_gamma,
        )

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

        reparam_values, reparam_uncertainties = transform_nlsq_to_reparam_space(
            nlsq_values,
            nlsq_uncertainties,
            t_ref,
            reparam_config,
        )

        from heterodyne.optimization.cmc.scaling import ParameterScaling

        scalings: dict[str, ParameterScaling] = {}
        prefactor_to_log: dict[str, str] = {}
        for prefactor, exponent in reparam_config.enabled_pairs:
            if prefactor in varying_names and exponent in varying_names:
                prefactor_to_log[prefactor] = reparam_config.get_reparam_name(prefactor)

        for name in varying_names:
            if name in prefactor_to_log:
                sname = prefactor_to_log[name]
            else:
                sname = name

            center = reparam_values.get(sname, space.values.get(name, 0.0))
            unc = reparam_uncertainties.get(sname, 0.0)
            scale = unc * config.nlsq_prior_width_factor if unc > 0 else 1.0
            scale = max(scale, 1e-10)
            scale = (
                scale * prior_width_multiplier
            )  # temper reparam prior width for CMC shards

            if sname.startswith("log_"):
                low = center - 10.0 * scale
                high = center + 10.0 * scale
            else:
                low, high = space.bounds[name]

            scalings[sname] = ParameterScaling(
                name=sname,
                center=center,
                scale=scale,
                low=low,
                high=high,
            )

        prior_std_dict = {}
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

        # priors_override is now handled by forcing use_reparam=False above,
        # so this branch only runs when no override is present.
        numpyro_model = get_heterodyne_model_reparam(
            t=t_for_model,
            q=model.q,
            dt=model.dt,
            phi_angle=phi_angle,
            c2_data=c2_jax,
            noise_scale=noise_scale,
            space=space,
            reparam_config=reparam_config,
            scalings=scalings,
            contrast=contrast,
            offset=offset,
        )
    else:
        numpyro_model = get_heterodyne_model(
            t=t_for_model,
            q=model.q,
            dt=model.dt,
            phi_angle=phi_angle,
            c2_data=c2_jax,
            noise_scale=noise_scale,
            space=space,
            contrast=contrast,
            offset=offset,
            priors_override=priors_override,
        )

    # --- Phase 3: sampling ---
    logger.info("[CMC] Phase 3/4: NUTS sampling")
    from numpyro.infer import initialization as numpyro_init

    _init_strategy_map = {
        "init_to_median": numpyro_init.init_to_median,
        "init_to_sample": numpyro_init.init_to_sample,
        "init_to_value": numpyro_init.init_to_value,
    }
    init_fn = _init_strategy_map.get(config.init_strategy, numpyro_init.init_to_median)

    # Elevate target acceptance for high-correlation regimes: when Z-space
    # reparameterization is active the power-law pair geometry is decorrelated
    # within each pair, but cross-pair correlations remain. A floor of 0.9
    # (matching homodyne's laminar-flow policy) improves leapfrog step quality.
    _MIN_TARGET_ACCEPT_REPARAM = 0.9
    effective_target_accept = (
        max(config.target_accept_prob, _MIN_TARGET_ACCEPT_REPARAM)
        if use_reparam
        else config.target_accept_prob
    )
    if use_reparam and effective_target_accept > config.target_accept_prob:
        logger.info(
            "[CMC] Elevating target_accept_prob %.2f → %.2f (reparam active)",
            config.target_accept_prob,
            effective_target_accept,
        )

    kernel = NUTS(
        numpyro_model,
        target_accept_prob=effective_target_accept,
        max_tree_depth=config.max_tree_depth,
        dense_mass=config.dense_mass,
        init_strategy=init_fn(),
    )

    rng_seed = config.seed if config.seed is not None else secrets.randbelow(2**31)

    init_params = None
    if config.use_nlsq_warmstart and nlsq_result is not None and nlsq_result.success:
        logger.info("[CMC] Using NLSQ result for chain initialization")

        if use_reparam and scalings is not None:
            init_params = {}
            perturb_key = jax.random.PRNGKey(rng_seed + 1)
            for sname, sc in scalings.items():
                perturb_key, subkey = jax.random.split(perturb_key)
                reparam_val = reparam_values.get(sname, sc.center)
                z_init = sc.to_normalized(reparam_val)
                base = jnp.full((config.num_chains,), jnp.float64(z_init))
                perturbation = 0.01 * jax.random.normal(
                    subkey, shape=(config.num_chains,)
                )
                init_params[f"{sname}_z"] = base + perturbation
            # sigma is sampled as a posterior site; initialise at prior centre so
            # NUTS init_strategy doesn't call the model without a seed handler.
            init_params["sigma"] = jnp.full(
                (config.num_chains,), jnp.float64(noise_scale * 1.5)
            )
        else:
            init_params = {}
            perturb_key = jax.random.PRNGKey(rng_seed + 1)
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
            init_params["sigma"] = jnp.full(
                (config.num_chains,), jnp.float64(noise_scale * 1.5)
            )

    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(rng_seed)

    try:
        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy", "diverging"))
        samples = _block_until_ready_pytree(mcmc.get_samples())
    except (RuntimeError, ValueError) as e:
        logger.error("[CMC] MCMC sampling failed: %s", e)
        return _create_failed_result(varying_names, str(e))

    # --- Phase 4: diagnostics and output ---
    sample_count = max(
        (np.asarray(values).shape[0] for values in samples.values()), default=0
    )
    logger.info(
        "[CMC] NUTS sampling complete: collected %d posterior draws", sample_count
    )
    logger.info("[CMC] Phase 4/4: diagnostics and result construction")
    # arviz_base.io_numpyro accesses numpyro.infer.initialization as an attribute.
    # Heterodyne's import chain loads the submodule but doesn't always register it
    # as a package attribute (Python import-order quirk). Set it explicitly.
    import sys as _sys

    import numpyro.infer as _numpyro_infer

    if not hasattr(_numpyro_infer, "initialization"):
        _init_mod = _sys.modules.get("numpyro.infer.initialization")
        if _init_mod is not None:
            _numpyro_infer.initialization = _init_mod
    idata = az.from_numpyro(mcmc)

    if use_reparam and reparam_config is not None:
        raw_samples = {k: np.asarray(v) for k, v in samples.items()}
        physics_samples = transform_to_physics_space(raw_samples, reparam_config)
        output_names = varying_names
        available_names = [n for n in output_names if n in idata.posterior]
        summary = (
            az.summary(idata, var_names=available_names, ci_prob=0.95)
            if available_names
            else None
        )
    else:
        physics_samples = {k: np.asarray(v) for k, v in samples.items()}
        output_names = varying_names
        summary = az.summary(idata, var_names=output_names, ci_prob=0.95)

    posterior_mean, posterior_std, r_hat, ess_bulk, ess_tail = _extract_posterior_stats(
        output_names,
        physics_samples,
        summary,
    )

    credible_intervals = _extract_credible_intervals(
        output_names, physics_samples, summary
    )

    bfmi, bfmi_compute_failed = _compute_bfmi(idata)

    samples_dict = {
        name: physics_samples[name] for name in output_names if name in physics_samples
    }
    map_estimate = posterior_mean.copy()

    wall_time = time.perf_counter() - start_time

    r_hat_finite = r_hat[~np.isnan(r_hat)]
    ess_finite = ess_bulk[~np.isnan(ess_bulk)]
    convergence_passed = bool(
        len(r_hat_finite) > 0
        and np.all(r_hat_finite < config.max_r_hat)
        and len(ess_finite) > 0
        and np.all(ess_finite > config.min_ess)
    )
    # BFMI is advisory only (homodyne parity). Homodyne check_convergence uses
    # R-hat + ESS as the sole hard gates. Applying BFMI as a hard gate here
    # silently kills all shards when chains start near a parameter boundary
    # (boundary reflection → low BFMI is expected and normal).
    if bfmi is not None and not bfmi_compute_failed:
        _min_bfmi = float(np.nanmin(np.asarray(bfmi, dtype=float)))
        if _min_bfmi < config.min_bfmi:
            logger.warning(
                "[CMC] Low BFMI=%.3f < %.2f — poor HMC energy exploration "
                "(advisory only; does not affect convergence gate)",
                _min_bfmi,
                config.min_bfmi,
            )
    if bfmi_compute_failed:
        logger.debug(
            "[CMC] BFMI unavailable (az.bfmi computation failed); "
            "convergence determined by R-hat and ESS only"
        )

    metadata: dict[str, Any] = {}
    # Store divergence_rate so fit_cmc_sharded can filter high-divergence shards
    # before consensus combination (CM-02 fix).
    _extra = mcmc.get_extra_fields()
    _div = _extra.get("diverging", None)
    if _div is not None:
        _div_arr = np.asarray(_div, dtype=bool)
        metadata["divergence_rate"] = (
            float(np.mean(_div_arr)) if _div_arr.size > 0 else 0.0
        )
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

    conv_report = validate_convergence(
        result, config.max_r_hat, config.min_ess, config.min_bfmi
    )
    for msg in conv_report.messages:
        logger.info(msg)

    logger.info(
        "[CMC] Complete in %.1fs, convergence: %s",
        wall_time,
        "PASSED" if convergence_passed else "FAILED",
    )

    # Divergence analysis (parity with homodyne)
    div_report = analyze_divergences(result)
    for msg in div_report.messages:
        logger.warning(msg)

    # Structured analysis summary (parity with homodyne)
    _r_hat_dict = (
        {n: float(result.r_hat[i]) for i, n in enumerate(result.parameter_names)}
        if result.r_hat is not None
        else {}
    )
    _ess_dict = (
        {n: float(result.ess_bulk[i]) for i, n in enumerate(result.parameter_names)}
        if result.ess_bulk is not None
        else {}
    )
    log_analysis_summary(
        convergence_status=result.convergence_status
        or ("converged" if result.convergence_passed else "not_converged"),
        r_hat=_r_hat_dict,
        ess_bulk=_ess_dict,
        divergences=result.divergences or 0,
        n_samples=config.num_samples,
        n_chains=config.num_chains,
        n_shards=1,
        shards_succeeded=1 if result.convergence_passed else 0,
        execution_time=wall_time,
    )

    return result


# ---------------------------------------------------------------------------
# Public: sharded CMC entry point
# ---------------------------------------------------------------------------


def fit_cmc_sharded(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: CMCConfig | None = None,
    sigma: np.ndarray | float | None = None,
    nlsq_result: NLSQResult | None = None,
    num_shards: int = 4,
    sharding_strategy: str = "random",
    shard_seed: int | None = None,
) -> CMCResult:
    """Fit heterodyne model using sharded Consensus Monte Carlo.

    Splits the observed c2 matrix into ``num_shards`` independent data
    subsets, runs NUTS on each shard sub-posterior (sequentially), then
    combines the shard posteriors via inverse-variance weighted consensus.

    Prior tempering is applied automatically: each shard's prior distribution
    is widened by ``sqrt(num_shards)`` (i.e., ``prior^(1/K)``) while sigma
    is passed unscaled. This is the correct Consensus Monte Carlo approach
    (Scott et al., 2016).

    Args:
        model: HeterodyneModel with configured parameters.
        c2_data: Observed two-time correlation matrix (N x N).
        phi_angle: Detector phi angle (degrees).
        config: CMC configuration (defaults to CMCConfig()).
        sigma: Measurement uncertainty (estimated if None).
        nlsq_result: Optional NLSQ result for warm-starting each shard.
        num_shards: Number of data shards (K). Must be >= 2.
        sharding_strategy: One of ``"random"`` (default) or
            ``"contiguous"``.  Random sharding breaks temporal
            autocorrelation between shards.  Contiguous sharding uses
            diagonal time-blocks, which preserves the two-time structure
            within each shard.
        shard_seed: Integer seed for deterministic shard assignment.
            If ``None``, a random seed is drawn from the OS.

    Returns:
        CMCResult with combined posterior and per-shard diagnostics stored
        in ``result.metadata["shard_diagnostics"]``.

    Raises:
        ValueError: If inputs fail validation or ``num_shards < 2``.
    """
    if config is None:
        config = CMCConfig()

    if num_shards < 2:
        raise ValueError(f"num_shards must be >= 2 for sharded CMC, got {num_shards}")

    # Validate inputs before touching JAX
    _validate_cmc_inputs(c2_data, sigma, model.param_manager.space)

    logger.info(
        "[CMC-sharded] Starting: %d shards, strategy=%s, chains=%d, samples=%d",
        num_shards,
        sharding_strategy,
        config.num_chains,
        config.num_samples,
    )

    start_time = time.perf_counter()

    # --- Phase 1: data preparation ---
    logger.info("[CMC-sharded] Phase 1/5: data preparation")
    c2_np = np.asarray(c2_data, dtype=np.float64)

    if sigma is None:
        sigma_jax_full = estimate_sigma(jnp.asarray(c2_np), method="diagonal")
        sigma_np: np.ndarray | float = np.asarray(sigma_jax_full)
        logger.info(
            "[CMC-sharded] Estimated sigma = %.4e",
            float(np.mean(np.asarray(sigma_np))),
        )
    else:
        sigma_np = np.asarray(sigma) if not isinstance(sigma, float) else sigma

    # --- Phase 2: shard creation ---
    logger.info("[CMC-sharded] Phase 2/5: creating %d shards", num_shards)
    effective_seed = shard_seed if shard_seed is not None else secrets.randbelow(2**31)
    shards = _create_shards(
        c2_np, sigma_np, num_shards, sharding_strategy, effective_seed
    )

    logger.info(
        "[CMC-sharded] Shards created: sizes=%s",
        [len(s["indices"]) for s in shards],
    )

    # --- Build tempered priors for CMC shards ---
    # Correct CMC tempering: widen prior by sqrt(K) per shard (prior^(1/K)).
    # Workers rebuild priors locally; _base_priors here is computed so that
    # build_nlsq_informed_priors / build_default_priors / temper_priors are
    # imported at module level so tests can patch them on this module.
    # Actual tempering flows via prior_width_multiplier passed to run_shards().

    _space = model.param_manager.space
    _scaling_active = [
        n for n in _space.varying_names if n not in _space.varying_physics_names
    ]
    if _scaling_active:
        logger.debug(
            "[CMC-sharded] ParameterSpace has scaling params active (%s); "
            "workers will use varying_physics_names (physics-only) for NUTS.",
            _scaling_active,
        )
    if nlsq_result is not None and nlsq_result.success:
        base_priors = build_nlsq_informed_priors(
            nlsq_result, _space, width_factor=config.nlsq_prior_width_factor
        )
    else:
        base_priors = build_default_priors(_space)
    _shard_priors = temper_priors(base_priors, num_shards)
    prior_width_mult = math.sqrt(num_shards)

    # --- Phase 3: per-shard sampling (parallel) ---
    base_seed = config.seed if config.seed is not None else secrets.randbelow(2**31)
    # Derive time axis from C2 matrix shape, not model.t. NLSQ trim calls
    # sync_time_axis(np.arange(1000)) which shrinks model.t to 1000 elements,
    # but CMC receives the full (1001×1001) C2 — shard t1_idx/t2_idx reach
    # index 1000, causing OOB if t_np is taken from the trimmed model.t.
    t_np = np.arange(c2_np.shape[0], dtype=np.float64)
    contrast, offset = model.scaling.get_for_angle(0)
    q_val = float(model.q)
    dt_val = float(model.dt)

    # Build reparameterization config for shard workers (parity with fit_cmc_jax).
    # Workers use reparam_config_dict to sample D0/alpha in log-space, which
    # greatly reduces the D0–alpha correlation and improves NUTS acceptance rate.
    # Requires an NLSQ warm-start to compute t_ref; falls back to None (raw space).
    _use_reparam = (
        config.use_reparam and nlsq_result is not None and nlsq_result.success
    )
    _reparam_config: ReparamConfig | None = None
    if _use_reparam:
        _t_max_val = float(t_np[-1]) if len(t_np) > 0 else 1.0
        _t_ref = compute_t_ref(dt_val, _t_max_val, fallback_value=1.0)
        _reparam_config = ReparamConfig(
            t_ref=_t_ref,
            enable_d_ref=config.reparameterization_d_total,
            enable_d_sample=config.reparameterization_d_total,
            enable_v_ref=config.reparameterization_log_gamma,
        )
        logger.info(
            "[CMC-sharded] Reparameterization enabled: t_ref=%.4e "
            "(d_ref=%s, d_sample=%s, v_ref=%s)",
            _t_ref,
            config.reparameterization_d_total,
            config.reparameterization_d_total,
            config.reparameterization_log_gamma,
        )
    _reparam_config_dict: dict[str, Any] | None = (
        {
            "enable_d_ref": _reparam_config.enable_d_ref,
            "enable_d_sample": _reparam_config.enable_d_sample,
            "enable_v_ref": _reparam_config.enable_v_ref,
            "t_ref": _reparam_config.t_ref,
        }
        if _reparam_config is not None
        else None
    )

    # Translate _create_shards output format to the dict format run_shards() expects.
    # Two wire formats are supported depending on the sharding strategy:
    #
    # Element-wise format (random strategy, t1_idx/t2_idx present):
    #   "t1"/"t2" — per-pair time values; "time_grid" — full axis for ShardGrid
    #   Worker uses compute_c2_elementwise → 1-D output matching flat c2_data.
    #
    # Meshgrid format (contiguous strategy, t_indices present):
    #   "t" — 1-D time axis for the shard block
    #   Worker uses compute_c2_heterodyne → 2-D output matching square c2_data.
    parallel_shards: list[dict[str, Any]] = []
    for shard in shards:
        _sigma_arr = shard["sigma_shard"]
        _sigma_wire = (
            np.asarray(_sigma_arr) if not isinstance(_sigma_arr, float) else _sigma_arr
        )
        _noise_scale = float(
            np.mean(np.asarray(_sigma_arr))
            if not isinstance(_sigma_arr, float)
            else _sigma_arr
        )
        _base: dict[str, Any] = {
            "c2_data": np.asarray(shard["c2_shard"]),
            "sigma": _sigma_wire,
            "noise_scale": _noise_scale,
            "q": q_val,
            "dt": dt_val,
            "phi_angle": phi_angle,
            "contrast": float(contrast),
            "offset": float(offset),
            "n_phi": 1,
            "reparam_config_dict": _reparam_config_dict,
        }
        if "t1_idx" in shard:
            # Element-wise (random): pass paired time values + full axis
            _base["t1"] = t_np[shard["t1_idx"]]
            _base["t2"] = t_np[shard["t2_idx"]]
            _base["time_grid"] = t_np
        else:
            # Meshgrid (contiguous): pass 1-D sub-axis
            _base["t"] = t_np[shard["t_indices"]]
        parallel_shards.append(_base)

    # NLSQ warm-start values passed to workers for chain initialisation.
    initial_values: dict[str, Any] | None = None
    nlsq_uncertainties_dict: dict[str, float] | None = None
    if nlsq_result is not None and nlsq_result.success:
        initial_values = {
            name: float(nlsq_result.get_param(name))
            for name in nlsq_result.parameter_names
        }
        # Also pass uncertainties so each worker can build NLSQ-informed
        # (TruncatedNormal centered on NLSQ value) priors locally and apply
        # CMC tempering on TOP of them. Without this, workers fall back to
        # registry defaults and the NLSQ posterior contraction is lost.
        unc_dict: dict[str, float] = {}
        for name in nlsq_result.parameter_names:
            unc = nlsq_result.get_uncertainty(name)
            if unc is not None and float(unc) > 0:
                unc_dict[name] = float(unc)
        nlsq_uncertainties_dict = unc_dict if unc_dict else None

    # Log rough runtime estimate before blocking and warn if it exceeds timeout.
    avg_pts = sum(int(np.asarray(s["c2_data"]).size) for s in parallel_shards) // max(
        num_shards, 1
    )
    _n_workers = _estimate_n_workers()
    _estimated_total = _log_runtime_estimate(
        logger,
        n_shards=num_shards,
        n_chains=config.num_chains,
        n_warmup=config.num_warmup,
        n_samples=config.num_samples,
        avg_points_per_shard=avg_pts,
        n_workers=_n_workers,
    )
    _batches = (num_shards + _n_workers - 1) // _n_workers
    _estimated_per_shard = _estimated_total / max(_batches, 1)
    if _estimated_per_shard > config.per_shard_timeout:
        logger.warning(
            "[CMC-sharded] Estimated per-shard time (%.0fs = %.1fh) exceeds "
            "per_shard_timeout=%ds. Shards will likely timeout. "
            "avg_points_per_shard=%d exceeds the ~100K NUTS limit. "
            "Use num_shards='auto' or reduce max_points_per_shard.",
            _estimated_per_shard,
            _estimated_per_shard / 3600,
            config.per_shard_timeout,
            avg_pts,
        )

    # Early abort guard: without an NLSQ warm-start, NUTS starts from the
    # default prior (identity mass matrix) and must discover the posterior
    # geometry from scratch during warmup.  For the 14-parameter heterodyne
    # model with shards >10K points, warmup alone exceeds 7200s because NUTS
    # saturates max_tree_depth (1024 leapfrog steps) on every iteration.
    # This is the het_c7548ee8 failure mode: all 47 shards timeout with 0
    # posterior samples collected.
    _NO_NLSQ_SHARD_LIMIT = 10_000
    if nlsq_result is None and avg_pts > _NO_NLSQ_SHARD_LIMIT:
        # Hard abort: 3 separate runs (het_c7548ee8, het_e34fa942, het_dd0f825b)
        # prove that CMC without NLSQ on >10K-point shards ALWAYS timeouts after
        # 8+ hours with 0 posterior samples. NUTS must discover a 14-parameter
        # posterior geometry from scratch (identity mass matrix, no warm start).
        # Warmup alone saturates max_tree_depth=10 (1024 leapfrog steps/step)
        # on every iteration, far exceeding per_shard_timeout=7200s.
        # Abort immediately to prevent silent 8-hour waste.
        raise RuntimeError(
            f"[CMC-sharded] Aborting: no NLSQ warm-start provided and "
            f"avg_points_per_shard={avg_pts} > {_NO_NLSQ_SHARD_LIMIT}. "
            f"Without a warm-start, all {num_shards} shards will timeout "
            f"({config.per_shard_timeout}s) with 0 posterior samples collected. "
            "Fix: run NLSQ first (optimizer: nlsq) then re-run CMC, or use "
            "optimizer: both to run NLSQ→CMC in one pass. "
            "To override (e.g. for small pilot runs), reduce max_points_per_shard "
            f"below {_NO_NLSQ_SHARD_LIMIT} in the CMC config."
        )

    logger.info(
        "[CMC-sharded] Phase 3/5: dispatching %d shards to MultiprocessingBackend",
        num_shards,
    )

    from heterodyne.optimization.cmc.backends.multiprocessing_backend import (
        MultiprocessingBackend,
    )

    _backend = MultiprocessingBackend()
    raw_results = _backend.run_shards(
        shards=parallel_shards,
        config=config,
        initial_values=initial_values,
        parameter_space=_space,
        prior_width_multiplier=prior_width_mult,
        nlsq_uncertainties=nlsq_uncertainties_dict
        if config.use_nlsq_informed_priors
        else None,
        nlsq_prior_width_factor=float(config.nlsq_prior_width_factor),
        progress_bar=True,
    )

    # Convert worker result dicts → CMCResult objects for _combine_shard_posteriors().
    shard_results: list[CMCResult] = [
        _result_dict_to_cmc_result(r, config) for r in raw_results
    ]

    # Pad with failed placeholders for any shards dropped by run_shards() (timeout/crash).
    if len(shard_results) < num_shards:
        n_missing = num_shards - len(shard_results)
        logger.warning(
            "[CMC-sharded] %d/%d shards failed or timed out",
            n_missing,
            num_shards,
        )
        _fallback_names: list[str] = (
            list(raw_results[0]["param_names"])
            if raw_results
            else list(_space.varying_names)
        )
        for _ in range(n_missing):
            shard_results.append(
                _create_failed_result(_fallback_names, "shard failed or timed out")
            )

    # --- Phase 4: consensus combination ---
    logger.info("[CMC-sharded] Phase 4/5: combining shard posteriors (consensus)")
    combined_result = _combine_shard_posteriors(
        shard_results,
        config,
        num_shards,
        base_seed,
    )

    # --- Bimodal detection across shards ---
    # Run after combination so the combine path's Gaussian approximation
    # can be checked for mode collapse. Results stored in metadata only —
    # the caller gets a normal CMCResult but can inspect metadata["bimodal"].
    bimodal_metadata: dict[str, Any] = {}
    successful_with_samples = [
        sr for sr in shard_results if sr.convergence_passed and sr.samples is not None
    ]
    if len(successful_with_samples) >= 2:
        from heterodyne.optimization.cmc.diagnostics import check_shard_bimodality

        shard_sample_dict = {
            i: sr.samples  # type: ignore[misc]
            for i, sr in enumerate(successful_with_samples)
        }
        bimodal_results = check_shard_bimodality(
            shard_sample_dict,
            min_weight=config.bimodal_min_weight,
            min_separation=config.bimodal_min_separation,
        )
        bimodal_params = [
            p for p, rs in bimodal_results.items() if any(r.is_bimodal for r in rs)
        ]
        bimodal_metadata["bimodal_detected"] = len(bimodal_params) > 0
        bimodal_metadata["bimodal_params"] = bimodal_params
        if bimodal_params:
            logger.warning(
                "[CMC-sharded] Bimodal posteriors detected for %d parameters: %s. "
                "Gaussian consensus approximation may be inaccurate.",
                len(bimodal_params),
                bimodal_params,
            )

    # --- Phase 5: finalize ---
    wall_time = time.perf_counter() - start_time
    logger.info(
        "[CMC-sharded] Phase 5/5: finalizing (total wall time=%.1fs)", wall_time
    )

    # Attach per-shard diagnostics to metadata
    shard_diagnostics = [
        {
            "convergence_passed": r.convergence_passed,
            "r_hat": r.r_hat.tolist() if r.r_hat is not None else None,
            "ess_bulk": r.ess_bulk.tolist() if r.ess_bulk is not None else None,
            "bfmi": r.bfmi,
            "wall_time_seconds": r.wall_time_seconds,
        }
        for r in shard_results
    ]

    metadata = dict(combined_result.metadata)
    metadata["num_shards"] = num_shards
    metadata["sharding_strategy"] = sharding_strategy
    metadata["shard_seed"] = effective_seed
    metadata["shard_diagnostics"] = shard_diagnostics
    metadata["n_failed_shards"] = sum(
        1 for r in shard_results if not r.convergence_passed
    )
    metadata.update(bimodal_metadata)

    final = CMCResult(
        parameter_names=combined_result.parameter_names,
        posterior_mean=combined_result.posterior_mean,
        posterior_std=combined_result.posterior_std,
        credible_intervals=combined_result.credible_intervals,
        convergence_passed=combined_result.convergence_passed,
        r_hat=combined_result.r_hat,
        ess_bulk=combined_result.ess_bulk,
        ess_tail=combined_result.ess_tail,
        bfmi=combined_result.bfmi,
        samples=combined_result.samples,
        map_estimate=combined_result.map_estimate,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples * num_shards,
        num_chains=config.num_chains,
        wall_time_seconds=wall_time,
        metadata=metadata,
    )

    logger.info(
        "[CMC-sharded] Complete in %.1fs, convergence: %s, failed shards: %d/%d",
        wall_time,
        "PASSED" if final.convergence_passed else "FAILED",
        metadata["n_failed_shards"],
        num_shards,
    )

    return final


# ---------------------------------------------------------------------------
# Shard creation
# ---------------------------------------------------------------------------


def _create_shards(
    c2_np: np.ndarray,
    sigma_np: np.ndarray | float,
    num_shards: int,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Partition correlation data into shards for Consensus Monte Carlo.

    Two strategies are supported:

    - ``"random"``: randomly shuffles the flat index set of the upper
      triangle (including diagonal), then cuts into equal-sized groups.
      Each shard receives a sub-matrix assembled from its assigned
      element indices.  This breaks temporal autocorrelation across
      shards.

    - ``"contiguous"``: partitions the time axis into equal-width
      contiguous blocks and takes the diagonal sub-matrix for each
      block.  Preserves the two-time structure within each shard.

    In both cases the per-shard sigma is sliced to match the shard shape.

    Args:
        c2_np: Full two-time correlation matrix, shape (N, N), float64.
        sigma_np: Uncertainty array of the same shape as ``c2_np``, or a
            scalar float.  Scalar sigma is broadcast per shard.
        num_shards: Number of partitions K.
        strategy: ``"random"`` or ``"contiguous"``.
        seed: Integer seed for reproducible random shard assignment.

    Returns:
        List of K dicts, each containing:

        - ``"c2_shard"``: JAX array of shape (n_shard_times, n_shard_times)
          or (n_elements,) depending on strategy.
        - ``"sigma_shard"``: matching uncertainty array or scalar.
        - ``"indices"``: 1-D NumPy array of flat matrix indices assigned
          to this shard (for auditing and reconstruction).

    Raises:
        ValueError: If ``strategy`` is not ``"random"`` or ``"contiguous"``.
    """
    if strategy not in {"random", "contiguous"}:
        raise ValueError(
            f"sharding_strategy must be 'random' or 'contiguous', got '{strategy}'"
        )

    n = c2_np.shape[0]
    sigma_is_scalar = isinstance(sigma_np, float) or (
        isinstance(sigma_np, np.ndarray) and sigma_np.ndim == 0
    )

    if strategy == "random":
        return _create_shards_random(
            c2_np, sigma_np, sigma_is_scalar, num_shards, seed, n
        )

    # contiguous: diagonal blocks along the time axis
    return _create_shards_contiguous(c2_np, sigma_np, sigma_is_scalar, num_shards, n)


def _create_shards_random(
    c2_np: np.ndarray,
    sigma_np: np.ndarray | float,
    sigma_is_scalar: bool,
    num_shards: int,
    seed: int,
    n: int,
) -> list[dict[str, Any]]:
    """Random element-wise sharding — flat per-pair representation.

    Each shard contains a 1-D array of selected c2 values together with their
    (row, col) time indices.  The worker builds a ShardGrid from these indices
    and calls ``compute_c2_elementwise``, which avoids the O(N²) meshgrid
    allocation and produces a 1-D prediction that matches the flat c2 data.

    Previous implementation reconstructed a (shard_n, shard_n) zero-padded
    sub-matrix, which caused two bugs:
      1. Shape metadata was lost during shared-memory serialisation, producing
         a (1002001,) flat array in the worker instead of (1001, 1001).
      2. ``compute_c2_heterodyne`` (NLSQ meshgrid path) returned (N, N) while
         the obs array had the wrong shape → BroadcastError.
    """
    rng = np.random.default_rng(seed)

    # All N² flat indices (symmetric matrix — use every pair, not just triu)
    all_indices = np.arange(n * n, dtype=np.int64)
    rng.shuffle(all_indices)
    splits = np.array_split(all_indices, num_shards)

    shards: list[dict[str, Any]] = []
    for split_indices in splits:
        rows, cols = np.divmod(split_indices, n)
        c2_flat = c2_np[rows, cols]
        sigma_flat: np.ndarray | float = (
            float(sigma_np) if sigma_is_scalar else np.asarray(sigma_np)[rows, cols]  # type: ignore[arg-type]
        )
        shards.append(
            {
                "c2_shard": c2_flat,  # shape (n_elements,)
                "sigma_shard": sigma_flat,
                "indices": split_indices,
                "t1_idx": rows.astype(np.int64),  # row time indices
                "t2_idx": cols.astype(np.int64),  # col time indices
            }
        )

    return shards


def _create_shards_contiguous(
    c2_np: np.ndarray,
    sigma_np: np.ndarray | float,
    sigma_is_scalar: bool,
    num_shards: int,
    n: int,
) -> list[dict[str, Any]]:
    """Contiguous diagonal-block sharding along the time axis."""
    boundaries = np.linspace(0, n, num_shards + 1, dtype=int)

    shards: list[dict[str, Any]] = []
    for i in range(num_shards):
        start = int(boundaries[i])
        stop = int(boundaries[i + 1])

        c2_block = c2_np[start:stop, start:stop]

        if sigma_is_scalar:
            sigma_block: np.ndarray | float = float(sigma_np)  # type: ignore[arg-type]
        else:
            sigma_block = np.asarray(sigma_np)[start:stop, start:stop]

        # Flat index set for this diagonal block
        row_idx, col_idx = np.meshgrid(
            np.arange(start, stop), np.arange(start, stop), indexing="ij"
        )
        flat_indices = (row_idx * n + col_idx).ravel().astype(np.int64)

        shards.append(
            {
                "c2_shard": jnp.asarray(c2_block),
                "sigma_shard": sigma_block,
                "indices": flat_indices,
                "t_indices": np.arange(start, stop, dtype=np.int64),
            }
        )

    return shards


# ---------------------------------------------------------------------------
# Posterior combination
# ---------------------------------------------------------------------------


def _combine_shard_posteriors(
    shard_results: list[CMCResult],
    config: CMCConfig,
    num_shards: int,
    base_seed: int,
) -> CMCResult:
    """Combine per-shard posteriors via inverse-variance weighted consensus.

    Implements the Consensus Monte Carlo estimator (Scott et al., 2016):

    .. math::

        \\mu^* = \\left(\\sum_k \\Sigma_k^{-1}\\right)^{-1}
                 \\sum_k \\Sigma_k^{-1} \\mu_k

    where :math:`\\mu_k` and :math:`\\Sigma_k` are the per-shard posterior
    mean and (diagonal) variance.  The combined variance is:

    .. math::

        \\Sigma^* = \\left(\\sum_k \\Sigma_k^{-1}\\right)^{-1}

    The worst-case R-hat across shards is used as the combined convergence
    diagnostic.  Combined ESS is the sum of per-shard ESS values (an
    approximation; true combined ESS would require cross-shard
    autocorrelation analysis).

    Args:
        shard_results: List of CMCResult objects, one per shard.  All
            must share the same ``parameter_names``.
        config: Global CMC config for convergence thresholds.
        num_shards: Number of shards (used only for logging).
        base_seed: Base random seed (not used here; reserved for future
            importance-resampling extension).

    Returns:
        CMCResult with combined posterior statistics.

    Raises:
        ValueError: If ``shard_results`` is empty or parameter names
            are inconsistent across shards.
    """
    if not shard_results:
        raise ValueError("shard_results must be non-empty")

    param_names = shard_results[0].parameter_names
    for i, sr in enumerate(shard_results[1:], start=1):
        if sr.parameter_names != param_names:
            raise ValueError(
                f"Shard {i} parameter_names mismatch: "
                f"expected {param_names}, got {sr.parameter_names}"
            )

    n_params = len(param_names)

    # --- Filter: exclude failed/degenerate/high-divergence shards ---
    _max_div_rate = getattr(config, "max_divergence_rate", 0.10)

    def _shard_has_valid_samples(sr: CMCResult) -> bool:
        return sr.posterior_std is not None and bool(np.any(sr.posterior_std > 0))

    def _shard_diagnostics_unknown(sr: CMCResult) -> bool:
        # r_hat all-NaN means ArviZ failed to build InferenceData (e.g. API mismatch)
        # but NUTS samples were collected — accept the shard on raw-sample basis.
        return sr.r_hat is not None and bool(np.all(np.isnan(sr.r_hat)))

    successful = [
        sr
        for sr in shard_results
        if _shard_has_valid_samples(sr)
        and getattr(sr, "metadata", {}).get("divergence_rate", 0.0) <= _max_div_rate
        and (sr.convergence_passed or _shard_diagnostics_unknown(sr))
    ]
    n_diag_unknown = sum(1 for sr in successful if not sr.convergence_passed)
    if n_diag_unknown > 0:
        logger.warning(
            "_combine_shard_posteriors: %d/%d accepted shards have unknown convergence "
            "(ArviZ diagnostics unavailable); combining on raw-sample basis",
            n_diag_unknown,
            len(successful),
        )

    if not successful:
        _n_no_samples = sum(
            1 for sr in shard_results if not _shard_has_valid_samples(sr)
        )
        _n_high_div = sum(
            1
            for sr in shard_results
            if _shard_has_valid_samples(sr)
            and getattr(sr, "metadata", {}).get("divergence_rate", 0.0) > _max_div_rate
        )
        _div_rates = [
            getattr(sr, "metadata", {}).get("divergence_rate")
            for sr in shard_results
            if _shard_has_valid_samples(sr)
        ]
        _finite_rates = [r for r in _div_rates if r is not None]
        _mean_div = float(np.mean(_finite_rates)) if _finite_rates else float("nan")
        logger.error(
            "_combine_shard_posteriors: all %d shards failed "
            "(no_samples=%d, high_divergence[>%.0f%%]=%d, bad_convergence=%d, "
            "mean_divergence_rate=%.1f%%) — "
            "if divergence is high, a warm-start parameter may be at a hard bound; "
            "returning degenerate result",
            len(shard_results),
            _n_no_samples,
            _max_div_rate * 100,
            _n_high_div,
            len(shard_results) - _n_no_samples - _n_high_div,
            _mean_div * 100,
        )
        return CMCResult(
            parameter_names=param_names,
            posterior_mean=np.zeros(n_params),
            posterior_std=np.full(n_params, np.nan),
            credible_intervals={},
            convergence_passed=False,
            r_hat=np.full(n_params, np.nan),
            ess_bulk=np.full(n_params, np.nan),
            ess_tail=np.full(n_params, np.nan),
            bfmi=None,
            samples=None,
            map_estimate=None,
            num_warmup=shard_results[0].num_warmup,
            num_samples=shard_results[0].num_samples,
            num_chains=shard_results[0].num_chains,
            wall_time_seconds=None,
            metadata={"all_shards_failed": True, "n_total_shards": len(shard_results)},
        )

    n_skipped = len(shard_results) - len(successful)
    if n_skipped > 0:
        logger.warning(
            "_combine_shard_posteriors: skipping %d failed/high-divergence shards "
            "(%d/%d successful remain)",
            n_skipped,
            len(successful),
            len(shard_results),
        )

    # --- Heterogeneity check: IQR-based CV, robust to near-zero parameters ---
    # α_ref, β, v_offset, φ₀ all default to ~0; std/|mean| diverges at zero.
    # IQR / max(|median|, 1e-3) stays finite and comparable across all params.
    if len(successful) >= 2:
        _smeans = np.stack([sr.posterior_mean for sr in successful], axis=0)
        _q75, _q25 = np.percentile(_smeans, [75, 25], axis=0)
        _iqr = _q75 - _q25
        _denom = np.maximum(np.abs(np.median(_smeans, axis=0)), 1e-3)
        _cv_robust = _iqr / _denom
        _max_cv_actual = float(np.max(_cv_robust))
        _cfg_max_cv = getattr(config, "max_parameter_cv", 1.0)
        if _max_cv_actual > _cfg_max_cv:
            _worst = param_names[int(np.argmax(_cv_robust))]
            _msg = (
                f"High cross-shard heterogeneity: max IQR-CV={_max_cv_actual:.2f} "
                f"(threshold {_cfg_max_cv}) on parameter {_worst!r}. "
                "Consider increasing min_points_per_shard or using NLSQ warm-start."
            )
            if getattr(config, "heterogeneity_abort", False):
                raise RuntimeError(_msg)
            logger.warning("_combine_shard_posteriors: %s", _msg)

    combination_method = (
        getattr(config, "combination_method", "consensus_mc") or "consensus_mc"
    )

    _known_methods = frozenset(
        {"consensus_mc", "simple_average", "robust_consensus_mc", "weighted_gaussian"}
    )
    if combination_method not in _known_methods:
        logger.warning(
            "_combine_shard_posteriors: unknown combination_method %r; "
            "falling back to inverse-variance (consensus_mc).",
            combination_method,
        )

    if combination_method == "simple_average":
        # Equal-weight mean and variance across shards
        combined_mean = np.mean(
            np.stack([sr.posterior_mean for sr in successful], axis=0), axis=0
        )
        combined_var = np.mean(
            np.stack([sr.posterior_std**2 for sr in successful], axis=0), axis=0
        )
        combined_std = np.sqrt(combined_var)
    elif combination_method == "robust_consensus_mc" and len(successful) >= 2:
        # Per-parameter z-score outlier detection before inverse-variance combination.
        # scale = max(std, 1e-4*(|mean|+1)) keeps near-zero params (α_ref, β,
        # v_offset, φ₀ ≈ 0) from producing infinite z-scores.
        _smeans = np.stack([sr.posterior_mean for sr in successful], axis=0)  # (K,P)
        _center = np.mean(_smeans, axis=0)
        _scale = np.maximum(
            np.std(_smeans, axis=0),
            1e-4 * (np.abs(_center) + 1.0),
        )
        _z_max = np.max(np.abs(_smeans - _center) / _scale, axis=1)  # (K,)
        _inlier = _z_max <= 3.0
        _n_excl = int(np.sum(~_inlier))
        if _n_excl > 0:
            logger.warning(
                "_combine_shard_posteriors: robust_consensus_mc excluded "
                "%d/%d outlier shards (z-score > 3)",
                _n_excl,
                len(successful),
            )
        _pool = [
            sr for sr, keep in zip(successful, _inlier.tolist(), strict=False) if keep
        ] or successful
        weight_sum = np.zeros(n_params)
        weighted_mean_sum = np.zeros(n_params)
        for sr in _pool:
            var_k = sr.posterior_std**2
            w_k = 1.0 / var_k
            weight_sum += w_k
            weighted_mean_sum += w_k * sr.posterior_mean
        combined_mean = weighted_mean_sum / np.where(weight_sum > 0, weight_sum, 1.0)
        combined_var = 1.0 / np.where(weight_sum > 0, weight_sum, 1.0)
        combined_std = np.sqrt(combined_var)
    else:
        # Default: inverse-variance weighting (consensus_mc / fallback)
        # Weight_k = 1 / Var_k (per-parameter, diagonal approximation)
        weight_sum = np.zeros(n_params)
        weighted_mean_sum = np.zeros(n_params)
        for sr in successful:
            var_k = sr.posterior_std**2
            w_k = 1.0 / var_k  # no clip needed — zero-std shards excluded above
            weight_sum += w_k
            weighted_mean_sum += w_k * sr.posterior_mean
        combined_mean = weighted_mean_sum / np.where(weight_sum > 0, weight_sum, 1.0)
        combined_var = 1.0 / np.where(weight_sum > 0, weight_sum, 1.0)
        combined_std = np.sqrt(combined_var)

    # --- Worst-case R-hat (conservative) ---
    r_hat_arrays = [sr.r_hat for sr in successful if sr.r_hat is not None]
    combined_r_hat = (
        np.nanmax(np.stack(r_hat_arrays, axis=0), axis=0)
        if r_hat_arrays
        else np.full(n_params, np.nan)
    )

    # --- Summed ESS (approximate) ---
    ess_bulk_arrays = [sr.ess_bulk for sr in successful if sr.ess_bulk is not None]
    combined_ess_bulk = (
        np.nansum(np.stack(ess_bulk_arrays, axis=0), axis=0)
        if ess_bulk_arrays
        else np.full(n_params, np.nan)
    )

    ess_tail_arrays = [sr.ess_tail for sr in successful if sr.ess_tail is not None]
    combined_ess_tail = (
        np.nansum(np.stack(ess_tail_arrays, axis=0), axis=0)
        if ess_tail_arrays
        else np.full(n_params, np.nan)
    )

    # --- BFMI: minimum across all shards and all chains ---
    all_bfmi_values: list[float] = []
    for sr in successful:
        if sr.bfmi is not None:
            all_bfmi_values.extend(sr.bfmi)
    combined_bfmi = all_bfmi_values if all_bfmi_values else None

    # --- Credible intervals from combined samples ---
    # Pool samples across shards for each parameter
    combined_samples: dict[str, np.ndarray] = {}
    if all(sr.samples is not None for sr in successful):
        for name in param_names:
            arrays = [
                np.asarray(sr.samples[name])  # type: ignore[index]
                for sr in successful
                if sr.samples is not None and name in sr.samples
            ]
            if arrays:
                combined_samples[name] = np.concatenate(arrays, axis=0)

    credible_intervals: dict[str, dict[str, float]] = {}
    for name in param_names:
        if name in combined_samples:
            s = combined_samples[name]
            z95 = float(np.percentile(s, 97.5))
            l95 = float(np.percentile(s, 2.5))
            z89 = float(np.percentile(s, 94.5))
            l89 = float(np.percentile(s, 5.5))
            credible_intervals[name] = {
                "lower_95": l95,
                "upper_95": z95,
                "lower_89": l89,
                "upper_89": z89,
            }

    # --- MAP estimate ---
    map_estimate = combined_mean.copy()

    # --- Convergence gate ---
    # Only evaluate diagnostics over the successful shards (failed shards are
    # already excluded; any skipped shard is reflected in n_skipped above).
    r_hat_finite = combined_r_hat[~np.isnan(combined_r_hat)]
    ess_finite = combined_ess_bulk[~np.isnan(combined_ess_bulk)]
    convergence_passed = bool(
        len(successful) > 0
        and len(r_hat_finite) > 0
        and np.all(r_hat_finite < config.max_r_hat)
        and len(ess_finite) > 0
        and np.all(ess_finite > config.min_ess)
    )
    # BFMI is advisory for combined result (homodyne parity).
    # Low combined BFMI is a useful warning but not a hard gate — the
    # combined R-hat and ESS from pooled shards are the authoritative signal.
    if combined_bfmi is not None:
        _combined_min_bfmi = float(np.nanmin(np.asarray(combined_bfmi, dtype=float)))
        if _combined_min_bfmi < config.min_bfmi:
            logger.warning(
                "_combine_shard_posteriors: low combined BFMI=%.3f < %.2f "
                "(advisory; does not override R-hat/ESS convergence gate). "
                "Consider reparameterization or wider parameter bounds.",
                _combined_min_bfmi,
                config.min_bfmi,
            )

    logger.info(
        "[CMC-sharded] Consensus combination: %d/%d shards converged "
        "(%d skipped/failed), worst_rhat=%.3f, combined_ess_min=%.0f",
        len(successful),
        num_shards,
        n_skipped,
        float(np.nanmax(combined_r_hat)) if combined_r_hat.size > 0 else float("nan"),
        float(np.nanmin(combined_ess_bulk))
        if combined_ess_bulk.size > 0
        else float("nan"),
    )

    return CMCResult(
        parameter_names=param_names,
        posterior_mean=combined_mean,
        posterior_std=combined_std,
        credible_intervals=credible_intervals,
        convergence_passed=convergence_passed,
        r_hat=combined_r_hat,
        ess_bulk=combined_ess_bulk,
        ess_tail=combined_ess_tail,
        bfmi=combined_bfmi,
        samples=combined_samples if combined_samples else None,
        map_estimate=map_estimate,
        num_warmup=shard_results[0].num_warmup,
        num_samples=shard_results[0].num_samples,
        num_chains=shard_results[0].num_chains,
        wall_time_seconds=None,  # caller fills this in
        metadata={},
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_cmc_inputs(
    c2_data: np.ndarray | jnp.ndarray,
    sigma: np.ndarray | float | None,
    space: Any,
) -> None:
    """Validate inputs before starting any CMC analysis.

    Checks performed:

    1. ``c2_data`` is 2-D and square (the heterodyne two-time matrix
       is always N x N).
    2. ``c2_data`` does not contain NaN or Inf.
    3. ``c2_data`` is approximately symmetric:
       ``max |c2 - c2.T| / max |c2| < 1e-3``.
    4. If ``sigma`` is provided as an array, it is strictly positive
       and has no NaN values.
    5. The parameter space has at least one varying parameter.

    Args:
        c2_data: Observed correlation matrix.
        sigma: Measurement uncertainty, or ``None``.
        space: ParameterSpace object with ``varying_names`` and ``bounds``.

    Raises:
        ValueError: On any validation failure.
    """
    c2_np = np.asarray(c2_data)

    # 1. Shape
    if c2_np.ndim != 2:
        raise ValueError(
            f"c2_data must be 2-D for CMC analysis, got {c2_np.ndim}-D "
            f"with shape {c2_np.shape}"
        )
    if c2_np.shape[0] != c2_np.shape[1]:
        raise ValueError(
            f"c2_data must be square (heterodyne two-time matrix), "
            f"got shape {c2_np.shape}"
        )

    # 2. NaN / Inf
    n_nan = int(np.sum(np.isnan(c2_np)))
    if n_nan > 0:
        raise ValueError(f"c2_data contains {n_nan} NaN values; clean data before CMC")
    n_inf = int(np.sum(np.isinf(c2_np)))
    if n_inf > 0:
        raise ValueError(f"c2_data contains {n_inf} Inf values; clean data before CMC")

    # 3. Approximate symmetry
    max_abs = float(np.max(np.abs(c2_np)))
    if max_abs > 0:
        asymmetry = float(np.max(np.abs(c2_np - c2_np.T))) / max_abs
        if asymmetry > 1e-3:
            raise ValueError(
                f"c2_data is not approximately symmetric: "
                f"max |c2 - c2.T| / max |c2| = {asymmetry:.4e} > 1e-3. "
                "The heterodyne two-time matrix must be symmetric."
            )

    # 4. Sigma
    if sigma is not None and not isinstance(sigma, float):
        sigma_np = np.asarray(sigma)
        if np.any(sigma_np <= 0):
            raise ValueError("sigma array must be strictly positive everywhere")
        n_nan_sigma = int(np.sum(np.isnan(sigma_np)))
        if n_nan_sigma > 0:
            raise ValueError(f"sigma contains {n_nan_sigma} NaN values")

    # 5. Parameter space
    if not hasattr(space, "varying_names") or len(space.varying_names) == 0:
        raise ValueError(
            "Parameter space has no varying parameters; "
            "at least one parameter must be free for CMC"
        )

    logger.debug(
        "[CMC] Input validation passed: shape=%s, n_varying=%d",
        c2_np.shape,
        len(space.varying_names),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_shard_config(config: CMCConfig, seed: int) -> CMCConfig:
    """Return a copy of ``config`` with a new seed.

    All other fields are preserved exactly.  The shard-level config
    intentionally keeps the same warmup/sample counts and NUTS hyper-
    parameters as the full-data config.

    Args:
        config: Original CMCConfig.
        seed: New integer seed for this shard's sampling run.

    Returns:
        New CMCConfig instance with ``seed`` replaced.
    """
    import dataclasses

    return dataclasses.replace(config, seed=seed)


def _estimate_n_workers() -> int:
    """Estimate the number of parallel workers the MultiprocessingBackend will use."""
    import multiprocessing as _mp

    try:
        logical = _mp.cpu_count() or 1
    except NotImplementedError:
        logical = 4
    return max(1, logical // 2 - 1)


def _fmt_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}min"
    else:
        return f"{secs / 3600:.1f}h"


def _log_runtime_estimate(
    log,
    n_shards: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    avg_points_per_shard: int,
    n_workers: int | None = None,
) -> float:
    """Log a rough CMC runtime estimate and return it in seconds."""
    if n_workers is None:
        n_workers = _estimate_n_workers()

    jit_overhead = 45 + (avg_points_per_shard / 10_000) * 20
    iters = n_chains * (n_warmup + n_samples)
    secs_per_iter = 0.2 + (avg_points_per_shard / 100_000) * 0.3
    total_per_shard = jit_overhead + iters * secs_per_iter

    batches = (n_shards + n_workers - 1) // n_workers
    total = batches * total_per_shard

    log.info(
        "Runtime estimate: %s total (%d shards / %d workers, ~%s/shard)",
        _fmt_time(total),
        n_shards,
        n_workers,
        _fmt_time(total_per_shard),
    )
    return total


def _result_dict_to_cmc_result(
    result_dict: dict[str, Any],
    config: CMCConfig,
) -> CMCResult:
    """Convert a _run_shard_worker result dict to a CMCResult.

    Workers return raw sample dicts; this helper computes ArviZ diagnostics
    (R-hat, ESS, BFMI) and constructs the full CMCResult expected by
    _combine_shard_posteriors().
    """
    if not result_dict.get("success", False):
        param_names: list[str] = result_dict.get("param_names", [])
        return _create_failed_result(
            param_names, result_dict.get("error", "shard failed")
        )

    samples_np: dict[str, np.ndarray] = result_dict["samples"]
    param_names = result_dict["param_names"]
    n_chains: int = result_dict["n_chains"]
    n_samples: int = result_dict["n_samples"]
    extra_fields: dict[str, np.ndarray] = result_dict.get("extra_fields", {})
    duration: float = result_dict.get("duration", 0.0)
    stats: dict[str, Any] = result_dict.get("stats", {})
    num_divergent: int = stats.get("num_divergent", 0)
    n_warmup: int = stats.get("n_warmup", config.num_warmup)

    # Reshape (n_chains * n_samples,) → (n_chains, n_samples) for ArviZ
    idata: az.InferenceData | None = None
    summary: Any = None
    try:
        posterior_dict = {
            k: v.reshape(n_chains, n_samples) for k, v in samples_np.items()
        }
        idata_kwargs: dict[str, Any] = {"posterior": posterior_dict}
        if "energy" in extra_fields:
            try:
                idata_kwargs["sample_stats"] = {
                    "energy": extra_fields["energy"].reshape(n_chains, n_samples)
                }
            except (ValueError, AttributeError):
                pass
        idata = az.from_dict(idata_kwargs)
        if param_names:
            summary = az.summary(idata, var_names=param_names, ci_prob=0.95)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("ArviZ summary failed for shard result: %s", _exc)

    physics_samples = {k: np.asarray(v) for k, v in samples_np.items()}

    posterior_mean, posterior_std, r_hat, ess_bulk, ess_tail = _extract_posterior_stats(
        param_names, physics_samples, summary
    )
    credible_intervals = _extract_credible_intervals(
        param_names, physics_samples, summary
    )

    bfmi: list[float] | None = None
    bfmi_compute_failed = False
    if idata is not None:
        bfmi, bfmi_compute_failed = _compute_bfmi(idata)

    r_hat_finite = r_hat[~np.isnan(r_hat)]
    ess_finite = ess_bulk[~np.isnan(ess_bulk)]
    convergence_passed = bool(
        len(r_hat_finite) > 0
        and np.all(r_hat_finite < config.max_r_hat)
        and len(ess_finite) > 0
        and np.all(ess_finite > config.min_ess)
    )
    # BFMI is advisory for per-shard results (homodyne parity).
    # Homodyne check_convergence uses R-hat + ESS as the sole hard gates.
    # Per-shard BFMI < 0.3 is expected when chains start near a parameter
    # boundary (boundary reflection drives short trajectories → low BFMI).
    # Using BFMI as a hard gate here causes 100% shard rejection on
    # boundary-adjacent warm-starts such as alpha_sample=-2.0.
    _shard_min_bfmi: float | None = None
    if bfmi is not None and not bfmi_compute_failed:
        _shard_min_bfmi = float(np.nanmin(np.asarray(bfmi, dtype=float)))
        if _shard_min_bfmi < config.min_bfmi:
            logger.warning(
                "Shard: low BFMI=%.3f < %.2f (advisory; not a convergence gate). "
                "Chains may be near a parameter boundary.",
                _shard_min_bfmi,
                config.min_bfmi,
            )
    if bfmi_compute_failed:
        logger.debug(
            "Shard: BFMI unavailable (az.bfmi failed); "
            "convergence determined by R-hat and ESS only"
        )

    total_iters = n_chains * n_samples
    divergence_rate = num_divergent / total_iters if total_iters > 0 else 0.0

    logger.debug(
        "Shard diagnostics: convergence=%s, max_r_hat=%.3f, min_ess=%.0f, "
        "bfmi=%s, divergence_rate=%.1f%%, n_chains=%d, n_samples=%d",
        "PASS" if convergence_passed else "FAIL",
        float(np.nanmax(r_hat)) if len(r_hat_finite) > 0 else float("nan"),
        float(np.nanmin(ess_bulk)) if len(ess_finite) > 0 else float("nan"),
        f"{_shard_min_bfmi:.3f}" if _shard_min_bfmi is not None else "N/A",
        divergence_rate * 100,
        n_chains,
        n_samples,
    )

    return CMCResult(
        parameter_names=param_names,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        credible_intervals=credible_intervals,
        convergence_passed=convergence_passed,
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        bfmi=bfmi,
        samples=physics_samples,
        map_estimate=posterior_mean.copy(),
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        wall_time_seconds=duration,
        metadata={
            "num_divergent": num_divergent,
            "divergence_rate": divergence_rate,
        },
    )


def _extract_posterior_stats(
    output_names: list[str],
    physics_samples: dict[str, np.ndarray],
    summary: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract mean, std, R-hat, ESS-bulk, ESS-tail from an ArviZ summary.

    Falls back to direct sample statistics when the summary is unavailable.

    Args:
        output_names: Ordered list of parameter names.
        physics_samples: Dictionary mapping parameter names to 1-D sample
            arrays in physics space.
        summary: ArviZ summary DataFrame, or ``None``.

    Returns:
        5-tuple of NumPy arrays ``(mean, std, r_hat, ess_bulk, ess_tail)``,
        each of length ``len(output_names)``.
    """
    if summary is not None and len(summary) > 0:
        posterior_mean = np.array(
            [
                float(summary.loc[name, "mean"]) if name in summary.index else 0.0
                for name in output_names
            ]
        )
        posterior_std = np.array(
            [
                float(summary.loc[name, "sd"]) if name in summary.index else 0.0
                for name in output_names
            ]
        )
        r_hat = np.array(
            [
                float(summary.loc[name, "r_hat"]) if name in summary.index else np.nan
                for name in output_names
            ]
        )
        ess_bulk = np.array(
            [
                float(summary.loc[name, "ess_bulk"])
                if name in summary.index
                else np.nan
                for name in output_names
            ]
        )
        ess_tail = np.array(
            [
                float(summary.loc[name, "ess_tail"])
                if name in summary.index
                else np.nan
                for name in output_names
            ]
        )
    else:
        posterior_mean = np.array(
            [
                float(np.mean(physics_samples[name]))
                if name in physics_samples
                else 0.0
                for name in output_names
            ]
        )
        posterior_std = np.array(
            [
                float(np.std(physics_samples[name])) if name in physics_samples else 0.0
                for name in output_names
            ]
        )
        r_hat = np.full(len(output_names), np.nan)
        ess_bulk = np.full(len(output_names), np.nan)
        ess_tail = np.full(len(output_names), np.nan)

    return posterior_mean, posterior_std, r_hat, ess_bulk, ess_tail


def _extract_credible_intervals(
    output_names: list[str],
    physics_samples: dict[str, np.ndarray],
    summary: Any,
) -> dict[str, dict[str, float]]:
    """Extract 95 % credible intervals from ArviZ summary or raw samples.

    Args:
        output_names: Ordered list of parameter names.
        physics_samples: Dictionary mapping parameter names to sample arrays.
        summary: ArviZ summary DataFrame produced by ``az.summary(ci_prob=0.95)``,
            or ``None`` to fall back to raw percentiles.  Modern ArviZ (≥ 0.12)
            uses columns ``"eti_2.5%"`` / ``"eti_97.5%"``; older versions use
            ``"hdi_2.5%"`` / ``"hdi_97.5%"``.  Both are tried before the
            raw-percentile fallback.

    Returns:
        Dict mapping parameter names to ``{"2.5%": lb, "97.5%": ub}``.
    """
    credible_intervals: dict[str, dict[str, float]] = {}
    for name in output_names:
        if summary is not None and name in summary.index:
            try:
                lb = float(summary.loc[name, "eti_2.5%"])
                ub = float(summary.loc[name, "eti_97.5%"])
            except KeyError:
                try:
                    lb = float(summary.loc[name, "hdi_2.5%"])
                    ub = float(summary.loc[name, "hdi_97.5%"])
                except KeyError:
                    lb, ub = None, None
            if lb is not None and ub is not None:
                credible_intervals[name] = {"2.5%": lb, "97.5%": ub}
                continue
        if name in physics_samples:
            s = physics_samples[name]
            credible_intervals[name] = {
                "2.5%": float(np.percentile(s, 2.5)),
                "97.5%": float(np.percentile(s, 97.5)),
            }
    return credible_intervals


def _compute_bfmi(idata: az.InferenceData) -> tuple[list[float] | None, bool]:
    """Compute BFMI from ArviZ InferenceData, returning (bfmi, failed).

    Args:
        idata: ArviZ InferenceData object with sample stats.

    Returns:
        Tuple of ``(bfmi_list, compute_failed)``.  ``bfmi_list`` is a
        list of per-chain BFMI values, or ``None`` if unavailable.
        ``compute_failed`` is ``True`` when an exception was raised.
    """
    bfmi: list[float] | None = None
    bfmi_compute_failed = False
    try:
        bfmi_result = az.bfmi(idata)
        # az.bfmi() return type varies across ArviZ versions:
        #   - xr.DataArray  → .values is a numpy array attribute (non-callable)
        #   - dict          → .values is a bound method (callable)
        #   - list / ndarray → iterate directly
        if isinstance(bfmi_result, dict):
            bfmi = list(bfmi_result.values())
        elif hasattr(bfmi_result, "values"):
            attr = bfmi_result.values
            bfmi = list(attr() if callable(attr) else attr)
        elif isinstance(bfmi_result, (list, np.ndarray)):
            bfmi = list(bfmi_result)
    except (TypeError, KeyError) as e:
        logger.warning("Could not compute BFMI: %s", e)
        bfmi_compute_failed = True
    return bfmi, bfmi_compute_failed


def _create_failed_result(parameter_names: list[str], message: str) -> CMCResult:
    """Create a failed CMC result with zero statistics.

    Args:
        parameter_names: List of parameter names for the result.
        message: Error message to store in ``metadata["error"]``.

    Returns:
        CMCResult with ``convergence_passed=False`` and zero arrays.
    """
    n_params = len(parameter_names)
    return CMCResult(
        parameter_names=parameter_names,
        posterior_mean=np.zeros(n_params),
        posterior_std=np.zeros(n_params),
        credible_intervals={},
        convergence_passed=False,
        metadata={"error": message},
    )


def run_cmc_analysis(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    config: CMCConfig | None = None,
    **kwargs: Any,
) -> CMCResult:
    """Convenience wrapper around :func:`fit_cmc_jax` (homodyne parity).

    Accepts the same arguments as :func:`fit_cmc_jax` and delegates directly.
    """
    return fit_cmc_jax(model, c2_data, config=config, **kwargs)
