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

    if sigma is None:
        sigma = estimate_sigma(c2_jax, method="diagonal")
        logger.info("[CMC] Estimated sigma = %.4e", float(jnp.mean(sigma)))

    sigma_jax = jnp.asarray(sigma) if isinstance(sigma, np.ndarray) else sigma

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
    if (
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
        t_array = np.asarray(model.t)
        dt_val = (
            float(t_array[1] - t_array[0]) if len(t_array) > 1 else float(t_array[0])
        )
        t_max_val = float(t_array[-1])
        t_ref = compute_t_ref(dt_val, t_max_val, fallback_value=1.0)

        reparam_config = ReparamConfig(t_ref=t_ref)
        logger.info("[CMC] Reference-time reparameterization: t_ref=%.4e", t_ref)

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
            contrast=contrast,
            offset=offset,
        )
    else:
        numpyro_model = get_heterodyne_model(
            t=model.t,
            q=model.q,
            dt=model.dt,
            phi_angle=phi_angle,
            c2_data=c2_jax,
            sigma=sigma_jax,
            space=space,
            contrast=contrast,
            offset=offset,
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

    kernel = NUTS(
        numpyro_model,
        target_accept_prob=config.target_accept_prob,
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

    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(rng_seed)

    try:
        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))
    except (RuntimeError, ValueError) as e:
        logger.error("[CMC] MCMC sampling failed: %s", e)
        return _create_failed_result(varying_names, str(e))

    # --- Phase 4: diagnostics and output ---
    logger.info("[CMC] Phase 4/4: diagnostics and result construction")
    samples = mcmc.get_samples()
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
    if bfmi is not None:
        convergence_passed = convergence_passed and min(bfmi) > config.min_bfmi
    if bfmi_compute_failed:
        convergence_passed = False

    metadata: dict[str, Any] = {}
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

    Prior tempering is applied automatically: each shard's likelihood
    contributes only 1/K of the full data, so prior scales are multiplied
    by ``sqrt(num_shards)`` to preserve the correct posterior geometry.

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

    # --- Phase 3: per-shard sampling ---
    logger.info("[CMC-sharded] Phase 3/5: sampling %d shards sequentially", num_shards)

    shard_results: list[CMCResult] = []
    base_seed = config.seed if config.seed is not None else secrets.randbelow(2**31)

    for shard_idx, shard in enumerate(shards):
        logger.info(
            "[CMC-sharded] Shard %d/%d: %d data points",
            shard_idx + 1,
            num_shards,
            len(shard["indices"]),
        )

        # Build per-shard sigma (tempered)
        shard_sigma_raw = shard["sigma_shard"]
        tempered_sigma = _temper_sigma(shard_sigma_raw, num_shards)

        # Per-shard config: unique seed, same NUTS hyper-parameters
        shard_config = _make_shard_config(config, seed=base_seed + shard_idx)

        shard_result = fit_cmc_jax(
            model=model,
            c2_data=shard["c2_shard"],
            phi_angle=phi_angle,
            config=shard_config,
            sigma=tempered_sigma,
            nlsq_result=nlsq_result,
        )
        shard_results.append(shard_result)

        logger.info(
            "[CMC-sharded] Shard %d/%d complete: convergence=%s, max_rhat=%.3f",
            shard_idx + 1,
            num_shards,
            "PASSED" if shard_result.convergence_passed else "FAILED",
            float(np.nanmax(shard_result.r_hat))
            if shard_result.r_hat is not None
            else float("nan"),
        )

    # --- Phase 4: consensus combination ---
    logger.info("[CMC-sharded] Phase 4/5: combining shard posteriors (consensus)")
    combined_result = _combine_shard_posteriors(
        shard_results,
        config,
        num_shards,
        base_seed,
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
    """Random element-wise sharding of the upper-triangle + diagonal."""
    rng = np.random.default_rng(seed)

    # Work with all N*N elements (c2 is symmetric, so we use the full matrix)
    all_indices = np.arange(n * n, dtype=np.int64)
    rng.shuffle(all_indices)

    # Split into num_shards roughly equal groups
    splits = np.array_split(all_indices, num_shards)

    shards: list[dict[str, Any]] = []
    for split_indices in splits:
        rows, cols = np.divmod(split_indices, n)

        # Build a square sub-matrix: use unique row/col indices, then slice
        # to the bounding box.  For random shards the "sub-matrix" is really
        # a vector of selected elements; we reshape into a 1-D correlation
        # vector for the shard likelihood.
        c2_shard_vals = c2_np[rows, cols]

        if sigma_is_scalar:
            sigma_shard: np.ndarray | float = float(sigma_np)  # type: ignore[arg-type]
        else:
            sigma_shard = np.asarray(sigma_np)[rows, cols]

        # Reshape to a (k, k) matrix using unique sorted row/col unions so
        # the shard model can be evaluated efficiently.  We fall back to
        # a 1-D representation when the shard is not square.
        unique_rows = np.unique(rows)
        unique_cols = np.unique(cols)

        if len(unique_rows) == len(unique_cols) and np.array_equal(
            unique_rows, unique_cols
        ):
            # Shard forms a square sub-matrix
            idx_map = {int(v): i for i, v in enumerate(unique_rows)}
            shard_n = len(unique_rows)
            c2_sq = np.zeros((shard_n, shard_n), dtype=np.float64)
            for flat_r, flat_c, val in zip(rows, cols, c2_shard_vals, strict=True):
                c2_sq[idx_map[int(flat_r)], idx_map[int(flat_c)]] = val

            if not sigma_is_scalar:
                sigma_sq = np.zeros((shard_n, shard_n), dtype=np.float64)
                for flat_r, flat_c, s_val in zip(
                    rows, cols, np.asarray(sigma_shard), strict=True
                ):
                    sigma_sq[idx_map[int(flat_r)], idx_map[int(flat_c)]] = s_val
                sigma_shard_out: np.ndarray | float = sigma_sq
            else:
                sigma_shard_out = float(sigma_np)  # type: ignore[arg-type]

            shards.append(
                {
                    "c2_shard": jnp.asarray(c2_sq),
                    "sigma_shard": sigma_shard_out,
                    "indices": split_indices,
                }
            )
        else:
            # Non-square: store as flattened 1-D array
            shards.append(
                {
                    "c2_shard": jnp.asarray(c2_shard_vals),
                    "sigma_shard": sigma_shard
                    if not sigma_is_scalar
                    else float(sigma_np),  # type: ignore[arg-type]
                    "indices": split_indices,
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

    # --- Inverse-variance weighting ---
    # Weight_k = 1 / Var_k (per-parameter, diagonal approximation)
    weight_sum = np.zeros(n_params)
    weighted_mean_sum = np.zeros(n_params)

    for sr in shard_results:
        var_k = sr.posterior_std**2
        # Clip to avoid division by zero from degenerate shards
        var_k_clipped = np.where(var_k > 1e-30, var_k, 1e-30)
        w_k = 1.0 / var_k_clipped
        weight_sum += w_k
        weighted_mean_sum += w_k * sr.posterior_mean

    combined_mean = weighted_mean_sum / np.where(weight_sum > 0, weight_sum, 1.0)
    combined_var = 1.0 / np.where(weight_sum > 0, weight_sum, 1.0)
    combined_std = np.sqrt(combined_var)

    # --- Worst-case R-hat (conservative) ---
    r_hat_stacked = np.stack(
        [sr.r_hat for sr in shard_results if sr.r_hat is not None],
        axis=0,
    )
    combined_r_hat = (
        np.nanmax(r_hat_stacked, axis=0)
        if r_hat_stacked.size > 0
        else np.full(n_params, np.nan)
    )

    # --- Summed ESS (approximate) ---
    ess_bulk_stacked = np.stack(
        [sr.ess_bulk for sr in shard_results if sr.ess_bulk is not None],
        axis=0,
    )
    combined_ess_bulk = (
        np.nansum(ess_bulk_stacked, axis=0)
        if ess_bulk_stacked.size > 0
        else np.full(n_params, np.nan)
    )

    ess_tail_stacked = np.stack(
        [sr.ess_tail for sr in shard_results if sr.ess_tail is not None],
        axis=0,
    )
    combined_ess_tail = (
        np.nansum(ess_tail_stacked, axis=0)
        if ess_tail_stacked.size > 0
        else np.full(n_params, np.nan)
    )

    # --- BFMI: minimum across all shards and all chains ---
    all_bfmi_values: list[float] = []
    for sr in shard_results:
        if sr.bfmi is not None:
            all_bfmi_values.extend(sr.bfmi)
    combined_bfmi = all_bfmi_values if all_bfmi_values else None

    # --- Credible intervals from combined samples ---
    # Pool samples across shards for each parameter
    combined_samples: dict[str, np.ndarray] = {}
    if all(sr.samples is not None for sr in shard_results):
        for name in param_names:
            arrays = [
                np.asarray(sr.samples[name])  # type: ignore[index]
                for sr in shard_results
                if sr.samples is not None and name in sr.samples
            ]
            if arrays:
                combined_samples[name] = np.concatenate(arrays, axis=0)

    credible_intervals: dict[str, dict[str, float]] = {}
    for name in param_names:
        if name in combined_samples:
            s = combined_samples[name]
            credible_intervals[name] = {
                "2.5%": float(np.percentile(s, 2.5)),
                "97.5%": float(np.percentile(s, 97.5)),
            }

    # --- MAP estimate ---
    map_estimate = combined_mean.copy()

    # --- Convergence gate ---
    r_hat_finite = combined_r_hat[~np.isnan(combined_r_hat)]
    ess_finite = combined_ess_bulk[~np.isnan(combined_ess_bulk)]
    n_failed = sum(1 for sr in shard_results if not sr.convergence_passed)

    convergence_passed = bool(
        n_failed == 0
        and len(r_hat_finite) > 0
        and np.all(r_hat_finite < config.max_r_hat)
        and len(ess_finite) > 0
        and np.all(ess_finite > config.min_ess)
    )
    if combined_bfmi is not None:
        convergence_passed = convergence_passed and min(combined_bfmi) > config.min_bfmi

    logger.info(
        "[CMC-sharded] Consensus combination: %d/%d shards converged, "
        "worst_rhat=%.3f, combined_ess_min=%.0f",
        num_shards - n_failed,
        num_shards,
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
# Prior tempering
# ---------------------------------------------------------------------------


def _temper_sigma(
    sigma: np.ndarray | float,
    num_shards: int,
) -> np.ndarray | float:
    """Scale measurement uncertainty for CMC prior tempering.

    In Consensus Monte Carlo each shard receives 1/K of the total data.
    To preserve the correct posterior geometry, the effective likelihood
    contribution of each shard must be inflated by K (i.e., the log-
    likelihood is multiplied by K).  Equivalently, the noise standard
    deviation is divided by ``sqrt(K)``:

    .. math::

        \\sigma_{\\text{shard}} = \\sigma_{\\text{full}} / \\sqrt{K}

    This ensures that the product of K shard sub-posteriors (each with
    tempered sigma) approximates the full-data posterior.

    Args:
        sigma: Original measurement uncertainty (scalar or array).
        num_shards: Number of shards K.  Must be >= 1.

    Returns:
        Tempered sigma of the same type as the input.

    Raises:
        ValueError: If ``num_shards < 1``.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    scale_factor = 1.0 / math.sqrt(num_shards)

    if isinstance(sigma, float):
        return sigma * scale_factor

    return np.asarray(sigma) * scale_factor


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
        summary: ArviZ summary DataFrame with ``eti95_lb`` / ``eti95_ub``
            columns, or ``None`` to fall back to raw percentiles.

    Returns:
        Dict mapping parameter names to ``{"2.5%": lb, "97.5%": ub}``.
    """
    credible_intervals: dict[str, dict[str, float]] = {}
    for name in output_names:
        if summary is not None and name in summary.index:
            credible_intervals[name] = {
                "2.5%": float(summary.loc[name, "eti95_lb"]),
                "97.5%": float(summary.loc[name, "eti95_ub"]),
            }
        elif name in physics_samples:
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
        if hasattr(bfmi_result, "values"):
            bfmi = list(bfmi_result.values)
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
