<!-- Package: heterodyne | Last verified: 2026-05-08 -->

# CMC Fitting Architecture

## Table of Contents

- [Overview](#overview)
- [Component Map](#component-map)
- [Execution Flow](#execution-flow)
- [NumPyro Model](#numpyro-model)
- [Prior Construction](#prior-construction)
- [Z-Space Reparameterization](#z-space-reparameterization)
- [Smooth Bounded Transforms](#smooth-bounded-transforms)
- [Sampler Infrastructure](#sampler-infrastructure)
- [Backend Selection](#backend-selection)
- [Consensus Monte Carlo Combination](#consensus-monte-carlo-combination)
- [CMCConfig](#cmcconfig)
- [Convergence Diagnostics](#convergence-diagnostics)
- [CMCResult](#cmcresult)
- [Data Preparation](#data-preparation)
- [NLSQ-to-CMC Pipeline](#nlsq-to-cmc-pipeline)
- [I/O Serialization Pipeline](#i-o-serialization-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [Architectural Invariants & Historical Fixes](#architectural-invariants--historical-fixes)

## Overview

The CMC (Consensus Monte Carlo) subsystem provides CPU-optimized Bayesian
posterior inference for the heterodyne 14-parameter two-component correlation
model using NumPyro's NUTS sampler. It is the second stage of the analysis
pipeline, warm-started from NLSQ point estimates. The subsystem implements
data sharding, per-shard NUTS sampling, and precision-weighted consensus
combination following Scott et al. (2016).

A Z-space reparameterization reduces posterior correlation for three
power-law parameter pairs (D0/alpha, v0/beta), and a runtime backend
selector dispatches to sequential CPU, multi-process CPU, JAX pjit
distributed, GPU, PBS cluster, or manual worker-pool execution.

---

## Key Files Reference

| File | Responsibility |
|---|---|
| `heterodyne/core/physics_cmc.py` | 14-parameter JAX physics kernel (element-wise path for CMC) |
| `heterodyne/optimization/cmc/model.py` | Log-probability and prior definitions |
| `heterodyne/optimization/cmc/core.py` | `fit_cmc_jax()` entry point and shard orchestration |
| `heterodyne/optimization/cmc/sampler.py` | `NUTSSampler` + `SamplingPlan` |
| `heterodyne/optimization/cmc/config.py` | `CMCConfig` |

---

## Component Map

```
optimization/cmc/
├── core.py               # fit_cmc_jax() unified entry, fit_cmc_sharded()
├── model.py              # NumPyro models (meshgrid + element-wise paths)
├── priors.py             # build_default_priors(), build_nlsq_informed_priors(),
│                         #   build_log_space_priors(), temper_priors(),
│                         #   estimate_contrast_offset_from_data()
├── sampler.py            # SamplingPlan (+ chain_method), NUTSSampler, AdaptiveSamplingPlan,
│                         #   run_nuts_with_retry(), SamplingStats
├── reparameterization.py # ReparamConfig, compute_t_ref(), power-law decorrelation
├── scaling.py            # ParameterScaling, smooth_bound() (tanh-based)
├── diagnostics.py        # R-hat, ESS, BFMI, divergence analysis,
│                         #   bimodal detection, cross-shard clustering,
│                         #   check_convergence(), create_diagnostics_dict(),
│                         #   log_analysis_summary(), get_convergence_recommendations()
├── config.py             # CMCConfig dataclass (14 config sections)
├── results.py            # CMCResult (+ get_samples_array, get_posterior_stats),
│                         #   ParameterStats, merge_shard_cmc_results(), compare_cmc_nlsq()
├── data_prep.py          # ShardingStrategy, PreparedData, sigma estimation
├── plotting.py           # CMC-specific plotting: convergence traces, diagonal overlays,
│                         #   residual maps, parameter sensitivity, pair correlations
├── io.py                 # Full result serialization: save_samples_npz, load_samples_npz,
│                         #   samples_to_arviz, save_fitted_data_npz, save_parameters_json,
│                         #   save_diagnostics_json, save_all_results;
│                         #   also per-shard NPZ archives, ArviZ NetCDF, list_shards()
└── backends/
    ├── base.py           # MCMCBackend protocol, CMCBackend ABC,
    │                     #   select_backend(), consensus_mc(), robust_consensus_mc()
    ├── cpu_backend.py    # CPUBackend: sequential NUTS chains
    ├── multiprocessing_backend.py  # Process-pool parallel (recommended for CPU)
    ├── pjit_backend.py   # JAX pjit distributed (multi-device)
    ├── pbs.py            # PBS/Torque HPC cluster backend
    └── worker_pool.py    # WorkerPoolBackend: manual process management
```

---

## Execution Flow

### Single-run path: `fit_cmc_jax()`

```
fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result,
            t_override=None, priors_override=None, prior_width_multiplier=1.0)
        │
        ├─ estimate_sigma(c2_data, method)
        │     diagonal / constant / local / residual / bootstrap
        │
        ├─ ReparamConfig + compute_t_ref(dt, t_max)
        │     t_ref = sqrt(dt × t_max)   [uses t_override if provided]
        │     ReparamConfig flags wired from CMCConfig:
        │       enable_d_ref    = config.reparameterization_d_total
        │       enable_d_sample = config.reparameterization_d_total
        │       enable_v_ref    = config.reparameterization_log_gamma
        │     [Previously all flags defaulted to False regardless of config]
        │
        ├─ transform_nlsq_to_reparam_space(nlsq_values, nlsq_unc, t_ref)
        │     delta-method uncertainty propagation to Z-space
        │     scale *= prior_width_multiplier  [applied when called from sharded path]
        │
        ├─ effective_target_accept = max(config.target_accept_prob, 0.9)
        │     [reparameterized path only; Z-space posteriors are well-conditioned
        │      but have longer correlation lengths → higher acceptance floor
        │      prevents NUTS from choosing too-large step sizes]
        │
        ├─ get_heterodyne_model_reparam(...)   # reparameterized path
        │   or get_heterodyne_model(..., priors_override=priors_override)
        │                                      # physics-space path; injected priors
        │                                      #   override space.priors when provided
        │
        ├─ mcmc.run(..., extra_fields=("energy", "diverging"))
        │     [both fields required: "energy" for BFMI, "diverging" for
        │      get_divergence_stats() and high-divergence shard filtering]
        │
        ├─ patch numpyro.infer.initialization attribute if missing
        │     [import-order quirk: submodule may be in sys.modules but not
        │      set as package attribute; az.from_numpyro() requires it]
        │
        ├─ transform_to_physics_space(samples, reparam_config)
        │
        └─ validate_convergence(result)
              R-hat, ESS, BFMI, posterior contraction checks
```

**Key parameters:**

| Parameter | Purpose |
|---|---|
| `t_override` | Shard time slice — replaces `model.t` for model construction. Used by `fit_cmc_sharded` to match shard C2 shape. Without this, shard C2 of shape `(M,M)` is observed against a model built on full `model.t` → shape mismatch crash. |
| `priors_override` | Pre-built (tempered) NumPyro distributions to use instead of `space.priors`. Used by `fit_cmc_sharded` to inject `temper_priors(priors, K)` for CMC correctness. |
| `prior_width_multiplier` | Multiplier applied to reparam-path `ParameterScaling.scale` — equivalent to prior tempering for the reparameterized path. Set to `sqrt(K)` by `fit_cmc_sharded`. |

### Sharded path: `fit_cmc_sharded()`

```
fit_cmc_sharded(model, c2_data, config, nlsq_result)
        │
        ├─ Data preparation: validate + estimate sigma (full data, unscaled)
        │
        ├─ Sharding: random / contiguous
        │     Each shard dict stores:
        │       c2_shard   — sub-matrix of C2 (shape M×M)
        │       sigma_shard — matched sigma slice
        │       t_indices  — time-axis indices for this shard
        │       indices    — flat C2 matrix indices (for audit)
        │     [Random non-square shards raise ValueError — use contiguous]
        │
        ├─ Prior tempering (correct CMC math, Scott et al. 2016):
        │     base_priors = build_nlsq_informed_priors() or build_default_priors()
        │     shard_priors = temper_priors(base_priors, K)
        │       → widens prior scale by sqrt(K) per shard
        │       → sigma passed unscaled (dividing sigma by sqrt(K) was WRONG:
        │          it multiplied shard likelihood by K, not prior by 1/K)
        │
        ├─ Per-shard: fit_cmc_jax(c2_shard, sigma_shard, t_override=t_shard,
        │               priors_override=shard_priors, prior_width_mult=sqrt(K))
        │
        ├─ Filter successful shards before combination:
        │     successful = [sr for sr in results
        │                   if sr.convergence_passed and any(sr.posterior_std > 0)]
        │     [Failed zero-std shards excluded — the 1e-30 variance clip
        │      gave them weight 1/1e-30, dominating the consensus]
        │     [All-failed → return CMCResult(convergence_passed=False,
        │                                   metadata={"all_shards_failed": True})]
        │
        ├─ Consensus MC combination over successful shards
        │     precision-weighted posterior: Λ_combined = Σ_k Λ_k
        │     μ_combined = Λ_combined⁻¹ Σ_k Λ_k μ_k
        │
        ├─ check_shard_bimodality(shard_sample_dict,
        │       min_weight=config.bimodal_min_weight,
        │       min_separation=config.bimodal_min_separation)
        │     → stored in CMCResult.metadata["bimodal_detected"]
        │                          metadata["bimodal_params"]
        │
        ├─ Diagnostics: R-hat, ESS, BFMI, divergence rate,
        │     cross-shard clustering
        │
        └─ Output: CMCResult with posterior samples + convergence metrics
```

---

## NumPyro Model

The NumPyro model is constructed in `model.py`. Six model constructors serve
different per-angle scaling modes and parameterization strategies:

### Model constructors

| Constructor | Purpose |
|---|---|
| `get_heterodyne_model(priors_override=None)` | Basic model: direct prior sampling, meshgrid or element-wise physics. When `priors_override` is provided (e.g. from `fit_cmc_sharded`), those distributions are used instead of `space.priors`. |
| `get_heterodyne_model_reparam(nlsq_params, reparam_config, scalings)` | Reparameterized: Z-space + smooth bounds. Scalings encode NLSQ-informed prior centres and widths (already tempered when called from the sharded path). Note: `nlsq_result` and `prior_width_factor` were removed from this signature — they were never read; `scalings` carries all prior-width information. |
| `get_heterodyne_model_constant()` | Fixed per-angle contrast/offset from NLSQ |
| `get_heterodyne_model_constant_averaged()` | Fixed angle-averaged contrast/offset |
| `get_heterodyne_model_individual()` | Per-angle sampled contrast/offset via `numpyro.plate` |
| `get_model_for_mode()` | Factory that dispatches to the correct constructor based on `per_angle_mode` |

### Per-angle scaling modes

| Mode | Behavior |
|---|---|
| `individual` | Sample `contrast_z` and `offset_z` per angle in z-space, transform via `smooth_bound()` |
| `auto` | Use `get_heterodyne_model` or `get_heterodyne_model_reparam` depending on reparam config |
| `constant` | Fixed contrast/offset arrays from NLSQ (not sampled) |
| `constant_averaged` | Single scalar contrast/offset averaged over all angles (not sampled) |

### Physics dispatch

Both meshgrid and element-wise integral paths are supported within every
model constructor:

- When `shard_grid` is `None`: calls `compute_c2_heterodyne()` (N x N meshgrid
  path from `jax_backend.py`)
- When `shard_grid` is provided: calls `compute_c2_elementwise()` (O(n_pairs)
  element-wise path from `physics_cmc.py`, no N x N allocation)

### Likelihood

All model variants use the same Normal observation likelihood:

```python
c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
numpyro.sample("obs", dist.Normal(c2_model, sigma), obs=c2_data)
```

### Sigma estimation

`estimate_sigma()` in `model.py` supports five methods:

| Method | Description |
|---|---|
| `diagonal` | Standard deviation of matrix diagonal relative to mean, floored at 1% of data scale |
| `constant` | Overall standard deviation of data |
| `local` | Spatially smoothed local variance via `scipy.ndimage.uniform_filter` |
| `residual` | RMS of NLSQ residuals (requires `nlsq_result`); falls back to `diagonal` |
| `bootstrap` | Bootstrap replicate means of the diagonal (200 replicates default) |

---

## Prior Construction

### `build_default_priors()` (priors.py)

Registry-based priors for each varying parameter:

- **BetaScaled** for bounded fraction parameters (`f0`, `f3`, `contrast`)
  when `prior_mean`/`prior_std` are available and bounds are finite
- **TruncatedNormal** when `prior_mean` and `prior_std` are set in the
  registry
- **Uniform** fallback when prior statistics are unavailable

### `build_nlsq_informed_priors()`

Centers TruncatedNormal priors on NLSQ best-fit values with width =
`nlsq_uncertainty * width_factor`. Falls back to registry `prior_std` or
1/6 of bounds range when NLSQ uncertainty is unavailable.

### `build_log_space_priors()`

Constructs LogNormal distributions for parameters with `log_space=True` in
the registry (D0_ref, D0_sample, v0). Median matches `prior_mean`;
log-space sigma derived from coefficient of variation.

### Prior tempering: `temper_priors()`

Scales prior widths by `sqrt(K)` for K-shard Consensus MC:

| Distribution type | Tempering |
|---|---|
| TruncatedNormal | scale multiplied by sqrt(K) |
| LogNormal | scale multiplied by sqrt(K) |
| Uniform | Unchanged (uninformative) |
| TransformedDistribution (BetaScaled) | Unchanged (with warning) |

### Prior validation: `validate_priors()`

Checks that all varying parameters have priors, prior support overlaps
parameter bounds, and no degenerate (scale < 1e-12) priors exist.

### `estimate_contrast_offset_from_data()`

Physics-informed quantile estimator for initializing contrast and offset priors
from C2 data:

```python
contrast, offset = estimate_contrast_offset_from_data(c2_data, t1, t2)
```

Uses the correlation decay structure `C2 = contrast × g1² + offset`:
- **Offset** → 10th-percentile of the large-lag region (top 20% of lags; where g1² ≈ 0)
- **Contrast** → 90th-percentile of the small-lag ceiling (bottom 20% of lags) minus offset

Both estimates are clipped to configurable `contrast_bounds`/`offset_bounds`. Falls back to
`(bounds_mid_contrast, bounds_mid_offset)` when fewer than 100 data points are present.
Inputs are ravelled so 2-D C2 matrices are handled correctly.

---

## Z-Space Reparameterization

We map constrained physical parameters θ (e.g., D₀ > 0, α ∈ (−1, 2)) to unconstrained Z-space z ∈ (−∞, ∞) using a Bijector chain (Log/Sigmoid + Affine transforms). This prevents NUTS from hitting parameter boundaries, which causes divergent transitions and destroys sampler efficiency. Without this transform, constrained posteriors near zero produce pathological funnel geometry where NUTS step sizes collapse — a well-known HMC failure mode. The transform also improves mixing for the correlated D₀/α and v₀/β parameter pairs, which exhibit strong posterior correlation in the unconstrained space.

Power-law pairs (D0, alpha) form banana-shaped posteriors because
`D0 * t^alpha` is approximately constant at the data's characteristic time
scale. NUTS struggles to explore this geometry efficiently.

The reparameterization replaces (D0, alpha) with (log_D_at_tref, alpha) where:

```
log_D_at_tref = log(D0) + alpha * log(t_ref)
```

`t_ref` is the geometric mean of `dt` and `t_max` (`t_ref = sqrt(dt * t_max)`).
At this reference time the log product is well-constrained by data, making
the posterior approximately elliptical and NUTS-friendly.

Three pairs are reparameterized independently, controlled by `ReparamConfig`:

| Flag | Pair | Reparam name |
|---|---|---|
| `enable_d_ref` | D0_ref / alpha_ref | `log_D0_ref_at_tref` |
| `enable_d_sample` | D0_sample / alpha_sample | `log_D0_sample_at_tref` |
| `enable_v_ref` | v0 / beta | `log_v0_at_tref` |

### Transform chain

1. **Forward** (`transform_nlsq_to_reparam_space`): NLSQ physics-space values
   to Z-space for warm-starting, with delta-method uncertainty propagation:
   `Var(log_A_at_tref) ~ (sigma_A0/A0)^2 + (log(t_ref) * sigma_alpha)^2`

2. **Model-internal** (`reparam_to_physics_jax`): Back-transform during NUTS
   evaluation: `A0 = exp(log_at_tref - alpha * log(t_ref))`

3. **Post-sampling** (`transform_to_physics_space`): Convert posterior samples
   from Z-space back to physics-space units (vectorized over sample dimension)

---

## Smooth Bounded Transforms

`scaling.py` replaces `jnp.clip()` (zero gradient at bounds) with tanh-based
smooth bounding:

```
smooth_bound(raw, low, high) = mid + half * tanh((raw - mid) / half)
```

where `mid = (low + high) / 2` and `half = (high - low) / 2`. This maps
`(-inf, +inf) -> (low, high)` with nonzero gradient everywhere, preventing
NUTS mass matrix adaptation from stalling at boundaries.

`ParameterScaling` manages the z-space to physics-space round trip:
- `to_normalized(value)`: physics -> z-space (`z = (value - center) / scale`)
- `to_original(z)`: z-space -> bounded physics (`center + scale * z`, then
  `smooth_bound`)

---

## Sampler Infrastructure

### SamplingPlan

Immutable configuration specifying a sampling run:

| Field | Default | Description |
|---|---|---|
| `num_warmup` | 500 | NUTS warm-up (adaptation) steps per chain |
| `num_samples` | 1000 | Posterior draws per chain after warmup |
| `num_chains` | 4 | Independent MCMC chains |
| `target_accept` | 0.8 | NUTS dual-averaging target acceptance probability |
| `max_tree_depth` | 10 | NUTS binary tree depth limit |
| `dense_mass` | True | Full-covariance vs diagonal mass matrix. The 14-param model has three correlated power-law pairs (D0/alpha ×2, v0/beta); a diagonal mass matrix inflates divergences on these banana-shaped posteriors. |
| `chain_method` | `"sequential"` | NumPyro chain execution: `"sequential"`, `"parallel"`, or `"vectorized"`. Propagated through `from_config()`, `for_shard()`, `AdaptiveSamplingPlan.get_plan()`, and retry loop. |
| `seed` | None | Explicit random seed (crypto-random if None) |

`SamplingPlan.from_config()` builds a plan from `CMCConfig` with optional
adaptive scaling. `for_shard()` returns a scaled-down plan for a single
CMC shard.

### AdaptiveSamplingPlan

Adjusts warmup/sample counts based on shard size relative to a 10K reference:

```
scale = sqrt(shard_size / 10_000)
num_warmup  = max(min_warmup_floor,  int(base.num_warmup  * scale))
num_samples = max(min_samples_floor, int(base.num_samples * scale))
```

Parameter-aware floors:
- `min_warmup = max(50, 5 * n_params)` (70 for 14-param model)
- `min_samples = max(100, 10 * n_params)` (140 for 14-param model)

### NUTSSampler

High-level wrapper around NumPyro's MCMC:

- `from_plan()`: factory that constructs NUTS kernel and MCMC object
- `run()`: executes sampling with per-chain perturbation of init params.
  Requests `extra_fields=("energy", "diverging")` — both fields are required:
  `"energy"` for BFMI computation, `"diverging"` for `get_divergence_stats()`.
  Requesting only `"energy"` caused divergence counts to always be zero.
  After `mcmc.run()`, calls `jax.block_until_ready(mcmc.last_state)` to force
  XLA lazy evaluation to complete before the timer stops — without this,
  `wall_time_seconds` underestimates true compute time.
- `run_with_init_values()`: warm-start from NLSQ MAP with preflight log-density
  validation via `numpyro.infer.util.log_density(kernel.model, (), {}, init_params)`.
  The previous approach used `kernel._potential_fn`, which is `None` before any
  sampling run and caused the preflight check to always be silently skipped.
- `get_divergence_stats()`: reads `extra["diverging"]` for true divergence rate,
  `extra["tree_depth"]` for mean depth and max-depth fraction
- `get_diagnostics()`: returns ArviZ `InferenceData` via `az.from_numpyro()`

**ArviZ compatibility:** `az.from_numpyro()` accesses `numpyro.infer.initialization`
as a package attribute. Heterodyne's import chain loads the submodule into
`sys.modules` but due to Python's circular-import timing the attribute is not
always set on the package object. `fit_cmc_jax` patches this explicitly before
calling `az.from_numpyro()`:

```python
if not hasattr(numpyro.infer, "initialization"):
    _mod = sys.modules.get("numpyro.infer.initialization")
    if _mod is not None:
        numpyro.infer.initialization = _mod
```

### run_nuts_with_retry()

Automatic retry with step-size adjustment on high divergence:

- Runs NUTS, checks divergence rate after each attempt
- If rate > 5% (`DIVERGENCE_RATE_HIGH`): reduces `target_accept` by
  `step_size_factor` (default 0.5) and rebuilds sampler
- Up to `max_retries` additional attempts (default 3)
- Returns best result (lowest divergence rate) regardless of success

### SamplingStats

Frozen summary from a completed run: `num_samples`, `num_warmup`,
`num_divergences`, `divergence_rate`, `mean_accept_prob`,
`max_tree_depth_fraction`, `wall_time_seconds`. The `is_healthy` property
checks divergence rate < 5% and acceptance probability > 0.6.

---

## Backend Selection

`select_backend(config)` in `backends/base.py` inspects `jax.devices()` at
runtime:

```
jax.devices()
    │
    ├── len(devices) > 1  → PjitBackend (multi-device parallel)
    └── single CPU device → CPUBackend (sequential chains)
```

| Backend | Strategy | Use case |
|---|---|---|
| `CPUBackend` | Sequential NUTS per chain | Testing/debugging, single CPU |
| `MultiprocessingBackend` | Process-pool parallel | CPU production (recommended) |
| `PjitBackend` | JAX pjit distributed | Multiple CPU devices |
| `PBSBackend` | PBS/Torque scheduler | HPC cluster |
| `WorkerPoolBackend` | Manual process management | Custom workflows |

All backends implement the `MCMCBackend` protocol, which exposes a single
`run(model, config, rng_key, init_params)` method returning a dict of sample
arrays shaped `(num_samples * num_chains,)`.

The `CMCBackend` abstract base class extends this with `get_capabilities()`,
`validate_resources()`, `estimate_memory()`, and `cleanup()` for resource
management.

---

## Consensus Monte Carlo Combination

### Standard combination: `consensus_mc()`

Each shard's posterior is summarized by mean and covariance. The combined
posterior uses full precision-matrix weighting:

```
Λ_combined = Σ_k Λ_k                    (sum of precision matrices)
μ_combined = Λ_combined⁻¹ Σ_k Λ_k μ_k  (precision-weighted means)
```

This is exact when sub-posteriors are Gaussian and the prior factorizes
across shards.

### Robust combination: `robust_consensus_mc()`

Identifies outlier shards via median absolute deviation (MAD) of per-shard
means. Shards deviating by more than `outlier_sigma` (default 3.0) MAD-scaled
standard deviations on any parameter have their precision downweighted by
`1/n_shards`.

### merge_shard_cmc_results()

Simpler inverse-variance combination operating on `CMCResult` objects.

**Failed-shard filtering (critical):** Before weighting, shards where
`convergence_passed=False`, `all(posterior_std == 0)`, or
`divergence_rate > config.max_divergence_rate` are excluded. The divergence
rate is stored in `CMCResult.metadata["divergence_rate"]` by `fit_cmc_jax`
from `mcmc.get_extra_fields()["diverging"]`. Without this gate, shards with
corrupt posteriors from excessive NUTS divergences bias the consensus.

When all shards fail the function returns immediately with a degenerate
`CMCResult(convergence_passed=False, posterior_std=NaN,
metadata={"all_shards_failed": True})` rather than crashing on `np.stack([])`.

```
# Effective implementation:
_max_div_rate = config.max_divergence_rate  # default 0.10
successful = [sr for sr in shard_results
              if sr.convergence_passed
              and any(sr.posterior_std > 0)
              and sr.metadata.get("divergence_rate", 0.0) <= _max_div_rate]
precision_i = 1 / std_i^2          # over successful shards only
combined_mean = Σ(precision_i × mean_i) / Σ(precision_i)
combined_std  = 1 / sqrt(Σ(precision_i))
```

**Heterogeneity check:** After filtering, before combination, an IQR-based
cross-shard CV is computed. Raw `std/|mean|` is avoided because α_ref, β,
v_offset, and φ₀ all default to ~0 and would produce infinite CV. Instead:

```
IQR-CV = (Q75 - Q25) / max(|median|, 1e-3)   # per parameter
```

If `max(IQR-CV) > config.max_parameter_cv` (default 1.0): raises `RuntimeError`
when `config.heterogeneity_abort=True`, otherwise logs a warning and continues.

Diagnostics use worst-case values: maximum R-hat, minimum ESS, minimum BFMI.
Credible intervals are reconstructed from combined Gaussian approximation.

### Combination methods

`_combine_shard_posteriors` reads `config.combination_method` and dispatches:

| Method | Implementation |
|---|---|
| `consensus_mc` (default) | Inverse-variance weighting (Scott et al. 2016) |
| `simple_average` | Equal-weight mean and variance across successful shards |
| `robust_consensus_mc` | Per-parameter z-score outlier detection, then inverse-variance on inliers |
| `weighted_gaussian` | Falls back to `consensus_mc` (not separately implemented) |

**`robust_consensus_mc` outlier detection:** Uses z-score rather than raw MAD
to handle near-zero parameters. Scale = `max(std, 1e-4 * (|mean| + 1))` keeps
α, β, v_offset, φ₀ finite. Shards where `max|z| > 3` across any parameter are
excluded; the inlier pool then undergoes standard inverse-variance combination.
Falls back to the full `successful` set if outlier removal would exclude all shards.

Unknown method names produce a `logger.warning` and fall back to `consensus_mc`.

The full `backends/base.py` also exposes `consensus_mc()` and `robust_consensus_mc()`
operating on numpy arrays (not `CMCResult` objects); these are used by the backend
layer, not by `_combine_shard_posteriors`.

---

## CMCConfig

CMCConfig is organized into 14 logical sections:

### 1. Enable gating

| Field | Default | Description |
|---|---|---|
| `enable` | `"auto"` | Master switch: `"auto"`, `"always"`, `"never"` |
| `min_points_for_cmc` | — | Minimum data points for auto-enable |

### 2. Per-angle mode

| Field | Default | Description |
|---|---|---|
| `per_angle_mode` | `"auto"` | `"auto"`, `"constant"`, `"constant_averaged"`, `"individual"` |
| `constant_scaling_threshold` | — | Min phi angles before switching from constant to individual |

### 3. Sharding

| Field | Default | Description |
|---|---|---|
| `num_shards` | `"auto"` | Number of shards K (or auto-derive from dataset) |
| `sharding_strategy` | `"stratified"` | `"stratified"`, `"random"`, `"contiguous"` |
| `max_points_per_shard` | `"auto"` | Upper bound on shard size |
| `min_points_per_shard` | — | Lower bound on shard size |
| `min_points_per_param` | 1500 | Minimum data-to-parameter ratio per shard |

### 4. Backend

| Field | Default | Description |
|---|---|---|
| `backend_name` | `"auto"` | `"auto"`, `"multiprocessing"`, `"pjit"`, `"cpu"` (no GPU backend) |
| `chain_method` | `"sequential"` | `"sequential"` or `"parallel"` within each worker |
| `enable_checkpoints` | — | Persist intermediate shard results |
| `checkpoint_dir` | — | Directory for shard checkpoint files |

### 5. Per-shard MCMC

| Field | Default | Description |
|---|---|---|
| `num_warmup` | 500 | NUTS warm-up steps per chain |
| `num_samples` | 1000 | Posterior draws per chain |
| `num_chains` | 4 | Independent chains per shard |
| `target_accept_prob` | 0.8 | Dual-averaging target (0.5-0.99) |
| `max_tree_depth` | 10 | NUTS tree depth limit |
| `dense_mass` | True | Full-covariance mass matrix |
| `init_strategy` | `"init_to_median"` | NUTS initialization |
| `adaptive_sampling` | — | Scale warmup/samples by shard size |
| `min_warmup` | — | Adaptive warmup floor |
| `min_samples` | — | Adaptive sample floor |
| `seed` | None | Base random seed |

### 6. Validation

| Field | Default | Description |
|---|---|---|
| `max_r_hat` | 1.05 | Maximum acceptable R-hat |
| `min_ess` | — | Minimum effective sample size |
| `min_bfmi` | — | Minimum BFMI |
| `max_divergence_rate` | — | Maximum divergent transition fraction |
| `max_parameter_cv` | — | Maximum coefficient of variation |
| `require_nlsq_warmstart` | — | Abort if NLSQ warm-start unavailable |
| `heterogeneity_abort` | — | Abort on incompatible shard posteriors |

### 7. NLSQ warm-start

| Field | Default | Description |
|---|---|---|
| `use_nlsq_warmstart` | — | Initialize NUTS from NLSQ MAP |
| `use_nlsq_informed_priors` | — | Center priors on NLSQ estimates |
| `nlsq_prior_width_factor` | 2.0 | Multiplier on NLSQ uncertainty for prior width |

### 8. Prior tempering

| Field | Default | Description |
|---|---|---|
| `prior_tempering` | — | Scale priors by 1/K for shard consistency |

### 9. Combination

| Field | Default | Description |
|---|---|---|
| `combination_method` | `"consensus_mc"` | Posterior combination algorithm |
| `min_success_rate` | — | Minimum fraction of shards that must converge |
| `min_success_rate_warning` | — | Warning threshold |

### 10. Timeout

| Field | Default | Description |
|---|---|---|
| `per_shard_timeout` | — | Wall-clock seconds per shard |
| `heartbeat_timeout` | — | Seconds before declaring worker dead |

### 11. Reparameterization

| Field | Default | Description |
|---|---|---|
| `use_reparam` | — | Enable parameter reparameterizations |
| `reparameterization_d_total` | — | Reparameterize d_total as unconstrained sum |
| `reparameterization_log_gamma` | — | Log-scale gamma reparameterization |

### 12. Bimodal detection

| Field | Default | Description |
|---|---|---|
| `bimodal_min_weight` | — | Minimum minor-mode weight |
| `bimodal_min_separation` | — | Minimum normalized distance between modes |

### 13. Seed and run identity

| Field | Default | Description |
|---|---|---|
| `seed` | None | Base random seed |
| `run_id` | — | Optional identifier for checkpoint namespacing |

### 14. Checkpointing

| Field | Default | Description |
|---|---|---|
| `enable_checkpoints` | — | Persist shard results to disk |
| `checkpoint_dir` | — | Checkpoint directory |

**Attribute naming convention:** Use the current names (`target_accept_prob`,
`max_r_hat`, `nlsq_prior_width_factor`). `from_dict()` handles legacy keys;
internal code must use the new names.

---

## Convergence Diagnostics

### Per-shard diagnostics

| Diagnostic | Threshold | Meaning |
|---|---|---|
| R-hat (split, rank-normalized) | > 1.05 | Poor chain mixing |
| ESS (bulk) | < 100 | Insufficient effective samples |
| ESS (tail) | < 100 | Insufficient tail ESS |
| BFMI | < 0.3 | Missing energy information |
| Divergence rate | < 5% good, 5-10% warning, 10-20% high, > 20% critical | NUTS geometry problems |

### Additional diagnostics

- **Posterior Contraction Ratio (PCR)**: `1 - posterior_std / prior_std`.
  Values near 1.0 = well-constrained; near 0 = prior-dominated; negative =
  possible misspecification.

- **Trace diagnostics** (`compute_trace_diagnostics()`): Autocorrelation at
  lags 1/5/10, stationarity flag, mixing quality classification.

- **Pair correlations** (`compute_pair_correlations()`): Pairwise Pearson
  correlations between parameters; `|r| > 0.9` triggers degeneracy warning.

- **Bimodal detection** (`detect_bimodal()`): GMM 1-vs-2 component BIC
  comparison per parameter. `delta_BIC > 10` declares bimodality (strong
  evidence on Raftery scale). Two post-conditions gate the final flag:
  `min_weight` (minor-mode weight must exceed threshold) and
  `min_separation` (mode distance in posterior std-devs must exceed threshold).
  Both are wired from `CMCConfig.bimodal_min_weight` and
  `CMCConfig.bimodal_min_separation`.

- **Cross-shard bimodality** (`check_shard_bimodality()`): Runs bimodal
  detection for every (parameter, shard) combination. Called from
  `fit_cmc_sharded` immediately after `_combine_shard_posteriors`; results
  stored in `CMCResult.metadata["bimodal_detected"]` and
  `metadata["bimodal_params"]`.

- **Cross-shard clustering** (`cluster_shard_modes()`): 2-means clustering
  of shard means on bimodal parameters to identify mode populations.

- **Cross-shard summary** (`summarize_cross_shard_bimodality()`): Aggregates
  mode statistics, separation significance, and checks whether consensus mean
  falls in the density trough between modes.

### Posterior quality functions (homodyne parity)

Three diagnostic utilities in `diagnostics.py` quantify the *quality* of the
CMC posterior relative to the NLSQ warm-start and the prior:

| Function | Signature | Returns |
|---|---|---|
| `compute_posterior_contraction(result, prior_std)` | `CMCResult, dict[str, float]` | `dict[str, float]` — PCR per parameter: `1 - posterior_std / prior_std`. Values near 1 = well-constrained; near 0 = prior-dominated; negative = possible misspecification. |
| `compute_nlsq_comparison_metrics(result, nlsq_result, tolerance_sigma=3.0)` | `CMCResult, dict or NLSQResult` | `dict` — per-parameter `{diff_pct, z_score, status}`. Flags parameters exceeding `tolerance_sigma` (default 3σ). |
| `compute_precision_analysis(result, nlsq_result=None)` | `CMCResult, ...` | `dict` — precision loss diagnostics comparing CMC posterior width to NLSQ uncertainty and prior width. |

These are pure diagnostic — they do not modify the result or raise exceptions.
Typical call site: after `fit_cmc_sharded()`, pass the result and the
`nlsq_result` to surface parameter-level discrepancies before writing output.

### High-level convergence helpers (homodyne parity)

```python
DEFAULT_MIN_ESS = 400.0          # minimum acceptable bulk ESS
DEFAULT_MAX_RHAT = 1.05          # maximum acceptable R-hat
DEFAULT_MAX_DIVERGENCE_RATE = 0.05  # maximum divergence rate (5%)
```

| Function | Returns | Description |
|---|---|---|
| `check_convergence(r_hat, ess_bulk, divergences, n_samples, n_chains, ..., num_shards=1)` | `tuple[str, list[str]]` | Returns `("converged"\|"divergences"\|"not_converged", warnings)`. `"divergences"` takes priority. `num_shards` scales the denominator for CMC. |
| `create_diagnostics_dict(r_hat, ess_bulk, ess_tail, divergences, ...)` | `dict[str, Any]` | JSON-serializable dict with `convergence_status`, `divergence_rate`, `max_r_hat`, `min_ess_bulk`, `per_parameter`, `sampling_config`, `timing`. |
| `log_analysis_summary(convergence_status, r_hat, ess_bulk, ..., n_shards, shards_succeeded, execution_time)` | `None` | Logs formatted summary at INFO/ERROR with OK/FAIL indicators. |
| `get_convergence_recommendations(max_rhat, min_ess, divergences, n_samples, n_chains, num_shards=1)` | `list[str]` | Returns actionable recommendation strings for high R-hat, low ESS, or high divergence rates. |

These functions use `dict[str, float]` for r_hat/ess (keyed by parameter name). Heterodyne's `CMCResult` stores these as `np.ndarray` — callers must convert via `_r_hat_dict(result)` / `_ess_bulk_dict(result)` helpers in `io.py`.

### Sharded convergence

`validate_convergence_sharded()` runs per-shard validation and returns a
combined `ConvergenceReport` with worst-case R-hat, minimum ESS, and minimum
BFMI across all shards. A single failing shard causes the combined report
to fail.

---

## CMCResult

| Field | Type | Description |
|---|---|---|
| `parameter_names` | `list[str]` | Names in canonical order |
| `posterior_mean` | `np.ndarray` | Per-parameter posterior means |
| `posterior_std` | `np.ndarray` | Per-parameter posterior standard deviations |
| `credible_intervals` | `dict[str, dict[str, float]]` | 89% and 95% credible intervals |
| `convergence_passed` | `bool` | True if all diagnostics pass |
| `r_hat` | `np.ndarray \| None` | Split R-hat per parameter |
| `ess_bulk` | `np.ndarray \| None` | Bulk ESS per parameter |
| `ess_tail` | `np.ndarray \| None` | Tail ESS per parameter |
| `bfmi` | `list[float] \| None` | BFMI per chain |
| `samples` | `dict[str, np.ndarray] \| None` | Full posterior samples |
| `map_estimate` | `np.ndarray \| None` | Maximum a posteriori estimate |
| `num_warmup` | `int` | Warmup steps used |
| `num_samples` | `int` | Posterior draws |
| `num_chains` | `int` | Number of chains |
| `wall_time_seconds` | `float \| None` | Elapsed wall-clock time |
| `metadata` | `dict[str, Any]` | Additional metadata (n_shards, combination_method, divergence_rate, etc.) |
| `convergence_status` | `str` | `"converged"` \| `"divergences"` \| `"not_converged"` (homodyne parity) |
| `warmup_time` | `float \| None` | Wall time for warmup phase only |
| `per_angle_mode` | `str` | Effective per-angle scaling mode (default `"auto"`) |
| `chi_squared` | `float \| None` | Post-combination chi-squared |
| `quality_flag` | `str \| None` | `"good"` \| `"warning"` \| `"poor"` |
| `mean_contrast` | `np.ndarray \| None` | Per-angle posterior contrast means |
| `std_contrast` | `np.ndarray \| None` | Per-angle posterior contrast standard deviations |
| `mean_offset` | `np.ndarray \| None` | Per-angle posterior offset means |
| `std_offset` | `np.ndarray \| None` | Per-angle posterior offset standard deviations |

`convergence_status` is populated by `merge_shard_cmc_results()` and
`fit_cmc_jax()`: `"divergences"` when any shard's `metadata["divergence_rate"] > 0.05`,
`"converged"` when `convergence_passed=True`, `"not_converged"` otherwise.

### CMCResult methods

- `get_samples_array() -> np.ndarray`: Returns samples as `(num_chains, num_samples, n_params)`.
  Flat 1-D arrays of shape `num_chains * num_samples` are automatically reshaped. Missing
  parameters are filled with zeros.
- `get_posterior_stats() -> dict[str, dict[str, float]]`: Returns per-parameter dict with
  `mean`, `std`, `median`, `hdi_5%`, `hdi_95%`, `r_hat`, `ess_bulk`, `ess_tail`.
  Diagnostic fields come from the array-form `r_hat`/`ess_bulk`/`ess_tail` fields indexed
  by position. Parameters absent from `self.samples` are omitted.

### ParameterStats

`ParameterStats(dict)` is a hybrid dict/sequence class for CLI and plotting compatibility:

```python
ps = ParameterStats(["D0_ref", "alpha_ref"], [1e4, 0.5])
ps["D0_ref"]   # → 1e4  (dict-style)
ps[0]          # → 1e4  (int index)
ps.as_array    # → np.array([1e4, 0.5])
np.asarray(ps) # → array via __array__ protocol
```

Exported from `heterodyne.optimization.cmc` as part of the public API.

### Standalone functions

- `cmc_result_to_arviz()`: Converts `CMCResult` to ArviZ `InferenceData`
  with proper chain-draw reshaping.
- `compare_cmc_nlsq()`: Compares CMC posterior means with NLSQ point
  estimates; reports per-parameter z-scores and consistency flags.
- `merge_shard_cmc_results(shard_results, parameter_names=None)`:
  Inverse-variance combination of per-shard `CMCResult` objects. For
  K > 500 shards uses hierarchical chunking (groups of 500, recursive)
  to bound peak memory at O(500) × ceil(K/500) rather than O(K).
  Populates `convergence_status` on the returned result.
- `cmc_result_summary_table()`: Formatted text table with posterior means,
  standard deviations, credible intervals, R-hat, and ESS.

### SamplingStats

`SamplingStats` (`sampler.py`) is the frozen summary returned by
`run_nuts_with_retry()` after each shard sampling attempt:

| Field | Type | Description |
|---|---|---|
| `num_samples` | `int` | Posterior draws collected |
| `num_warmup` | `int` | Warmup steps used |
| `num_divergences` | `int` | Count of divergent transitions |
| `divergence_rate` | `float` | `num_divergences / (num_samples * num_chains)` |
| `mean_accept_prob` | `float` | Mean NUTS acceptance probability |
| `max_tree_depth_fraction` | `float` | Fraction of steps hitting `max_tree_depth` |
| `wall_time_seconds` | `float` | Total sampling + warmup wall time |
| `is_healthy` | `bool` | `divergence_rate < 5%` and `mean_accept_prob > 0.6` |

`is_healthy` is the fast shard-quality gate used by `run_nuts_with_retry()`
to decide whether to retry with a reduced step size before returning.
- `cmc_result_summary_table()`: Formatted text table with posterior means,
  standard deviations, credible intervals, R-hat, and ESS.

### Bimodal consensus types

`results.py` also defines two types for mode-aware combination:

| Type | Description |
|---|---|
| `ModeCluster` | Single posterior mode: mean, std, weight, and supporting shard indices |
| `BimodalConsensusResult` | Two-mode combination result: list of `ModeCluster` objects + consensus mean/std from precision-weighted combination across modes |

These are populated by `cluster_shard_modes()` (in `diagnostics.py`) when
cross-shard bimodal detection finds distinct mode populations.

---

## Data Preparation

`data_prep.py` provides sharding infrastructure:

### Sharding strategies

| Strategy | Description |
|---|---|
| `RANDOM` | Randomly assign data points with fixed seed |
| `CONTIGUOUS` | Split along time axis into contiguous blocks |
| `STRATIFIED` | Stratified by time range so each shard covers all epochs |
| `ANGLE_BALANCED` | Each shard gets proportional representation from every phi angle |

### Shard size constraints

- `min_points_per_shard`: prevents degenerate under-determined shards
- `max_points_per_shard`: `"auto"` recommended; NUTS is O(n) per leapfrog
  step, so never use 100K+ shard size
- `min_points_per_param`: default 1500 (21K minimum for 14-param model)

---

## NLSQ-to-CMC Pipeline

The warm-start pipeline extracts NLSQ values and uncertainties for CMC
initialization:

1. `extract_nlsq_values_for_cmc()`: Converts array-based `NLSQResult` to
   float dicts, filtering non-finite values
2. `validate_initial_value_bounds()`: Checks init values against registry
   bounds
3. `build_init_values_dict()`: Resolves NLSQ > `prior_mean` > `default`
   fallback chain with bound clamping
4. `build_nlsq_informed_priors()`: Centers priors on NLSQ MAP with
   width = `nlsq_unc * width_factor`
5. `transform_nlsq_to_reparam_space()`: Maps NLSQ values to Z-space
   with delta-method uncertainty propagation

---

## I/O Serialization Pipeline

`io.py` provides a complete result serialization API (homodyne parity):

| Function | Output | Description |
|---|---|---|
| `save_samples_npz(result, path)` | `samples.npz` | Shape `(n_chains, n_samples, n_params)`; schema version 1.0; r_hat/ESS arrays; `n_phi` from `result.metadata["n_phi"]` |
| `load_samples_npz(path)` | `dict[str, Any]` | Context-manager load (no file-descriptor leak); validates `.npz` extension and existence |
| `samples_to_arviz(data)` | `az.InferenceData` | Converts loaded dict to ArviZ posterior group |
| `save_fitted_data_npz(result, c2_exp, c2_fitted, c2_fitted_std, t1, t2, phi_angles, q, path)` | `fitted_data.npz` | Stores `c2_exp`, `c2_fitted`, `residuals`, 90% CI bands (`1.645 × std`) |
| `save_parameters_json(result, path)` | `parameters.json` | Calls `result.get_posterior_stats()`; NaN→`null`, Inf→`"Infinity"` |
| `save_diagnostics_json(result, path, warnings=None)` | `diagnostics.json` | Calls `create_diagnostics_dict()`; numpy-safe JSON converter |
| `save_all_results(result, output_dir, ...)` | `dict[str, Path]` | Orchestrates all above; `fitted_data.npz` only when all data arrays provided |

**Schema version** `SAMPLES_SCHEMA_VERSION = (1, 0)` allows future format evolution with backward-compatible loading.

---

## Key Design Decisions

1. **CPU-only optimization**: All backends assume CPU execution. No GPU backend
   exists — `backend_name="gpu"` is not a valid option.

2. **No analytical integrals**: The physics model always uses numerical
   integration (`trapezoid_cumsum`). Transport coefficient integrals have no
   closed-form solutions for the general power-law parameterization.

3. **Gradient-safe floors**: Uses `jnp.where(x > eps, x, eps)` instead of
   `jnp.maximum(x, eps)` to preserve non-zero gradients for NUTS leapfrog
   and NLSQ Jacobian.

4. **Smooth bounds over clip**: `smooth_bound()` (tanh) replaces `jnp.clip()`
   everywhere in the MCMC model to maintain differentiability at parameter
   boundaries.

5. **Prior tempering (not sigma-scaling)**: CMC shard sub-posteriors must be
   `prior^(1/K) · likelihood(data_k|θ)`. Widening the prior by `sqrt(K)` via
   `temper_priors(priors, K)` is correct. Dividing sigma by `sqrt(K)` is wrong:
   it multiplies the likelihood by K per shard, giving K² over-weighting in the
   combined posterior. `_temper_sigma()` has been removed.

6. **Element-wise path for CMC**: Per-shard NUTS evaluation uses `ShardGrid`
   + `compute_c2_elementwise()` to avoid O(N²) memory allocation per leapfrog
   step.

7. **Shard time-slice alignment**: Each shard's C2 sub-matrix has shape `(M,M)`
   where M < N. The NumPyro model must be built with `t[shard_t_indices]` (M
   time points), not `model.t` (N time points). Mismatch causes a shape error
   at NUTS runtime. `t_indices` is stored in every shard dict; `fit_cmc_jax`
   accepts `t_override` for this purpose.

8. **Collect both extra fields**: NUTS must request `extra_fields=("energy",
   "diverging")`. `"energy"` is needed for BFMI; `"diverging"` is needed for
   `get_divergence_stats()`. Requesting only `"energy"` silently returns zero
   divergences regardless of actual NUTS behavior.

9. **Physics-space priors in MP workers**: The `multiprocessing_backend.py`
   worker's `_shard_model` samples directly from `parameter_space.priors[name]`
   (physics-space distributions). No back-transform is needed or valid there.
   The prior tempering is handled via `priors_override` passed from
   `fit_cmc_sharded`, not by a post-sampling reparameterization.

---

## Architectural Invariants & Historical Fixes

### Design Invariants

The following invariants were established through production experience. New code must not violate them.

- **Shard size MUST use `"auto"`** — NUTS is O(n) per leapfrog step. Fixed large shard sizes cause memory exhaustion or pathologically slow sampling.
- **Priors for diffusion coefficients MUST use LogNormal** — enforces positivity in the unconstrained Z-space. Normal priors on D₀ allow negative samples.
- **`jax.block_until_ready()` MUST be called after `mcmc.run()`** — XLA lazy evaluation defers compute to the first `device_get()`; without this, `wall_time_seconds` measures only dispatch overhead.
- **Divergence rate MUST be stored in `CMCResult.metadata["divergence_rate"]`** — enables shard-level quality filtering before consensus combination.
- **Cross-shard heterogeneity MUST be checked before combination** — IQR/max(|median|, 1e-3) CV, not std/|mean|, because α, β, φ₀ parameters have near-zero medians.
- **Bimodal detection MUST run after `_combine_shard_posteriors`** — calling it before defeats its purpose (mode separation only apparent in the combined posterior).
- **Preflight log-density check uses `numpyro.infer.util.log_density()`** — `kernel._potential_fn` is `None` before sampling; checking it always silently no-ops.

### Critical Features & Fixes

| Version | Fix | Files |
|---|---|---|
| 2026-05-08 | **C1 — Sharded likelihood shape mismatch**: shard C2 of shape `(M,M)` was observed against model built with full `model.t` (N points) → crash. Added `t_indices` to shard dicts; `fit_cmc_jax` accepts `t_override`. Random non-square shards raise `ValueError`. | `core.py` |
| 2026-05-08 | **C4 — Divergence tracking disabled**: `extra_fields=("energy",)` only; `get_divergence_stats()` reads `"diverging"` → always zero. Added `"diverging"` to all 4 `mcmc.run()` call sites. | `core.py`, `sampler.py`, `backends/multiprocessing_backend.py` |
| 2026-05-08 | **C2 — Wrong CMC prior tempering**: `_temper_sigma()` divided sigma by `sqrt(K)`, multiplying each shard likelihood by K → K²-over-weighted posterior. Replaced with `temper_priors(priors, K)`. `temper_priors()` already existed in `priors.py` but was unused. | `core.py`, `model.py` |
| 2026-05-08 | **C3 — Failed-shard contamination**: zero-std failed shards received weight `1/1e-30 = 1e30` in consensus; `np.stack([])` crashed when all shards failed. Added pre-combination filter on `successful` shards; all-failed returns degenerate `CMCResult`. | `core.py` |
| 2026-05-08 | **C5 — MP worker reparam crash**: worker called `reparam_to_physics_jax(params, reparam_config)` but signature is `(log_at_tref, alpha, t_ref)`. Worker samples physics-space priors directly — no back-transform needed. Removed broken block. | `backends/multiprocessing_backend.py` |
| 2026-05-08 | **ArviZ/NumPyro import-order incompatibility**: `arviz_base.io_numpyro` accesses `numpyro.infer.initialization` as a package attribute, but heterodyne's import chain loads it into `sys.modules` without setting the attribute (circular-import timing). Fixed by explicit attribute patch before `az.from_numpyro()`. | `core.py` |
| 2026-05-08 | **Dead parameters removed from `get_heterodyne_model_reparam`**: `nlsq_result` and `prior_width_factor` were accepted but never read. The new path uses `scalings`; the legacy clip path hardcodes `scale = (bounds[1]-bounds[0])/6`. Both removed from signature and call sites. | `model.py`, `core.py` |
| 2026-05-08 | **`dense_mass` default changed to `True`**: The 14-param model has three correlated power-law pairs that produce banana-shaped posteriors. A diagonal mass matrix (`False`) cannot navigate these cross-correlations and inflates divergences. Changed in `CMCConfig`, `SamplingPlan`, and PBS backend fallback. | `config.py`, `sampler.py`, `backends/pbs.py` |
| 2026-05-08 | **W2 — Reparam path uses too-low acceptance target**: `config.target_accept_prob` (default 0.8) was used unchanged for the reparameterized path. Z-space posteriors have longer correlation lengths; NUTS picks excessively large step sizes at 0.8, causing divergences. Fixed: `effective_target_accept = max(config.target_accept_prob, 0.9)` for the reparam path only. | `core.py` |
| 2026-05-08 | **W3 — ReparamConfig flags not wired from CMCConfig**: `ReparamConfig()` was constructed with all enable flags at their defaults (`False`), making `config.reparameterization_d_total` and `config.reparameterization_log_gamma` silently ignored. Fixed: flags are now explicitly set from the config fields. | `core.py` |
| 2026-05-08 | **W4 — Bimodal detection not called after shard combination**: `check_shard_bimodality()` was never invoked in `fit_cmc_sharded`. CMCConfig fields `bimodal_min_weight` and `bimodal_min_separation` had no effect. Fixed: call added after `_combine_shard_posteriors`; results stored in `CMCResult.metadata`. | `core.py` |
| 2026-05-08 | **W5 — Bimodal detection post-conditions not enforced**: `detect_bimodal()` / `check_shard_bimodality()` had no `min_weight` or `min_separation` parameters, so `CMCConfig.bimodal_min_weight` and `bimodal_min_separation` were unreachable. Fixed: both parameters added and applied as post-conditions after the BIC test. | `diagnostics.py` |
| 2026-05-08 | **W6 — Preflight log-density check always skipped**: `_validate_init_log_density` checked `kernel._potential_fn` which is `None` before any sampling run → preflight silently no-ops on every call. Fixed: now uses `numpyro.infer.util.log_density(kernel.model, (), {}, init_params)` which is always callable. | `sampler.py` |
| 2026-05-08 | **Phase 3 — NL-07: `classify_fit_quality` bounds-proximity parameter**: Added optional `n_at_bounds: int = 0` parameter. When `n_at_bounds > 0`, a chi-squared-"good" result is demoted to "marginal" — a bound-saturated parameter may absorb residual error and mask convergence issues. Existing callers unaffected (backward-compatible default). | `optimization/nlsq/validation/fit_quality.py` |
| 2026-05-08 | **Phase 3 — CM-07: CMCResult homodyne-parity fields**: Added `convergence_status`, `warmup_time`, `per_angle_mode`, `chi_squared`, `quality_flag`, `mean_contrast`, `std_contrast`, `mean_offset`, `std_offset` as optional fields (all default to `None`/`"not_converged"`/`"auto"`). `merge_shard_cmc_results` populates `convergence_status` automatically. | `optimization/cmc/results.py` |
| 2026-05-08 | **Phase 3 — CM-09: hierarchical combination for K > 500 shards**: `merge_shard_cmc_results` now chunks into groups of 500 and combines recursively when `len(shard_results) > 500`, bounding peak memory to O(500) × ceil(K/500). | `optimization/cmc/results.py` |
| 2026-05-08 | **Phase 3 — CM-04: `jax.block_until_ready()` in `NUTSSampler.run()`**: timing measurement was unreliable — XLA's lazy evaluation deferred actual compute to the first `device_get()` call, so `wall_time_seconds` measured only dispatch time. Fixed by calling `jax.block_until_ready(mcmc.last_state)` immediately after `mcmc.run()`. | `sampler.py` |
| 2026-05-08 | **Phase 3 — CM-02: divergence-rate shard filter**: `fit_cmc_jax` now extracts `mcmc.get_extra_fields()["diverging"]` and stores `divergence_rate` in `CMCResult.metadata`. `_combine_shard_posteriors` gates the `successful` filter on `metadata.get("divergence_rate", 0.0) <= config.max_divergence_rate` (default 10%), preventing high-divergence shards from contaminating the consensus posterior. | `core.py` |
| 2026-05-08 | **Phase 3 — CM-03: cross-shard heterogeneity detection**: `max_parameter_cv` and `heterogeneity_abort` config fields existed but had no enforcement. Added IQR-based CV check (`IQR / max(\|median\|, 1e-3)`) between shard filtering and combination — uses IQR rather than std/\|mean\| so near-zero params (α, β, v_offset, φ₀ ≈ 0) stay finite. Raises `RuntimeError` or logs warning per `heterogeneity_abort`. | `core.py` |
| 2026-05-08 | **Phase 3 — CM-01: `robust_consensus_mc` wiring**: `_combine_shard_posteriors` routed all non-`simple_average` methods (including `robust_consensus_mc`) to the same inverse-variance `else` branch — the config option was a silent no-op. Added dedicated `elif` branch with per-parameter z-score outlier detection before inverse-variance combination. `scale = max(std, 1e-4*(|mean|+1))` prevents near-zero parameters from producing infinite z-scores. | `core.py` |
| 2026-05-08 | **Phase 2 — homodyne parity additions**: (1) `ParameterStats` hybrid dict/sequence; `CMCResult.get_samples_array()`, `CMCResult.get_posterior_stats()`; (2) `DEFAULT_MIN_ESS/MAX_RHAT/MAX_DIVERGENCE_RATE` constants; `check_convergence()`, `create_diagnostics_dict()`, `log_analysis_summary()`, `get_convergence_recommendations()`; (3) full io.py save pipeline (7 functions); (4) `SamplingPlan.chain_method` wired through `from_config()`, `for_shard()`, `AdaptiveSamplingPlan.get_plan()`, and retry loop; (5) `estimate_contrast_offset_from_data()` in priors; (6) `combination_method` dispatch in `_combine_shard_posteriors`. | `results.py`, `diagnostics.py`, `io.py`, `sampler.py`, `config.py`, `priors.py`, `core.py` |

For the full bug narrative, see `docs/changelog/cmc-architecture-fixes.md`.
