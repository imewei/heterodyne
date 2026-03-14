# CMC Fitting Architecture

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

## Component Map

```
optimization/cmc/
├── core.py               # fit_cmc_jax() unified entry, fit_cmc_sharded()
├── model.py              # NumPyro models (meshgrid + element-wise paths)
├── priors.py             # build_default_priors(), build_nlsq_informed_priors(),
│                         #   build_log_space_priors(), temper_priors()
├── sampler.py            # SamplingPlan, NUTSSampler, AdaptiveSamplingPlan,
│                         #   run_nuts_with_retry(), SamplingStats
├── reparameterization.py # ReparamConfig, compute_t_ref(), power-law decorrelation
├── scaling.py            # ParameterScaling, smooth_bound() (tanh-based)
├── diagnostics.py        # R-hat, ESS, BFMI, divergence analysis,
│                         #   bimodal detection, cross-shard clustering
├── config.py             # CMCConfig dataclass (14 config sections)
├── results.py            # CMCResult, merge_shard_cmc_results(), compare_cmc_nlsq()
├── data_prep.py          # ShardingStrategy, PreparedData, sigma estimation
├── plotting.py           # CMC-specific plotting utilities
├── io.py                 # Posterior serialization
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
fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result)
        │
        ├─ estimate_sigma(c2_data, method)
        │     diagonal / constant / local / residual / bootstrap
        │
        ├─ select_backend(config)
        │     ├── len(devices) > 1  → PjitBackend
        │     └── single CPU device → CPUBackend
        │
        ├─ ReparamConfig + compute_t_ref(dt, t_max)
        │     t_ref = sqrt(dt × t_max)
        │
        ├─ transform_nlsq_to_reparam_space(nlsq_values, nlsq_unc, t_ref)
        │     delta-method uncertainty propagation to Z-space
        │
        ├─ get_heterodyne_model_reparam(...)   # reparameterized path
        │   or get_heterodyne_model(...)       # physics-space path
        │
        ├─ backend.run(model, config, rng_key, init_params)
        │
        ├─ transform_to_physics_space(samples, reparam_config)
        │
        └─ validate_convergence(result)
              R-hat, ESS, BFMI, posterior contraction checks
```

### Sharded path: `fit_cmc_sharded()`

```
fit_cmc_sharded(model, c2_data, config, nlsq_result)
        │
        ├─ Data preparation: validate + estimate sigma
        │
        ├─ Sharding: stratified / random / contiguous / angle-balanced
        │     partitioning of data into K shards
        │
        ├─ Prior tempering: scale prior width by sqrt(K)
        │     preserves correct posterior under shard factorization
        │
        ├─ Per-shard NUTS sampling via selected backend
        │     (parallel or sequential across shards)
        │
        ├─ Consensus MC combination
        │     precision-weighted posterior: Λ_combined = Σ_k Λ_k
        │     μ_combined = Λ_combined⁻¹ Σ_k Λ_k μ_k
        │
        ├─ Diagnostics: R-hat, ESS, BFMI, divergence rate,
        │     bimodal detection, cross-shard clustering
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
| `get_heterodyne_model()` | Basic model: direct prior sampling, meshgrid or element-wise physics |
| `get_heterodyne_model_reparam()` | Reparameterized: Z-space + smooth bounds + NLSQ-informed priors (or legacy clip path) |
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

---

## Z-Space Reparameterization

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
| `dense_mass` | False | Full-covariance vs diagonal mass matrix |
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
- `run()`: executes sampling with per-chain perturbation of init params
- `run_with_init_values()`: warm-start from NLSQ MAP with preflight log-density
  validation
- `get_divergence_stats()`: divergence rate, mean tree depth, max-depth fraction
- `get_diagnostics()`: returns ArviZ `InferenceData` via `az.from_numpyro()`

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

Simpler inverse-variance combination operating on `CMCResult` objects:

```
precision_i = 1 / std_i^2
combined_mean = Σ(precision_i × mean_i) / Σ(precision_i)
combined_std  = 1 / sqrt(Σ(precision_i))
```

Diagnostics use worst-case values: maximum R-hat, minimum ESS, minimum BFMI.
Credible intervals are reconstructed from combined Gaussian approximation.

### Combination methods

| Method | Description |
|---|---|
| `consensus_mc` | Full precision-matrix weighting |
| `robust_consensus_mc` | Outlier-resistant precision weighting |
| `weighted_gaussian` | Weighted Gaussian approximation |
| `simple_average` | Unweighted mean of shard posteriors |

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
| `backend_name` | `"auto"` | `"auto"`, `"multiprocessing"`, `"pjit"`, `"cpu"`, `"gpu"` |
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
| `dense_mass` | False | Full-covariance mass matrix |
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
  evidence on Raftery scale).

- **Cross-shard bimodality** (`check_shard_bimodality()`): Runs bimodal
  detection for every (parameter, shard) combination.

- **Cross-shard clustering** (`cluster_shard_modes()`): 2-means clustering
  of shard means on bimodal parameters to identify mode populations.

- **Cross-shard summary** (`summarize_cross_shard_bimodality()`): Aggregates
  mode statistics, separation significance, and checks whether consensus mean
  falls in the density trough between modes.

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
| `metadata` | `dict[str, Any]` | Additional metadata (n_shards, combination_method, etc.) |

### Standalone functions

- `cmc_result_to_arviz()`: Converts `CMCResult` to ArviZ `InferenceData`
  with proper chain-draw reshaping.
- `compare_cmc_nlsq()`: Compares CMC posterior means with NLSQ point
  estimates; reports per-parameter z-scores and consistency flags.
- `merge_shard_cmc_results()`: Inverse-variance combination of per-shard
  `CMCResult` objects into a consensus result.
- `cmc_result_summary_table()`: Formatted text table with posterior means,
  standard deviations, credible intervals, R-hat, and ESS.

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

## Key Design Decisions

1. **CPU-only optimization**: All backends assume CPU execution. The GPU
   backend is a wrapper but heterodyne is not designed for GPU workloads.

2. **No analytical integrals**: The physics model always uses numerical
   integration (`trapezoid_cumsum`). Transport coefficient integrals have no
   closed-form solutions for the general power-law parameterization.

3. **Gradient-safe floors**: Uses `jnp.where(x > eps, x, eps)` instead of
   `jnp.maximum(x, eps)` to preserve non-zero gradients for NUTS leapfrog
   and NLSQ Jacobian.

4. **Smooth bounds over clip**: `smooth_bound()` (tanh) replaces `jnp.clip()`
   everywhere in the MCMC model to maintain differentiability at parameter
   boundaries.

5. **Prior tempering**: Shard sub-posteriors use priors widened by `sqrt(K)`
   so that the K-fold product of sub-posteriors recovers the full-data
   posterior (when Gaussian).

6. **Element-wise path for CMC**: Per-shard NUTS evaluation uses `ShardGrid`
   + `compute_c2_elementwise()` to avoid O(N^2) memory allocation per
   leapfrog step.
