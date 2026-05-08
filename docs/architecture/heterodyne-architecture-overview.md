<!-- Package: heterodyne | Last verified: 2026-05-08 -->

# Heterodyne Package Architecture Overview

## Overview

`heterodyne` is a CPU-optimized JAX package for two-component heterodyne X-ray
Photon Correlation Spectroscopy (XPCS) analysis under nonequilibrium conditions.
It fits the two-time correlation function c2(t1, t2) to a 14-parameter physics
model plus 2 per-angle scaling parameters (contrast and offset) using a two-stage
pipeline: non-linear least squares (NLSQ) warm-start followed by Bayesian
posterior sampling via Consensus Monte Carlo (NumPyro NUTS).

**Stack:** Python 3.12+, JAX >=0.8.2 (CPU-only), NumPy >=2.3, NLSQ >=0.6.10.
Float64 precision is required (`JAX_ENABLE_X64=1` set before first JAX import).

---

## Terminology Note: g2 vs c2

The homodyne package uses `g2` (normalized intensity autocorrelation). The heterodyne package uses `c2` (two-component heterodyne correlation). These are equivalent concepts in different physical models:

- `g2(q, τ) = 1 + β|f(q,τ)|²` — homodyne, single-component
- `c2(q, τ) = c₁(τ) + c₂(τ)` — heterodyne, two-component (reference + sample dynamics)

Both represent the normalized time autocorrelation of scattered intensity. When reading homodyne documentation, mentally substitute `g2 → c2` and note that heterodyne adds a second decay channel via the reference beam.

---

## Module Map

```
heterodyne/
├── core/                        # Physics kernels and model abstractions
│   ├── jax_backend.py                # JIT-compiled c2 computation (meshgrid path, N×N)
│   ├── physics_cmc.py                # Element-wise c2 computation (CMC path, O(n_pairs))
│   ├── physics_nlsq.py               # NLSQ residual/Jacobian adapters (upper-triangle)
│   ├── physics_utils.py              # Shared primitives: trapezoid_cumsum, smooth_abs, rate functions
│   ├── physics.py                    # PhysicsConstants, PARAMETER_BOUNDS, ValidationResult
│   ├── theory.py                     # TheoryEngine: compute_c2_theory(), compute_chi2_theory(),
│   │                                 #   compute_g1_decay(), compute_velocity_field()
│   ├── models.py                     # TwoComponentModel, ReducedModel, create_model()
│   ├── heterodyne_model.py           # HeterodyneModel stateful wrapper + ParameterManager
│   ├── fitting.py                    # UnifiedHeterodyneEngine, ParameterSpace, DatasetSize,
│   │                                 #   FitResult, compute_c2_batch(), fit utilities
│   ├── scaling_utils.py              # Per-angle contrast/offset estimation (quantile-based)
│   ├── physics_factors.py            # PhysicsFactors + CachedMatrices (pre-computed q²/2·dt)
│   ├── diagonal_correction.py        # Autocorrelation peak removal (basic/statistical/interpolation)
│   ├── numpy_gradients.py            # Numerical differentiation fallback (Richardson, complex-step)
│   ├── model_mixins.py               # TransportMixin, FractionMixin, VelocityMixin (reusable physics)
│   └── backend_api.py                # Backend enum, BackendConfig, get_current_backend()
│
├── config/                      # Parameter metadata and configuration management
│   ├── parameter_registry.py         # Immutable registry (MappingProxyType) - 16 entries (14+2)
│   ├── parameter_names.py            # ALL_PARAM_NAMES canonical ordering
│   ├── parameter_manager.py          # ParameterManager: vary flags, bounds, defaults
│   ├── parameter_space.py            # ParameterSpace for priors + single-angle stabilization
│   ├── physics_validators.py         # Declarative constraint rules per parameter
│   ├── types.py                      # TypedDict config structures
│   └── manager.py                    # ConfigManager (YAML/JSON loading)
│
├── data/                        # I/O, preprocessing, and data quality
│   ├── xpcs_loader.py                # HDF5/NPZ/MAT/NPY loading -> XPCSData
│   ├── angle_filtering.py            # Phi-angle range selection
│   ├── preprocessing.py              # PreprocessingPipeline
│   ├── memory_manager.py             # MemoryManager budget tracking
│   ├── quality_controller.py         # Data quality scoring
│   ├── validators.py                 # Shape, dtype, finiteness checks
│   ├── filtering_utils.py            # NaN masking helpers
│   ├── optimization.py               # Chunk-size optimization
│   ├── performance_engine.py         # Profiling instrumentation
│   ├── types.py                      # AngleRange dataclass
│   └── config.py                     # DataConfig dataclass
│
├── optimization/
│   ├── nlsq/                    # Primary: Trust-region Levenberg-Marquardt optimizer
│   │   ├── core.py                   # fit_nlsq_jax(), fit_nlsq_multi_phi() - main entry
│   │   ├── adapter.py                # NLSQAdapter (JAX-traced, primary)
│   │   ├── adapter_base.py           # NLSQAdapterBase
│   │   ├── config.py                 # NLSQConfig, HybridRecoveryConfig, NLSQValidationConfig
│   │   ├── results.py                # NLSQResult dataclass
│   │   ├── fourier_reparam.py        # FourierReparameterizer for angular scaling
│   │   ├── cmaes_wrapper.py          # CMA-ES global optimization
│   │   ├── multistart.py             # Multi-start with LHS sampling
│   │   ├── fallback_chain.py         # Strategy selection + automatic degradation
│   │   ├── recovery.py               # 3-attempt error recovery
│   │   ├── anti_degeneracy_controller.py  # 4-layer defense system
│   │   ├── hierarchical.py           # Two-stage phys/scaling optimization
│   │   ├── adaptive_regularization.py     # CV-based regularization
│   │   ├── gradient_monitor.py       # Real-time gradient diagnostics
│   │   ├── jacobian.py               # Jacobian analysis
│   │   ├── memory.py                 # Memory-aware strategy selection
│   │   ├── data_prep.py              # Data preparation + weight computation
│   │   ├── parameter_utils.py        # Parameter utilities
│   │   ├── parameter_index_mapper.py # Varying/fixed index mapping
│   │   ├── parallel_accumulator.py   # Parallel residual accumulation
│   │   ├── transforms.py             # Parameter transforms
│   │   ├── progress.py               # Progress reporting
│   │   ├── result_builder.py         # NLSQResult construction
│   │   ├── strategies/
│   │   │   ├── base.py               # FittingStrategy abstract base
│   │   │   ├── stratified_ls.py      # Stratified least squares + anti-degeneracy
│   │   │   ├── hybrid_streaming.py   # Streaming + gradient accumulation
│   │   │   ├── out_of_core.py        # Disk-based JTJ accumulation
│   │   │   ├── sequential.py         # Per-angle sequential fitting
│   │   │   ├── jit_strategy.py       # JAX JIT residual evaluation
│   │   │   └── chunked.py            # Chunked evaluation
│   │   └── validation/
│   │       ├── input_validator.py
│   │       ├── convergence.py
│   │       ├── fit_quality.py
│   │       ├── bounds.py
│   │       ├── result_validator.py
│   │       └── result.py
│   │
│   ├── cmc/                     # Secondary: Consensus Monte Carlo (NumPyro)
│   │   ├── core.py                   # fit_cmc_jax() - unified entry (sharding + consensus)
│   │   ├── model.py                  # NumPyro model (meshgrid + element-wise paths)
│   │   ├── priors.py                 # Default + NLSQ-informed prior construction
│   │   ├── sampler.py                # SamplingPlan, NUTS wrapper, retry logic
│   │   ├── reparameterization.py     # t_ref transforms for power-law pairs
│   │   ├── scaling.py                # ParameterScaling (smooth bounded transforms)
│   │   ├── diagnostics.py            # R-hat, ESS, BFMI, bimodal detection
│   │   ├── config.py                 # CMCConfig (14 config sections)
│   │   ├── results.py                # CMCResult + merge/compare utilities
│   │   ├── data_prep.py              # Shard preparation, sigma estimation
│   │   ├── io.py                     # Posterior serialization
│   │   └── backends/
│   │       ├── base.py               # MCMCBackend protocol + select_backend()
│   │       ├── cpu_backend.py        # Sequential NUTS
│   │       ├── multiprocessing_backend.py  # Process-pool parallel
│   │       ├── pjit_backend.py       # JAX pjit distributed
│   │       ├── pbs.py                # PBS/Torque cluster
│   │       └── worker_pool.py        # WorkerPoolBackend
│   │
│   ├── exceptions.py                 # Unified exception hierarchy
│   ├── checkpoint_manager.py         # SHA-256 checksums, atomic writes
│   ├── batch_statistics.py           # Batch diagnostics
│   ├── gradient_diagnostics.py       # Gradient analysis
│   ├── numerical_validation.py       # NaN/Inf detection
│   └── recovery_strategies.py        # Failure recovery patterns
│
├── viz/                         # MCMC diagnostics, comparison, dashboard, ArviZ
├── cli/                         # CLI entry points (heterodyne, heterodyne-config, etc.)
├── device/                      # CPU/NUMA detection, XLA flag configuration
├── io/                          # JSON/NPZ serialization
├── utils/                       # Logging (log_phase), checkpoints, misc
└── runtime/                     # Shell completion system
```

---

## Two-Path Integral Architecture

The physics model uses two distinct integral evaluation paths that share
primitives from `physics_utils.py` (trapezoid_cumsum, rate functions, smooth_abs):

**1. Meshgrid path (NLSQ):** `core/jax_backend.py` builds a full N x N
time-integral matrix via `create_time_integral_matrix()`. The cumsum-based
construction is fast under JIT compilation but requires O(N^2) memory. Used by
`fit_nlsq_jax()` where the entire c2 matrix is evaluated at once.

**2. Element-wise path (CMC):** `core/physics_cmc.py` uses `ShardGrid` with
`precompute_shard_grid()` for O(n_pairs) cumsum lookup. No N x N matrix is ever
materialized. Designed for per-shard NUTS evaluation where each leapfrog step
evaluates only the shard's (t1, t2) pairs.

Both paths produce identical c2 values for the same parameters. The NLSQ path
feeds residuals and Jacobians through `core/physics_nlsq.py` (upper-triangle
extraction). The CMC path is called directly from the NumPyro model in
`optimization/cmc/model.py`.

---

## Analysis Pipeline

```
YAML config
    |
    v
ConfigManager (config/manager.py)
    |
    v
XPCSDataLoader (data/xpcs_loader.py)
    |   loads HDF5/NPZ/MAT/NPY, validates shape/dtype/finiteness
    v
AngleFiltering / PreprocessingPipeline
    |   phi-range selection, quality scoring, NaN masking
    v
HeterodyneModel.from_config() or create_model()
    |   builds model with ParameterManager (vary flags, bounds)
    v
fit_nlsq_jax()  -  NLSQ warm-start
    |   strategy selection (stratified/streaming/out-of-core/sequential)
    |   anti-degeneracy controller, fallback chain, 3-attempt recovery
    v
NLSQResult (best-fit parameters, covariance, chi-squared)
    |
    v
fit_cmc_jax()  -  Bayesian posterior via NumPyro NUTS
    |   NLSQ-informed priors, data sharding, consensus merge
    v
CMC diagnostics (R-hat, ESS, BFMI)
    |
    v
CMCResult (posterior samples, diagnostics) -> JSON + NPZ
```

---

## 14 Physics Parameters + 2 Scaling Parameters

All 14 physics parameters plus 2 per-angle scaling parameters (contrast, offset),
organized in five groups:

**Reference transport** — `J_r(t) = D0_ref·t^alpha_ref + D_offset_ref`

| Parameter | Description | Default | Units |
|---|---|---|---|
| D0_ref | Reference diffusion prefactor | 1e4 | Å²/s^α |
| alpha_ref | Reference transport exponent | 0.0 | — |
| D_offset_ref | Reference transport rate offset | 0.0 | Å²/s |

**Sample transport** — `J_s(t) = D0_sample·t^alpha_sample + D_offset_sample`

| Parameter | Description | Default | Units |
|---|---|---|---|
| D0_sample | Sample diffusion prefactor | 1e4 | Å²/s^α |
| alpha_sample | Sample transport exponent | 0.0 | — |
| D_offset_sample | Sample transport rate offset | 0.0 | Å²/s |

**Velocity** — `v(t) = v0·t^beta + v_offset`

| Parameter | Description | Default | Units |
|---|---|---|---|
| v0 | Velocity prefactor | 1e3 | Å/s^β |
| beta | Velocity exponent (0 = constant) | 0.0 | — |
| v_offset | Velocity offset (negative = reversal) | 0.0 | Å/s |

**Sample fraction** — `f_s(t) = clip(f0·exp(f1·(t − f2)) + f3, 0, 1)`

| Parameter | Description | Default | Units |
|---|---|---|---|
| f0 | Fraction amplitude | 0.5 | — |
| f1 | Exponential rate (0 = constant) | 0.0 | 1/s |
| f2 | Time shift | 0.0 | s |
| f3 | Baseline offset | 0.0 | — |

**Flow angle**

| Parameter | Description | Default | Units |
|---|---|---|---|
| phi0 | Flow angle offset relative to q-vector | 0.0 | degrees |

**Per-angle scaling** (not in the physics array — passed separately)

| Parameter | Description | Default | Units |
|---|---|---|---|
| contrast | Speckle contrast | estimated | — |
| offset | Baseline offset | estimated | — |

The parameter registry in `config/parameter_registry.py` stores all 16 entries
as an immutable `MappingProxyType`. Bounds, defaults, and prior statistics
(prior_mean, prior_std) cannot be mutated at runtime.

---

## Design Principles

**JAX-first, stateless physics.**
`compute_c2_heterodyne` in `core/jax_backend.py` is a pure function decorated
with `@jax.jit`. It accepts a flat parameter array and returns the c2 matrix.
No mutable state. All JAX transformations (jit, vmap, grad, jacobian) apply
directly. Gradient-safe floors use `jnp.where(x > eps, x, eps)` instead of
`jnp.maximum` to preserve gradients for NLSQ Jacobian and NUTS leapfrog.

**CPU-only.**
The package targets CPU execution exclusively. `device/cpu.py` handles
CPU/NUMA detection and XLA flag configuration. There is no GPU acceleration
path for the physics kernels.

**Immutable parameter configuration.**
`parameter_registry.py` uses `MappingProxyType` for the default registry.
Bounds, defaults, and prior statistics cannot be mutated at runtime. This
prevents accidental configuration drift between the NLSQ and CMC stages.

**Dual prior system.**
`parameter_registry.py` (prior_mean/prior_std) and `parameter_space.py`
(`_DEFAULT_PRIOR_SPECS`) must stay in sync. The registry is consumed by
`cmc/priors.py` for building default and log-space priors. The
`_DEFAULT_PRIOR_SPECS` are consumed by `parameter_space.py:_default_prior()`
for ParameterSpace initialization.

**Analysis mode factory.**
`create_model(mode)` in `core/models.py` returns a `TwoComponentModel` for
`"two_component"` (all 14 parameters free) or a `ReducedModel` that freezes
inactive parameters at canonical defaults. Registered modes: `"static_ref"`,
`"static_both"`, `"two_component"`.

**Two-path integral architecture.**
The meshgrid path (N x N matrix) serves NLSQ where the full c2 surface is
needed for residual/Jacobian computation. The element-wise path (O(n_pairs)
cumsum lookup) serves CMC where per-shard evaluation avoids materializing the
full matrix. Both paths share numerical primitives and produce identical results.

**Backend selection at runtime.**
`select_backend(config)` in `optimization/cmc/backends/base.py` returns the
appropriate CMC backend: `CPUBackend` (sequential), `MultiprocessingBackend`
(process-pool), `PjitBackend` (JAX distributed), `PBSBackend` (cluster), or
`WorkerPoolBackend`.

**Numerical integration only.**
Transport coefficient integrals always use numerical integration
(`trapezoid_cumsum`). The general power-law parameterization has no closed-form
solutions; analytical shortcuts introduce silent approximation errors.

**Units.**
All quantities use Angstroms: q in A^-1, D0 in A^2/s^alpha, velocities in A/s.
Wavelength follows lambda = 12.398 / E[keV] A.

**Float64 precision.**
`JAX_ENABLE_X64=1` must be set before the first JAX import.
`heterodyne/__init__.py` and `cli/main.py` both call
`os.environ.setdefault("JAX_ENABLE_X64", "1")`. Multiprocessing workers re-set
the variable since spawn-mode starts fresh.

---

## NLSQ Optimization

The primary optimization path uses trust-region Levenberg-Marquardt via
`fit_nlsq_jax()` in `optimization/nlsq/core.py`.

### Per-Angle Scaling Modes

```yaml
optimization:
  nlsq:
    anti_degeneracy:
      per_angle_mode: "auto"  # "auto", "constant", "individual", "fourier"
```

| Mode | Behavior |
|---|---|
| auto | n_phi >= 3: averaged scaling, else individual |
| constant | Fixed scaling from quantile estimation |
| individual | Independent contrast/offset per angle |
| fourier | Truncated Fourier series for angular variation |

Multi-angle fitting uses `fit_nlsq_multi_phi()` for joint optimization when
Fourier or individual mode is selected with more than one angle.

### Strategy Selection

The fallback chain in `nlsq/fallback_chain.py` selects a fitting strategy based
on data size and available memory:

- **Stratified LS** (`stratified_ls.py`): Primary strategy with anti-degeneracy.
- **JIT strategy** (`jit_strategy.py`): JAX JIT-compiled residual evaluation.
- **Sequential** (`sequential.py`): Per-angle sequential fitting.
- **Hybrid streaming** (`hybrid_streaming.py`): Gradient accumulation for large datasets.
- **Out-of-core** (`out_of_core.py`): Disk-based JTJ accumulation for memory-constrained runs.
- **Chunked** (`chunked.py`): Chunked evaluation.

### CMA-ES Global Optimization

```yaml
optimization:
  nlsq:
    cmaes:
      enable: true
      preset: "cmaes-global"
      refine_with_nlsq: true
      nlsq_warmstart: true
      warmstart_auto_skip: true
```

CMA-ES (`cmaes_wrapper.py`) provides global search for multi-scale problems,
optionally refined with a subsequent NLSQ fit.

### Error Recovery

`recovery.py` implements a 3-attempt error recovery protocol with parameter
perturbation and strategy degradation between attempts.

---

## CMC (Consensus Monte Carlo)

The secondary optimization path uses NumPyro NUTS sampling via `fit_cmc_jax()`
in `optimization/cmc/core.py`.

### Warm-Start from NLSQ

```python
from heterodyne.optimization.nlsq import fit_nlsq_jax
from heterodyne.optimization.cmc import fit_cmc_jax

nlsq_result = fit_nlsq_jax(model, c2_data, phi_angle, config)
cmc_result = fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result=nlsq_result)
```

NLSQ provides warm-start initialization for CMC: the MAP estimate sets NUTS chain starting positions and the diagonal of JᵀJ provides the initial mass matrix estimate. This reduces burn-in by approximately 60% vs. cold-start sampling. **For the full data contract (input shapes, covariance fallback, reparameterization step, failure behavior), see `cmc-fitting-architecture.md §NLSQ-to-CMC Pipeline`.**

### Data Sharding

Data is partitioned into shards for parallel NUTS evaluation. Each shard runs
independent chains; results are merged via consensus.

```yaml
optimization:
  cmc:
    sharding:
      max_points_per_shard: "auto"  # Always use auto
```

NUTS is O(n) per leapfrog step. Shard sizes above 100K points degrade
performance severely.

### Reparameterization

`reparameterization.py` implements t_ref transforms for power-law parameter
pairs. `scaling.py` provides smooth bounded transforms for constrained sampling.

### Diagnostics

`diagnostics.py` computes mandatory convergence diagnostics: R-hat (split),
effective sample size (ESS), Bayesian Fraction of Missing Information (BFMI),
and bimodal detection. ArviZ integration is available through `viz/`.

### CMCConfig Attribute Names

Current attribute names (not legacy):

- `target_accept_prob` (not `target_accept`)
- `max_r_hat` (not `r_hat_threshold`)
- `nlsq_prior_width_factor` (not `prior_width_factor`)

`from_dict()` handles legacy keys; internal code must use the current names.

---

## Key Entry Points

| Task | Function / Class |
|---|---|
| Load data | `XPCSDataLoader(path).load()` |
| Build model | `HeterodyneModel.from_config(config)` or `create_model("two_component")` |
| NLSQ fit | `fit_nlsq_jax(model, c2_data, phi_angle, config)` |
| Multi-angle NLSQ | `fit_nlsq_multi_phi(model, c2_data_batch, phi_angles, config)` |
| Bayesian fit | `fit_cmc_jax(model, c2_data, phi_angle, config, nlsq_result=nlsq_result)` |
| Physics kernel (meshgrid) | `compute_c2_heterodyne(params, t, q, dt, phi_angle)` |
| Physics kernel (element-wise) | `compute_c2_elementwise(params, shard_grid, q, dt, phi_angle)` |
| Parameter info | `DEFAULT_REGISTRY["D0_ref"]` |

---

## CLI Entry Points

| Command | Purpose |
|---|---|
| `heterodyne` | Main analysis (NLSQ/CMC) |
| `heterodyne-config` | Config generation/validation |
| `heterodyne-config-xla` | XLA device configuration |
| `heterodyne-post-install` | Shell completion setup |
| `heterodyne-cleanup` | Remove shell completion files |
| `heterodyne-validate` | System validation |

---

## Checkpoint and Recovery

`optimization/checkpoint_manager.py` provides SHA-256 checksums, version
tracking, and atomic writes for intermediate results. `find_latest_valid()`
locates the most recent valid checkpoint for warm-restart after failures.

---

## Contributor Quick Reference

### How to add a physics parameter

Touch these files in order:

1. `config/parameter_registry.py` — add `ParameterInfo` with bounds, `prior_mean`, `prior_std`, `vary_default`, `is_scaling`
2. `core/physics.py` — add physical constant if needed
3. `core/models.py` — update `ParameterSpace` if the parameter changes the model dimension
4. `core/heterodyne_model.py` — update `HeterodyneModel` constructor if initialization changes
5. `optimization/nlsq/` — update residual/Jacobian adapters if the parameter appears in the physics kernel
6. `optimization/cmc/priors.py` — `build_default_priors()` and `build_log_space_priors()` both read from the registry; verify the new parameter appears with correct prior type
7. `optimization/cmc/reparameterization.py` — if the parameter needs Z-space transform, add a bijector entry
8. `core/parameter_space.py` — update `_DEFAULT_PRIOR_SPECS` to match the registry (`prior_mean`/`prior_std`)
9. Tests — add a test verifying the parameter appears in `DEFAULT_REGISTRY` and `fit_nlsq_jax()` runs without error

> **Critical:** The dual prior system (`parameter_registry.py` + `parameter_space.py:_DEFAULT_PRIOR_SPECS`) must stay in sync. The registry feeds `cmc/priors.py`; `_DEFAULT_PRIOR_SPECS` feeds `ParameterSpace._default_prior()`.

### How to debug NUTS divergences

1. Check `result.metadata["divergence_rate"]` — if > 5 %, sampling is unhealthy.
2. Increase `target_accept_prob` in `CMCConfig` (e.g., 0.8 → 0.9) — forces smaller step sizes.
3. Enable dense mass matrix: `CMCConfig(dense_mass=True)` — adapts to parameter correlations.
4. Toggle reparameterization: if `use_reparameterization=True`, try `False` — funnel geometry may be in a different space.
5. Check ArviZ BFMI: if BFMI < 0.3, the energy transition is degenerate — usually indicates prior-posterior conflict.
6. Run a single shard to isolate the problematic data region: `fit_cmc_jax(shard_data, config, nlsq_result=nlsq_result)`.

### How to run a single CMC shard manually

```python
from heterodyne.optimization.cmc import fit_cmc_jax
from heterodyne.optimization.cmc.config import CMCConfig

shard_data = full_data[:1000]  # slice to one shard's worth of time pairs
config = CMCConfig(num_samples=500, num_warmup=200)
result = fit_cmc_jax(shard_data, config, nlsq_result=nlsq_result)
print(result.convergence_status, result.metadata.get("divergence_rate"))
```

---

## Data Flow Summary

```
YAML -> ConfigManager -> XPCSDataLoader(HDF5) -> HeterodyneModel -> fit_nlsq_jax() -> fit_cmc_jax() -> Result(JSON+NPZ)
```
