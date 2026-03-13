# NLSQ Fitting Architecture

## Overview

The NLSQ subsystem provides CPU-optimized non-linear least squares fitting
of the heterodyne two-component c2 correlation model (14 physics parameters
+ 2 scaling parameters per angle). It is the primary warm-start stage of the
analysis pipeline, producing parameter estimates and uncertainties that
initialize the subsequent Bayesian (CMC/NUTS) sampling.

The subsystem is layered: global optimization methods (CMA-ES, multi-start)
sit above local trust-region fitting, which itself uses an adapter/wrapper
fallback chain with memory-aware strategy selection. A 4-layer
anti-degeneracy defense system addresses the structural parameter
correlations inherent to the 14-parameter model.

---

## Component Map

```
optimization/nlsq/
├── core.py                        # fit_nlsq_jax(), fit_nlsq_multi_phi() — main entry points
├── adapter.py                     # NLSQAdapter (JAX-traced primary), NLSQWrapper (scipy fallback)
├── adapter_base.py                # NLSQAdapterBase shared protocol
├── config.py                      # NLSQConfig, HybridRecoveryConfig, NLSQValidationConfig
├── results.py                     # NLSQResult dataclass
├── fourier_reparam.py             # FourierReparameterizer + FourierReparamConfig
├── cmaes_wrapper.py               # CMA-ES global optimization with NLSQ warm-start
├── multistart.py                  # Multi-start with Latin Hypercube Sampling
├── fallback_chain.py              # OptimizationStrategy enum, automatic strategy degradation
├── recovery.py                    # 3-attempt error recovery with diagnostics
├── anti_degeneracy_controller.py  # Degeneracy detection (correlation, bound saturation, plateau)
├── hierarchical.py                # Two-stage physics/scaling optimization
├── adaptive_regularization.py     # CV-based regularization
├── gradient_monitor.py            # Real-time gradient norm tracking
├── jacobian.py                    # Jacobian condition number analysis
├── memory.py                      # NLSQStrategy enum, memory-aware strategy selection
├── data_prep.py                   # Data preparation and weight computation
├── parameter_utils.py             # Parameter index/name utilities
├── parameter_index_mapper.py      # Varying/fixed parameter index mapping
├── parallel_accumulator.py        # Parallel residual accumulation
├── transforms.py                  # Parameter scaling/centering
├── progress.py                    # Progress reporting
├── result_builder.py              # NLSQResult factory (build_result_from_nlsq, build_failed_result)
├── fit_computation.py             # Fit computation helpers
├── strategies/
│   ├── base.py                    # FittingStrategy ABC
│   ├── stratified_ls.py           # Stratified angle-aware least squares
│   ├── hybrid_streaming.py        # Streaming gradient accumulation (100M+ points)
│   ├── out_of_core.py             # Disk-based JTJ accumulation
│   ├── sequential.py              # Per-angle sequential fitting
│   ├── jit_strategy.py            # JAX JIT residual (small-medium problems)
│   ├── chunked.py                 # Chunked evaluation
│   ├── residual.py                # Residual computation
│   ├── residual_jit.py            # JIT-compiled residual variants
│   └── executors.py               # Strategy executors
└── validation/
    ├── input_validator.py         # Pre-fit validation (NaN, bounds, shape)
    ├── convergence.py             # Convergence assessment
    ├── fit_quality.py             # Chi2, bounds proximity, quality classification
    ├── bounds.py                  # Bounds validation
    ├── result_validator.py        # Post-fit validation
    └── result.py                  # ValidationReport, ValidationIssue
```

---

## Execution Flow

### Single-Angle: `fit_nlsq_jax()`

```
fit_nlsq_jax(model, c2_data, phi_angle, config)
        │
        ├─ 1. Global optimization check (if not skipped)
        │     ├─ CMA-ES enabled?  → _fit_cmaes()  [3-phase]
        │     └─ Multi-start enabled?  → _fit_multistart()  [LHS sampling]
        │
        └─ 2. Local optimization: _fit_local()
              │
              ├─ Memory-aware strategy check (select_nlsq_strategy)
              │     Warns if peak memory exceeds threshold
              │
              ├─ NLSQAdapter.fit_jax()  [JAX-traced, primary]
              │     Uses nlsq.CurveFit with LRU model cache (max 64)
              │     Automatic memory-tier routing
              │
              ├─ On failure → NLSQWrapper.fit()  [scipy.optimize.least_squares fallback]
              │     Progressive recovery via HybridRecoveryConfig
              │
              └─ Post-fit: compute fitted correlation, update model
                    Returns NLSQResult
```

### Multi-Angle: `fit_nlsq_multi_phi()`

```
fit_nlsq_multi_phi(model, c2_data, phi_angles, config)
        │
        ├─ Determine fitting mode from config.per_angle_mode
        │
        ├─ Joint mode ("fourier", "independent", "auto" with >1 angle)
        │     └─ _fit_joint_multi_phi()
        │           Parameter vector: [physics_varying | fourier_coeffs]
        │           Single optimization across all angles simultaneously
        │           FourierReparameterizer converts coefficients → per-angle scaling
        │           Returns list[NLSQResult], one per angle
        │
        └─ Sequential mode (single angle or fallback)
              Per-angle warm-start chain: each angle initializes from previous
              Returns list[NLSQResult], one per angle
```

---

## Backend Adapters

`NLSQAdapter` and `NLSQWrapper` are both defined in `adapter.py` and
implement the `NLSQAdapterBase` protocol from `adapter_base.py`.

| Class | Backend | Method | Notes |
|---|---|---|---|
| `NLSQAdapter` | `nlsq` library (JAX-traced) | `fit_jax()` | Primary; CurveFit with LRU cache (max 64 instances) |
| `NLSQWrapper` | `scipy.optimize.least_squares` | `fit()` | Fallback; NumPy residuals, progressive recovery |

The adapter is tried first. On failure (ValueError, RuntimeError, TypeError,
ImportError, OSError), the wrapper provides automatic retry with progressive
recovery controlled by `HybridRecoveryConfig`.

### NLSQAdapter Model Cache

`NLSQAdapter` caches compiled `nlsq.CurveFit` instances keyed by
`(n_data, n_params, phi_angles, scaling_mode)`. This avoids re-JIT-compiling
for identical problem shapes. The cache is bounded at 64 entries with LRU
eviction.

---

## CMA-ES Global Optimization

When `config.enable_cmaes = True`, `fit_nlsq_jax()` delegates to a 3-phase
CMA-ES pipeline:

```
Phase 1: NLSQ warm-start
    Run local trust-region fit to get a warm-start point.
    If it fails, CMA-ES proceeds from raw initial parameters.

Phase 2: CMA-ES global search
    Uses evosax JAX-accelerated backend via CMAESConfig:
    - sigma0 (initial step size), popsize, maxiter, tolx, tolfun
    - Optional diagonal_filtering and anti_degeneracy penalty

Phase 3: Comparison
    Compare NLSQ vs CMA-ES results by final cost.
    Keep the lower-cost result.
    Classify fit quality (good/marginal/poor) via classify_fit_quality().
```

### CMA-ES Configuration

| Field | Default | Description |
|---|---|---|
| `enable_cmaes` | `False` | Enable CMA-ES global search |
| `cmaes_sigma0` | 0.3 | Initial step size |
| `cmaes_max_iterations` | 1000 | Maximum CMA-ES generations |
| `cmaes_population_size` | `None` (auto) | Population size |
| `cmaes_tolx` | 1e-6 | Parameter convergence tolerance |
| `cmaes_tolfun` | 1e-8 | Cost function convergence tolerance |
| `cmaes_diagonal_filtering` | `"none"` | `"none"` or `"remove"` |
| `cmaes_anti_degeneracy` | `False` | Apply anti-degeneracy penalty |

---

## Multi-Start Optimization

When `config.multistart = True`, `fit_nlsq_jax()` delegates to
`MultiStartOptimizer`:

- Generates `multistart_n` (default 10) starting points via Latin Hypercube
  Sampling (LHS), Sobol, or random sampling (`config.sampling_strategy`).
- Runs parallel local fits from each starting point.
- Screens candidates by cost (`screen_keep_fraction`), refines top-k
  (`refine_top_k`), selects the best result.

---

## Fourier Reparameterization

Replaces per-angle independent contrast/offset values with truncated Fourier
series to reduce structural degeneracy in joint multi-angle fits.

### Mathematical Formulation

```
contrast(phi) = c0 + sum_k [ck * cos(k * phi) + sk * sin(k * phi)]   k = 1..order
offset(phi)   = o0 + sum_k [ok * cos(k * phi) + tk * sin(k * phi)]   k = 1..order
```

### Parameter Count Reduction (order = 2)

| n_phi | Independent | Fourier | Reduction |
|-------|-------------|---------|-----------|
| 2     | 4           | 4       | 0%        |
| 3     | 6           | 6       | 0%        |
| 10    | 20          | 10      | 50%       |
| 23    | 46          | 10      | 78%       |
| 100   | 200         | 10      | 95%       |

For `n_phi <= 2*(order+1)`, independent mode is used automatically.

### Configuration

| Field | Default | Description |
|---|---|---|
| `per_angle_mode` | `"auto"` | `"independent"`, `"fourier"`, or `"auto"` |
| `fourier_order` | 2 | Number of Fourier harmonics |
| `fourier_auto_threshold` | 6 | Use Fourier when n_phi > threshold in auto mode |

---

## 4-Layer Anti-Degeneracy Defense

The heterodyne 14-parameter model has known structural degeneracies
(D0_ref/D0_sample correlation, alpha/D0 compensation, v0/v_offset trading).
Four defense layers address these:

| Layer | Module | Mechanism |
|---|---|---|
| 1 | `fourier_reparam.py` | Fourier/constant reparameterization reduces per-angle parameter count |
| 2 | `hierarchical.py` | Two-stage optimization: physics params first, then scaling |
| 3 | `adaptive_regularization.py` | CV-based regularization penalizes cross-group variance |
| 4 | `gradient_monitor.py` | Real-time gradient collapse detection with consecutive-trigger thresholds |

The `anti_degeneracy_controller.py` module provides post-fit degeneracy
diagnostics: correlation degeneracy (|r| > threshold), bound saturation
(parameters at bounds), and cost-function plateau detection.

---

## Recovery: 3-Attempt Error Recovery

`recovery.py` provides attempt-level retry within a single strategy,
complementing the strategy-level fallback chain. Error diagnosis categorizes
failures (OOM, convergence, bounds, ill-conditioned, NaN) and selects
recovery actions.

The three attempts:

1. **Original parameters** -- unperturbed initial values.
2. **Perturbed parameters** -- Gaussian perturbation scaled by
   `perturb_scale` (default 10%) of parameter range.
3. **Relaxed convergence** -- loosened tolerance thresholds.

`HybridRecoveryConfig` controls progressive scaling per retry attempt *k*:

| Setting | Scale per attempt | Default |
|---|---|---|
| Learning rate | `lr_decay ** k` | 0.5 |
| Regularization | `lambda_growth ** k` | 10.0 |
| Trust radius | `trust_decay ** k` | 0.5 |

---

## Strategy Selection (Memory-Aware)

`memory.py` estimates peak memory from Jacobian dimensions and routes to the
appropriate strategy:

```
Decision tree:
    Index array alone > threshold  →  STREAMING  (extreme scale, 100M+ points)
    Peak Jacobian memory > threshold  →  LARGE  (chunked JTJ accumulation)
    Otherwise  →  STANDARD  (full in-memory Jacobian, fastest)
```

Memory threshold = `nlsq_memory_fraction` (default 0.75) of system RAM,
with `nlsq_memory_fallback_gb` (default 16 GB) when detection fails.

`fallback_chain.py` provides automatic strategy degradation: if the selected
strategy fails, the chain tries strategies in descending robustness order
(STREAMING > LARGE > STANDARD) until one succeeds.

---

## Fitting Strategies

All strategies implement `FittingStrategy` (ABC in `strategies/base.py`):

| Strategy | Module | Use Case |
|---|---|---|
| Stratified LS | `stratified_ls.py` | Angle-aware least squares with anti-degeneracy |
| Hybrid Streaming | `hybrid_streaming.py` | L-BFGS warmup + streaming Gauss-Newton for 100M+ points |
| Out-of-Core | `out_of_core.py` | Disk-based JTJ accumulation for memory-limited systems |
| Sequential | `sequential.py` | Per-angle sequential fitting with warm-starting |
| JIT | `jit_strategy.py` | JAX JIT-compiled residual for small-medium problems |
| Chunked | `chunked.py` | Chunked residual evaluation |

---

## NLSQResult Dataclass

Key fields returned from every fit:

| Field | Type | Description |
|---|---|---|
| `parameters` | `np.ndarray` | Fitted parameter values |
| `parameter_names` | `list[str]` | Names in canonical order |
| `success` | `bool` | Whether optimizer converged |
| `message` | `str` | Optimizer status message |
| `uncertainties` | `np.ndarray \| None` | 1-sigma from covariance diagonal |
| `covariance` | `np.ndarray \| None` | Full parameter covariance matrix |
| `final_cost` | `float \| None` | Residual sum of squares at solution |
| `reduced_chi_squared` | `float \| None` | chi-squared / degrees of freedom |
| `n_iterations` | `int` | Number of optimizer iterations |
| `n_function_evals` | `int` | Number of function evaluations |
| `convergence_reason` | `str` | Why the optimizer stopped |
| `residuals` | `np.ndarray \| None` | Residual vector at solution |
| `jacobian` | `np.ndarray \| None` | Jacobian at solution |
| `fitted_correlation` | `np.ndarray \| None` | Model correlation at fitted params |
| `wall_time_seconds` | `float \| None` | Total wall-clock time |
| `metadata` | `dict[str, Any]` | Additional diagnostics (optimizer, fallback info, etc.) |

Helper methods: `params_dict`, `get_param(name)`, `get_uncertainty(name)`,
`get_correlation_matrix()`, `validate()`, `summary()`.

---

## Validation

All validators produce a `ValidationReport` containing `ValidationIssue`
instances with one of three severity levels:

| Severity | Effect | Examples |
|---|---|---|
| `ERROR` | Sets `is_valid = False` | NaN in data, inverted bounds, optimization failed |
| `WARNING` | Logged but does not block | Reduced chi-squared > threshold, large uncertainty, near-bound solution |
| `INFO` | Logged as informational | Good fit quality |

### Pre-Fit (`input_validator.py`)

Checks: empty data, NaN/Inf values, bounds shape, inverted bounds, initial
parameters out of bounds.

### Post-Fit (`result_validator.py`, `fit_quality.py`, `convergence.py`)

Checks: convergence flag, chi-squared thresholds, relative uncertainty,
parameter correlations, bounds proximity. `classify_fit_quality()` returns
good/marginal/poor classification.

---

## Configuration

### NLSQConfig (Master)

Core solver fields:

| Field | Default | Description |
|---|---|---|
| `max_iterations` | 1000 | Maximum optimizer iterations |
| `tolerance` | 1e-8 | Convergence tolerance |
| `method` | `"trf"` | Trust-region algorithm (`"trf"`, `"lm"`, `"dogbox"`) |
| `loss` | `"soft_l1"` | Robust loss kernel (`"linear"`, `"soft_l1"`, `"huber"`, `"cauchy"`, `"arctan"`) |
| `ftol` | 1e-8 | Relative cost function tolerance |
| `xtol` | 1e-8 | Relative parameter step tolerance |
| `gtol` | 1e-8 | Absolute projected gradient tolerance |
| `use_jac` | `True` | Supply analytic Jacobian |
| `x_scale` | `"jac"` | Parameter scaling (`"jac"` or explicit list) |
| `verbose` | 1 | Solver verbosity (0=silent, 1=summary, 2=detailed) |

Workflow and goal presets:

| Field | Default | Values |
|---|---|---|
| `workflow` | `"auto"` | `"auto"`, `"auto_global"`, `"hpc"` |
| `goal` | `"robust"` | `"fast"`, `"robust"`, `"quality"`, `"memory_efficient"` |
| `analysis_mode` | `"two_component"` | `"static_ref"`, `"static_both"`, `"two_component"` |

Multi-start and streaming:

| Field | Default | Description |
|---|---|---|
| `multistart` | `False` | Enable multi-start optimization |
| `multistart_n` | 10 | Number of random starts |
| `sampling_strategy` | `"lhs"` | `"lhs"`, `"sobol"`, `"random"` |
| `screen_keep_fraction` | 0.5 | Fraction of starts to keep after screening |
| `refine_top_k` | 3 | Number of top candidates to refine |
| `enable_streaming` | `False` | Enable streaming gradient accumulation |
| `streaming_chunk_size` | 50000 | Points per streaming chunk |
| `enable_stratified` | `False` | Enable stratified sampling |
| `target_chunk_size` | 10000 | Points per stratified chunk |

Recovery, diagnostics, and anti-degeneracy:

| Field | Default | Description |
|---|---|---|
| `enable_recovery` | `True` | Enable automatic retry on failure |
| `max_recovery_attempts` | 3 | Maximum recovery retries |
| `enable_diagnostics` | `True` | Emit convergence/quality diagnostics |
| `enable_anti_degeneracy` | `True` | Apply anti-degeneracy constraints |

Hierarchical optimization:

| Field | Default | Description |
|---|---|---|
| `enable_hierarchical` | `False` | Enable two-stage physics/scaling optimization |
| `hierarchical_max_outer_iterations` | 20 | Max outer loop iterations |
| `hierarchical_inner_tolerance` | 1e-6 | Inner optimization tolerance |
| `hierarchical_outer_tolerance` | 1e-4 | Outer convergence tolerance |

Adaptive regularization:

| Field | Default | Description |
|---|---|---|
| `regularization_mode` | `"none"` | `"none"`, `"tikhonov"`, `"adaptive"` |
| `group_variance_lambda` | 0.01 | Group variance penalty weight |
| `regularization_target_cv` | 0.5 | Target coefficient of variation |

Gradient monitoring:

| Field | Default | Description |
|---|---|---|
| `enable_gradient_monitoring` | `False` | Enable gradient collapse detection |
| `gradient_ratio_threshold` | 100.0 | Gradient norm ratio trigger |
| `gradient_consecutive_triggers` | 3 | Consecutive trigger count before action |

Hybrid streaming optimizer:

| Field | Default | Description |
|---|---|---|
| `hybrid_enable` | `False` | Enable hybrid streaming optimizer |
| `hybrid_method` | `"gauss_newton"` | `"lbfgs"` or `"gauss_newton"` |
| `hybrid_warmup_fraction` | 0.1 | Fraction of data for warmup phase |
| `hybrid_max_phases` | 4 | Maximum number of streaming phases |

NLSQ package integration:

| Field | Default | Description |
|---|---|---|
| `use_nlsq_library` | `True` | Prefer nlsq JAX library over scipy |
| `nlsq_stability` | `"auto"` | `"auto"`, `"check"`, `"off"` |
| `nlsq_memory_fraction` | 0.75 | Fraction of RAM for NLSQ |
| `nlsq_memory_fallback_gb` | 16.0 | Fallback threshold if detection fails |
| `n_params` | 14 | Number of model parameters |

### HybridRecoveryConfig

| Field | Default | Description |
|---|---|---|
| `max_retries` | 3 | Maximum recovery attempts |
| `lr_decay` | 0.5 | Learning rate decay per retry |
| `lambda_growth` | 10.0 | Regularization growth per retry |
| `trust_decay` | 0.5 | Trust radius decay per retry |
| `perturb_scale` | 0.1 | Parameter perturbation scale (fraction of range) |

### NLSQValidationConfig

| Field | Default | Description |
|---|---|---|
| `chi2_warn_low` | 0.5 | chi-squared reduced below this triggers warning (overfitting) |
| `chi2_warn_high` | 2.0 | chi-squared reduced above this triggers warning |
| `chi2_fail_high` | 10.0 | chi-squared reduced above this triggers error |
| `max_relative_uncertainty` | 1.0 | Relative uncertainty above 100% triggers warning |
| `correlation_warn` | 0.95 | Correlation coefficient magnitude above this triggers warning |
