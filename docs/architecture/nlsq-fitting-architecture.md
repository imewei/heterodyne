<!-- Package: heterodyne | Last verified: 2026-05-08 -->

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

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Setup Phase](#1-setup-phase)
3. [Component Map](#component-map)
4. [Execution Flow](#execution-flow)
5. [Backend Adapters](#backend-adapters)
6. [CMA-ES Global Optimization](#cma-es-global-optimization)
7. [Multi-Start Optimization](#multi-start-optimization)
8. [Fourier Reparameterization](#fourier-reparameterization)
9. [4-Layer Anti-Degeneracy Defense](#4-layer-anti-degeneracy-defense)
10. [Recovery: 3-Attempt Error Recovery](#recovery-3-attempt-error-recovery)
11. [Stratification Decision](#stratification-decision)
12. [Residual Function Setup](#residual-function-setup)
13. [Strategy Selection (Memory-Aware)](#strategy-selection-memory-aware)
14. [Fitting Strategies](#fitting-strategies)
15. [NLSQResult Dataclass](#nlsqresult-dataclass)
16. [Result Building](#result-building)
17. [Validation](#validation)
18. [Configuration](#configuration)
19. [NLSQ as CMC Warm-Start Provider](#nlsq-as-cmc-warm-start-provider)
20. [Quick Reference Tables](#quick-reference-tables)
21. [Key Files Reference](#key-files-reference)

---

## High-Level Architecture

```
fit_nlsq_jax() / fit_nlsq_multi_phi()
        │
        ├─ 1. Setup & Input Validation
        ├─ 2. Global Optimization (CMA-ES / Multi-start)  [optional]
        ├─ 3. Adapter Selection (NLSQAdapter → NLSQWrapper fallback)
        ├─ 4. Memory & Strategy Selection
        ├─ 5. Stratification Decision
        ├─ 6. Residual Function Setup
        ├─ 7. 4-Layer Anti-Degeneracy
        ├─ 8. Strategy Execution
        ├─ 9. 3-Attempt Recovery
        └─ 10. Result Building

        ╔══════════════════════════════════════════╗
        ║  NLSQ ALSO SERVES AS CMC WARM-START      ║
        ║  (--method both)  → see §19              ║
        ╚══════════════════════════════════════════╝
```

---

## 1. Setup Phase

`fit_nlsq_jax()` (`optimization/nlsq/core.py`) and `fit_nlsq_multi_phi()`
share the same setup sequence:

1. **Input validation** — `validation/input_validator.py` checks for empty
   data, NaN/Inf values, bounds shape consistency, inverted bounds, and
   initial parameters that lie outside the configured bounds.
2. **Initial values & bounds** — `param_manager.get_initial_values()` and
   `param_manager.get_bounds()` return varying-only arrays; values are
   then `np.clip(initial, lower, upper)`-clamped before optimization.
3. **JAX-side constants** — `t`, `q`, `dt` from the model and the full
   parameter array `fixed_values_jax = jnp.asarray(pm.get_full_values())`
   are pre-converted to JAX device arrays once. `varying_indices_jax` is
   captured as `jnp.int32` for the per-call `at[].set()` scatter.
4. **Weight handling** — `weights_jax` is `jnp.asarray(weights)` when
   supplied; shape must match `c2_jax.shape` or a `ValueError` is raised
   in `_fit_local()`.

### Parameter Vector Layout

The full canonical parameter array always has 14 entries in
`config/parameter_names.py:ALL_PARAM_NAMES` order. `ParameterIndexMapper`
(in `optimization/nlsq/parameter_index_mapper.py`) bridges three
representations:

| Space | What it contains | Built from |
|---|---|---|
| **Full** | All 14 physics parameters in canonical order | `pm.get_full_values()` |
| **Varying** | Only parameters with `vary=True` | `pm.varying_names`, `pm.varying_indices` |
| **Optimizer** | Varying params, optionally log-transformed | `log_mask` from `DEFAULT_REGISTRY[name].log_space` |

Index conversion methods: `full_to_varying(i)`, `varying_to_full(j)`,
`get_name(j)`, `name_to_varying(name)`, `is_log_transformed(j)`. Reverse
lookups are O(1) via cached dicts built in `__init__`.

### Time Axis Handling

`model.t` is a 1-D array of frame times (seconds). For multi-angle joint
fits, `_fit_joint_constant_multi_phi()` builds the meshgrid explicitly:

```python
t1_mesh, t2_mesh = np.meshgrid(np.asarray(t), np.asarray(t), indexing="ij")
```

The `_fit_joint_multi_phi()` and single-angle paths delegate the meshgrid
construction to the JAX backend (`compute_residuals` / `compute_multi_angle_residuals`).

### Analysis Mode Selection

`NLSQConfig.analysis_mode` selects which subset of physics parameters
varies. The total optimizer-vector length is `n_physics_varying + 2*n_phi`
in independent scaling mode (or `n_physics_varying + 2` in
constant-averaged mode):

| Mode | Class hint | Physics varying | Scaling | Total per n_phi=1 |
|---|---|---|---|---|
| `static_ref` | reduced model | 3 | 2 | 5 |
| `static_both` | reduced model | 6 | 2 | 8 |
| `two_component` | full two-component model | 14 | 2 | 16 |

Validated against `_VALID_ANALYSIS_MODES = {"static_ref", "static_both", "two_component"}`
in `config.py`.

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
├── result_builder.py              # NLSQResult factory + TimedContext
├── fit_computation.py             # compute_c2_batch(), compute_theoretical_fits(), etc.
├── strategies/
│   ├── base.py                    # FittingStrategy ABC, StrategyResult
│   ├── stratified_ls.py           # StratifiedLSStrategy
│   ├── hybrid_streaming.py        # HybridStreamingStrategy (4-phase)
│   ├── out_of_core.py             # OutOfCoreStrategy
│   ├── sequential.py              # Per-angle sequential fitting
│   ├── jit_strategy.py            # JITStrategy (LRU-cached)
│   ├── chunked.py                 # ChunkedStrategy
│   ├── residual.py                # ResidualStrategy
│   ├── residual_jit.py            # ResidualJITStrategy
│   └── executors.py               # Strategy executors
└── validation/
    ├── input_validator.py         # Pre-fit validation (NaN, bounds, shape)
    ├── convergence.py             # Convergence assessment
    ├── fit_quality.py             # classify_fit_quality(), FitQualityValidator
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
        ├─ 1. Global optimization check (if not _skip_global_selection)
        │     ├─ enable_cmaes? → _fit_cmaes()  [3-phase: NLSQ→CMA-ES→compare]
        │     └─ multistart?    → _fit_multistart()  [LHS sampling]
        │
        └─ 2. Local optimization: _fit_local()
              │
              ├─ Memory-aware strategy check (select_nlsq_strategy)
              │     Warns if peak memory exceeds threshold
              │
              ├─ NLSQAdapter.fit_jax()  [JAX-traced, primary]
              │     Uses nlsq.CurveFit with LRU model cache (max 64)
              │
              ├─ On failure → NLSQWrapper.fit()  [scipy.optimize.least_squares]
              │     Progressive recovery via HybridRecoveryConfig
              │
              └─ Post-fit: compute fitted correlation, σ²-corrected chi²
                    Returns NLSQResult
```

### Multi-Angle: `fit_nlsq_multi_phi()`

```
fit_nlsq_multi_phi(model, c2_data, phi_angles, config)
        │
        ├─ enable_cmaes? → _fit_joint_cmaes_multi_phi()
        │
        ├─ Constant-averaged mode (per_angle_mode == "constant", or
        │   "auto" with n_phi >= constant_scaling_threshold)
        │     └─ _fit_joint_constant_multi_phi()
        │           Per-angle quantile estimates → averaged contrast/offset
        │           Optimizer vector: [physics_varying | contrast | offset]
        │
        ├─ Joint Fourier mode ("fourier"/"independent"/"auto" with multi-angle)
        │     └─ _fit_joint_multi_phi()
        │           Optimizer vector: [physics_varying | fourier_coeffs]
        │           FourierReparameterizer.fourier_to_per_angle() at each call
        │
        └─ Sequential mode (single angle or fallback)
              Per-angle warm-start chain: each angle initializes from previous
```

All paths return `list[NLSQResult]`, one per angle.

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
CMA-ES pipeline (`_fit_cmaes()` in `core.py`):

```
Phase 1: NLSQ warm-start
    Run local trust-region fit to get a warm-start point.
    If it fails or `cmaes_warmstart_auto_skip` triggers,
    CMA-ES proceeds (or is skipped) accordingly.

Phase 2: CMA-ES global search
    Uses cma.fmin2 via CMAESWrapper:
    - sigma0 (initial step size), popsize, maxiter, tolx, tolfun
    - Optional diagonal_filtering ("none"/"remove") and
      anti_degeneracy penalty wrapping (build_anti_degeneracy_objective)
    - Adaptive popsize/maxiter via compute_adaptive_cmaes_params()
      when popsize is None or maxiter is the default

Phase 3: Comparison
    Compare NLSQ vs CMA-ES results by final cost.
    Keep the lower-cost result.
    Classify fit quality (good/marginal/poor) via classify_fit_quality().
```

For multi-angle joint runs, `_fit_joint_cmaes_multi_phi()` follows the same
structure but auto-skips CMA-ES when `cmaes_warmstart_auto_skip` is `True`
and the warm-start reduced χ² is below `cmaes_warmstart_skip_threshold`
(default 5.0).

### CMA-ES Configuration

Verified against `cmaes_wrapper.py:CMAESConfig` and `config.py:NLSQConfig`:

| Field | Default | Description |
|---|---|---|
| `enable_cmaes` | `False` | Enable CMA-ES global search |
| `cmaes_sigma0` | 0.3 | Initial step size |
| `cmaes_max_iterations` | 1000 | Maximum CMA-ES generations |
| `cmaes_population_size` | `None` (auto) | Population size; `None` triggers `compute_adaptive_cmaes_params()` |
| `cmaes_tolx` | 1e-6 | Parameter convergence tolerance |
| `cmaes_tolfun` | 1e-8 | Cost function convergence tolerance |
| `cmaes_diagonal_filtering` | `"remove"` | `"none"` or `"remove"` |
| `cmaes_anti_degeneracy` | `False` | Wrap objective with degeneracy penalty |
| `cmaes_warmstart_auto_skip` | `True` | Skip CMA-ES when NLSQ warm-start is already good |
| `cmaes_warmstart_skip_threshold` | 5.0 | Reduced-χ² ceiling that triggers skip |

**Note on `cmaes_preset`:** The string preset (e.g. `"cmaes-global"`)
referenced in user-facing CLI/YAML docs is a config-loader convenience,
not a field on `NLSQConfig`; presets are expanded into the explicit
`cmaes_*` fields above before reaching `_fit_cmaes()`.

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

### Problem Statement

The heterodyne 14-parameter model exhibits known structural degeneracies:

- **D0_ref / D0_sample correlation** — both describe diffusion, often
  correlated across temperature and concentration; the residual landscape
  has a long, narrow valley along the D0_ref ≈ D0_sample line.
- **alpha / D0 compensation** — the product `D0 * t^alpha` is
  approximately constant at the characteristic time `t* = exp(1/alpha)`,
  yielding a banana-shaped posterior in (alpha, D0) space.
- **v0 / v_offset trading** — at constant velocity (`beta = 0`), the
  model can express the same flow as either `v0` or `v_offset`, with
  flat sensitivity along the trade-off direction.
- **Explosion with phi angles** — independent per-angle scaling adds
  `2*n_phi` parameters; for 23 angles the optimizer faces 14 + 46 = 60
  parameters. Gradient cancellation across angles (parameters that pull
  in opposite directions for different φ) accelerates degeneracy growth.

The four defense layers attack these correlations from different angles:

| Layer | Module | Activation | Mechanism | What it prevents |
|---|---|---|---|---|
| 1 | `fourier_reparam.py` | Joint multi-angle (`per_angle_mode != "constant"`) | Truncated Fourier basis collapses `2*n_phi` per-angle parameters into `2*(2*order+1)` coefficients | Parameter explosion at large `n_phi`; enforces smooth angular variation |
| 2 | `hierarchical.py` | `enable_hierarchical = True` | Two-stage outer/inner loop: physics first (fixed scaling), then scaling (fixed physics), repeat until both converge | Cross-group cancellation between physics and scaling gradients |
| 3 | `adaptive_regularization.py` | `regularization_mode in {"tikhonov","adaptive"}` | CV-based λ that penalises group-variance; λ grows when group CV exceeds `regularization_target_cv` | Flat-direction drift in (alpha, D0) and similar pairs |
| 4 | `gradient_monitor.py` | `enable_gradient_monitoring = True` | Real-time gradient-norm ratio tracker; triggers when ratio > `gradient_ratio_threshold` for `gradient_consecutive_triggers` consecutive iterations | Late-stage gradient collapse where one parameter group dominates |

Diagnostic post-pass: `anti_degeneracy_controller.py` reports correlation
degeneracy (|r| > threshold via `_KNOWN_DEGENERATE_PAIRS`), bound
saturation, and cost-function plateau detection.

> **Homodyne Layer 5 absent by design.** Homodyne's shear-sensitivity
> weighting (sinc decorrelation) penalizes parameters based on their
> sensitivity to shear flow direction. In heterodyne, velocity enters as
> a phase term `cos(q·cos(φ)·v_integral)` in the cross-term — not as
> amplitude decorrelation. Angular weighting by shear sensitivity is not
> physically applicable; Layer 5 was deliberately removed.

---

## Recovery: 3-Attempt Error Recovery

`recovery.py` provides attempt-level retry within a single strategy,
complementing the strategy-level fallback chain. Error diagnosis categorizes
failures (OOM, convergence, bounds, ill-conditioned, NaN) and selects
recovery actions.

The three attempts:

1. **Original parameters** — unperturbed initial values.
2. **Perturbed parameters** — Gaussian perturbation scaled by
   `perturb_scale` (default 10%) of parameter range.
3. **Relaxed convergence** — loosened tolerance thresholds.

`HybridRecoveryConfig` controls progressive scaling per retry attempt *k*:

| Setting | Scale per attempt | Default |
|---|---|---|
| Learning rate | `lr_decay ** k` | 0.5 |
| Regularization | `lambda_growth ** k` | 10.0 |
| Trust radius | `trust_decay ** k` | 0.5 |

---

## Stratification Decision

The `StratifiedLSStrategy` (`strategies/stratified_ls.py`) is selected when
`config.enable_stratified` is `True` and the dataset has non-uniform
information density across the q-point range.

### When stratification activates

- Direct: `enable_stratified = True` in `NLSQConfig`. The optimizer dispatches
  to `StratifiedLSStrategy.fit()`, which consumes `target_chunk_size`.
- The strategy ultimately delegates to `nlsq.curve_fit_large` with the
  user's bounds and method (with `dogbox` coerced to `trf`, `lm` coerced
  to `trf`).

### What happens during the fit

- A pure-JAX `residual_fn(varying)` is built that scatters varying values
  into the full 14-element parameter array via `at[].set()`, then calls
  `compute_residuals(full_params, t, q, dt, phi_angle, c2_jax, weights_jax)`.
- `curve_fit_large` accepts `f(xdata, *params) → predictions`; the wrapper
  passes `ydata = zeros` and returns the negated `residual_fn` so that
  `ydata - f(...) = residual_fn(...)`.
- Final covariance is estimated post-hoc from the returned Jacobian via
  `(JᵀJ)⁻¹ · s²` with `s² = ‖r‖² / (n − p)`.

### Decision conditions

| Condition | Strategy invoked |
|---|---|
| `enable_stratified=True` and `target_chunk_size` set | `StratifiedLSStrategy` |
| Hybrid streaming requested (`hybrid_enable=True`) | `HybridStreamingStrategy` |
| Memory threshold exceeded (`select_nlsq_strategy → LARGE/STREAMING`) | warning emitted; adapter still dispatches to its memory-tier path |
| Otherwise | direct adapter path (single-pass `compute_residuals`) |

---

## Residual Function Setup

The strategy layer offers two complementary residual evaluators:

### `ResidualStrategy` (`strategies/residual.py`)

- **Class:** `ResidualStrategy`
- **Tracing:** Python-traced — calls `compute_residuals` and (when
  `config.use_jac=True`) `compute_residuals_jacobian` once per scipy
  iteration.
- **Shapes:** dynamic — `n_data` and `n_params` are recomputed on each
  call; no padding.
- **Solver:** `nlsq.CurveFit(flength=n_data)` with explicit Jacobian.
- **Use case:** small datasets (< 10 k residuals), debugging, baseline
  comparisons.

### `ResidualJITStrategy` (`strategies/residual_jit.py`)

- **Class:** `ResidualJITStrategy`
- **Tracing:** JIT-compiled — `_jit_residuals(varying_jax)` is wrapped in
  `@jax.jit` and warmed up once before the optimizer loop.
- **Shapes:** padded static shapes via the JIT cache; finite-difference
  Jacobian on the scipy side (no analytic Jacobian).
- **vmap:** the multi-angle joint path uses the batched
  `compute_multi_angle_residuals` (jit + vmap) elsewhere; `ResidualJITStrategy`
  itself is single-angle.
- **Use case:** when analytic Jacobian compilation is slow or fails, or
  when fast iteration outweighs Jacobian accuracy.

### Computation flow (both strategies)

1. Cache constants: `t`, `q`, `dt`, `c2_jax`, `weights_jax`, `fixed_values`
   (full 14-element JAX array), `varying_idx`.
2. On each call, `at[varying_idx].set(varying)` reconstructs the full
   parameter vector (immutable JAX scatter).
3. Call `compute_residuals(full_params, t, q, dt, phi_angle, c2_jax,
   weights_jax)` — returns the flattened residual vector.
4. Convert back to NumPy at the boundary so scipy/nlsq can consume it.

### Parameter vector layout (single-angle)

```
[ varying physics params... ]    # length = pm.n_varying (≤ 14)
```

### Parameter vector layout (multi-angle joint)

| Mode | Layout |
|---|---|
| Constant-averaged | `[ varying physics... | contrast | offset ]` |
| Independent | `[ varying physics... | c_0..c_{n_phi-1} | o_0..o_{n_phi-1} ]` |
| Fourier (order *K*) | `[ varying physics... | fourier_contrast(2K+1) | fourier_offset(2K+1) ]` |

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

### Peak memory formula

`memory.py:estimate_peak_memory_gb` returns:

```
peak_memory_gib = n_points × n_params × bytes_per_element × _JACOBIAN_OVERHEAD / 1024³
                = n_points × n_params × 8 × 6.5 / 1024³
```

The overhead factor `_JACOBIAN_OVERHEAD = 6.5` is documented in
`memory.py` as covering "base Jacobian + autodiff intermediates + JIT +
workspace":

| Component | Approx. share |
|---|---|
| Base Jacobian (n_points × n_params × float64) | 1.0× |
| JAX autodiff intermediates (vjp / jvp tapes) | ~3.0× |
| JIT compilation buffer + XLA workspace | ~1.5× |
| Optimizer scratch (JᵀJ, Cholesky, etc.) | ~1.0× |

### Worked example

5-angle run with `n_points = 5,000,000` (e.g., five 1000² C₂ matrices) and
`n_params = 24` (14 physics + 2×5 scaling):

```
peak ≈ 5e6 × 24 × 8 × 6.5 / 1024³  ≈  5.81 GiB
```

On a 16 GiB workstation with default `nlsq_memory_fraction = 0.75`, this
falls under the 12.0 GiB threshold → STANDARD strategy. Doubling the
matrix to 1414² per angle pushes peak past 11.6 GiB; STANDARD still fits
but a third angle would force LARGE.

### Environment override

`memory.py:MEMORY_FRACTION_ENV_VAR = "HETERODYNE_MEMORY_FRACTION"` —
setting `HETERODYNE_MEMORY_FRACTION=0.5` clamps the threshold to 50% of
detected RAM. The value is clamped to `[0.1, 0.9]`.

### Detection priority chain

1. `psutil.virtual_memory().total` — preferred, cross-platform.
2. `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")` — Linux/Unix.
3. Fallback: `FALLBACK_THRESHOLD_GB = 16.0` GiB.

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
| Residual | `residual.py` | Direct evaluation, dynamic shapes |
| Residual JIT | `residual_jit.py` | JIT residual + finite-difference Jacobian |

### Stratified LS

- **Memory pattern:** single full Jacobian; `nlsq.curve_fit_large` handles
  internal chunking when needed.
- **Selection:** `enable_stratified=True`, or memory tier LARGE.
- **Pseudocode:**

```python
fixed_values = jnp.asarray(pm.get_full_values())
def residual_fn(varying):
    full = fixed_values.at[varying_idx].set(varying)
    return compute_residuals(full, t, q, dt, phi_angle, c2_jax, weights_jax)
result = curve_fit_large(f=-residual_fn, xdata, ydata=zeros, p0=initial,
                         bounds=(lower, upper), method=method)
```

### Hybrid Streaming

4-phase pipeline from `hybrid_streaming.py`:

1. **Phase 1 — Normalization:** `_param_scales = (upper - lower)` when
   `hybrid_normalization=True`, else ones; transforms parameters to
   comparable magnitudes for better optimizer conditioning.
2. **Phase 2 — L-BFGS warmup:** `warmup_iterations = max_iterations *
   hybrid_warmup_fraction`. Provides fast global progress.
3. **Phase 3 — Gauss-Newton refinement (chunk accumulation of JᵀJ and
   Jᵀr):** `gauss_newton_max_iterations = max_iterations - warmup_iterations`,
   chunk size from `streaming_chunk_size` (default 50 000). All inside
   `nlsq.AdaptiveHybridStreamingOptimizer` (single call).
4. **Phase 4 — Denormalization + covariance:** parameters returned in
   original space; covariance assembled from accumulated JᵀJ.

Falls back to `nlsq.curve_fit_large` if `AdaptiveHybridStreamingOptimizer`
is unavailable.

### Out-of-Core

- **Memory pattern:** memory-mapped `c2_data`; chunk-wise Jacobian rows
  accumulated into JᵀJ on disk.
- **Selection:** memory tier STREAMING or explicit user request via the
  strategy.
- **Pseudocode:**

```python
chunk_size = self._chunk_size or config.chunk_size or self._auto_chunk_size(...)
n_chunks = ceil(n_data / chunk_size)
# Build residual_fn(varying) closing over c2_jax, weights_jax
nlsq_result = curve_fit_large(f=-residual_fn, xdata, ydata=zeros,
                              p0=initial, bounds=(lower, upper),
                              method=method)
# Recompute residuals & estimate covariance from final Jacobian
```

The module also exposes utilities for parallel chunk evaluation
(`accumulate_chunks_parallel` / `accumulate_chunks_sequential` patterns)
through `parallel_accumulator.py`.

### JIT Strategy

- **Memory pattern:** full Jacobian held in device memory; LRU cache keeps
  compiled XLA programs warm across calls.
- **Cache key:** `(n_data, n_params, phi_angles, scaling_mode)` — same key
  used by `NLSQAdapter`'s model cache.
- **Selection:** memory tier STANDARD; default for small-medium problems.

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

## Result Building

`result_builder.py` centralizes `NLSQResult` construction so every strategy
emits a consistent payload with covariance, uncertainties, reduced χ², and
metadata.

| Factory | Input | When used |
|---|---|---|
| `build_result_from_scipy(opt_result, parameter_names, n_data, ...)` | `scipy.optimize.OptimizeResult` | NLSQWrapper / scipy paths |
| `build_result_from_arrays(parameters, parameter_names, residuals, n_data, ...)` | Raw arrays | CMA-ES (`fit_with_cmaes`), non-scipy backends |
| `build_result_from_nlsq(nlsq_result, parameter_names, n_data, ...)` | nlsq library return (dict / tuple / object with `.x`/`.popt`) | Hybrid streaming, NLSQAdapter normalization |
| `build_failed_result(parameter_names, message, initial_params, ...)` | Failure description | All adapter / wrapper failure paths |

### Covariance computation

`_compute_covariance(jacobian, residuals, n_data, n_params)` uses the
Gauss-Newton approximation `cov = s² * (JᵀJ)⁻¹` with `s² = Σr² / (n−p)`.
A condition-number guard adds `1e-10·I` Tikhonov regularization when
`cond(JᵀJ) > 1e14`, falls back to `pinv` on `LinAlgError`, and returns
`None` when both fail.

---

## Parallel Chunk Accumulation

`optimization/nlsq/parallel_accumulator.py` provides the Gauss-Newton JᵀJ
accumulation layer used by the Out-of-Core and Hybrid Streaming strategies.

### `GaussNewtonAccumulation`

Dataclass holding the accumulated normal-equation components for one or more
chunks: `JtJ` (n_params × n_params), `Jtr` (n_params,), `chi2` (scalar),
`n_points` (int). Supports `+` for combining partial results from parallel
workers.

### `accumulate_chunks_sequential(chunks) → GaussNewtonAccumulation`

Iterates over a list of chunk callables sequentially, accumulating JᵀJ, Jᵀr,
and chi² into a single `GaussNewtonAccumulation`. Used as a fallback and for
small chunk counts (< 10).

### `accumulate_chunks_parallel(chunks, n_workers) → GaussNewtonAccumulation`

Distributes chunks across a thread pool (gate: `n_chunks ≥ 10`). Exploits
the associativity of matrix addition — partial JᵀJ sums can be reduced in
any order without affecting the result. Falls back to sequential on
`OSError`, `RuntimeError`, `PicklingError`, or timeout.

### `create_ooc_kernels(physics_config, ...) → tuple[Callable, Callable]`

JIT-kernel factory for out-of-core workers. Returns two `@jax.jit` kernels:
- `compute_chunk_accumulators(p, data...) → (JtJ, Jtr, chi2)` — full Jacobian
- `compute_chunk_chi2(p, data...) → chi2` — cost probe without Jacobian

Both kernels close over physics constants (q, dt, n_phi, t_unique) so JIT
compilation happens once per worker init, not per iteration.

### `should_use_parallel_accumulation(n_chunks, threshold=10) → bool`

Decision helper: returns `True` when `n_chunks ≥ threshold`. Keeps the
sequential path for small chunk counts where thread-pool overhead exceeds
the parallelism benefit.

### TimedContext

```python
class TimedContext:
    def __enter__(self): self._start = time.perf_counter(); return self
    def __exit__(self, *args): self.elapsed = time.perf_counter() - self._start
```

Used as `with timer: result = optimizer.run(...)`; `timer.elapsed` is then
attached to the result's `wall_time_seconds`.

### Quality Flag

`validation/fit_quality.py:classify_fit_quality(reduced_chi_squared, n_at_bounds=0)`
returns one of three flags:

| Flag | Condition |
|---|---|
| `"good"` | χ² < 1.5 **and** `n_at_bounds == 0` |
| `"marginal"` | 1.5 ≤ χ² < 3.0, **or** χ² < 1.5 but `n_at_bounds > 0` |
| `"poor"` | χ² ≥ 3.0 or `None` |

The `n_at_bounds` parameter (default `0`) counts parameters that landed at
their optimization bounds. A bound-saturated parameter may absorb residual
error and mask convergence problems — so a chi-squared-good fit is demoted
to "marginal" when any parameter hits a bound.  Existing callers that omit
`n_at_bounds` see unchanged behavior.

`FitQualityValidator` adds bounds-proximity checks (`edge_fraction = 0.005`
of the bound span) and stricter chi² thresholds for `ValidationReport`
ERROR/WARNING severity (`chi2_warn = 10.0`, `chi2_fail = 100.0`).

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

### Analysis Modes

The `analysis_mode` field selects which subset of the 14 physics parameters
participates in optimization. Total optimizer-vector length is
`n_physics_varying + 2*n_phi` for independent scaling
(`n_physics_varying + 2` in constant-averaged mode).

| Mode | Physics varying | Scaling | Total per n_phi=1 |
|---|---|---|---|
| `static_ref` | 3 | 2 | 5 |
| `static_both` | 6 | 2 | 8 |
| `two_component` | 14 | 2 | 16 |

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

---

## NLSQ as CMC Warm-Start Provider

NLSQ provides warm-start initialization for CMC: the MAP estimate sets NUTS chain starting positions and the diagonal of JᵀJ provides the initial mass matrix estimate. This reduces burn-in by approximately 60% vs. cold-start sampling. **For the full data contract (input shapes, covariance fallback, reparameterization step, failure behavior), see `cmc-fitting-architecture.md §NLSQ-to-CMC Pipeline`.**

### `--method` switch (CLI)

| Value | Behaviour | Source ref |
|---|---|---|
| `nlsq` | Run NLSQ only; no CMC. | `commands.py` line 98 |
| `cmc` | Skip NLSQ; load existing `nlsq_data.npz` from disk via `resolve_nlsq_warmstart()`; abort with warning if not found. | `commands.py` lines 113–133 |
| `both` | Run NLSQ first (`run_nlsq`), then CMC (`run_cmc(..., nlsq_results=...)`). | `commands.py` lines 98–148 |

### Programmatic API

```python
from heterodyne.optimization.nlsq import fit_nlsq_jax
from heterodyne.optimization.cmc import fit_cmc_jax

nlsq_result = fit_nlsq_jax(model, c2_data, phi_angle, config)
cmc_result = fit_cmc_jax(
    model, c2_data, phi_angle, config,
    nlsq_result=nlsq_result,
)
```

### Prior recentering

The CMC layer reads `nlsq_result.parameters` and `nlsq_result.uncertainties`
and recenters its parameter priors around the MAP point. The width is
controlled by `CMCConfig.nlsq_prior_width_factor` (NOT the legacy
`prior_width_factor`; `from_dict()` accepts the legacy spelling but
internal code uses the current name).

### Benefits

- **Improved initialization:** NUTS chains start near the posterior mode
  rather than from broad priors that are typically 5–10σ away.
- **Reduced divergence risk:** the warm-start is in the well-conditioned
  region of the likelihood, so the leapfrog integrator's step size adapts
  cleanly rather than thrashing on unusable curvature.
- **Faster mixing:** R-hat ≈ 1 and ESS comparable to chain length are
  reachable in fewer warmup samples; without warm-start the typical
  failure mode is R-hat ≫ 1 and ESS ≈ n_chains (zero effective mixing).

> **Note:** Heterodyne-specific CMC convergence benchmarks have not yet
> been published. Quantitative speed-ups are reported anecdotally and are
> sensitive to dataset size, prior tightness, and the specific
> degeneracies present in a given fit.

---

## Quick Reference Tables

### Strategy selection

| Condition | Strategy class | Executor | Memory pattern |
|---|---|---|---|
| `enable_stratified=True` | `StratifiedLSStrategy` | `nlsq.curve_fit_large` | full Jacobian |
| `hybrid_enable=True` | `HybridStreamingStrategy` | `AdaptiveHybridStreamingOptimizer` | streaming JᵀJ accumulation |
| Memory tier STREAMING (`select_nlsq_strategy`) | `OutOfCoreStrategy` | `curve_fit_large` (mmap) | disk-backed JᵀJ |
| Memory tier LARGE | adapter LARGE path | `nlsq.CurveFit` chunked | chunked JᵀJ |
| Memory tier STANDARD (default) | adapter STANDARD path / `JITStrategy` | `nlsq.CurveFit` (LRU-cached) | full Jacobian in RAM |
| Small dataset, debug | `ResidualStrategy` | `nlsq.CurveFit` | full Jacobian, dynamic shapes |
| Analytic Jacobian unreliable | `ResidualJITStrategy` | `nlsq.CurveFit` (FD Jacobian) | JIT residual only |

### Analysis mode parameter counts

| Mode | Physics varying | Scaling | Total per n_phi=1 |
|---|---|---|---|
| `static_ref` | 3 | 2 | 5 |
| `static_both` | 6 | 2 | 8 |
| `two_component` | 14 | 2 | 16 |

### Key config fields

| Concern | Field | Default |
|---|---|---|
| Convergence tolerance | `tolerance` / `ftol` / `xtol` / `gtol` | `1e-8` each |
| Solver method | `method` | `"trf"` |
| Robust loss | `loss` | `"soft_l1"` |
| Iteration cap | `max_iterations` | `1000` |
| Function-eval cap | `max_nfev` | `None` (defaults to 100×n_params) |
| Memory threshold | `nlsq_memory_fraction` | `0.75` |

---

## Key Files Reference

| File | Purpose |
|---|---|
| `optimization/nlsq/core.py` | `fit_nlsq_jax()`, `fit_nlsq_multi_phi()`, joint multi-angle dispatchers, σ²-corrected χ² calculation. |
| `optimization/nlsq/adapter.py` | `NLSQAdapter` (JAX-traced primary) and `NLSQWrapper` (scipy fallback). |
| `optimization/nlsq/adapter_base.py` | `NLSQAdapterBase` shared protocol. |
| `optimization/nlsq/config.py` | `NLSQConfig`, `HybridRecoveryConfig`, `NLSQValidationConfig`, YAML round-tripping. |
| `optimization/nlsq/results.py` | `NLSQResult` dataclass + helper methods. |
| `optimization/nlsq/result_builder.py` | Four `NLSQResult` factories + `TimedContext`. |
| `optimization/nlsq/parameter_index_mapper.py` | Bidirectional mapping between full / varying / optimizer parameter spaces. |
| `optimization/nlsq/fourier_reparam.py` | `FourierReparameterizer` for joint multi-angle scaling. |
| `optimization/nlsq/cmaes_wrapper.py` | `CMAESWrapper`, `fit_with_cmaes`, adaptive popsize, anti-degeneracy objective. |
| `optimization/nlsq/multistart.py` | Latin-hypercube multi-start optimizer. |
| `optimization/nlsq/fallback_chain.py` | Strategy degradation chain. |
| `optimization/nlsq/recovery.py` | 3-attempt error recovery and diagnostics. |
| `optimization/nlsq/anti_degeneracy_controller.py` | Post-fit degeneracy diagnostics. |
| `optimization/nlsq/hierarchical.py` | Two-stage physics/scaling optimization (Layer 2). |
| `optimization/nlsq/adaptive_regularization.py` | CV-based regularization (Layer 3). |
| `optimization/nlsq/gradient_monitor.py` | Gradient-collapse detection (Layer 4). |
| `optimization/nlsq/jacobian.py` | Jacobian condition number analysis. |
| `optimization/nlsq/memory.py` | Peak-memory estimation, `select_nlsq_strategy`. |
| `optimization/nlsq/data_prep.py` | Data preparation, `compute_degrees_of_freedom`. |
| `optimization/nlsq/parameter_utils.py` | Parameter index/name utilities. |
| `optimization/nlsq/parallel_accumulator.py` | Parallel residual / chunk accumulation. |
| `optimization/nlsq/transforms.py` | Parameter scaling/centering. |
| `optimization/nlsq/progress.py` | Progress reporting. |
| `optimization/nlsq/fit_computation.py` | Batched theoretical fit helpers. |
| `optimization/nlsq/strategies/base.py` | `FittingStrategy` ABC and `StrategyResult`. |
| `optimization/nlsq/strategies/stratified_ls.py` | `StratifiedLSStrategy`. |
| `optimization/nlsq/strategies/hybrid_streaming.py` | `HybridStreamingStrategy` (4-phase). |
| `optimization/nlsq/strategies/out_of_core.py` | `OutOfCoreStrategy`. |
| `optimization/nlsq/strategies/sequential.py` | Per-angle sequential fitting. |
| `optimization/nlsq/strategies/jit_strategy.py` | `JITStrategy`. |
| `optimization/nlsq/strategies/chunked.py` | `ChunkedStrategy`. |
| `optimization/nlsq/strategies/residual.py` | `ResidualStrategy` (dynamic shapes, full Jacobian). |
| `optimization/nlsq/strategies/residual_jit.py` | `ResidualJITStrategy` (JIT residual + FD Jacobian). |
| `optimization/nlsq/strategies/executors.py` | Strategy executor utilities. |
| `optimization/nlsq/validation/input_validator.py` | Pre-fit data / bounds checks. |
| `optimization/nlsq/validation/convergence.py` | Convergence assessment. |
| `optimization/nlsq/validation/fit_quality.py` | `classify_fit_quality()`, `FitQualityValidator`. |
| `optimization/nlsq/validation/bounds.py` | Bounds validation. |
| `optimization/nlsq/validation/result_validator.py` | Post-fit validation. |
| `optimization/nlsq/validation/result.py` | `ValidationReport`, `ValidationIssue`, `ValidationSeverity`. |
