# NLSQ Fitting Redesign: Port Homodyne Algorithms to Heterodyne

**Date:** 2026-03-09
**Status:** Approved

## Problem

Heterodyne calls `scipy.optimize.least_squares` directly in 10 locations across 8 files,
bypassing the NLSQ package's JAX-first optimizers, JIT caching, memory-aware routing,
streaming for large datasets, and fallback chain. Homodyne's NLSQ implementation is
tested, proven, and correct — heterodyne should mirror its algorithms.

## Design Principle

**Port homodyne's algorithms, not its code.** Heterodyne keeps its own model layer
(HeterodyneModel, ParameterTransform, NLSQResult, parameter registry). We adopt
homodyne's optimizer integration patterns — how it calls NLSQ functions, routes by
dataset size, handles fallback and recovery.

## Scope

### Files to Modify (port homodyne algorithms)

| File | Current State | Target State |
|------|--------------|-------------|
| `core/fitting.py` | ScaledFittingEngine (scipy wrapper) | Mirror homodyne: UnifiedEngine + 3 JAX solvers + DatasetSize + FitResult |
| `optimization/nlsq/adapter.py` | Lightweight CurveFit + ScipyNLSQAdapter fallback | Mirror homodyne: NLSQAdapter (CurveFit + model cache) + NLSQWrapper (curve_fit/curve_fit_large/streaming) |
| `optimization/nlsq/core.py` | _fit_local() uses scipy; _fit_joint_multi_phi() uses scipy | Mirror homodyne: dual-adapter routing (NLSQAdapter → NLSQWrapper), NLSQ package for joint fit |
| `optimization/nlsq/strategies/residual.py` | scipy.optimize.least_squares | nlsq.CurveFit (small datasets) |
| `optimization/nlsq/strategies/jit_strategy.py` | scipy.optimize.least_squares | nlsq.CurveFit with JIT caching (medium datasets) |
| `optimization/nlsq/strategies/chunked.py` | scipy.optimize.least_squares | nlsq.curve_fit_large (large datasets) |
| `optimization/nlsq/strategies/residual_jit.py` | scipy.optimize.least_squares | nlsq.CurveFit (JIT residual variant) |
| `optimization/nlsq/strategies/out_of_core.py` | scipy.optimize.least_squares | nlsq.curve_fit_large or streaming |
| `optimization/nlsq/strategies/stratified_ls.py` | scipy.optimize.least_squares | nlsq.curve_fit_large with stratified residual |
| `optimization/nlsq/strategies/hybrid_streaming.py` | scipy (Gauss-Newton phase) | nlsq.AdaptiveHybridStreamingOptimizer |
| `optimization/nlsq/config.py` | Missing NLSQ-specific fields | Add streaming config, memory limits, stability settings |
| `optimization/nlsq/result_builder.py` | Builds from scipy result | Normalize NLSQ package returns (popt, pcov) → NLSQResult |
| `optimization/nlsq/memory.py` (create) | Does not exist | Memory-based strategy selection (from homodyne) |
| `optimization/nlsq/fallback_chain.py` | May need updates | NLSQ result normalization + fallback: STREAMING → LARGE → STANDARD |

### Files NOT Modified

- `core/heterodyne_model.py`, `core/jax_backend.py`, `core/physics_cmc.py` — physics untouched
- `config/parameter_registry.py`, `config/parameter_space.py` — parameter bounds untouched
- `optimization/nlsq/results.py` — NLSQResult stays
- `optimization/nlsq/transforms.py` — ParameterTransform stays
- `optimization/nlsq/validation/` — stays

### Deletions

- `ScipyNLSQAdapter` class from `adapter.py` — NLSQ package wraps scipy internally
- Current `ScaledFittingEngine` from `core/fitting.py` — replaced by ported UnifiedEngine

## Architecture Detail

### Layer 1: `core/fitting.py` — JAX-Accelerated Fitting Engine

Mirror homodyne's `core/fitting.py` with heterodyne adaptations:

```
UnifiedHeterodyneEngine (mirrors UnifiedHomodyneEngine)
  ├─ estimate_scaling_parameters()  — JAX batch LS for contrast/offset
  ├─ compute_likelihood()           — NLL for heterodyne c2 model
  ├─ detect_dataset_size()          — categorize + log optimization strategy
  ├─ validate_inputs()              — input validation
  └─ get_parameter_info()           — parameter space introspection

ParameterSpace (adapted for heterodyne's 14+2 parameters)
DatasetSize (identical algorithm)
FitResult (adapted field names)

JAX solvers (identical algorithms):
  ├─ solve_least_squares_jax()          — batch 2×2 normal equations
  ├─ solve_least_squares_general_jax()  — N-param with Cholesky/SVD switching
  └─ solve_least_squares_chunked_jax()  — lax.scan chunked solver

ScaledFittingEngine = UnifiedHeterodyneEngine  (alias, backward compat)
```

**Heterodyne adaptations:**
- `compute_likelihood()` calls `compute_c2_heterodyne()` (not `compute_g1()`)
- `ParameterSpace` uses heterodyne's 14 physics + 2 scaling parameters
- No `analysis_mode` static/laminar_flow split (heterodyne has single transport model)
- No shear-related parameters (gamma_dot, phi0, etc.)

### Layer 2: `optimization/nlsq/adapter.py` — Dual Adapter

Mirror homodyne's dual adapter pattern:

```
NLSQAdapter (primary, v2.11.0+ equivalent)
  ├─ Uses nlsq.CurveFit class
  ├─ 64-entry LRU model cache (key: phi_angles, q, scaling mode)
  ├─ JIT compilation reuse across multi-start
  ├─ get_or_create_model() with cache stats
  └─ fit() → returns NLSQResult

NLSQWrapper (stable fallback, production equivalent)
  ├─ Memory-based routing:
  │   ├─ STANDARD: nlsq.curve_fit
  │   ├─ LARGE: nlsq.curve_fit_large
  │   └─ STREAMING: nlsq.AdaptiveHybridStreamingOptimizer
  ├─ Angle-stratified chunking for per-angle scaling
  ├─ 3-attempt error recovery
  ├─ Fallback chain: STREAMING → LARGE → STANDARD
  └─ fit() → returns NLSQResult
```

**No ScipyNLSQAdapter.** NLSQ package wraps scipy internally.

### Layer 3: Strategy Layer — NLSQ Package Integration

Each strategy replaces its scipy call with the appropriate NLSQ function:

| Strategy | Dataset Size | NLSQ Function | Homodyne Equivalent |
|----------|-------------|---------------|---------------------|
| ResidualStrategy | < 10k | nlsq.CurveFit | StandardExecutor |
| ResidualJITStrategy | < 10k (JIT residual only) | nlsq.CurveFit | StandardExecutor |
| JITStrategy | 10k–250k | nlsq.CurveFit (cached) | StandardExecutor |
| ChunkedStrategy | 250k–100M | nlsq.curve_fit_large | LargeDatasetExecutor |
| OutOfCoreStrategy | 100M+ (disk) | nlsq.curve_fit_large | LargeDatasetExecutor |
| StratifiedLSStrategy | varies | nlsq.curve_fit_large | StratifiedResidualFunctionJIT |
| HybridStreamingStrategy | 100M+ | nlsq.AdaptiveHybridStreamingOptimizer | StreamingExecutor |

### Layer 4: Entry Point — `optimization/nlsq/core.py`

```
fit_nlsq_jax()
  ├─ Configure CPU threading
  ├─ Check global optimization (CMA-ES / Multi-Start)
  │
  ├─ _fit_local():
  │   ├─ Try NLSQAdapter.fit() [primary, JAX-traced]
  │   └─ Fallback: NLSQWrapper.fit() [memory-aware routing]
  │
  ├─ Post-process results
  └─ Return NLSQResult

_fit_joint_multi_phi():
  └─ nlsq.CurveFit with Fourier-parameterized joint residual
```

### Memory-Based Routing (from homodyne)

```python
def select_nlsq_strategy(n_points, n_params, available_memory_gb):
    jacobian_bytes = n_points * n_params * 8  # float64
    total_bytes = jacobian_bytes * 3  # J, J^T J, workspace

    if total_bytes < 0.5 * available_memory_gb * 1e9:
        return NLSQStrategy.STANDARD    # nlsq.curve_fit
    elif total_bytes < 2.0 * available_memory_gb * 1e9:
        return NLSQStrategy.LARGE       # nlsq.curve_fit_large
    else:
        return NLSQStrategy.STREAMING   # AdaptiveHybridStreamingOptimizer
```

### Fallback Chain (from homodyne)

```
STREAMING → LARGE → STANDARD → fail (with actionable diagnostics)
```

Each tier catches exceptions, logs diagnostics, and demotes. No scipy escape hatch.

### NLSQ Result Normalization (from homodyne)

The NLSQ package returns different formats:
- `curve_fit` → `(popt, pcov)` or `(popt, pcov, info)`
- `curve_fit_large` → same with `full_output=True`
- `AdaptiveHybridStreamingOptimizer.fit()` → dict with `'x'`, `'pcov'`, `'streaming_diagnostics'`
- `CurveFit.fit()` → result object with `.x`, `.pcov`

`result_builder.py` normalizes all formats → `NLSQResult`.

## Implementation Phases

### Phase 1: Foundation (no dependencies)
1. Create `optimization/nlsq/memory.py` — memory-based strategy selection
2. Update `optimization/nlsq/config.py` — add NLSQ-specific fields
3. Update `optimization/nlsq/result_builder.py` — NLSQ result normalization

### Phase 2: Core fitting engine (depends on Phase 1)
4. Rewrite `core/fitting.py` — port UnifiedEngine + JAX solvers from homodyne

### Phase 3: Adapter layer (depends on Phase 1)
5. Rewrite `optimization/nlsq/adapter.py` — NLSQAdapter + NLSQWrapper, delete ScipyNLSQAdapter
6. Update `optimization/nlsq/fallback_chain.py` — NLSQ-native fallback

### Phase 4: Strategy layer (depends on Phase 3)
7. Update all 6 strategy files — replace scipy calls with NLSQ package calls

### Phase 5: Entry point (depends on Phases 3–4)
8. Update `optimization/nlsq/core.py` — dual-adapter routing, NLSQ for joint fit

### Phase 6: Cleanup and tests
9. Remove ScipyNLSQAdapter, remove scipy.optimize imports from NLSQ path
10. Update/create tests for new NLSQ integration
