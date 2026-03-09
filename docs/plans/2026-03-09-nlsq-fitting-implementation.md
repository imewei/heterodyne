# NLSQ Fitting Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all 10 direct `scipy.optimize.least_squares` call sites with NLSQ package integration, mirroring homodyne's proven algorithms.

**Architecture:** Port homodyne's dual-adapter pattern (NLSQAdapter + NLSQWrapper) with memory-based routing (STANDARD → LARGE → STREAMING), angle-stratified chunking, and fallback chain. Heterodyne keeps its own model layer (HeterodyneModel, ParameterTransform, NLSQResult, parameter registry).

**Tech Stack:** Python 3.13, JAX, nlsq>=0.6.4, NumPy, NumPyro

**Reference files (homodyne, read-only):**
- `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/memory.py`
- `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/wrapper.py`
- `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/adapter.py`
- `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/fallback_chain.py`
- `/home/wei/Documents/GitHub/homodyne/homodyne/core/fitting.py`

---

## Phase 1: Foundation (no dependencies)

### Task 1: Create memory-based strategy selection module

Port homodyne's `memory.py` algorithm for memory-aware NLSQ strategy routing.

**Files:**
- Create: `heterodyne/optimization/nlsq/memory.py`
- Test: `tests/unit/nlsq/test_memory.py`

**Reference:** `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/memory.py`

**What to port (algorithm, not code):**
- `NLSQStrategy` enum: `STANDARD`, `LARGE`, `STREAMING` (drop `OUT_OF_CORE` — heterodyne uses `ChunkedStrategy` for that range)
- `StrategyDecision` frozen dataclass: strategy, threshold_gb, peak_memory_gb, reason
- `detect_total_system_memory()` → `float | None` using `psutil` with fallback
- `estimate_peak_memory_gb(n_points, n_params)` with Jacobian overhead factor 6.5
- `select_nlsq_strategy(n_points, n_params, memory_fraction=0.75)` → `StrategyDecision`
- Constants: `DEFAULT_MEMORY_FRACTION = 0.75`, `FALLBACK_THRESHOLD_GB = 16.0`

**Heterodyne adaptations:**
- Use `heterodyne.utils.logging.get_logger` (not homodyne's)
- No `log_phase` decorator (heterodyne doesn't have it)
- No `nlsq.caching.get_memory_manager` import (optional in homodyne, skip)

**Step 1:** Write test file `tests/unit/nlsq/test_memory.py`:
- `test_strategy_enum_values()` — verify STANDARD/LARGE/STREAMING
- `test_estimate_peak_memory_small()` — 1000 points, 16 params → small memory
- `test_estimate_peak_memory_large()` — 10M points, 16 params → large memory
- `test_select_standard_strategy()` — small dataset → STANDARD
- `test_select_large_strategy()` — medium dataset → LARGE
- `test_select_streaming_strategy()` — huge dataset → STREAMING
- `test_strategy_decision_frozen()` — StrategyDecision is immutable

**Step 2:** Run: `uv run pytest tests/unit/nlsq/test_memory.py -v` → expect FAIL

**Step 3:** Implement `heterodyne/optimization/nlsq/memory.py`

**Step 4:** Run: `uv run pytest tests/unit/nlsq/test_memory.py -v` → expect PASS

**Step 5:** Commit: `feat(nlsq): add memory-based strategy selection`

---

### Task 2: Add NLSQ-specific fields to NLSQConfig

Add streaming, memory, and NLSQ integration fields to the existing config.

**Files:**
- Modify: `heterodyne/optimization/nlsq/config.py`
- Test: `tests/unit/nlsq/test_config.py` (extend existing)

**What to add to `NLSQConfig` dataclass (after line ~373):**
```python
# NLSQ package integration (mirrors homodyne wrapper.py)
nlsq_stability: str = "auto"  # 'auto', 'check', False-like
nlsq_rescale_data: bool = False  # xdata is indices, not physical
nlsq_x_scale: str | np.ndarray = "jac"  # trust-region scaling
nlsq_memory_fraction: float = 0.75  # fraction of RAM for NLSQ
nlsq_memory_fallback_gb: float = 16.0  # fallback if detection fails
```

**Step 1:** Add tests to existing test file or create `tests/unit/nlsq/test_config_nlsq.py`:
- `test_nlsq_config_defaults()` — verify new field defaults
- `test_nlsq_config_from_dict()` — verify from_dict handles new keys
- `test_nlsq_config_to_dict()` — verify round-trip

**Step 2:** Run tests → expect FAIL

**Step 3:** Add fields to `NLSQConfig` in config.py. Update `from_dict()` and `to_dict()`.

**Step 4:** Run tests → expect PASS

**Step 5:** Commit: `feat(nlsq): add NLSQ package integration fields to config`

---

### Task 3: Add NLSQ result normalization to result_builder

Port homodyne's `handle_nlsq_result()` algorithm for normalizing different NLSQ return formats.

**Files:**
- Modify: `heterodyne/optimization/nlsq/result_builder.py`
- Test: `tests/unit/nlsq/test_result_builder.py` (extend existing)

**Reference:** homodyne's `fallback_chain.py:handle_nlsq_result()` (lines 87-210)

**What to add:**
```python
def build_result_from_nlsq(
    nlsq_result: Any,
    parameter_names: list[str],
    n_data: int,
    wall_time: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Normalize any NLSQ package return format → NLSQResult.

    Handles:
    - dict with 'x', 'pcov' keys (streaming optimizer)
    - (popt, pcov) tuple (curve_fit)
    - (popt, pcov, info) tuple (curve_fit with full_output)
    - object with .x, .pcov attributes (CurveFit result)
    """
```

**Step 1:** Write tests:
- `test_build_from_nlsq_tuple()` — (popt, pcov) input
- `test_build_from_nlsq_triple()` — (popt, pcov, info) input
- `test_build_from_nlsq_dict()` — dict input from streaming
- `test_build_from_nlsq_object()` — object with .x, .pcov
- `test_build_from_nlsq_none_pcov()` — pcov=None handling

**Step 2:** Run → FAIL

**Step 3:** Implement `build_result_from_nlsq()` in result_builder.py

**Step 4:** Run → PASS

**Step 5:** Commit: `feat(nlsq): add NLSQ result normalization to result_builder`

---

## Phase 2: Core fitting engine (depends on Phase 1)

### Task 4: Rewrite core/fitting.py — port UnifiedEngine + JAX solvers

Replace `ScaledFittingEngine` (scipy wrapper) with homodyne-mirrored `UnifiedHeterodyneEngine`.

**Files:**
- Rewrite: `heterodyne/core/fitting.py`
- Modify: `heterodyne/core/__init__.py` (update exports)
- Test: `tests/unit/core/test_fitting.py`

**Reference:** `/home/wei/Documents/GitHub/homodyne/homodyne/core/fitting.py` (866 lines)

**What to port (algorithms, adapted for heterodyne):**

1. **`ParameterSpace` dataclass** — adapted for heterodyne's 14+2 params:
   - Scaling: contrast_bounds, offset_bounds, contrast_prior, offset_prior
   - Transport: D0_ref_bounds, D0_sample_bounds, alpha_ref_bounds, alpha_sample_bounds, v0_bounds, v_offset_bounds, etc.
   - `get_param_bounds()`→ returns bounds list from ParameterManager
   - `get_param_priors()` → returns priors list
   - Optional `config_manager` for bound override

2. **`DatasetSize`** — identical algorithm:
   - SMALL (<1M), MEDIUM (1-10M), LARGE (>20M)
   - `categorize(data_size: int) -> str`

3. **`FitResult` dataclass** — adapted for heterodyne:
   - params, contrast, offset, chi_squared, reduced_chi_squared, degrees_of_freedom
   - param_errors, residual_std, max_residual, fit_iterations, converged
   - computation_time, backend, dataset_size
   - `get_summary() -> dict`

4. **`UnifiedHeterodyneEngine`** — adapted:
   - `__init__(parameter_space)` — no `analysis_mode` param (heterodyne has single model)
   - `estimate_scaling_parameters(data, theory, validate_bounds)` → (contrast, offset) via JAX batch LS
   - `compute_likelihood(params, contrast, offset, data, sigma, t, phi, q, dt)` → NLL using `compute_c2_heterodyne`
   - `detect_dataset_size(data)` → category string with logging
   - `validate_inputs(data, sigma, t, phi, q)` → ValueError on invalid
   - `get_parameter_info()` → dict

5. **`ScaledFittingEngine = UnifiedHeterodyneEngine`** — backward compat alias

6. **JAX solvers** — identical algorithms:
   - `solve_least_squares_jax(theory_batch, exp_batch)` — batch 2×2 normal equations
   - `solve_least_squares_general_jax(design_matrix, target_vector, regularization)` — N-param with Cholesky/SVD
   - `solve_least_squares_chunked_jax(theory_chunks, exp_chunks)` — lax.scan chunked

**Heterodyne adaptations:**
- Import `compute_c2_heterodyne` from `heterodyne.core.jax_backend` (not `compute_g1`)
- `ParameterSpace` uses heterodyne's 14 physics params (D0_ref, D0_sample, alpha_ref, alpha_sample, v0, v_offset, etc.) — get names from `heterodyne.config.parameter_registry`
- No `analysis_mode` parameter (no static/laminar_flow split)
- No shear parameters (gamma_dot, phi0, beta)
- `compute_likelihood()` uses c2 correlation (not g1²)
- JAX fallback stubs when JAX unavailable (identical pattern)

**Step 1:** Write `tests/unit/core/test_fitting.py`:
- `test_parameter_space_defaults()` — bounds and priors populated
- `test_parameter_space_get_bounds()` — returns correct count
- `test_dataset_size_categorize()` — SMALL/MEDIUM/LARGE thresholds
- `test_fit_result_get_summary()` — dict structure
- `test_solve_least_squares_jax_identity()` — theory=data → contrast≈1, offset≈0
- `test_solve_least_squares_jax_scaling()` — known contrast/offset recovery
- `test_solve_least_squares_general_jax()` — N-param regression
- `test_solve_least_squares_chunked_jax()` — matches non-chunked result
- `test_engine_estimate_scaling()` — contrast/offset estimation
- `test_engine_validate_inputs_empty()` — raises on empty data
- `test_engine_validate_inputs_negative_q()` — raises on q<=0

**Step 2:** Run → FAIL

**Step 3:** Implement the full `core/fitting.py`

**Step 4:** Update `core/__init__.py` exports: keep `ScaledFittingEngine`, add new exports

**Step 5:** Run → PASS

**Step 6:** Commit: `feat(core): port UnifiedEngine and JAX solvers from homodyne`

---

## Phase 3: Adapter layer (depends on Phase 1)

### Task 5: Rewrite adapter.py — NLSQAdapter + NLSQWrapper

Replace current adapter.py (lightweight CurveFit + ScipyNLSQAdapter) with homodyne-mirrored dual adapter.

**Files:**
- Rewrite: `heterodyne/optimization/nlsq/adapter.py`
- Test: `tests/unit/nlsq/test_adapter.py`

**Reference:**
- homodyne `adapter.py` for NLSQAdapter (CurveFit model caching)
- homodyne `wrapper.py` for NLSQWrapper (curve_fit/curve_fit_large/streaming routing)

**What to implement:**

**A. Model cache utilities (top of file):**
```python
ModelCacheKey = frozen dataclass(analysis_mode, phi_angles: tuple, q, per_angle_scaling)
CachedModel = dataclass(model, model_func, created_at, n_hits)
_model_cache: dict  # 64-entry LRU
get_or_create_model() → (model, model_func, cache_hit)
clear_model_cache()
get_cache_stats() → dict
```

**B. NLSQAdapter class (primary, JAX-traced):**
- Uses `nlsq.CurveFit` class for JIT compilation caching
- `fit()` method: creates/gets cached CurveFit instance, calls `fitter.curve_fit()`, normalizes result via `build_result_from_nlsq()`
- Model caching across multi-start calls
- Convergence assessment (NaN check, χ² sanity, no-progress detection)

**C. NLSQWrapper class (stable fallback):**
- **CRITICAL import order:** `from nlsq import curve_fit, curve_fit_large` BEFORE JAX imports
- Memory-based routing via `select_nlsq_strategy()` from `memory.py`:
  - STANDARD → `nlsq.curve_fit()`
  - LARGE → `nlsq.curve_fit_large()`
  - STREAMING → `nlsq.AdaptiveHybridStreamingOptimizer`
- Fallback chain: STREAMING → LARGE → STANDARD
- 3-attempt error recovery
- `fit()` method: selects strategy, executes with fallback, normalizes result

**D. Delete ScipyNLSQAdapter entirely.**

**Step 1:** Write `tests/unit/nlsq/test_adapter.py`:
- `test_model_cache_hit()` — second call returns cached
- `test_model_cache_eviction()` — >64 entries triggers eviction
- `test_cache_stats()` — hits/misses tracked
- `test_nlsq_adapter_fit_returns_result()` — mock CurveFit, verify NLSQResult
- `test_nlsq_wrapper_standard_strategy()` — small data → curve_fit called
- `test_nlsq_wrapper_large_strategy()` — large data → curve_fit_large called
- `test_nlsq_wrapper_fallback_on_error()` — STREAMING fails → LARGE tried
- `test_no_scipy_adapter()` — ScipyNLSQAdapter does not exist

**Step 2:** Run → FAIL

**Step 3:** Implement adapter.py

**Step 4:** Run → PASS

**Step 5:** Commit: `feat(nlsq): port dual-adapter pattern from homodyne`

---

### Task 6: Update fallback_chain.py — NLSQ-native fallback

Update the fallback chain to use NLSQ result normalization and remove scipy dependencies.

**Files:**
- Modify: `heterodyne/optimization/nlsq/fallback_chain.py`
- Test: `tests/unit/nlsq/test_fallback_chain.py`

**Reference:** homodyne `fallback_chain.py` (lines 87-210 for result normalization, 213-406 for execution)

**What to change:**
- `execute_optimization_with_fallback()`: route to NLSQWrapper methods instead of strategy objects
- Add `handle_nlsq_result()` for normalizing NLSQ returns (or delegate to result_builder)
- Ensure fallback order matches homodyne: STREAMING → CHUNKED → LARGE → STANDARD
- Remove any scipy.optimize imports

**Step 1:** Write/update tests:
- `test_fallback_order()` — STREAMING → CHUNKED → LARGE → STANDARD → None
- `test_handle_nlsq_result_tuple()` — (popt, pcov) → normalized
- `test_handle_nlsq_result_dict()` — streaming dict → normalized
- `test_execute_with_fallback_success()` — first strategy succeeds
- `test_execute_with_fallback_demotes()` — first fails, second succeeds

**Step 2-5:** TDD cycle + commit: `feat(nlsq): update fallback chain for NLSQ-native routing`

---

## Phase 4: Strategy layer (depends on Phase 3)

### Task 7: Replace scipy in all strategy files

Update each strategy to use NLSQ package calls instead of scipy.optimize.least_squares.

**Files to modify (6 strategy files):**
- `heterodyne/optimization/nlsq/strategies/residual.py` (line 42, 256)
- `heterodyne/optimization/nlsq/strategies/jit_strategy.py` (line 45, 338)
- `heterodyne/optimization/nlsq/strategies/residual_jit.py` (line 18, 115)
- `heterodyne/optimization/nlsq/strategies/chunked.py` (line 39, 331)
- `heterodyne/optimization/nlsq/strategies/out_of_core.py` (line 15, 121)
- `heterodyne/optimization/nlsq/strategies/stratified_ls.py` (line 15, 96)

**Test:** `tests/unit/nlsq/test_strategies_nlsq.py`

**Pattern for each file — replace:**
```python
from scipy.optimize import least_squares
# ...
result = least_squares(residual_fn, x0, bounds=..., method=..., ...)
```

**With (small/medium datasets):**
```python
from nlsq import CurveFit
# ...
fitter = CurveFit(flength=n_data)
popt, pcov = fitter.curve_fit(
    f=model_fn, xdata=xdata, ydata=ydata,
    p0=x0, bounds=bounds, method=config.method,
    ftol=config.ftol, gtol=config.gtol,
    loss=config.loss, stability=config.nlsq_stability,
)
result = build_result_from_nlsq((popt, pcov), ...)
```

**With (large datasets — chunked.py, out_of_core.py):**
```python
from nlsq import curve_fit_large
# ...
popt, pcov = curve_fit_large(
    f=model_fn, xdata=xdata, ydata=ydata,
    p0=x0, bounds=bounds,
    memory_limit_gb=available_memory * config.nlsq_memory_fraction,
)
result = build_result_from_nlsq((popt, pcov), ...)
```

**With (streaming — hybrid_streaming.py):**
```python
from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
# ...
stream_config = HybridStreamingConfig(
    chunk_size=config.streaming_chunk_size,
    enable_multistart=config.multistart,
    n_starts=config.multistart_n,
)
optimizer = AdaptiveHybridStreamingOptimizer(config=stream_config)
stream_result = optimizer.fit(
    data_source=(xdata, ydata), func=model_fn,
    p0=x0, bounds=bounds,
)
result = build_result_from_nlsq(stream_result, ...)
```

**Each strategy also needs a wrapper to convert residual_fn to curve_fit's (f, xdata, ydata) format:**
```python
def _make_curve_fit_func(residual_fn, n_data):
    """Convert residual_fn(params) → curve_fit-compatible f(x, *params)."""
    xdata = np.arange(n_data, dtype=np.float64)
    ydata = np.zeros(n_data, dtype=np.float64)

    def model_fn(x, *params):
        return residual_fn(np.array(params)) + ydata  # residual = model - data → model = residual + data
        # Actually: curve_fit minimizes ||f(x,p) - ydata||², and residual_fn returns (model - data)*w
        # So we need: f(x, *params) = model(params)[x_indices]
        # The exact conversion depends on how each strategy's residual_fn is structured.
    return model_fn, xdata, ydata
```

**IMPORTANT:** The exact residual → curve_fit conversion depends on each strategy's internals. Read each strategy's `fit()` method carefully to understand what `residual_fn` returns before converting. Some return weighted residuals, some return raw model-data differences.

**Step 1:** Write `tests/unit/nlsq/test_strategies_nlsq.py`:
- `test_residual_strategy_no_scipy_import()` — verify no scipy.optimize import
- `test_jit_strategy_no_scipy_import()` — same
- `test_chunked_strategy_uses_curve_fit_large()` — mock nlsq, verify called
- For each strategy: mock the NLSQ function, call fit(), verify NLSQResult returned

**Step 2:** Run → FAIL

**Step 3:** Update each strategy file (one at a time, test between each)

**Step 4:** Run → PASS

**Step 5:** Commit: `feat(nlsq): replace scipy with NLSQ package in all strategies`

---

### Task 8: Update hybrid_streaming.py for AdaptiveHybridStreamingOptimizer

Special handling — this strategy's L-BFGS warmup phase uses `scipy.optimize.minimize`, which is separate from least_squares. The Gauss-Newton phase should use NLSQ.

**Files:**
- Modify: `heterodyne/optimization/nlsq/strategies/hybrid_streaming.py`
- Test: `tests/unit/nlsq/test_hybrid_streaming.py`

**What to change:**
- Replace scipy Gauss-Newton phase with `nlsq.AdaptiveHybridStreamingOptimizer`
- L-BFGS warmup can stay (it's `scipy.optimize.minimize`, not `least_squares`) OR port to NLSQ's built-in warmup
- Mirror homodyne's `create_multistart_warmup_func()` pattern

**Step 1-5:** TDD cycle + commit: `feat(nlsq): integrate AdaptiveHybridStreamingOptimizer in hybrid strategy`

---

## Phase 5: Entry point (depends on Phases 3-4)

### Task 9: Update core.py — dual-adapter routing

Update `fit_nlsq_jax()` and `_fit_joint_multi_phi()` to use the new adapter layer.

**Files:**
- Modify: `heterodyne/optimization/nlsq/core.py`
- Test: `tests/unit/nlsq/test_core.py` (extend existing)

**What to change in `_fit_local()` (~line 639-711):**
- Primary: `NLSQAdapter.fit()` (JAX-traced CurveFit)
- Fallback: `NLSQWrapper.fit()` (memory-aware routing)
- Remove: `ScipyNLSQAdapter` fallback (deleted in Task 5)

**What to change in `_fit_joint_multi_phi()` (~line 343):**
- Replace `scipy.optimize.least_squares(joint_residual_fn, ...)` with `nlsq.CurveFit` call
- Use `build_result_from_nlsq()` for result normalization

**Step 1:** Write/extend tests:
- `test_fit_local_uses_nlsq_adapter()` — mock NLSQAdapter, verify called
- `test_fit_local_fallback_to_wrapper()` — adapter fails → wrapper tried
- `test_fit_joint_multi_phi_no_scipy()` — verify no scipy import
- `test_fit_nlsq_jax_end_to_end()` — small synthetic dataset → valid NLSQResult

**Step 2-5:** TDD cycle + commit: `feat(nlsq): update core.py with dual-adapter routing`

---

## Phase 6: Cleanup and verification

### Task 10: Remove all scipy.optimize.least_squares imports from NLSQ path

Final sweep to ensure zero scipy in the optimization pipeline.

**Files to verify (all under `heterodyne/`):**
- `core/fitting.py` — should have NO scipy imports
- `optimization/nlsq/adapter.py` — should have NO scipy imports
- `optimization/nlsq/core.py` — should have NO scipy.optimize.least_squares
- `optimization/nlsq/strategies/residual.py` — NO scipy
- `optimization/nlsq/strategies/jit_strategy.py` — NO scipy
- `optimization/nlsq/strategies/residual_jit.py` — NO scipy
- `optimization/nlsq/strategies/chunked.py` — NO scipy
- `optimization/nlsq/strategies/out_of_core.py` — NO scipy
- `optimization/nlsq/strategies/stratified_ls.py` — NO scipy
- `optimization/nlsq/strategies/hybrid_streaming.py` — `scipy.optimize.minimize` OK (L-BFGS), but NO `least_squares`

**Test:** `tests/unit/nlsq/test_no_scipy.py`

```python
import importlib
import ast

NLSQ_FILES = [
    "heterodyne/core/fitting.py",
    "heterodyne/optimization/nlsq/adapter.py",
    "heterodyne/optimization/nlsq/core.py",
    "heterodyne/optimization/nlsq/strategies/residual.py",
    "heterodyne/optimization/nlsq/strategies/jit_strategy.py",
    "heterodyne/optimization/nlsq/strategies/residual_jit.py",
    "heterodyne/optimization/nlsq/strategies/chunked.py",
    "heterodyne/optimization/nlsq/strategies/out_of_core.py",
    "heterodyne/optimization/nlsq/strategies/stratified_ls.py",
]

def test_no_scipy_least_squares_in_nlsq_path():
    """Ensure no file imports scipy.optimize.least_squares."""
    for filepath in NLSQ_FILES:
        source = open(filepath).read()
        assert "from scipy.optimize import least_squares" not in source, f"{filepath} still imports scipy least_squares"
        assert "scipy.optimize.least_squares" not in source, f"{filepath} still references scipy least_squares"
```

**Step 1:** Write `tests/unit/nlsq/test_no_scipy.py`

**Step 2:** Run → expect PASS (if all previous tasks done correctly) or FAIL (find stragglers)

**Step 3:** Fix any remaining scipy imports

**Step 4:** Run full test suite: `uv run pytest tests/ -v --tb=short`

**Step 5:** Commit: `refactor(nlsq): remove all scipy.optimize.least_squares from NLSQ path`

---

### Task 11: Update existing tests for new adapter API

Some existing tests may import `ScipyNLSQAdapter` or `ScaledFittingEngine` with old signatures.

**Files:** Search for and update:
- `tests/unit/nlsq/test_adapter.py` (if exists) — remove ScipyNLSQAdapter tests
- `tests/unit/core/test_fitting.py` (if exists) — update for new engine
- `tests/integration/` — update any integration tests that use old fitting API
- `tests/unit/nlsq/test_core.py` (if exists) — update for dual-adapter routing

**Step 1:** Run: `uv run pytest tests/ -v --tb=short 2>&1 | head -100` — identify failures

**Step 2:** Fix each failing test

**Step 3:** Run full suite → all PASS

**Step 4:** Commit: `test(nlsq): update tests for NLSQ package integration`

---

## Summary

| Phase | Tasks | Scope | Key Deliverable |
|-------|-------|-------|-----------------|
| 1 | 1-3 | Foundation | memory.py, config fields, result normalization |
| 2 | 4 | Core engine | fitting.py with JAX solvers |
| 3 | 5-6 | Adapters | NLSQAdapter + NLSQWrapper + fallback chain |
| 4 | 7-8 | Strategies | All 6 strategies use NLSQ package |
| 5 | 9 | Entry point | core.py dual-adapter routing |
| 6 | 10-11 | Cleanup | Zero scipy, all tests pass |

**Total: 11 tasks, ~6 phases, estimated ~2500 LOC changed**

**Verification command (run after each phase):**
```bash
uv run pytest tests/ -v --tb=short -x
uv run ruff check heterodyne/optimization/nlsq/ heterodyne/core/fitting.py
```
