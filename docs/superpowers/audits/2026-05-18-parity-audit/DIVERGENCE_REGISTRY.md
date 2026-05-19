# Divergence Registry — Heterodyne → Homodyne Strict 1:1 Parity

Records every gap from `REPORT.md` that was dispositioned `DROP` in `DISPOSITIONS.md`. Each entry names the divergence and the one-line rationale.

Populated incrementally by Phase 4 PRs as they encounter DROP-dispositioned rows. Use this file as the authoritative source of "what is intentionally different between heterodyne and homodyne" for future audit re-runs.

## D1 — Physics-exempt files (5 rows)

- file_inventory:missing_py_file | `optimization.nlsq.shear_weighting` | Layer 5 (shear-sensitivity weighting); not applicable to heterodyne's velocity-phase physics model (D1, spec §2)

## D2 — Physics-specific docs (10 rows)

_Populated by Phase 4 PR 10._

## D3 — Parameter-content / shear-physics divergence (~148 rows)

_Populated incrementally by Phase 4 PRs 1–11 as each PR encounters D3-matching rows in its scope. Each entry: `category:kind | qualname | rationale (one of: Layer-5 shear-only / homodyne-only param / homodyne-only physics class / homodyne-only CLI mode)`._

- file_inventory:extra_py_file | `optimization.nlsq.strategies.chunked` | heterodyne-only ChunkedStrategy class (memory-bounded NLSQ chunking); no homodyne equivalent. KEEP (D3 heterodyne-required). Phase 4 PR 1 Task 1.2 Step 2.5 verification classified the chunked.py↔chunking.py pair as "different files" — homodyne's chunking.py contains stratification utility functions (analyze_angle_distribution, StratificationDiagnostics, create_angle_stratified_data, etc.) deferred to a separate port task (not in scope for PR 1 Task 1.2 as originally written).
- file_inventory:extra_py_file | `optimization.nlsq.strategies.base` | heterodyne-only strategy dispatch abstraction (`StrategyResult` dataclass, `FittingStrategy` Protocol, `select_strategy` function, `_estimate_available_memory`). Referenced by 11 source files: every strategy class (`chunked`, `jit_strategy`, `residual`, `residual_jit`, `sequential`, `hybrid_streaming`, `out_of_core`, `stratified_ls`), `executors.py`, `strategies/__init__.py`, and `nlsq/__init__.py`. Homodyne removed its equivalent `selection.py` in v2.12.0 (note: "Use NLSQ's WorkflowSelector instead") and replaced `FittingStrategy` with a raw-Callable ABC interface (`OptimizationExecutor.execute(residual_fn, xdata, ydata, ...)`). Migrating heterodyne to homodyne's pattern would require restructuring all 11 source files and 2 test files with no structural target to migrate to. KEEP (D3, heterodyne architectural choice). Phase 4 PR 1 Task 1.3.
- file_inventory:extra_py_file | `optimization.nlsq.strategies.jit_strategy` | heterodyne-only `JITStrategy` class: JAX-JIT-compiled full Jacobian + residual fitting strategy with XLA compilation-cache key logging, covariance estimation from the Jacobian (`_estimate_covariance_from_jac`), compile-time/run-time accounting, and a `dogbox→trf` coercion guard. Referenced from 10 distinct files: `strategies/__init__.py`, `strategies/base.py` (`select_strategy` default), `strategies/executors.py` (lazy import), `strategies/residual.py` (docstring cross-reference), `strategies/residual_jit.py` (docstring comparison), `strategies/sequential.py` (strategy dispatch map), `nlsq/__init__.py` (public re-export), and 3 test files (`test_strategies_nlsq.py`, `test_strategies_base.py`, `test_no_scipy.py`). Homodyne has no `jit_strategy.py` in its strategies directory (homodyne strategies: `residual_jit`, `executors`, `stratified_ls`, `out_of_core`, `residual`, `chunking`, `sequential`, `hybrid_streaming`). KEEP (D3, heterodyne architectural choice — JIT strategy is the primary high-performance path for the heterodyne two-path integral model). Phase 4 PR 1 Task 1.4.
- file_inventory:extra_py_file | `optimization.nlsq.validation.bounds` | heterodyne-only `BoundsValidator` class: validates fitted parameters against the heterodyne 14-parameter `DEFAULT_REGISTRY` (hard bounds check + soft edge-fraction proximity warning). Imports `DEFAULT_REGISTRY` directly and issues severity-graduated `ValidationReport` entries. Homodyne has no `bounds.py` in its validation subpackage (homodyne validation: `result_validator.py`, `fit_quality.py`, `input_validator.py`). Only 2 distinct import sites outside the file itself, both `__init__.py` re-exports (`validation/__init__.py` and `nlsq/__init__.py`), but the functionality is inseparable from heterodyne's parameter registry. KEEP (D3, heterodyne-required — bounds contract tied to the 14-param registry absent from homodyne). Phase 4 PR 1 Task 1.4.
- file_inventory:extra_py_file | `optimization.nlsq.validation.convergence` | heterodyne-only `ConvergenceValidator` class: assesses NLSQ optimizer convergence quality (termination status, iteration-limit detection, outlier-residual ratio check). Imports base types (`ValidationIssue`, `ValidationReport`, `ValidationSeverity`) from `validation/result.py`. Homodyne has no `convergence.py` in its validation subpackage; convergence checks are folded into `result_validator.py` as individual `validate_*` functions without a dedicated class. Only 2 distinct import sites outside the file itself, both `__init__.py` re-exports. KEEP (D3, heterodyne architectural choice — separate convergence class retained from heterodyne's original validation architecture). Phase 4 PR 1 Task 1.4.
- file_inventory:extra_py_file | `optimization.nlsq.validation.result` | heterodyne-only base-type module: defines `ValidationSeverity` (Enum), `ValidationIssue` (dataclass), `ValidationReport` (dataclass with `.errors`/`.warnings`/`.summary` helpers), and the class-based `ResultValidator`. These base types are imported by `validation/bounds.py`, `validation/convergence.py`, and `validation/__init__.py`. Heterodyne additionally has `validation/result_validator.py` (the homodyne-parity functional API introduced in the NLSQ parity fixes of 2026-05-11). Homodyne's `result_validator.py` defines its own local `ValidationReport` dataclass inline and exports `validate_covariance`, `validate_optimized_params`, `validate_result_consistency` as standalone functions — it has no separate base-type module. The heterodyne `result.py` cannot be collapsed into `result_validator.py` without also migrating `bounds.py` and `convergence.py` (both D3 KEEP above), which would require dissolving those two files. KEEP (D3, foundation dependency for three D3-kept files; elimination requires a coordinated multi-file restructure deferred to a future parity PR). Phase 4 PR 1 Task 1.4.
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.get_shear_weights` | Layer 5 method; heterodyne has no shear term in the g2 formula (D3, spec §2). Phase 4 PR 2 Task 2.4.
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.update_shear_phi0` | Layer 5 method; heterodyne has no shear term in the g2 formula (D3, spec §2). Phase 4 PR 2 Task 2.4.
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.use_shear_weighting` | Layer 5 property; always False in heterodyne — no shear physics (D3, spec §2). Phase 4 PR 2 Task 2.4.
- classes:missing_field  | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.shear_weighter` | Layer 5 field; heterodyne has no shear term in the g2 formula (D3, spec §2). Phase 4 PR 2 Task 2.4.
- missing_method | `optimization.nlsq.adapter.NLSQAdapter.fit` kwarg `shear_transforms` | Layer-5 shear-only; heterodyne has no shear sinc term in the g2 formula, so this kwarg is omitted from the ported `fit()` signature. `analysis_mode` default changed from homodyne's `'static_isotropic'` to `'full'` — heterodyne always uses its 14-parameter full model. Phase 4 PR 3 Batch 1.

## Manual-narrative-diff escalations (3 rows; from REPORT.md §"Manual narrative diffs")

_Populated as the corresponding PRs land:_

- (PR 5) `optimization.cmc.sampler.run_nuts_with_retry` ported from homodyne — **not a divergence**, fixed in scope. Registry entry will be deleted when PR 5 lands.
- (PR 4) `optimization.cmc.warmstart.clamp_params_to_interior` + `clamp_to_interior` signatures preserved during absorption into `priors.py` — **not a divergence**, fixed in scope. Registry entry will be deleted when PR 4 lands.

---

## How to add an entry

When a Phase 4 PR encounters a DROP-dispositioned row, append a line under the matching D1/D2/D3 section:

```
- {category}:{kind} | `{qualname or path}` | {one-line rationale}
```

Example:
```
- file_inventory:missing_py_file | `core.homodyne_model` | physics wrapper analog; heterodyne has core.heterodyne_model (D1)
- file_inventory:extra_py_file   | `core.physics_kernel` | heterodyne-only physics formula body, spec §2 exempt (D1)
```
