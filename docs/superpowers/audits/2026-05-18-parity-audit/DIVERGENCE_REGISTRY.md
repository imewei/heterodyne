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

## Manual-narrative-diff escalations (3 rows; from REPORT.md §"Manual narrative diffs")

_Populated as the corresponding PRs land:_

- (PR 5) `optimization.cmc.sampler.run_nuts_with_retry` ported from homodyne — **not a divergence**, fixed in scope. Registry entry will be deleted when PR 5 lands.
- (PR 2) `optimization.nlsq.anti_degeneracy_controller` active-orchestrator architecture ported from homodyne — **not a divergence**, fixed in scope. Registry entry will be deleted when PR 2 lands.
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
