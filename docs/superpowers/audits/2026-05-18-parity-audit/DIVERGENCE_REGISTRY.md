# Divergence Registry — Heterodyne → Homodyne Strict 1:1 Parity

Records every gap from `REPORT.md` that was dispositioned `DROP` in `DISPOSITIONS.md`. Each entry names the divergence and the one-line rationale.

Populated incrementally by Phase 4 PRs as they encounter DROP-dispositioned rows. Use this file as the authoritative source of "what is intentionally different between heterodyne and homodyne" for future audit re-runs.

## D1 — Physics-exempt files (5 rows)

_Populated by Phase 4 PR 1._

## D2 — Physics-specific docs (10 rows)

_Populated by Phase 4 PR 10._

## D3 — Parameter-content / shear-physics divergence (~148 rows)

_Populated incrementally by Phase 4 PRs 1–11 as each PR encounters D3-matching rows in its scope. Each entry: `category:kind | qualname | rationale (one of: Layer-5 shear-only / homodyne-only param / homodyne-only physics class / homodyne-only CLI mode)`._

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
