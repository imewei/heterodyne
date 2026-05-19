# Phase 2 Dispositions — Heterodyne → Homodyne Parity Audit

**Audit source:** `REPORT.md` in this directory (4503 gaps, pinned to
homodyne `0368cbdb` and heterodyne `3ca229f`).

**Disposition tokens:**
- `KEEP` — fix in Phase 4
- `DROP` — accepted divergence; logged in `DIVERGENCE_REGISTRY.md` with rationale
- `DEFER` — backlogged for later (none used here — user chose "execute everything")
- `WAIVE` — overlaps with another row; not separately actioned

**Default:** every row in `REPORT.md` defaults to `KEEP` unless matched by a `DROP` or `WAIVE` rule below.

---

## Disposition summary

| Disposition | Rows | % of total |
|---|---|---|
| `KEEP` | ~4,322 | ~96% |
| `DROP` | ~168 | ~3.7% |
| `WAIVE` | 13 | ~0.3% |
| `DEFER` | 0 | 0% |
| **Total** | **4,503** | **100%** |

---

## DROP rules

Each rule names every row it covers by qualname/path. Rule numbering matches the framework agreed in brainstorming Phase 2.

### D1 — Physics-exempt files (5 rows)

**Rationale:** Per spec §2 narrow physics exemption, formula-body files and their immediate analogs do not need to mirror. `heterodyne_model.py` is the structural analog of `homodyne_model.py`; `physics_kernel.py` is a heterodyne-only formula body; `backend_api.py` is a heterodyne-only physics-backend dispatch surface used only by physics files; `shear_weighting.py` is homodyne's Layer 5 of the 5-layer anti-degeneracy system that heterodyne explicitly does not implement (no sinc shear term in the heterodyne physics model).

Rows DROPped:

```
file_inventory:
- missing_py_file: core.homodyne_model
- missing_py_file: optimization.nlsq.shear_weighting
- extra_py_file:   core.heterodyne_model
- extra_py_file:   core.physics_kernel
- extra_py_file:   core.backend_api
```

### D2 — Physics-specific docs (10 rows)

**Rationale:** These docs describe physics that does not apply to heterodyne. `theory/homodyne_scattering.rst`, `theory/yielding_dynamics.rst`, and `user_guide/03_advanced_topics/laminar_flow.rst` describe homodyne-only physics. `api/homodyne_model.rst` documents the homodyne wrapper (the heterodyne-equivalent doc will live under a different filename and is tracked separately as a KEEP). The two `theory/anti_degeneracy*.rst` pages will be ported as **one** consolidated page (the `_defense` variant is the same content under a different filename); marking one DROP, the other KEEP.

Rows DROPped:

```
file_inventory:
- missing_doc_file: api/homodyne_model.rst
- missing_doc_file: theory/homodyne_scattering.rst
- missing_doc_file: theory/yielding_dynamics.rst
- missing_doc_file: theory/anti_degeneracy_defense.rst   # consolidate with theory/anti_degeneracy.rst
- missing_doc_file: user_guide/03_advanced_topics/laminar_flow.rst
- missing_doc_file: user_guide/01_fundamentals/homodyne_overview.rst

docs (heading_drift on cross-package pages):
- heading_drift: architecture/nlsq-fitting-architecture.md  →  "Example: laminar_flow mode"
- heading_drift: architecture/nlsq-fitting-architecture.md  →  "Layer 5: Shear-Sensitivity Weighting"
- heading_drift: theory/analysis_modes.rst                  →  "Shear Integral"
- heading_drift: user_guide/01_fundamentals/parameter_guide.rst  →  "Shear Parameters (laminar_flow only)"
```

### D3 — Parameter-content / shear-physics divergence (148 rows)

**Rationale:** Per spec §2, "parameter registry content (14 vs N params) is allowed to diverge but structure/API mirrors." These rows are all **content** divergences arising from the different physics models (heterodyne has no shear term, no laminar-flow mode, no `gamma_dot_*` parameters, no `sinc_prefactor`, no separate `DiffusionModel`/`ShearModel`/`CombinedModel`/`HomodyneModel` classes). Structural/API mirroring (e.g., the ParameterRegistry class interface) is enforced via the KEEP default; only the parameter *names* and *values* may differ.

**Pattern-match (any row whose qualname/path matches one of these substrings):**

```
- "shear_weighting"
- "ShearModel"
- "ShearSensitivityWeighting"
- "ShearWeightingConfig"
- "DiffusionModel"
- "CombinedModel"
- "HomodyneModel"          # specifically the homodyne wrapper class
- "PhysicsFactors"         # homodyne's physics-factor data container (heterodyne uses different shape)
- "core.physics_factors"   # the entire module
- ".gamma_dot_"            # any gamma_dot_* parameter or method arg
- ".sinc_prefactor"
- ".shear_phi0"
- ".shear_weights"
- ".shear_weighter"
- ".shear_transforms"      # NLSQ adapter kwarg
- "--laminar-flow"         # CLI flag
- "g1_shear"
- "compute_g1_shear"
- "is_laminar_flow"
- "use_shear_weighting"
```

This pattern matches approximately **126 rows of static drift** (classes, methods, fields, configs, signatures) **+ 22 rows of P2 log_format_drift** mentioning shear/laminar, for a total of ~148 rows.

If any row would qualify under both D2 and D3, treat D2 as the operative rule (it carries the docs-specific context).

---

## WAIVE rule

### W1 — Aggregate `__all__` export rows are covered by individual signature/class rows (13 rows)

**Rationale:** Each `missing_export` row lists 10–130+ symbol names that are *also* present individually as `signatures:missing_in_heterodyne` or `classes:missing_class` rows. Fixing the underlying signature/class gaps will close these aggregate rows automatically (the implementer will re-export from `__init__.py` as part of the per-PR work). Tracking the aggregate row separately would double-count.

Rows WAIVEd:

```
exports (all 13 P0 missing_export rows):
- exports:missing_export at  ``  (root __init__, 10 names)
- exports:missing_export at  `cli`  (1 name)
- exports:missing_export at  `config`  (15 names)
- exports:missing_export at  `core`  (18 names)
- exports:missing_export at  `data`  (5 names)
- exports:missing_export at  `io`  (3 names)
- exports:missing_export at  `optimization`  (20 names)
- exports:missing_export at  `optimization.cmc.backends`  (1 name)
- exports:missing_export at  `optimization.nlsq`  (130+ names)
- exports:missing_export at  `optimization.nlsq.strategies`  (19 names)
- exports:missing_export at  `optimization.nlsq.validation`  (7 names)
- exports:missing_export at  `utils`  (4 names)
- exports:missing_export at  `viz`  (10 names)
```

**Phase 4 expectation:** the PR that fixes the underlying signature/class drift in a subpackage MUST also update that subpackage's `__init__.py` to re-export the previously-missing names. The audit re-run will verify this.

---

## Everything else → KEEP

Default disposition. Concretely covers:

| Category | Rows | Notes |
|---|---|---|
| `signatures` (P0 + P1) | 1244 | All function signature drift; aligns in Phase 4 PRs 1–8 |
| `classes` (P0) | 573 | Missing methods/fields; aligns in Phase 4 PRs 1–8 |
| `configs` (P0 + P1) | 988 | Missing/extra config keys; aligns in Phase 4 PRs 3–4 |
| `cli` (P0 + P1) | 52 | Flag drift; aligns in Phase 4 PR 5 |
| `docs` (P0 + P1 + P2) | 912 | Broken autodoc, missing pages, heading drift; aligns in Phase 4 PRs 9–10 |
| `logs_errors` (P0 + P2) | ~635 | Exit codes + format strings; aligns in Phase 4 PR 8 |
| `file_inventory` (P0 + P1) | ~37 | After D1/D2 drops, mostly file renames and absorb-then-delete; aligns in Phase 4 PRs 1, 3, 7 |
| **Total KEEP** | **~4,322** | |

Heterodyne-only files that remain `KEEP` (i.e., must be absorbed-then-deleted or renamed in Phase 4):

```
file_inventory extras (after D1 drops):
- optimization.cmc.backends.cpu_backend           → absorb into multiprocessing.py OR remove
- optimization.cmc.backends.multiprocessing_backend → rename to multiprocessing.py
- optimization.cmc.backends.pjit_backend          → rename to pjit.py
- optimization.cmc.prior_builder                  → absorb into priors.py (preserves T1–T4 work)
- optimization.cmc.warmstart                      → absorb into priors.py (preserves T1–T4 work)
- optimization.nlsq.strategies.base               → absorb into strategies/__init__.py OR keep as private
- optimization.nlsq.strategies.chunked            → rename to strategies/chunking.py
- optimization.nlsq.strategies.jit_strategy       → port to homodyne OR remove
- optimization.nlsq.validation.bounds             → absorb into validation/__init__.py OR keep as submodule (homodyne has validation/ as a dir too — verify)
- optimization.nlsq.validation.convergence        → absorb (same)
- optimization.nlsq.validation.result             → absorb (same)
```

Homodyne-only files that remain `KEEP` (i.e., must be ported to heterodyne in Phase 4):

```
file_inventory missing (after D1 drops):
- optimization.cmc.backends.multiprocessing       → satisfied by the rename above
- optimization.cmc.backends.pjit                  → satisfied by the rename above
- optimization.nlsq.strategies.chunking           → satisfied by the rename above
- optimization.nlsq.wrapper                       → genuine port (NLSQWrapper alternate adapter)
```

CLAUDE.md edits required (Phase 4 PR 3, covered by KEEP):

```
- Remove CLAUDE.md lines 21–25 (CMC config attribute rename note) once attributes revert to target_accept / max_r_hat / prior_width_factor
- Remove any "Critical Rule" line in CLAUDE.md whose behavior is reverted to homodyne's (e.g., NUTS num_warmup default, if homodyne uses a different default — Phase 4 will surface this when reverting)
```

---

## Note on the manual narrative diffs (REPORT.md §"Manual narrative diffs")

The 7 narrative diffs at the end of REPORT.md are commentary that informs Phase 3 plan, not separately dispositionable rows. The narratives surface three findings worth carrying into Phase 3:

1. **Heterodyne `cmc/sampler.py` is missing `run_nuts_with_retry`** — Phase 4 PR 4 must port this from homodyne (genuine behavioral gap, not just a rename).
2. **`anti_degeneracy_controller.py` has fundamentally different architecture** (5-layer active orchestrator in homodyne vs 3-check passive diagnostic in heterodyne) — Phase 4 PR 1/2 work needs to either port the active orchestrator from homodyne (large change) or `DROP` this architecture difference under the strict-1:1 commitment. **Action required:** decide before Phase 3 plan generation. Default for this disposition file: KEEP (port the active orchestrator from homodyne).
3. **NLSQ → CMC warm-start handoff has two distinct call signatures** that the Phase 4 PR 3 absorb-then-delete of `warmstart.py` must preserve (`clamp_params_to_interior` and `clamp_to_interior` with geometric-margin log-space math and `fixed_param_overrides` arg).

---

## Phase 3 hand-off

`superpowers:writing-plans` consumes this file plus REPORT.md to produce the Phase 4 implementation plan. The plan must:

1. Honor every `DROP` rule by adding the row's qualname/path to `DIVERGENCE_REGISTRY.md` with the rationale from this file.
2. Honor every `WAIVE` rule by treating the aggregate `__all__` row as auto-closed when the underlying signature/class gap is fixed.
3. Treat every other row as a `KEEP` that must be closed by some PR in the plan, with the targeted gap category re-counted (must drop to 0 for that PR's category) before the PR can merge.
