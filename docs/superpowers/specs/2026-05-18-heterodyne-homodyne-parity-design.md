# Heterodyne → Homodyne Strict 1:1 Parity — Design

**Date:** 2026-05-18
**Author:** Wei Chen (with Claude)
**Status:** Approved (brainstorming complete)
**Next:** writing-plans → audit execution

---

## 1. Goal

Make `heterodyne` a **strict 1:1 structural and behavioral mirror** of `homodyne`, with the only allowed divergence being the bodies of physics formula files and the absence of the homodyne-only shear-sensitivity weighting layer.

After this work, any silent drift between the two packages — public API signatures, config keys, CLI flags, log formats, exit codes, file structure, docs structure — should be impossible to introduce without a failing CI gate catching it.

## 2. Scope decisions (locked)

| Decision | Choice |
|---|---|
| Mirror goal | Full behavioral parity audit |
| Deliverable | Audit + plan + execute everything to 1:1 parity |
| Divergence policy | Strict 1:1 mirror — **true zero divergence** |
| Physics exemption | Narrow — only formula bodies in `core/theory.py`, `core/physics.py`, `core/physics_cmc.py`, `core/physics_nlsq.py`, `core/physics_utils.py`, `core/jax_backend.py`. Their public function signatures still must mirror homodyne where the homodyne file exists. |
| Execution shape | Stage-gated: audit → user review → plan → execute |
| Docs in scope | Yes — Sphinx tree, autodoc, cross-refs, file count |

### True zero divergence — concrete consequences

1. **CMC config attribute names revert** to homodyne names:
   - `target_accept_prob` → `target_accept`
   - `max_r_hat` → `r_hat_threshold`
   - `nlsq_prior_width_factor` → `prior_width_factor`
   - `from_dict()` legacy-key shim removed
   - CLAUDE.md lines 21–25 deleted
2. **CLAUDE.md "Critical Rules"** that conflict with homodyne behavior revert; the rule line in CLAUDE.md is deleted. Rules that don't conflict (e.g. gradient-safe floors as a coding pattern that applies to both packages) stay.
3. **Heterodyne-only files** (`cmc/warmstart.py`, `cmc/prior_builder.py`, `cmc/backends/cpu_backend.py`) are absorbed into the homodyne-named equivalent (`cmc/priors.py`, `cmc/backends/multiprocessing.py`) and then deleted.
4. **CMC backend file renames**: `multiprocessing_backend.py` → `multiprocessing.py`, `pjit_backend.py` → `pjit.py`.

### Physics-exempt content divergence (allowed)

The parameter-registry exemption permits **values** in the registry to differ (14 params vs N homodyne params, different default bounds/priors). It does **not** permit registry **structure or consumer-API** drift.

NUTS warmup floor (currently 1500 in heterodyne for dense_mass on 14 params) is a config default. If the audit shows homodyne defaults lower, the dispositioner can mark it `DROP` (keep heterodyne's 1500) under the parameter-registry exemption — but this must be logged in `DIVERGENCE_REGISTRY.md` with `WARN: physics-required value`. Every such case is decided individually during Phase 2 review.

## 3. Architecture — four stage-gated phases

```
Phase 1 (AUDIT)
  ├─ Run 7 extractors against both packages, side-by-side
  ├─ Diff, categorize by severity (P0/P1/P2/P3)
  ├─ Manual narrative pass for 5–10 dynamic-behavior modules
  └─ ARTIFACT → docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md

Phase 2 (REVIEW)
  ├─ User reads REPORT.md
  ├─ Marks each gap row: KEEP / DROP / DEFER / WAIVE
  └─ ARTIFACT → docs/superpowers/audits/2026-05-18-parity-audit/DISPOSITIONS.md

Phase 3 (PLAN)
  ├─ Invoke superpowers:writing-plans against KEEP rows
  ├─ Group into PR-sized batches by subpackage
  └─ ARTIFACT → docs/superpowers/plans/2026-05-18-heterodyne-homodyne-parity-plan.md

Phase 4 (EXECUTE)
  ├─ Execute via superpowers:executing-plans (separate session)
  ├─ Each PR: implement + tests + re-run audit to verify gap closed
  └─ ARTIFACT → merged PRs + final audit re-run showing 0 must-fix gaps
```

**Source-of-truth pinning:** Phase 1 records exact homodyne git SHA in `homodyne_sha.txt`. Phase 4 PRs reference this SHA. Homodyne drift mid-execution is out of scope; a future "homodyne-drift refresh" audit handles it.

## 4. Phase 1 — Audit mechanics

### 4.1 Extractors

Stored in `tools/parity_audit/`. Pure Python, stdlib-only (no external deps).

| # | Extractor | Source | Output schema |
|---|---|---|---|
| 1 | `extract_signatures.py` | Every `*.py` not in `tests/`, not in physics-exempt files | `module.qualname(arg: Type = default, …) -> Return` |
| 2 | `extract_exports.py` | Every `__init__.py` | Sorted `__all__` list per module |
| 3 | `extract_classes.py` | All public classes | Name, base classes, public method signatures, dataclass fields |
| 4 | `extract_configs.py` | All `config.get(`/`config["`/YAML key strings; dataclass fields in `*/config.py` | Hierarchical key paths (e.g. `optimization.cmc.target_accept`) |
| 5 | `extract_cli.py` | `cli/args_parser.py`, `cli/commands.py`, every `add_argument` call; main.py subparser registrations | Flag name, dest, default, choices, help; command + alias list |
| 6 | `extract_logs_errors.py` | All `logger.{info,warning,error,debug}(…)`; all `raise X(…)` first-arg literals; all `sys.exit(N)` | Format strings, exception class + message stems, exit codes |
| 7 | `extract_docs.py` | All `docs/source/**/*.rst` and `*.md` | Relative path, top-level headings, autodoc directives (`automodule`, `autoclass`, `autofunction`), toctree entries, cross-references (`:func:`, `:class:`, `:ref:`) |
| — | `extract_file_inventory.py` | `find heterodyne homodyne -name '*.py' -o -name '*.rst' -o -name '*.md'` | Pair-wise inventory: present-in-both, homodyne-only, heterodyne-only |
| — | `diff_extracts.py` | Reads extractor JSON outputs | Categorized human-readable diff → `REPORT.md` |

### 4.2 Severity rubric

| Severity | Definition | Examples |
|---|---|---|
| **P0 — silent breakage** | API signature, public class shape, config key, CLI flag, or exit code differs in a way that would break a caller written against homodyne. **Also: Sphinx build fails on heterodyne due to broken autodoc/xref.** | `fit_nlsq_jax(data, config)` vs `fit_nlsq_jax(config, data)`; missing CLI flag `--no-cmc`; exit code 2 vs 1 for same condition; missing `cmc/multiprocessing.py` referenced by `automodule::` |
| **P1 — structural drift** | File missing/extra/renamed; module name differs; `__all__` differs; **missing top-level docs page or section** | `multiprocessing_backend.py` vs `multiprocessing.py`; heterodyne-only `warmstart.py`; missing `docs/source/theory/anti_degeneracy_defense.rst` |
| **P2 — observable drift** | Log format strings differ; error message stems differ; help text differs; **heading text drift, missing example, missing config reference** | Different banner text; different log phase names |
| **P3 — cosmetic** | Docstring text differs; type hint style differs; comment density differs; **prose-only differences** | Docstring difference only |

P3 is **recorded but not in the must-fix tier even with "execute everything"**. P3 is fixed opportunistically when a file is touched for P0/P1/P2. Otherwise we'd ship hundreds of low-value PRs.

### 4.3 Manual narrative diffs

AST can't see runtime behavior. These 5–10 modules get a 1-paragraph narrative diff appended to `REPORT.md`:

1. **NLSQ fallback chain ordering** (`optimization/nlsq/fallback_chain.py`) — strategy retry order
2. **CMC sampler loop sequencing** (`optimization/cmc/sampler.py`) — retry-on-divergence semantics, warmup ramp
3. **CLI dispatcher** (`cli/commands.py`) — subcommand routing precedence
4. **Anti-degeneracy controller orchestration** (`optimization/nlsq/anti_degeneracy_controller.py`) — layer invocation order, trigger conditions
5. **Recovery tactics** (`optimization/nlsq/recovery.py`) — action ordering
6. **NLSQ → CMC warm-start handoff** — what data flows in what shape (especially relevant after `warmstart.py` is absorbed into `priors.py`)
7. **Data pipeline gating** (`cli/data_pipeline.py`) — validation order, fail-fast conditions

### 4.4 Audit artifact location

```
heterodyne/
├── tools/parity_audit/                                # NEW
│   ├── __init__.py
│   ├── extract_signatures.py
│   ├── extract_exports.py
│   ├── extract_classes.py
│   ├── extract_configs.py
│   ├── extract_cli.py
│   ├── extract_logs_errors.py
│   ├── extract_docs.py
│   ├── extract_file_inventory.py
│   ├── diff_extracts.py
│   └── README.md
└── docs/superpowers/audits/                           # NEW
    └── 2026-05-18-parity-audit/
        ├── homodyne_sha.txt
        ├── heterodyne_sha.txt
        ├── extracts/                                  # raw JSON per extractor, per package
        ├── REPORT.md                                  # categorized human-readable diff
        ├── DISPOSITIONS.md                            # Phase 2 output
        └── DIVERGENCE_REGISTRY.md                     # final list of accepted divergences
```

## 5. Phase 2 — Review gate

User walks `REPORT.md` row-by-row and assigns a disposition token:

| Token | Meaning |
|---|---|
| `KEEP` | Fix in Phase 4 (default) |
| `DROP` | Heterodyne stays as-is; logged in `DIVERGENCE_REGISTRY.md` with one-line rationale |
| `DEFER` | Backlog. Tracked but not in this work |
| `WAIVE` | Already covered by another row; skip |

Output → `DISPOSITIONS.md`. To make review cheap, all rows default to `KEEP` — user only overrides exceptions.

## 6. Phase 3 — Plan generation

Invoke `superpowers:writing-plans` against `KEEP` rows. Output organized by subpackage × severity:

```
PR 1: NLSQ structural alignment (P1)
PR 2: NLSQ signature drift (P0)
PR 3: CMC structural alignment (P1)
  ├─ Rename multiprocessing_backend.py → multiprocessing.py + all imports
  ├─ Rename pjit_backend.py → pjit.py + all imports
  ├─ Decide cpu_backend.py: merge into multiprocessing.py or delete
  ├─ Absorb warmstart.py into priors.py
  ├─ Absorb prior_builder.py into priors.py
  ├─ Revert CMC config attribute names (target_accept_prob → target_accept, etc.)
  ├─ Remove from_dict() legacy-key shim
  └─ Update CLAUDE.md (delete lines 21–25)
PR 4: CMC signature/config drift (P0)
PR 5: CLI parity (P0/P1)
PR 6: Viz parity (P1/P2)
PR 7: Data/IO/Utils/Device parity (P1/P2)
PR 8: Logs and error messages (P2)
PR 9: Docs structural fill (P1) — port missing pages from homodyne, adapt for 14-param model + no-shear
PR 10: Docs autodoc + xref fixup (P0/P2) — make `make docs` warning-free
PR N: P3 cleanup (opportunistic, no standalone PR)
```

Each PR is independently mergeable, ships with regression tests, and re-runs the audit extractors to verify the targeted gap category dropped to zero.

## 7. Phase 4 — Execution rules

### Rule 1: Absorb-then-delete (no orphan deletions)

For every heterodyne-only file slated for removal:

1. Identify the homodyne file that owns the equivalent functionality (e.g., `priors.py` owns warm-start in homodyne).
2. Port the heterodyne-only functionality into that homodyne-named file as a new function/method.
3. Replace all heterodyne call sites to use the new location.
4. Run full test suite — must be green.
5. Only then delete the heterodyne-only file.
6. Re-run tests — still green.

This protects recent T1–T4 / 19-finding-closeout stability work. The 438-test CMC suite must remain green throughout.

### Rule 2: Re-run audit per PR

Every PR's CI step runs `python -m tools.parity_audit.diff_extracts` and asserts:

- The targeted gap category dropped to 0
- No previously-closed gap reopened

PR cannot merge if it didn't close what it claimed to close, or if it reopened something else.

### Rule 3: TDD for new ports

Genuine ports (e.g., `optimization/nlsq/wrapper.py` if NLSQWrapper is dispositioned `KEEP`) follow `superpowers:test-driven-development`: write the regression test first against the homodyne-equivalent behavior (adapted to 14 params), watch it fail, then port.

### Rule 4: Docs ride with code

Docs updates ship in the same PR as the code change that caused the drift. The `multiprocessing_backend.py` → `multiprocessing.py` rename PR also updates every `automodule::` directive, every `:class:` ref, and every prose mention. No "docs cleanup" PR — docs build never breaks on `main`.

## 8. Cross-cutting concerns

| Concern | Handling |
|---|---|
| CMC backend renames break worker spawn-mode imports | Rule 1 absorb-then-delete + full test suite includes multiprocessing/PBS spawn tests |
| `__all__` reordering creates noise diffs | Extractor sorts `__all__`; only set membership matters |
| Heterodyne param count (14) differs from homodyne | Registry **content** diverges (explicit physics exemption); registry **structure and consumer API** must still mirror |
| CLI flag drift may break user shell history | P0 rename ships with one-release deprecation alias, then removed in follow-up |
| NUTS warmup-1500 default may conflict with homodyne default | Per-row disposition decides; if `DROP`, logged in `DIVERGENCE_REGISTRY.md` as physics-required content divergence under parameter-registry exemption |

## 9. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Reverting CMC attr names + warmup floor regresses recent stability work (438 CMC tests) | Absorb-then-delete + full test suite after every step. Regressed-but-accepted tests documented in `DIVERGENCE_REGISTRY.md` |
| R2 | Audit extractors miss runtime behavior | 5–10 manual narrative diffs (§4.3) |
| R3 | Homodyne drifts mid-execution | `homodyne_sha.txt` pin. Newer homodyne is a future audit |
| R4 | CMC backend renames break PBS / pjit spawn-mode workers | Rule 1 + full spawn-mode test suite |
| R5 | Aggressive deletion strands functionality | Phase 2 review gate — user marks `DROP`/`DEFER` on anything they want to keep |
| R6 | NUTS warmup revert causes CMC sampler instability | Audit row carries `WARN: physics-required value` tag; user dispositions per-row |
| R7 | Audit re-run produces noisy churn | Extractors normalize: sort `__all__`, ignore whitespace, ignore docstring trivia at P3 |
| R8 | Tests break in non-obvious ways during absorb-then-delete | Rule 1's 6-step sequence requires green tests at each step; bisect is small |
| R9 | Docs build regressions go unnoticed | New CI gate runs `make docs` with `-W` (warnings-as-errors); fails build on any Sphinx warning |

## 10. Testing strategy

### Per PR (defined in Phase 3 plan, enforced in Phase 4)

1. **Full test suite stays green:** `make test` (~3000 tests). No `--no-verify`.
2. **Targeted gap regression:** Re-run audit extractor for the category this PR closes; assert it dropped to 0.
3. **TDD for ports:** New code ships with characterization tests written before implementation.
4. **PBS / multiprocessing spawn-mode tests run:** `tests/regression/test_cmc_*` + any spawn-mode PBS test.
5. **Docs build:** `uv run sphinx-build -W docs/source docs/build/html` succeeds (warnings-as-errors).

### Cross-PR (post-merge to main)

6. **Full audit re-run after each PR merge:** Output posted as comment on the next PR; user watches gap count trend monotonically downward.
7. **CI gate `parity-audit-no-regression.yml`:** Runs extractors on every PR; fails if any closed gap reopens. Makes parity a forever-property.

## 11. Success criteria

The work is complete when:

1. `python -m tools.parity_audit.diff_extracts` reports **zero P0, zero P1, zero P2** gaps (excluding `DROP`/`DEFER` rows in `DISPOSITIONS.md`).
2. `DIVERGENCE_REGISTRY.md` contains exactly the dispositioned `DROP` rows, each with one-line rationale (parameter-registry-exempt rows like NUTS warmup if kept).
3. Full test suite green (~3000 tests).
4. CI gate `parity-audit-no-regression.yml` is active.
5. Five manual-narrative modules have "no behavior drift" sign-off in `REPORT.md`.
6. `make docs` completes with **zero warnings**, all autodoc targets resolve, all cross-references resolve.
7. Heterodyne docs file count matches homodyne (minus any pages dispositioned as `DROP` for being physics-formula-specific).

## 12. Out of scope (explicitly)

| Out of scope | Why |
|---|---|
| Physics formula body parity | Narrow physics exemption |
| Shear-sensitivity weighting | Homodyne-only by the brief |
| Homodyne back-ports of heterodyne enhancements | Strict 1:1, not bidirectional |
| Performance benchmarking | Mirror is structural; perf is separate |
| `graphify-out/` | Generated artifact |
| `docs/build/`, `docs/_build/`, `docs/_autosummary/` | Generated; gitignored or regenerated |
| `docs/superpowers/` | Project management, not public docs |
| Test scaffolding/fixtures (factories/, conftest.py) | Tests must pass; infrastructure can diverge |

## 13. Artifact summary

| Phase | Artifact | Path |
|---|---|---|
| Design | This spec | `docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md` |
| Phase 1 | Audit report | `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md` |
| Phase 1 | Extractor outputs | `docs/superpowers/audits/2026-05-18-parity-audit/extracts/` |
| Phase 1 | SHA pins | `docs/superpowers/audits/2026-05-18-parity-audit/{homodyne,heterodyne}_sha.txt` |
| Phase 1 | Audit tooling | `tools/parity_audit/` |
| Phase 2 | Dispositions | `docs/superpowers/audits/2026-05-18-parity-audit/DISPOSITIONS.md` |
| Phase 2 | Divergence registry | `docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md` |
| Phase 3 | Implementation plan | `docs/superpowers/plans/2026-05-18-heterodyne-homodyne-parity-plan.md` |
| Phase 4 | Merged PRs + CI gate | `.github/workflows/parity-audit-no-regression.yml`; merged into `main` |
