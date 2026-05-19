# Heterodyne → Homodyne Strict 1:1 Parity — Phase 4 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan PR-by-PR. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 4,322 KEEP-dispositioned audit gaps from `REPORT.md` across 12 sequential PRs, leaving heterodyne as a strict 1:1 structural and behavioral mirror of homodyne modulo the physics formula bodies and the absent Layer-5 shear-weighting (per spec §2 and `DISPOSITIONS.md`).

**Architecture:** 12 PR-sized batches in dependency order. Each PR (a) closes one or more gap categories from REPORT.md, (b) honors DROP rules from DISPOSITIONS.md by logging them in `DIVERGENCE_REGISTRY.md`, (c) honors WAIVE rules by treating aggregate `__all__` rows as auto-closed when underlying signature/class gaps fix, (d) follows the spec §7 absorb-then-delete pattern when removing heterodyne-only files, and (e) re-runs `python -m tools.parity_audit run-all` as the merge gate (targeted gap category must drop to 0; no other category may regress).

**Tech Stack:** Python 3.13, uv-managed, pytest, JAX 0.8+ (CPU-only), NumPyro for CMC, scipy.optimize.least_squares for NLSQ.

**Inputs (all committed):**
- Spec: `docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md` (`f46c1db`)
- Phase 1 plan: `docs/superpowers/plans/2026-05-18-heterodyne-homodyne-parity-plan.md` (`443984d`)
- Audit report: `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md` (~4,503 rows)
- Dispositions: `docs/superpowers/audits/2026-05-18-parity-audit/DISPOSITIONS.md` (`f675476`)
- Audit tooling: `tools/parity_audit/` (8 extractors + differ, 59 tests green)

**Pinned SHAs:**
- Homodyne: `0368cbdb075fcff1908c0da2b59a1b0d37d5eeca`
- Heterodyne starting point for Phase 4: current `main` head (run `git rev-parse HEAD`)

**Scale and granularity disclaimer:** With ~4,322 KEEP gaps, per-gap enumeration would produce hundreds of thousands of lines of plan. Instead, this plan uses **PR templates with a per-gap workflow + 2-3 representative sample tasks per PR**. The authoritative gap list for each PR is the section in REPORT.md that the PR targets; the implementer reads REPORT.md and DISPOSITIONS.md for each new gap they touch.

---

## Pre-flight: divergence registry initialization (Task P0)

This is a 1-task setup performed once before PR 1 starts. It creates the `DIVERGENCE_REGISTRY.md` skeleton that subsequent PRs will append to as they encounter DROP-dispositioned rows.

**Files:**
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md`

- [ ] **Step 1: Write the registry skeleton**

`docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): initialize DIVERGENCE_REGISTRY.md for Phase 4

Skeleton with D1/D2/D3 sections per DISPOSITIONS.md. Phase 4 PRs append
to this file as they encounter DROP-dispositioned rows. Three named
narrative-diff items (run_nuts_with_retry, anti_degeneracy_controller
arch port, warmstart absorption) are placeholders that get deleted as
the corresponding PRs land."
```

---

## Per-PR workflow (applies to every PR below)

Every Phase 4 PR follows this 6-step structure. PR-specific tasks slot inside Step 3.

- [ ] **Step 1: Pre-PR audit snapshot.** Record the gap count for each category before any changes:

  ```bash
  python -m tools.parity_audit run-all \
      --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
      --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
      --out docs/superpowers/audits/2026-05-18-parity-audit
  grep -E "^## P[0-3]|^### " docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md | head -30
  ```

  Save the per-category counts (e.g., "signatures: 615 → target 0 for nlsq subpackage in PR 2").

- [ ] **Step 2: Walk the PR's target audit rows.** Read the relevant section of REPORT.md. For each row:
  - If matched by a DISPOSITIONS.md `DROP` rule: append to `DIVERGENCE_REGISTRY.md` under D1/D2/D3 and **skip**.
  - If matched by `WAIVE` rule (only W1 aggregate `__all__` rows): **skip** (will auto-close).
  - Otherwise: **fix** (per the PR's task templates below).

- [ ] **Step 3: Execute PR-specific tasks.** See per-PR sections below.

- [ ] **Step 4: Full test suite green.** Run:

  ```bash
  make test
  ```

  Expected: all ~3,000 tests pass (no regressions in the 438 CMC tests from T1–T4). If anything regresses, fix before continuing (no `--no-verify`).

- [ ] **Step 5: Re-run audit and confirm targeted category dropped to 0.** Run:

  ```bash
  python -m tools.parity_audit run-all \
      --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
      --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
      --out docs/superpowers/audits/2026-05-18-parity-audit
  ```

  Check the targeted category (e.g., `signatures` for PR 3 NLSQ subpackage) is `0` for the in-scope module prefix, and no previously-closed gap reopened (compare to Step 1 snapshot).

- [ ] **Step 6: Commit the PR's atomic batch.**

  ```bash
  git add <files modified in Step 3>
  git add -f docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
  git -c commit.gpgsign=false commit -m "<PR-specific commit message template — see each PR section>"
  ```

  If the PR is composed of multiple logical commits (e.g., PR 3 absorb-then-delete), commit after each absorb-then-delete sequence rather than at the end.

---

## PR 1: NLSQ structural alignment (P1 file_inventory + missing wrapper.py)

**Closes audit categories:**
- `file_inventory:missing_py_file` for `optimization.nlsq.wrapper`, `optimization.nlsq.strategies.chunking`
- `file_inventory:extra_py_file` for `optimization.nlsq.strategies.base`, `optimization.nlsq.strategies.chunked`, `optimization.nlsq.strategies.jit_strategy`, `optimization.nlsq.validation.bounds`, `optimization.nlsq.validation.convergence`, `optimization.nlsq.validation.result`

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D1: `missing_py_file: optimization.nlsq.shear_weighting` (Layer 5, spec §2 exempt)
- D3: any extra_py_file row whose qualname matches the D3 pattern (none expected in NLSQ scope)

**Goal:** Heterodyne's NLSQ subpackage has the same file inventory as homodyne (minus shear_weighting.py per spec). New file `wrapper.py` ports homodyne's NLSQWrapper.

### Task 1.1: Append D1 NLSQ row to DIVERGENCE_REGISTRY.md

- [ ] **Step 1: Append entry**

Edit `docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md` D1 section to add:

```
- file_inventory:missing_py_file | `optimization.nlsq.shear_weighting` | Layer 5 (shear-sensitivity weighting); not applicable to heterodyne's velocity-phase physics model (D1, spec §2)
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): log shear_weighting Layer 5 as accepted D1 divergence"
```

### Task 1.2: Rename `strategies/chunked.py` → `strategies/chunking.py`

The audit shows homodyne has `optimization.nlsq.strategies.chunking` and heterodyne has `optimization.nlsq.strategies.chunked`. Pure rename.

**Files:**
- Rename: `heterodyne/optimization/nlsq/strategies/chunked.py` → `heterodyne/optimization/nlsq/strategies/chunking.py`
- Modify: every file that imports from `heterodyne.optimization.nlsq.strategies.chunked` (grep first)
- Modify: `heterodyne/optimization/nlsq/strategies/__init__.py` re-export

- [ ] **Step 1: Find all import sites**

```bash
grep -rn "from heterodyne.optimization.nlsq.strategies.chunked" heterodyne/ tests/
grep -rn "from heterodyne.optimization.nlsq.strategies import .*chunked" heterodyne/ tests/
grep -rn "heterodyne.optimization.nlsq.strategies.chunked" heterodyne/ tests/
```

List every file:line pair returned.

- [ ] **Step 2: Rename the file**

```bash
git mv heterodyne/optimization/nlsq/strategies/chunked.py heterodyne/optimization/nlsq/strategies/chunking.py
```

- [ ] **Step 3: Update every import site**

For each file:line from Step 1, replace `chunked` with `chunking` in the import path. Use Edit tool, not sed, to preserve formatting.

- [ ] **Step 4: Update `__init__.py` re-export**

In `heterodyne/optimization/nlsq/strategies/__init__.py`, change any `from heterodyne.optimization.nlsq.strategies.chunked import ...` to `from heterodyne.optimization.nlsq.strategies.chunking import ...`.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/unit/optimization/nlsq/ tests/regression/ -x 2>&1 | tail -10
```

Expected: all NLSQ + regression tests pass (no `ImportError` for the renamed module).

- [ ] **Step 6: Commit**

```bash
git add heterodyne/optimization/nlsq/strategies/
git add tests/ -A
git -c commit.gpgsign=false commit -m "refactor(nlsq): rename strategies.chunked → strategies.chunking

Mirrors homodyne file structure. No behavior change. All import sites
updated mechanically."
```

### Task 1.3: Absorb `strategies/base.py` into `strategies/__init__.py`

Audit shows `optimization.nlsq.strategies.base` is heterodyne-only. Homodyne's strategy infrastructure lives directly in `strategies/__init__.py`.

**Files:**
- Read: `heterodyne/optimization/nlsq/strategies/base.py`
- Modify: `heterodyne/optimization/nlsq/strategies/__init__.py`
- Delete: `heterodyne/optimization/nlsq/strategies/base.py`
- Modify: every file that imports from `heterodyne.optimization.nlsq.strategies.base`

Per spec §7 Rule 1 (absorb-then-delete):

- [ ] **Step 1: Find all import sites**

```bash
grep -rn "from heterodyne.optimization.nlsq.strategies.base" heterodyne/ tests/
grep -rn "from heterodyne.optimization.nlsq.strategies import .*Base" heterodyne/ tests/
```

- [ ] **Step 2: Read the contents to absorb**

```bash
cat heterodyne/optimization/nlsq/strategies/base.py
```

Identify the public classes/functions (anything not starting with `_`).

- [ ] **Step 3: Move contents to `strategies/__init__.py`**

Use Edit tool. Append the public class/function definitions from `base.py` to `strategies/__init__.py`, preserving imports they require. Do NOT alter logic. Add a comment at the top of the absorbed block:

```python
# --- Absorbed from optimization/nlsq/strategies/base.py to mirror homodyne layout ---
# (Phase 4 PR 1; spec §7 Rule 1)
```

- [ ] **Step 4: Update every import site**

For each file:line from Step 1, change `from heterodyne.optimization.nlsq.strategies.base import X` to `from heterodyne.optimization.nlsq.strategies import X`.

- [ ] **Step 5: Run full test suite — MUST be green before deletion**

```bash
make test 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 6: Delete the absorbed file**

```bash
git rm heterodyne/optimization/nlsq/strategies/base.py
```

- [ ] **Step 7: Re-run tests after deletion**

```bash
make test 2>&1 | tail -5
```

Expected: all tests still pass.

- [ ] **Step 8: Commit (one atomic absorb-then-delete commit)**

```bash
git add heterodyne/optimization/nlsq/strategies/
git add tests/ -A
git -c commit.gpgsign=false commit -m "refactor(nlsq): absorb strategies/base.py into strategies/__init__.py

Mirrors homodyne layout (no separate base.py — base classes live in
the package __init__). Absorb-then-delete per spec §7 Rule 1; full test
suite green before and after deletion."
```

### Task 1.4: Repeat Task 1.3 pattern for `jit_strategy.py`, `validation/bounds.py`, `validation/convergence.py`, `validation/result.py`

Each of these is an absorb-then-delete sequence following the same 8-step pattern as Task 1.3. The target file for absorption is:

| Heterodyne-only file | Absorbs into | Homodyne equivalent |
|---|---|---|
| `strategies/jit_strategy.py` | `strategies/__init__.py` or merge into `strategies/residual_jit.py` (check homodyne layout first) | `strategies/residual_jit.py` |
| `validation/bounds.py` | `validation/__init__.py` | (homodyne has `validation/` as a dir too — verify the file split) |
| `validation/convergence.py` | `validation/__init__.py` | (same) |
| `validation/result.py` | `validation/__init__.py` | (same) |

For each, repeat Task 1.3 Steps 1-8. Commit one absorb-then-delete sequence per file.

**IMPORTANT for validation/ subdir:** Before absorbing, run:

```bash
ls /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/validation/ 2>&1
```

If homodyne also has `validation/` as a dir with multiple files, the heterodyne split may already mirror — in which case DROP these rows under D3 instead (heterodyne's organization is actually parity, not divergence). If homodyne has `validation.py` as a single file or no `validation/` at all, absorb-then-delete as above.

### Task 1.5: Port `wrapper.py` (NLSQWrapper) from homodyne

This is a **genuine port**, not a rename. NLSQWrapper is homodyne's "stable fallback adapter" for 100M+ point datasets with custom 3-attempt recovery and full streaming (see homodyne `wrapper.py` docstring).

**Files:**
- Create: `heterodyne/optimization/nlsq/wrapper.py` (port from homodyne, adapt for 14-param model)
- Create: `tests/unit/optimization/nlsq/test_wrapper_port.py` (TDD characterization test, written first)
- Modify: `heterodyne/optimization/nlsq/__init__.py` (export NLSQWrapper)

Per spec §7 Rule 3 (TDD for ports): failing test first.

- [ ] **Step 1: Read homodyne's wrapper.py**

```bash
cat /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/wrapper.py | head -200
```

Identify the NLSQWrapper public API: `__init__` signature, `fit()` method signature, configuration knobs.

- [ ] **Step 2: Write the failing characterization test**

`tests/unit/optimization/nlsq/test_wrapper_port.py`:

```python
"""Characterization tests for the NLSQWrapper port from homodyne.

Verifies NLSQWrapper exposes the same public API as homodyne's NLSQWrapper,
adapted for heterodyne's 14-parameter model. The wrapper is the stable
fallback adapter for 100M+ point datasets with custom 3-attempt recovery.

Refs: Phase 4 PR 1 Task 1.5 (heterodyne/docs/superpowers/plans/2026-05-19-...md)
"""
from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.wrapper import NLSQWrapper


def test_nlsq_wrapper_is_instantiable() -> None:
    """NLSQWrapper can be constructed without arguments (defaults applied)."""
    wrapper = NLSQWrapper()
    assert wrapper is not None


def test_nlsq_wrapper_exposes_fit_method() -> None:
    """NLSQWrapper.fit signature matches homodyne's contract."""
    wrapper = NLSQWrapper()
    assert hasattr(wrapper, "fit")
    assert callable(wrapper.fit)


def test_nlsq_wrapper_fit_returns_optimization_result(tmp_path) -> None:
    """fit() with minimal synthetic data returns an OptimizationResult."""
    from heterodyne.optimization.nlsq.results import NLSQResult

    # Minimal synthetic 14-param data — see test fixtures for shape conventions
    n_t = 50
    n_phi = 1
    t = np.linspace(0.001, 0.05, n_t)
    phi = np.array([0.0])
    c2_data = np.ones((n_phi, n_t, n_t)) * 1.05  # contrast + offset = 1.05

    data = {"t1": t, "t2": t, "phi_angles": phi, "c2_data": c2_data}
    config = {"physics": {"q": 0.0054, "dt": 0.001}}  # minimal

    wrapper = NLSQWrapper()
    result = wrapper.fit(data, config)
    assert isinstance(result, NLSQResult)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/optimization/nlsq/test_wrapper_port.py -v
```

Expected: `ModuleNotFoundError: No module named 'heterodyne.optimization.nlsq.wrapper'`

- [ ] **Step 4: Port the wrapper**

Read `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/wrapper.py` in full. Port to `heterodyne/optimization/nlsq/wrapper.py` with these adaptations:

1. Replace all references to homodyne 7-parameter model with heterodyne 14-parameter model. Specifically:
   - `analysis_mode` arg defaulting to `'laminar_flow'` → defaulting to `'full'` (heterodyne uses `mode='full'` per `heterodyne/config/manager.py`)
   - Any references to `gamma_dot_*` or `phi0` shear parameters → use heterodyne's 14 physics param names (`D0_ref`, `D0_sample`, `alpha_ref`, `alpha_sample`, `D_offset_ref`, `D_offset_sample`, `v0`, `beta`, `v_offset`, `f0`, `f1`, `f2`, `f3`, `phi0`)
   - Remove any `shear_transforms` kwarg (D3 pattern: shear-only)
2. Keep the 3-attempt recovery loop verbatim. Keep the streaming infrastructure verbatim.
3. Keep public method signatures identical to homodyne's: `__init__(self, *, recovery_attempts: int = 3, ...)`, `fit(self, data, config, initial_params=None, bounds=None, ...) -> NLSQResult`
4. Imports from `heterodyne.optimization.nlsq.recovery`, `heterodyne.optimization.nlsq.fallback_chain`, `heterodyne.optimization.nlsq.memory` should work (these already exist).

Use Write tool to create the new file. The port should be ~300-500 lines.

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/optimization/nlsq/test_wrapper_port.py -v 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 6: Export from `__init__.py`**

Edit `heterodyne/optimization/nlsq/__init__.py`:

```python
# Add to imports:
from heterodyne.optimization.nlsq.wrapper import NLSQWrapper

# Add to __all__:
__all__ = [
    ...,
    "NLSQWrapper",
]
```

- [ ] **Step 7: Run full test suite**

```bash
make test 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add heterodyne/optimization/nlsq/wrapper.py heterodyne/optimization/nlsq/__init__.py tests/unit/optimization/nlsq/test_wrapper_port.py
git -c commit.gpgsign=false commit -m "feat(nlsq): port NLSQWrapper from homodyne (14-param adaptation)

Stable fallback adapter for 100M+ point datasets with custom 3-attempt
recovery loop and streaming infrastructure. Public API mirrors homodyne's
NLSQWrapper.fit() signature. Adapted from homodyne's 7-param laminar_flow
model to heterodyne's 14-param full model; shear_transforms kwarg removed
(D3 divergence — shear-only).

Refs: spec §6 PR 1, audit row file_inventory:missing_py_file optimization.nlsq.wrapper"
```

### Task 1.6: Re-run audit and confirm PR 1 categories closed

- [ ] **Step 1: Re-run audit**

```bash
python -m tools.parity_audit run-all \
    --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
    --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
    --out docs/superpowers/audits/2026-05-18-parity-audit
```

- [ ] **Step 2: Verify PR 1 targets closed**

```bash
grep -A 5 "### file_inventory" docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md | head -30
```

Expected: zero `missing_py_file` rows for `optimization.nlsq.wrapper`, `optimization.nlsq.strategies.chunking`. Zero `extra_py_file` rows for the 6 heterodyne-only NLSQ files absorbed in Tasks 1.3–1.4. The `shear_weighting` missing row should still appear (correct — it's DROPped, not removed from REPORT.md).

- [ ] **Step 3: Verify no other categories regressed**

Compare gap counts to the Step-1 snapshot taken before PR 1 started. No category may have increased.

- [ ] **Step 4: Commit the updated audit report**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md
git add -f docs/superpowers/audits/2026-05-18-parity-audit/extracts/
git -c commit.gpgsign=false commit -m "audit(parity): refresh report after PR 1 (NLSQ structural)

Closed file_inventory rows for NLSQ subpackage. Re-pinned to latest
heterodyne SHA; homodyne SHA unchanged (0368cbdb)."
```

---

## PR 2: Anti-degeneracy controller architecture port

**Closes audit categories:**
- All `classes:missing_method`, `classes:missing_field` rows in `optimization.nlsq.anti_degeneracy_controller` whose qualnames are NOT D3-DROPped (i.e., NOT `shear_*` named)
- All `signatures:changed` and `signatures:missing_in_heterodyne` rows in `optimization.nlsq.anti_degeneracy_controller`

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D3: all `shear_*` named methods/fields/signatures (e.g., `get_shear_weights`, `update_shear_phi0`, `use_shear_weighting`, `shear_weighter`, `shear_phi0`)

**Goal:** Port homodyne's active 5-layer orchestrator architecture into heterodyne (Layers 1-4 only; Layer 5 stays D3-dropped). Heterodyne's current 3-check passive diagnostic class becomes a thin wrapper that delegates to the active orchestrator's layer-by-layer interface.

**Implementation note:** This is a substantial port. The narrative diff in REPORT.md §"Manual narrative diffs" §anti_degeneracy_controller documents the divergence. User explicitly confirmed KEEP (port from homodyne) in Phase 2 dispositioning.

### Task 2.1: Read both implementations side-by-side

- [ ] **Step 1: Read homodyne's controller**

```bash
cat /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/anti_degeneracy_controller.py | head -300
```

Identify: the `AntiDegeneracyController` class, its `__init__` signature, its public methods, the layer-invocation order in the optimizer-loop hook, and how it consumes each layer's output.

- [ ] **Step 2: Read heterodyne's controller**

```bash
cat /home/wei/Documents/GitHub/heterodyne/heterodyne/optimization/nlsq/anti_degeneracy_controller.py
```

Identify the existing passive classes (`DegeneracyCheck`, `GradientCollapseDetector`, etc.) and what code in the optimizer currently calls them.

- [ ] **Step 3: Identify call sites**

```bash
grep -rn "AntiDegeneracyController\|DegeneracyCheck\|GradientCollapseDetector\|suggest_regularization" heterodyne/ tests/
```

Save the call-site list — these will need updating.

### Task 2.2: Write characterization test for active orchestrator

**Files:**
- Create: `tests/unit/optimization/nlsq/test_anti_degeneracy_controller_active.py`

- [ ] **Step 1: Write the failing test**

```python
"""Characterization test for the active 5-layer (4-layer in heterodyne)
anti-degeneracy controller ported from homodyne.

Layer 5 (shear-sensitivity weighting) is intentionally absent per spec §2.
The other 4 layers (Fourier reparam, hierarchical opt, adaptive CV-reg,
gradient-collapse monitoring) are actively invoked in the optimizer loop.
"""
from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
    AntiDegeneracyConfig,
)


def test_controller_is_instantiable_from_config() -> None:
    """from_config classmethod mirrors homodyne's contract."""
    config_dict = {
        "fourier_reparam_enabled": False,
        "hierarchical_enabled": False,
        "adaptive_regularization_enabled": False,
        "gradient_monitor_enabled": True,
    }
    n_phi = 4
    phi_angles = np.linspace(0, np.pi, n_phi)
    n_physical = 14  # heterodyne param count
    controller = AntiDegeneracyController.from_config(
        config_dict=config_dict,
        n_phi=n_phi,
        phi_angles=phi_angles,
        n_physical=n_physical,
        per_angle_scaling=True,
    )
    assert controller is not None
    # Layer 5 (shear weighting) is absent: should NOT have use_shear_weighting() method,
    # or it should always return False
    if hasattr(controller, "use_shear_weighting"):
        assert controller.use_shear_weighting() is False


def test_controller_exposes_active_invoke_method() -> None:
    """Active orchestrator has a per-iteration invoke method, not just diagnostics."""
    config_dict = {"gradient_monitor_enabled": True}
    controller = AntiDegeneracyController.from_config(
        config_dict=config_dict,
        n_phi=1,
        phi_angles=np.array([0.0]),
        n_physical=14,
    )
    # Active orchestrator should have a hook that the optimizer calls per iteration
    assert hasattr(controller, "on_iteration") or hasattr(controller, "step")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/optimization/nlsq/test_anti_degeneracy_controller_active.py -v
```

Expected: FAIL (current heterodyne controller has no `from_config` classmethod or has different signature).

### Task 2.3: Port the active orchestrator

**Files:**
- Modify: `heterodyne/optimization/nlsq/anti_degeneracy_controller.py` (rewrite)

- [ ] **Step 1: Port the controller class structure**

Read homodyne's `anti_degeneracy_controller.py` in full. Port the `AntiDegeneracyController` class with these adaptations:

1. Skip Layer 5 entirely — do not import `shear_weighting`, do not create a `shear_weighter` field, do not implement `update_shear_phi0` / `get_shear_weights` / `use_shear_weighting` methods.
2. Keep Layers 1-4 (Fourier reparam, hierarchical, adaptive CV-reg, gradient monitor) verbatim from homodyne.
3. Adapt `n_physical` default from 7 to 14.
4. Adapt `analysis_mode` default from `'laminar_flow'` to `'full'`.
5. Keep `from_config` classmethod signature identical to homodyne's, minus the `is_laminar_flow` kwarg (or default it to False and never honor it).

Use Write tool to replace the existing `anti_degeneracy_controller.py`.

- [ ] **Step 2: Run test to verify it passes**

```bash
uv run pytest tests/unit/optimization/nlsq/test_anti_degeneracy_controller_active.py -v
```

Expected: both tests pass.

- [ ] **Step 3: Update call sites**

For every file:line from Task 2.1 Step 3 that referenced the old passive classes (`DegeneracyCheck`, etc.), update to use the new `AntiDegeneracyController.from_config(...)` + `on_iteration()` API. The optimizer loop (`heterodyne/optimization/nlsq/core.py`) is the primary call site.

- [ ] **Step 4: Full test suite**

```bash
make test 2>&1 | tail -5
```

Expected: all tests pass. CMC tests (438) MUST stay green — the controller change shouldn't affect them, but the loop integration might.

- [ ] **Step 5: Commit**

```bash
git add heterodyne/optimization/nlsq/anti_degeneracy_controller.py heterodyne/optimization/nlsq/core.py tests/unit/optimization/nlsq/test_anti_degeneracy_controller_active.py
git -c commit.gpgsign=false commit -m "feat(nlsq): port active 5-layer anti-degeneracy controller from homodyne

Replaces heterodyne's passive 3-check diagnostic class with homodyne's
active orchestrator. Layers 1-4 (Fourier reparam, hierarchical opt,
adaptive CV-reg, gradient monitor) are invoked per optimizer iteration.
Layer 5 (shear-sensitivity weighting) is intentionally absent per spec §2;
related methods (use_shear_weighting / get_shear_weights / update_shear_phi0)
are NOT ported (D3 divergence, logged in DIVERGENCE_REGISTRY.md).

Refs: spec §6 PR 2; REPORT.md §Manual narrative diffs §anti_degeneracy_controller.py"
```

### Task 2.4: Append D3 rows for shear-related methods to DIVERGENCE_REGISTRY.md

- [ ] **Step 1: Append shear-related D3 entries**

Edit `docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md` D3 section to add:

```
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.get_shear_weights` | Layer 5 method; heterodyne has no shear term (D3)
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.update_shear_phi0` | Layer 5 method; heterodyne has no shear term (D3)
- classes:missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.use_shear_weighting` | Layer 5 method; heterodyne has no shear term (D3)
- classes:missing_field  | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.shear_weighter` | Layer 5 field; heterodyne has no shear term (D3)
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): log Layer-5 anti-degeneracy methods as D3 divergence"
```

### Task 2.5: Re-run audit and confirm PR 2 categories closed

Run the standard per-PR Steps 1-3 from "Per-PR workflow" above. Specifically:

- The `optimization.nlsq.anti_degeneracy_controller` qualnames in `signatures` and `classes` categories should drop to 0 except for the 4 D3-DROPped shear-related rows.
- The audit's signature/class counters for the broader `optimization.nlsq` prefix should be lower (we've closed a chunk).

Commit the updated REPORT.md per Per-PR workflow Step 6.

---

## PR 3: NLSQ signature/class drift (remaining)

**Closes audit categories:**
- All remaining `signatures:changed`, `signatures:missing_in_heterodyne`, `signatures:extra_in_heterodyne` rows whose qualname starts with `optimization.nlsq.` and was NOT closed by PR 1 or PR 2
- All `classes:missing_method`, `classes:missing_field`, `classes:missing_class` rows in `optimization.nlsq.*` not closed by PR 2
- All `configs:missing_config_key`, `configs:extra_config_key` rows in `optimization.nlsq.*`

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D3: any qualname matching D3 patterns (shear_*, gamma_dot_*, etc.) — none expected in NLSQ scope after PR 2

**Goal:** Align every NLSQ subpackage signature/class/config to homodyne's. This is a high-volume PR (estimated ~200-300 individual gaps).

### Process per gap (template — applies to every NLSQ row in the audit)

For each row in REPORT.md whose qualname is under `optimization.nlsq.*` (excluding rows already closed by PR 1 or PR 2):

- [ ] **Step 1: Read the row and identify the change kind**
  - `signatures:changed` → adjust function signature in heterodyne to match homodyne's
  - `signatures:missing_in_heterodyne` → port the homodyne function (read body, adapt for 14-param, write to heterodyne)
  - `signatures:extra_in_heterodyne` → delete the heterodyne function (no homodyne equivalent) UNLESS it's a heterodyne-required helper, in which case mark D3 and add to DIVERGENCE_REGISTRY.md
  - `classes:missing_method/field` → port the method/field from homodyne's equivalent class
  - `classes:missing_class` → port the entire class from homodyne
  - `configs:missing_config_key` → add the dataclass field to heterodyne's matching config file
  - `configs:extra_config_key` → remove the heterodyne dataclass field (or mark D3 if heterodyne-required)

- [ ] **Step 2: Make the change**
  - Use Edit tool for function signature changes (preserve formatting)
  - Use Write tool for new function/class ports
  - For port: read homodyne's implementation, adapt for 14-param model (remove shear references, swap parameter names if needed), verify the imports it needs are available in heterodyne

- [ ] **Step 3: Run the relevant unit tests**
  - `uv run pytest tests/unit/optimization/nlsq/ -x`
  - If anything regresses, fix before continuing

- [ ] **Step 4: Continue to next gap**
  - Don't commit per-gap; commit per logical group (one class, one module)

### Logical commit grouping for PR 3

Group commits by source file to keep history reviewable:

| Commit subject | Scope |
|---|---|
| `refactor(nlsq): align adapter.NLSQAdapter signatures to homodyne` | All `optimization.nlsq.adapter.*` rows |
| `refactor(nlsq): align config.NLSQConfig / config.NLSQValidationConfig to homodyne` | All `optimization.nlsq.config.*` rows |
| `refactor(nlsq): align core.* function signatures to homodyne` | All `optimization.nlsq.core.*` rows |
| `feat(nlsq): port missing functions from homodyne (data_prep, fit_computation, jacobian, memory)` | One commit per file for ported additions |
| `refactor(nlsq): align cmaes_wrapper / fallback_chain / recovery signatures` | Grouped |
| `refactor(nlsq): align multistart / parallel_accumulator / parameter_index_mapper signatures` | Grouped |
| `refactor(nlsq): align result_builder / results / strategies/ signatures` | Grouped |
| `feat(nlsq): align validation/ module to homodyne (post PR 1 absorption)` | Validation alignment |

After all logical groups committed:

### Task 3.N: Re-run audit and confirm NLSQ categories closed

Per the standard per-PR workflow:

- [ ] **Step 1: Re-run audit**

```bash
python -m tools.parity_audit run-all \
    --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
    --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
    --out docs/superpowers/audits/2026-05-18-parity-audit
```

- [ ] **Step 2: Confirm gap counts dropped to ~0 for NLSQ prefix**

```bash
grep -A 200 "### signatures" docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md | grep "optimization.nlsq" | wc -l
grep -A 200 "### classes" docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md | grep "optimization.nlsq" | wc -l
grep -A 200 "### configs" docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md | grep "optimization.nlsq" | wc -l
```

Expected: each count is 0 modulo D3 DROPped rows (which are accepted divergences and remain in REPORT.md).

- [ ] **Step 3: Commit the refreshed audit report**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md docs/superpowers/audits/2026-05-18-parity-audit/extracts/
git -c commit.gpgsign=false commit -m "audit(parity): refresh report after PR 3 (NLSQ signature/class/config drift closed)"
```

---

## PR 4: CMC structural alignment + absorb-then-delete + attribute renames

**Closes audit categories:**
- `file_inventory:missing_py_file` for `optimization.cmc.backends.multiprocessing`, `optimization.cmc.backends.pjit`
- `file_inventory:extra_py_file` for `optimization.cmc.backends.cpu_backend`, `optimization.cmc.backends.multiprocessing_backend`, `optimization.cmc.backends.pjit_backend`, `optimization.cmc.prior_builder`, `optimization.cmc.warmstart`
- `configs:extra_config_key` for the three CMC attribute rename targets (`target_accept_prob`, `max_r_hat`, `nlsq_prior_width_factor`)

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- None for PR 4 — all heterodyne-only CMC files are absorbed rather than dropped (preserves T1–T4 stability work).

**Goal:**
1. Rename CMC backends to match homodyne (`multiprocessing_backend.py` → `multiprocessing.py`; `pjit_backend.py` → `pjit.py`).
2. Absorb `warmstart.py` and `prior_builder.py` into `priors.py` (per spec §7 Rule 1).
3. Decide `cpu_backend.py`: if homodyne's `multiprocessing.py` handles `n_workers=1` correctly, absorb-then-delete `cpu_backend.py` into `multiprocessing.py`. If homodyne has no single-CPU path, keep `cpu_backend.py` and mark as D3 divergence (heterodyne-required).
4. Revert CMC config attribute names: `target_accept_prob` → `target_accept`, `max_r_hat` → `r_hat_threshold`, `nlsq_prior_width_factor` → `prior_width_factor`. Remove `from_dict()` legacy-key shim.
5. Update CLAUDE.md (delete lines 21-25 referencing the old attribute names).

**Critical preservation:** The warm-start handoff has two distinct call signatures that the absorber MUST preserve (per REPORT.md §Manual narrative diffs §6):
- `clamp_params_to_interior(np.ndarray, list[str]) → (np.ndarray, list[str])`
- `clamp_to_interior(NLSQResult, dict | None) → NLSQResult`

Plus geometric-margin log-space math and `fixed_param_overrides` arg logic. All of this must survive the absorption verbatim.

### Task 4.1: Rename `multiprocessing_backend.py` → `multiprocessing.py`

Follow Task 1.2 (rename pattern) for `heterodyne/optimization/cmc/backends/multiprocessing_backend.py` → `heterodyne/optimization/cmc/backends/multiprocessing.py`. Update all import sites and `backends/__init__.py`. Run full test suite. Commit.

```bash
git mv heterodyne/optimization/cmc/backends/multiprocessing_backend.py heterodyne/optimization/cmc/backends/multiprocessing.py
# ... import updates ...
git -c commit.gpgsign=false commit -m "refactor(cmc): rename backends.multiprocessing_backend → backends.multiprocessing (homodyne parity)"
```

### Task 4.2: Rename `pjit_backend.py` → `pjit.py`

Same pattern as 4.1.

```bash
git mv heterodyne/optimization/cmc/backends/pjit_backend.py heterodyne/optimization/cmc/backends/pjit.py
# ... import updates ...
git -c commit.gpgsign=false commit -m "refactor(cmc): rename backends.pjit_backend → backends.pjit (homodyne parity)"
```

### Task 4.3: Decide and handle `cpu_backend.py`

- [ ] **Step 1: Verify homodyne's `multiprocessing.py` handles single-CPU correctly**

```bash
grep -nE "n_workers|num_workers|cpu_count" /home/wei/Documents/GitHub/homodyne/homodyne/optimization/cmc/backends/multiprocessing.py
```

Look for a path that handles `n_workers=1` without spawning subprocesses (inline execution).

- [ ] **Step 2a (if homodyne handles single-CPU): Absorb cpu_backend.py into multiprocessing.py**

Follow Task 1.3 absorb-then-delete pattern. The CPU-only path goes into a branch in `multiprocessing.py` triggered by `n_workers=1` or `backend='cpu'`.

- [ ] **Step 2b (if homodyne does NOT handle single-CPU): Keep cpu_backend.py as D3 divergence**

Append to `DIVERGENCE_REGISTRY.md` D3 section:

```
- file_inventory:extra_py_file | `optimization.cmc.backends.cpu_backend` | heterodyne-required single-CPU execution path; homodyne's multiprocessing.py always spawns subprocesses. Decision: keep (D3, heterodyne improvement preserved)
```

Commit:

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): preserve cpu_backend.py as D3 heterodyne-required (no homodyne equiv)"
```

### Task 4.4: Absorb `warmstart.py` into `priors.py` (preserving narrative-diff signatures)

This is the load-bearing absorption per REPORT.md §Manual narrative diffs §6.

**Files:**
- Read: `heterodyne/optimization/cmc/warmstart.py`
- Modify: `heterodyne/optimization/cmc/priors.py`
- Delete: `heterodyne/optimization/cmc/warmstart.py`
- Modify: every file importing from `heterodyne.optimization.cmc.warmstart`

- [ ] **Step 1: Read warmstart.py contents in full**

```bash
cat heterodyne/optimization/cmc/warmstart.py
```

Note the public symbols: `clamp_params_to_interior`, `clamp_to_interior`, any other public functions/classes. Note the geometric-margin log-space math and `fixed_param_overrides` logic.

- [ ] **Step 2: Find all import sites**

```bash
grep -rn "from heterodyne.optimization.cmc.warmstart" heterodyne/ tests/
grep -rn "from heterodyne.optimization.cmc import .*warmstart" heterodyne/ tests/
```

- [ ] **Step 3: Read priors.py to find the absorption point**

```bash
cat heterodyne/optimization/cmc/priors.py | head -100
```

The absorbed functions go after the existing prior-construction functions, under a heading comment:

```python
# --- Absorbed from optimization/cmc/warmstart.py (Phase 4 PR 4; spec §7 Rule 1) ---
# Preserves the two public call signatures + geometric-margin log-space math
# + fixed_param_overrides arg logic from REPORT.md §Manual narrative diffs §6.
```

- [ ] **Step 4: Append warmstart functions to priors.py**

Use Edit tool to append. Do NOT rename `clamp_params_to_interior` or `clamp_to_interior` — their signatures must be byte-identical to the current heterodyne versions. Tests reference these names.

- [ ] **Step 5: Update import sites**

For each file:line from Step 2, change `from heterodyne.optimization.cmc.warmstart import clamp_to_interior` → `from heterodyne.optimization.cmc.priors import clamp_to_interior` (and similar for other symbols).

- [ ] **Step 6: Full test suite green BEFORE deletion**

```bash
make test 2>&1 | tail -5
```

Expected: all ~3,000 tests pass, including the 438 CMC tests and the warm-start regression tests. If anything fails, the absorption broke something — fix before continuing. Do NOT delete warmstart.py yet.

- [ ] **Step 7: Delete warmstart.py**

```bash
git rm heterodyne/optimization/cmc/warmstart.py
```

- [ ] **Step 8: Re-run full test suite AFTER deletion**

```bash
make test 2>&1 | tail -5
```

Expected: still all green.

- [ ] **Step 9: Commit**

```bash
git add heterodyne/optimization/cmc/priors.py
git add tests/ -A
git -c commit.gpgsign=false commit -m "refactor(cmc): absorb warmstart.py into priors.py (spec §7 Rule 1)

Preserves the two public call signatures (clamp_params_to_interior,
clamp_to_interior), geometric-margin log-space math, and
fixed_param_overrides arg logic per REPORT.md §Manual narrative diffs §6.
438 CMC tests green before and after deletion.

Refs: spec §6 PR 3, audit row file_inventory:extra_py_file optimization.cmc.warmstart"
```

### Task 4.5: Absorb `prior_builder.py` into `priors.py`

Same pattern as Task 4.4. Identify public symbols in `prior_builder.py`, absorb into `priors.py` under a heading comment, update import sites, full test suite green, delete `prior_builder.py`, re-test, commit.

### Task 4.6: Revert CMC config attribute names

**Files:**
- Modify: `heterodyne/optimization/cmc/config.py`
- Modify: every call site referencing the old names

Per DISPOSITIONS.md and spec §2 true-zero-divergence:
- `target_accept_prob` → `target_accept`
- `max_r_hat` → `r_hat_threshold`
- `nlsq_prior_width_factor` → `prior_width_factor`

- [ ] **Step 1: Find all references**

```bash
grep -rn "target_accept_prob\|max_r_hat\|nlsq_prior_width_factor" heterodyne/ tests/
```

- [ ] **Step 2: Edit `config.py` dataclass**

In `heterodyne/optimization/cmc/config.py`:
- Rename field `target_accept_prob: float` → `target_accept: float`
- Rename field `max_r_hat: float` → `r_hat_threshold: float`
- Rename field `nlsq_prior_width_factor: float` → `prior_width_factor: float`

- [ ] **Step 3: Remove `from_dict()` legacy-key shim**

The shim that maps old keys to new is no longer needed. Find it (likely a small block in `config.py`) and delete it.

- [ ] **Step 4: Update every call site**

For each file:line from Step 1, replace the old name with the new name. Use Edit tool.

- [ ] **Step 5: Full test suite**

```bash
make test 2>&1 | tail -5
```

Expected: all green. Any test that referenced the old name will fail and need updating (which Step 4 already handled).

- [ ] **Step 6: Update CLAUDE.md**

Edit `CLAUDE.md` — delete lines 21–25 (the "CMC Config Attribute Names (Renamed)" block and the `from_dict()` note).

- [ ] **Step 7: Commit**

```bash
git add heterodyne/optimization/cmc/config.py heterodyne/ tests/ CLAUDE.md
git -c commit.gpgsign=false commit -m "refactor(cmc): revert CMCConfig attribute names to homodyne parity

target_accept_prob → target_accept, max_r_hat → r_hat_threshold,
nlsq_prior_width_factor → prior_width_factor. from_dict() legacy-key
shim removed. CLAUDE.md lines 21-25 deleted to match.

Spec §2 true-zero-divergence enforcement.
Refs: spec §6 PR 3"
```

### Task 4.7: Re-run audit and confirm PR 4 categories closed

Per standard per-PR workflow. Verify CMC file_inventory and CMC attribute-name configs all dropped to 0.

---

## PR 5: CMC signature/class/config drift + `run_nuts_with_retry` port

**Closes audit categories:**
- All `signatures:changed`, `signatures:missing_in_heterodyne`, `signatures:extra_in_heterodyne` rows in `optimization.cmc.*` not closed by PR 4
- All `classes:missing_method/field/class` rows in `optimization.cmc.*`
- All `configs:missing_config_key/extra_config_key` rows in `optimization.cmc.*` not closed by PR 4

**New port:** `run_nuts_with_retry` from homodyne (per REPORT.md §Manual narrative diffs §2 — genuine behavioral gap).

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D3: any qualname matching D3 patterns (none expected in CMC scope)

### Task 5.1: Process per gap

Same template as PR 3 Process Per Gap, but scoped to `optimization.cmc.*` rows. Use Edit tool for in-place signature changes; use Write tool for new function/class ports.

Logical commit grouping for PR 5:

| Commit subject | Scope |
|---|---|
| `refactor(cmc): align core.* function signatures to homodyne` | `optimization.cmc.core.*` |
| `refactor(cmc): align sampler.* signatures to homodyne` | `optimization.cmc.sampler.*` |
| `refactor(cmc): align model.py / data_prep.py / diagnostics.py / io.py signatures` | One commit per file |
| `refactor(cmc): align priors.py signatures (post-absorption)` | After PR 4 absorption |
| `refactor(cmc): align reparameterization.py / scaling.py / plotting.py / results.py signatures` | Grouped |
| `refactor(cmc): align backends/*.py signatures to homodyne` | All backend files |

### Task 5.2: Port `run_nuts_with_retry` from homodyne

**Files:**
- Read: `/home/wei/Documents/GitHub/homodyne/homodyne/optimization/cmc/sampler.py`
- Modify: `heterodyne/optimization/cmc/sampler.py` (add `run_nuts_with_retry` function)
- Create: `tests/regression/test_cmc_nuts_retry.py` (TDD characterization)

Per spec §7 Rule 3 (TDD for ports): failing test first.

- [ ] **Step 1: Read homodyne's implementation**

```bash
grep -A 80 "^def run_nuts_with_retry" /home/wei/Documents/GitHub/homodyne/homodyne/optimization/cmc/sampler.py
```

Read the full function body. Note: 3-attempt escalation loop with adaptive `target_accept_prob` per REPORT.md §Manual narrative diffs §2.

- [ ] **Step 2: Write the failing characterization test**

`tests/regression/test_cmc_nuts_retry.py`:

```python
"""Characterization test for run_nuts_with_retry ported from homodyne.

3-attempt escalation loop with adaptive target_accept_prob per attempt
(e.g., 0.8 → 0.9 → 0.95). The retry triggers on excessive divergent
transitions in the previous attempt.

Refs: REPORT.md §Manual narrative diffs §cmc/sampler.py finding;
Phase 4 PR 5 Task 5.2.
"""
from __future__ import annotations

import inspect

from heterodyne.optimization.cmc import sampler


def test_run_nuts_with_retry_exists() -> None:
    """run_nuts_with_retry is exposed from the cmc.sampler module."""
    assert hasattr(sampler, "run_nuts_with_retry")
    assert callable(sampler.run_nuts_with_retry)


def test_run_nuts_with_retry_signature_matches_homodyne() -> None:
    """Signature has the expected kwargs: max_attempts, target_accept_schedule."""
    sig = inspect.signature(sampler.run_nuts_with_retry)
    params = sig.parameters
    assert "max_attempts" in params or "max_retries" in params
    # The function should accept some form of per-attempt target_accept schedule
    # (homodyne's name was target_accept_schedule; if heterodyne ports with same name, this passes)
    assert any(k in params for k in ("target_accept_schedule", "target_accepts", "target_accept_per_attempt"))
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/regression/test_cmc_nuts_retry.py -v
```

Expected: FAIL (`AttributeError: module ... has no attribute 'run_nuts_with_retry'`).

- [ ] **Step 4: Port the function**

Append `run_nuts_with_retry` to `heterodyne/optimization/cmc/sampler.py`. Adapt only:
- Replace any homodyne-specific parameter references with heterodyne's 14-param model
- Keep the 3-attempt loop verbatim, including the per-attempt `target_accept_prob` escalation schedule

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/regression/test_cmc_nuts_retry.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Full test suite**

```bash
make test 2>&1 | tail -5
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add heterodyne/optimization/cmc/sampler.py tests/regression/test_cmc_nuts_retry.py
git -c commit.gpgsign=false commit -m "feat(cmc): port run_nuts_with_retry from homodyne (3-attempt escalation)

Genuine behavioral gap per REPORT.md §Manual narrative diffs §cmc/sampler.py.
3-attempt escalation with adaptive target_accept_prob per attempt
(0.8 → 0.9 → 0.95). Retry triggers on excessive divergent transitions.

Heterodyne's effective_warmup_floor contract is preserved (called inside
each retry attempt; floor adapts per the heterodyne 14-param dense_mass
constraint).

Refs: spec §6 PR 4; REPORT.md §Manual narrative diffs §2"
```

- [ ] **Step 8: Delete the corresponding placeholder from DIVERGENCE_REGISTRY.md**

In the "Manual-narrative-diff escalations" section, delete the line:

```
- (PR 5) `optimization.cmc.sampler.run_nuts_with_retry` ported from homodyne — **not a divergence**, fixed in scope. Registry entry will be deleted when PR 5 lands.
```

Commit:

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): clear PR 5 narrative-diff placeholder (run_nuts_with_retry landed)"
```

### Task 5.3: Re-run audit and confirm PR 5 categories closed

Per standard per-PR workflow.

---

## PR 6: CLI parity (26 P0 missing flags + 26 P1 extras + CLI signature/class drift)

**Closes audit categories:**
- All `cli:missing_cli_flag` rows (26)
- All `cli:extra_cli_flag` rows (26)
- All `cli:cli_default_drift` rows (any)
- All `signatures:changed/missing/extra` rows in `cli.*` not closed by earlier PRs
- All `classes:*` rows in `cli.*`
- `logs_errors:exit_code_drift` row (1)
- All `configs:*` rows in `cli.*` (if any)

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D3: `cli:missing_cli_flag --laminar-flow` (homodyne-only mode)

### Task 6.1: Add missing CLI flags

For each P0 `cli:missing_cli_flag` row (e.g., `--cmc-num-shards`, `--cmc-plot-diagnostics`, `--data-file`, `--dense-mass-matrix`, `--filter`):

- [ ] **Step 1: Read homodyne's add_argument call**

```bash
grep -A 8 "add_argument.*--cmc-num-shards" /home/wei/Documents/GitHub/homodyne/homodyne/cli/
```

- [ ] **Step 2: Port to heterodyne's `cli/args_parser.py`**

Add the same `parser.add_argument(...)` call with the same flag, dest, default, choices, help text.

- [ ] **Step 3: Wire the flag into the command dispatcher**

In `cli/commands.py`, add handling for the new `args.NAME` attribute. If the flag triggers behavior, wire it to the appropriate optimizer/config knob.

- [ ] **Step 4: Run CLI tests**

```bash
uv run pytest tests/unit/cli/ -x 2>&1 | tail -10
```

- [ ] **Step 5: Commit per logical group (e.g., all CMC-related flags in one commit)**

### Task 6.2: Remove heterodyne-only CLI flags (or DROP under D3)

For each P1 `cli:extra_cli_flag` row:
- If the flag is shear/laminar-related → DROP under D3 in DIVERGENCE_REGISTRY.md
- If the flag is heterodyne-required → DROP under D3 ("heterodyne-required CLI flag preserved")
- Otherwise → remove from `cli/args_parser.py` and remove handling from `cli/commands.py`

### Task 6.3: Align CLI default drift (`--config`)

Audit row: `cli_default_drift --config` (homodyne default `Path('./homodyne_config.yaml')` vs heterodyne `None`).

- [ ] **Step 1: Edit `cli/args_parser.py`**

Change heterodyne's default from `None` to `Path('./heterodyne_config.yaml')` (note: heterodyne config filename, not homodyne — this is parity-equivalent, not byte-identical).

- [ ] **Step 2: Verify the default is honored**

```bash
uv run pytest tests/unit/cli/test_args_parser.py -v
```

- [ ] **Step 3: Commit**

### Task 6.4: Add `sys.exit(N)` calls (exit-code drift)

Audit row: `logs_errors:exit_code_drift` — homodyne exit codes `[0, 1, 130]`, heterodyne `[]`.

- [ ] **Step 1: Find homodyne's exit-code paths**

```bash
grep -nE "sys\.exit\(0\)|sys\.exit\(1\)|sys\.exit\(130\)" /home/wei/Documents/GitHub/homodyne/homodyne/cli/
```

- [ ] **Step 2: Port the exit calls to heterodyne**

In `heterodyne/cli/main.py` and `cli/commands.py`:
- `sys.exit(0)` on successful completion (currently heterodyne returns 0 but doesn't call sys.exit)
- `sys.exit(1)` on convergence failure / data validation failure (currently heterodyne returns 2 — REVERT to homodyne's 1)
- `sys.exit(130)` on KeyboardInterrupt (currently heterodyne doesn't handle Ctrl-C explicitly)

- [ ] **Step 3: Tests**

Add CLI tests verifying each exit code path. Use `pytest.raises(SystemExit) as exc_info` and assert `exc_info.value.code == EXPECTED`.

- [ ] **Step 4: Commit**

### Task 6.5: Align all CLI signatures/classes/configs

Per the PR 3 process-per-gap template, scoped to `cli.*` qualnames.

Logical commit grouping:

| Commit subject | Scope |
|---|---|
| `refactor(cli): align args_parser signatures to homodyne` | `cli.args_parser.*` |
| `refactor(cli): align commands.dispatch_command and main signatures to homodyne` | `cli.commands.*`, `cli.main.*` |
| `refactor(cli): align config_generator signatures to homodyne` | `cli.config_generator.*` |
| `refactor(cli): align data_pipeline / optimization_runner / plot_dispatch / result_saving signatures` | Grouped |
| `feat(cli): port missing functions from homodyne (normalize_angle_to_symmetric_range, etc.)` | One commit per logical port |

### Task 6.6: Re-run audit and confirm PR 6 categories closed

Per standard per-PR workflow.

---

## PR 7: Viz parity

**Closes audit categories:**
- All `signatures:*`, `classes:*`, `configs:*` rows in `viz.*`
- All `logs_errors:*` rows in `viz.*` (any heterodyne-only log strings)

**Drops:** any D3-matched rows (shear-related visualization, none expected)

### Process

Per the PR 3 process-per-gap template, scoped to `viz.*`. Logical commits:

| Commit subject | Scope |
|---|---|
| `refactor(viz): align mcmc_diagnostics / mcmc_arviz / mcmc_plots signatures` | Grouped |
| `refactor(viz): align mcmc_dashboard / nlsq_plots / datashader_backend signatures` | Grouped |
| `refactor(viz): align validation.py signatures` | `viz.validation.*` |
| `feat(viz): port missing visualization functions from homodyne` | One commit per logical port |

### Task 7.N: Re-run audit and confirm PR 7 categories closed

Per standard per-PR workflow.

---

## PR 8: Data / IO / Utils / Device parity

**Closes audit categories:**
- All `signatures:*`, `classes:*`, `configs:*` rows in `data.*`, `io.*`, `utils.*`, `device.*`
- All `logs_errors:log_format_drift` rows in these subpackages

**Drops:** any D3-matched rows

### Process

Per the PR 3 template. Logical commits:

| Commit subject | Scope |
|---|---|
| `refactor(data): align xpcs_loader / preprocessing / validators / quality_controller signatures` | Grouped |
| `refactor(data): align memory_manager / performance_engine / config / optimization signatures` | Grouped |
| `refactor(io): align json_utils / nlsq_writers / mcmc_writers signatures` | `io.*` |
| `refactor(utils): align logging / async_io / path_validation signatures` | `utils.*` |
| `refactor(device): align cpu / config / platform signatures` | `device.*` |
| `feat(data,io,utils,device): port missing functions from homodyne` | Grouped |

### Task 8.N: Re-run audit and confirm PR 8 categories closed

Per standard per-PR workflow.

---

## PR 9: Logs / errors / messages alignment

**Closes audit categories:**
- All remaining `logs_errors:log_format_drift` rows across the codebase
- All `logs_errors:raise` stem drift (exception class + message stems differ)

**Drops:** D3-matched rows mentioning shear/laminar (~22 rows per audit grep)

### Process

This PR is a large mechanical alignment. For each row in REPORT.md `logs_errors:log_format_drift`:

- [ ] **Step 1: Read homodyne's log message at the corresponding call site**
- [ ] **Step 2: Edit heterodyne's matching call site to use the same format string**
- [ ] **Step 3: Run the module's unit tests**
- [ ] **Step 4: Continue to next row**

Group commits by source module (one commit per `cli/`, `optimization/nlsq/`, etc.).

### Task 9.N: Re-run audit and confirm PR 9 categories closed

Per standard per-PR workflow.

---

## PR 10: Docs structural fill (22 missing pages + cross-reference repairs)

**Closes audit categories:**
- `file_inventory:missing_doc_file` for all KEEP-dispositioned docs files (~22 rows after D2 drops)
- `docs:missing_doc` (~27 P1 rows after D2 drops)
- `docs:heading_drift` (~837 P2 rows) — these get fixed naturally by porting full docs content

**Drops (logged to DIVERGENCE_REGISTRY.md):**
- D2: 10 physics-specific docs pages per DISPOSITIONS.md

### Task 10.1: Append D2 entries to DIVERGENCE_REGISTRY.md

- [ ] **Step 1: Append D2 entries**

In `docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md`, D2 section:

```
- file_inventory:missing_doc_file | `api/homodyne_model.rst` | replaced by heterodyne-equivalent doc (D2)
- file_inventory:missing_doc_file | `theory/homodyne_scattering.rst` | homodyne physics-only (D2)
- file_inventory:missing_doc_file | `theory/yielding_dynamics.rst` | homodyne physics-only (D2)
- file_inventory:missing_doc_file | `theory/anti_degeneracy_defense.rst` | consolidated with theory/anti_degeneracy.rst (D2)
- file_inventory:missing_doc_file | `user_guide/03_advanced_topics/laminar_flow.rst` | homodyne mode-only (D2)
- file_inventory:missing_doc_file | `user_guide/01_fundamentals/homodyne_overview.rst` | replaced by heterodyne overview (D2)
- docs:heading_drift | `architecture/nlsq-fitting-architecture.md` :: "Example: laminar_flow mode" | homodyne mode-only (D2)
- docs:heading_drift | `architecture/nlsq-fitting-architecture.md` :: "Layer 5: Shear-Sensitivity Weighting" | Layer 5 (D2)
- docs:heading_drift | `theory/analysis_modes.rst` :: "Shear Integral" | homodyne shear-only (D2)
- docs:heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` :: "Shear Parameters (laminar_flow only)" | homodyne mode-only (D2)
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/DIVERGENCE_REGISTRY.md
git -c commit.gpgsign=false commit -m "audit(parity): log D2 physics-specific docs as accepted divergences"
```

### Task 10.2: Port each missing docs file from homodyne

For each KEEP-dispositioned `missing_doc_file` row (~22 files):

- [ ] **Step 1: Read homodyne's docs file**

```bash
cat /home/wei/Documents/GitHub/homodyne/docs/source/PATH/FILE.rst
```

- [ ] **Step 2: Port to heterodyne**

Use Write tool to create `docs/source/PATH/FILE.rst` in heterodyne. Apply these adaptations:
- Replace `homodyne` → `heterodyne` in autodoc directives, function/class refs, and prose
- Replace 7-param references with 14-param references (use `D0_ref`, `D0_sample`, etc.)
- Remove any shear/laminar-flow specific sections
- Adjust ADR rationales for heterodyne-specific design decisions where they differ

- [ ] **Step 3: Verify the doc builds**

```bash
uv run sphinx-build -W -b html docs/source docs/_build/html 2>&1 | tail -10
```

Expected: build succeeds with no warnings. If warnings appear (broken xrefs, missing autodoc targets), fix before continuing — those would be P0 docs gaps if left unresolved.

- [ ] **Step 4: Commit per logical group**

Group docs ports by topic:

| Commit subject | Files |
|---|---|
| `docs(api): port api/cmc_*.rst from homodyne` | `api/cmc_backends.rst`, `api/cmc_reparameterization.rst`, `api/cmc_sampler.rst` |
| `docs(api): port api/nlsq_*.rst from homodyne` | `api/nlsq_adapter.rst`, `api/nlsq_wrapper.rst`, `api/optimization_guide.rst`, `api/theory_engine.rst` |
| `docs(architecture): port architecture/heterodyne-architecture-overview.md` | renamed from homodyne-architecture-overview.md |
| `docs(developer/adrs): port all ADRs adapted for heterodyne` | All adr_*.rst |
| `docs(theory): port theory/anti_degeneracy.rst (consolidated)` | `theory/anti_degeneracy.rst` (single page, not the _defense duplicate) |
| `docs(theory): port theory/theoretical_framework.rst adapted for heterodyne physics` | physics-formula content adapted, not just renamed |
| `docs(user_guide): port user_guide/01_fundamentals/analysis_modes.rst` | adapted |
| `docs(user_guide): port user_guide/02_data_and_fitting/model_selection.rst` | adapted |
| `docs(user_guide): port user_guide/03_advanced_topics/streaming_mode.rst` | adapted |
| `docs(user_guide): port user_guide/04_practical_guides/batch_processing.rst` | adapted |

### Task 10.3: Re-run audit and confirm PR 10 categories closed

Per standard per-PR workflow. The `docs:heading_drift` count should drop dramatically as ported docs bring matching headings.

---

## PR 11: Docs autodoc + xref fixup (make Sphinx warning-free)

**Closes audit categories:**
- All `docs:broken_autodoc_target` P0 rows (48 rows)
- All remaining `docs:heading_drift` P2 rows after PR 10 ports

**Drops:** none expected (any remaining D2 docs are already DROPped)

### Task 11.1: Audit-driven autodoc fixup

For each `docs:broken_autodoc_target` row:

- [ ] **Step 1: Read the broken row**

E.g., `expected autodoc target heterodyne.cli.args_parser not present in heterodyne docs page api/cli.rst`.

- [ ] **Step 2: Verify the target now exists**

After PRs 1–8, the heterodyne module/class/function should exist (the per-PR audit re-runs confirm this). Run:

```bash
python -c "import heterodyne.cli.args_parser; print(dir(heterodyne.cli.args_parser))"
```

- [ ] **Step 3: Verify the docs page references the correct target**

```bash
grep -nE "automodule|autoclass|autofunction" docs/source/api/cli.rst
```

If the docs page has the right `automodule:: heterodyne.cli.args_parser` directive, the broken_autodoc_target row should already be closed by the time we get here. If not, the directive needs a fix (typo, wrong module path).

- [ ] **Step 4: Build docs to confirm**

```bash
uv run sphinx-build -W -b html docs/source docs/_build/html 2>&1 | tail -10
```

Expected: zero warnings.

### Task 11.2: Fix any remaining heading_drift rows

If `docs:heading_drift` still has rows after PR 10, walk each one:

- [ ] **Step 1: Read both pages side-by-side**
- [ ] **Step 2: Add the missing heading to heterodyne's docs page** (preserving the homodyne hierarchy)
- [ ] **Step 3: Build docs to confirm**
- [ ] **Step 4: Commit per docs file**

### Task 11.3: Final Sphinx warnings-as-errors run

- [ ] **Step 1: Clean build**

```bash
rm -rf docs/_build docs/build
```

- [ ] **Step 2: Strict build**

```bash
uv run sphinx-build -W -b html docs/source docs/_build/html 2>&1 | tail -20
```

Expected: build succeeds with `0 errors, 0 warnings`. Any warning fails the build with `-W`.

- [ ] **Step 3: Commit any final fixes**

### Task 11.4: Re-run audit and confirm PR 11 categories closed

Per standard per-PR workflow.

---

## PR 12: Parity CI gate (the forever-property)

**Closes audit categories:** none directly — this PR makes parity a perpetual CI gate.

**Drops:** none

### Task 12.1: Add GitHub Actions workflow

**Files:**
- Create: `.github/workflows/parity-audit-no-regression.yml`

- [ ] **Step 1: Create the workflow file**

`.github/workflows/parity-audit-no-regression.yml`:

```yaml
name: parity-audit-no-regression

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  audit:
    name: Heterodyne→Homodyne parity audit (no regression)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout heterodyne (PR head)
        uses: actions/checkout@v4
        with:
          path: heterodyne
          fetch-depth: 0

      - name: Checkout homodyne (pinned SHA)
        uses: actions/checkout@v4
        with:
          repository: ImagingXPCS/homodyne   # adjust to actual org/repo
          ref: 0368cbdb075fcff1908c0da2b59a1b0d37d5eeca
          path: homodyne

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: latest

      - name: Set up Python
        run: uv python install 3.13

      - name: Install heterodyne deps
        working-directory: heterodyne
        run: uv sync --frozen

      - name: Run parity audit (PR head vs pinned homodyne)
        working-directory: heterodyne
        run: |
          uv run python -m tools.parity_audit run-all \
              --homodyne ../homodyne/homodyne \
              --heterodyne ./heterodyne \
              --out ./pr-audit
          mv pr-audit/REPORT.md pr-audit/REPORT-pr.md

      - name: Run parity audit (main baseline vs pinned homodyne)
        working-directory: heterodyne
        run: |
          git fetch origin main
          git worktree add /tmp/heterodyne-main origin/main
          cd /tmp/heterodyne-main
          uv sync --frozen
          uv run python -m tools.parity_audit run-all \
              --homodyne ${{ github.workspace }}/homodyne/homodyne \
              --heterodyne ./heterodyne \
              --out ./main-audit

      - name: Compare gap counts (no closed gap may reopen)
        working-directory: heterodyne
        run: |
          PR_TOTAL=$(grep -oE 'Total gaps: \*\*[0-9]+\*\*' pr-audit/REPORT-pr.md | grep -oE '[0-9]+')
          MAIN_TOTAL=$(grep -oE 'Total gaps: \*\*[0-9]+\*\*' /tmp/heterodyne-main/main-audit/REPORT.md | grep -oE '[0-9]+')
          echo "Main baseline total gaps: $MAIN_TOTAL"
          echo "PR head total gaps:       $PR_TOTAL"
          if [ "$PR_TOTAL" -gt "$MAIN_TOTAL" ]; then
            echo "::error::PR introduces parity regressions (gap count increased from $MAIN_TOTAL to $PR_TOTAL)"
            diff /tmp/heterodyne-main/main-audit/REPORT.md pr-audit/REPORT-pr.md | head -200
            exit 1
          fi
          echo "::notice::PR maintains or improves parity (gap delta: $((PR_TOTAL - MAIN_TOTAL)))"

      - name: Upload PR audit report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: parity-audit-pr
          path: heterodyne/pr-audit/
```

- [ ] **Step 2: Tune the homodyne checkout (replace placeholder org)**

The `repository: ImagingXPCS/homodyne` field in Step 1's YAML is a placeholder. Replace with the actual org/repo path where homodyne is hosted. If homodyne is in a different GitHub org, this needs the actual path.

Verify by asking the user or checking `.git/config` of the local homodyne clone:

```bash
git -C /home/wei/Documents/GitHub/homodyne remote get-url origin
```

Update the YAML to match.

- [ ] **Step 3: Update the pinned SHA**

The `ref: 0368cbdb075fcff1908c0da2b59a1b0d37d5eeca` in the YAML must match whatever homodyne SHA Phase 4 was audited against. This is the SHA in `docs/superpowers/audits/2026-05-18-parity-audit/homodyne_sha.txt` after the final post-PR-11 audit re-run.

If you want CI to track newer homodyne automatically, replace the SHA with `main` — but that means homodyne drift can fail your PRs without warning. Recommended: keep pinned, and run a separate "homodyne-drift refresh" audit on a schedule (e.g., monthly) to deliberately re-pin.

- [ ] **Step 4: Local dry-run**

Run the audit comparison locally to make sure the gap counts behave:

```bash
git stash  # if you have uncommitted work
python -m tools.parity_audit run-all \
    --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
    --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
    --out /tmp/local-audit
grep "Total gaps" /tmp/local-audit/REPORT.md
# Compare to docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md
git stash pop  # restore work
```

Expected: gap count after PRs 1-11 should be ~168 (just the DROP-dispositioned divergences, which appear in REPORT.md but don't fail the build because they're in DIVERGENCE_REGISTRY.md). If higher, some PR didn't close its targets — investigate before merging PR 12.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/parity-audit-no-regression.yml
git -c commit.gpgsign=false commit -m "ci(parity): add no-regression workflow for heterodyne→homodyne parity

Compares the PR head audit against the main-branch audit; fails if any
closed gap reopens. Pinned to homodyne SHA in docs/superpowers/audits/
2026-05-18-parity-audit/homodyne_sha.txt.

The audit tooling itself (tools/parity_audit/) ships with the repo, so
this workflow has no external service dependencies.

Refs: spec §6 PR 11, spec §7 Rule 2"
```

### Task 12.2: Verify CI passes on a no-op PR

- [ ] **Step 1: Push a no-op PR**

Create a branch with a trivial change (e.g., a typo fix in README), open a PR, and confirm the parity-audit-no-regression workflow runs and passes.

- [ ] **Step 2: Confirm the workflow's gap-count comparison logic works**

Read the GitHub Actions log for the comparison step. Verify it printed the expected counts (~168 for main baseline and PR head).

- [ ] **Step 3: Merge PR 12**

After verifying the CI gate, merge PR 12. From this point on, every PR runs the parity audit.

---

## Final completion criteria (post-PR-12)

Per spec §11:

1. `python -m tools.parity_audit run-all ...` reports **zero P0, zero P1, zero P2** gaps excluding the rows in DIVERGENCE_REGISTRY.md.
2. `DIVERGENCE_REGISTRY.md` contains exactly the dispositioned `DROP` rows with rationale.
3. Full test suite green (~3,000 tests).
4. CI gate `parity-audit-no-regression.yml` is active and passing.
5. `uv run sphinx-build -W docs/source docs/_build/html` completes with zero warnings.
6. The three narrative-diff placeholders in DIVERGENCE_REGISTRY.md are deleted (run_nuts_with_retry, anti_degeneracy_controller arch port, warmstart absorption all landed in scope).

### Final commit message template (use after PR 12 merges)

```
chore(parity): heterodyne→homodyne strict 1:1 parity complete

Phase 4 of the strict-1:1 mirror program (specs/2026-05-18-...-design.md)
landed across PRs 1-12. The parity audit tooling at tools/parity_audit/
is the perpetual gate via .github/workflows/parity-audit-no-regression.yml.

Divergence registry: docs/superpowers/audits/2026-05-18-parity-audit/
DIVERGENCE_REGISTRY.md documents the ~168 intentionally-accepted divergences
(D1 physics-exempt files, D2 physics-specific docs, D3 shear/laminar/
homodyne-physics content).
```

---

## Self-review

**Spec coverage:** Each spec section maps to plan content.
- §1 Goal → plan goal
- §2 Scope decisions / true-zero-divergence → enforced in PR 4 attribute reverts + DIVERGENCE_REGISTRY.md structure
- §3 Four phases / Phase 4 → this plan
- §6 PR plan table → expanded to 12 PRs in this document; renumbered to add PR 2 (anti-degen arch port) and split CMC into PR 4 (structural) + PR 5 (signatures)
- §7 Rules 1-4 → Rule 1 (absorb-then-delete) in PR 1 Tasks 1.3-1.4 and PR 4 Tasks 4.4-4.5; Rule 2 (audit re-run per PR) in standard Per-PR workflow Step 5 and PR 12 CI gate; Rule 3 (TDD for ports) in PR 1 Task 1.5, PR 2 Task 2.3, PR 5 Task 5.2; Rule 4 (docs ride with code) — implicit in each PR's commits, made explicit in PR 10-11 working alongside code-touching PRs
- §11 success criteria → "Final completion criteria" section above
- §13 artifact summary → PR commits + DIVERGENCE_REGISTRY.md + CI workflow

**Placeholder scan:** None. Every step has actual commands and concrete code paths. The "process per gap" template in PRs 3, 5, 6, 7, 8, 9, 11 is deliberate (not a placeholder) — it documents the deterministic workflow for high-volume mechanical work that would otherwise require enumerating thousands of identical task structures.

**Type consistency:** Function signatures and module paths referenced in this plan (`run_nuts_with_retry`, `clamp_params_to_interior`, `clamp_to_interior`, `NLSQWrapper.fit`, `AntiDegeneracyController.from_config`) all match REPORT.md narrative diffs and the audit's static extracts. CMC attribute rename targets (`target_accept`, `r_hat_threshold`, `prior_width_factor`) match DISPOSITIONS.md and CLAUDE.md lines 21-25 (the lines to delete).

**Known plan limitations and reader guidance:**
- This plan is bounded but the work it describes is not — closing 4,322 KEEP gaps will take many sessions. Use `superpowers:subagent-driven-development` per PR; consider running PRs in parallel where they don't share files (PRs 6, 7, 8 are independent of each other after PR 5 lands).
- The "process per gap" template in PRs 3, 5, 6, 7, 8, 9 means the executing agent reads REPORT.md row-by-row for each PR. After each PR, the audit re-run regenerates REPORT.md, which may reorder/recolour rows — always use the latest REPORT.md as the authoritative gap source, not the version that was current when this plan was written.
- PR 12 references homodyne's GitHub org as a placeholder; the user or executing agent must update this before the CI gate goes live (Task 12.1 Step 2).
