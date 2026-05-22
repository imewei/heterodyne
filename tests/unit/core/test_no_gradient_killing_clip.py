"""Lint test: no gradient-killing ops on gradient-bearing physics tensors.

CLAUDE.md rule #7 mandates ``jnp.where(x>eps, x, eps)`` or smooth alternatives
(``smooth_clip``, ``smooth_bound``, ``smooth_abs``) over ``jnp.clip`` /
``jnp.maximum`` on tensors that participate in NUTS leapfrog or NLSQ Jacobian
flow.  Hard ``jnp.clip`` and ``jnp.maximum`` zero the gradient at the
boundary and stall the sampler / optimizer.

This test scans gradient-bearing source files for both ops and asserts each
call matches an explicit allow-listed pattern.

When you add a new ``jnp.clip`` or ``jnp.maximum``:

  * Indexing context (``jnp.searchsorted`` etc.) — allowed.
  * Overflow guard inside ``jnp.exp`` / ``jnp.log`` — allowed (gradient is
    already vanishingly small at the cap).
  * Integer-literal scalar bounds like ``jnp.maximum(1, ...)`` — allowed
    (no autodiff flow).
  * Square-root-of-weights pattern ``jnp.sqrt(jnp.maximum(w, 0.0))`` —
    allowed (weights are typically static / non-differentiated).
  * Otherwise — switch to ``smooth_clip`` / ``smooth_bound`` /
    ``jnp.where(cond, x, floor)`` and document the gradient continuity.

This catches deep-RCA F8 regressions before they ship.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Allow patterns: each pattern is matched against the FULL source line.
# A jnp.clip call is acceptable iff at least one pattern matches its line.
_ALLOWED_CLIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Comments and docstrings referencing jnp.clip are not actual calls
    re.compile(r"^\s*#"),
    re.compile(r'^\s*[\'"]'),
    # Documentation placeholder syntax (ellipsis argument) — not a real call
    re.compile(r"jnp\.clip\(\s*\.\.\."),
    # Indexing: jnp.clip(jnp.searchsorted(...), 0, n_grid - 1)
    re.compile(r"jnp\.clip\(\s*jnp\.searchsorted"),
    # Overflow guard for exp on the same line: jnp.exp(jnp.clip(x, -limit, limit))
    re.compile(r"jnp\.exp\(\s*jnp\.clip\("),
    # Overflow guard for log on positive support: jnp.log(jnp.clip(x, eps, ...))
    re.compile(r"jnp\.log\(\s*jnp\.clip\("),
    # Symmetric exp overflow guard split across lines:
    # ``exponent = jnp.clip(x, -<int>, <int>)`` then ``jnp.exp(exponent)``.
    # The clip is bounded away from zero on both sides, so the gradient is
    # only zeroed in the overflow tails where exp would saturate anyway.
    re.compile(r"jnp\.clip\([^,]+,\s*-\d+\s*,\s*\d+\s*\)"),
    # CLAUDE.md canonical sample-fraction equation:
    # ``f_s(t) = clip(f0 * exp(f1 * (t - f2)) + f3, 0, 1)`` — the [0,1] bound
    # is part of the model definition, not a numerical band-aid. Match the
    # canonical literal bounds.
    re.compile(r"jnp\.clip\([^,]+,\s*0\.0\s*,\s*1\.0\s*\)"),
)

# Allow patterns for jnp.maximum.  The legitimate uses are:
#   - integer scalar bounds (no autodiff flow)
#   - sqrt-of-weights pattern (weights are non-differentiated inputs)
#   - inside abs-of-pair scale estimators where the result is a denominator
#     consumed by a stability check rather than a forward signal
_ALLOWED_MAXIMUM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*#"),
    re.compile(r'^\s*[\'"]'),
    # Documentation placeholder syntax (ellipsis argument) — not a real call
    re.compile(r"jnp\.maximum\(\s*\.\.\."),
    # Integer-literal scalar bound: jnp.maximum(1, ...) or jnp.maximum(0, ...)
    re.compile(r"jnp\.maximum\(\s*-?\d+\s*,"),
    # Sqrt-of-weights pattern: jnp.sqrt(jnp.maximum(w, 0.0))
    re.compile(r"jnp\.sqrt\(\s*jnp\.maximum\("),
    # Scale-estimator stability denominator: jnp.maximum(jnp.maximum(jnp.abs(...
    re.compile(r"jnp\.maximum\(\s*jnp\.maximum\(\s*jnp\.abs\("),
)

# Tech-debt grandfather list: source lines that currently violate rule #7
# but pre-date this lint expansion. Each entry is (file_basename, line_no).
# REMOVE entries as the underlying code is migrated to jnp.where /
# smooth_clip / smooth_bound. New violations are blocked by the lint.
_GRANDFATHERED_MAXIMUM: frozenset[tuple[str, int]] = frozenset(
    {
        # J_rate floor in t^alpha rate function — rate is a gradient-bearing
        # forward signal in the physics model; switch to jnp.where(J>0, J, 0).
        ("theory.py", 166),
        ("models.py", 193),
        ("models.py", 224),
        # Rate-function floor inside physics_utils.compute_rate_function.
        ("physics_utils.py", 287),
    }
)


def _gradient_bearing_source_files() -> list[Path]:
    """Find gradient-bearing files in heterodyne/core/.

    Includes the original physics*.py set plus theory.py, models.py,
    jax_backend.py, and scaling_utils.py — every file in core/ whose
    outputs participate in the NLSQ Jacobian or NUTS leapfrog flow.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # tests/unit/core/<file>.py → repo
    core_dir = repo_root / "heterodyne" / "core"
    extra_names = (
        "theory.py",
        "models.py",
        "jax_backend.py",
        "scaling_utils.py",
    )
    files = list(core_dir.glob("physics*.py"))
    for name in extra_names:
        candidate = core_dir / name
        if candidate.exists():
            files.append(candidate)
    return sorted(set(files))


# Backwards-compatible alias kept for any external test imports.
_physics_source_files = _gradient_bearing_source_files


def _iter_op_lines(path: Path, op_token: str):
    """Yield ``(lineno, line)`` for every line in ``path`` containing
    ``op_token`` (e.g. ``jnp.clip(`` — the open paren guarantees a call)."""
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if op_token in line:
            yield lineno, line


@pytest.mark.parametrize("path", _gradient_bearing_source_files(), ids=lambda p: p.name)
def test_no_unguarded_clip_in_gradient_bearing_module(path: Path) -> None:
    """Every ``jnp.clip(...)`` in a gradient-bearing core file must match
    one of the allow-listed patterns (indexing or overflow guard).

    If this test fails on a new ``jnp.clip`` call, the fix is to:
      1. Switch to ``smooth_clip(x, low, high)`` for physical bounds, OR
      2. Switch to ``jnp.where(cond, x, floor)`` if a one-sided floor is
         intended, OR
      3. Add a new allow-listed pattern here with a comment explaining why
         the call is safe (e.g. the operand has no autodiff dependency).
    """
    violations: list[str] = []
    for lineno, line in _iter_op_lines(path, "jnp.clip("):
        if any(p.search(line) for p in _ALLOWED_CLIP_PATTERNS):
            continue
        violations.append(f"  {path.name}:{lineno}: {line.strip()}")

    assert not violations, (
        "Unguarded jnp.clip() found in gradient-bearing module — violates "
        "CLAUDE.md rule #7 (use smooth_clip/smooth_bound/jnp.where instead).  "
        "See deep-RCA F8 for the failure mode.\n"
        "Offending lines:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize("path", _gradient_bearing_source_files(), ids=lambda p: p.name)
def test_no_unguarded_maximum_in_gradient_bearing_module(path: Path) -> None:
    """Every ``jnp.maximum(...)`` in a gradient-bearing core file must
    match an allow-listed pattern or be grandfathered as known tech debt.

    Adding a new ``jnp.maximum`` on a gradient-bearing tensor without a
    matching allow pattern fails the lint. The grandfather set documents
    pre-existing violations slated for migration to ``jnp.where``.
    """
    violations: list[str] = []
    for lineno, line in _iter_op_lines(path, "jnp.maximum("):
        if any(p.search(line) for p in _ALLOWED_MAXIMUM_PATTERNS):
            continue
        if (path.name, lineno) in _GRANDFATHERED_MAXIMUM:
            continue
        violations.append(f"  {path.name}:{lineno}: {line.strip()}")

    assert not violations, (
        "Unguarded jnp.maximum() found in gradient-bearing module — violates "
        "CLAUDE.md rule #7. ``jnp.maximum(x, eps)`` zeros the gradient below "
        "the floor; use ``jnp.where(x > eps, x, eps)`` for differentiable "
        "floors.\nOffending lines:\n" + "\n".join(violations)
    )


def test_lint_finds_gradient_bearing_files() -> None:
    """Smoke test: the lint must actually scan something, else a future
    file rename could silently disable the entire check."""
    files = _gradient_bearing_source_files()
    assert len(files) >= 5, (
        f"Expected at least 5 gradient-bearing core files, found {len(files)}: "
        f"{[p.name for p in files]}"
    )
    names = {p.name for p in files}
    assert "physics_utils.py" in names
    assert "physics_cmc.py" in names
    assert "theory.py" in names
    assert "models.py" in names
