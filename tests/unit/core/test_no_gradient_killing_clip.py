"""Lint test: no ``jnp.clip`` on gradient-bearing physics tensors.

CLAUDE.md rule #7 mandates ``jnp.where(x>eps, x, eps)`` or smooth alternatives
(``smooth_clip``, ``smooth_bound``, ``smooth_abs``) over ``jnp.clip`` /
``jnp.maximum`` on tensors that participate in NUTS leapfrog or NLSQ Jacobian
flow.  Hard ``jnp.clip`` zeros the gradient at the boundary and stalls the
sampler / optimizer.

This test scans ``heterodyne/core/physics*.py`` for ``jnp.clip`` calls and
asserts each one matches an explicit allow-listed pattern.  When you add a
new ``jnp.clip``:

  * If it indexes into an array (``jnp.searchsorted`` etc.) — allowed.
  * If it bounds the argument of an overflow-prone scalar function
    (``jnp.exp``, ``jnp.log``) — allowed (the gradient is already
    vanishingly small at the cap).
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
_ALLOWED_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Comments and docstrings referencing jnp.clip are not actual calls
    re.compile(r"^\s*#"),
    re.compile(r'^\s*[\'"]'),
    # Indexing: jnp.clip(jnp.searchsorted(...), 0, n_grid - 1)
    re.compile(r"jnp\.clip\(\s*jnp\.searchsorted"),
    # Overflow guard for exp: jnp.exp(jnp.clip(x, -limit, limit))
    re.compile(r"jnp\.exp\(\s*jnp\.clip\("),
    # Overflow guard for log on positive support: jnp.log(jnp.clip(x, eps, ...))
    re.compile(r"jnp\.log\(\s*jnp\.clip\("),
)


def _physics_source_files() -> list[Path]:
    """Find heterodyne/core/physics*.py files relative to this test."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # tests/unit/core/<file>.py → repo
    core_dir = repo_root / "heterodyne" / "core"
    return sorted(core_dir.glob("physics*.py"))


def _iter_clip_lines(path: Path):
    """Yield ``(lineno, line)`` for every line in ``path`` that contains
    ``jnp.clip(`` (open-paren guarantees it's a call, not a mention)."""
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if "jnp.clip(" in line:
            yield lineno, line


@pytest.mark.parametrize("path", _physics_source_files(), ids=lambda p: p.name)
def test_no_unguarded_clip_in_physics_module(path: Path) -> None:
    """Every ``jnp.clip(...)`` in ``heterodyne/core/physics*.py`` must match
    one of the allow-listed patterns (indexing or overflow guard).

    If this test fails on a new ``jnp.clip`` call, the fix is to:
      1. Switch to ``smooth_clip(x, low, high)`` for physical bounds, OR
      2. Switch to ``jnp.where(cond, x, floor)`` if a one-sided floor is
         intended, OR
      3. Add a new allow-listed pattern here with a comment explaining why
         the call is safe (e.g. the operand has no autodiff dependency).
    """
    violations: list[str] = []
    for lineno, line in _iter_clip_lines(path):
        if any(p.search(line) for p in _ALLOWED_PATTERNS):
            continue
        violations.append(f"  {path.name}:{lineno}: {line.strip()}")

    assert not violations, (
        "Unguarded jnp.clip() found in physics module — violates CLAUDE.md "
        "rule #7 (use smooth_clip/smooth_bound/jnp.where instead).  "
        "See deep-RCA F8 for the failure mode.\n"
        "Offending lines:\n" + "\n".join(violations)
    )


def test_lint_finds_physics_files() -> None:
    """Smoke test: the lint must actually scan something, else a future
    file rename could silently disable the entire check."""
    files = _physics_source_files()
    assert len(files) >= 3, (
        f"Expected at least 3 physics*.py files, found {len(files)}: "
        f"{[p.name for p in files]}"
    )
    names = {p.name for p in files}
    assert "physics_utils.py" in names
    assert "physics_cmc.py" in names
