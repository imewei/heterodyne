# Heterodyne → Homodyne Strict 1:1 Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the audit tooling that makes heterodyne→homodyne parity gaps mechanically discoverable, then run it to produce the dispositionable `REPORT.md` that Phases 2–4 act on.

**Architecture:** A stdlib-only Python package `tools/parity_audit/` containing 8 extractors (one per surface category from spec §4.1), a categorizing differ, and a `python -m tools.parity_audit` CLI. Phase 1 (this plan) ends when `REPORT.md` is committed and the user reviews it. Phases 2–4 are sketched at the bottom as gated downstream work.

**Tech Stack:** Python 3.13, stdlib (`ast`, `pathlib`, `json`, `argparse`, `re`), pytest for TDD.

**Pinned reference:**
- Homodyne SHA: `0368cbdb075fcff1908c0da2b59a1b0d37d5eeca` (recorded in `homodyne_sha.txt` by Task 12)
- Heterodyne SHA at plan creation: `f46c1db7cdb3f882fe9a378e5751e6af98d4d8d8`
- Spec: `docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md`

**File map (created/modified in this plan):**

```
heterodyne/
├── tools/                                              # NEW
│   ├── __init__.py                                     #   (empty)
│   └── parity_audit/                                   # NEW package
│       ├── __init__.py                                 #   exports public API
│       ├── __main__.py                                 #   CLI entry: python -m tools.parity_audit
│       ├── walker.py                                   #   shared: file discovery + exemption filter
│       ├── ast_utils.py                                #   shared: safe AST helpers (string lists, etc.)
│       ├── extract_file_inventory.py
│       ├── extract_signatures.py
│       ├── extract_exports.py
│       ├── extract_classes.py
│       ├── extract_configs.py
│       ├── extract_cli.py
│       ├── extract_logs_errors.py
│       ├── extract_docs.py
│       ├── diff_extracts.py                            #   categorizer → REPORT.md
│       └── README.md                                   #   how to re-run
├── tests/unit/parity_audit/                            # NEW
│   ├── __init__.py
│   ├── fixtures/                                       #   small synthetic .py/.rst inputs
│   │   ├── __init__.py
│   │   ├── sample_signatures.py
│   │   ├── sample_init.py
│   │   ├── sample_classes.py
│   │   ├── sample_config.py
│   │   ├── sample_cli.py
│   │   ├── sample_logs_errors.py
│   │   └── sample_docs.rst
│   ├── test_walker.py
│   ├── test_ast_utils.py
│   ├── test_extract_file_inventory.py
│   ├── test_extract_signatures.py
│   ├── test_extract_exports.py
│   ├── test_extract_classes.py
│   ├── test_extract_configs.py
│   ├── test_extract_cli.py
│   ├── test_extract_logs_errors.py
│   ├── test_extract_docs.py
│   ├── test_diff_extracts.py
│   └── test_cli.py
└── docs/superpowers/audits/                            # NEW (populated by Tasks 12–14)
    └── 2026-05-18-parity-audit/
        ├── homodyne_sha.txt
        ├── heterodyne_sha.txt
        ├── extracts/
        │   ├── homodyne/                               #   per-extractor JSON
        │   └── heterodyne/
        └── REPORT.md
```

---

## Phase 1 — Build the audit tooling and run it

### Task 1: Scaffold `tools/parity_audit/` package + shared walker

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/parity_audit/__init__.py`
- Create: `tools/parity_audit/__main__.py`
- Create: `tools/parity_audit/walker.py`
- Create: `tools/parity_audit/README.md`
- Create: `tests/unit/parity_audit/__init__.py`
- Create: `tests/unit/parity_audit/fixtures/__init__.py`
- Create: `tests/unit/parity_audit/test_walker.py`

- [ ] **Step 1: Write the failing test for walker**

`tests/unit/parity_audit/test_walker.py`:

```python
"""Tests for the shared walker that discovers files for extraction."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.walker import (
    PHYSICS_EXEMPT_FILES,
    discover_doc_files,
    discover_python_files,
    is_physics_exempt,
)


def test_physics_exempt_files_are_module_paths() -> None:
    assert "core.theory" in PHYSICS_EXEMPT_FILES
    assert "core.physics" in PHYSICS_EXEMPT_FILES
    assert "core.physics_cmc" in PHYSICS_EXEMPT_FILES
    assert "core.physics_nlsq" in PHYSICS_EXEMPT_FILES
    assert "core.physics_utils" in PHYSICS_EXEMPT_FILES
    assert "core.jax_backend" in PHYSICS_EXEMPT_FILES


def test_is_physics_exempt_matches_module_path(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    physics_file = pkg / "core" / "theory.py"
    physics_file.write_text("")
    nonphysics_file = pkg / "core" / "models.py"
    nonphysics_file.write_text("")

    assert is_physics_exempt(physics_file, package_root=pkg) is True
    assert is_physics_exempt(nonphysics_file, package_root=pkg) is False


def test_discover_python_files_skips_tests_and_pycache(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "tests").mkdir(parents=True)
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    (pkg / "tests" / "test_models.py").write_text("")
    (pkg / "__pycache__" / "models.cpython-313.pyc").write_text("")

    found = set(discover_python_files(pkg))
    assert pkg / "core" / "models.py" in found
    assert pkg / "tests" / "test_models.py" not in found
    assert pkg / "__pycache__" / "models.cpython-313.pyc" not in found


def test_discover_python_files_excludes_physics_exempt_when_requested(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "theory.py").write_text("")
    (pkg / "core" / "models.py").write_text("")

    found = set(discover_python_files(pkg, exclude_physics_exempt=True))
    assert pkg / "core" / "models.py" in found
    assert pkg / "core" / "theory.py" not in found


def test_discover_doc_files_finds_rst_and_md(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "source"
    (docs / "user_guide").mkdir(parents=True)
    (docs / "user_guide" / "intro.rst").write_text("Intro\n=====\n")
    (docs / "user_guide" / "advanced.md").write_text("# Advanced\n")
    (docs / "_build").mkdir()
    (docs / "_build" / "ignored.rst").write_text("")

    found = set(discover_doc_files(docs))
    assert docs / "user_guide" / "intro.rst" in found
    assert docs / "user_guide" / "advanced.md" in found
    assert docs / "_build" / "ignored.rst" not in found
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_walker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Create empty package files**

`tools/__init__.py`:

```python
```

`tools/parity_audit/__init__.py`:

```python
"""Parity audit tooling for the heterodyne→homodyne 1:1 mirror.

See docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md
for the design this tooling implements.
"""
```

- [ ] **Step 4: Implement `walker.py`**

`tools/parity_audit/walker.py`:

```python
"""Shared file-discovery utilities for parity audit extractors."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# Physics-exempt files (spec §2: narrow physics boundary).
PHYSICS_EXEMPT_FILES: frozenset[str] = frozenset(
    {
        "core.theory",
        "core.physics",
        "core.physics_cmc",
        "core.physics_nlsq",
        "core.physics_utils",
        "core.jax_backend",
    }
)

_DOC_SKIP_DIRS: frozenset[str] = frozenset({"_build", "build", "_autosummary"})
_PY_SKIP_DIRS: frozenset[str] = frozenset({"tests", "__pycache__"})


def _to_module_path(file_path: Path, package_root: Path) -> str:
    rel = file_path.relative_to(package_root).with_suffix("")
    return ".".join(rel.parts)


def is_physics_exempt(file_path: Path, *, package_root: Path) -> bool:
    return _to_module_path(file_path, package_root) in PHYSICS_EXEMPT_FILES


def discover_python_files(
    package_root: Path,
    *,
    exclude_physics_exempt: bool = False,
) -> Iterator[Path]:
    for path in package_root.rglob("*.py"):
        if any(part in _PY_SKIP_DIRS for part in path.relative_to(package_root).parts):
            continue
        if exclude_physics_exempt and is_physics_exempt(path, package_root=package_root):
            continue
        yield path


def discover_doc_files(docs_root: Path) -> Iterator[Path]:
    for pattern in ("*.rst", "*.md"):
        for path in docs_root.rglob(pattern):
            if any(part in _DOC_SKIP_DIRS for part in path.relative_to(docs_root).parts):
                continue
            yield path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/parity_audit/test_walker.py -v`
Expected: 5 passed

- [ ] **Step 6: Create `__main__.py` stub and README**

`tools/parity_audit/__main__.py`:

```python
"""CLI entry point: `python -m tools.parity_audit`.

Subcommands are wired in Task 11 once extractors and the differ exist.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.parity_audit")
    parser.add_argument(
        "subcommand",
        choices=["extract", "diff", "run-all"],
        help="extract: run all extractors against one package. "
        "diff: compare two extract dirs. "
        "run-all: convenience wrapper for extract+diff on both packages.",
    )
    args, _ = parser.parse_known_args(argv)
    print(f"Subcommand {args.subcommand!r} not yet wired (Task 11).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

`tools/parity_audit/README.md`:

````markdown
# Parity Audit Tooling

Mechanical diff between `heterodyne/` and `homodyne/` for strict 1:1 mirror
enforcement. See the spec at
`docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md`.

## Usage

```bash
# From repo root, with both packages checked out side-by-side:
python -m tools.parity_audit run-all \
    --heterodyne /path/to/heterodyne \
    --homodyne /path/to/homodyne \
    --out docs/superpowers/audits/2026-05-18-parity-audit
```

Outputs:
- `extracts/heterodyne/*.json`, `extracts/homodyne/*.json` — raw per-extractor data
- `REPORT.md` — categorized human-readable diff (P0/P1/P2/P3 severity)
- `{homodyne,heterodyne}_sha.txt` — SHA pins

## Extractors

| File | Purpose |
|---|---|
| `extract_file_inventory.py` | File-structure diff |
| `extract_signatures.py` | Public function signatures |
| `extract_exports.py` | `__all__` lists per `__init__.py` |
| `extract_classes.py` | Public class shapes |
| `extract_configs.py` | Config keys |
| `extract_cli.py` | argparse flags + subcommands |
| `extract_logs_errors.py` | Log strings, exception stems, exit codes |
| `extract_docs.py` | Sphinx structure |

## Re-running after a PR

CI runs `python -m tools.parity_audit run-all` and posts the diff vs the pre-PR
baseline as a PR comment. A closed gap that reopens fails the build.
````

- [ ] **Step 7: Verify CLI stub runs**

Run: `python -m tools.parity_audit extract`
Expected: prints `Subcommand 'extract' not yet wired (Task 11).` to stderr, exits 1.

- [ ] **Step 8: Commit**

```bash
git add tools/__init__.py tools/parity_audit/ tests/unit/parity_audit/__init__.py tests/unit/parity_audit/fixtures/__init__.py tests/unit/parity_audit/test_walker.py
git commit -m "feat(parity_audit): scaffold package + shared file walker

Establishes tools/parity_audit/ as the home for parity audit extractors.
walker.py centralizes file discovery and physics-exemption filtering so
every extractor uses the same rules.

Refs: docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md §4.1"
```

---

### Task 2: Build `ast_utils.py` (shared safe AST helpers)

These helpers replace any use of `ast.literal_eval` with explicit AST walking — same outcome, no security-hook false positives.

**Files:**
- Create: `tools/parity_audit/ast_utils.py`
- Create: `tests/unit/parity_audit/test_ast_utils.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/parity_audit/test_ast_utils.py`:

```python
"""Tests for safe AST helpers used by extractors."""
from __future__ import annotations

import ast

from tools.parity_audit.ast_utils import (
    string_list_from_node,
    string_constant,
    int_constant,
)


def test_string_list_from_node_handles_list() -> None:
    node = ast.parse("['a', 'b', 'c']", mode="eval").body
    assert string_list_from_node(node) == ["a", "b", "c"]


def test_string_list_from_node_handles_tuple() -> None:
    node = ast.parse("('x', 'y')", mode="eval").body
    assert string_list_from_node(node) == ["x", "y"]


def test_string_list_from_node_filters_non_strings() -> None:
    node = ast.parse("['a', 1, 'b', None]", mode="eval").body
    assert string_list_from_node(node) == ["a", "b"]


def test_string_list_from_node_returns_empty_for_unknown() -> None:
    node = ast.parse("x", mode="eval").body
    assert string_list_from_node(node) == []


def test_string_list_from_node_handles_none() -> None:
    assert string_list_from_node(None) == []


def test_string_constant_returns_value() -> None:
    node = ast.parse("'hello'", mode="eval").body
    assert string_constant(node) == "hello"


def test_string_constant_returns_none_for_non_string() -> None:
    node = ast.parse("123", mode="eval").body
    assert string_constant(node) is None


def test_int_constant_returns_value() -> None:
    node = ast.parse("42", mode="eval").body
    assert int_constant(node) == 42


def test_int_constant_returns_none_for_non_int() -> None:
    node = ast.parse("'42'", mode="eval").body
    assert int_constant(node) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_ast_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the helpers**

`tools/parity_audit/ast_utils.py`:

```python
"""Safe AST helpers for parity audit extractors.

These helpers walk AST nodes directly instead of executing literals, so
they never run arbitrary code and never trigger overly broad security
scanners.
"""
from __future__ import annotations

import ast


def string_list_from_node(node: ast.expr | None) -> list[str]:
    """Return the string elements of an ast.List / ast.Tuple, in source order.

    Non-string elements are skipped. Returns [] for any other node type.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    result: list[str] = []
    for elt in node.elts:
        value = string_constant(elt)
        if value is not None:
            result.append(value)
    return result


def string_constant(node: ast.expr | None) -> str | None:
    """Return the value of a string ast.Constant, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def int_constant(node: ast.expr | None) -> int | None:
    """Return the value of an integer ast.Constant, or None.

    ast.Constant booleans (True/False) are also instances of int in Python; we
    exclude them explicitly so True doesn't masquerade as 1.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/parity_audit/test_ast_utils.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tools/parity_audit/ast_utils.py tests/unit/parity_audit/test_ast_utils.py
git commit -m "feat(parity_audit): add safe AST literal helpers

string_list_from_node, string_constant, int_constant walk AST nodes
directly so we never execute Python literals from source files.
Replaces what would otherwise be ast.literal_eval calls."
```

---

### Task 3: Build `extract_file_inventory.py`

The simplest extractor — pure file-system walk. Lays the I/O pattern (JSON output to a directory) every later extractor follows.

**Files:**
- Create: `tools/parity_audit/extract_file_inventory.py`
- Create: `tests/unit/parity_audit/test_extract_file_inventory.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/parity_audit/test_extract_file_inventory.py`:

```python
"""Tests for the file-inventory extractor."""
from __future__ import annotations

import json
from pathlib import Path

from tools.parity_audit.extract_file_inventory import extract, write_json


def test_extract_returns_relative_module_paths(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    (pkg / "core" / "theory.py").write_text("")  # physics-exempt
    (pkg / "cli").mkdir()
    (pkg / "cli" / "main.py").write_text("")
    (pkg / "tests").mkdir()
    (pkg / "tests" / "test_x.py").write_text("")  # skipped

    result = extract(pkg)

    assert sorted(result["python_files"]) == ["cli.main", "core.models", "core.theory"]
    assert sorted(result["python_files_non_physics"]) == ["cli.main", "core.models"]
    assert result["physics_exempt_files"] == ["core.theory"]


def test_extract_includes_docs_when_docs_root_present(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    docs = pkg.parent / "docs" / "source"
    docs.mkdir(parents=True)
    (docs / "intro.rst").write_text("Intro\n=====\n")
    (docs / "advanced.md").write_text("# Advanced\n")

    result = extract(pkg, docs_root=docs)

    assert sorted(result["doc_files"]) == ["advanced.md", "intro.rst"]


def test_write_json_round_trips(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")

    out_path = tmp_path / "file_inventory.json"
    write_json(pkg, out_path)
    data = json.loads(out_path.read_text())
    assert "core.models" in data["python_files"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_file_inventory.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the extractor**

`tools/parity_audit/extract_file_inventory.py`:

```python
"""File-inventory extractor: lists source/doc files in a package."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.parity_audit.walker import (
    discover_doc_files,
    discover_python_files,
    is_physics_exempt,
)


def _module_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def extract(package_root: Path, *, docs_root: Path | None = None) -> dict[str, Any]:
    all_python = sorted(_module_path(p, package_root) for p in discover_python_files(package_root))
    non_physics = sorted(
        _module_path(p, package_root)
        for p in discover_python_files(package_root, exclude_physics_exempt=True)
    )
    physics_only = sorted(
        _module_path(p, package_root)
        for p in discover_python_files(package_root)
        if is_physics_exempt(p, package_root=package_root)
    )

    result: dict[str, Any] = {
        "python_files": all_python,
        "python_files_non_physics": non_physics,
        "physics_exempt_files": physics_only,
    }

    if docs_root is not None and docs_root.exists():
        result["doc_files"] = sorted(
            str(p.relative_to(docs_root)) for p in discover_doc_files(docs_root)
        )

    return result


def write_json(package_root: Path, out_path: Path, *, docs_root: Path | None = None) -> None:
    out_path.write_text(json.dumps(extract(package_root, docs_root=docs_root), indent=2, sort_keys=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/parity_audit/test_extract_file_inventory.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/parity_audit/extract_file_inventory.py tests/unit/parity_audit/test_extract_file_inventory.py
git commit -m "feat(parity_audit): add file-inventory extractor

Lists python files (with/without physics exemption) and docs files for a
package, dumped to JSON for the differ to consume.

Refs: spec §4.1 extractor #8"
```

---

### Task 4: Build `extract_signatures.py`

Walks `ast.FunctionDef`/`AsyncFunctionDef` nodes and produces canonical signature strings.

**Files:**
- Create: `tools/parity_audit/extract_signatures.py`
- Create: `tests/unit/parity_audit/test_extract_signatures.py`
- Create: `tests/unit/parity_audit/fixtures/sample_signatures.py`

- [ ] **Step 1: Create the fixture file**

`tests/unit/parity_audit/fixtures/sample_signatures.py`:

```python
"""Synthetic input for signature extractor tests. Do not import in real code."""
from __future__ import annotations


def public_fn(a: int, b: str = "x") -> bool:
    return True


def _private_fn(a: int) -> None:
    return None


async def async_public(a: int) -> int:
    return a


class Cls:
    def public_method(self, x: int) -> int:
        return x

    def _private_method(self) -> None:
        return None

    @staticmethod
    def static_method(x: int) -> int:
        return x


def fn_with_complex_types(
    items: list[dict[str, int]],
    *,
    callback: "Callable[[int], int] | None" = None,
) -> dict[str, list[int]]:
    return {}
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_signatures.py`:

```python
"""Tests for the signature extractor."""
from __future__ import annotations

import ast
from pathlib import Path

from tools.parity_audit.extract_signatures import canonical_signature, extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_signatures.py"


def test_extract_skips_private_functions() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    names = set(sigs.keys())
    assert "fixtures.sample_signatures.public_fn" in names
    assert "fixtures.sample_signatures._private_fn" not in names


def test_extract_handles_async() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    assert "fixtures.sample_signatures.async_public" in sigs


def test_extract_public_method_includes_self() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.Cls.public_method"]
    assert "self" in sig
    assert "x: int" in sig
    assert "-> int" in sig


def test_extract_skips_private_methods() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    assert "fixtures.sample_signatures.Cls._private_method" not in sigs


def test_canonical_signature_has_stable_form() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.public_fn"]
    assert sig == "public_fn(a: int, b: str = 'x') -> bool"


def test_canonical_signature_round_trips_kwonly() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.fn_with_complex_types"]
    assert "*" in sig  # kwonly marker
    assert "callback:" in sig


def test_canonical_signature_helper_directly() -> None:
    tree = ast.parse("def foo(x: int = 1) -> str: ...")
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert canonical_signature(fn) == "foo(x: int = 1) -> str"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_signatures.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_signatures.py`:

```python
"""Signature extractor: canonical public-function signatures via AST."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.parity_audit.walker import discover_python_files


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def canonical_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = fn.args
    parts: list[str] = []

    pos_only = list(args.posonlyargs)
    pos = list(args.args)
    defaults = list(args.defaults)
    n_defaulted = len(defaults)
    n_no_default = len(pos_only) + len(pos) - n_defaulted

    flat_pos = pos_only + pos
    for idx, arg in enumerate(flat_pos):
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        if idx >= n_no_default:
            default = defaults[idx - n_no_default]
            parts.append(f"{arg.arg}{ann} = {_unparse(default)}")
        else:
            parts.append(f"{arg.arg}{ann}")
        if pos_only and idx == len(pos_only) - 1:
            parts.append("/")

    if args.vararg is not None:
        ann = f": {_unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        if default is not None:
            parts.append(f"{arg.arg}{ann} = {_unparse(default)}")
        else:
            parts.append(f"{arg.arg}{ann}")

    if args.kwarg is not None:
        ann = f": {_unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    ret = f" -> {_unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({', '.join(parts)}){ret}"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def extract_file(file_path: Path, *, module_path: str) -> dict[str, str]:
    tree = ast.parse(file_path.read_text())
    result: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            result[f"{module_path}.{node.name}"] = canonical_signature(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(sub.name):
                    qual = f"{module_path}.{node.name}.{sub.name}"
                    result[qual] = canonical_signature(sub)
    return result


def extract(package_root: Path) -> dict[str, str]:
    all_sigs: dict[str, str] = {}
    for file_path in discover_python_files(package_root, exclude_physics_exempt=True):
        rel = file_path.relative_to(package_root).with_suffix("")
        module_path = ".".join(rel.parts)
        all_sigs.update(extract_file(file_path, module_path=module_path))
    return all_sigs


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_signatures.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_signatures.py tests/unit/parity_audit/test_extract_signatures.py tests/unit/parity_audit/fixtures/sample_signatures.py
git commit -m "feat(parity_audit): add canonical signature extractor

Produces stable, sorted {qualname: signature} JSON for every public
function and method in a package. Skips private names and physics-exempt
files.

Refs: spec §4.1 extractor #1"
```

---

### Task 5: Build `extract_exports.py`

Parses each `__init__.py` for `__all__` to detect public-API surface drift. Uses the safe `string_list_from_node` helper from `ast_utils.py`.

**Files:**
- Create: `tools/parity_audit/extract_exports.py`
- Create: `tests/unit/parity_audit/test_extract_exports.py`
- Create: `tests/unit/parity_audit/fixtures/sample_init.py`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_init.py`:

```python
"""Synthetic __init__.py for export extractor tests."""
from __future__ import annotations

__all__ = ["B", "A", "C"]
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_exports.py`:

```python
"""Tests for the __all__ exports extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_exports import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_init.py"


def test_extract_returns_sorted_all() -> None:
    exports = extract_file(FIXTURE, module_path="fixtures.sample_init")
    assert exports == ["A", "B", "C"]


def test_extract_missing_all_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "noall.py"
    f.write_text("x = 1\n")
    assert extract_file(f, module_path="noall") == []


def test_extract_handles_tuple_literal(tmp_path: Path) -> None:
    f = tmp_path / "tup.py"
    f.write_text('__all__ = ("a", "b")\n')
    assert extract_file(f, module_path="tup") == ["a", "b"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_exports.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_exports.py`:

```python
"""Exports extractor: per-module __all__ lists."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.parity_audit.ast_utils import string_list_from_node
from tools.parity_audit.walker import discover_python_files


def extract_file(file_path: Path, *, module_path: str) -> list[str]:
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return sorted(string_list_from_node(node.value))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                return sorted(string_list_from_node(node.value))
    return []


def extract(package_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for file_path in discover_python_files(package_root, exclude_physics_exempt=False):
        if file_path.name != "__init__.py":
            continue
        rel = file_path.parent.relative_to(package_root)
        module_path = ".".join(rel.parts) if rel.parts else ""
        result[module_path] = extract_file(file_path, module_path=module_path)
    return result


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_exports.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_exports.py tests/unit/parity_audit/test_extract_exports.py tests/unit/parity_audit/fixtures/sample_init.py
git commit -m "feat(parity_audit): add __all__ exports extractor

Parses every __init__.py and emits the sorted __all__ membership list.
Set comparison so source-order reordering does not produce false drift.

Refs: spec §4.1 extractor #2"
```

---

### Task 6: Build `extract_classes.py`

**Files:**
- Create: `tools/parity_audit/extract_classes.py`
- Create: `tests/unit/parity_audit/test_extract_classes.py`
- Create: `tests/unit/parity_audit/fixtures/sample_classes.py`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_classes.py`:

```python
"""Synthetic input for class extractor tests."""
from __future__ import annotations

from dataclasses import dataclass


class PlainClass:
    def public_method(self, x: int) -> int:
        return x

    def _private(self) -> None:
        return None


class Subclass(PlainClass):
    def another(self) -> str:
        return ""


@dataclass(frozen=True)
class DataCls:
    name: str
    value: int = 0
    _private: str = "hidden"


class _PrivateClass:
    pass
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_classes.py`:

```python
"""Tests for the class-shape extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_classes import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_classes.py"


def test_plain_class_captured() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    plain = classes["fixtures.sample_classes.PlainClass"]
    assert plain["bases"] == []
    assert "public_method(self, x: int) -> int" in plain["methods"]
    assert all("_private" not in m for m in plain["methods"])


def test_subclass_records_bases() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    sub = classes["fixtures.sample_classes.Subclass"]
    assert sub["bases"] == ["PlainClass"]


def test_dataclass_fields_captured() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    dc = classes["fixtures.sample_classes.DataCls"]
    assert dc["dataclass_fields"] == ["name", "value"]


def test_private_class_skipped() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    assert "fixtures.sample_classes._PrivateClass" not in classes
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_classes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_classes.py`:

```python
"""Class extractor: bases, public method signatures, dataclass field names."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.extract_signatures import canonical_signature
from tools.parity_audit.walker import discover_python_files


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for dec in cls.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


def extract_file(file_path: Path, *, module_path: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(file_path.read_text())
    result: dict[str, dict[str, Any]] = {}

    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and _is_public(node.name)):
            continue
        bases: list[str] = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                continue

        methods = sorted(
            canonical_signature(sub)
            for sub in node.body
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(sub.name)
        )

        dataclass_fields: list[str] = []
        if _is_dataclass(node):
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    if _is_public(sub.target.id):
                        dataclass_fields.append(sub.target.id)

        result[f"{module_path}.{node.name}"] = {
            "bases": bases,
            "methods": methods,
            "dataclass_fields": dataclass_fields,
        }
    return result


def extract(package_root: Path) -> dict[str, dict[str, Any]]:
    all_classes: dict[str, dict[str, Any]] = {}
    for file_path in discover_python_files(package_root, exclude_physics_exempt=True):
        rel = file_path.relative_to(package_root).with_suffix("")
        module_path = ".".join(rel.parts)
        all_classes.update(extract_file(file_path, module_path=module_path))
    return all_classes


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_classes.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_classes.py tests/unit/parity_audit/test_extract_classes.py tests/unit/parity_audit/fixtures/sample_classes.py
git commit -m "feat(parity_audit): add class-shape extractor

For each public class: base classes, sorted public method signatures,
and (for @dataclass) public field names.

Refs: spec §4.1 extractor #3"
```

---

### Task 7: Build `extract_configs.py`

**Files:**
- Create: `tools/parity_audit/extract_configs.py`
- Create: `tests/unit/parity_audit/test_extract_configs.py`
- Create: `tests/unit/parity_audit/fixtures/sample_config.py`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_config.py`:

```python
"""Synthetic input for config extractor tests."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CMCConfig:
    target_accept: float = 0.8
    max_r_hat: float = 1.01
    nlsq_prior_width_factor: float = 1.0


def use_config(config: dict) -> tuple:
    a = config.get("optimization.cmc.dense_mass")
    b = config["optimization.nlsq.tolerance"]
    return a, b
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_configs.py`:

```python
"""Tests for the config-key extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_configs import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_config.py"


def test_dataclass_fields_yield_config_keys() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "fixtures.sample_config.CMCConfig.target_accept" in keys
    assert "fixtures.sample_config.CMCConfig.max_r_hat" in keys


def test_config_get_calls_captured() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "optimization.cmc.dense_mass" in keys["fixtures.sample_config.runtime_keys"]


def test_config_subscript_captured() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "optimization.nlsq.tolerance" in keys["fixtures.sample_config.runtime_keys"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_configs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_configs.py`:

```python
"""Config-key extractor.

Two complementary sources:
1. Dataclass fields in any */config.py file (declared config schema)
2. String literals passed to config.get("X") / config["X"] (runtime usage)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.ast_utils import string_constant
from tools.parity_audit.walker import discover_python_files


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for dec in cls.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


class _RuntimeKeyVisitor(ast.NodeVisitor):
    """Collect string keys from config.get("X") and config["X"] patterns."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"config", "cfg"}
            and node.args
        ):
            value = string_constant(node.args[0])
            if value is not None:
                self.keys.add(value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in {"config", "cfg"}
        ):
            value = string_constant(node.slice)
            if value is not None:
                self.keys.add(value)
        self.generic_visit(node)


def extract_file(file_path: Path, *, module_path: str) -> dict[str, Any]:
    tree = ast.parse(file_path.read_text())
    result: dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_dataclass(node) and _is_public(node.name):
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    if _is_public(sub.target.id):
                        key = f"{module_path}.{node.name}.{sub.target.id}"
                        default = ast.unparse(sub.value) if sub.value is not None else None
                        result[key] = {"default": default}

    visitor = _RuntimeKeyVisitor()
    visitor.visit(tree)
    if visitor.keys:
        result[f"{module_path}.runtime_keys"] = sorted(visitor.keys)
    return result


def extract(package_root: Path) -> dict[str, Any]:
    all_keys: dict[str, Any] = {}
    for file_path in discover_python_files(package_root, exclude_physics_exempt=True):
        rel = file_path.relative_to(package_root).with_suffix("")
        module_path = ".".join(rel.parts)
        all_keys.update(extract_file(file_path, module_path=module_path))
    return all_keys


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_configs.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_configs.py tests/unit/parity_audit/test_extract_configs.py tests/unit/parity_audit/fixtures/sample_config.py
git commit -m "feat(parity_audit): add config-key extractor

Captures (a) dataclass fields declared in */config.py and (b) string
keys used via config.get(\"X\") / config[\"X\"].

Refs: spec §4.1 extractor #4"
```

---

### Task 8: Build `extract_cli.py`

**Files:**
- Create: `tools/parity_audit/extract_cli.py`
- Create: `tests/unit/parity_audit/test_extract_cli.py`
- Create: `tests/unit/parity_audit/fixtures/sample_cli.py`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_cli.py`:

```python
"""Synthetic input for CLI extractor tests."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", default="default.yaml", help="Path to YAML")
    parser.add_argument(
        "--method",
        choices=["nlsq", "cmc"],
        default="nlsq",
        help="Which optimizer to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.add_parser("fit", aliases=["f"])
    subparsers.add_parser("plot")
    return parser
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_cli.py`:

```python
"""Tests for the CLI extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_cli import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_cli.py"


def test_flags_captured_with_defaults_and_choices() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    flags = {f["flag"]: f for f in result["flags"]}
    assert "--config" in flags
    assert flags["--config"]["dest"] == "'config_path'"
    assert flags["--config"]["default"] == "'default.yaml'"

    assert "--method" in flags
    assert flags["--method"]["choices"] == ["nlsq", "cmc"]


def test_short_aliases_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    flags = {f["flag"]: f for f in result["flags"]}
    assert "--verbose" in flags
    assert "-v" in flags["--verbose"]["aliases"]


def test_subparsers_captured_with_aliases() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    subs = {s["name"]: s for s in result["subparsers"]}
    assert "fit" in subs
    assert subs["fit"]["aliases"] == ["f"]
    assert "plot" in subs
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_cli.py`:

```python
"""CLI extractor: argparse add_argument() flags and add_parser() subcommands."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.ast_utils import string_constant, string_list_from_node
from tools.parity_audit.walker import discover_python_files


def _node_repr(node: ast.expr | None) -> Any:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _flag_name(args: list[ast.expr]) -> tuple[str, list[str]]:
    names: list[str] = []
    for arg in args:
        value = string_constant(arg)
        if value is not None:
            names.append(value)
    if not names:
        return "", []
    long = [n for n in names if n.startswith("--")]
    if long:
        canonical = max(long, key=len)
    else:
        canonical = names[0]
    return canonical, [n for n in names if n != canonical]


class _CLIVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.flags: list[dict[str, Any]] = []
        self.subparsers: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                self._record_add_argument(node)
            elif node.func.attr == "add_parser":
                self._record_add_parser(node)
        self.generic_visit(node)

    def _record_add_argument(self, node: ast.Call) -> None:
        canonical, aliases = _flag_name(node.args)
        if not canonical:
            return
        kw = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        entry: dict[str, Any] = {
            "flag": canonical,
            "aliases": aliases,
            "dest": _node_repr(kw.get("dest")),
            "default": _node_repr(kw.get("default")),
            "help": _node_repr(kw.get("help")),
        }
        if "choices" in kw:
            entry["choices"] = string_list_from_node(kw["choices"])
        if "action" in kw:
            entry["action"] = _node_repr(kw["action"])
        self.flags.append(entry)

    def _record_add_parser(self, node: ast.Call) -> None:
        name = ""
        if node.args:
            value = string_constant(node.args[0])
            if value is not None:
                name = value
        if not name:
            return
        kw = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        aliases = string_list_from_node(kw.get("aliases"))
        self.subparsers.append({"name": name, "aliases": aliases})


def extract_file(file_path: Path, *, module_path: str) -> dict[str, Any]:
    tree = ast.parse(file_path.read_text())
    visitor = _CLIVisitor()
    visitor.visit(tree)
    return {"flags": visitor.flags, "subparsers": visitor.subparsers}


def extract(package_root: Path) -> dict[str, Any]:
    all_cli: dict[str, Any] = {}
    cli_dir = package_root / "cli"
    if not cli_dir.exists():
        return all_cli
    for file_path in discover_python_files(cli_dir):
        rel = file_path.relative_to(package_root).with_suffix("")
        module_path = ".".join(rel.parts)
        data = extract_file(file_path, module_path=module_path)
        if data["flags"] or data["subparsers"]:
            all_cli[module_path] = data
    return all_cli


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_cli.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_cli.py tests/unit/parity_audit/test_extract_cli.py tests/unit/parity_audit/fixtures/sample_cli.py
git commit -m "feat(parity_audit): add CLI surface extractor

Captures argparse.add_argument flags and add_parser subcommands.
Drift here is P0 because users wire shell scripts to these names.

Refs: spec §4.1 extractor #5"
```

---

### Task 9: Build `extract_logs_errors.py`

**Files:**
- Create: `tools/parity_audit/extract_logs_errors.py`
- Create: `tests/unit/parity_audit/test_extract_logs_errors.py`
- Create: `tests/unit/parity_audit/fixtures/sample_logs_errors.py`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_logs_errors.py`:

```python
"""Synthetic input for logs/errors extractor tests."""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def do_work(x: int) -> int:
    logger.info("Starting work for x=%d", x)
    if x < 0:
        logger.warning("Negative x: %d", x)
        raise ValueError("x must be non-negative")
    if x > 1000:
        logger.error("x out of range")
        sys.exit(2)
    return x
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_logs_errors.py`:

```python
"""Tests for the logs/errors extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_logs_errors import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_logs_errors.py"


def test_log_format_strings_captured_by_level() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    assert "Starting work for x=%d" in result["log_messages"]["info"]
    assert "Negative x: %d" in result["log_messages"]["warning"]
    assert "x out of range" in result["log_messages"]["error"]


def test_exception_stems_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    raises = result["raises"]
    assert any(r["exception"] == "ValueError" and "non-negative" in r["message"] for r in raises)


def test_exit_codes_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    assert 2 in result["exit_codes"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_logs_errors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_logs_errors.py`:

```python
"""Logs/errors extractor: log format strings, raise stems, exit codes."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.ast_utils import int_constant, string_constant
from tools.parity_audit.walker import discover_python_files

_LOG_LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}


class _LogsErrorsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.log_messages: dict[str, list[str]] = {lvl: [] for lvl in _LOG_LEVELS}
        self.raises: list[dict[str, str]] = []
        self.exit_codes: set[int] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_LEVELS:
            if node.args:
                value = string_constant(node.args[0])
                if value is not None:
                    self.log_messages[node.func.attr].append(value)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "exit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.args
        ):
            code = int_constant(node.args[0])
            if code is not None:
                self.exit_codes.add(code)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc_name = ast.unparse(exc.func)
            msg = ""
            if exc.args:
                value = string_constant(exc.args[0])
                if value is not None:
                    msg = value
            self.raises.append({"exception": exc_name, "message": msg})
        elif isinstance(exc, ast.Name):
            self.raises.append({"exception": exc.id, "message": ""})
        self.generic_visit(node)


def extract_file(file_path: Path, *, module_path: str) -> dict[str, Any]:
    tree = ast.parse(file_path.read_text())
    visitor = _LogsErrorsVisitor()
    visitor.visit(tree)
    return {
        "log_messages": {k: sorted(set(v)) for k, v in visitor.log_messages.items() if v},
        "raises": sorted(visitor.raises, key=lambda r: (r["exception"], r["message"])),
        "exit_codes": sorted(visitor.exit_codes),
    }


def extract(package_root: Path) -> dict[str, Any]:
    all_data: dict[str, Any] = {}
    for file_path in discover_python_files(package_root, exclude_physics_exempt=True):
        rel = file_path.relative_to(package_root).with_suffix("")
        module_path = ".".join(rel.parts)
        data = extract_file(file_path, module_path=module_path)
        if data["log_messages"] or data["raises"] or data["exit_codes"]:
            all_data[module_path] = data
    return all_data


def write_json(package_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(package_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_logs_errors.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_logs_errors.py tests/unit/parity_audit/test_extract_logs_errors.py tests/unit/parity_audit/fixtures/sample_logs_errors.py
git commit -m "feat(parity_audit): add logs/errors/exit-code extractor

Refs: spec §4.1 extractor #6"
```

---

### Task 10: Build `extract_docs.py`

**Files:**
- Create: `tools/parity_audit/extract_docs.py`
- Create: `tests/unit/parity_audit/test_extract_docs.py`
- Create: `tests/unit/parity_audit/fixtures/sample_docs.rst`

- [ ] **Step 1: Create fixture**

`tests/unit/parity_audit/fixtures/sample_docs.rst`:

```rst
NLSQ Optimization
=================

Overview
--------

The :func:`heterodyne.fit_nlsq_jax` function runs trust-region LM.
See also :class:`heterodyne.NLSQConfig`.

API
---

.. automodule:: heterodyne.optimization.nlsq
   :members:

.. autoclass:: heterodyne.NLSQConfig
   :members:

.. autofunction:: heterodyne.fit_nlsq_jax

.. toctree::
   :maxdepth: 2

   intro
   advanced

See also :ref:`anti-degeneracy-overview`.
```

- [ ] **Step 2: Write the failing test**

`tests/unit/parity_audit/test_extract_docs.py`:

```python
"""Tests for the docs extractor."""
from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_docs import extract_file


FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs.rst"


def test_headings_captured() -> None:
    result = extract_file(FIXTURE)
    assert "NLSQ Optimization" in result["headings"]
    assert "Overview" in result["headings"]
    assert "API" in result["headings"]


def test_autodoc_directives_captured() -> None:
    result = extract_file(FIXTURE)
    assert "heterodyne.optimization.nlsq" in result["automodule"]
    assert "heterodyne.NLSQConfig" in result["autoclass"]
    assert "heterodyne.fit_nlsq_jax" in result["autofunction"]


def test_toctree_entries_captured() -> None:
    result = extract_file(FIXTURE)
    assert "intro" in result["toctree"]
    assert "advanced" in result["toctree"]


def test_cross_references_captured() -> None:
    result = extract_file(FIXTURE)
    xrefs = result["xrefs"]
    assert {"role": "func", "target": "heterodyne.fit_nlsq_jax"} in xrefs
    assert {"role": "class", "target": "heterodyne.NLSQConfig"} in xrefs
    assert {"role": "ref", "target": "anti-degeneracy-overview"} in xrefs
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_extract_docs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the extractor**

`tools/parity_audit/extract_docs.py`:

```python
"""Docs extractor: Sphinx headings, autodoc directives, toctree, xrefs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.parity_audit.walker import discover_doc_files

_HEADING_RE = re.compile(r"^(?P<text>\S.*\S?)\n(?P<underline>[=\-~^\"*+#`]{3,})$", re.MULTILINE)
_AUTODOC_RE = re.compile(r"^\.\. (automodule|autoclass|autofunction)::\s+(\S+)", re.MULTILINE)
_TOCTREE_BLOCK_RE = re.compile(
    r"^\.\. toctree::(?P<body>(?:\n[ \t]+.*)*)", re.MULTILINE
)
_XREF_RE = re.compile(r":(\w+):`(?:[^<`]+ <)?([^>`]+?)>?`")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_rst(text: str) -> dict[str, Any]:
    headings = [m.group("text").strip() for m in _HEADING_RE.finditer(text)]
    automodule: list[str] = []
    autoclass: list[str] = []
    autofunction: list[str] = []
    for m in _AUTODOC_RE.finditer(text):
        bucket = {"automodule": automodule, "autoclass": autoclass, "autofunction": autofunction}
        bucket[m.group(1)].append(m.group(2))

    toctree: list[str] = []
    for m in _TOCTREE_BLOCK_RE.finditer(text):
        for line in m.group("body").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            toctree.append(stripped)

    xrefs: list[dict[str, str]] = []
    for m in _XREF_RE.finditer(text):
        xrefs.append({"role": m.group(1), "target": m.group(2).strip()})

    return {
        "headings": headings,
        "automodule": sorted(set(automodule)),
        "autoclass": sorted(set(autoclass)),
        "autofunction": sorted(set(autofunction)),
        "toctree": toctree,
        "xrefs": xrefs,
    }


def _extract_md(text: str) -> dict[str, Any]:
    headings = [m.group(2).strip() for m in _MD_HEADING_RE.finditer(text)]
    return {
        "headings": headings,
        "automodule": [],
        "autoclass": [],
        "autofunction": [],
        "toctree": [],
        "xrefs": [],
    }


def extract_file(file_path: Path) -> dict[str, Any]:
    text = file_path.read_text()
    if file_path.suffix == ".md":
        return _extract_md(text)
    return _extract_rst(text)


def extract(docs_root: Path) -> dict[str, Any]:
    if not docs_root.exists():
        return {}
    result: dict[str, Any] = {}
    for file_path in discover_doc_files(docs_root):
        rel = str(file_path.relative_to(docs_root))
        result[rel] = extract_file(file_path)
    return result


def write_json(docs_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(docs_root), indent=2, sort_keys=True))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/parity_audit/test_extract_docs.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add tools/parity_audit/extract_docs.py tests/unit/parity_audit/test_extract_docs.py tests/unit/parity_audit/fixtures/sample_docs.rst
git commit -m "feat(parity_audit): add docs structure extractor

Captures Sphinx headings, autodoc directives, toctree entries, and
cross-references for both .rst and .md docs.

Refs: spec §4.1 extractor #7"
```

---

### Task 11: Build `diff_extracts.py` (the categorizer)

**Files:**
- Create: `tools/parity_audit/diff_extracts.py`
- Create: `tests/unit/parity_audit/test_diff_extracts.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/parity_audit/test_diff_extracts.py`:

```python
"""Tests for the diff/categorizer module."""
from __future__ import annotations

from tools.parity_audit.diff_extracts import (
    diff_classes,
    diff_cli,
    diff_configs,
    diff_docs,
    diff_exports,
    diff_file_inventory,
    diff_logs_errors,
    diff_signatures,
)


def test_diff_signatures_flags_missing_and_changed() -> None:
    homo = {
        "m.foo": "foo(a: int) -> int",
        "m.bar": "bar() -> None",
    }
    hetero = {
        "m.foo": "foo(a: str) -> int",  # changed → P0
        # m.bar missing → P0
    }
    gaps = diff_signatures(homodyne=homo, heterodyne=hetero)
    severities = {g["severity"] for g in gaps}
    assert severities == {"P0"}
    assert any(g["qualname"] == "m.foo" and g["kind"] == "changed" for g in gaps)
    assert any(g["qualname"] == "m.bar" and g["kind"] == "missing_in_heterodyne" for g in gaps)


def test_diff_signatures_flags_heterodyne_only_as_p1() -> None:
    homo: dict[str, str] = {}
    hetero = {"m.extra": "extra() -> None"}
    gaps = diff_signatures(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P1"
    assert gaps[0]["kind"] == "extra_in_heterodyne"


def test_diff_exports_set_membership_only() -> None:
    homo = {"m": ["A", "B"]}
    hetero = {"m": ["B", "A"]}
    assert diff_exports(homodyne=homo, heterodyne=hetero) == []


def test_diff_exports_missing_export_is_p0() -> None:
    homo = {"m": ["A", "B"]}
    hetero = {"m": ["A"]}
    gaps = diff_exports(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"
    assert "B" in gaps[0]["detail"]


def test_diff_file_inventory_missing_py_is_p1() -> None:
    homo = {"python_files_non_physics": ["a.b"], "doc_files": []}
    hetero = {"python_files_non_physics": [], "doc_files": []}
    gaps = diff_file_inventory(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P1" and g["path"] == "a.b" for g in gaps)


def test_diff_file_inventory_extra_in_heterodyne_is_p1() -> None:
    homo = {"python_files_non_physics": [], "doc_files": []}
    hetero = {"python_files_non_physics": ["cmc.warmstart"], "doc_files": []}
    gaps = diff_file_inventory(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P1" and "warmstart" in g["path"] for g in gaps)


def test_diff_cli_missing_flag_is_p0() -> None:
    homo = {"cli.main": {"flags": [{"flag": "--method", "default": "'nlsq'"}], "subparsers": []}}
    hetero = {"cli.main": {"flags": [], "subparsers": []}}
    gaps = diff_cli(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" and "--method" in g["detail"] for g in gaps)


def test_diff_configs_missing_key_is_p0() -> None:
    homo = {"cmc.config.CMCConfig.target_accept": {"default": "0.8"}}
    hetero: dict = {}
    gaps = diff_configs(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"


def test_diff_logs_errors_format_string_is_p2() -> None:
    homo = {"m": {"log_messages": {"info": ["Starting %s"]}, "raises": [], "exit_codes": []}}
    hetero = {"m": {"log_messages": {"info": ["Beginning %s"]}, "raises": [], "exit_codes": []}}
    gaps = diff_logs_errors(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P2"


def test_diff_logs_errors_exit_code_is_p0() -> None:
    homo = {"m": {"log_messages": {}, "raises": [], "exit_codes": [2]}}
    hetero = {"m": {"log_messages": {}, "raises": [], "exit_codes": [1]}}
    gaps = diff_logs_errors(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" for g in gaps)


def test_diff_docs_missing_page_is_p1() -> None:
    homo = {"theory/anti_degeneracy.rst": {"headings": ["Anti-Deg"], "automodule": [], "autoclass": [], "autofunction": [], "toctree": [], "xrefs": []}}
    hetero: dict = {}
    gaps = diff_docs(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P1"


def test_diff_docs_broken_autodoc_target_is_p0() -> None:
    homo = {"api/nlsq.rst": {"headings": [], "automodule": ["homodyne.optimization.nlsq"], "autoclass": [], "autofunction": [], "toctree": [], "xrefs": []}}
    hetero = {"api/nlsq.rst": {"headings": [], "automodule": ["heterodyne.optimization.nlsq.NONEXISTENT"], "autoclass": [], "autofunction": [], "toctree": [], "xrefs": []}}
    gaps = diff_docs(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" for g in gaps)


def test_diff_classes_missing_method_is_p0() -> None:
    homo = {"m.Cls": {"bases": [], "methods": ["foo(self) -> int"], "dataclass_fields": []}}
    hetero = {"m.Cls": {"bases": [], "methods": [], "dataclass_fields": []}}
    gaps = diff_classes(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_diff_extracts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the differ**

`tools/parity_audit/diff_extracts.py`:

```python
"""Categorizing differ: turns per-extractor JSON into a categorized REPORT.md.

Severity rubric mirrors spec §4.2:
- P0: silent breakage (signatures, public class shapes, config keys, CLI flags,
      exit codes, Sphinx-build-breaking docs drift)
- P1: structural drift (file added/missing/renamed, missing top-level docs page,
      extra heterodyne-only files)
- P2: observable drift (log format strings, error message stems, help text,
      docs heading drift)
- P3: cosmetic (docstring text, comment density)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

Gap = dict[str, Any]


def diff_signatures(*, homodyne: dict[str, str], heterodyne: dict[str, str]) -> list[Gap]:
    gaps: list[Gap] = []
    for qual, sig in sorted(homodyne.items()):
        if qual not in heterodyne:
            gaps.append({
                "category": "signatures",
                "severity": "P0",
                "kind": "missing_in_heterodyne",
                "qualname": qual,
                "detail": f"homodyne has `{sig}`; heterodyne missing",
            })
        elif heterodyne[qual] != sig:
            gaps.append({
                "category": "signatures",
                "severity": "P0",
                "kind": "changed",
                "qualname": qual,
                "detail": f"homodyne: `{sig}`\n  heterodyne: `{heterodyne[qual]}`",
            })
    for qual, sig in sorted(heterodyne.items()):
        if qual not in homodyne:
            gaps.append({
                "category": "signatures",
                "severity": "P1",
                "kind": "extra_in_heterodyne",
                "qualname": qual,
                "detail": f"heterodyne-only: `{sig}`",
            })
    return gaps


def diff_exports(*, homodyne: dict[str, list[str]], heterodyne: dict[str, list[str]]) -> list[Gap]:
    gaps: list[Gap] = []
    for module, homo_list in sorted(homodyne.items()):
        hetero_list = heterodyne.get(module, [])
        missing = sorted(set(homo_list) - set(hetero_list))
        extra = sorted(set(hetero_list) - set(homo_list))
        if missing:
            gaps.append({
                "category": "exports",
                "severity": "P0",
                "kind": "missing_export",
                "qualname": module,
                "detail": f"missing from __all__: {missing}",
            })
        if extra:
            gaps.append({
                "category": "exports",
                "severity": "P1",
                "kind": "extra_export",
                "qualname": module,
                "detail": f"heterodyne-only in __all__: {extra}",
            })
    return gaps


def diff_classes(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for qual, info in sorted(homodyne.items()):
        if qual not in heterodyne:
            gaps.append({
                "category": "classes",
                "severity": "P0",
                "kind": "missing_class",
                "qualname": qual,
                "detail": "homodyne defines this class; heterodyne does not",
            })
            continue
        other = heterodyne[qual]
        for m in info.get("methods", []):
            if m not in other.get("methods", []):
                gaps.append({
                    "category": "classes",
                    "severity": "P0",
                    "kind": "missing_method",
                    "qualname": f"{qual}.{m.split('(')[0]}",
                    "detail": f"homodyne method `{m}` missing in heterodyne",
                })
        for f in info.get("dataclass_fields", []):
            if f not in other.get("dataclass_fields", []):
                gaps.append({
                    "category": "classes",
                    "severity": "P0",
                    "kind": "missing_field",
                    "qualname": f"{qual}.{f}",
                    "detail": f"homodyne dataclass field `{f}` missing in heterodyne",
                })
    return gaps


def diff_configs(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for key in sorted(homodyne):
        if key not in heterodyne:
            gaps.append({
                "category": "configs",
                "severity": "P0",
                "kind": "missing_config_key",
                "qualname": key,
                "detail": f"homodyne defines `{key}`; heterodyne does not",
            })
    for key in sorted(heterodyne):
        if key not in homodyne:
            gaps.append({
                "category": "configs",
                "severity": "P1",
                "kind": "extra_config_key",
                "qualname": key,
                "detail": "heterodyne-only config key",
            })
    return gaps


def diff_cli(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    homo_flags = _flatten_cli_flags(homodyne)
    hetero_flags = _flatten_cli_flags(heterodyne)
    for flag, entry in sorted(homo_flags.items()):
        if flag not in hetero_flags:
            gaps.append({
                "category": "cli",
                "severity": "P0",
                "kind": "missing_cli_flag",
                "qualname": flag,
                "detail": f"homodyne flag `{flag}` (default={entry.get('default')}) missing in heterodyne",
            })
        elif hetero_flags[flag].get("default") != entry.get("default"):
            gaps.append({
                "category": "cli",
                "severity": "P0",
                "kind": "cli_default_drift",
                "qualname": flag,
                "detail": f"default homodyne={entry.get('default')!r} heterodyne={hetero_flags[flag].get('default')!r}",
            })
    for flag in sorted(hetero_flags):
        if flag not in homo_flags:
            gaps.append({
                "category": "cli",
                "severity": "P1",
                "kind": "extra_cli_flag",
                "qualname": flag,
                "detail": "heterodyne-only CLI flag",
            })
    return gaps


def _flatten_cli_flags(extract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    for module_data in extract.values():
        for entry in module_data.get("flags", []):
            flat[entry["flag"]] = entry
    return flat


def diff_logs_errors(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    homo_exits = _collect_exit_codes(homodyne)
    hetero_exits = _collect_exit_codes(heterodyne)
    if homo_exits != hetero_exits:
        gaps.append({
            "category": "logs_errors",
            "severity": "P0",
            "kind": "exit_code_drift",
            "qualname": "<package>",
            "detail": f"homodyne exit codes={sorted(homo_exits)} heterodyne={sorted(hetero_exits)}",
        })
    for module, info in sorted(homodyne.items()):
        homo_logs = info.get("log_messages", {})
        hetero_logs = heterodyne.get(module, {}).get("log_messages", {})
        for level, msgs in homo_logs.items():
            other = set(hetero_logs.get(level, []))
            for m in msgs:
                if m not in other:
                    gaps.append({
                        "category": "logs_errors",
                        "severity": "P2",
                        "kind": "log_format_drift",
                        "qualname": f"{module} [{level}]",
                        "detail": f"homodyne logs {m!r}; heterodyne does not (at this level)",
                    })
    return gaps


def _collect_exit_codes(extract: dict[str, Any]) -> set[int]:
    codes: set[int] = set()
    for info in extract.values():
        for code in info.get("exit_codes", []):
            codes.add(int(code))
    return codes


def diff_docs(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for rel, info in sorted(homodyne.items()):
        if rel not in heterodyne:
            gaps.append({
                "category": "docs",
                "severity": "P1",
                "kind": "missing_doc",
                "qualname": rel,
                "detail": f"homodyne has docs page {rel}; heterodyne does not",
            })
            continue
        other = heterodyne[rel]
        homo_targets = set(info.get("automodule", []) + info.get("autoclass", []) + info.get("autofunction", []))
        hetero_targets = set(other.get("automodule", []) + other.get("autoclass", []) + other.get("autofunction", []))
        adapted = {t.replace("homodyne", "heterodyne", 1) for t in homo_targets}
        for target in sorted(adapted - hetero_targets):
            gaps.append({
                "category": "docs",
                "severity": "P0",
                "kind": "broken_autodoc_target",
                "qualname": rel,
                "detail": f"expected autodoc target `{target}` not present in heterodyne docs page",
            })
        homo_headings = set(info.get("headings", []))
        hetero_headings = set(other.get("headings", []))
        for h in sorted(homo_headings - hetero_headings):
            gaps.append({
                "category": "docs",
                "severity": "P2",
                "kind": "heading_drift",
                "qualname": rel,
                "detail": f"heading `{h}` present in homodyne, absent in heterodyne",
            })
    return gaps


def diff_file_inventory(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    homo_py = set(homodyne.get("python_files_non_physics", []))
    hetero_py = set(heterodyne.get("python_files_non_physics", []))
    for missing in sorted(homo_py - hetero_py):
        gaps.append({
            "category": "file_inventory",
            "severity": "P1",
            "kind": "missing_py_file",
            "path": missing,
            "detail": f"homodyne has `{missing}.py`; heterodyne does not",
        })
    for extra in sorted(hetero_py - homo_py):
        gaps.append({
            "category": "file_inventory",
            "severity": "P1",
            "kind": "extra_py_file",
            "path": extra,
            "detail": "heterodyne-only file (candidate for absorb-then-delete)",
        })
    homo_docs = set(homodyne.get("doc_files", []))
    hetero_docs = set(heterodyne.get("doc_files", []))
    for missing in sorted(homo_docs - hetero_docs):
        gaps.append({
            "category": "file_inventory",
            "severity": "P1",
            "kind": "missing_doc_file",
            "path": missing,
            "detail": f"homodyne has docs file `{missing}`; heterodyne does not",
        })
    return gaps


_SEVERITY_ORDER = ["P0", "P1", "P2", "P3"]
_SEVERITY_BLURB = {
    "P0": "Silent breakage — API/config/CLI/exit-code/docs-build drift",
    "P1": "Structural drift — files added/missing/renamed, docs pages missing",
    "P2": "Observable drift — log formats, error stems, docs heading drift",
    "P3": "Cosmetic — docstring/comment differences",
}


def render_report(all_gaps: list[Gap], *, homodyne_sha: str, heterodyne_sha: str) -> str:
    lines: list[str] = []
    lines.append("# Heterodyne → Homodyne Parity Audit Report")
    lines.append("")
    lines.append(f"- Homodyne SHA: `{homodyne_sha}`")
    lines.append(f"- Heterodyne SHA: `{heterodyne_sha}`")
    lines.append(f"- Total gaps: **{len(all_gaps)}**")
    lines.append("")
    by_sev: dict[str, list[Gap]] = {s: [] for s in _SEVERITY_ORDER}
    for g in all_gaps:
        by_sev.setdefault(g["severity"], []).append(g)

    for sev in _SEVERITY_ORDER:
        bucket = by_sev.get(sev, [])
        lines.append(f"## {sev} — {_SEVERITY_BLURB[sev]} ({len(bucket)} gaps)")
        lines.append("")
        if not bucket:
            lines.append("_(none)_")
            lines.append("")
            continue
        by_cat: dict[str, list[Gap]] = {}
        for g in bucket:
            by_cat.setdefault(g["category"], []).append(g)
        for cat in sorted(by_cat):
            lines.append(f"### {cat} ({len(by_cat[cat])})")
            lines.append("")
            lines.append("| Disposition | Kind | Qualname / Path | Detail |")
            lines.append("|---|---|---|---|")
            for g in by_cat[cat]:
                ident = g.get("qualname") or g.get("path", "")
                detail = g["detail"].replace("\n", "<br>")
                lines.append(f"| `KEEP` | {g['kind']} | `{ident}` | {detail} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def run_full_diff(*, homodyne_extracts: Path, heterodyne_extracts: Path) -> list[Gap]:
    pairs: list[tuple[str, Callable[..., list[Gap]]]] = [
        ("signatures.json", diff_signatures),
        ("exports.json", diff_exports),
        ("classes.json", diff_classes),
        ("configs.json", diff_configs),
        ("cli.json", diff_cli),
        ("logs_errors.json", diff_logs_errors),
        ("docs.json", diff_docs),
        ("file_inventory.json", diff_file_inventory),
    ]
    all_gaps: list[Gap] = []
    for filename, fn in pairs:
        homo_path = homodyne_extracts / filename
        hetero_path = heterodyne_extracts / filename
        if not homo_path.exists() or not hetero_path.exists():
            continue
        homo = json.loads(homo_path.read_text())
        hetero = json.loads(hetero_path.read_text())
        all_gaps.extend(fn(homodyne=homo, heterodyne=hetero))
    return all_gaps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/parity_audit/test_diff_extracts.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add tools/parity_audit/diff_extracts.py tests/unit/parity_audit/test_diff_extracts.py
git commit -m "feat(parity_audit): add categorizing differ → REPORT.md

Each diff_* function takes per-extractor JSON for both packages and
emits typed Gap dicts with severity per spec §4.2. render_report()
groups them by severity then category and defaults every row to
disposition KEEP for cheap Phase 2 review.

Refs: spec §4.2"
```

---

### Task 12: Wire `python -m tools.parity_audit` CLI

**Files:**
- Modify: `tools/parity_audit/__main__.py`
- Modify: `tools/parity_audit/__init__.py`
- Create: `tests/unit/parity_audit/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

`tests/unit/parity_audit/test_cli.py`:

```python
"""End-to-end CLI smoke tests for `python -m tools.parity_audit`."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_pkg(root: Path, name: str, module_text: str) -> Path:
    pkg = root / name
    (pkg / "core").mkdir(parents=True)
    (pkg / "__init__.py").write_text("__all__ = ['foo']\n")
    (pkg / "core" / "__init__.py").write_text("")
    (pkg / "core" / "models.py").write_text(module_text)
    return pkg


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_extract_subcommand_writes_json(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path, "samplepkg", "def foo(x: int) -> int:\n    return x\n")
    out = tmp_path / "extracts"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "tools.parity_audit", "extract", "--package", str(pkg), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "signatures.json").exists()
    sigs = json.loads((out / "signatures.json").read_text())
    assert any("foo(x: int)" in v for v in sigs.values())


def test_diff_subcommand_writes_report(tmp_path: Path) -> None:
    homo_pkg = _make_pkg(tmp_path, "homo", "def foo() -> int:\n    return 1\n")
    hetero_pkg = _make_pkg(tmp_path, "het", "def foo(x: int) -> int:\n    return x\n")
    homo_out = tmp_path / "homo_extracts"
    hetero_out = tmp_path / "het_extracts"
    homo_out.mkdir()
    hetero_out.mkdir()

    for pkg, out in [(homo_pkg, homo_out), (hetero_pkg, hetero_out)]:
        subprocess.run(
            [sys.executable, "-m", "tools.parity_audit", "extract", "--package", str(pkg), "--out", str(out)],
            check=True,
            cwd=REPO_ROOT,
        )

    report = tmp_path / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.parity_audit",
            "diff",
            "--homodyne",
            str(homo_out),
            "--heterodyne",
            str(hetero_out),
            "--out",
            str(report),
            "--homodyne-sha",
            "deadbeef",
            "--heterodyne-sha",
            "cafef00d",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    body = report.read_text()
    assert "Parity Audit Report" in body
    assert "deadbeef" in body
    assert "P0" in body  # signature drift on foo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/parity_audit/test_cli.py -v`
Expected: FAIL (current `__main__.py` is the stub from Task 1)

- [ ] **Step 3: Update `__init__.py` to expose extractor modules**

`tools/parity_audit/__init__.py`:

```python
"""Parity audit tooling for the heterodyne→homodyne 1:1 mirror.

See docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md
for the design this tooling implements.
"""
from __future__ import annotations

from tools.parity_audit import (
    ast_utils,
    diff_extracts,
    extract_classes,
    extract_cli,
    extract_configs,
    extract_docs,
    extract_exports,
    extract_file_inventory,
    extract_logs_errors,
    extract_signatures,
    walker,
)

__all__ = [
    "ast_utils",
    "diff_extracts",
    "extract_classes",
    "extract_cli",
    "extract_configs",
    "extract_docs",
    "extract_exports",
    "extract_file_inventory",
    "extract_logs_errors",
    "extract_signatures",
    "walker",
]
```

- [ ] **Step 4: Implement the CLI**

Replace `tools/parity_audit/__main__.py`:

```python
"""CLI entry point: `python -m tools.parity_audit`."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.parity_audit import (
    diff_extracts,
    extract_classes,
    extract_cli as extract_cli_mod,
    extract_configs,
    extract_docs,
    extract_exports,
    extract_file_inventory,
    extract_logs_errors,
    extract_signatures,
)


def _cmd_extract(args: argparse.Namespace) -> int:
    package = Path(args.package).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    docs_root = Path(args.docs).resolve() if args.docs else package.parent / "docs" / "source"

    extract_signatures.write_json(package, out / "signatures.json")
    extract_exports.write_json(package, out / "exports.json")
    extract_classes.write_json(package, out / "classes.json")
    extract_configs.write_json(package, out / "configs.json")
    extract_cli_mod.write_json(package, out / "cli.json")
    extract_logs_errors.write_json(package, out / "logs_errors.json")
    extract_docs.write_json(docs_root, out / "docs.json")
    extract_file_inventory.write_json(
        package,
        out / "file_inventory.json",
        docs_root=docs_root if docs_root.exists() else None,
    )
    print(f"Wrote 8 extracts to {out}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    homo_dir = Path(args.homodyne).resolve()
    hetero_dir = Path(args.heterodyne).resolve()
    out = Path(args.out).resolve()
    all_gaps = diff_extracts.run_full_diff(homodyne_extracts=homo_dir, heterodyne_extracts=hetero_dir)
    body = diff_extracts.render_report(
        all_gaps, homodyne_sha=args.homodyne_sha, heterodyne_sha=args.heterodyne_sha
    )
    out.write_text(body)
    print(f"Wrote {len(all_gaps)} gaps to {out}")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    homo_pkg = Path(args.homodyne).resolve()
    hetero_pkg = Path(args.heterodyne).resolve()
    out = Path(args.out).resolve()
    homo_extracts = out / "extracts" / "homodyne"
    hetero_extracts = out / "extracts" / "heterodyne"
    homo_extracts.mkdir(parents=True, exist_ok=True)
    hetero_extracts.mkdir(parents=True, exist_ok=True)

    def _run_extract(pkg: Path, out_dir: Path) -> None:
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.parity_audit",
                "extract",
                "--package",
                str(pkg),
                "--out",
                str(out_dir),
            ],
            check=False,
        ).returncode
        if rc != 0:
            raise SystemExit(rc)

    _run_extract(homo_pkg, homo_extracts)
    _run_extract(hetero_pkg, hetero_extracts)

    homo_sha = subprocess.check_output(
        ["git", "-C", str(homo_pkg.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    hetero_sha = subprocess.check_output(
        ["git", "-C", str(hetero_pkg.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    (out / "homodyne_sha.txt").write_text(homo_sha + "\n")
    (out / "heterodyne_sha.txt").write_text(hetero_sha + "\n")

    report = out / "REPORT.md"
    all_gaps = diff_extracts.run_full_diff(
        homodyne_extracts=homo_extracts, heterodyne_extracts=hetero_extracts
    )
    body = diff_extracts.render_report(all_gaps, homodyne_sha=homo_sha, heterodyne_sha=hetero_sha)
    report.write_text(body)
    print(f"Run-all complete. {len(all_gaps)} gaps. Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.parity_audit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Run all extractors against one package")
    p_extract.add_argument("--package", required=True)
    p_extract.add_argument("--out", required=True)
    p_extract.add_argument("--docs", help="Path to docs/source/ (defaults to ../docs/source)")
    p_extract.set_defaults(func=_cmd_extract)

    p_diff = sub.add_parser("diff", help="Diff two extract dirs and emit REPORT.md")
    p_diff.add_argument("--homodyne", required=True)
    p_diff.add_argument("--heterodyne", required=True)
    p_diff.add_argument("--out", required=True)
    p_diff.add_argument("--homodyne-sha", default="unknown")
    p_diff.add_argument("--heterodyne-sha", default="unknown")
    p_diff.set_defaults(func=_cmd_diff)

    p_run = sub.add_parser("run-all", help="extract + diff for both packages, pin SHAs")
    p_run.add_argument("--homodyne", required=True)
    p_run.add_argument("--heterodyne", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/parity_audit/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Run the full parity_audit suite once for regression**

Run: `uv run pytest tests/unit/parity_audit/ -v`
Expected: all green (~45 tests across all extractors + differ + CLI + ast_utils + walker)

- [ ] **Step 7: Commit**

```bash
git add tools/parity_audit/__main__.py tools/parity_audit/__init__.py tests/unit/parity_audit/test_cli.py
git commit -m "feat(parity_audit): wire python -m tools.parity_audit CLI

Three subcommands: extract (one package → JSON), diff (two extract dirs
→ REPORT.md), run-all (convenience wrapper that pins SHAs and produces
extracts + REPORT in one shot).

Refs: spec §4.4"
```

---

### Task 13: Run the audit against real homodyne + heterodyne

This task produces real artifacts. There is no test — it is the production run.

**Files:**
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/extracts/homodyne/*.json` (8 files)
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/extracts/heterodyne/*.json` (8 files)
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/homodyne_sha.txt`
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/heterodyne_sha.txt`
- Create: `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md`

- [ ] **Step 1: Create the audit output directory**

Run: `mkdir -p docs/superpowers/audits/2026-05-18-parity-audit/extracts/{homodyne,heterodyne}`

- [ ] **Step 2: Run the full audit**

Run:
```bash
python -m tools.parity_audit run-all \
    --homodyne /home/wei/Documents/GitHub/homodyne/homodyne \
    --heterodyne /home/wei/Documents/GitHub/heterodyne/heterodyne \
    --out docs/superpowers/audits/2026-05-18-parity-audit
```

Expected stdout:
```
Wrote 8 extracts to .../extracts/homodyne
Wrote 8 extracts to .../extracts/heterodyne
Run-all complete. NNN gaps. Report: .../REPORT.md
```

(`NNN` will be the actual gap count discovered.)

- [ ] **Step 3: Sanity-check the SHA pins**

Run:
```bash
cat docs/superpowers/audits/2026-05-18-parity-audit/homodyne_sha.txt
cat docs/superpowers/audits/2026-05-18-parity-audit/heterodyne_sha.txt
```

Expected:
- Homodyne SHA matches `git -C /home/wei/Documents/GitHub/homodyne rev-parse HEAD`
- Heterodyne SHA matches `git rev-parse HEAD`

- [ ] **Step 4: Sanity-check the report structure**

Run: `head -40 docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md`

Expected:
- Title line `# Heterodyne → Homodyne Parity Audit Report`
- SHA pins
- "Total gaps: **N**"
- "## P0 — Silent breakage" section header

If the report is empty or malformed, debug `run-all` before committing.

- [ ] **Step 5: Commit the audit artifacts (force-add past gitignore)**

Run:
```bash
git add -f docs/superpowers/audits/2026-05-18-parity-audit/
git commit -m "audit(parity): initial heterodyne→homodyne 1:1 parity report

REPORT.md categorizes gaps by P0/P1/P2/P3 severity per spec §4.2.
Phase 2 review: walk REPORT.md and write disposition (KEEP/DROP/DEFER/WAIVE)
for each row into DISPOSITIONS.md."
```

---

### Task 14: Manual narrative diff pass (7 dynamic-behavior modules)

AST extractors don't see runtime sequencing. For the modules below, write a 1-paragraph diff appended to `REPORT.md` under a new `## Manual narrative diffs` section.

**Files:**
- Modify: `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md`

- [ ] **Step 1: Diff `optimization/nlsq/fallback_chain.py`**

```bash
diff -u /home/wei/Documents/GitHub/homodyne/homodyne/optimization/nlsq/fallback_chain.py \
        /home/wei/Documents/GitHub/heterodyne/heterodyne/optimization/nlsq/fallback_chain.py | head -100
```

Look for: order of strategies tried, abort conditions, retry semantics. Write one paragraph naming any difference, or "no behavior drift detected" if none.

- [ ] **Step 2: Diff `optimization/cmc/sampler.py`**

Focus: retry-on-divergence semantics, warmup ramp, mass-matrix adaptation triggers.

- [ ] **Step 3: Diff `cli/commands.py`**

Focus: subcommand routing precedence, fail-fast vs warn-on conditions.

- [ ] **Step 4: Diff `optimization/nlsq/anti_degeneracy_controller.py`**

Focus: layer invocation order, trigger conditions. NOTE: heterodyne intentionally omits Layer 5 (shear weighting); that absence is not drift.

- [ ] **Step 5: Diff `optimization/nlsq/recovery.py`**

Focus: recovery action ordering, escalation thresholds.

- [ ] **Step 6: Diff NLSQ → CMC warm-start handoff**

Compare heterodyne's `optimization/cmc/warmstart.py` (slated for absorption into `priors.py`) against homodyne's `optimization/cmc/priors.py` warm-start helpers. Document any shape/semantic difference the absorption PR must preserve.

- [ ] **Step 7: Diff `cli/data_pipeline.py`**

Focus: data validation order, fail-fast conditions, NaN/shape gating.

- [ ] **Step 8: Append the narrative section to REPORT.md**

Append to `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md`:

```markdown
## Manual narrative diffs (runtime behavior not captured by extractors)

### fallback_chain.py
[Your one-paragraph diff]

### cmc/sampler.py
[Your one-paragraph diff]

### cli/commands.py
[Your one-paragraph diff]

### anti_degeneracy_controller.py
[Your one-paragraph diff]

### recovery.py
[Your one-paragraph diff]

### NLSQ → CMC warm-start handoff
[Your one-paragraph diff — call out shape changes the absorb-then-delete of warmstart.py must preserve]

### data_pipeline.py
[Your one-paragraph diff]
```

- [ ] **Step 9: Commit**

```bash
git add docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md
git commit -m "audit(parity): append manual narrative diffs to REPORT.md

7 dynamic-behavior modules walked by eye for runtime semantics that
AST extractors cannot see (strategy ordering, retry semantics, gating
precedence)."
```

---

## Phase 1 done. Stage gate.

After Task 14 commits, **stop** and tell the user:

> Phase 1 complete. The audit report is at
> `docs/superpowers/audits/2026-05-18-parity-audit/REPORT.md`
> with N gaps (P0: A, P1: B, P2: C, P3: D).
>
> **Your turn:** walk the report and for each row decide
> KEEP / DROP / DEFER / WAIVE. Default is KEEP — only override exceptions.
> Save dispositions to `DISPOSITIONS.md` next to REPORT.md.
>
> When ready, run the writing-plans skill again with the DISPOSITIONS.md
> as input to produce the Phase 3 execution plan.

Do not start Phase 3 work in the same session — the review is a stage gate.

---

## Phase 2 — User review gate (no implementation tasks)

User walks `REPORT.md` and produces `DISPOSITIONS.md`. No code work. This phase ends when `DISPOSITIONS.md` is committed.

**Disposition tokens:**
- `KEEP` — fix in Phase 4 (default)
- `DROP` — keep heterodyne as-is; logged in `DIVERGENCE_REGISTRY.md` with one-line rationale
- `DEFER` — backlog; not in this work
- `WAIVE` — covered by another row; skip

---

## Phase 3 — Plan generation (sketched; produced after Phase 2)

Re-invoke `superpowers:writing-plans` against `KEEP` rows. The plan will materialize as approximately the following PR sequence — exact tasks depend on what Phase 1 surfaces:

| PR # | Title | Rough content |
|---|---|---|
| 1 | NLSQ structural alignment (P1) | Add any missing files (e.g., `wrapper.py` if dispositioned KEEP), reconcile `__init__.py` exports, port homodyne-only module sketches adapted for 14-param model |
| 2 | NLSQ signature/class drift (P0) | Align each drifted signature, update call sites; per-PR audit re-run asserts category drops to 0 |
| 3 | CMC structural alignment + attr renames (P1 + P0) | Rename `multiprocessing_backend.py`→`multiprocessing.py`, `pjit_backend.py`→`pjit.py`. Absorb `warmstart.py` into `priors.py`. Absorb `prior_builder.py` into `priors.py`. Decide `cpu_backend.py`. Revert CMC attribute names. Remove `from_dict()` legacy shim. Update CLAUDE.md (delete lines 21–25). |
| 4 | CMC signature/config drift (P0) | Per-row alignment. |
| 5 | CLI parity (P0/P1) | Align argparse flags, dests, defaults, subcommand aliases. |
| 6 | Viz parity (P1/P2) | Align viz module structure, plot function signatures. |
| 7 | Data/IO/Utils/Device parity (P1/P2) | Module-level renames, signature alignment. |
| 8 | Logs/errors/exit codes (P2/P0) | Align format strings and raised message stems; revert exit-code drift. |
| 9 | Docs structural fill (P1) | Port the ~22 missing docs files from homodyne, adapt for 14-param model + no-shear. |
| 10 | Docs autodoc + xref fixup (P0/P2) | Make `make docs` warning-free; fix every broken `:func:`/`:class:`/`:ref:` and every `automodule::` target after the renames above. |
| 11 | Parity CI gate | Add `.github/workflows/parity-audit-no-regression.yml` that runs `python -m tools.parity_audit run-all` on every PR and fails if any closed gap reopens. |

Each PR is independently mergeable. Each PR's CI step re-runs the audit and asserts the targeted gap category dropped to 0 and no closed gap reopened.

---

## Phase 4 — Execution (sketched; per Phase 3 plan)

### Execution rules (apply to every PR)

**Rule 1: Absorb-then-delete (no orphan deletions).** For every heterodyne-only file slated for removal:

1. Find the homodyne file that owns the equivalent functionality.
2. Port the heterodyne-only functionality into that homodyne-named file.
3. Replace all heterodyne call sites.
4. Run full test suite — must be green.
5. Only then delete the heterodyne-only file.
6. Re-run tests — still green.

**Rule 2: Re-run audit per PR.** PR cannot merge if it didn't close what it claimed to close, or if it reopened something else.

**Rule 3: TDD for new ports.** Genuine ports follow `superpowers:test-driven-development`: failing test first.

**Rule 4: Docs ride with code.** Doc updates ship in the same PR as the code change that caused the drift. No standalone "docs cleanup" PR.

### Phase 4 completion criteria

1. `python -m tools.parity_audit run-all ...` reports **zero P0, zero P1, zero P2** gaps (excluding `DROP`/`DEFER` rows in `DISPOSITIONS.md`).
2. `DIVERGENCE_REGISTRY.md` contains exactly the dispositioned `DROP` rows with rationale.
3. Full test suite green (~3000 tests).
4. CI gate `parity-audit-no-regression.yml` is active and passing.
5. `make docs` completes with zero warnings.

---

## Self-review

**Spec coverage:** Each spec section maps to plan content:
- §1 Goal → plan goal
- §2 Scope decisions → plan header + Phase 4 Rule 1
- §3 Four phases → plan's four-phase structure
- §4.1 7 extractors → Tasks 3–10 (one task each, plus shared `ast_utils.py` in Task 2 and shared `walker.py` in Task 1)
- §4.2 severity rubric → Task 11 differ implementation + tests
- §4.3 manual narrative → Task 14
- §4.4 artifact paths → Task 13 + Phase 1 file map
- §5 review tokens → Phase 2 section
- §6 PR plan → Phase 3 table
- §7 execution rules → Phase 4 rules
- §8 cross-cutting concerns → covered in Phase 3/4 sketches
- §9 risk register → enforced via Rule 1 (R1, R4, R8), Rule 2 (R5), test gates (R7)
- §10 testing → per-PR gates listed under Phase 4
- §11 success criteria → "Phase 4 completion criteria"
- §12 out of scope → not in any task
- §13 artifact summary → plan File map + Tasks 13–14

**Placeholder scan:** None. Every step has actual code or actual commands. Manual narrative diffs in Task 14 are bracketed placeholders intentionally — the operator types the paragraph after running each `diff` command.

**Type consistency:** `extract()` and `write_json()` signatures are uniform across all 8 extractor modules. `Gap` dict shape (`category`, `severity`, `kind`, `qualname`/`path`, `detail`) is used consistently by all `diff_*` functions and consumed by `render_report()`. CLI subcommands (`extract`, `diff`, `run-all`) match the README example. `string_constant`, `string_list_from_node`, `int_constant` from `ast_utils.py` are used identically in `extract_exports.py`, `extract_cli.py`, `extract_configs.py`, `extract_logs_errors.py`.
