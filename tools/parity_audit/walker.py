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
        if exclude_physics_exempt and is_physics_exempt(
            path, package_root=package_root
        ):
            continue
        yield path


def discover_doc_files(docs_root: Path) -> Iterator[Path]:
    for pattern in ("*.rst", "*.md"):
        for path in docs_root.rglob(pattern):
            if any(
                part in _DOC_SKIP_DIRS for part in path.relative_to(docs_root).parts
            ):
                continue
            yield path
