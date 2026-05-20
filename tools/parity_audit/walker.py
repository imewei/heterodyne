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
_PY_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "tests",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "node_modules",
        "graphify-out",
        ".git",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "_build",
    }
)


def _to_module_path(file_path: Path, package_root: Path) -> str:
    rel = file_path.relative_to(package_root).with_suffix("")
    return ".".join(rel.parts)


def is_physics_exempt(file_path: Path, *, package_root: Path) -> bool:
    return _to_module_path(file_path, package_root) in PHYSICS_EXEMPT_FILES


def _iter_files(root: Path, pattern: str, skip_dirs: frozenset[str]) -> Iterator[Path]:
    """Walk ``root`` for files matching ``pattern``, skipping ``skip_dirs`` and symlinked dirs.

    Symlinked directories are skipped entirely to prevent infinite traversal on
    cyclic links (e.g. ``.venv/lib64 -> .``); symlinked files are still yielded.
    """
    resolved_root = root.resolve()
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_symlink() and entry.is_dir():
                    continue
                if entry.is_dir():
                    if entry.name in skip_dirs:
                        continue
                    # Guard: don't escape root via symlinks to ancestor paths.
                    try:
                        if not entry.resolve().is_relative_to(resolved_root):
                            continue
                    except (OSError, ValueError):
                        continue
                    stack.append(entry)
                elif entry.is_file() and entry.match(pattern):
                    yield entry
            except OSError:
                continue


def discover_python_files(
    package_root: Path,
    *,
    exclude_physics_exempt: bool = False,
) -> Iterator[Path]:
    for path in _iter_files(package_root, "*.py", _PY_SKIP_DIRS):
        if exclude_physics_exempt and is_physics_exempt(
            path, package_root=package_root
        ):
            continue
        yield path


def discover_doc_files(docs_root: Path) -> Iterator[Path]:
    for pattern in ("*.rst", "*.md"):
        yield from _iter_files(docs_root, pattern, _DOC_SKIP_DIRS)
