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
    all_python = sorted(
        _module_path(p, package_root) for p in discover_python_files(package_root)
    )
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


def write_json(
    package_root: Path, out_path: Path, *, docs_root: Path | None = None
) -> None:
    out_path.write_text(
        json.dumps(extract(package_root, docs_root=docs_root), indent=2, sort_keys=True)
    )
