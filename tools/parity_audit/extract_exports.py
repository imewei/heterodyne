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
