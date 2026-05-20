"""Class extractor: bases, public method signatures, dataclass field names."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.ast_utils import safe_parse
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
    tree = safe_parse(file_path)
    if tree is None:
        return {}
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
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _is_public(sub.name)
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
