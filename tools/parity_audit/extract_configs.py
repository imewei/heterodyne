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
        if isinstance(node.value, ast.Name) and node.value.id in {"config", "cfg"}:
            value = string_constant(node.slice)
            if value is not None:
                self.keys.add(value)
        self.generic_visit(node)


def extract_file(file_path: Path, *, module_path: str) -> dict[str, Any]:
    tree = ast.parse(file_path.read_text())
    result: dict[str, Any] = {}

    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and _is_dataclass(node)
            and _is_public(node.name)
        ):
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    if _is_public(sub.target.id):
                        key = f"{module_path}.{node.name}.{sub.target.id}"
                        default = (
                            ast.unparse(sub.value) if sub.value is not None else None
                        )
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
