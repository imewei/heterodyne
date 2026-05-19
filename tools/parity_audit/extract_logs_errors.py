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
        "log_messages": {
            k: sorted(set(v)) for k, v in visitor.log_messages.items() if v
        },
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
