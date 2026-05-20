"""CLI extractor: argparse add_argument() flags and add_parser() subcommands."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from tools.parity_audit.ast_utils import (
    safe_parse,
    string_constant,
    string_list_from_node,
)
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
    del module_path  # included in signature for parity with other extractors
    tree = safe_parse(file_path)
    if tree is None:
        return {"flags": [], "subparsers": []}
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
