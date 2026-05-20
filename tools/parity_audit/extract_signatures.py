"""Signature extractor: canonical public-function signatures via AST."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.parity_audit.ast_utils import safe_parse
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
    tree = safe_parse(file_path)
    if tree is None:
        return {}
    result: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
            node.name
        ):
            result[f"{module_path}.{node.name}"] = canonical_signature(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            for sub in node.body:
                if isinstance(
                    sub, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and _is_public(sub.name):
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
