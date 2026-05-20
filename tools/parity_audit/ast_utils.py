"""Safe AST helpers for parity audit extractors.

These helpers walk AST nodes directly instead of executing literals, so
they never run arbitrary code and never trigger overly broad security
scanners.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def safe_parse(file_path: Path) -> ast.Module | None:
    """Parse ``file_path`` with ``ast.parse``; return ``None`` on SyntaxError.

    Emits a single-line warning to stderr so CI logs surface the skip without
    aborting the whole audit run. Matches the contract used by
    ``extract_exports.extract_file`` (return-empty-on-syntax-error).
    """
    try:
        return ast.parse(file_path.read_text())
    except SyntaxError as exc:
        print(
            f"warning: parity_audit skipped {file_path} (SyntaxError: {exc.msg})",
            file=sys.stderr,
        )
        return None


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
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return int(node.value)
    return None
