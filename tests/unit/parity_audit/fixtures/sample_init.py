"""Synthetic __init__.py for export extractor tests.

The extractor parses this file via AST; it is never imported, so the
stub names below exist only to satisfy Pyright's __all__ membership check.
"""

from __future__ import annotations

A = B = C = None

__all__ = ["B", "A", "C"]
