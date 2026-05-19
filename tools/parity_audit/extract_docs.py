"""Docs extractor: Sphinx headings, autodoc directives, toctree, xrefs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.parity_audit.walker import discover_doc_files

_HEADING_RE = re.compile(
    r"^(?P<text>\S.*\S?)\n(?P<underline>[=\-~^\"*+#`]{3,})$", re.MULTILINE
)
_AUTODOC_RE = re.compile(
    r"^\.\. (automodule|autoclass|autofunction)::\s+(\S+)", re.MULTILINE
)
_TOCTREE_BLOCK_RE = re.compile(
    r"^\.\. toctree::(?P<body>(?:\n(?:[ \t]+.*)?)*)", re.MULTILINE
)
_XREF_RE = re.compile(r":(\w+):`(?:[^<`]+ <)?([^>`]+?)>?`")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_rst(text: str) -> dict[str, Any]:
    headings = [m.group("text").strip() for m in _HEADING_RE.finditer(text)]
    automodule: list[str] = []
    autoclass: list[str] = []
    autofunction: list[str] = []
    for m in _AUTODOC_RE.finditer(text):
        bucket = {
            "automodule": automodule,
            "autoclass": autoclass,
            "autofunction": autofunction,
        }
        bucket[m.group(1)].append(m.group(2))

    toctree: list[str] = []
    for m in _TOCTREE_BLOCK_RE.finditer(text):
        for line in m.group("body").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            toctree.append(stripped)

    xrefs: list[dict[str, str]] = []
    for m in _XREF_RE.finditer(text):
        xrefs.append({"role": m.group(1), "target": m.group(2).strip()})

    return {
        "headings": headings,
        "automodule": sorted(set(automodule)),
        "autoclass": sorted(set(autoclass)),
        "autofunction": sorted(set(autofunction)),
        "toctree": toctree,
        "xrefs": xrefs,
    }


def _extract_md(text: str) -> dict[str, Any]:
    headings = [m.group(2).strip() for m in _MD_HEADING_RE.finditer(text)]
    return {
        "headings": headings,
        "automodule": [],
        "autoclass": [],
        "autofunction": [],
        "toctree": [],
        "xrefs": [],
    }


def extract_file(file_path: Path) -> dict[str, Any]:
    text = file_path.read_text()
    if file_path.suffix == ".md":
        return _extract_md(text)
    return _extract_rst(text)


def extract(docs_root: Path) -> dict[str, Any]:
    if not docs_root.exists():
        return {}
    result: dict[str, Any] = {}
    for file_path in discover_doc_files(docs_root):
        rel = str(file_path.relative_to(docs_root))
        result[rel] = extract_file(file_path)
    return result


def write_json(docs_root: Path, out_path: Path) -> None:
    out_path.write_text(json.dumps(extract(docs_root), indent=2, sort_keys=True))
