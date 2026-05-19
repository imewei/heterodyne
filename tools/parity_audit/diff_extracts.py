"""Categorizing differ: turns per-extractor JSON into a categorized REPORT.md.

Severity rubric mirrors spec §4.2:
- P0: silent breakage (signatures, public class shapes, config keys, CLI flags,
      exit codes, Sphinx-build-breaking docs drift)
- P1: structural drift (file added/missing/renamed, missing top-level docs page,
      extra heterodyne-only files)
- P2: observable drift (log format strings, error message stems, help text,
      docs heading drift)
- P3: cosmetic (docstring text, comment density)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Gap = dict[str, Any]


def diff_signatures(
    *, homodyne: dict[str, str], heterodyne: dict[str, str]
) -> list[Gap]:
    gaps: list[Gap] = []
    for qual, sig in sorted(homodyne.items()):
        if qual not in heterodyne:
            gaps.append(
                {
                    "category": "signatures",
                    "severity": "P0",
                    "kind": "missing_in_heterodyne",
                    "qualname": qual,
                    "detail": f"homodyne has `{sig}`; heterodyne missing",
                }
            )
        elif heterodyne[qual] != sig:
            gaps.append(
                {
                    "category": "signatures",
                    "severity": "P0",
                    "kind": "changed",
                    "qualname": qual,
                    "detail": f"homodyne: `{sig}`\n  heterodyne: `{heterodyne[qual]}`",
                }
            )
    for qual, sig in sorted(heterodyne.items()):
        if qual not in homodyne:
            gaps.append(
                {
                    "category": "signatures",
                    "severity": "P1",
                    "kind": "extra_in_heterodyne",
                    "qualname": qual,
                    "detail": f"heterodyne-only: `{sig}`",
                }
            )
    return gaps


def diff_exports(
    *, homodyne: dict[str, list[str]], heterodyne: dict[str, list[str]]
) -> list[Gap]:
    gaps: list[Gap] = []
    for module, homo_list in sorted(homodyne.items()):
        hetero_list = heterodyne.get(module, [])
        missing = sorted(set(homo_list) - set(hetero_list))
        extra = sorted(set(hetero_list) - set(homo_list))
        if missing:
            gaps.append(
                {
                    "category": "exports",
                    "severity": "P0",
                    "kind": "missing_export",
                    "qualname": module,
                    "detail": f"missing from __all__: {missing}",
                }
            )
        if extra:
            gaps.append(
                {
                    "category": "exports",
                    "severity": "P1",
                    "kind": "extra_export",
                    "qualname": module,
                    "detail": f"heterodyne-only in __all__: {extra}",
                }
            )
    return gaps


def diff_classes(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for qual, info in sorted(homodyne.items()):
        if qual not in heterodyne:
            gaps.append(
                {
                    "category": "classes",
                    "severity": "P0",
                    "kind": "missing_class",
                    "qualname": qual,
                    "detail": "homodyne defines this class; heterodyne does not",
                }
            )
            continue
        other = heterodyne[qual]
        for m in info.get("methods", []):
            if m not in other.get("methods", []):
                gaps.append(
                    {
                        "category": "classes",
                        "severity": "P0",
                        "kind": "missing_method",
                        "qualname": f"{qual}.{m.split('(')[0]}",
                        "detail": f"homodyne method `{m}` missing in heterodyne",
                    }
                )
        for f in info.get("dataclass_fields", []):
            if f not in other.get("dataclass_fields", []):
                gaps.append(
                    {
                        "category": "classes",
                        "severity": "P0",
                        "kind": "missing_field",
                        "qualname": f"{qual}.{f}",
                        "detail": f"homodyne dataclass field `{f}` missing in heterodyne",
                    }
                )
    return gaps


def diff_configs(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for key in sorted(homodyne):
        if key not in heterodyne:
            gaps.append(
                {
                    "category": "configs",
                    "severity": "P0",
                    "kind": "missing_config_key",
                    "qualname": key,
                    "detail": f"homodyne defines `{key}`; heterodyne does not",
                }
            )
    for key in sorted(heterodyne):
        if key not in homodyne:
            gaps.append(
                {
                    "category": "configs",
                    "severity": "P1",
                    "kind": "extra_config_key",
                    "qualname": key,
                    "detail": "heterodyne-only config key",
                }
            )
    return gaps


def diff_cli(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    homo_flags = _flatten_cli_flags(homodyne)
    hetero_flags = _flatten_cli_flags(heterodyne)
    for flag, entry in sorted(homo_flags.items()):
        if flag not in hetero_flags:
            gaps.append(
                {
                    "category": "cli",
                    "severity": "P0",
                    "kind": "missing_cli_flag",
                    "qualname": flag,
                    "detail": f"homodyne flag `{flag}` (default={entry.get('default')}) missing in heterodyne",
                }
            )
        elif hetero_flags[flag].get("default") != entry.get("default"):
            gaps.append(
                {
                    "category": "cli",
                    "severity": "P0",
                    "kind": "cli_default_drift",
                    "qualname": flag,
                    "detail": f"default homodyne={entry.get('default')!r} heterodyne={hetero_flags[flag].get('default')!r}",
                }
            )
    for flag in sorted(hetero_flags):
        if flag not in homo_flags:
            gaps.append(
                {
                    "category": "cli",
                    "severity": "P1",
                    "kind": "extra_cli_flag",
                    "qualname": flag,
                    "detail": "heterodyne-only CLI flag",
                }
            )
    return gaps


def _flatten_cli_flags(extract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    for module_data in extract.values():
        for entry in module_data.get("flags", []):
            flat[entry["flag"]] = entry
    return flat


def diff_logs_errors(
    *, homodyne: dict[str, Any], heterodyne: dict[str, Any]
) -> list[Gap]:
    gaps: list[Gap] = []
    homo_exits = _collect_exit_codes(homodyne)
    hetero_exits = _collect_exit_codes(heterodyne)
    if homo_exits != hetero_exits:
        gaps.append(
            {
                "category": "logs_errors",
                "severity": "P0",
                "kind": "exit_code_drift",
                "qualname": "<package>",
                "detail": f"homodyne exit codes={sorted(homo_exits)} heterodyne={sorted(hetero_exits)}",
            }
        )
    for module, info in sorted(homodyne.items()):
        homo_logs = info.get("log_messages", {})
        hetero_logs = heterodyne.get(module, {}).get("log_messages", {})
        for level, msgs in homo_logs.items():
            other = set(hetero_logs.get(level, []))
            for m in msgs:
                if m not in other:
                    gaps.append(
                        {
                            "category": "logs_errors",
                            "severity": "P2",
                            "kind": "log_format_drift",
                            "qualname": f"{module} [{level}]",
                            "detail": f"homodyne logs {m!r}; heterodyne does not (at this level)",
                        }
                    )
    return gaps


def _collect_exit_codes(extract: dict[str, Any]) -> set[int]:
    codes: set[int] = set()
    for info in extract.values():
        for code in info.get("exit_codes", []):
            codes.add(int(code))
    return codes


def diff_docs(*, homodyne: dict[str, Any], heterodyne: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for rel, info in sorted(homodyne.items()):
        if rel not in heterodyne:
            gaps.append(
                {
                    "category": "docs",
                    "severity": "P1",
                    "kind": "missing_doc",
                    "qualname": rel,
                    "detail": f"homodyne has docs page {rel}; heterodyne does not",
                }
            )
            continue
        other = heterodyne[rel]
        homo_targets = set(
            info.get("automodule", [])
            + info.get("autoclass", [])
            + info.get("autofunction", [])
        )
        hetero_targets = set(
            other.get("automodule", [])
            + other.get("autoclass", [])
            + other.get("autofunction", [])
        )
        adapted = {t.replace("homodyne", "heterodyne", 1) for t in homo_targets}
        for target in sorted(adapted - hetero_targets):
            gaps.append(
                {
                    "category": "docs",
                    "severity": "P0",
                    "kind": "broken_autodoc_target",
                    "qualname": rel,
                    "detail": f"expected autodoc target `{target}` not present in heterodyne docs page",
                }
            )
        homo_headings = set(info.get("headings", []))
        hetero_headings = set(other.get("headings", []))
        for h in sorted(homo_headings - hetero_headings):
            gaps.append(
                {
                    "category": "docs",
                    "severity": "P2",
                    "kind": "heading_drift",
                    "qualname": rel,
                    "detail": f"heading `{h}` present in homodyne, absent in heterodyne",
                }
            )
    return gaps


def diff_file_inventory(
    *, homodyne: dict[str, Any], heterodyne: dict[str, Any]
) -> list[Gap]:
    gaps: list[Gap] = []
    homo_py = set(homodyne.get("python_files_non_physics", []))
    hetero_py = set(heterodyne.get("python_files_non_physics", []))
    for missing in sorted(homo_py - hetero_py):
        gaps.append(
            {
                "category": "file_inventory",
                "severity": "P1",
                "kind": "missing_py_file",
                "path": missing,
                "detail": f"homodyne has `{missing}.py`; heterodyne does not",
            }
        )
    for extra in sorted(hetero_py - homo_py):
        gaps.append(
            {
                "category": "file_inventory",
                "severity": "P1",
                "kind": "extra_py_file",
                "path": extra,
                "detail": "heterodyne-only file (candidate for absorb-then-delete)",
            }
        )
    homo_docs = set(homodyne.get("doc_files", []))
    hetero_docs = set(heterodyne.get("doc_files", []))
    for missing in sorted(homo_docs - hetero_docs):
        gaps.append(
            {
                "category": "file_inventory",
                "severity": "P1",
                "kind": "missing_doc_file",
                "path": missing,
                "detail": f"homodyne has docs file `{missing}`; heterodyne does not",
            }
        )
    return gaps


_SEVERITY_ORDER = ["P0", "P1", "P2", "P3"]
_SEVERITY_BLURB = {
    "P0": "Silent breakage — API/config/CLI/exit-code/docs-build drift",
    "P1": "Structural drift — files added/missing/renamed, docs pages missing",
    "P2": "Observable drift — log formats, error stems, docs heading drift",
    "P3": "Cosmetic — docstring/comment differences",
}


def render_report(
    all_gaps: list[Gap], *, homodyne_sha: str, heterodyne_sha: str
) -> str:
    lines: list[str] = []
    lines.append("# Heterodyne → Homodyne Parity Audit Report")
    lines.append("")
    lines.append(f"- Homodyne SHA: `{homodyne_sha}`")
    lines.append(f"- Heterodyne SHA: `{heterodyne_sha}`")
    lines.append(f"- Total gaps: **{len(all_gaps)}**")
    lines.append("")
    by_sev: dict[str, list[Gap]] = {s: [] for s in _SEVERITY_ORDER}
    for g in all_gaps:
        by_sev.setdefault(g["severity"], []).append(g)

    for sev in _SEVERITY_ORDER:
        bucket = by_sev.get(sev, [])
        lines.append(f"## {sev} — {_SEVERITY_BLURB[sev]} ({len(bucket)} gaps)")
        lines.append("")
        if not bucket:
            lines.append("_(none)_")
            lines.append("")
            continue
        by_cat: dict[str, list[Gap]] = {}
        for g in bucket:
            by_cat.setdefault(g["category"], []).append(g)
        for cat in sorted(by_cat):
            lines.append(f"### {cat} ({len(by_cat[cat])})")
            lines.append("")
            lines.append("| Disposition | Kind | Qualname / Path | Detail |")
            lines.append("|---|---|---|---|")
            for g in by_cat[cat]:
                ident = g.get("qualname") or g.get("path", "")
                detail = g["detail"].replace("\n", "<br>")
                lines.append(f"| `KEEP` | {g['kind']} | `{ident}` | {detail} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def run_full_diff(*, homodyne_extracts: Path, heterodyne_extracts: Path) -> list[Gap]:
    pairs: list[tuple[str, Callable[..., list[Gap]]]] = [
        ("signatures.json", diff_signatures),
        ("exports.json", diff_exports),
        ("classes.json", diff_classes),
        ("configs.json", diff_configs),
        ("cli.json", diff_cli),
        ("logs_errors.json", diff_logs_errors),
        ("docs.json", diff_docs),
        ("file_inventory.json", diff_file_inventory),
    ]
    all_gaps: list[Gap] = []
    for filename, fn in pairs:
        homo_path = homodyne_extracts / filename
        hetero_path = heterodyne_extracts / filename
        if not homo_path.exists() or not hetero_path.exists():
            continue
        homo = json.loads(homo_path.read_text())
        hetero = json.loads(hetero_path.read_text())
        all_gaps.extend(fn(homodyne=homo, heterodyne=hetero))
    return all_gaps
