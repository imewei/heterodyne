"""Tests for the diff/categorizer module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.parity_audit.diff_extracts import (
    MissingExtractError,
    diff_classes,
    diff_cli,
    diff_configs,
    diff_docs,
    diff_exports,
    diff_file_inventory,
    diff_logs_errors,
    diff_signatures,
    render_report,
    run_full_diff,
)


def test_diff_signatures_flags_missing_and_changed() -> None:
    homo = {
        "m.foo": "foo(a: int) -> int",
        "m.bar": "bar() -> None",
    }
    hetero = {
        "m.foo": "foo(a: str) -> int",  # changed → P0
        # m.bar missing → P0
    }
    gaps = diff_signatures(homodyne=homo, heterodyne=hetero)
    severities = {g["severity"] for g in gaps}
    assert severities == {"P0"}
    assert any(g["qualname"] == "m.foo" and g["kind"] == "changed" for g in gaps)
    assert any(
        g["qualname"] == "m.bar" and g["kind"] == "missing_in_heterodyne" for g in gaps
    )


def test_diff_signatures_flags_heterodyne_only_as_p1() -> None:
    homo: dict[str, str] = {}
    hetero = {"m.extra": "extra() -> None"}
    gaps = diff_signatures(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P1"
    assert gaps[0]["kind"] == "extra_in_heterodyne"


def test_diff_exports_set_membership_only() -> None:
    homo = {"m": ["A", "B"]}
    hetero = {"m": ["B", "A"]}
    assert diff_exports(homodyne=homo, heterodyne=hetero) == []


def test_diff_exports_missing_export_is_p0() -> None:
    homo = {"m": ["A", "B"]}
    hetero = {"m": ["A"]}
    gaps = diff_exports(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"
    assert "B" in gaps[0]["detail"]


def test_diff_file_inventory_missing_py_is_p1() -> None:
    homo = {"python_files_non_physics": ["a.b"], "doc_files": []}
    hetero = {"python_files_non_physics": [], "doc_files": []}
    gaps = diff_file_inventory(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P1" and g["path"] == "a.b" for g in gaps)


def test_diff_file_inventory_extra_in_heterodyne_is_p1() -> None:
    homo = {"python_files_non_physics": [], "doc_files": []}
    hetero = {"python_files_non_physics": ["cmc.warmstart"], "doc_files": []}
    gaps = diff_file_inventory(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P1" and "warmstart" in g["path"] for g in gaps)


def test_diff_cli_missing_flag_is_p0() -> None:
    homo = {
        "cli.main": {
            "flags": [{"flag": "--method", "default": "'nlsq'"}],
            "subparsers": [],
        }
    }
    hetero = {"cli.main": {"flags": [], "subparsers": []}}
    gaps = diff_cli(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" and "--method" in g["detail"] for g in gaps)


def test_diff_configs_missing_key_is_p0() -> None:
    homo = {"cmc.config.CMCConfig.target_accept": {"default": "0.8"}}
    hetero: dict = {}
    gaps = diff_configs(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"


def test_diff_logs_errors_format_string_is_p2() -> None:
    homo = {
        "m": {"log_messages": {"info": ["Starting %s"]}, "raises": [], "exit_codes": []}
    }
    hetero = {
        "m": {
            "log_messages": {"info": ["Beginning %s"]},
            "raises": [],
            "exit_codes": [],
        }
    }
    gaps = diff_logs_errors(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P2"


def test_diff_logs_errors_exit_code_is_p0() -> None:
    homo = {"m": {"log_messages": {}, "raises": [], "exit_codes": [2]}}
    hetero = {"m": {"log_messages": {}, "raises": [], "exit_codes": [1]}}
    gaps = diff_logs_errors(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" for g in gaps)


def test_diff_docs_missing_page_is_p1() -> None:
    homo = {
        "theory/anti_degeneracy.rst": {
            "headings": ["Anti-Deg"],
            "automodule": [],
            "autoclass": [],
            "autofunction": [],
            "toctree": [],
            "xrefs": [],
        }
    }
    hetero: dict = {}
    gaps = diff_docs(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P1"


def test_diff_docs_broken_autodoc_target_is_p0() -> None:
    homo = {
        "api/nlsq.rst": {
            "headings": [],
            "automodule": ["homodyne.optimization.nlsq"],
            "autoclass": [],
            "autofunction": [],
            "toctree": [],
            "xrefs": [],
        }
    }
    hetero = {
        "api/nlsq.rst": {
            "headings": [],
            "automodule": ["heterodyne.optimization.nlsq.NONEXISTENT"],
            "autoclass": [],
            "autofunction": [],
            "toctree": [],
            "xrefs": [],
        }
    }
    gaps = diff_docs(homodyne=homo, heterodyne=hetero)
    assert any(g["severity"] == "P0" for g in gaps)


def test_diff_classes_missing_method_is_p0() -> None:
    homo = {
        "m.Cls": {"bases": [], "methods": ["foo(self) -> int"], "dataclass_fields": []}
    }
    hetero = {"m.Cls": {"bases": [], "methods": [], "dataclass_fields": []}}
    gaps = diff_classes(homodyne=homo, heterodyne=hetero)
    assert gaps[0]["severity"] == "P0"


# ---------------------------------------------------------------------------
# Integration tests for run_full_diff + render_report (closes review N4).
# ---------------------------------------------------------------------------


_EMPTY_EXTRACTS: dict[str, object] = {
    "signatures.json": {},
    "exports.json": {},
    "classes.json": {},
    "configs.json": {},
    "cli.json": {},
    "logs_errors.json": {},
    "docs.json": {},
    "file_inventory.json": {
        "python_files_non_physics": [],
        "doc_files": [],
    },
}


def _materialise_extracts(directory: Path, contents: dict[str, object]) -> None:
    """Write each ``filename -> payload`` pair as JSON under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        (directory / name).write_text(json.dumps(payload))


def test_run_full_diff_zero_gaps_when_inputs_match(tmp_path: Path) -> None:
    """Identical extracts on both sides → empty gap list, no exceptions."""
    homo_dir = tmp_path / "homo"
    hetero_dir = tmp_path / "hetero"
    _materialise_extracts(homo_dir, _EMPTY_EXTRACTS)
    _materialise_extracts(hetero_dir, _EMPTY_EXTRACTS)

    gaps = run_full_diff(
        homodyne_extracts=homo_dir,
        heterodyne_extracts=hetero_dir,
    )
    assert gaps == []


def test_run_full_diff_raises_on_missing_extract(tmp_path: Path) -> None:
    """Silent-skip regression guard: missing extracts must raise (not return [])."""
    homo_dir = tmp_path / "homo"
    hetero_dir = tmp_path / "hetero"
    # Only populate the heterodyne side; homodyne side is empty → MissingExtractError.
    _materialise_extracts(hetero_dir, _EMPTY_EXTRACTS)
    homo_dir.mkdir()

    with pytest.raises(MissingExtractError) as excinfo:
        run_full_diff(
            homodyne_extracts=homo_dir,
            heterodyne_extracts=hetero_dir,
        )

    msg = str(excinfo.value)
    # The error message must enumerate the missing files so CI logs
    # point at the broken extractor instead of silently passing.
    assert "signatures.json" in msg
    assert "configs.json" in msg


def test_render_report_emits_severity_buckets_and_no_keep_hardcoding() -> None:
    """render_report must surface severity headers and stop hardcoding ``KEEP``."""
    gaps: list[dict[str, object]] = [
        {
            "category": "signatures",
            "severity": "P0",
            "kind": "missing_in_heterodyne",
            "qualname": "m.foo",
            "detail": "homodyne has `foo()`; heterodyne missing",
        },
        {
            "category": "docs",
            "severity": "P2",
            "kind": "heading_drift",
            "qualname": "api/nlsq.rst",
            "detail": "heading `Bounds` present in homodyne, absent in heterodyne",
            "disposition": "WAIVE",
        },
    ]
    report = render_report(gaps, homodyne_sha="abc123", heterodyne_sha="def456")

    # SHA pinning and total appear in the header.
    assert "abc123" in report
    assert "def456" in report
    assert "Total gaps: **2**" in report

    # Severity sections both render with their counts.
    assert "P0 —" in report
    assert "(1 gaps)" in report

    # Disposition column uses the gap's value when supplied, ``?`` otherwise.
    assert "| `WAIVE` |" in report
    assert "| `?` |" in report

    # Regression guard: KEEP must not be hardcoded for every row (N6).
    assert "| `KEEP` |" not in report
