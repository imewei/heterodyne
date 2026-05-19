"""Tests for the diff/categorizer module."""

from __future__ import annotations

from tools.parity_audit.diff_extracts import (
    diff_classes,
    diff_cli,
    diff_configs,
    diff_docs,
    diff_exports,
    diff_file_inventory,
    diff_logs_errors,
    diff_signatures,
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
