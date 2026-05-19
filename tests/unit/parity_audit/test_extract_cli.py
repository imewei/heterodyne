"""Tests for the CLI extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_cli import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cli.py"


def test_flags_captured_with_defaults_and_choices() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    flags = {f["flag"]: f for f in result["flags"]}
    assert "--config" in flags
    assert flags["--config"]["dest"] == "'config_path'"
    assert flags["--config"]["default"] == "'default.yaml'"

    assert "--method" in flags
    assert flags["--method"]["choices"] == ["nlsq", "cmc"]


def test_short_aliases_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    flags = {f["flag"]: f for f in result["flags"]}
    assert "--verbose" in flags
    assert "-v" in flags["--verbose"]["aliases"]


def test_subparsers_captured_with_aliases() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_cli")
    subs = {s["name"]: s for s in result["subparsers"]}
    assert "fit" in subs
    assert subs["fit"]["aliases"] == ["f"]
    assert "plot" in subs
