"""Tests for the __all__ exports extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_exports import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_init.py"


def test_extract_returns_sorted_all() -> None:
    exports = extract_file(FIXTURE, module_path="fixtures.sample_init")
    assert exports == ["A", "B", "C"]


def test_extract_missing_all_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "noall.py"
    f.write_text("x = 1\n")
    assert extract_file(f, module_path="noall") == []


def test_extract_handles_tuple_literal(tmp_path: Path) -> None:
    f = tmp_path / "tup.py"
    f.write_text('__all__ = ("a", "b")\n')
    assert extract_file(f, module_path="tup") == ["a", "b"]
