"""Tests for the logs/errors extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_logs_errors import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_logs_errors.py"


def test_log_format_strings_captured_by_level() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    assert "Starting work for x=%d" in result["log_messages"]["info"]
    assert "Negative x: %d" in result["log_messages"]["warning"]
    assert "x out of range" in result["log_messages"]["error"]


def test_exception_stems_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    raises = result["raises"]
    assert any(
        r["exception"] == "ValueError" and "non-negative" in r["message"]
        for r in raises
    )


def test_exit_codes_captured() -> None:
    result = extract_file(FIXTURE, module_path="fixtures.sample_logs_errors")
    assert 2 in result["exit_codes"]
