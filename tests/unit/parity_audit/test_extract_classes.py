"""Tests for the class-shape extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_classes import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_classes.py"


def test_plain_class_captured() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    plain = classes["fixtures.sample_classes.PlainClass"]
    assert plain["bases"] == []
    assert "public_method(self, x: int) -> int" in plain["methods"]
    assert all("_private" not in m for m in plain["methods"])


def test_subclass_records_bases() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    sub = classes["fixtures.sample_classes.Subclass"]
    assert sub["bases"] == ["PlainClass"]


def test_dataclass_fields_captured() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    dc = classes["fixtures.sample_classes.DataCls"]
    assert dc["dataclass_fields"] == ["name", "value"]


def test_private_class_skipped() -> None:
    classes = extract_file(FIXTURE, module_path="fixtures.sample_classes")
    assert "fixtures.sample_classes._PrivateClass" not in classes
