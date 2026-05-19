"""Tests for the file-inventory extractor."""

from __future__ import annotations

import json
from pathlib import Path

from tools.parity_audit.extract_file_inventory import extract, write_json


def test_extract_returns_relative_module_paths(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    (pkg / "core" / "theory.py").write_text("")  # physics-exempt
    (pkg / "cli").mkdir()
    (pkg / "cli" / "main.py").write_text("")
    (pkg / "tests").mkdir()
    (pkg / "tests" / "test_x.py").write_text("")  # skipped

    result = extract(pkg)

    assert sorted(result["python_files"]) == ["cli.main", "core.models", "core.theory"]
    assert sorted(result["python_files_non_physics"]) == ["cli.main", "core.models"]
    assert result["physics_exempt_files"] == ["core.theory"]


def test_extract_includes_docs_when_docs_root_present(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    docs = pkg.parent / "docs" / "source"
    docs.mkdir(parents=True)
    (docs / "intro.rst").write_text("Intro\n=====\n")
    (docs / "advanced.md").write_text("# Advanced\n")

    result = extract(pkg, docs_root=docs)

    assert sorted(result["doc_files"]) == ["advanced.md", "intro.rst"]


def test_write_json_round_trips(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")

    out_path = tmp_path / "file_inventory.json"
    write_json(pkg, out_path)
    data = json.loads(out_path.read_text())
    assert "core.models" in data["python_files"]
