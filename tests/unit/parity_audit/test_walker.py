"""Tests for the shared walker that discovers files for extraction."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.walker import (
    PHYSICS_EXEMPT_FILES,
    discover_doc_files,
    discover_python_files,
    is_physics_exempt,
)


def test_physics_exempt_files_are_module_paths() -> None:
    assert "core.theory" in PHYSICS_EXEMPT_FILES
    assert "core.physics" in PHYSICS_EXEMPT_FILES
    assert "core.physics_cmc" in PHYSICS_EXEMPT_FILES
    assert "core.physics_nlsq" in PHYSICS_EXEMPT_FILES
    assert "core.physics_utils" in PHYSICS_EXEMPT_FILES
    assert "core.jax_backend" in PHYSICS_EXEMPT_FILES


def test_is_physics_exempt_matches_module_path(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    physics_file = pkg / "core" / "theory.py"
    physics_file.write_text("")
    nonphysics_file = pkg / "core" / "models.py"
    nonphysics_file.write_text("")

    assert is_physics_exempt(physics_file, package_root=pkg) is True
    assert is_physics_exempt(nonphysics_file, package_root=pkg) is False


def test_discover_python_files_skips_tests_and_pycache(tmp_path: Path) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "tests").mkdir(parents=True)
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "core" / "models.py").write_text("")
    (pkg / "tests" / "test_models.py").write_text("")
    (pkg / "__pycache__" / "models.cpython-313.pyc").write_text("")

    found = set(discover_python_files(pkg))
    assert pkg / "core" / "models.py" in found
    assert pkg / "tests" / "test_models.py" not in found
    assert pkg / "__pycache__" / "models.cpython-313.pyc" not in found


def test_discover_python_files_excludes_physics_exempt_when_requested(
    tmp_path: Path,
) -> None:
    pkg = tmp_path / "heterodyne"
    (pkg / "core").mkdir(parents=True)
    (pkg / "core" / "theory.py").write_text("")
    (pkg / "core" / "models.py").write_text("")

    found = set(discover_python_files(pkg, exclude_physics_exempt=True))
    assert pkg / "core" / "models.py" in found
    assert pkg / "core" / "theory.py" not in found


def test_discover_doc_files_finds_rst_and_md(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "source"
    (docs / "user_guide").mkdir(parents=True)
    (docs / "user_guide" / "intro.rst").write_text("Intro\n=====\n")
    (docs / "user_guide" / "advanced.md").write_text("# Advanced\n")
    (docs / "_build").mkdir()
    (docs / "_build" / "ignored.rst").write_text("")

    found = set(discover_doc_files(docs))
    assert docs / "user_guide" / "intro.rst" in found
    assert docs / "user_guide" / "advanced.md" in found
    assert docs / "_build" / "ignored.rst" not in found
