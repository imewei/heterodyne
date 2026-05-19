"""End-to-end CLI smoke tests for `python -m tools.parity_audit`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_pkg(root: Path, name: str, module_text: str) -> Path:
    pkg = root / name
    (pkg / "core").mkdir(parents=True)
    (pkg / "__init__.py").write_text("__all__ = ['foo']\n")
    (pkg / "core" / "__init__.py").write_text("")
    (pkg / "core" / "models.py").write_text(module_text)
    return pkg


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_extract_subcommand_writes_json(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path, "samplepkg", "def foo(x: int) -> int:\n    return x\n")
    out = tmp_path / "extracts"
    out.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.parity_audit",
            "extract",
            "--package",
            str(pkg),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "signatures.json").exists()
    sigs = json.loads((out / "signatures.json").read_text())
    assert any("foo(x: int)" in v for v in sigs.values())


def test_diff_subcommand_writes_report(tmp_path: Path) -> None:
    homo_pkg = _make_pkg(tmp_path, "homo", "def foo() -> int:\n    return 1\n")
    hetero_pkg = _make_pkg(tmp_path, "het", "def foo(x: int) -> int:\n    return x\n")
    homo_out = tmp_path / "homo_extracts"
    hetero_out = tmp_path / "het_extracts"
    homo_out.mkdir()
    hetero_out.mkdir()

    for pkg, out in [(homo_pkg, homo_out), (hetero_pkg, hetero_out)]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.parity_audit",
                "extract",
                "--package",
                str(pkg),
                "--out",
                str(out),
            ],
            check=True,
            cwd=REPO_ROOT,
        )

    report = tmp_path / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.parity_audit",
            "diff",
            "--homodyne",
            str(homo_out),
            "--heterodyne",
            str(hetero_out),
            "--out",
            str(report),
            "--homodyne-sha",
            "deadbeef",
            "--heterodyne-sha",
            "cafef00d",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    body = report.read_text()
    assert "Parity Audit Report" in body
    assert "deadbeef" in body
    assert "P0" in body  # signature drift on foo
