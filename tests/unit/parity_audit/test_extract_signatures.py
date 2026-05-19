"""Tests for the signature extractor."""

from __future__ import annotations

import ast
from pathlib import Path

from tools.parity_audit.extract_signatures import canonical_signature, extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_signatures.py"


def test_extract_skips_private_functions() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    names = set(sigs.keys())
    assert "fixtures.sample_signatures.public_fn" in names
    assert "fixtures.sample_signatures._private_fn" not in names


def test_extract_handles_async() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    assert "fixtures.sample_signatures.async_public" in sigs


def test_extract_public_method_includes_self() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.Cls.public_method"]
    assert "self" in sig
    assert "x: int" in sig
    assert "-> int" in sig


def test_extract_skips_private_methods() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    assert "fixtures.sample_signatures.Cls._private_method" not in sigs


def test_canonical_signature_has_stable_form() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.public_fn"]
    assert sig == "public_fn(a: int, b: str = 'x') -> bool"


def test_canonical_signature_round_trips_kwonly() -> None:
    sigs = extract_file(FIXTURE, module_path="fixtures.sample_signatures")
    sig = sigs["fixtures.sample_signatures.fn_with_complex_types"]
    assert "*" in sig  # kwonly marker
    assert "callback:" in sig


def test_canonical_signature_helper_directly() -> None:
    tree = ast.parse("def foo(x: int = 1) -> str: ...")
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert canonical_signature(fn) == "foo(x: int = 1) -> str"
