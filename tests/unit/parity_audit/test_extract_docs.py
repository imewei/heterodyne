"""Tests for the docs extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_docs import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs.rst"


def test_headings_captured() -> None:
    result = extract_file(FIXTURE)
    assert "NLSQ Optimization" in result["headings"]
    assert "Overview" in result["headings"]
    assert "API" in result["headings"]


def test_autodoc_directives_captured() -> None:
    result = extract_file(FIXTURE)
    assert "heterodyne.optimization.nlsq" in result["automodule"]
    assert "heterodyne.NLSQConfig" in result["autoclass"]
    assert "heterodyne.fit_nlsq_jax" in result["autofunction"]


def test_toctree_entries_captured() -> None:
    result = extract_file(FIXTURE)
    assert "intro" in result["toctree"]
    assert "advanced" in result["toctree"]


def test_cross_references_captured() -> None:
    result = extract_file(FIXTURE)
    xrefs = result["xrefs"]
    assert {"role": "func", "target": "heterodyne.fit_nlsq_jax"} in xrefs
    assert {"role": "class", "target": "heterodyne.NLSQConfig"} in xrefs
    assert {"role": "ref", "target": "anti-degeneracy-overview"} in xrefs
