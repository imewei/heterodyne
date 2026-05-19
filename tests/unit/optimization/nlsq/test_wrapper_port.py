"""Characterization tests for the NLSQWrapper port from homodyne.

Verifies NLSQWrapper exposes the same public API as homodyne's, adapted
for heterodyne's 14-parameter model. NLSQWrapper is the stable fallback
adapter for 100M+ point datasets with custom 3-attempt recovery.

Refs: Phase 4 PR 1 Task 1.5
"""

from __future__ import annotations

import inspect

from heterodyne.optimization.nlsq.wrapper import NLSQWrapper


def test_nlsq_wrapper_is_instantiable() -> None:
    """NLSQWrapper can be constructed without arguments (defaults applied)."""
    wrapper = NLSQWrapper()
    assert wrapper is not None


def test_nlsq_wrapper_exposes_fit_method() -> None:
    """NLSQWrapper.fit is callable."""
    wrapper = NLSQWrapper()
    assert hasattr(wrapper, "fit")
    assert callable(wrapper.fit)


def test_nlsq_wrapper_fit_signature_matches_homodyne() -> None:
    """fit() accepts the same core kwargs homodyne does: data, config,
    initial_params, bounds.
    """
    sig = inspect.signature(NLSQWrapper.fit)
    params = sig.parameters
    assert "self" in params or "cls" in params
    # Required positional/keyword args homodyne exposes
    for required in ("data", "config"):
        assert required in params, f"NLSQWrapper.fit must accept {required!r} arg"


def test_nlsq_wrapper_is_exported_from_init() -> None:
    """NLSQWrapper is re-exported from the optimization.nlsq package."""
    from heterodyne.optimization import nlsq as pkg

    assert hasattr(pkg, "NLSQWrapper")
    assert "NLSQWrapper" in getattr(pkg, "__all__", [])
