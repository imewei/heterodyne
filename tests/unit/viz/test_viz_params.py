"""Verify visualization modules use heterodyne parameter names."""

from __future__ import annotations

import pathlib
import re

import pytest

VIZ_DIR = pathlib.Path("heterodyne/viz")

# Homodyne-only parameter names that should never appear in heterodyne viz.
# These are unambiguous names that don't collide with matplotlib kwargs.
HOMODYNE_ONLY_PARAMS = {
    "gamma_dot_t0",
    "gamma_dot_t_offset",
    "gamma_dot_0",
    "gamma_dot_offset",
    "gamma_dot",
    "shear_rate",
}

# Ambiguous names that are also common Python/matplotlib identifiers.
# Check these only as quoted strings (parameter name references).
AMBIGUOUS_PARAMS = {"D0", "alpha", "D_offset"}


class TestVizParameterNames:
    """Ensure no homodyne parameter names leak into viz code."""

    @pytest.mark.parametrize("py_file", sorted(VIZ_DIR.glob("*.py")))
    def test_no_homodyne_params(self, py_file: pathlib.Path) -> None:
        """Check that viz file does not reference homodyne-only params."""
        source = py_file.read_text()
        for param in HOMODYNE_ONLY_PARAMS:
            assert param not in source, (
                f"Homodyne param '{param}' found in {py_file.name}"
            )

    @pytest.mark.parametrize("py_file", sorted(VIZ_DIR.glob("*.py")))
    def test_no_ambiguous_homodyne_refs(self, py_file: pathlib.Path) -> None:
        """Check that ambiguous homodyne names don't appear as quoted param refs."""
        source = py_file.read_text()
        for param in AMBIGUOUS_PARAMS:
            # Match "D0" or 'D0' as a bare param name string literal, but
            # exclude matplotlib/dict key usage like "alpha": 0.8
            pattern = rf"""(['"]){re.escape(param)}\1(?!\s*:)"""
            match = re.search(pattern, source)
            assert match is None, (
                f"Homodyne param reference '{param}' found in {py_file.name}"
            )
