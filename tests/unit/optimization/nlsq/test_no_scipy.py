"""Guard test: ensure no scipy.optimize.least_squares in the NLSQ pipeline.

This test verifies that the NLSQ fitting redesign successfully eliminated
all scipy.optimize.least_squares calls from the optimization path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Files that must NOT contain scipy.optimize.least_squares
NLSQ_FILES = [
    "heterodyne/core/fitting.py",
    "heterodyne/optimization/nlsq/adapter.py",
    "heterodyne/optimization/nlsq/core.py",
    "heterodyne/optimization/nlsq/fallback_chain.py",
    "heterodyne/optimization/nlsq/strategies/residual.py",
    "heterodyne/optimization/nlsq/strategies/jit_strategy.py",
    "heterodyne/optimization/nlsq/strategies/residual_jit.py",
    "heterodyne/optimization/nlsq/strategies/chunked.py",
    "heterodyne/optimization/nlsq/strategies/out_of_core.py",
    "heterodyne/optimization/nlsq/strategies/stratified_ls.py",
    "heterodyne/optimization/nlsq/strategies/hybrid_streaming.py",
]

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class TestNoScipyLeastSquares:
    """Verify that scipy.optimize.least_squares is absent from the NLSQ path."""

    @pytest.mark.parametrize("filepath", NLSQ_FILES)
    def test_no_scipy_least_squares_import(self, filepath: str) -> None:
        """No file in the NLSQ path imports scipy.optimize.least_squares."""
        source = (PROJECT_ROOT / filepath).read_text()
        assert "from scipy.optimize import least_squares" not in source, (
            f"{filepath} still imports scipy.optimize.least_squares"
        )

    @pytest.mark.parametrize("filepath", NLSQ_FILES)
    def test_no_scipy_least_squares_call(self, filepath: str) -> None:
        """No file calls scipy.optimize.least_squares(...)."""
        source = (PROJECT_ROOT / filepath).read_text()
        # Check for actual calls (not just docstring references)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Check: least_squares(...)
                if isinstance(func, ast.Name) and func.id == "least_squares":
                    pytest.fail(f"{filepath} calls least_squares() directly")
                # Check: scipy.optimize.least_squares(...)
                if isinstance(func, ast.Attribute) and func.attr == "least_squares":
                    pytest.fail(f"{filepath} calls least_squares as an attribute")

    def test_no_scipy_nlsq_adapter_class(self) -> None:
        """ScipyNLSQAdapter class must not exist in adapter.py."""
        source = (PROJECT_ROOT / "heterodyne/optimization/nlsq/adapter.py").read_text()
        assert "class ScipyNLSQAdapter" not in source

    def test_no_scipy_optimize_import_in_adapter(self) -> None:
        """adapter.py must have zero scipy.optimize imports."""
        source = (PROJECT_ROOT / "heterodyne/optimization/nlsq/adapter.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "scipy.optimize" in node.module:
                    pytest.fail(f"adapter.py imports from {node.module}")
