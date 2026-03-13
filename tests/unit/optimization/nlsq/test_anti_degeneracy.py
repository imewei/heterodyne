"""Unit tests for AntiDegeneracyController and multi-start LHS utilities.

Tests the controller creation, correlation/bound/plateau checks, and
the standalone `check_zero_volume_bounds` / `generate_lhs_starts` helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
    DegeneracyCheck,
)
from heterodyne.optimization.nlsq.multistart import (
    check_zero_volume_bounds,
    generate_lhs_starts,
)

# ---------------------------------------------------------------------------
# AntiDegeneracyController
# ---------------------------------------------------------------------------


class TestAntiDegeneracyController:
    """Tests for AntiDegeneracyController creation and check methods."""

    def test_controller_creation_default(self) -> None:
        """Default construction uses sensible threshold values."""
        ctrl = AntiDegeneracyController()
        assert hasattr(ctrl, "check")
        assert callable(ctrl.check)

    def test_controller_creation_custom(self) -> None:
        """Custom thresholds are accepted without error."""
        ctrl = AntiDegeneracyController(
            correlation_threshold=0.95,
            bound_tolerance=1e-3,
            plateau_min_iterations=5,
            plateau_cost_rtol=1e-8,
        )
        assert ctrl is not None

    def test_controller_invalid_correlation_threshold(self) -> None:
        """correlation_threshold <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="correlation_threshold"):
            AntiDegeneracyController(correlation_threshold=0.0)

    def test_controller_invalid_bound_tolerance(self) -> None:
        """Negative bound_tolerance raises ValueError."""
        with pytest.raises(ValueError, match="bound_tolerance"):
            AntiDegeneracyController(bound_tolerance=-1e-4)

    def test_degeneracy_check_dataclass(self) -> None:
        """DegeneracyCheck fields are accessible and have correct types."""
        chk = DegeneracyCheck(
            is_degenerate=True,
            affected_params=["D0_ref", "D0_sample"],
            message="test message",
            suggested_action="fix it",
        )
        assert chk.is_degenerate is True
        assert isinstance(chk.affected_params, list)
        assert "D0_ref" in chk.affected_params
        assert isinstance(chk.message, str)
        assert isinstance(chk.suggested_action, str)


# ---------------------------------------------------------------------------
# check_zero_volume_bounds
# ---------------------------------------------------------------------------


class TestCheckZeroVolumeBounds:
    """Tests for the standalone check_zero_volume_bounds utility."""

    def test_check_zero_volume_bounds_detects_fixed(self) -> None:
        """Dimensions where lower == upper are returned as fixed indices."""
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([0.0, 5.0, 2.0])  # dims 0 and 2 are fixed
        fixed = check_zero_volume_bounds(lower, upper)
        assert sorted(fixed) == [0, 2]

    def test_check_zero_volume_bounds_none(self) -> None:
        """All different → empty list returned."""
        lower = np.array([0.0, 1.0, -5.0])
        upper = np.array([1.0, 2.0, 5.0])
        fixed = check_zero_volume_bounds(lower, upper)
        assert fixed == []

    def test_check_zero_volume_bounds_all_fixed(self) -> None:
        """Every dimension fixed → all indices returned."""
        lower = np.array([3.0, 7.0])
        upper = np.array([3.0, 7.0])
        fixed = check_zero_volume_bounds(lower, upper)
        assert sorted(fixed) == [0, 1]

    def test_check_zero_volume_bounds_shape_mismatch(self) -> None:
        """Mismatched array lengths raise ValueError."""
        lower = np.array([0.0, 1.0])
        upper = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            check_zero_volume_bounds(lower, upper)

    def test_check_zero_volume_bounds_single_element(self) -> None:
        """Single element array with lower == upper → [0]."""
        fixed = check_zero_volume_bounds(np.array([5.0]), np.array([5.0]))
        assert fixed == [0]


# ---------------------------------------------------------------------------
# generate_lhs_starts
# ---------------------------------------------------------------------------


class TestGenerateLhsStarts:
    """Tests for the standalone generate_lhs_starts utility."""

    def test_generate_lhs_starts_shape(self) -> None:
        """Output shape is exactly (n_starts, n_params)."""
        lower = np.array([0.0, -1.0, 10.0])
        upper = np.array([1.0, 1.0, 100.0])
        n_starts = 8
        starts = generate_lhs_starts(n_starts, lower, upper, seed=0)
        assert starts.shape == (n_starts, len(lower))

    def test_generate_lhs_starts_within_bounds(self) -> None:
        """All generated points satisfy lower <= x <= upper."""
        lower = np.array([0.0, -5.0, 1.0])
        upper = np.array([10.0, 5.0, 100.0])
        starts = generate_lhs_starts(20, lower, upper, seed=42)
        assert np.all(starts >= lower[np.newaxis, :])
        assert np.all(starts <= upper[np.newaxis, :])

    def test_generate_lhs_starts_fixed_dims(self) -> None:
        """Zero-volume dimensions (lower == upper) are constant across all starts."""
        lower = np.array([0.0, 3.0, 1.0])
        upper = np.array([1.0, 3.0, 5.0])  # dim 1 is fixed at 3.0
        starts = generate_lhs_starts(10, lower, upper, seed=7)
        # All rows for the fixed dimension must equal the fixed value
        assert np.all(starts[:, 1] == pytest.approx(3.0))

    def test_generate_lhs_starts_reproducible(self) -> None:
        """Same seed produces identical starting points."""
        lower = np.zeros(4)
        upper = np.ones(4)
        a = generate_lhs_starts(5, lower, upper, seed=99)
        b = generate_lhs_starts(5, lower, upper, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_generate_lhs_starts_different_seeds(self) -> None:
        """Different seeds produce different starting points."""
        lower = np.zeros(3)
        upper = np.ones(3)
        a = generate_lhs_starts(6, lower, upper, seed=1)
        b = generate_lhs_starts(6, lower, upper, seed=2)
        assert not np.array_equal(a, b)

    def test_generate_lhs_starts_invalid_n_starts(self) -> None:
        """n_starts < 1 raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([1.0])
        with pytest.raises(ValueError, match="n_starts"):
            generate_lhs_starts(0, lower, upper)

    def test_generate_lhs_starts_single_start(self) -> None:
        """n_starts=1 is valid and returns shape (1, n_params)."""
        lower = np.array([0.0, -1.0])
        upper = np.array([1.0, 1.0])
        starts = generate_lhs_starts(1, lower, upper, seed=0)
        assert starts.shape == (1, 2)
