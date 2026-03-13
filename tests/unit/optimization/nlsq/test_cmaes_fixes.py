"""Tests for CMA-ES homodyne parity fixes.

Covers:
- Fix #0: CMAES_AVAILABLE flag and fit_with_cmaes convenience function
- Fix #2: Adaptive population sizing via compute_adaptive_cmaes_params
- Fix #3: Adaptive max_generations scaling
- Fix #4: Diagonal filtering configuration
- Fix #5: 3-level quality flag classification
- Fix #6: Anti-degeneracy objective wrapping
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.cmaes_wrapper import (
    CMAES_AVAILABLE,
    CMAESConfig,
    build_anti_degeneracy_objective,
    compute_adaptive_cmaes_params,
)
from heterodyne.optimization.nlsq.validation.fit_quality import classify_fit_quality

# ---------------------------------------------------------------------------
# Fix #0: CMAES_AVAILABLE flag
# ---------------------------------------------------------------------------


class TestCMAESAvailable:
    """Tests for the CMAES_AVAILABLE module-level flag."""

    def test_cmaes_available_is_bool(self) -> None:
        assert isinstance(CMAES_AVAILABLE, bool)

    def test_cmaes_available_reflects_cma_import(self) -> None:
        try:
            import cma  # noqa: F401

            assert CMAES_AVAILABLE is True
        except ImportError:
            assert CMAES_AVAILABLE is False


# ---------------------------------------------------------------------------
# Fix #2 + #3: Adaptive population sizing and max_generations
# ---------------------------------------------------------------------------


class TestComputeAdaptiveCMAESParams:
    """Tests for compute_adaptive_cmaes_params."""

    def test_uniform_scale_returns_minimum(self) -> None:
        """When all parameter ranges are equal, scale ratio ~ 1 → min values."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        popsize, max_gen = compute_adaptive_cmaes_params((lower, upper))
        assert popsize == 50
        assert max_gen == 200

    def test_high_scale_ratio_returns_maximum(self) -> None:
        """When scale ratio is very large → max values."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1e6, 1.0])  # ratio = 1e6
        popsize, max_gen = compute_adaptive_cmaes_params((lower, upper))
        assert popsize == 200
        assert max_gen == 500

    def test_moderate_scale_ratio_interpolates(self) -> None:
        """Intermediate scale ratio → intermediate values."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1000.0, 1.0])  # ratio = 1000, log10=3, t=0.5
        popsize, max_gen = compute_adaptive_cmaes_params((lower, upper))
        assert 50 < popsize < 200
        assert 200 < max_gen < 500

    def test_single_active_dimension_returns_minimum(self) -> None:
        """With only one active dimension (others fixed), returns minimum."""
        lower = np.array([0.0, 5.0])
        upper = np.array([10.0, 5.0])  # dim 1 is fixed
        popsize, max_gen = compute_adaptive_cmaes_params((lower, upper))
        assert popsize == 50
        assert max_gen == 200

    def test_fixed_dimensions_excluded(self) -> None:
        """Fixed dimensions (lower == upper) are excluded from ratio calculation."""
        lower = np.array([0.0, 5.0, 0.0])
        upper = np.array([1e6, 5.0, 1.0])  # dim 1 fixed; ratio from dim 0 & 2
        popsize, max_gen = compute_adaptive_cmaes_params((lower, upper))
        assert popsize == 200
        assert max_gen == 500


# ---------------------------------------------------------------------------
# Fix #4: Diagonal filtering configuration
# ---------------------------------------------------------------------------


class TestDiagonalFiltering:
    """Tests for CMAESConfig.diagonal_filtering."""

    def test_default_is_none(self) -> None:
        cfg = CMAESConfig()
        assert cfg.diagonal_filtering == "none"

    def test_remove_accepted(self) -> None:
        cfg = CMAESConfig(diagonal_filtering="remove")
        assert cfg.diagonal_filtering == "remove"

    def test_none_accepted(self) -> None:
        cfg = CMAESConfig(diagonal_filtering="none")
        assert cfg.diagonal_filtering == "none"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="diagonal_filtering"):
            CMAESConfig(diagonal_filtering="invalid")


# ---------------------------------------------------------------------------
# Fix #5: 3-level quality flags
# ---------------------------------------------------------------------------


class TestClassifyFitQuality:
    """Tests for classify_fit_quality."""

    def test_good_quality(self) -> None:
        assert classify_fit_quality(0.5) == "good"
        assert classify_fit_quality(1.0) == "good"
        assert classify_fit_quality(1.49) == "good"

    def test_marginal_quality(self) -> None:
        assert classify_fit_quality(1.5) == "marginal"
        assert classify_fit_quality(2.0) == "marginal"
        assert classify_fit_quality(2.99) == "marginal"

    def test_poor_quality(self) -> None:
        assert classify_fit_quality(3.0) == "poor"
        assert classify_fit_quality(100.0) == "poor"

    def test_none_returns_poor(self) -> None:
        assert classify_fit_quality(None) == "poor"

    def test_zero_chi2_is_good(self) -> None:
        assert classify_fit_quality(0.0) == "good"


# ---------------------------------------------------------------------------
# Fix #6: Anti-degeneracy objective wrapping
# ---------------------------------------------------------------------------


class TestBuildAntiDegeneracyObjective:
    """Tests for build_anti_degeneracy_objective."""

    def test_no_degenerate_pairs_returns_base(self) -> None:
        """When no known pairs are present, returns the base objective unchanged."""
        base = lambda x: float(np.sum(x**2))  # noqa: E731
        names = ["unknown_a", "unknown_b"]
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        wrapped = build_anti_degeneracy_objective(base, (lower, upper), names)
        # Should be the exact same function object (no wrapping needed)
        assert wrapped is base

    def test_penalty_increases_cost(self) -> None:
        """When degenerate pairs are present, wrapped cost >= base cost."""
        base = lambda x: float(np.sum(x**2))  # noqa: E731
        names = ["D0_ref", "D0_sample", "v0"]
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1e6, 1e6, 1e3])
        wrapped = build_anti_degeneracy_objective(
            base,
            (lower, upper),
            names,
            penalty_weight=0.01,
        )
        x = np.array([500.0, 500.0, 100.0])
        assert wrapped(x) >= base(x)

    def test_separated_params_lower_penalty(self) -> None:
        """Well-separated degenerate pair parameters have lower penalty."""
        base = lambda x: 0.0  # noqa: E731
        names = ["D0_ref", "D0_sample"]
        lower = np.array([0.0, 0.0])
        upper = np.array([1e6, 1e6])
        wrapped = build_anti_degeneracy_objective(
            base,
            (lower, upper),
            names,
            penalty_weight=1.0,
        )
        # Close together (both at 0.5 normalized)
        x_close = np.array([5e5, 5e5])
        # Far apart (0.1 vs 0.9 normalized)
        x_far = np.array([1e5, 9e5])
        assert wrapped(x_close) > wrapped(x_far)

    def test_penalty_weight_scales(self) -> None:
        """Higher penalty_weight produces higher total cost."""
        base = lambda x: 1.0  # noqa: E731
        names = ["v0", "v_offset"]
        lower = np.array([0.0, -100.0])
        upper = np.array([1e3, 100.0])
        x = np.array([500.0, 0.0])
        low_penalty = build_anti_degeneracy_objective(
            base,
            (lower, upper),
            names,
            penalty_weight=0.001,
        )
        high_penalty = build_anti_degeneracy_objective(
            base,
            (lower, upper),
            names,
            penalty_weight=1.0,
        )
        assert high_penalty(x) > low_penalty(x)
