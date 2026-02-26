"""Tests for heterodyne/optimization/cmc/scaling.py.

Covers:
- smooth_bound and smooth_bound_inverse
- Round-trip consistency (bound → inverse → bound)
- ParameterScaling z-space transforms
- Gradient nonzero at bounds (differentiability)
- compute_scaling_factors from parameter space
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.optimization.cmc.scaling import (
    ParameterScaling,
    compute_scaling_factors,
    smooth_bound,
    smooth_bound_inverse,
)

# ============================================================================
# smooth_bound
# ============================================================================


class TestSmoothBound:
    """Tests for tanh-based smooth bounding."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_center_maps_to_center(self) -> None:
        """Value at midpoint stays at midpoint."""
        low, high = 0.0, 10.0
        mid = (low + high) / 2.0
        result = float(smooth_bound(jnp.float64(mid), low, high))
        assert result == pytest.approx(mid, abs=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_output_within_bounds(self) -> None:
        """Output is always within [low, high] (tanh saturates at extremes)."""
        low, high = -1.0, 1.0
        for raw in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            result = float(smooth_bound(jnp.float64(raw), low, high))
            assert low <= result <= high, f"raw={raw} → {result} not in [{low}, {high}]"
        # Non-extreme values are strictly interior
        for raw in [-1.0, 0.0, 1.0]:
            result = float(smooth_bound(jnp.float64(raw), low, high))
            assert low < result < high

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_monotonic(self) -> None:
        """smooth_bound is monotonically increasing."""
        low, high = 0.0, 10.0
        raws = jnp.linspace(-20.0, 20.0, 100)
        results = jax.vmap(lambda r: smooth_bound(r, low, high))(raws)
        diffs = jnp.diff(results)
        assert jnp.all(diffs > 0), "smooth_bound should be monotonically increasing"

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_gradient_nonzero_at_bounds(self) -> None:
        """Gradient is nonzero everywhere (unlike jnp.clip)."""
        low, high = 0.0, 10.0

        def f(x):
            return smooth_bound(x, low, high)

        # Test at various points including near bounds
        for raw in [0.0, 0.01, 5.0, 9.99, 10.0, -5.0, 15.0]:
            grad = float(jax.grad(f)(jnp.float64(raw)))
            assert grad > 0, f"Gradient at raw={raw} should be positive, got {grad}"

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_gradient_nonzero_far_from_center(self) -> None:
        """Even far from center, gradient is nonzero (just small)."""
        low, high = 0.0, 10.0

        def f(x):
            return smooth_bound(x, low, high)

        grad = float(jax.grad(f)(jnp.float64(100.0)))
        assert grad > 0

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_degenerate_bounds_returns_midpoint_not_nan(self) -> None:
        """When low == high, smooth_bound returns the bound value, not NaN."""
        result = float(smooth_bound(jnp.float64(5.0), 5.0, 5.0))
        assert np.isfinite(result), f"smooth_bound returned {result} for degenerate bounds"
        assert result == pytest.approx(5.0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_degenerate_bounds_any_input(self) -> None:
        """Degenerate bounds always return the bound regardless of input."""
        for raw_val in [-100.0, 0.0, 5.0, 100.0]:
            result = float(smooth_bound(jnp.float64(raw_val), 3.0, 3.0))
            assert np.isfinite(result)
            assert result == pytest.approx(3.0)


# ============================================================================
# smooth_bound_inverse
# ============================================================================


class TestSmoothBoundInverse:
    """Tests for arctanh-based inverse."""

    @pytest.mark.unit
    def test_inverse_at_center(self) -> None:
        """Inverse of midpoint is midpoint."""
        low, high = 0.0, 10.0
        mid = (low + high) / 2.0
        result = smooth_bound_inverse(mid, low, high)
        assert result == pytest.approx(mid, abs=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_round_trip_bound_then_inverse(self) -> None:
        """smooth_bound → smooth_bound_inverse recovers input (within clamp region)."""
        low, high = 0.0, 10.0
        for raw in [1.0, 3.0, 5.0, 7.0, 9.0]:
            bounded = float(smooth_bound(jnp.float64(raw), low, high))
            recovered = smooth_bound_inverse(bounded, low, high)
            assert recovered == pytest.approx(raw, abs=1e-6)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_round_trip_inverse_then_bound(self) -> None:
        """smooth_bound_inverse → smooth_bound recovers input."""
        low, high = -5.0, 5.0
        for value in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            raw = smooth_bound_inverse(value, low, high)
            bounded = float(smooth_bound(jnp.float64(raw), low, high))
            assert bounded == pytest.approx(value, abs=1e-6)

    @pytest.mark.unit
    def test_near_bounds_clamped(self) -> None:
        """Values very near bounds are clamped to avoid arctanh(±1)."""
        low, high = 0.0, 10.0
        # Should not raise, even at the boundary
        result_low = smooth_bound_inverse(0.001, low, high)
        result_high = smooth_bound_inverse(9.999, low, high)
        assert np.isfinite(result_low)
        assert np.isfinite(result_high)


# ============================================================================
# ParameterScaling
# ============================================================================


class TestParameterScaling:
    """Tests for ParameterScaling dataclass."""

    @pytest.mark.unit
    def test_to_normalized(self) -> None:
        """z = (value - center) / scale."""
        sc = ParameterScaling(name="p", center=5.0, scale=2.0, low=0.0, high=10.0)
        assert sc.to_normalized(5.0) == pytest.approx(0.0)
        assert sc.to_normalized(7.0) == pytest.approx(1.0)
        assert sc.to_normalized(3.0) == pytest.approx(-1.0)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_to_original(self) -> None:
        """to_original maps z=0 to center (bounded)."""
        sc = ParameterScaling(name="p", center=5.0, scale=2.0, low=0.0, high=10.0)
        result = float(sc.to_original(jnp.float64(0.0)))
        assert result == pytest.approx(5.0, abs=1e-6)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_to_original_respects_bounds(self) -> None:
        """to_original stays within bounds even for extreme z."""
        sc = ParameterScaling(name="p", center=5.0, scale=2.0, low=0.0, high=10.0)
        result_low = float(sc.to_original(jnp.float64(-100.0)))
        result_high = float(sc.to_original(jnp.float64(100.0)))
        assert 0.0 <= result_low <= 10.0
        assert 0.0 <= result_high <= 10.0

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_z_space_round_trip(self) -> None:
        """to_normalized → to_original recovers value (approximately, due to smooth bound)."""
        sc = ParameterScaling(name="p", center=5.0, scale=2.0, low=0.0, high=10.0)
        # At center, round-trip is exact
        z = sc.to_normalized(5.0)
        recovered = float(sc.to_original(jnp.float64(z)))
        assert recovered == pytest.approx(5.0, abs=1e-6)


# ============================================================================
# compute_scaling_factors
# ============================================================================


class TestComputeScalingFactors:
    """Tests for compute_scaling_factors()."""

    @pytest.mark.unit
    def test_with_nlsq_values(self) -> None:
        """Uses NLSQ values as centers."""
        space = MagicMock()
        space.varying_names = ["D0_ref", "alpha_ref"]
        space.bounds = {"D0_ref": (0.0, 10.0), "alpha_ref": (0.0, 2.0)}

        nlsq_vals = {"D0_ref": 1.5, "alpha_ref": 0.8}
        nlsq_unc = {"D0_ref": 0.15, "alpha_ref": 0.08}

        scalings = compute_scaling_factors(
            space, nlsq_values=nlsq_vals, nlsq_uncertainties=nlsq_unc,
            width_factor=2.0,
        )

        assert scalings["D0_ref"].center == 1.5
        assert scalings["D0_ref"].scale == pytest.approx(0.15 * 2.0)
        assert scalings["alpha_ref"].center == 0.8
        assert scalings["alpha_ref"].scale == pytest.approx(0.08 * 2.0)

    @pytest.mark.unit
    def test_fallback_without_nlsq(self) -> None:
        """Falls back to bounds midpoint and range/6."""
        space = MagicMock()
        space.varying_names = ["D0_ref"]
        space.bounds = {"D0_ref": (0.0, 6.0)}

        scalings = compute_scaling_factors(space)

        assert scalings["D0_ref"].center == pytest.approx(3.0)
        assert scalings["D0_ref"].scale == pytest.approx(1.0)  # (6-0)/6 = 1.0

    @pytest.mark.unit
    def test_zero_uncertainty_gets_bounds_fallback(self) -> None:
        """Zero uncertainty falls back to bounds-based scale."""
        space = MagicMock()
        space.varying_names = ["p"]
        space.bounds = {"p": (0.0, 12.0)}

        scalings = compute_scaling_factors(
            space, nlsq_values={"p": 6.0}, nlsq_uncertainties={"p": 0.0},
        )

        assert scalings["p"].center == 6.0
        assert scalings["p"].scale == pytest.approx(2.0)  # (12-0)/6

    @pytest.mark.unit
    def test_minimum_scale_enforced(self) -> None:
        """Scale never goes below 1e-10."""
        space = MagicMock()
        space.varying_names = ["p"]
        space.bounds = {"p": (5.0, 5.0)}  # degenerate bounds

        scalings = compute_scaling_factors(space)
        assert scalings["p"].scale >= 1e-10

    @pytest.mark.unit
    def test_custom_width_factor(self) -> None:
        """width_factor multiplies NLSQ uncertainty."""
        space = MagicMock()
        space.varying_names = ["p"]
        space.bounds = {"p": (0.0, 10.0)}

        scalings = compute_scaling_factors(
            space, nlsq_values={"p": 5.0}, nlsq_uncertainties={"p": 0.5},
            width_factor=3.0,
        )

        assert scalings["p"].scale == pytest.approx(0.5 * 3.0)
