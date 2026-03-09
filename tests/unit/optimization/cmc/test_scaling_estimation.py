"""Tests for quantile-based scaling estimation and parameter validation.

Covers:
- estimate_contrast_offset_from_quantiles: edge cases and physics correctness
- estimate_per_angle_scaling: multi-angle vectorized estimation
- compute_averaged_scaling: averaged mode
- PerAngleScaling: auto and constant_averaged modes
- validate_parameters: bounds checking and violation reporting
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.core.physics import (
    ValidationResult,
    validate_parameters,
)
from heterodyne.core.scaling_utils import (
    PerAngleScaling,
    ScalingConfig,
    compute_averaged_scaling,
    estimate_contrast_offset_from_quantiles,
    estimate_per_angle_scaling,
)

# ============================================================================
# estimate_contrast_offset_from_quantiles
# ============================================================================


class TestEstimateContrastOffset:
    """Tests for quantile-based contrast/offset estimation."""

    @pytest.mark.unit
    def test_synthetic_decay_curve(self) -> None:
        """Known synthetic data: C2 = offset + contrast * exp(-dt/tau)."""
        rng = np.random.default_rng(42)
        n = 1000
        true_contrast = 0.3
        true_offset = 1.02
        tau = 20.0

        dt = rng.uniform(0.1, 200.0, size=n)
        c2 = true_offset + true_contrast * np.exp(-dt / tau) + rng.normal(0, 0.005, n)

        est_contrast, est_offset = estimate_contrast_offset_from_quantiles(
            c2, dt, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        # Quantile estimation should be within ~20% for synthetic data
        assert est_offset == pytest.approx(true_offset, abs=0.05)
        assert est_contrast == pytest.approx(true_contrast, abs=0.1)

    @pytest.mark.unit
    def test_insufficient_data_returns_midpoints(self) -> None:
        """Less than 100 points → return midpoints of bounds."""
        c2 = np.ones(50)
        dt = np.ones(50)
        c, o = estimate_contrast_offset_from_quantiles(
            c2, dt, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )
        assert c == pytest.approx(0.5)
        assert o == pytest.approx(1.0)

    @pytest.mark.unit
    def test_nan_values_filtered(self) -> None:
        """NaN values are filtered before estimation."""
        rng = np.random.default_rng(42)
        n = 500
        c2 = rng.uniform(0.9, 1.3, size=n)
        dt = rng.uniform(0.1, 100.0, size=n)
        c2[:20] = np.nan  # Inject NaN

        c, o = estimate_contrast_offset_from_quantiles(c2, dt)
        assert np.isfinite(c)
        assert np.isfinite(o)

    @pytest.mark.unit
    def test_all_nan_below_threshold(self) -> None:
        """All NaN data with <100 finite points returns midpoints."""
        c2 = np.full(200, np.nan)
        c2[:50] = 1.0  # Only 50 finite — below 100 threshold
        dt = np.ones(200)
        c, o = estimate_contrast_offset_from_quantiles(c2, dt)
        # Should return midpoint defaults
        assert c == pytest.approx(0.5)
        assert o == pytest.approx(1.0)

    @pytest.mark.unit
    def test_constant_data(self) -> None:
        """Constant C2 → contrast=0, offset=constant."""
        c2 = np.full(200, 1.05)
        dt = np.linspace(0, 100, 200)
        c, o = estimate_contrast_offset_from_quantiles(
            c2, dt, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )
        assert c == pytest.approx(0.0, abs=1e-6)
        assert o == pytest.approx(1.05, abs=0.01)

    @pytest.mark.unit
    def test_bounds_clipping(self) -> None:
        """Estimates are clipped to specified bounds."""
        # Data with huge contrast that exceeds bounds
        dt = np.linspace(0, 100, 500)
        c2 = 2.0 + 5.0 * np.exp(-dt / 10.0)  # contrast ~5, offset ~2

        c, o = estimate_contrast_offset_from_quantiles(
            c2, dt, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )
        assert 0.0 <= c <= 1.0
        assert 0.5 <= o <= 1.5

    @pytest.mark.unit
    def test_custom_quantile_parameters(self) -> None:
        """Custom quantile parameters are respected."""
        rng = np.random.default_rng(42)
        c2 = rng.uniform(0.9, 1.3, size=500)
        dt = rng.uniform(0.1, 100.0, size=500)

        c1, o1 = estimate_contrast_offset_from_quantiles(
            c2, dt, lag_floor_quantile=0.90, lag_ceiling_quantile=0.10,
        )
        c2_, o2 = estimate_contrast_offset_from_quantiles(
            c2, dt, lag_floor_quantile=0.70, lag_ceiling_quantile=0.30,
        )
        # Different quantile settings should produce different estimates
        # (not necessarily, but for random data they typically do)
        # At minimum, both should be finite
        assert np.isfinite(c1) and np.isfinite(o1)
        assert np.isfinite(c2_) and np.isfinite(o2)


# ============================================================================
# estimate_per_angle_scaling
# ============================================================================


class TestEstimatePerAngleScaling:
    """Tests for vectorized per-angle estimation."""

    @pytest.mark.unit
    def test_single_angle(self) -> None:
        """Single angle with sufficient data."""
        rng = np.random.default_rng(42)
        n = 500
        c2 = 1.0 + 0.3 * np.exp(-rng.uniform(0, 100, n) / 20.0)
        t1 = rng.uniform(0, 100, n)
        t2 = rng.uniform(0, 100, n)
        phi_idx = np.zeros(n, dtype=int)

        result = estimate_per_angle_scaling(
            c2, t1, t2, phi_idx, n_phi=1,
            contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        assert "contrast_0" in result
        assert "offset_0" in result
        assert 0.0 <= result["contrast_0"] <= 1.0
        assert 0.5 <= result["offset_0"] <= 1.5

    @pytest.mark.unit
    def test_multiple_angles(self) -> None:
        """Three angles, each with different statistics."""
        rng = np.random.default_rng(42)
        n_per = 200
        n_phi = 3

        c2_parts, t1_parts, t2_parts, phi_parts = [], [], [], []
        for i in range(n_phi):
            offset = 0.95 + 0.05 * i  # Different offset per angle
            t1_i = rng.uniform(0, 100, n_per)
            t2_i = rng.uniform(0, 100, n_per)
            dt_i = np.abs(t1_i - t2_i)
            c2_i = offset + 0.2 * np.exp(-dt_i / 15.0) + rng.normal(0, 0.005, n_per)
            c2_parts.append(c2_i)
            t1_parts.append(t1_i)
            t2_parts.append(t2_i)
            phi_parts.append(np.full(n_per, i, dtype=int))

        c2 = np.concatenate(c2_parts)
        t1 = np.concatenate(t1_parts)
        t2 = np.concatenate(t2_parts)
        phi_idx = np.concatenate(phi_parts)

        result = estimate_per_angle_scaling(
            c2, t1, t2, phi_idx, n_phi=n_phi,
            contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        # All angles should have entries
        for i in range(n_phi):
            assert f"contrast_{i}" in result
            assert f"offset_{i}" in result

        # Offsets should increase with angle index (by construction)
        offsets = [result[f"offset_{i}"] for i in range(n_phi)]
        assert offsets[0] < offsets[2], "Offsets should reflect per-angle differences"

    @pytest.mark.unit
    def test_insufficient_data_all_angles(self) -> None:
        """All angles below 100-point threshold → midpoint defaults."""
        c2 = np.ones(30)
        t1 = np.ones(30)
        t2 = np.ones(30)
        phi_idx = np.array([0] * 10 + [1] * 10 + [2] * 10)

        result = estimate_per_angle_scaling(
            c2, t1, t2, phi_idx, n_phi=3,
            contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        for i in range(3):
            assert result[f"contrast_{i}"] == pytest.approx(0.5)
            assert result[f"offset_{i}"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_mixed_sufficient_insufficient(self) -> None:
        """Some angles have enough data, others don't."""
        rng = np.random.default_rng(42)
        # Angle 0: 200 points (sufficient)
        # Angle 1: 50 points (insufficient)
        c2_0 = rng.uniform(0.9, 1.3, size=200)
        c2_1 = rng.uniform(0.9, 1.3, size=50)
        c2 = np.concatenate([c2_0, c2_1])

        t1 = rng.uniform(0, 100, len(c2))
        t2 = rng.uniform(0, 100, len(c2))
        phi_idx = np.array([0] * 200 + [1] * 50)

        result = estimate_per_angle_scaling(
            c2, t1, t2, phi_idx, n_phi=2,
            contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        # Angle 0 should have data-driven estimate
        assert 0.0 <= result["contrast_0"] <= 1.0
        # Angle 1 should have midpoint default
        assert result["contrast_1"] == pytest.approx(0.5)
        assert result["offset_1"] == pytest.approx(1.0)


# ============================================================================
# compute_averaged_scaling
# ============================================================================


class TestComputeAveragedScaling:
    """Tests for averaged scaling computation."""

    @pytest.mark.unit
    def test_averaged_values(self) -> None:
        """Averaged scaling returns mean of per-angle estimates."""
        rng = np.random.default_rng(42)
        n_per = 200
        n_phi = 3

        c2_parts, t1_parts, t2_parts, phi_parts = [], [], [], []
        for i in range(n_phi):
            t1_i = rng.uniform(0, 100, n_per)
            t2_i = rng.uniform(0, 100, n_per)
            dt_i = np.abs(t1_i - t2_i)
            c2_i = 1.0 + 0.2 * np.exp(-dt_i / 15.0) + rng.normal(0, 0.005, n_per)
            c2_parts.append(c2_i)
            t1_parts.append(t1_i)
            t2_parts.append(t2_i)
            phi_parts.append(np.full(n_per, i, dtype=int))

        c2 = np.concatenate(c2_parts)
        t1 = np.concatenate(t1_parts)
        t2 = np.concatenate(t2_parts)
        phi_idx = np.concatenate(phi_parts)

        c_avg, o_avg, c_per, o_per = compute_averaged_scaling(
            c2, t1, t2, phi_idx, n_phi=n_phi,
            contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
        )

        assert c_avg == pytest.approx(float(np.mean(c_per)))
        assert o_avg == pytest.approx(float(np.mean(o_per)))
        assert len(c_per) == n_phi
        assert len(o_per) == n_phi


# ============================================================================
# PerAngleScaling — new modes
# ============================================================================


class TestPerAngleScalingModes:
    """Tests for auto and constant_averaged scaling modes."""

    @pytest.mark.unit
    def test_auto_mode_all_vary(self) -> None:
        """Auto mode: all contrast and offset parameters vary."""
        config = ScalingConfig(n_angles=4, mode="auto")
        scaling = PerAngleScaling.from_config(config)

        assert np.all(scaling.vary_contrast)
        assert np.all(scaling.vary_offset)
        assert scaling.n_varying_scaling == 8  # 4 contrast + 4 offset

    @pytest.mark.unit
    def test_constant_averaged_mode_first_only(self) -> None:
        """Constant_averaged mode: only first angle varies."""
        config = ScalingConfig(n_angles=4, mode="constant_averaged")
        scaling = PerAngleScaling.from_config(config)

        assert scaling.vary_contrast[0] is np.True_
        assert not np.any(scaling.vary_contrast[1:])
        assert scaling.vary_offset[0] is np.True_
        assert not np.any(scaling.vary_offset[1:])
        assert scaling.n_varying_scaling == 2

    @pytest.mark.unit
    def test_invalid_mode_raises(self) -> None:
        """Unknown mode raises ValueError."""
        config = ScalingConfig(n_angles=2, mode="nonexistent")
        with pytest.raises(ValueError, match="Unknown scaling mode"):
            PerAngleScaling.from_config(config)

    @pytest.mark.unit
    def test_initialize_from_data(self) -> None:
        """initialize_from_data sets values from quantile estimation."""
        rng = np.random.default_rng(42)
        n_per = 200
        n_angles = 2

        config = ScalingConfig(n_angles=n_angles, mode="auto")
        scaling = PerAngleScaling.from_config(config)

        # Build synthetic data
        c2_parts, t1_parts, t2_parts, phi_parts = [], [], [], []
        for i in range(n_angles):
            offset = 0.95 + 0.1 * i
            t1_i = rng.uniform(0, 100, n_per)
            t2_i = rng.uniform(0, 100, n_per)
            dt_i = np.abs(t1_i - t2_i)
            c2_i = offset + 0.2 * np.exp(-dt_i / 15.0)
            c2_parts.append(c2_i)
            t1_parts.append(t1_i)
            t2_parts.append(t2_i)
            phi_parts.append(np.full(n_per, i, dtype=int))

        scaling.initialize_from_data(
            c2_data=np.concatenate(c2_parts),
            t1=np.concatenate(t1_parts),
            t2=np.concatenate(t2_parts),
            phi_indices=np.concatenate(phi_parts),
        )

        # Values should have been updated from defaults
        assert not np.allclose(scaling.contrast, 0.5)
        assert not np.allclose(scaling.offset, 1.0)

    @pytest.mark.unit
    def test_initialize_from_data_constant_averaged(self) -> None:
        """constant_averaged mode collapses to single averaged value."""
        rng = np.random.default_rng(42)
        n_per = 200
        n_angles = 3

        config = ScalingConfig(n_angles=n_angles, mode="constant_averaged")
        scaling = PerAngleScaling.from_config(config)

        c2_parts, t1_parts, t2_parts, phi_parts = [], [], [], []
        for i in range(n_angles):
            t1_i = rng.uniform(0, 100, n_per)
            t2_i = rng.uniform(0, 100, n_per)
            dt_i = np.abs(t1_i - t2_i)
            c2_i = 1.0 + 0.25 * np.exp(-dt_i / 15.0)
            c2_parts.append(c2_i)
            t1_parts.append(t1_i)
            t2_parts.append(t2_i)
            phi_parts.append(np.full(n_per, i, dtype=int))

        scaling.initialize_from_data(
            c2_data=np.concatenate(c2_parts),
            t1=np.concatenate(t1_parts),
            t2=np.concatenate(t2_parts),
            phi_indices=np.concatenate(phi_parts),
        )

        # All angles should have the same value (averaged)
        assert np.allclose(scaling.contrast, scaling.contrast[0])
        assert np.allclose(scaling.offset, scaling.offset[0])


# ============================================================================
# validate_parameters
# ============================================================================


class TestValidateParameters:
    """Tests for physics parameter validation."""

    @pytest.mark.unit
    def test_valid_parameters(self) -> None:
        """All parameters within bounds."""
        result = validate_parameters({
            "D0_ref": 1e4,
            "alpha_ref": 0.5,
            "v0": 100.0,
        })
        assert result.valid is True
        assert result.parameters_checked == 3
        assert len(result.violations) == 0
        assert "OK" in str(result)

    @pytest.mark.unit
    def test_lower_bound_violation(self) -> None:
        """Value below lower bound is reported."""
        result = validate_parameters({"D0_ref": 10.0})  # min is 100
        assert result.valid is False
        assert result.parameters_checked == 1
        assert len(result.violations) == 1
        assert "lower bound" in result.violations[0]

    @pytest.mark.unit
    def test_upper_bound_violation(self) -> None:
        """Value above upper bound is reported."""
        result = validate_parameters({"alpha_ref": 5.0})  # max is 2.0
        assert result.valid is False
        assert "upper bound" in result.violations[0]

    @pytest.mark.unit
    def test_nan_detected(self) -> None:
        """NaN value is reported as 'not finite'."""
        result = validate_parameters({"D0_ref": float("nan")})
        assert result.valid is False
        assert "not finite" in result.violations[0]

    @pytest.mark.unit
    def test_inf_detected(self) -> None:
        """Infinity is reported as 'not finite'."""
        result = validate_parameters({"D0_ref": float("inf")})
        assert result.valid is False
        assert "not finite" in result.violations[0]

    @pytest.mark.unit
    def test_unknown_params_skipped(self) -> None:
        """Parameters not in bounds dict are silently skipped."""
        result = validate_parameters({"unknown_param": 42.0, "another": -1.0})
        assert result.valid is True
        assert result.parameters_checked == 0

    @pytest.mark.unit
    def test_multiple_violations(self) -> None:
        """Multiple violations are all reported."""
        result = validate_parameters({
            "D0_ref": -1.0,       # below min 100
            "alpha_ref": 99.0,    # above max 2.0
            "v0": float("nan"),   # not finite
        })
        assert result.valid is False
        assert result.parameters_checked == 3
        assert len(result.violations) == 3

    @pytest.mark.unit
    def test_custom_bounds(self) -> None:
        """Custom bounds override defaults."""
        custom = {"x": (0.0, 10.0)}
        result = validate_parameters({"x": 5.0}, bounds=custom)
        assert result.valid is True

        result = validate_parameters({"x": 15.0}, bounds=custom)
        assert result.valid is False

    @pytest.mark.unit
    def test_exact_boundary_values_valid(self) -> None:
        """Values exactly at bounds are valid."""
        result = validate_parameters({
            "D0_ref": 100.0,   # exact lower bound
            "alpha_ref": 2.0,  # exact upper bound
        })
        assert result.valid is True

    @pytest.mark.unit
    def test_validation_result_str(self) -> None:
        """ValidationResult __str__ formats correctly."""
        ok = ValidationResult(valid=True, message="All good")
        assert "OK All good" in str(ok)

        fail = ValidationResult(
            valid=False,
            violations=["x too low", "y too high"],
            message="2 violations",
        )
        s = str(fail)
        assert "FAIL" in s
        assert "x too low" in s
        assert "y too high" in s
