"""Tests for physics validators module.

Tests validate_parameters, validate_time_integral_safety, and
validate_correlation_inputs functions for physics constraint checking.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.config.physics_validators import (
    ValidationResult,
    validate_correlation_inputs,
    validate_parameters,
    validate_time_integral_safety,
)

# ============================================================================
# ValidationResult Tests
# ============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    @pytest.mark.unit
    def test_valid_result_bool_true(self) -> None:
        """Valid result evaluates to True."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert bool(result) is True

    @pytest.mark.unit
    def test_invalid_result_bool_false(self) -> None:
        """Invalid result evaluates to False."""
        result = ValidationResult(is_valid=False, errors=["error"], warnings=[])
        assert bool(result) is False

    @pytest.mark.unit
    def test_result_with_warnings_still_valid(self) -> None:
        """Result with only warnings is still valid."""
        result = ValidationResult(is_valid=True, errors=[], warnings=["warning"])
        assert bool(result) is True


# ============================================================================
# validate_parameters Tests
# ============================================================================


class TestValidateParameters:
    """Tests for validate_parameters function."""

    @pytest.mark.unit
    def test_valid_array_14_params(self) -> None:
        """Valid 14-parameter array passes validation."""
        params = np.array([
            1.0, 1.0, 0.0,  # D0_ref, alpha_ref, offset_ref
            1.0, 1.0, 0.0,  # D0_sample, alpha_sample, offset_sample
            0.0, 1.0, 0.0,  # v0, beta, phase
            0.5, 0.1, 0.0, 0.1, 0.0,  # f0, f1, f2, f3, background
        ])
        result = validate_parameters(params)
        assert result.is_valid
        assert len(result.errors) == 0

    @pytest.mark.unit
    def test_wrong_array_length_fails(self) -> None:
        """Array with wrong length fails validation."""
        params = np.array([1.0, 2.0, 3.0])  # Only 3 params
        result = validate_parameters(params)
        assert not result.is_valid
        assert "Expected 14 parameters" in result.errors[0]

    @pytest.mark.unit
    def test_valid_dict_params(self) -> None:
        """Valid parameter dictionary passes validation."""
        params = {
            "D0_ref": 1.0,
            "alpha_ref": 1.0,
            "D0_sample": 1.0,
            "f0": 0.5,
            "f3": 0.1,
        }
        result = validate_parameters(params)
        assert result.is_valid

    @pytest.mark.unit
    def test_negative_D0_ref_error(self) -> None:
        """Negative D0_ref is an error."""
        params = {"D0_ref": -1.0}
        result = validate_parameters(params)
        assert not result.is_valid
        assert any("D0_ref" in e and "non-negative" in e for e in result.errors)

    @pytest.mark.unit
    def test_negative_D0_sample_error(self) -> None:
        """Negative D0_sample is an error."""
        params = {"D0_sample": -0.5}
        result = validate_parameters(params)
        assert not result.is_valid
        assert any("D0_sample" in e for e in result.errors)

    @pytest.mark.unit
    def test_f0_out_of_range_error(self) -> None:
        """f0 outside [0, 1] is an error."""
        # f0 > 1
        result = validate_parameters({"f0": 1.5})
        assert not result.is_valid
        assert any("f0" in e and "[0, 1]" in e for e in result.errors)

        # f0 < 0
        result = validate_parameters({"f0": -0.1})
        assert not result.is_valid

    @pytest.mark.unit
    def test_f3_out_of_range_error(self) -> None:
        """f3 outside [0, 1] is an error."""
        result = validate_parameters({"f3": 2.0})
        assert not result.is_valid
        assert any("f3" in e for e in result.errors)

    @pytest.mark.unit
    def test_f0_plus_f3_exceeds_one_error(self) -> None:
        """f0 + f3 > 1 is a physical impossibility — produces error."""
        params = {"f0": 0.8, "f3": 0.5}
        result = validate_parameters(params)
        assert not result.is_valid  # Error, not just a warning
        assert any("f0 + f3" in e for e in result.errors)

    @pytest.mark.unit
    def test_unusual_alpha_warning(self) -> None:
        """Unusual exponent values generate warnings."""
        params = {"alpha_ref": 5.0}  # |alpha| > 2
        result = validate_parameters(params)
        assert result.is_valid
        assert any("alpha_ref" in w and "unusual" in w for w in result.warnings)

    @pytest.mark.unit
    def test_unusual_beta_warning(self) -> None:
        """Unusual beta exponent generates warning."""
        params = {"beta": -3.0}  # |beta| > 2
        result = validate_parameters(params)
        assert result.is_valid
        assert any("beta" in w for w in result.warnings)

    @pytest.mark.unit
    def test_large_diffusion_coefficient_warning(self) -> None:
        """Very large D0 generates warning."""
        params = {"D0_ref": 1e6}  # > 1e5
        result = validate_parameters(params)
        assert result.is_valid
        assert any("D0_ref" in w and "large" in w for w in result.warnings)

    @pytest.mark.unit
    def test_large_velocity_warning(self) -> None:
        """Very large velocity generates warning."""
        params = {"v0": 5000.0}  # > 1e3
        result = validate_parameters(params)
        assert result.is_valid
        assert any("v0" in w and "large" in w for w in result.warnings)

    @pytest.mark.unit
    def test_large_f1_warning(self) -> None:
        """Large f1 (rapid fraction change) generates warning."""
        params = {"f1": 10.0}  # |f1| > 5
        result = validate_parameters(params)
        assert result.is_valid
        assert any("f1" in w and "rapidly" in w for w in result.warnings)


# ============================================================================
# validate_time_integral_safety Tests
# ============================================================================


class TestValidateTimeIntegralSafety:
    """Tests for validate_time_integral_safety function."""

    @pytest.mark.unit
    def test_normal_alpha_valid(self) -> None:
        """Normal alpha values pass validation."""
        result = validate_time_integral_safety(alpha=1.0, t_min=0.01, t_max=100.0)
        assert result.is_valid
        assert len(result.errors) == 0

    @pytest.mark.unit
    def test_negative_alpha_with_t_min_zero_error(self) -> None:
        """Negative alpha with t_min=0 is an error (singularity at t=0)."""
        result = validate_time_integral_safety(alpha=-0.5, t_min=0.0, t_max=100.0)
        assert not result.is_valid
        assert any("t_min > 0" in e for e in result.errors)

    @pytest.mark.unit
    def test_negative_alpha_with_positive_t_min_valid(self) -> None:
        """Negative alpha with t_min > 0 is valid."""
        result = validate_time_integral_safety(alpha=-0.5, t_min=0.01, t_max=100.0)
        assert result.is_valid

    @pytest.mark.unit
    def test_very_negative_alpha_warning(self) -> None:
        """Very negative alpha (< -1) generates instability warning."""
        result = validate_time_integral_safety(alpha=-1.5, t_min=0.01, t_max=100.0)
        assert result.is_valid
        assert any("numerical instability" in w for w in result.warnings)

    @pytest.mark.unit
    def test_large_alpha_overflow_warning(self) -> None:
        """Large alpha with large t_max generates overflow warning."""
        # Need t_max^alpha > 1e15 to trigger warning
        # 10000^4 = 1e16 > 1e15
        result = validate_time_integral_safety(alpha=4.0, t_min=0.01, t_max=10000.0)
        assert result.is_valid
        assert any("overflow" in w for w in result.warnings)

    @pytest.mark.unit
    def test_moderate_alpha_no_overflow(self) -> None:
        """Moderate alpha with small t_max doesn't warn."""
        result = validate_time_integral_safety(alpha=3.5, t_min=0.01, t_max=10.0)
        # 10^3.5 ≈ 3162, much less than 1e15
        assert result.is_valid
        assert len(result.warnings) == 0


# ============================================================================
# validate_correlation_inputs Tests
# ============================================================================


class TestValidateCorrelationInputs:
    """Tests for validate_correlation_inputs function."""

    @pytest.mark.unit
    def test_valid_inputs(self) -> None:
        """Valid correlation inputs pass validation."""
        t1 = np.array([0.0, 1.0, 2.0, 3.0])
        t2 = np.array([0.0, 1.0, 2.0, 3.0])
        c2_data = np.random.rand(4, 4)

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert result.is_valid

    @pytest.mark.unit
    def test_shape_mismatch_error(self) -> None:
        """Shape mismatch between time grids and data is an error."""
        t1 = np.array([0.0, 1.0, 2.0])  # 3 points
        t2 = np.array([0.0, 1.0, 2.0, 3.0])  # 4 points
        c2_data = np.random.rand(5, 5)  # Wrong shape

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid
        assert any("shape" in e for e in result.errors)

    @pytest.mark.unit
    def test_nan_values_error(self) -> None:
        """NaN values in c2_data is an error."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 1.0, 2.0])
        c2_data = np.array([[1.0, np.nan, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    @pytest.mark.unit
    def test_inf_values_error(self) -> None:
        """Infinite values in c2_data is an error."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 1.0, 2.0])
        c2_data = np.array([[1.0, np.inf, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid
        assert any("infinite" in e for e in result.errors)

    @pytest.mark.unit
    def test_negative_values_warning(self) -> None:
        """Negative values in c2_data generates warning."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 1.0, 2.0])
        c2_data = np.array([[1.0, -0.1, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert result.is_valid  # Only a warning
        assert any("negative" in w for w in result.warnings)

    @pytest.mark.unit
    def test_values_greater_than_two_warning(self) -> None:
        """Values > 2 generates warning for normalized correlation."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 1.0, 2.0])
        c2_data = np.array([[1.0, 2.5, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert result.is_valid
        assert any("> 2" in w for w in result.warnings)

    @pytest.mark.unit
    def test_non_monotonic_t1_error(self) -> None:
        """Non-monotonic t1 is an error."""
        t1 = np.array([0.0, 2.0, 1.0])  # Not monotonic
        t2 = np.array([0.0, 1.0, 2.0])
        c2_data = np.ones((3, 3))

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid
        assert any("t1" in e and "increasing" in e for e in result.errors)

    @pytest.mark.unit
    def test_non_monotonic_t2_error(self) -> None:
        """Non-monotonic t2 is an error."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 2.0, 1.0])  # Not monotonic
        c2_data = np.ones((3, 3))

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid
        assert any("t2" in e and "increasing" in e for e in result.errors)

    @pytest.mark.unit
    def test_equal_consecutive_times_error(self) -> None:
        """Equal consecutive times (not strictly increasing) is an error."""
        t1 = np.array([0.0, 1.0, 1.0, 2.0])  # Duplicate value
        t2 = np.array([0.0, 1.0, 2.0, 3.0])
        c2_data = np.ones((4, 4))

        result = validate_correlation_inputs(t1, t2, c2_data)
        assert not result.is_valid


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhysicsValidatorsIntegration:
    """Integration tests combining multiple validators."""

    @pytest.mark.integration
    def test_full_valid_scenario(self) -> None:
        """Full valid scenario passes all validators."""
        # Valid parameters
        params = np.array([
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.5, 0.1, 0.0, 0.1, 0.0,
        ])
        param_result = validate_parameters(params)
        assert param_result.is_valid

        # Valid time integral settings
        time_result = validate_time_integral_safety(
            alpha=params[1], t_min=0.01, t_max=100.0
        )
        assert time_result.is_valid

        # Valid correlation inputs
        t = np.linspace(0.01, 100.0, 50)
        c2_data = np.random.rand(50, 50) + 0.5  # Values in [0.5, 1.5]
        corr_result = validate_correlation_inputs(t, t, c2_data)
        assert corr_result.is_valid

    @pytest.mark.integration
    def test_edge_case_minimal_fraction(self) -> None:
        """Edge case with f0=0, f3=0 (no scattering fraction)."""
        params = {"f0": 0.0, "f3": 0.0, "D0_ref": 1.0, "D0_sample": 1.0}
        result = validate_parameters(params)
        assert result.is_valid

    @pytest.mark.integration
    def test_edge_case_maximal_fraction(self) -> None:
        """Edge case with f0=1, f3=0."""
        params = {"f0": 1.0, "f3": 0.0, "D0_ref": 1.0, "D0_sample": 1.0}
        result = validate_parameters(params)
        assert result.is_valid
