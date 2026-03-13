"""Tests for model output validation and parameter counting in cmc/model.py."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from heterodyne.optimization.cmc.model import (
    get_model_param_count,
    validate_model_output,
)

# ---------------------------------------------------------------------------
# validate_model_output
# ---------------------------------------------------------------------------


class TestValidateModelOutput:
    """Tests for validate_model_output()."""

    def test_valid_c2(self) -> None:
        c2 = jnp.array([0.5, 0.8, 1.0, 0.3])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is True

    def test_negative_c2_in_range(self) -> None:
        """Heterodyne C2 can be negative (velocity phase), down to -1.0."""
        c2 = jnp.array([-0.5, 0.0, 0.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is True

    def test_boundary_values(self) -> None:
        """Exactly at bounds [-1.0, 10.0] should pass."""
        c2 = jnp.array([-1.0, 10.0])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is True

    def test_nan_detected(self) -> None:
        c2 = jnp.array([1.0, float("nan"), 0.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_inf_detected(self) -> None:
        c2 = jnp.array([1.0, float("inf")])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_neg_inf_detected(self) -> None:
        c2 = jnp.array([float("-inf"), 0.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_below_lower_bound(self) -> None:
        c2 = jnp.array([-1.5, 0.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_above_upper_bound(self) -> None:
        c2 = jnp.array([0.5, 10.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_all_nan(self) -> None:
        c2 = jnp.array([float("nan"), float("nan")])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False

    def test_single_element_valid(self) -> None:
        c2 = jnp.array([0.5])
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is True

    def test_2d_array_valid(self) -> None:
        """Validation should work on 2D C2 arrays (meshgrid path)."""
        c2 = jnp.ones((5, 5)) * 0.5
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is True

    def test_2d_array_with_nan(self) -> None:
        c2 = jnp.ones((3, 3)).at[1, 1].set(float("nan"))
        params = jnp.ones(14)
        assert validate_model_output(c2, params) is False


# ---------------------------------------------------------------------------
# get_model_param_count
# ---------------------------------------------------------------------------


class TestGetModelParamCount:
    """Tests for get_model_param_count()."""

    def test_auto_mode(self) -> None:
        """Auto mode: physics params only (contrast/offset in param space)."""
        count = get_model_param_count(n_phi=4, per_angle_mode="auto")
        # All 16 params have vary_default=True, but get_model_param_count
        # counts only ALL_PARAM_NAMES (14 physics) that have vary_default.
        # No per-angle additions for "auto".
        assert count >= 14
        assert isinstance(count, int)

    def test_constant_mode(self) -> None:
        """Constant mode: same as auto (no additional per-angle params)."""
        count_const = get_model_param_count(n_phi=4, per_angle_mode="constant")
        count_auto = get_model_param_count(n_phi=4, per_angle_mode="auto")
        assert count_const == count_auto

    def test_constant_averaged_mode(self) -> None:
        count = get_model_param_count(n_phi=4, per_angle_mode="constant_averaged")
        count_auto = get_model_param_count(n_phi=4, per_angle_mode="auto")
        assert count == count_auto

    def test_individual_mode(self) -> None:
        """Individual mode adds 2 * n_phi per-angle params."""
        n_phi = 5
        count_individual = get_model_param_count(
            n_phi=n_phi, per_angle_mode="individual"
        )
        count_auto = get_model_param_count(n_phi=n_phi, per_angle_mode="auto")
        assert count_individual == count_auto + 2 * n_phi

    def test_individual_mode_scaling(self) -> None:
        """Doubling n_phi should double the per-angle contribution."""
        count_4 = get_model_param_count(n_phi=4, per_angle_mode="individual")
        count_8 = get_model_param_count(n_phi=8, per_angle_mode="individual")
        assert count_8 - count_4 == 2 * (8 - 4)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown per_angle_mode"):
            get_model_param_count(n_phi=1, per_angle_mode="invalid")

    def test_n_phi_zero_individual(self) -> None:
        """n_phi=0 with individual mode should equal physics-only count."""
        count = get_model_param_count(n_phi=0, per_angle_mode="individual")
        count_auto = get_model_param_count(n_phi=0, per_angle_mode="auto")
        assert count == count_auto

    def test_return_type(self) -> None:
        count = get_model_param_count(n_phi=3, per_angle_mode="auto")
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# get_model_for_mode (mode validation only - no physics)
# ---------------------------------------------------------------------------


class TestGetModelForModeValidation:
    """Test mode validation in get_model_for_mode without running physics."""

    def test_invalid_mode_raises(self) -> None:
        from heterodyne.optimization.cmc.model import get_model_for_mode

        with pytest.raises(ValueError, match="Unknown per_angle_mode"):
            get_model_for_mode(
                per_angle_mode="nonexistent",
                t=jnp.linspace(0.01, 1.0, 10),
                q=0.01,
                dt=0.01,
                phi_angle=0.0,
                c2_data=jnp.ones(10),
                sigma=0.1,
                space=None,  # type: ignore[arg-type]
            )
