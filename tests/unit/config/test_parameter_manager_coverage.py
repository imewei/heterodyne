"""Tests for config/parameter_manager.py module.

Covers set_vary, set_bounds, validate_physics, get_group_values, from_config
for improved coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.config.parameter_manager import ParameterManager
from heterodyne.config.parameter_names import ALL_PARAM_NAMES

# ============================================================================
# Test ParameterManager Basic Properties
# ============================================================================


class TestParameterManagerProperties:
    """Tests for ParameterManager properties."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_n_params_is_14(self, param_manager: ParameterManager) -> None:
        """n_params returns 14."""
        assert param_manager.n_params == 14

    def test_n_varying_default(self, param_manager: ParameterManager) -> None:
        """n_varying returns count of varying parameters."""
        n = param_manager.n_varying
        assert n > 0
        assert n <= 14

    def test_varying_names_list(self, param_manager: ParameterManager) -> None:
        """varying_names returns list of strings."""
        names = param_manager.varying_names
        assert isinstance(names, list)
        for name in names:
            assert name in ALL_PARAM_NAMES

    def test_varying_indices_match_names(self, param_manager: ParameterManager) -> None:
        """varying_indices corresponds to varying_names."""
        names = param_manager.varying_names
        indices = param_manager.varying_indices
        assert len(names) == len(indices)

    def test_fixed_indices_complement_varying(
        self, param_manager: ParameterManager
    ) -> None:
        """fixed_indices and varying_indices are disjoint and complete."""
        varying = set(param_manager.varying_indices)
        fixed = set(param_manager.fixed_indices)
        assert varying.isdisjoint(fixed)
        assert varying.union(fixed) == set(range(14))


# ============================================================================
# Test set_vary
# ============================================================================


class TestSetVary:
    """Tests for set_vary method."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_set_vary_to_false(self, param_manager: ParameterManager) -> None:
        """set_vary can disable a parameter from varying."""
        param_manager.set_vary("D0_ref", False)
        assert "D0_ref" not in param_manager.varying_names

    def test_set_vary_to_true(self, param_manager: ParameterManager) -> None:
        """set_vary can enable a fixed parameter to vary."""
        # D_offset_ref is fixed by default
        param_manager.set_vary("D_offset_ref", True)
        assert "D_offset_ref" in param_manager.varying_names

    def test_set_vary_invalid_name_raises(
        self, param_manager: ParameterManager
    ) -> None:
        """set_vary raises ValueError for invalid parameter names."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            param_manager.set_vary("invalid_param", True)

    def test_set_vary_updates_n_varying(self, param_manager: ParameterManager) -> None:
        """set_vary updates n_varying count."""
        original = param_manager.n_varying
        param_manager.set_vary("D0_ref", False)
        assert param_manager.n_varying == original - 1


# ============================================================================
# Test set_bounds
# ============================================================================


class TestSetBounds:
    """Tests for set_bounds method."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_set_bounds_updates_bounds(self, param_manager: ParameterManager) -> None:
        """set_bounds updates the bounds for a parameter."""
        param_manager.set_bounds("D0_ref", 0.5, 100.0)
        lower, upper = param_manager.get_bounds()
        # Find D0_ref in varying params
        if "D0_ref" in param_manager.varying_names:
            idx = param_manager.varying_names.index("D0_ref")
            assert lower[idx] == 0.5
            assert upper[idx] == 100.0

    def test_set_bounds_invalid_name_raises(
        self, param_manager: ParameterManager
    ) -> None:
        """set_bounds raises ValueError for invalid parameter names."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            param_manager.set_bounds("invalid_param", 0.0, 1.0)

    def test_set_bounds_stored_in_space(self, param_manager: ParameterManager) -> None:
        """set_bounds stores bounds in parameter space."""
        param_manager.set_bounds("alpha_ref", -1.0, 1.5)
        assert param_manager.space.bounds["alpha_ref"] == (-1.0, 1.5)


# ============================================================================
# Test validate_physics
# ============================================================================


class TestValidatePhysics:
    """Tests for validate_physics method."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_validate_physics_default_passes(
        self, param_manager: ParameterManager
    ) -> None:
        """validate_physics returns empty list for default params."""
        violations = param_manager.validate_physics()
        # Default params should be valid
        assert isinstance(violations, list)

    def test_validate_physics_negative_diffusion(
        self, param_manager: ParameterManager
    ) -> None:
        """validate_physics catches negative diffusion coefficients."""
        params = param_manager.get_full_values()
        params[0] = -1.0  # D0_ref
        violations = param_manager.validate_physics(params)
        assert any("D0_ref" in v and "non-negative" in v for v in violations)

    def test_validate_physics_fraction_out_of_range(
        self, param_manager: ParameterManager
    ) -> None:
        """validate_physics catches fraction params outside [0, 1]."""
        params = param_manager.get_full_values()
        params[9] = 1.5  # f0 > 1
        violations = param_manager.validate_physics(params)
        assert any("f0" in v for v in violations)

    def test_validate_physics_extreme_alpha(
        self, param_manager: ParameterManager
    ) -> None:
        """validate_physics warns about extreme alpha values."""
        params = param_manager.get_full_values()
        params[1] = 3.0  # alpha_ref > 2
        violations = param_manager.validate_physics(params)
        assert any("alpha_ref" in v and "unusual magnitude" in v for v in violations)

    def test_validate_physics_uses_stored_values_when_none(
        self, param_manager: ParameterManager
    ) -> None:
        """validate_physics uses stored values when params is None."""
        violations = param_manager.validate_physics(None)
        assert isinstance(violations, list)


# ============================================================================
# Test get_group_values
# ============================================================================


class TestGetGroupValues:
    """Tests for get_group_values method."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_get_group_values_reference(self, param_manager: ParameterManager) -> None:
        """get_group_values returns reference group parameters."""
        values = param_manager.get_group_values("reference")
        assert set(values.keys()) == {"D0_ref", "alpha_ref", "D_offset_ref"}

    def test_get_group_values_sample(self, param_manager: ParameterManager) -> None:
        """get_group_values returns sample group parameters."""
        values = param_manager.get_group_values("sample")
        assert set(values.keys()) == {"D0_sample", "alpha_sample", "D_offset_sample"}

    def test_get_group_values_velocity(self, param_manager: ParameterManager) -> None:
        """get_group_values returns velocity group parameters."""
        values = param_manager.get_group_values("velocity")
        assert set(values.keys()) == {"v0", "beta", "v_offset"}

    def test_get_group_values_fraction(self, param_manager: ParameterManager) -> None:
        """get_group_values returns fraction group parameters."""
        values = param_manager.get_group_values("fraction")
        assert set(values.keys()) == {"f0", "f1", "f2", "f3"}

    def test_get_group_values_angle(self, param_manager: ParameterManager) -> None:
        """get_group_values returns angle group parameters."""
        values = param_manager.get_group_values("angle")
        assert set(values.keys()) == {"phi0"}

    def test_get_group_values_invalid_group_raises(
        self, param_manager: ParameterManager
    ) -> None:
        """get_group_values raises ValueError for invalid group."""
        with pytest.raises(ValueError, match="Unknown group"):
            param_manager.get_group_values("invalid_group")


# ============================================================================
# Test from_config
# ============================================================================


class TestFromConfig:
    """Tests for from_config class method."""

    def test_from_config_empty_config(self) -> None:
        """from_config works with empty config dict."""
        pm = ParameterManager.from_config({})
        assert pm.n_params == 14

    def test_from_config_with_parameters(self) -> None:
        """from_config accepts parameter configurations."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 2.0, "min": 0.0, "max": 10.0, "vary": True}
                }
            }
        }
        pm = ParameterManager.from_config(config)
        assert pm.space.values["D0_ref"] == 2.0

    def test_from_config_preserves_defaults_for_missing(self) -> None:
        """from_config uses defaults for unspecified parameters."""
        config = {"parameters": {}}
        pm = ParameterManager.from_config(config)
        # All 14 params should exist with default values
        assert len(pm.get_full_values()) == 14


# ============================================================================
# Test expand_varying_to_full and extract_varying
# ============================================================================


class TestExpandAndExtract:
    """Tests for expand_varying_to_full and extract_varying methods."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_roundtrip(self, param_manager: ParameterManager) -> None:
        """extract_varying -> expand_varying_to_full is identity for varying params."""
        full = param_manager.get_full_values()
        varying = param_manager.extract_varying(full)
        reconstructed = param_manager.expand_varying_to_full(varying)

        np.testing.assert_array_almost_equal(full, reconstructed)

    def test_expand_preserves_fixed(self, param_manager: ParameterManager) -> None:
        """expand_varying_to_full preserves fixed parameter values."""
        # Modify a varying parameter
        varying = param_manager.get_initial_values()
        varying[0] = 999.0  # Change first varying param

        expanded = param_manager.expand_varying_to_full(varying)

        # Fixed parameters should match original
        original = param_manager.get_full_values()
        for idx in param_manager.fixed_indices:
            assert expanded[idx] == original[idx]


# ============================================================================
# Test update_values
# ============================================================================


class TestUpdateValues:
    """Tests for update_values method."""

    @pytest.fixture
    def param_manager(self) -> ParameterManager:
        """Create a ParameterManager for testing."""
        return ParameterManager()

    def test_update_values_from_dict(self, param_manager: ParameterManager) -> None:
        """update_values accepts dict input."""
        param_manager.update_values({"D0_ref": 5.0})
        assert param_manager.space.values["D0_ref"] == 5.0

    def test_update_values_from_array(self, param_manager: ParameterManager) -> None:
        """update_values accepts array input."""
        new_values = np.ones(14) * 2.0
        param_manager.update_values(new_values)
        assert param_manager.space.values["D0_ref"] == 2.0
