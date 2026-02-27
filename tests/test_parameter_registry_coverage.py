"""Tests for config/parameter_registry.py module.

Covers ParameterInfo, ParameterRegistry methods for improved coverage.
"""

from __future__ import annotations

import pytest

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import (
    DEFAULT_REGISTRY,
    ParameterInfo,
    ParameterRegistry,
)

# ============================================================================
# Test ParameterInfo
# ============================================================================


class TestParameterInfo:
    """Tests for ParameterInfo dataclass."""

    @pytest.fixture
    def param_info(self) -> ParameterInfo:
        """Create a ParameterInfo for testing."""
        return ParameterInfo(
            name="test_param",
            default=1.0,
            min_bound=0.0,
            max_bound=10.0,
            description="A test parameter",
            unit="nm",
            group="test",
            vary_default=True,
        )

    def test_validate_value_within_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns True for value within bounds."""
        assert param_info.validate_value(5.0) is True

    def test_validate_value_at_lower_bound(self, param_info: ParameterInfo) -> None:
        """validate_value returns True at lower bound."""
        assert param_info.validate_value(0.0) is True

    def test_validate_value_at_upper_bound(self, param_info: ParameterInfo) -> None:
        """validate_value returns True at upper bound."""
        assert param_info.validate_value(10.0) is True

    def test_validate_value_below_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns False below lower bound."""
        assert param_info.validate_value(-1.0) is False

    def test_validate_value_above_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns False above upper bound."""
        assert param_info.validate_value(11.0) is False

    def test_clip_value_within_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value returns same value if within bounds."""
        assert param_info.clip_value(5.0) == 5.0

    def test_clip_value_below_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value clips to lower bound."""
        assert param_info.clip_value(-5.0) == 0.0

    def test_clip_value_above_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value clips to upper bound."""
        assert param_info.clip_value(15.0) == 10.0

    def test_frozen_dataclass(self, param_info: ParameterInfo) -> None:
        """ParameterInfo is frozen (immutable)."""
        with pytest.raises(AttributeError):
            param_info.default = 2.0  # type: ignore[misc]


# ============================================================================
# Test ParameterRegistry
# ============================================================================


class TestParameterRegistry:
    """Tests for ParameterRegistry class."""

    @pytest.fixture
    def registry(self) -> ParameterRegistry:
        """Create a ParameterRegistry for testing."""
        return ParameterRegistry()

    def test_len(self, registry: ParameterRegistry) -> None:
        """len(registry) returns 14."""
        assert len(registry) == 14

    def test_getitem_valid_name(self, registry: ParameterRegistry) -> None:
        """__getitem__ returns ParameterInfo for valid name."""
        info = registry["D0_ref"]
        assert isinstance(info, ParameterInfo)
        assert info.name == "D0_ref"

    def test_getitem_invalid_name_raises(self, registry: ParameterRegistry) -> None:
        """__getitem__ raises KeyError for invalid name."""
        with pytest.raises(KeyError, match="Unknown parameter"):
            _ = registry["invalid_name"]

    def test_iter_yields_canonical_order(self, registry: ParameterRegistry) -> None:
        """__iter__ yields names in canonical order."""
        names = list(registry)
        assert names == list(ALL_PARAM_NAMES)

    def test_get_defaults(self, registry: ParameterRegistry) -> None:
        """get_defaults returns dict of default values."""
        defaults = registry.get_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) == 14
        for name in ALL_PARAM_NAMES:
            assert name in defaults

    def test_get_bounds(self, registry: ParameterRegistry) -> None:
        """get_bounds returns lower and upper bounds lists."""
        lower, upper = registry.get_bounds()
        assert len(lower) == 14
        assert len(upper) == 14
        # All bounds should be sensible
        for lo, hi in zip(lower, upper, strict=True):
            assert lo <= hi

    def test_get_group_reference(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for reference group."""
        group = registry.get_group("reference")
        assert len(group) == 3
        assert all(isinstance(p, ParameterInfo) for p in group)
        names = [p.name for p in group]
        assert "D0_ref" in names
        assert "alpha_ref" in names
        assert "D_offset_ref" in names

    def test_get_group_sample(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for sample group."""
        group = registry.get_group("sample")
        assert len(group) == 3

    def test_get_group_velocity(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for velocity group."""
        group = registry.get_group("velocity")
        assert len(group) == 3

    def test_get_group_fraction(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for fraction group."""
        group = registry.get_group("fraction")
        assert len(group) == 4

    def test_get_group_angle(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for angle group."""
        group = registry.get_group("angle")
        assert len(group) == 1
        assert group[0].name == "phi0"

    def test_get_group_invalid_raises(self, registry: ParameterRegistry) -> None:
        """get_group raises KeyError for invalid group name."""
        with pytest.raises(KeyError, match="Unknown group"):
            registry.get_group("invalid_group")

    def test_get_varying_indices_default_flags(
        self, registry: ParameterRegistry
    ) -> None:
        """get_varying_indices uses default vary flags when empty dict passed."""
        indices = registry.get_varying_indices({})
        # Should return indices of params with vary_default=True
        assert len(indices) > 0
        assert all(isinstance(i, int) for i in indices)

    def test_get_varying_indices_custom_flags(
        self, registry: ParameterRegistry
    ) -> None:
        """get_varying_indices uses provided vary flags."""
        # All fixed
        indices = registry.get_varying_indices(dict.fromkeys(ALL_PARAM_NAMES, False))
        assert len(indices) == 0

        # All varying
        indices = registry.get_varying_indices(dict.fromkeys(ALL_PARAM_NAMES, True))
        assert len(indices) == 14


# ============================================================================
# Test DEFAULT_REGISTRY module constant
# ============================================================================


class TestDefaultRegistry:
    """Tests for DEFAULT_REGISTRY module constant."""

    def test_default_registry_exists(self) -> None:
        """DEFAULT_REGISTRY is a ParameterRegistry instance."""
        assert isinstance(DEFAULT_REGISTRY, ParameterRegistry)

    def test_default_registry_has_all_params(self) -> None:
        """DEFAULT_REGISTRY contains all 14 parameters."""
        assert len(DEFAULT_REGISTRY) == 14

    def test_default_registry_d0_ref_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct D0_ref bounds."""
        info = DEFAULT_REGISTRY["D0_ref"]
        assert info.min_bound == 1e-12  # Nonzero to prevent degenerate fits
        assert info.max_bound == 1e6

    def test_default_registry_alpha_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct alpha bounds."""
        info = DEFAULT_REGISTRY["alpha_ref"]
        assert info.min_bound == -2.0
        assert info.max_bound == 2.0

    def test_default_registry_fraction_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct fraction parameter bounds."""
        info = DEFAULT_REGISTRY["f0"]
        assert info.min_bound == 0.0
        assert info.max_bound == 1.0

    def test_default_registry_vary_defaults(self) -> None:
        """DEFAULT_REGISTRY has sensible vary_default flags."""
        # Offsets typically fixed by default
        assert DEFAULT_REGISTRY["D_offset_ref"].vary_default is False
        # Main parameters vary by default
        assert DEFAULT_REGISTRY["D0_ref"].vary_default is True
