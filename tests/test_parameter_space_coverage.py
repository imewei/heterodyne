"""Tests for config/parameter_space.py module.

Covers validate, from_config, PriorDistribution.to_numpyro for improved coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_space import (
    ParameterSpace,
    PriorDistribution,
    PriorType,
)

# ============================================================================
# Test PriorType Enum
# ============================================================================


class TestPriorType:
    """Tests for PriorType enum."""

    def test_enum_values(self) -> None:
        """PriorType has expected values."""
        assert PriorType.UNIFORM.value == "uniform"
        assert PriorType.NORMAL.value == "normal"
        assert PriorType.LOGNORMAL.value == "lognormal"
        assert PriorType.HALFNORMAL.value == "halfnormal"
        assert PriorType.EXPONENTIAL.value == "exponential"


# ============================================================================
# Test PriorDistribution
# ============================================================================


class TestPriorDistribution:
    """Tests for PriorDistribution class."""

    def test_uniform_factory(self) -> None:
        """PriorDistribution.uniform creates uniform prior."""
        prior = PriorDistribution.uniform(0.0, 10.0)
        assert prior.prior_type == PriorType.UNIFORM
        assert prior.params == {"low": 0.0, "high": 10.0}

    def test_normal_factory(self) -> None:
        """PriorDistribution.normal creates normal prior."""
        prior = PriorDistribution.normal(1.0, 0.5)
        assert prior.prior_type == PriorType.NORMAL
        assert prior.params == {"loc": 1.0, "scale": 0.5}

    def test_lognormal_factory(self) -> None:
        """PriorDistribution.lognormal creates lognormal prior."""
        prior = PriorDistribution.lognormal(0.0, 1.0)
        assert prior.prior_type == PriorType.LOGNORMAL
        assert prior.params == {"loc": 0.0, "scale": 1.0}

    def test_halfnormal_factory(self) -> None:
        """PriorDistribution.halfnormal creates halfnormal prior."""
        prior = PriorDistribution.halfnormal(2.0)
        assert prior.prior_type == PriorType.HALFNORMAL
        assert prior.params == {"scale": 2.0}

    def test_to_numpyro_uniform(self) -> None:
        """to_numpyro converts uniform prior correctly."""
        prior = PriorDistribution.uniform(0.0, 10.0)
        dist = prior.to_numpyro("test")
        # Check it's a distribution object
        assert hasattr(dist, "sample")

    def test_to_numpyro_normal(self) -> None:
        """to_numpyro converts normal prior correctly."""
        prior = PriorDistribution.normal(1.0, 0.5)
        dist = prior.to_numpyro("test")
        assert hasattr(dist, "sample")

    def test_to_numpyro_lognormal(self) -> None:
        """to_numpyro converts lognormal prior correctly."""
        prior = PriorDistribution.lognormal(0.0, 1.0)
        dist = prior.to_numpyro("test")
        assert hasattr(dist, "sample")

    def test_to_numpyro_halfnormal(self) -> None:
        """to_numpyro converts halfnormal prior correctly."""
        prior = PriorDistribution.halfnormal(2.0)
        dist = prior.to_numpyro("test")
        assert hasattr(dist, "sample")

    def test_to_numpyro_exponential(self) -> None:
        """to_numpyro converts exponential prior correctly."""
        prior = PriorDistribution(PriorType.EXPONENTIAL, {"rate": 1.0})
        dist = prior.to_numpyro("test")
        assert hasattr(dist, "sample")

    def test_to_numpyro_exponential_default_rate(self) -> None:
        """to_numpyro uses default rate for exponential."""
        prior = PriorDistribution(PriorType.EXPONENTIAL, {})
        dist = prior.to_numpyro("test")
        assert hasattr(dist, "sample")


# ============================================================================
# Test ParameterSpace Basic Operations
# ============================================================================


class TestParameterSpaceBasics:
    """Tests for ParameterSpace basic functionality."""

    @pytest.fixture
    def space(self) -> ParameterSpace:
        """Create a ParameterSpace for testing."""
        return ParameterSpace()

    def test_post_init_populates_values(self, space: ParameterSpace) -> None:
        """__post_init__ populates values for all parameters."""
        assert len(space.values) == 14
        for name in ALL_PARAM_NAMES:
            assert name in space.values

    def test_post_init_populates_vary(self, space: ParameterSpace) -> None:
        """__post_init__ populates vary flags for all parameters."""
        assert len(space.vary) == 14

    def test_post_init_populates_bounds(self, space: ParameterSpace) -> None:
        """__post_init__ populates bounds for all parameters."""
        assert len(space.bounds) == 14

    def test_post_init_populates_priors(self, space: ParameterSpace) -> None:
        """__post_init__ populates priors for all parameters."""
        assert len(space.priors) == 14
        # Default priors are uniform
        for name in ALL_PARAM_NAMES:
            assert space.priors[name].prior_type == PriorType.UNIFORM

    def test_n_total(self, space: ParameterSpace) -> None:
        """n_total returns 14."""
        assert space.n_total == 14

    def test_n_varying(self, space: ParameterSpace) -> None:
        """n_varying returns count of varying parameters."""
        n = space.n_varying
        assert n > 0
        assert n <= 14

    def test_varying_names(self, space: ParameterSpace) -> None:
        """varying_names returns list of varying parameter names."""
        names = space.varying_names
        assert len(names) == space.n_varying
        for name in names:
            assert space.vary[name] is True

    def test_fixed_names(self, space: ParameterSpace) -> None:
        """fixed_names returns list of fixed parameter names."""
        names = space.fixed_names
        for name in names:
            assert space.vary[name] is False


# ============================================================================
# Test ParameterSpace Array Operations
# ============================================================================


class TestParameterSpaceArrays:
    """Tests for ParameterSpace array operations."""

    @pytest.fixture
    def space(self) -> ParameterSpace:
        """Create a ParameterSpace for testing."""
        return ParameterSpace()

    def test_get_initial_array(self, space: ParameterSpace) -> None:
        """get_initial_array returns array of shape (14,)."""
        arr = space.get_initial_array()
        assert arr.shape == (14,)
        assert isinstance(arr, np.ndarray)

    def test_get_bounds_arrays(self, space: ParameterSpace) -> None:
        """get_bounds_arrays returns (lower, upper) arrays."""
        lower, upper = space.get_bounds_arrays()
        assert lower.shape == (14,)
        assert upper.shape == (14,)
        # Bounds should be sensible
        assert np.all(lower <= upper)

    def test_get_vary_mask(self, space: ParameterSpace) -> None:
        """get_vary_mask returns boolean array."""
        mask = space.get_vary_mask()
        assert mask.shape == (14,)
        assert mask.dtype == bool
        assert np.sum(mask) == space.n_varying

    def test_array_to_dict(self, space: ParameterSpace) -> None:
        """array_to_dict converts array to dict."""
        arr = np.ones(14) * 5.0
        d = space.array_to_dict(arr)
        assert len(d) == 14
        for name in ALL_PARAM_NAMES:
            assert d[name] == 5.0

    def test_update_from_dict(self, space: ParameterSpace) -> None:
        """update_from_dict updates values."""
        space.update_from_dict({"D0_ref": 99.0, "alpha_ref": 0.5})
        assert space.values["D0_ref"] == 99.0
        assert space.values["alpha_ref"] == 0.5

    def test_update_from_dict_rejects_unknown(self, space: ParameterSpace) -> None:
        """update_from_dict raises ValueError for unknown keys."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            space.update_from_dict({"unknown_param": 1.0})


# ============================================================================
# Test ParameterSpace.validate
# ============================================================================


class TestParameterSpaceValidate:
    """Tests for ParameterSpace.validate method."""

    def test_validate_default_space_passes(self) -> None:
        """validate returns empty list for default space."""
        space = ParameterSpace()
        errors = space.validate()
        assert errors == []

    def test_validate_detects_out_of_bounds(self) -> None:
        """validate detects values outside bounds."""
        space = ParameterSpace()
        space.values["D0_ref"] = -100.0  # Below bound of 0
        errors = space.validate()
        assert len(errors) > 0
        assert any("D0_ref" in e for e in errors)

    def test_validate_detects_missing_value(self) -> None:
        """validate detects missing values."""
        space = ParameterSpace()
        del space.values["D0_ref"]
        errors = space.validate()
        assert len(errors) > 0
        assert any("Missing value" in e and "D0_ref" in e for e in errors)

    def test_validate_detects_missing_bounds(self) -> None:
        """validate detects missing bounds."""
        space = ParameterSpace()
        del space.bounds["alpha_ref"]
        errors = space.validate()
        assert len(errors) > 0
        assert any("Missing bounds" in e and "alpha_ref" in e for e in errors)

    def test_validate_multiple_errors(self) -> None:
        """validate reports all errors."""
        space = ParameterSpace()
        space.values["D0_ref"] = -100.0
        space.values["f0"] = 5.0  # Outside [0, 1]
        errors = space.validate()
        assert len(errors) >= 2


# ============================================================================
# Test ParameterSpace.from_config
# ============================================================================


class TestParameterSpaceFromConfig:
    """Tests for ParameterSpace.from_config class method."""

    def test_from_config_empty(self) -> None:
        """from_config works with empty config."""
        space = ParameterSpace.from_config({})
        assert space.n_total == 14

    def test_from_config_no_parameters_key(self) -> None:
        """from_config works without 'parameters' key."""
        config = {"temporal": {"dt": 0.001}}
        space = ParameterSpace.from_config(config)
        assert space.n_total == 14

    def test_from_config_with_value(self) -> None:
        """from_config reads parameter values."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 2.5}
                }
            }
        }
        space = ParameterSpace.from_config(config)
        assert space.values["D0_ref"] == 2.5

    def test_from_config_with_bounds(self) -> None:
        """from_config reads parameter bounds."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 1.0, "min": 0.1, "max": 100.0}
                }
            }
        }
        space = ParameterSpace.from_config(config)
        assert space.bounds["D0_ref"] == (0.1, 100.0)

    def test_from_config_with_vary(self) -> None:
        """from_config reads vary flag."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 1.0, "vary": False}
                }
            }
        }
        space = ParameterSpace.from_config(config)
        assert space.vary["D0_ref"] is False

    def test_from_config_multiple_groups(self) -> None:
        """from_config handles multiple parameter groups."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 1.0}
                },
                "sample": {
                    "D0_sample": {"value": 2.0}
                },
                "velocity": {
                    "v0": {"value": 100.0}
                }
            }
        }
        space = ParameterSpace.from_config(config)
        assert space.values["D0_ref"] == 1.0
        assert space.values["D0_sample"] == 2.0
        assert space.values["v0"] == 100.0

    def test_from_config_preserves_defaults(self) -> None:
        """from_config preserves defaults for unspecified params."""
        config = {
            "parameters": {
                "reference": {
                    "D0_ref": {"value": 5.0}
                }
            }
        }
        space = ParameterSpace.from_config(config)
        # alpha_ref should have default value
        default_space = ParameterSpace()
        assert space.values["alpha_ref"] == default_space.values["alpha_ref"]
