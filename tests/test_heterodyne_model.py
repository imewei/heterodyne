"""Tests for HeterodyneModel class.

Tests the main heterodyne correlation model wrapper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.core.heterodyne_model import HeterodyneModel

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def basic_config() -> dict:
    """Basic configuration for model creation."""
    return {
        "temporal": {
            "dt": 1.0,
            "time_length": 50,
            "t_start": 0,
        },
        "scattering": {
            "wavevector_q": 0.01,
        },
        "parameters": {
            "reference": {
                "D0": {"value": 1.0, "vary": True, "min": 0.0, "max": 10.0},
                "alpha": {"value": 1.0, "vary": False},
                "offset": {"value": 0.0, "vary": False},
            },
            "sample": {
                "D0": {"value": 1.0, "vary": True, "min": 0.0, "max": 10.0},
                "alpha": {"value": 1.0, "vary": False},
                "offset": {"value": 0.0, "vary": False},
            },
            "velocity": {
                "v0": {"value": 0.0, "vary": False},
                "beta": {"value": 1.0, "vary": False},
                "phase": {"value": 0.0, "vary": False},
            },
            "fraction": {
                "f0": {"value": 0.5, "vary": True, "min": 0.0, "max": 1.0},
                "f1": {"value": 0.0, "vary": False},
                "f2": {"value": 0.0, "vary": False},
                "f3": {"value": 0.0, "vary": False},
            },
            "other": {
                "background": {"value": 0.0, "vary": False},
            },
        },
    }


@pytest.fixture
def model(basic_config: dict) -> HeterodyneModel:
    """Create model from basic config."""
    return HeterodyneModel.from_config(basic_config)


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestHeterodyneModelFromConfig:
    """Tests for from_config factory method."""

    @pytest.mark.unit
    def test_from_config_creates_model(self, basic_config: dict) -> None:
        """from_config creates HeterodyneModel."""
        model = HeterodyneModel.from_config(basic_config)
        assert isinstance(model, HeterodyneModel)

    @pytest.mark.unit
    def test_from_config_sets_physics_factors(self, basic_config: dict) -> None:
        """from_config initializes physics factors."""
        model = HeterodyneModel.from_config(basic_config)
        assert model._factors is not None
        assert model.n_times == 50
        assert model.dt == 1.0
        assert model.q == 0.01

    @pytest.mark.unit
    def test_from_config_sets_time_array(self, basic_config: dict) -> None:
        """from_config initializes time array."""
        model = HeterodyneModel.from_config(basic_config)
        assert model._t is not None
        assert len(model.t) == 50


# ============================================================================
# Property Tests
# ============================================================================


class TestHeterodyneModelProperties:
    """Tests for model properties."""

    @pytest.mark.unit
    def test_n_params(self, model: HeterodyneModel) -> None:
        """n_params returns 14."""
        assert model.n_params == 14

    @pytest.mark.unit
    def test_n_varying(self, model: HeterodyneModel) -> None:
        """n_varying returns count of varying parameters."""
        # D0_ref, D0_sample, f0 vary in basic_config, plus background might be default vary
        n_varying = model.n_varying
        assert n_varying >= 3  # At least D0_ref, D0_sample, f0
        assert n_varying <= 14  # At most all params

    @pytest.mark.unit
    def test_param_names(self, model: HeterodyneModel) -> None:
        """param_names returns all 14 names."""
        assert model.param_names == ALL_PARAM_NAMES
        assert len(model.param_names) == 14

    @pytest.mark.unit
    def test_varying_names(self, model: HeterodyneModel) -> None:
        """varying_names returns list of varying parameter names."""
        varying = model.varying_names
        assert "D0_ref" in varying
        assert "D0_sample" in varying
        assert "f0" in varying

    @pytest.mark.unit
    def test_q_property(self, model: HeterodyneModel) -> None:
        """q property returns wavevector."""
        assert model.q == 0.01

    @pytest.mark.unit
    def test_dt_property(self, model: HeterodyneModel) -> None:
        """dt property returns time step."""
        assert model.dt == 1.0

    @pytest.mark.unit
    def test_t_property(self, model: HeterodyneModel) -> None:
        """t property returns time array."""
        t = model.t
        assert len(t) == 50
        assert t[0] == 0.0

    @pytest.mark.unit
    def test_n_times_property(self, model: HeterodyneModel) -> None:
        """n_times property returns time count."""
        assert model.n_times == 50


# ============================================================================
# Parameter Management Tests
# ============================================================================


class TestHeterodyneModelParameters:
    """Tests for parameter get/set methods."""

    @pytest.mark.unit
    def test_get_params(self, model: HeterodyneModel) -> None:
        """get_params returns full parameter array."""
        params = model.get_params()
        assert params.shape == (14,)
        assert isinstance(params, np.ndarray)

    @pytest.mark.unit
    def test_get_params_dict(self, model: HeterodyneModel) -> None:
        """get_params_dict returns parameter dictionary."""
        params_dict = model.get_params_dict()
        assert len(params_dict) == 14
        assert "D0_ref" in params_dict
        assert "f0" in params_dict

    @pytest.mark.unit
    def test_set_params_array(self, model: HeterodyneModel) -> None:
        """set_params accepts array."""
        new_params = np.ones(14) * 2.0
        model.set_params(new_params)
        assert_allclose(model.get_params(), new_params)

    @pytest.mark.unit
    def test_set_params_dict(self, model: HeterodyneModel) -> None:
        """set_params accepts dictionary."""
        model.set_params({"D0_ref": 5.0})
        assert model.get_params_dict()["D0_ref"] == 5.0


# ============================================================================
# Compute Correlation Tests
# ============================================================================


class TestHeterodyneModelComputeCorrelation:
    """Tests for compute_correlation method."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_correlation_returns_array(self, model: HeterodyneModel) -> None:
        """compute_correlation returns jax array."""
        c2 = model.compute_correlation(phi_angle=0.0)
        assert isinstance(c2, jnp.ndarray)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_correlation_shape(self, model: HeterodyneModel) -> None:
        """compute_correlation returns (N, N) matrix."""
        c2 = model.compute_correlation(phi_angle=0.0)
        assert c2.shape == (50, 50)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_correlation_with_params(self, model: HeterodyneModel) -> None:
        """compute_correlation accepts params argument."""
        params = np.ones(14)
        c2 = model.compute_correlation(phi_angle=0.0, params=params)
        assert c2.shape == (50, 50)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_correlation_no_nan(self, model: HeterodyneModel) -> None:
        """compute_correlation doesn't produce NaN."""
        c2 = model.compute_correlation(phi_angle=45.0)
        assert not jnp.any(jnp.isnan(c2))

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_correlation_positive(self, model: HeterodyneModel) -> None:
        """compute_correlation produces positive values."""
        c2 = model.compute_correlation(phi_angle=0.0)
        assert jnp.all(c2 >= 0)


# ============================================================================
# Compute Residuals Tests
# ============================================================================


class TestHeterodyneModelComputeResiduals:
    """Tests for compute_residuals method."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_residuals_returns_flat_array(
        self, model: HeterodyneModel
    ) -> None:
        """compute_residuals returns flattened array."""
        c2_data = np.ones((50, 50))
        residuals = model.compute_residuals(c2_data, phi_angle=0.0)
        assert residuals.shape == (50 * 50,)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_residuals_zero_for_model_data(
        self, model: HeterodyneModel
    ) -> None:
        """Residuals are zero when data equals model."""
        c2_model = model.compute_correlation(phi_angle=0.0)
        residuals = model.compute_residuals(c2_model, phi_angle=0.0)
        assert_allclose(residuals, 0.0, atol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_residuals_with_weights(self, model: HeterodyneModel) -> None:
        """compute_residuals accepts weights."""
        c2_data = np.ones((50, 50))
        weights = np.ones((50, 50)) * 2.0
        residuals = model.compute_residuals(c2_data, phi_angle=0.0, weights=weights)
        assert residuals.shape == (50 * 50,)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_residuals_with_params(self, model: HeterodyneModel) -> None:
        """compute_residuals accepts params argument."""
        c2_data = np.ones((50, 50))
        params = np.ones(14)
        residuals = model.compute_residuals(c2_data, phi_angle=0.0, params=params)
        assert residuals.shape == (50 * 50,)


# ============================================================================
# Compute g1 and Fraction Tests
# ============================================================================


class TestHeterodyneModelComputeComponents:
    """Tests for g1 and fraction computation methods."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_g1_reference(self, model: HeterodyneModel) -> None:
        """compute_g1_reference returns g1 array."""
        g1_ref = model.compute_g1_reference()
        assert g1_ref.shape == (50,)
        assert jnp.all(g1_ref >= 0)
        assert jnp.all(g1_ref <= 1)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_g1_sample(self, model: HeterodyneModel) -> None:
        """compute_g1_sample returns g1 array."""
        g1_sample = model.compute_g1_sample()
        assert g1_sample.shape == (50,)
        assert jnp.all(g1_sample >= 0)
        assert jnp.all(g1_sample <= 1)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_fraction(self, model: HeterodyneModel) -> None:
        """compute_fraction returns fraction array."""
        frac = model.compute_fraction()
        assert frac.shape == (50,)
        assert jnp.all(frac >= 0)
        assert jnp.all(frac <= 1)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_components_with_params(self, model: HeterodyneModel) -> None:
        """Component methods accept params argument."""
        params = np.ones(14) * 0.5
        params[9] = 0.5  # f0 in bounds

        g1_ref = model.compute_g1_reference(params)
        g1_sample = model.compute_g1_sample(params)
        frac = model.compute_fraction(params)

        assert g1_ref.shape == (50,)
        assert g1_sample.shape == (50,)
        assert frac.shape == (50,)


# ============================================================================
# Create Residual Function Tests
# ============================================================================


class TestHeterodyneModelResidualFunction:
    """Tests for create_residual_function method."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_create_residual_function_returns_callable(
        self, model: HeterodyneModel
    ) -> None:
        """create_residual_function returns a callable."""
        c2_data = np.ones((50, 50))
        residual_fn = model.create_residual_function(c2_data, phi_angle=0.0)
        assert callable(residual_fn)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_residual_function_accepts_varying_params(
        self, model: HeterodyneModel
    ) -> None:
        """Residual function takes varying params only."""
        c2_data = np.ones((50, 50))
        residual_fn = model.create_residual_function(c2_data, phi_angle=0.0)

        # Should accept array of length n_varying
        n_varying = model.n_varying
        varying_params = jnp.ones(n_varying)
        residuals = residual_fn(varying_params)
        assert residuals.shape == (50 * 50,)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_residual_function_with_weights(self, model: HeterodyneModel) -> None:
        """Residual function handles weights."""
        c2_data = np.ones((50, 50))
        weights = np.ones((50, 50)) * 2.0
        residual_fn = model.create_residual_function(
            c2_data, phi_angle=0.0, weights=weights
        )

        n_varying = model.n_varying
        varying_params = jnp.ones(n_varying)
        residuals = residual_fn(varying_params)
        assert residuals.shape == (50 * 50,)


# ============================================================================
# Summary Tests
# ============================================================================


class TestHeterodyneModelSummary:
    """Tests for summary method."""

    @pytest.mark.unit
    def test_summary_returns_string(self, model: HeterodyneModel) -> None:
        """summary() returns a string."""
        summary = model.summary()
        assert isinstance(summary, str)

    @pytest.mark.unit
    def test_summary_contains_title(self, model: HeterodyneModel) -> None:
        """Summary contains title."""
        summary = model.summary()
        assert "HeterodyneModel Summary" in summary

    @pytest.mark.unit
    def test_summary_contains_physics(self, model: HeterodyneModel) -> None:
        """Summary contains physics info."""
        summary = model.summary()
        assert "Time points" in summary
        assert "Time step" in summary
        assert "Wavevector q" in summary

    @pytest.mark.unit
    def test_summary_contains_parameters(self, model: HeterodyneModel) -> None:
        """Summary contains parameter info."""
        summary = model.summary()
        assert "D0_ref" in summary
        assert "f0" in summary

    @pytest.mark.unit
    def test_summary_shows_vary_status(self, model: HeterodyneModel) -> None:
        """Summary shows vary/fixed status."""
        summary = model.summary()
        assert "vary" in summary
        assert "fixed" in summary


# ============================================================================
# Edge Cases
# ============================================================================


class TestHeterodyneModelEdgeCases:
    """Edge case tests for HeterodyneModel."""

    @pytest.mark.unit
    def test_uninitialized_factors_raises(self) -> None:
        """Accessing q/dt/t on uninitialized model raises."""
        model = HeterodyneModel()

        with pytest.raises(ValueError, match="Physics factors"):
            _ = model.q

        with pytest.raises(ValueError, match="Physics factors"):
            _ = model.dt

        with pytest.raises(ValueError, match="Time array"):
            _ = model.t

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_different_phi_angles(self, model: HeterodyneModel) -> None:
        """Model works with different phi angles."""
        c2_0 = model.compute_correlation(phi_angle=0.0)
        c2_90 = model.compute_correlation(phi_angle=90.0)

        assert c2_0.shape == c2_90.shape
        # Different phi angles might give different results depending on velocity
