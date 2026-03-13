"""Tests for create_model() factory and ReducedModel analysis modes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.core.models import (
    ANALYSIS_MODES,
    ReducedModel,
    TwoComponentModel,
    create_model,
)


class TestTwoComponentMode:
    """create_model('two_component') returns the full 14-param model."""

    def test_two_component_mode(self) -> None:
        model = create_model("two_component")
        assert isinstance(model, TwoComponentModel)
        assert model.n_params == 14

    def test_two_component_param_names(self) -> None:
        model = create_model("two_component")
        from heterodyne.config.parameter_names import ALL_PARAM_NAMES

        assert model.param_names == ALL_PARAM_NAMES


class TestStaticRefMode:
    """create_model('static_ref') returns a 3-param reference-only model."""

    def test_static_ref_mode(self) -> None:
        model = create_model("static_ref")
        assert isinstance(model, ReducedModel)
        assert model.n_params == 3

    def test_static_ref_param_names(self) -> None:
        model = create_model("static_ref")
        assert model.param_names == ("D0_ref", "alpha_ref", "D_offset_ref")

    def test_static_ref_defaults(self) -> None:
        model = create_model("static_ref")
        defaults = model.get_default_params()
        assert defaults.shape == (3,)
        # D0_ref=1e4, alpha_ref=0.0, D_offset_ref=0.0
        np.testing.assert_allclose(defaults, [1e4, 0.0, 0.0])


class TestStaticBothMode:
    """create_model('static_both') returns a 6-param model."""

    def test_static_both_mode(self) -> None:
        model = create_model("static_both")
        assert isinstance(model, ReducedModel)
        assert model.n_params == 6

    def test_static_both_param_names(self) -> None:
        model = create_model("static_both")
        assert model.param_names == (
            "D0_ref",
            "alpha_ref",
            "D_offset_ref",
            "D0_sample",
            "alpha_sample",
            "D_offset_sample",
        )

    def test_static_both_compute(self) -> None:
        """Integration test: compute_correlation runs without error."""
        model = create_model("static_both")
        params = jnp.array(model.get_default_params())
        t = jnp.linspace(0.0, 1.0, 8)
        result = model.compute_correlation(
            params,
            t=t,
            q=0.01,
            dt=0.1,
            phi_angle=0.0,
            contrast=1.0,
            offset=1.0,
        )
        # Should produce an (N, N) matrix with finite values
        assert result.shape == (8, 8)
        assert jnp.all(jnp.isfinite(result))


class TestInvalidMode:
    """Invalid modes raise ValueError with a descriptive message."""

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown analysis mode"):
            create_model("nonexistent_mode")

    def test_invalid_mode_message_contains_valid_modes(self) -> None:
        with pytest.raises(ValueError, match="static_ref"):
            create_model("bad_mode")


class TestAnalysisModesRegistry:
    """ANALYSIS_MODES registry has the expected structure."""

    def test_registry_keys(self) -> None:
        assert set(ANALYSIS_MODES.keys()) == {
            "static_ref",
            "static_both",
            "two_component",
        }

    def test_static_ref_length(self) -> None:
        assert len(ANALYSIS_MODES["static_ref"]) == 3

    def test_static_both_length(self) -> None:
        assert len(ANALYSIS_MODES["static_both"]) == 6

    def test_two_component_length(self) -> None:
        assert len(ANALYSIS_MODES["two_component"]) == 14


class TestReducedModelValidation:
    """ReducedModel rejects unknown parameter names."""

    def test_invalid_param_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter names"):
            ReducedModel(_active_params=("D0_ref", "not_a_param"))

    def test_valid_single_param(self) -> None:
        model = ReducedModel(_active_params=("f0",))
        assert model.n_params == 1
        assert model.param_names == ("f0",)

    def test_defaults_single_param(self) -> None:
        model = ReducedModel(_active_params=("f0",))
        defaults = model.get_default_params()
        np.testing.assert_allclose(defaults, [0.5])
