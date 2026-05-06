"""Tests for NLSQ package integration fields on NLSQConfig."""

from __future__ import annotations

import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig


class TestNLSQConfigDefaults:
    """Verify default values for the new NLSQ integration fields."""

    def test_nlsq_config_defaults(self) -> None:
        cfg = NLSQConfig()
        assert cfg.nlsq_stability == "auto"
        assert cfg.nlsq_rescale_data is False
        assert cfg.nlsq_x_scale == "jac"
        assert cfg.nlsq_memory_fraction == 0.75
        assert cfg.nlsq_memory_fallback_gb == 16.0
        assert cfg.cmaes_diagonal_filtering == "remove"
        assert cfg.cmaes_warmstart_auto_skip is True
        assert cfg.cmaes_warmstart_skip_threshold == 5.0


class TestNLSQConfigFromDict:
    """Verify from_dict handles the new keys."""

    def test_nlsq_config_from_dict(self) -> None:
        d = {
            "nlsq_stability": "check",
            "nlsq_rescale_data": True,
            "nlsq_x_scale": "jac",
            "nlsq_memory_fraction": 0.5,
            "nlsq_memory_fallback_gb": 32.0,
        }
        cfg = NLSQConfig.from_dict(d)
        assert cfg.nlsq_stability == "check"
        assert cfg.nlsq_rescale_data is True
        assert cfg.nlsq_x_scale == "jac"
        assert cfg.nlsq_memory_fraction == 0.5
        assert cfg.nlsq_memory_fallback_gb == 32.0

    def test_nlsq_config_from_dict_missing_keys_uses_defaults(self) -> None:
        cfg = NLSQConfig.from_dict({})
        assert cfg.nlsq_stability == "auto"
        assert cfg.nlsq_rescale_data is False
        assert cfg.nlsq_memory_fraction == 0.75

    def test_nlsq_config_from_dict_x_scale_array(self) -> None:
        """nlsq_x_scale should pass through numpy arrays."""
        arr = [1.0, 2.0, 3.0]
        cfg = NLSQConfig.from_dict({"nlsq_x_scale": arr})
        assert cfg.nlsq_x_scale == arr

    def test_nlsq_config_from_dict_cmaes_warmstart_keys(self) -> None:
        cfg = NLSQConfig.from_dict(
            {
                "cmaes_diagonal_filtering": "none",
                "cmaes_warmstart_auto_skip": False,
                "cmaes_warmstart_skip_threshold": 2.5,
            }
        )

        assert cfg.cmaes_diagonal_filtering == "none"
        assert cfg.cmaes_warmstart_auto_skip is False
        assert cfg.cmaes_warmstart_skip_threshold == 2.5

    def test_nlsq_config_from_dict_nested_homodyne_style_blocks(self) -> None:
        """Homodyne-style nested anti_degeneracy/cmaes blocks should parse."""
        cfg = NLSQConfig.from_dict(
            {
                "anti_degeneracy": {
                    "per_angle_mode": "auto",
                    "fourier_order": 3,
                    "fourier_auto_threshold": 8,
                    "constant_scaling_threshold": 4,
                    "hierarchical": {
                        "enable": False,
                        "max_outer_iterations": 7,
                        "outer_tolerance": 2e-6,
                    },
                    "regularization": {
                        "mode": "relative",
                        "lambda": 2.5,
                        "target_cv": 0.2,
                    },
                    "gradient_monitoring": {
                        "enable": False,
                        "ratio_threshold": 0.03,
                        "consecutive_triggers": 9,
                    },
                },
                "cmaes": {
                    "enable": True,
                    "sigma": 0.2,
                    "max_generations": 321,
                    "popsize": 12,
                    "tol_x": 3e-5,
                    "tol_fun": 4e-7,
                    "diagonal_filtering": "remove",
                    "anti_degeneracy": True,
                    "warmstart_auto_skip": False,
                    "warmstart_skip_threshold": 1.25,
                },
            }
        )

        assert cfg.per_angle_mode == "auto"
        assert cfg.fourier_order == 3
        assert cfg.fourier_auto_threshold == 8
        assert cfg.constant_scaling_threshold == 4
        assert cfg.enable_hierarchical is False
        assert cfg.hierarchical_max_outer_iterations == 7
        assert cfg.hierarchical_outer_tolerance == pytest.approx(2e-6)
        assert cfg.regularization_mode == "relative"
        assert cfg.group_variance_lambda == pytest.approx(2.5)
        assert cfg.regularization_target_cv == pytest.approx(0.2)
        assert cfg.enable_gradient_monitoring is False
        assert cfg.gradient_ratio_threshold == pytest.approx(0.03)
        assert cfg.gradient_consecutive_triggers == 9
        assert cfg.enable_cmaes is True
        assert cfg.cmaes_sigma0 == pytest.approx(0.2)
        assert cfg.cmaes_max_iterations == 321
        assert cfg.cmaes_population_size == 12
        assert cfg.cmaes_tolx == pytest.approx(3e-5)
        assert cfg.cmaes_tolfun == pytest.approx(4e-7)
        assert cfg.cmaes_diagonal_filtering == "remove"
        assert cfg.cmaes_anti_degeneracy is True
        assert cfg.cmaes_warmstart_auto_skip is False
        assert cfg.cmaes_warmstart_skip_threshold == pytest.approx(1.25)


class TestNLSQConfigToDict:
    """Verify round-trip serialisation of the new fields."""

    def test_nlsq_config_to_dict(self) -> None:
        cfg = NLSQConfig(
            nlsq_stability="off",
            nlsq_rescale_data=True,
            nlsq_x_scale="jac",
            nlsq_memory_fraction=0.6,
            nlsq_memory_fallback_gb=8.0,
        )
        d = cfg.to_dict()
        assert d["nlsq_stability"] == "off"
        assert d["nlsq_rescale_data"] is True
        assert d["nlsq_x_scale"] == "jac"
        assert d["nlsq_memory_fraction"] == 0.6
        assert d["nlsq_memory_fallback_gb"] == 8.0

    def test_nlsq_config_round_trip(self) -> None:
        original = NLSQConfig(
            nlsq_stability="check",
            nlsq_memory_fraction=0.9,
        )
        restored = NLSQConfig.from_dict(original.to_dict())
        assert restored.nlsq_stability == original.nlsq_stability
        assert restored.nlsq_memory_fraction == original.nlsq_memory_fraction
        assert restored.nlsq_memory_fallback_gb == original.nlsq_memory_fallback_gb
        assert restored.nlsq_rescale_data == original.nlsq_rescale_data


class TestNLSQConfigValidation:
    """Validate boundary checks on the new fields."""

    @pytest.mark.parametrize("bad_stability", ["invalid", "on", ""])
    def test_nlsq_config_invalid_stability(self, bad_stability: str) -> None:
        cfg = NLSQConfig(nlsq_stability=bad_stability)
        errors = cfg.validate()
        assert any("nlsq_stability" in e for e in errors)

    @pytest.mark.parametrize("bad_frac", [0.0, -0.1, 1.1, 2.0])
    def test_nlsq_config_memory_fraction_bounds(self, bad_frac: float) -> None:
        cfg = NLSQConfig(nlsq_memory_fraction=bad_frac)
        errors = cfg.validate()
        assert any("nlsq_memory_fraction" in e for e in errors)

    def test_nlsq_config_memory_fraction_valid_edge(self) -> None:
        """Fraction of exactly 1.0 should be valid."""
        cfg = NLSQConfig(nlsq_memory_fraction=1.0)
        errors = cfg.validate()
        assert not any("nlsq_memory_fraction" in e for e in errors)

    def test_nlsq_config_memory_fallback_negative(self) -> None:
        cfg = NLSQConfig(nlsq_memory_fallback_gb=-1.0)
        errors = cfg.validate()
        assert any("nlsq_memory_fallback_gb" in e for e in errors)

    def test_nlsq_config_valid_stability_values(self) -> None:
        for val in ("auto", "check", "off"):
            cfg = NLSQConfig(nlsq_stability=val)
            errors = cfg.validate()
            assert not any("nlsq_stability" in e for e in errors)
