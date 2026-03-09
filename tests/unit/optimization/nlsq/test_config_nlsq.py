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
