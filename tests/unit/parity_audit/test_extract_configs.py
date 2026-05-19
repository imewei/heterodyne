"""Tests for the config-key extractor."""

from __future__ import annotations

from pathlib import Path

from tools.parity_audit.extract_configs import extract_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_config.py"


def test_dataclass_fields_yield_config_keys() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "fixtures.sample_config.CMCConfig.target_accept" in keys
    assert "fixtures.sample_config.CMCConfig.max_r_hat" in keys


def test_config_get_calls_captured() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "optimization.cmc.dense_mass" in keys["fixtures.sample_config.runtime_keys"]


def test_config_subscript_captured() -> None:
    keys = extract_file(FIXTURE, module_path="fixtures.sample_config")
    assert "optimization.nlsq.tolerance" in keys["fixtures.sample_config.runtime_keys"]
