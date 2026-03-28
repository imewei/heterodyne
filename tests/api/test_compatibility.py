"""API compatibility tests for heterodyne public interface.

Ensures the public API surface remains stable across versions.
Tests verify that all exported symbols exist, have expected types,
and that function signatures haven't changed in breaking ways.
"""

from __future__ import annotations

import inspect

import pytest


# ============================================================================
# Public API Surface
# ============================================================================


class TestPublicAPIExports:
    """Verify all __all__ exports are importable."""

    EXPECTED_EXPORTS = [
        "__version__",
        "__version_tuple__",
        "HeterodyneModel",
        "TwoComponentModel",
        "ConfigManager",
        "load_xpcs_config",
        "ParameterManager",
        "XPCSDataLoader",
        "load_xpcs_data",
        "fit_nlsq_jax",
        "NLSQConfig",
        "NLSQResult",
        "fit_cmc_jax",
        "CMCConfig",
        "CMCResult",
        "get_device_config",
        "HAS_CORE",
        "HAS_DATA",
        "HAS_CONFIG",
        "HAS_OPTIMIZATION",
        "HAS_DEVICE",
        "HAS_VIZ",
        "HAS_CLI",
    ]

    @pytest.mark.unit
    def test_all_exports_importable(self) -> None:
        """Every name in __all__ is importable via getattr."""
        import heterodyne

        for name in self.EXPECTED_EXPORTS:
            obj = getattr(heterodyne, name, _SENTINEL)
            assert obj is not _SENTINEL, f"Missing export: {name}"

    @pytest.mark.unit
    def test_all_matches_expected(self) -> None:
        """__all__ contains exactly the expected exports."""
        import heterodyne

        assert set(heterodyne.__all__) == set(self.EXPECTED_EXPORTS)

    @pytest.mark.unit
    def test_version_is_string(self) -> None:
        """__version__ is a non-empty string."""
        import heterodyne

        assert isinstance(heterodyne.__version__, str)
        assert len(heterodyne.__version__) > 0


# ============================================================================
# Core Class Signatures
# ============================================================================


class TestCoreClassSignatures:
    """Verify core classes have expected methods and attributes."""

    @pytest.mark.unit
    def test_heterodyne_model_has_from_config(self) -> None:
        """HeterodyneModel exposes from_config classmethod."""
        from heterodyne import HeterodyneModel

        assert hasattr(HeterodyneModel, "from_config")
        assert callable(HeterodyneModel.from_config)

    @pytest.mark.unit
    def test_config_manager_has_from_yaml(self) -> None:
        """ConfigManager exposes from_yaml classmethod."""
        from heterodyne import ConfigManager

        assert hasattr(ConfigManager, "from_yaml")
        assert callable(ConfigManager.from_yaml)

    @pytest.mark.unit
    def test_config_manager_has_from_dict(self) -> None:
        """ConfigManager exposes from_dict classmethod."""
        from heterodyne import ConfigManager

        assert hasattr(ConfigManager, "from_dict")
        assert callable(ConfigManager.from_dict)

    @pytest.mark.unit
    def test_config_manager_properties(self) -> None:
        """ConfigManager exposes expected property accessors."""
        from heterodyne import ConfigManager

        expected_properties = [
            "dt",
            "start_frame",
            "end_frame",
            "time_length",
            "t_start",
            "wavevector_q",
            "phi_angles",
            "stator_rotor_gap",
        ]
        for prop in expected_properties:
            assert hasattr(ConfigManager, prop), f"Missing property: {prop}"


# ============================================================================
# Optimization Function Signatures
# ============================================================================


class TestOptimizationSignatures:
    """Verify optimization entry points have stable signatures."""

    @pytest.mark.unit
    def test_fit_nlsq_jax_callable(self) -> None:
        """fit_nlsq_jax is callable."""
        from heterodyne import fit_nlsq_jax

        assert callable(fit_nlsq_jax)

    @pytest.mark.unit
    def test_fit_cmc_jax_callable(self) -> None:
        """fit_cmc_jax is callable."""
        from heterodyne import fit_cmc_jax

        assert callable(fit_cmc_jax)

    @pytest.mark.unit
    def test_nlsq_config_is_dataclass(self) -> None:
        """NLSQConfig is a dataclass."""
        import dataclasses

        from heterodyne import NLSQConfig

        assert dataclasses.is_dataclass(NLSQConfig)

    @pytest.mark.unit
    def test_cmc_config_is_dataclass(self) -> None:
        """CMCConfig is a dataclass."""
        import dataclasses

        from heterodyne import CMCConfig

        assert dataclasses.is_dataclass(CMCConfig)

    @pytest.mark.unit
    def test_cmc_config_has_jax_profiling(self) -> None:
        """CMCConfig includes JAX profiling fields."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert hasattr(cfg, "enable_jax_profiling")
        assert hasattr(cfg, "jax_profile_dir")
        assert cfg.enable_jax_profiling is False

    @pytest.mark.unit
    def test_cmc_config_defaults_aligned(self) -> None:
        """CMCConfig defaults match homodyne parity targets."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert cfg.num_samples == 1500
        assert cfg.target_accept_prob == 0.85
        assert cfg.min_ess == 400
        assert cfg.seed == 42


# ============================================================================
# Module Availability Flags
# ============================================================================


class TestModuleAvailability:
    """Verify HAS_* flags are booleans."""

    @pytest.mark.unit
    def test_has_flags_are_booleans(self) -> None:
        """All HAS_* flags are boolean."""
        import heterodyne

        flags = ["HAS_CORE", "HAS_DATA", "HAS_CONFIG", "HAS_OPTIMIZATION",
                 "HAS_DEVICE", "HAS_VIZ", "HAS_CLI"]
        for flag in flags:
            val = getattr(heterodyne, flag)
            assert isinstance(val, bool), f"{flag} is {type(val)}, expected bool"

    @pytest.mark.unit
    def test_core_modules_available(self) -> None:
        """Core modules (CORE, DATA, CONFIG, OPTIMIZATION) are available."""
        import heterodyne

        assert heterodyne.HAS_CORE is True
        assert heterodyne.HAS_DATA is True
        assert heterodyne.HAS_CONFIG is True
        assert heterodyne.HAS_OPTIMIZATION is True


_SENTINEL = object()
