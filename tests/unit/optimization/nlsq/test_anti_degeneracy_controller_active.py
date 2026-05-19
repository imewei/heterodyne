"""Characterization test for the active 4-layer anti-degeneracy controller.

Tests the active orchestrator ported from homodyne (Layers 1-4 only).
Layer 5 (shear-sensitivity weighting) is intentionally absent per spec §2
because heterodyne uses a velocity-phase physics model with no shear term.

These tests are "characterization" tests: they describe the *contract*
that the new active-orchestrator API must satisfy.  They run against the
ported implementation after Task 2.3 completes.

Version: 2.0.0
Author: Claude Code
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config_dict() -> dict[str, Any]:
    """Minimal valid config dict for AntiDegeneracyConfig.from_dict()."""
    return {
        "enable": True,
        "per_angle_mode": "auto",
        "fourier_order": 2,
        "fourier_auto_threshold": 8,
        "constant_scaling_threshold": 3,
        "hierarchical": {"enable": False},
        "regularization": {"enable": False},
        "gradient_monitoring": {"enable": False},
    }


@pytest.fixture
def simple_phi_angles() -> np.ndarray:
    """Four equally-spaced phi angles in radians."""
    return np.linspace(0, np.pi, 4)


# ---------------------------------------------------------------------------
# AntiDegeneracyConfig
# ---------------------------------------------------------------------------


class TestAntiDegeneracyConfig:
    """Tests for the AntiDegeneracyConfig dataclass."""

    def test_from_dict_creates_config(self, base_config_dict: dict[str, Any]) -> None:
        """from_dict() parses a config dictionary without error."""
        cfg = AntiDegeneracyConfig.from_dict(base_config_dict)
        assert isinstance(cfg, AntiDegeneracyConfig)
        assert cfg.enable is True
        assert cfg.per_angle_mode == "auto"
        assert cfg.fourier_order == 2

    def test_from_dict_empty_dict_uses_defaults(self) -> None:
        """An empty dict produces a valid config using all defaults."""
        cfg = AntiDegeneracyConfig.from_dict({})
        assert isinstance(cfg, AntiDegeneracyConfig)
        # Defaults: enable=True, per_angle_mode="auto"
        assert cfg.enable is True

    def test_from_dict_disable_all(self) -> None:
        """enable=False propagates correctly."""
        cfg = AntiDegeneracyConfig.from_dict({"enable": False})
        assert cfg.enable is False

    def test_shear_weighting_fields_absent(self) -> None:
        """Config must NOT have shear_weighting fields (Layer 5 is D3-dropped)."""
        cfg = AntiDegeneracyConfig.from_dict({})
        # The exact attribute names that homodyne has but heterodyne drops:
        for shear_attr in (
            "shear_weighting_enable",
            "shear_weighting_min_weight",
            "shear_weighting_alpha",
            "shear_weighting_update_frequency",
            "shear_weighting_normalize",
        ):
            assert not hasattr(cfg, shear_attr), (
                f"AntiDegeneracyConfig must not have {shear_attr} "
                "(Layer 5 shear fields are D3-dropped in heterodyne)"
            )


# ---------------------------------------------------------------------------
# AntiDegeneracyController construction and from_config
# ---------------------------------------------------------------------------


class TestAntiDegeneracyControllerFromConfig:
    """Tests for the from_config() classmethod (active orchestrator API)."""

    def test_from_config_returns_controller(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """from_config() returns an AntiDegeneracyController instance."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert isinstance(controller, AntiDegeneracyController)

    def test_is_enabled_when_config_enable_true(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """controller.is_enabled is True when config.enable=True and per_angle_scaling=True."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
            per_angle_scaling=True,
        )
        assert controller.is_enabled is True

    def test_is_disabled_when_config_enable_false(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """controller.is_enabled is False when config.enable=False."""
        controller = AntiDegeneracyController.from_config(
            config_dict={"enable": False},
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.is_enabled is False

    def test_is_disabled_when_per_angle_scaling_false(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """controller.is_enabled is False when per_angle_scaling=False."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
            per_angle_scaling=False,
        )
        assert controller.is_enabled is False

    def test_n_physical_stored(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """n_physical=14 (heterodyne) is stored on the controller."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.n_physical == 14

    def test_n_phi_stored(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """n_phi is stored on the controller."""
        n_phi = len(simple_phi_angles)
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=n_phi,
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.n_phi == n_phi

    def test_config_stored(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """controller.config is an AntiDegeneracyConfig instance."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert isinstance(controller.config, AntiDegeneracyConfig)


# ---------------------------------------------------------------------------
# Layer properties
# ---------------------------------------------------------------------------


class TestAntiDegeneracyControllerProperties:
    """Tests for the per-layer boolean properties."""

    def test_use_fourier_false_for_auto_mode_few_angles(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """With n_phi=4 and per_angle_mode='auto', Fourier is not active."""
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "fourier_auto_threshold": 8,  # n_phi=4 < 8, so NOT fourier
            "constant_scaling_threshold": 3,  # n_phi=4 >= 3, so auto_averaged
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.use_fourier is False

    def test_use_fourier_true_for_fourier_mode(self) -> None:
        """With per_angle_mode='fourier' and n_phi >= 3, Fourier is active."""
        n_phi = 8
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        config_dict = {
            "enable": True,
            "per_angle_mode": "fourier",
            "fourier_order": 2,
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=n_phi,
            phi_angles=phi_angles,
            n_physical=14,
        )
        assert controller.use_fourier is True

    def test_use_shear_weighting_always_false(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """use_shear_weighting must always be False (Layer 5 is D3-dropped)."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        # The property must exist but must always return False
        assert hasattr(controller, "use_shear_weighting")
        assert controller.use_shear_weighting is False

    def test_shear_methods_absent_or_noop(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """Layer 5 methods (get_shear_weights, update_shear_phi0) must either
        be absent or return None / no-op — they must NOT raise AttributeError
        if called from shared infrastructure code."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        # get_shear_weights: absent or returns None
        if hasattr(controller, "get_shear_weights"):
            result = controller.get_shear_weights()
            assert result is None

    def test_use_hierarchical_false_when_disabled(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """use_hierarchical is False when hierarchical.enable=False."""
        config_dict = {
            "enable": True,
            "hierarchical": {"enable": False},
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.use_hierarchical is False


# ---------------------------------------------------------------------------
# NLSQ callbacks (active loop integration)
# ---------------------------------------------------------------------------


class TestAntiDegeneracyControllerCallbacks:
    """Tests for create_nlsq_callbacks() active-loop integration."""

    def test_create_nlsq_callbacks_returns_dict(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """create_nlsq_callbacks() returns a dict (possibly empty if not enabled)."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        callbacks = controller.create_nlsq_callbacks()
        assert isinstance(callbacks, dict)

    def test_create_nlsq_callbacks_empty_when_disabled(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """create_nlsq_callbacks() returns {} when controller is disabled."""
        controller = AntiDegeneracyController.from_config(
            config_dict={"enable": False},
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        callbacks = controller.create_nlsq_callbacks()
        assert callbacks == {}

    def test_create_nlsq_callbacks_callable_values(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """All values in the callbacks dict are callable."""
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "constant_scaling_threshold": 3,
            "regularization": {"enable": True},
            "gradient_monitoring": {"enable": True},
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        callbacks = controller.create_nlsq_callbacks()
        for key, val in callbacks.items():
            assert callable(val), f"callback['{key}'] must be callable"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestAntiDegeneracyControllerDiagnostics:
    """Tests for get_diagnostics() method."""

    def test_get_diagnostics_returns_dict(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """get_diagnostics() returns a dict with at least 'enabled' key."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        diag = controller.get_diagnostics()
        assert isinstance(diag, dict)
        assert "enabled" in diag

    def test_get_diagnostics_enabled_matches_is_enabled(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """diag['enabled'] matches controller.is_enabled."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        diag = controller.get_diagnostics()
        assert diag["enabled"] == controller.is_enabled

    def test_reset_monitor_does_not_raise(
        self,
        base_config_dict: dict[str, Any],
        simple_phi_angles: np.ndarray,
    ) -> None:
        """reset_monitor() can be called without error."""
        controller = AntiDegeneracyController.from_config(
            config_dict=base_config_dict,
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        # Should not raise even if monitor is None
        controller.reset_monitor()


# ---------------------------------------------------------------------------
# per_angle_mode_actual selection
# ---------------------------------------------------------------------------


class TestPerAngleMode:
    """Tests for per_angle_mode_actual auto-selection logic."""

    def test_auto_mode_large_n_phi_gives_auto_averaged(self) -> None:
        """auto mode with n_phi >= constant_scaling_threshold → 'auto_averaged'."""
        n_phi = 8
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "fourier_auto_threshold": 10,  # n_phi=8 < 10 → not fourier
            "constant_scaling_threshold": 3,  # n_phi=8 >= 3 → auto_averaged
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=n_phi,
            phi_angles=phi_angles,
            n_physical=14,
        )
        assert controller.per_angle_mode_actual == "auto_averaged"

    def test_auto_mode_small_n_phi_gives_individual(self) -> None:
        """auto mode with n_phi < constant_scaling_threshold → 'individual'."""
        n_phi = 2
        phi_angles = np.linspace(0, np.pi, n_phi)
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "constant_scaling_threshold": 3,  # n_phi=2 < 3 → individual
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=n_phi,
            phi_angles=phi_angles,
            n_physical=14,
        )
        assert controller.per_angle_mode_actual == "individual"

    def test_explicit_fourier_mode(self) -> None:
        """Explicit fourier mode → per_angle_mode_actual='fourier'."""
        n_phi = 6
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        config_dict = {
            "enable": True,
            "per_angle_mode": "fourier",
            "fourier_order": 2,
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=n_phi,
            phi_angles=phi_angles,
            n_physical=14,
        )
        assert controller.per_angle_mode_actual == "fourier"
        assert controller.use_fourier is True

    def test_disabled_controller_per_angle_mode_is_disabled(
        self,
        simple_phi_angles: np.ndarray,
    ) -> None:
        """Disabled controller has per_angle_mode_actual='disabled'."""
        controller = AntiDegeneracyController.from_config(
            config_dict={"enable": False},
            n_phi=len(simple_phi_angles),
            phi_angles=simple_phi_angles,
            n_physical=14,
        )
        assert controller.per_angle_mode_actual == "disabled"
