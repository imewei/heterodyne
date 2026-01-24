"""Additional tests for nlsq/core.py to improve coverage.

Covers:
- fit_nlsq_jax with default config (line 45)
- fit_nlsq_jax with multistart (lines 121-132, 158-169)
- fit_nlsq_jax ImportError fallback (lines 145-148)
- fit_nlsq_multi_phi function (lines 223-251)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from heterodyne import HeterodyneModel, NLSQConfig


# ============================================================================
# Test fit_nlsq_jax with default config
# ============================================================================


class TestFitNLSQJaxDefaultConfig:
    """Tests for fit_nlsq_jax when config is None."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_with_none_config(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test fit_nlsq_jax creates default config when None passed."""
        from heterodyne import fit_nlsq_jax

        # Pass config=None to trigger default config creation (line 45)
        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=None,  # Triggers default config creation
            use_nlsq_library=False,
        )

        assert result is not None
        assert hasattr(result, "parameters")
        assert len(result.parameters) == small_heterodyne_model.n_varying


# ============================================================================
# Test fit_nlsq_jax with multistart
# ============================================================================


class TestFitNLSQJaxMultistart:
    """Tests for multistart optimization paths."""

    @pytest.fixture
    def multistart_config(self) -> NLSQConfig:
        """Create config with multistart enabled."""
        from heterodyne import NLSQConfig

        return NLSQConfig(
            max_iterations=10,
            tolerance=1e-4,
            method="trf",
            multistart=True,
            multistart_n=2,  # Small for speed
            verbose=0,
        )

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_multistart_with_scipy(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        multistart_config: NLSQConfig,
    ) -> None:
        """Test multistart optimization using scipy fallback (lines 158-169)."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=multistart_config,
            use_nlsq_library=False,  # Force scipy path
        )

        assert result is not None
        assert "multistart" in result.metadata
        assert result.metadata["multistart"]["n_starts"] == 2

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_multistart_with_nlsq(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        multistart_config: NLSQConfig,
    ) -> None:
        """Test multistart optimization using nlsq library (lines 121-132)."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=multistart_config,
            use_nlsq_library=True,  # Use nlsq path
        )

        assert result is not None
        assert "multistart" in result.metadata
        assert result.metadata["multistart"]["n_starts"] == 2
        assert result.metadata["multistart"]["n_successful"] >= 0


# ============================================================================
# Test fit_nlsq_jax ImportError fallback
# ============================================================================


class TestFitNLSQJaxImportErrorFallback:
    """Tests for nlsq import error fallback to scipy."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_falls_back_on_import_error(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test that fit_nlsq_jax falls back to scipy when nlsq import fails (lines 145-148)."""
        from heterodyne.optimization.nlsq import core

        # Mock NLSQAdapter to raise ImportError
        original_nlsq_adapter = core.NLSQAdapter

        def mock_adapter_init(*args, **kwargs):
            raise ImportError("nlsq not available")

        with patch.object(core, "NLSQAdapter", side_effect=mock_adapter_init):
            result = core.fit_nlsq_jax(
                model=small_heterodyne_model,
                c2_data=small_c2_data,
                phi_angle=0.0,
                config=fast_nlsq_config,
                use_nlsq_library=True,  # Try nlsq, should fall back
            )

        assert result is not None
        assert len(result.parameters) == small_heterodyne_model.n_varying


# ============================================================================
# Test fit_nlsq_multi_phi
# ============================================================================


class TestFitNLSQMultiPhi:
    """Tests for fit_nlsq_multi_phi function (lines 223-251)."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_single_angle(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi with single angle."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Single angle case - data needs to be 3D
        c2_3d = small_c2_data[np.newaxis, ...]  # Shape: (1, N, N)
        phi_angles = [0.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
        )

        assert len(results) == 1
        assert results[0].metadata["phi_angle"] == 0.0

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_multiple_angles(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi with multiple angles."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Create data for 3 angles
        c2_3d = np.stack([small_c2_data, small_c2_data * 0.9, small_c2_data * 0.8])
        phi_angles = [0.0, 45.0, 90.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
        )

        assert len(results) == 3
        for i, phi in enumerate(phi_angles):
            assert results[i].metadata["phi_angle"] == phi

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_2d_data_expansion(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi expands 2D data for single angle (line 225-226)."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Pass 2D data (should be auto-expanded to 3D)
        phi_angles = [0.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=small_c2_data,  # 2D, not 3D
            phi_angles=phi_angles,
            config=fast_nlsq_config,
        )

        assert len(results) == 1

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_mismatched_shapes_raises(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi raises on shape mismatch (lines 228-232)."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Create data for 2 angles but provide 3 angles
        c2_3d = np.stack([small_c2_data, small_c2_data])  # 2 matrices
        phi_angles = [0.0, 45.0, 90.0]  # 3 angles - mismatch!

        with pytest.raises(ValueError, match="doesn't match"):
            fit_nlsq_multi_phi(
                model=small_heterodyne_model,
                c2_data=c2_3d,
                phi_angles=phi_angles,
                config=fast_nlsq_config,
            )

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_with_weights(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi with 3D weights (line 239)."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Create data and weights for 2 angles
        c2_3d = np.stack([small_c2_data, small_c2_data * 0.9])
        weights_3d = np.ones_like(c2_3d)  # 3D weights
        phi_angles = [0.0, 45.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
            weights=weights_3d,
        )

        assert len(results) == 2

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_with_2d_weights(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi with shared 2D weights."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Create data for 2 angles with shared 2D weights
        c2_3d = np.stack([small_c2_data, small_c2_data * 0.9])
        weights_2d = np.ones_like(small_c2_data)  # 2D weights (shared)
        phi_angles = [0.0, 45.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
            weights=weights_2d,
        )

        assert len(results) == 2

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_numpy_phi_angles(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_multi_phi accepts numpy array for phi_angles (line 223)."""
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        c2_3d = np.stack([small_c2_data, small_c2_data])
        phi_angles = np.array([0.0, 90.0])  # numpy array, not list

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
        )

        assert len(results) == 2
