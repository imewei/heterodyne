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
    def test_per_angle_chi2_not_duplicated_from_joint(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Joint fit must not duplicate the aggregate chi2 to every per-angle result.

        Regression for the bug where _fit_joint_averaged_multi_phi copied
        joint_result.reduced_chi_squared to all per-angle NLSQResult objects,
        making all three values identical.
        """
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Three angles with intentionally different scales → different residuals
        c2_3d = np.stack(
            [
                small_c2_data,
                small_c2_data * 0.85,
                small_c2_data * 0.70,
            ]
        )
        phi_angles = [0.0, 45.0, 90.0]

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fast_nlsq_config,
        )

        assert len(results) == 3
        chi2_vals = [r.reduced_chi_squared for r in results]

        # All chi2 values must be set (not None) and finite
        assert all(v is not None for v in chi2_vals), f"Some chi2 are None: {chi2_vals}"
        assert all(
            np.isfinite(v)
            for v in chi2_vals  # type: ignore[arg-type]
        ), f"Some chi2 are non-finite: {chi2_vals}"

        # Per-angle values must differ — identical values mean the joint chi2
        # was duplicated rather than computed per angle
        assert len(set(chi2_vals)) > 1, (
            f"All per-angle chi2 are identical ({chi2_vals[0]:.6f}), "
            "which indicates the joint result was duplicated instead of "
            "computing per-angle statistics."
        )

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

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_fourier_joint_fit(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test fit_nlsq_multi_phi with Fourier joint fit (per_angle_mode='fourier')."""
        from heterodyne.optimization.nlsq.config import NLSQConfig as _NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        # Need enough angles for Fourier mode (> 2*order+1 = 5 for order=2)
        n_angles = 8
        c2_3d = np.stack([small_c2_data * (1.0 - 0.02 * i) for i in range(n_angles)])
        phi_angles = np.linspace(0, 315, n_angles)

        # Enable Fourier mode
        fourier_config = _NLSQConfig(
            max_iterations=10,
            tolerance=1e-4,
            method="trf",
            verbose=0,
            per_angle_mode="fourier",
            fourier_order=2,
        )

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=fourier_config,
        )

        assert len(results) == n_angles
        # All results should have joint Fourier metadata
        for r in results:
            assert r.metadata["optimizer"] == "joint_fourier"
            assert "fourier_coeffs" in r.metadata
            assert r.metadata["fourier_order"] == 2
            assert r.metadata["n_angles_joint"] == n_angles
            assert "contrast" in r.metadata
            assert "offset" in r.metadata

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_individual_joint_fit(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test individual mode with >1 angle uses joint fit with identity."""
        from heterodyne.optimization.nlsq.config import NLSQConfig as _NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        c2_3d = np.stack([small_c2_data, small_c2_data * 0.95])
        phi_angles = [0.0, 45.0]

        ind_config = _NLSQConfig(
            max_iterations=10,
            tolerance=1e-4,
            method="trf",
            verbose=0,
            per_angle_mode="individual",
        )

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=ind_config,
        )

        assert len(results) == 2
        # Independent mode with >1 angle still uses joint fit path
        for r in results:
            assert r.metadata["optimizer"] == "joint_fourier"

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_multi_phi_auto_mode_few_angles(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test auto mode falls back to sequential for few angles."""
        from heterodyne.optimization.nlsq.config import NLSQConfig as _NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

        c2_3d = np.stack([small_c2_data, small_c2_data * 0.9])
        phi_angles = [0.0, 90.0]

        auto_config = _NLSQConfig(
            max_iterations=10,
            tolerance=1e-4,
            method="trf",
            verbose=0,
            per_angle_mode="auto",
            fourier_auto_threshold=6,  # 2 angles < threshold
        )

        results = fit_nlsq_multi_phi(
            model=small_heterodyne_model,
            c2_data=c2_3d,
            phi_angles=phi_angles,
            config=auto_config,
        )

        assert len(results) == 2


# ============================================================================
# Test FourierReparameterizer
# ============================================================================


class TestFourierReparameterizer:
    """Tests for homodyne-parity FourierReparameterizer."""

    def test_fourier_mode_roundtrip(self) -> None:
        """Fourier coefficients -> per-angle -> coefficients roundtrip."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        assert fourier.use_fourier
        assert fourier.n_coeffs_per_param == 5  # 1 + 2*2
        assert fourier.n_coeffs == 10  # 2 * 5

        # Create smooth per-angle values
        contrast = 0.3 + 0.1 * np.cos(phi)
        offset = 1.0 + 0.05 * np.cos(2 * phi)

        # Convert to Fourier and back
        coeffs = fourier.per_angle_to_fourier(contrast, offset)
        assert coeffs.shape == (10,)

        c_out, o_out = fourier.fourier_to_per_angle(coeffs)
        np.testing.assert_allclose(c_out, contrast, atol=1e-10)
        np.testing.assert_allclose(o_out, offset, atol=1e-10)

    def test_individual_mode_passthrough(self) -> None:
        """Individual mode is identity transform."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, np.pi, 5)
        config = FourierReparamConfig(mode="individual")
        fourier = FourierReparameterizer(phi, config)

        assert not fourier.use_fourier
        assert fourier.n_coeffs == 10  # 2 * 5

        contrast = np.array([0.3, 0.35, 0.28, 0.32, 0.31])
        offset = np.array([1.0, 1.02, 0.98, 1.01, 0.99])

        coeffs = fourier.per_angle_to_fourier(contrast, offset)
        c_out, o_out = fourier.fourier_to_per_angle(coeffs)
        np.testing.assert_allclose(c_out, contrast)
        np.testing.assert_allclose(o_out, offset)

    def test_auto_mode_threshold(self) -> None:
        """Auto mode uses Fourier only above threshold."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        config = FourierReparamConfig(mode="auto", auto_threshold=6)

        # Below threshold: independent
        phi_small = np.linspace(0, 2 * np.pi, 4)
        f_small = FourierReparameterizer(phi_small, config)
        assert not f_small.use_fourier

        # Above threshold: Fourier
        phi_large = np.linspace(0, 2 * np.pi, 10)
        f_large = FourierReparameterizer(phi_large, config)
        assert f_large.use_fourier

    def test_get_bounds(self) -> None:
        """Bounds have correct shape and values."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 12)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        lower, upper = fourier.get_bounds()
        assert lower.shape == (10,)
        assert upper.shape == (10,)

        # c0 bounds = c0_bounds
        assert lower[0] == config.c0_bounds[0]
        assert upper[0] == config.c0_bounds[1]

        # Harmonic bounds = ck_bounds
        assert lower[1] == config.ck_bounds[0]

    def test_get_initial_coefficients_scalar(self) -> None:
        """Scalar init creates uniform Fourier coefficients."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        coeffs = fourier.get_initial_coefficients(0.3, 1.0)
        c_out, o_out = fourier.fourier_to_per_angle(coeffs)

        # Uniform input -> only DC component, others ~0
        np.testing.assert_allclose(c_out, 0.3, atol=1e-10)
        np.testing.assert_allclose(o_out, 1.0, atol=1e-10)

    def test_get_coefficient_labels(self) -> None:
        """Labels match expected Fourier coefficient naming."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 10)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        labels = fourier.get_coefficient_labels()
        assert labels[0] == "contrast_c0"
        assert "contrast_c1" in labels
        assert "contrast_s1" in labels
        assert "offset_c0" in labels
        assert len(labels) == 10

    def test_jacobian_transform(self) -> None:
        """Jacobian has correct shape and structure."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        config = FourierReparamConfig(mode="fourier", fourier_order=1)
        fourier = FourierReparameterizer(phi, config)

        J = fourier.get_jacobian_transform()
        # 2*n_phi rows (per-angle), n_coeffs columns
        assert J.shape == (16, 6)  # 2*8, 2*3

    def test_diagnostics(self) -> None:
        """Diagnostics dict contains expected keys."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 20)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        diag = fourier.get_diagnostics()
        assert diag["use_fourier"] is True
        assert diag["n_phi"] == 20
        assert diag["n_coeffs"] == 10
        assert diag["reduction_ratio"] < 1.0

    def test_fourier_fallback_too_few_angles(self) -> None:
        """Fourier mode falls back to independent when n_phi < min required."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.array([0.0, 1.0, 2.0])  # 3 angles < 5 (min for order=2)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)
        assert not fourier.use_fourier  # Falls back

    def test_to_from_fourier_single_group(self) -> None:
        """to_fourier/from_fourier roundtrip for single parameter group."""
        from heterodyne.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        phi = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi, config)

        values = 0.5 + 0.1 * np.cos(phi) + 0.05 * np.sin(2 * phi)
        coeffs = fourier.to_fourier(values)
        reconstructed = fourier.from_fourier(coeffs)
        np.testing.assert_allclose(reconstructed, values, atol=1e-10)

    def test_from_dict_config(self) -> None:
        """FourierReparamConfig.from_dict creates correct config."""
        from heterodyne.optimization.nlsq.fourier_reparam import FourierReparamConfig

        config = FourierReparamConfig.from_dict(
            {
                "per_angle_mode": "fourier",
                "fourier_order": 3,
                "fourier_auto_threshold": 10,
            }
        )
        assert config.mode == "fourier"
        assert config.fourier_order == 3
        assert config.auto_threshold == 10


class TestComputePerAngleChi2:
    """Unit tests for the _compute_per_angle_chi2 helper.

    Prevention: locks down the per-angle cost/chi2 computation so that
    future changes to joint-fit result-splitting cannot silently reintroduce
    the bug where all angles received the same aggregate chi2.
    """

    def _make_c2(self, n: int, noise_std: float = 0.01) -> np.ndarray:
        rng = np.random.default_rng(42)
        base = np.ones((n, n), dtype=np.float64)
        return base + rng.normal(0, noise_std, (n, n))

    def test_returns_positive_cost_and_chi2(self) -> None:
        """Cost and chi2 are strictly positive for non-zero residuals."""
        from heterodyne.optimization.nlsq.core import _compute_per_angle_chi2

        n = 20
        c2 = self._make_c2(n, noise_std=0.02)
        residuals = np.random.default_rng(0).normal(0, 0.05, (n - 1) * (n - 2))
        cost, chi2 = _compute_per_angle_chi2(residuals, c2, n_params=5)
        assert cost > 0.0
        assert chi2 > 0.0

    def test_cost_equals_half_ssr(self) -> None:
        """cost == 0.5 * sum(residuals²) by definition."""
        from heterodyne.optimization.nlsq.core import _compute_per_angle_chi2

        n = 15
        c2 = self._make_c2(n)
        residuals = np.ones((n - 1) * (n - 2)) * 0.1
        cost, _ = _compute_per_angle_chi2(residuals, c2, n_params=3)
        expected = 0.5 * float(np.sum(residuals**2))
        assert abs(cost - expected) < 1e-12

    def test_chi2_normalized_by_noise(self) -> None:
        """chi2 is SSR/(σ²_noise * DOF), not raw MSE, when noise is detectable."""
        from heterodyne.optimization.nlsq.core import _compute_per_angle_chi2

        n = 30
        noise_std = 0.05
        rng = np.random.default_rng(7)
        c2 = np.ones((n, n)) + rng.normal(0, noise_std, (n, n))
        residuals = rng.normal(0, noise_std, (n - 1) * (n - 2))

        _, chi2 = _compute_per_angle_chi2(residuals, c2, n_params=5)
        raw_mse = float(np.sum(residuals**2)) / ((n - 1) * (n - 2) - 5)

        # Normalized chi2 should differ from raw MSE when sigma2_noise ≠ 1
        assert not np.isclose(chi2, raw_mse, rtol=0.01), (
            "chi2 == raw MSE suggests the noise normalization did not apply"
        )

    def test_fallback_to_mse_when_noise_near_zero(self) -> None:
        """Falls back to MSE (no σ²_noise division) for constant c2 data."""
        from heterodyne.optimization.nlsq.core import _compute_per_angle_chi2

        n = 20
        c2 = np.ones((n, n))  # zero variance → sigma2_noise ≈ 0
        # Residuals exclude t=0 boundary AND diagonal: (n-1)*(n-2) entries
        residuals = np.full((n - 1) * (n - 2), 0.1)
        _, chi2 = _compute_per_angle_chi2(residuals, c2, n_params=3)
        n_dof = max((n - 1) * (n - 2) - 3, 1)
        expected_mse = float(np.sum(residuals**2)) / n_dof
        assert abs(chi2 - expected_mse) < 1e-10

    def test_per_angle_values_differ_for_different_data(self) -> None:
        """Different c2 matrices must produce different chi2 — not identical."""
        from heterodyne.optimization.nlsq.core import _compute_per_angle_chi2

        n = 25
        rng = np.random.default_rng(99)
        residuals = rng.normal(0, 0.05, (n - 1) * (n - 2))

        c2_a = np.ones((n, n)) + rng.normal(0, 0.02, (n, n))
        c2_b = np.ones((n, n)) + rng.normal(0, 0.08, (n, n))  # higher noise

        _, chi2_a = _compute_per_angle_chi2(residuals, c2_a, n_params=5)
        _, chi2_b = _compute_per_angle_chi2(residuals, c2_b, n_params=5)
        assert not np.isclose(chi2_a, chi2_b, rtol=1e-6), (
            "Different noise levels must produce different chi2 values"
        )
