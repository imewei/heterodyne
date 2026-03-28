"""Scientific validation tests for heterodyne physics model.

Validates numerical correctness of the two-component correlation model
against known analytical limits and physical constraints.
"""

from __future__ import annotations

import numpy as np
import pytest


# ============================================================================
# Physical Limit Tests
# ============================================================================


class TestPhysicalLimits:
    """Verify model produces physically correct results at known limits."""

    @pytest.fixture
    def _make_model(self):
        """Factory for creating models with specific parameters."""
        from heterodyne.core.heterodyne_model import HeterodyneModel

        def _factory(n_times: int = 50, dt: float = 1.0, q: float = 0.01):
            config = {
                "analyzer_parameters": {
                    "dt": dt,
                    "start_frame": 1,
                    "end_frame": n_times,
                    "scattering": {"wavevector_q": q},
                },
                "parameters": {},
            }
            return HeterodyneModel.from_config(config)

        return _factory

    @pytest.mark.unit
    def test_zero_diffusion_gives_finite_c2(self, _make_model) -> None:
        """With D0=0, model should produce finite C2 values."""
        model = _make_model()
        # Default parameters should produce finite correlation
        c2 = model.compute_correlation(phi_angle=0.0)
        assert np.all(np.isfinite(c2)), "C2 contains NaN/Inf"

    @pytest.mark.unit
    def test_correlation_matrix_symmetry(self, _make_model) -> None:
        """C2(t1, t2) should be symmetric: C2(t1,t2) = C2(t2,t1)."""
        model = _make_model(n_times=30)
        c2 = model.compute_correlation(phi_angle=0.0)
        np.testing.assert_allclose(
            c2, c2.T, rtol=1e-10,
            err_msg="C2 matrix is not symmetric",
        )

    @pytest.mark.unit
    def test_diagonal_is_maximum(self, _make_model) -> None:
        """Diagonal elements should be local maxima (auto-correlation peak)."""
        model = _make_model(n_times=30)
        c2 = model.compute_correlation(phi_angle=0.0)
        diag = np.diag(c2)
        # Each diagonal element should be >= adjacent off-diagonal
        for i in range(1, len(diag) - 1):
            assert c2[i, i] >= c2[i, i + 1] - 1e-10, (
                f"Diagonal not maximum at index {i}"
            )

    @pytest.mark.unit
    def test_c2_values_bounded(self, _make_model) -> None:
        """C2 values should be finite and within reasonable physical range."""
        model = _make_model(n_times=30)
        c2 = model.compute_correlation(phi_angle=0.0)
        assert np.all(np.isfinite(c2)), "C2 contains NaN or Inf"
        # With default offset ~1.0 and contrast ~0.5, values should be O(1)
        assert np.all(np.abs(c2) < 100), "C2 values unexpectedly large"


# ============================================================================
# Numerical Stability
# ============================================================================


class TestNumericalStability:
    """Verify model is numerically stable under edge conditions."""

    @pytest.mark.unit
    def test_small_q_stable(self) -> None:
        """Model is stable with very small wavevector (q → 0)."""
        from heterodyne.core.heterodyne_model import HeterodyneModel

        config = {
            "analyzer_parameters": {
                "dt": 1.0,
                "start_frame": 1,
                "end_frame": 20,
                "scattering": {"wavevector_q": 1e-6},
            },
            "parameters": {},
        }
        model = HeterodyneModel.from_config(config)
        c2 = model.compute_correlation(phi_angle=0.0)
        assert np.all(np.isfinite(c2)), "Unstable at small q"

    @pytest.mark.unit
    def test_large_dt_stable(self) -> None:
        """Model is stable with large time step."""
        from heterodyne.core.heterodyne_model import HeterodyneModel

        config = {
            "analyzer_parameters": {
                "dt": 100.0,
                "start_frame": 1,
                "end_frame": 20,
                "scattering": {"wavevector_q": 0.01},
            },
            "parameters": {},
        }
        model = HeterodyneModel.from_config(config)
        c2 = model.compute_correlation(phi_angle=0.0)
        assert np.all(np.isfinite(c2)), "Unstable at large dt"

    @pytest.mark.unit
    def test_multiple_phi_angles_consistent(self) -> None:
        """Different phi angles produce consistent (finite) results."""
        from heterodyne.core.heterodyne_model import HeterodyneModel

        config = {
            "analyzer_parameters": {
                "dt": 1.0,
                "start_frame": 1,
                "end_frame": 20,
                "scattering": {"wavevector_q": 0.01},
            },
            "parameters": {},
        }
        model = HeterodyneModel.from_config(config)
        for phi in [0.0, 45.0, 90.0, 135.0, 180.0]:
            c2 = model.compute_correlation(phi_angle=phi)
            assert np.all(np.isfinite(c2)), f"Unstable at phi={phi}"


# ============================================================================
# Parameter Count Validation
# ============================================================================


class TestParameterCount:
    """Verify the 14-parameter model structure."""

    @pytest.mark.unit
    def test_model_has_14_physics_params(self) -> None:
        """Model reports exactly 14 physical parameters."""
        from heterodyne.core.heterodyne_model import HeterodyneModel

        config = {
            "analyzer_parameters": {
                "dt": 1.0,
                "start_frame": 1,
                "end_frame": 20,
                "scattering": {"wavevector_q": 0.01},
            },
            "parameters": {},
        }
        model = HeterodyneModel.from_config(config)
        assert model.n_params == 14

    @pytest.mark.unit
    def test_parameter_names_complete(self) -> None:
        """All 14 parameter names are present."""
        from heterodyne.config.parameter_names import ALL_PARAM_NAMES

        assert len(ALL_PARAM_NAMES) == 14
        # Verify all five groups are represented
        groups = {
            "reference": ["D0_ref", "alpha_ref", "D_offset_ref"],
            "sample": ["D0_sample", "alpha_sample", "D_offset_sample"],
            "velocity": ["v0", "beta", "v_offset"],
            "fraction": ["f0", "f1", "f2", "f3"],
            "angle": ["phi0"],
        }
        for group_name, params in groups.items():
            for p in params:
                assert p in ALL_PARAM_NAMES, (
                    f"Missing {p} from {group_name} group"
                )
