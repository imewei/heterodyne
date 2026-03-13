"""Basic tests for heterodyne core functionality."""

import numpy as np
import pytest


class TestHeterodyneModel:
    """Tests for HeterodyneModel class."""

    def test_model_creation(self) -> None:
        """Test model can be created from config."""
        from heterodyne import HeterodyneModel

        config = {
            "temporal": {"dt": 1.0, "time_length": 100},
            "scattering": {"wavevector_q": 0.01},
            "parameters": {},
        }

        model = HeterodyneModel.from_config(config)

        assert model.n_params == 14
        assert model.n_times == 100
        assert model.q == 0.01
        assert model.dt == 1.0

    def test_compute_correlation(self) -> None:
        """Test correlation computation."""
        from heterodyne import HeterodyneModel

        config = {
            "temporal": {"dt": 1.0, "time_length": 50},
            "scattering": {"wavevector_q": 0.01},
            "parameters": {},
        }

        model = HeterodyneModel.from_config(config)

        # Compute correlation
        c2 = model.compute_correlation(phi_angle=0.0)

        # Check shape
        assert c2.shape == (50, 50)

        # Correlation should be positive
        assert np.all(c2 >= 0)

        # Diagonal should be close to offset + contrast = 1.5
        # (offset=1.0, contrast=0.5 from PerAngleScaling default)
        diag = np.diag(np.asarray(c2))
        assert np.allclose(diag[0], 1.5, rtol=0.1)


class TestJaxBackend:
    """Tests for JAX backend functions."""

    def test_compute_c2_heterodyne(self) -> None:
        """Test core c2 computation."""
        import jax.numpy as jnp

        from heterodyne.core.jax_backend import compute_c2_heterodyne

        # Default parameters
        params = jnp.array(
            [
                1.0,
                1.0,
                0.0,  # D0_ref, alpha_ref, D_offset_ref
                1.0,
                1.0,
                0.0,  # D0_sample, alpha_sample, D_offset_sample
                0.0,
                0.0,
                0.0,  # v0, beta, v_offset
                0.5,
                0.0,
                0.0,
                0.0,  # f0, f1, f2, f3
                0.0,  # phi0
            ]
        )

        t = jnp.arange(20) * 1.0
        q = 0.01
        dt = 1.0
        phi = 0.0

        c2 = compute_c2_heterodyne(params, t, q, dt, phi)

        assert c2.shape == (20, 20)
        assert not np.any(np.isnan(c2))
        assert not np.any(np.isinf(c2))

    def test_g1_transport(self) -> None:
        """Test g1 correlation computation."""
        import jax.numpy as jnp

        from heterodyne.core.jax_backend import compute_g1_transport

        J = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        q = 0.1

        g1 = compute_g1_transport(J, q)

        # g1 should decay from 1
        assert np.isclose(g1[0], 1.0)
        assert np.all(g1[1:] <= 1.0)
        assert np.all(np.diff(g1) <= 0)  # Monotonically decreasing


class TestParameterManager:
    """Tests for parameter management."""

    def test_parameter_names(self) -> None:
        """Test parameter name constants."""
        from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_GROUPS

        assert len(ALL_PARAM_NAMES) == 14

        # Check all groups sum to 16 (14 physics + 2 scaling)
        total = sum(len(names) for names in PARAM_GROUPS.values())
        assert total == 16

    def test_parameter_space(self) -> None:
        """Test ParameterSpace defaults."""
        from heterodyne.config.parameter_space import ParameterSpace

        space = ParameterSpace()

        assert space.n_total == 14  # Physics params only
        assert len(space.values) == 16  # 14 physics + 2 scaling
        assert len(space.bounds) == 16

        # Check defaults are within bounds
        for name in space.values:
            val = space.values[name]
            low, high = space.bounds[name]
            assert low <= val <= high


class TestNLSQConfig:
    """Tests for NLSQ configuration."""

    def test_default_config(self) -> None:
        """Test default NLSQ configuration."""
        from heterodyne.optimization.nlsq.config import NLSQConfig

        config = NLSQConfig()

        assert config.max_iterations == 1000
        assert config.tolerance == 1e-8
        assert config.method == "trf"

    def test_config_from_dict(self) -> None:
        """Test creating config from dictionary."""
        from heterodyne.optimization.nlsq.config import NLSQConfig

        config = NLSQConfig.from_dict(
            {
                "max_iterations": 50,
                "tolerance": 1e-6,
                "multistart": True,
            }
        )

        assert config.max_iterations == 50
        assert config.tolerance == 1e-6
        assert config.multistart is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
