"""Tests for CMC test factory fixtures.

Provides reusable factory functions that create pre-configured CMC test objects
(CMCConfig, NLSQResult, ReparamConfig, synthetic c2 data, HeterodyneModel mock)
and validates that each factory produces correct, self-consistent objects.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.config.parameter_names import ALL_PARAM_NAMES

# ---------------------------------------------------------------------------
# Factory functions (importable by other test modules)
# ---------------------------------------------------------------------------


def make_cmc_config(**overrides: Any):
    """Return a :class:`CMCConfig` with minimal CI-friendly defaults.

    Defaults are intentionally tiny (num_warmup=10, num_samples=20,
    num_chains=1) so that tests importing this factory run as fast as
    possible.  Checkpoints and adaptive sampling are disabled for
    determinism.

    Args:
        **overrides: Any ``CMCConfig`` field can be overridden by name.

    Returns:
        A configured ``CMCConfig`` instance.
    """
    from heterodyne.optimization.cmc.config import CMCConfig

    defaults: dict[str, Any] = {
        "num_warmup": 10,
        "num_samples": 20,
        "num_chains": 1,
        "seed": 42,
        "enable_checkpoints": False,
        "adaptive_sampling": False,
        "chain_method": "sequential",
    }
    defaults.update(overrides)
    return CMCConfig(**defaults)


def make_nlsq_result(
    n_params: int = 14,
    success: bool = True,
    **overrides: Any,
):
    """Return an :class:`NLSQResult` with realistic synthetic values.

    Parameters are populated with registry defaults, and uncertainties are
    set to 10% of each parameter value (floored at 1e-6).

    Args:
        n_params: Number of parameters (default 14).
        success: Whether the result reports convergence.
        **overrides: Any ``NLSQResult`` field can be overridden by name.

    Returns:
        A configured ``NLSQResult`` instance.
    """
    from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
    from heterodyne.optimization.nlsq.results import NLSQResult

    # Use registry defaults for the first n_params physics parameters
    names = list(ALL_PARAM_NAMES[:n_params])
    values = np.array(
        [DEFAULT_REGISTRY[n].default for n in names], dtype=np.float64
    )
    uncertainties = np.maximum(np.abs(values) * 0.1, 1e-6)

    defaults: dict[str, Any] = {
        "parameters": values,
        "parameter_names": names,
        "success": success,
        "message": "Converged" if success else "Failed to converge",
        "uncertainties": uncertainties,
        "covariance": np.diag(uncertainties**2),
        "final_cost": 1.23e-4,
        "reduced_chi_squared": 1.05,
        "n_iterations": 15,
        "n_function_evals": 42,
        "wall_time_seconds": 0.5,
    }
    defaults.update(overrides)
    return NLSQResult(**defaults)


def make_reparam_config(**overrides: Any):
    """Return a :class:`ReparamConfig` with sensible defaults.

    All three power-law pairs are enabled by default with ``t_ref=1.0``.

    Args:
        **overrides: Any ``ReparamConfig`` field can be overridden by name.

    Returns:
        A configured ``ReparamConfig`` instance.
    """
    from heterodyne.optimization.cmc.reparameterization import ReparamConfig

    defaults: dict[str, Any] = {
        "enable_d_ref": True,
        "enable_d_sample": True,
        "enable_v_ref": True,
        "t_ref": 1.0,
    }
    defaults.update(overrides)
    return ReparamConfig(**defaults)


def make_synthetic_c2(n_times: int = 16, seed: int = 42) -> np.ndarray:
    """Return a synthetic two-time correlation matrix.

    The matrix is symmetric positive-definite with realistic structure:
    diagonal values near 1.0 and off-diagonal elements decaying with
    distance from the diagonal.

    Args:
        n_times: Number of time points (output shape is ``(n_times, n_times)``).
        seed: NumPy random seed for reproducibility.

    Returns:
        2-D array of shape ``(n_times, n_times)``.
    """
    rng = np.random.default_rng(seed)

    # Exponential decay kernel + small noise → symmetric positive-definite
    idx = np.arange(n_times, dtype=np.float64)
    dist = np.abs(idx[:, None] - idx[None, :])
    c2 = np.exp(-dist / max(n_times / 4.0, 1.0))
    # Add small positive noise to keep finite & break exact symmetry before re-symmetrising
    noise = rng.uniform(0.0, 0.01, size=(n_times, n_times))
    c2 = c2 + noise
    # Enforce exact symmetry
    c2 = (c2 + c2.T) / 2.0
    return c2


def make_model_mock(n_params: int = 14) -> MagicMock:
    """Return a :class:`~unittest.mock.MagicMock` mimicking :class:`HeterodyneModel`.

    The mock exposes ``param_manager`` with working ``varying_names``,
    ``get_initial_values()``, and ``get_bounds()`` methods that return
    arrays of the correct length.

    Args:
        n_params: Number of varying parameters.

    Returns:
        A ``MagicMock`` with a ``param_manager`` sub-mock.
    """
    from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

    names = list(ALL_PARAM_NAMES[:n_params])
    values = np.array(
        [DEFAULT_REGISTRY[n].default for n in names], dtype=np.float64
    )
    lower = np.array(
        [DEFAULT_REGISTRY[n].min_bound for n in names], dtype=np.float64
    )
    upper = np.array(
        [DEFAULT_REGISTRY[n].max_bound for n in names], dtype=np.float64
    )

    param_manager = MagicMock()
    param_manager.varying_names = names
    param_manager.n_varying = n_params
    param_manager.get_initial_values.return_value = values
    param_manager.get_bounds.return_value = (lower, upper)
    param_manager.get_full_values.return_value = values

    model = MagicMock()
    model.param_manager = param_manager
    model.n_params = 14
    model.n_varying = n_params
    model.varying_names = names

    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCMCConfigFactory:
    """Validate :func:`make_cmc_config` factory."""

    def test_default_config_valid(self) -> None:
        """Factory output passes CMCConfig validation (no errors on construction)."""
        cfg = make_cmc_config()
        # CMCConfig.__post_init__ runs validation; reaching here means it passed.
        assert cfg is not None

    def test_override_num_chains(self) -> None:
        """Keyword overrides are applied correctly."""
        cfg = make_cmc_config(num_chains=4)
        assert cfg.num_chains == 4

    def test_ci_friendly_defaults(self) -> None:
        """Default warmup/samples are small enough for CI."""
        cfg = make_cmc_config()
        assert cfg.num_warmup == 10
        assert cfg.num_samples == 20
        assert cfg.num_chains == 1

    def test_checkpoints_disabled(self) -> None:
        """Checkpoints are disabled by default for test determinism."""
        cfg = make_cmc_config()
        assert cfg.enable_checkpoints is False

    def test_adaptive_sampling_disabled(self) -> None:
        """Adaptive sampling is disabled by default for test determinism."""
        cfg = make_cmc_config()
        assert cfg.adaptive_sampling is False

    def test_seed_deterministic(self) -> None:
        """Default seed is set for reproducibility."""
        cfg = make_cmc_config()
        assert cfg.seed == 42


@pytest.mark.unit
class TestNLSQResultFactory:
    """Validate :func:`make_nlsq_result` factory."""

    def test_default_result_success(self) -> None:
        """Default result reports success=True."""
        result = make_nlsq_result()
        assert result.success is True

    def test_parameter_count(self) -> None:
        """Result contains the requested number of parameters."""
        result = make_nlsq_result(n_params=14)
        assert result.n_params == 14
        assert len(result.parameter_names) == 14
        assert result.parameters.shape == (14,)

    def test_failure_result(self) -> None:
        """success=False produces a valid (non-converged) result."""
        result = make_nlsq_result(success=False)
        assert result.success is False
        assert "Failed" in result.message

    def test_result_has_parameter_names(self) -> None:
        """parameter_names is populated with canonical physics names."""
        result = make_nlsq_result()
        assert isinstance(result.parameter_names, list)
        assert all(isinstance(n, str) for n in result.parameter_names)
        # First name should be D0_ref per canonical order
        assert result.parameter_names[0] == "D0_ref"

    def test_uncertainties_shape(self) -> None:
        """Uncertainties array matches parameter count."""
        result = make_nlsq_result(n_params=14)
        assert result.uncertainties is not None
        assert result.uncertainties.shape == (14,)

    def test_covariance_shape(self) -> None:
        """Covariance matrix is square with correct dimensions."""
        result = make_nlsq_result(n_params=14)
        assert result.covariance is not None
        assert result.covariance.shape == (14, 14)

    def test_params_dict_property(self) -> None:
        """params_dict property returns all parameter names."""
        result = make_nlsq_result()
        d = result.params_dict
        assert isinstance(d, dict)
        assert len(d) == 14

    def test_custom_n_params(self) -> None:
        """Factory respects custom n_params values."""
        for n in (3, 7, 14):
            result = make_nlsq_result(n_params=n)
            assert result.n_params == n


@pytest.mark.unit
class TestReparamConfigFactory:
    """Validate :func:`make_reparam_config` factory."""

    def test_default_config(self) -> None:
        """Factory produces a valid ReparamConfig with all pairs enabled."""
        cfg = make_reparam_config()
        assert cfg.enable_d_ref is True
        assert cfg.enable_d_sample is True
        assert cfg.enable_v_ref is True
        assert cfg.t_ref == 1.0

    def test_override_flags(self) -> None:
        """Can override individual reparameterization flags."""
        cfg = make_reparam_config(enable_d_ref=False, t_ref=10.0)
        assert cfg.enable_d_ref is False
        assert cfg.enable_d_sample is True
        assert cfg.t_ref == 10.0

    def test_enabled_pairs_all(self) -> None:
        """All three power-law pairs are reported when fully enabled."""
        cfg = make_reparam_config()
        pairs = cfg.enabled_pairs
        assert len(pairs) == 3

    def test_enabled_pairs_subset(self) -> None:
        """Disabling a flag removes that pair from enabled_pairs."""
        cfg = make_reparam_config(enable_v_ref=False)
        pairs = cfg.enabled_pairs
        assert len(pairs) == 2
        pair_names = [p[0] for p in pairs]
        assert "v0" not in pair_names

    def test_frozen(self) -> None:
        """ReparamConfig is frozen (immutable)."""
        cfg = make_reparam_config()
        with pytest.raises(AttributeError):
            cfg.t_ref = 99.0  # type: ignore[misc]


@pytest.mark.unit
class TestSyntheticDataFactory:
    """Validate :func:`make_synthetic_c2` factory."""

    def test_shape(self) -> None:
        """Output shape is (n_times, n_times)."""
        c2 = make_synthetic_c2(n_times=16)
        assert c2.shape == (16, 16)

    def test_reproducible(self) -> None:
        """Same seed produces identical data."""
        a = make_synthetic_c2(n_times=8, seed=123)
        b = make_synthetic_c2(n_times=8, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seed(self) -> None:
        """Different seeds produce different data."""
        a = make_synthetic_c2(n_times=8, seed=1)
        b = make_synthetic_c2(n_times=8, seed=2)
        assert not np.array_equal(a, b)

    def test_finite(self) -> None:
        """No NaN or inf values in the output."""
        c2 = make_synthetic_c2(n_times=32)
        assert np.all(np.isfinite(c2))

    def test_positive_diagonal(self) -> None:
        """Diagonal elements are strictly positive."""
        c2 = make_synthetic_c2(n_times=16)
        diag = np.diag(c2)
        assert np.all(diag > 0)

    def test_symmetric(self) -> None:
        """Output matrix is symmetric."""
        c2 = make_synthetic_c2(n_times=16)
        np.testing.assert_array_equal(c2, c2.T)

    def test_custom_size(self) -> None:
        """Factory respects custom n_times."""
        for n in (4, 32, 64):
            c2 = make_synthetic_c2(n_times=n)
            assert c2.shape == (n, n)


@pytest.mark.unit
class TestModelMockFactory:
    """Validate :func:`make_model_mock` factory."""

    def test_has_param_manager(self) -> None:
        """Mock has a param_manager attribute."""
        model = make_model_mock()
        assert hasattr(model, "param_manager")
        assert model.param_manager is not None

    def test_varying_names(self) -> None:
        """param_manager.varying_names returns a list of strings."""
        model = make_model_mock(n_params=14)
        names = model.param_manager.varying_names
        assert isinstance(names, list)
        assert len(names) == 14
        assert all(isinstance(n, str) for n in names)

    def test_varying_names_canonical_order(self) -> None:
        """varying_names follow canonical parameter order."""
        model = make_model_mock()
        names = model.param_manager.varying_names
        assert names == list(ALL_PARAM_NAMES)

    def test_get_initial_values(self) -> None:
        """get_initial_values returns array of correct length."""
        model = make_model_mock(n_params=14)
        values = model.param_manager.get_initial_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (14,)
        assert np.all(np.isfinite(values))

    def test_get_bounds(self) -> None:
        """get_bounds returns (lower, upper) tuple of correct shape."""
        model = make_model_mock(n_params=14)
        lower, upper = model.param_manager.get_bounds()
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (14,)
        assert upper.shape == (14,)
        # Lower bounds must be strictly less than upper bounds
        assert np.all(lower < upper)

    def test_bounds_contain_initial_values(self) -> None:
        """Initial values are within bounds."""
        model = make_model_mock(n_params=14)
        values = model.param_manager.get_initial_values()
        lower, upper = model.param_manager.get_bounds()
        assert np.all(values >= lower)
        assert np.all(values <= upper)

    def test_custom_n_params(self) -> None:
        """Factory respects custom n_params for sub-selections."""
        model = make_model_mock(n_params=7)
        assert len(model.param_manager.varying_names) == 7
        assert model.param_manager.get_initial_values().shape == (7,)

    def test_n_varying_attribute(self) -> None:
        """Mock exposes n_varying on both model and param_manager."""
        model = make_model_mock(n_params=10)
        assert model.param_manager.n_varying == 10
        assert model.n_varying == 10
