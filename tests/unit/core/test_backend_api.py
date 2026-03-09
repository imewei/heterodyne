"""Tests for heterodyne.core.backend_api — backend selection and abstraction.

Covers:
- Backend enum
- BackendConfig dataclass
- _check_jax_available
- get_current_backend
- set_backend
- get_array_module
- ensure_array
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from heterodyne.core.backend_api import (
    Backend,
    BackendConfig,
    ensure_array,
    get_array_module,
    get_current_backend,
    set_backend,
)


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------


class TestBackendEnum:
    """Tests for Backend enum values."""

    def test_jax_value(self) -> None:
        assert Backend.JAX.value == "jax"

    def test_numpy_value(self) -> None:
        assert Backend.NUMPY.value == "numpy"

    def test_unique_values(self) -> None:
        values = [b.value for b in Backend]
        assert len(values) == len(set(values))

    def test_members(self) -> None:
        assert set(Backend.__members__.keys()) == {"JAX", "NUMPY"}


# ---------------------------------------------------------------------------
# BackendConfig
# ---------------------------------------------------------------------------


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_defaults(self) -> None:
        config = BackendConfig()
        assert config.backend is Backend.JAX
        assert config.device == "cpu"
        assert config.precision == "float64"

    def test_custom_values(self) -> None:
        config = BackendConfig(backend=Backend.NUMPY, device="gpu", precision="float32")
        assert config.backend is Backend.NUMPY
        assert config.device == "gpu"
        assert config.precision == "float32"


# ---------------------------------------------------------------------------
# get_current_backend
# ---------------------------------------------------------------------------


class TestGetCurrentBackend:
    """Tests for backend auto-detection."""

    def test_returns_jax_when_available(self) -> None:
        # JAX should be available in the test environment
        result = get_current_backend()
        assert result is Backend.JAX

    def test_returns_backend_enum(self) -> None:
        result = get_current_backend()
        assert isinstance(result, Backend)


# ---------------------------------------------------------------------------
# set_backend
# ---------------------------------------------------------------------------


class TestSetBackend:
    """Tests for backend configuration."""

    def test_set_jax_backend(self) -> None:
        # Should not raise
        set_backend(Backend.JAX)

    def test_set_numpy_backend(self) -> None:
        # Should not raise
        set_backend(Backend.NUMPY)

    def test_jax_sets_env_defaults(self) -> None:
        import os

        # set_backend uses setdefault, so it won't overwrite existing values
        set_backend(Backend.JAX)
        assert os.environ.get("JAX_ENABLE_X64") is not None
        assert os.environ.get("JAX_PLATFORMS") is not None


# ---------------------------------------------------------------------------
# get_array_module
# ---------------------------------------------------------------------------


class TestGetArrayModule:
    """Tests for array module retrieval."""

    def test_jax_backend_returns_jnp(self) -> None:
        import jax.numpy as jnp

        xp = get_array_module(Backend.JAX)
        assert xp is jnp

    def test_numpy_backend_returns_np(self) -> None:
        xp = get_array_module(Backend.NUMPY)
        assert xp is np

    def test_none_auto_detects(self) -> None:
        xp = get_array_module(None)
        assert isinstance(xp, ModuleType)

    def test_returned_module_has_linspace(self) -> None:
        xp = get_array_module(Backend.JAX)
        assert hasattr(xp, "linspace")

    def test_returned_module_has_array(self) -> None:
        xp = get_array_module(Backend.NUMPY)
        assert hasattr(xp, "array")

    def test_jax_unavailable_raises(self) -> None:
        """When JAX is requested but unavailable, RuntimeError is raised."""
        import heterodyne.core.backend_api as mod

        original = mod._jax_available
        try:
            mod._jax_available = False
            with pytest.raises(RuntimeError, match="JAX backend requested"):
                get_array_module(Backend.JAX)
        finally:
            mod._jax_available = original


# ---------------------------------------------------------------------------
# ensure_array
# ---------------------------------------------------------------------------


class TestEnsureArray:
    """Tests for array conversion."""

    def test_list_to_jax_array(self) -> None:
        import jax.numpy as jnp

        result = ensure_array([1.0, 2.0, 3.0], Backend.JAX)
        assert isinstance(result, jnp.ndarray)
        npt.assert_allclose(np.asarray(result), [1.0, 2.0, 3.0])

    def test_list_to_numpy_array(self) -> None:
        result = ensure_array([1.0, 2.0], Backend.NUMPY)
        assert isinstance(result, np.ndarray)
        npt.assert_allclose(result, [1.0, 2.0])

    def test_scalar_conversion(self) -> None:
        result = ensure_array(3.14, Backend.NUMPY)
        assert isinstance(result, np.ndarray)
        npt.assert_allclose(float(result), 3.14)

    def test_numpy_array_passthrough(self) -> None:
        arr = np.array([1.0, 2.0])
        result = ensure_array(arr, Backend.NUMPY)
        npt.assert_allclose(result, arr)

    def test_auto_detect_backend(self) -> None:
        result = ensure_array([1.0, 2.0])
        # Should work without specifying backend
        assert hasattr(result, "shape")

    def test_preserves_values(self) -> None:
        data = [1.5, -2.3, 0.0, 100.0]
        for backend in [Backend.JAX, Backend.NUMPY]:
            result = ensure_array(data, backend)
            npt.assert_allclose(np.asarray(result), data, rtol=1e-7)
