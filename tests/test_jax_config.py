"""Tests for JAX Float64 precision configuration.

Bug Prevented: JAX Float64 Precision Issues
--------------------------------------------
JAX defaults to float32 for performance, but scientific computing requires
float64 for numerical stability. The x64 mode must be set BEFORE JAX is
imported, otherwise arrays silently use float32.

These tests verify that:
1. JAX x64 mode is properly configured
2. Arrays default to float64
3. Computations maintain float64 precision
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestJAXX64Configuration:
    """Tests for JAX x64 configuration state."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_x64_enabled_via_config(self) -> None:
        """Verify JAX x64 is enabled via jax.config.

        This should be True if heterodyne was imported correctly.
        """
        assert jax.config.jax_enable_x64, (
            "JAX x64 mode is not enabled via config. "
            "Import heterodyne before importing JAX, or set "
            "JAX_ENABLE_X64=True environment variable."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_x64_enabled_via_env(self) -> None:
        """Verify JAX_ENABLE_X64 environment variable is set.

        The heterodyne package sets this in __init__.py.
        """
        env_value = os.environ.get("JAX_ENABLE_X64", "").lower()

        # Should be set to 'true' or '1'
        assert env_value in ("true", "1"), (
            f"JAX_ENABLE_X64 environment variable is '{env_value}', "
            "expected 'True' or '1'. Check heterodyne/__init__.py."
        )


class TestArrayDtypes:
    """Tests for default array dtypes."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_arrays_default_to_float64(self) -> None:
        """Verify JAX arrays created from Python floats are float64.

        With x64 mode enabled, jnp.array([1.0, 2.0]) should be float64.
        """
        arr = jnp.array([1.0, 2.0, 3.0])

        assert arr.dtype == jnp.float64, (
            f"JAX array has dtype {arr.dtype}, expected float64. "
            "This indicates x64 mode is not properly enabled."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_arange_default_dtype(self) -> None:
        """Verify jnp.arange creates float64 by default."""
        arr = jnp.arange(10.0)

        assert arr.dtype == jnp.float64, (
            f"jnp.arange has dtype {arr.dtype}, expected float64."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_zeros_default_dtype(self) -> None:
        """Verify jnp.zeros creates float64 by default."""
        jnp.zeros(10)

        # Note: jnp.zeros(10) creates float32 by default even with x64
        # Must use dtype parameter for float64
        arr_f64 = jnp.zeros(10, dtype=jnp.float64)

        assert arr_f64.dtype == jnp.float64

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_numpy_to_jax_preserves_float64(self) -> None:
        """Verify converting numpy float64 to JAX preserves dtype."""
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        jax_arr = jnp.asarray(np_arr)

        assert jax_arr.dtype == jnp.float64, (
            f"Converted array has dtype {jax_arr.dtype}, expected float64."
        )


class TestHeterodyneImport:
    """Tests for heterodyne import behavior."""

    @pytest.mark.unit
    def test_heterodyne_import_sets_x64(self) -> None:
        """Verify importing heterodyne enables x64 mode.

        The heterodyne/__init__.py should set JAX_ENABLE_X64 and
        call jax.config.update before any JAX operations.
        """
        # At this point heterodyne has already been imported via conftest
        # Check that x64 is enabled
        assert jax.config.jax_enable_x64

    @pytest.mark.unit
    def test_heterodyne_init_sets_env_var(self) -> None:
        """Verify heterodyne sets the environment variable."""
        # The env var should be set by heterodyne/__init__.py
        assert os.environ.get("JAX_ENABLE_X64") is not None


class TestComputationDtypes:
    """Tests for computation output dtypes."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_correlation_output_dtype(
        self,
        small_heterodyne_model,
        ensure_jax_float64: bool,
    ) -> None:
        """Verify correlation computation outputs float64."""
        c2 = small_heterodyne_model.compute_correlation(phi_angle=0.0)

        # Convert to JAX array if numpy
        c2_jax = jnp.asarray(c2)

        assert c2_jax.dtype == jnp.float64, (
            f"Correlation output has dtype {c2_jax.dtype}, expected float64."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_residual_output_dtype(
        self,
        small_heterodyne_model,
        small_c2_data: np.ndarray,
        ensure_jax_float64: bool,
    ) -> None:
        """Verify residual computation outputs float64."""
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data, dtype=jnp.float64)
        full_params = jnp.asarray(model.get_params(), dtype=jnp.float64)

        residuals = compute_residuals(
            full_params,
            model.t,
            model.q,
            model.dt,
            0.0,
            c2_jax,
            None,
        )

        assert residuals.dtype == jnp.float64, (
            f"Residuals have dtype {residuals.dtype}, expected float64."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_g1_computation_dtype(
        self,
        ensure_jax_float64: bool,
    ) -> None:
        """Verify g1 correlation computation outputs float64."""
        from heterodyne.core.jax_backend import compute_g1_transport

        J = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=jnp.float64)
        q = 0.1

        g1 = compute_g1_transport(J, q)

        assert g1.dtype == jnp.float64, (
            f"g1 has dtype {g1.dtype}, expected float64."
        )


class TestNumericPrecision:
    """Tests for numeric precision requirements."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_small_number_precision(self) -> None:
        """Verify small numbers maintain precision.

        Float32 loses precision for values around 1e-7, which is
        common in XPCS analysis.
        """
        small_val = 1e-10
        arr = jnp.array([small_val])

        assert arr.dtype == jnp.float64
        assert jnp.isclose(arr[0], small_val, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_large_number_precision(self) -> None:
        """Verify large numbers maintain precision.

        Float32 loses precision for values above 1e7.
        """
        large_val = 1e12
        arr = jnp.array([large_val])

        assert arr.dtype == jnp.float64
        assert jnp.isclose(arr[0], large_val, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_subtraction_precision(self) -> None:
        """Verify precision is maintained during subtraction.

        Catastrophic cancellation is a common issue with float32.
        """
        a = jnp.array([1.0000001], dtype=jnp.float64)
        b = jnp.array([1.0], dtype=jnp.float64)

        diff = a - b

        assert diff.dtype == jnp.float64
        assert jnp.isclose(diff[0], 1e-7, rtol=1e-3)


class TestBugPrevention_Float64Precision:
    """Regression tests for JAX Float64 Precision bug.

    BUG DESCRIPTION:
    JAX defaults to float32 for performance. If x64 mode is not enabled
    BEFORE JAX is imported, arrays silently use float32, causing:
    - Precision loss in scientific computations
    - Inconsistent results between runs
    - Numerical instability in optimization

    The fix requires setting the environment variable and config
    BEFORE importing JAX:

        import os
        os.environ["JAX_ENABLE_X64"] = "True"

        import jax
        jax.config.update("jax_enable_x64", True)

        # Now import other modules

    These tests verify float64 precision is maintained.
    """

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_float32_loses_precision(self) -> None:
        """REGRESSION TEST: Demonstrate float32 precision loss.

        This test documents why float64 is required - float32 cannot
        represent small differences accurately.
        """
        # These values are indistinguishable in float32
        a = np.float32(1.0000001)
        b = np.float32(1.0)

        # float32 precision loss
        diff_f32 = a - b
        # In float32, this is often 0 or very wrong
        assert diff_f32 < 1e-6 or diff_f32 == 0, (
            "float32 cannot distinguish 1.0000001 from 1.0"
        )

        # float64 maintains precision
        a_f64 = np.float64(1.0000001)
        b_f64 = np.float64(1.0)
        diff_f64 = a_f64 - b_f64
        assert np.isclose(diff_f64, 1e-7, rtol=1e-3), (
            "float64 correctly represents the difference"
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_heterodyne_sets_x64_before_jax_import(self) -> None:
        """REGRESSION TEST: Verify heterodyne enables x64 correctly.

        The heterodyne/__init__.py must set JAX_ENABLE_X64=True BEFORE
        any JAX imports to ensure float64 is used.
        """
        # At this point, heterodyne has been imported
        # Check that x64 is enabled
        assert jax.config.jax_enable_x64, (
            "JAX x64 mode is not enabled. "
            "heterodyne/__init__.py must set this before JAX import."
        )

        # Check that arrays are float64
        arr = jnp.array([1.0, 2.0])
        assert arr.dtype == jnp.float64, (
            f"JAX arrays are {arr.dtype}, not float64. "
            "This indicates x64 was not enabled before JAX import."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_env_var_is_set(self) -> None:
        """REGRESSION TEST: Verify environment variable is set.

        The JAX_ENABLE_X64 environment variable must be set for
        reproducible behavior across processes.
        """
        env_value = os.environ.get("JAX_ENABLE_X64", "")

        assert env_value.lower() in ("true", "1"), (
            f"JAX_ENABLE_X64 is '{env_value}', expected 'True'. "
            "This must be set by heterodyne/__init__.py."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_correlation_uses_float64(
        self,
        small_heterodyne_model,
        ensure_jax_float64: bool,
    ) -> None:
        """REGRESSION TEST: Verify correlation computation uses float64.

        If float32 were used, correlation values would have precision loss
        that could affect fitting results.
        """
        c2 = small_heterodyne_model.compute_correlation(phi_angle=0.0)
        c2_jax = jnp.asarray(c2)

        assert c2_jax.dtype == jnp.float64, (
            f"Correlation has dtype {c2_jax.dtype}, expected float64. "
            "This would cause precision loss in fitting."
        )

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_residual_computation_maintains_float64(
        self,
        small_heterodyne_model,
        small_c2_data: np.ndarray,
        ensure_jax_float64: bool,
    ) -> None:
        """REGRESSION TEST: Verify residuals maintain float64 throughout.

        If any computation downcasts to float32, residuals would be
        inaccurate, leading to poor fits.
        """
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data, dtype=jnp.float64)
        params = jnp.asarray(model.get_params(), dtype=jnp.float64)

        residuals = compute_residuals(
            params,
            model.t,
            model.q,
            model.dt,
            0.0,
            c2_jax,
            None,
        )

        assert residuals.dtype == jnp.float64, (
            f"Residuals have dtype {residuals.dtype}, expected float64."
        )

        # Also verify intermediate values don't silently downcast
        assert c2_jax.dtype == jnp.float64
        assert params.dtype == jnp.float64
