"""Additional tests for nlsq/adapter.py to improve coverage.

Covers:
- NLSQAdapter.fit() delegation to scipy (lines 71-73)
- NLSQAdapter.fit_jax() exception handling - covariance warning (lines 148-149)
- NLSQAdapter.fit_jax() exception handling - failure path (lines 169-173)
- ScipyNLSQAdapter.fit() exception handling - failure path (lines 293-297)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.optimization.nlsq.adapter import NLSQAdapter, ScipyNLSQAdapter
from heterodyne.optimization.nlsq.config import NLSQConfig


# ============================================================================
# Test NLSQAdapter.fit() delegation
# ============================================================================


class TestNLSQAdapterFitDelegation:
    """Tests for NLSQAdapter.fit() method (lines 71-73)."""

    @pytest.mark.unit
    def test_nlsq_adapter_fit_delegates_to_scipy(self) -> None:
        """Test NLSQAdapter.fit() delegates to ScipyNLSQAdapter for numpy functions."""

        def residual_fn(params: np.ndarray) -> np.ndarray:
            a, b = params
            x = np.linspace(0, 1, 10)
            y_true = 2.0 * x + 1.0
            y_pred = a * x + b
            return y_pred - y_true

        adapter = NLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=50, tolerance=1e-6)

        result = adapter.fit(
            residual_fn=residual_fn,
            initial_params=np.array([1.0, 0.5]),
            bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            config=config,
        )

        # Should succeed via scipy delegation
        assert result.success
        np.testing.assert_allclose(result.parameters[0], 2.0, atol=0.1)
        np.testing.assert_allclose(result.parameters[1], 1.0, atol=0.1)

    @pytest.mark.unit
    def test_nlsq_adapter_fit_with_jacobian(self) -> None:
        """Test NLSQAdapter.fit() passes jacobian to scipy."""

        def residual_fn(params: np.ndarray) -> np.ndarray:
            a, b = params
            x = np.linspace(0, 1, 10)
            return a * x + b - (2.0 * x + 1.0)

        def jacobian_fn(params: np.ndarray) -> np.ndarray:
            x = np.linspace(0, 1, 10)
            jac = np.zeros((10, 2))
            jac[:, 0] = x  # d/da
            jac[:, 1] = 1.0  # d/db
            return jac

        adapter = NLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=50, tolerance=1e-6, use_jac=True)

        result = adapter.fit(
            residual_fn=residual_fn,
            initial_params=np.array([1.0, 0.5]),
            bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            config=config,
            jacobian_fn=jacobian_fn,
        )

        assert result.success


# ============================================================================
# Test NLSQAdapter.fit_jax() exception handling
# ============================================================================


class TestNLSQAdapterFitJaxExceptionHandling:
    """Tests for NLSQAdapter.fit_jax() exception paths."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_fit_jax_covariance_extraction_warning(self) -> None:
        """Test fit_jax handles covariance extraction failure gracefully (lines 148-149)."""
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        # Mock CurveFit to return covariance that will fail diag extraction
        # Patch at nlsq module level since that's where it's imported from
        with patch("nlsq.CurveFit") as MockCurveFit:
            mock_instance = MagicMock()
            # Return covariance that causes sqrt to fail (negative diagonal)
            bad_covariance = np.array([[-1.0, 0.0], [0.0, -1.0]])
            mock_instance.curve_fit.return_value = (
                np.array([1.0, 1.0]),
                bad_covariance,
            )
            MockCurveFit.return_value = mock_instance

            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

            # Should still succeed but uncertainties may be None or NaN
            assert result.success

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_fit_jax_optimization_failure(self) -> None:
        """Test fit_jax returns failure result on exception (lines 169-173)."""
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        # Mock CurveFit to raise an exception
        with patch("nlsq.CurveFit") as MockCurveFit:
            mock_instance = MagicMock()
            mock_instance.curve_fit.side_effect = RuntimeError("Optimization diverged")
            MockCurveFit.return_value = mock_instance

            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

            # Should return failure result
            assert result.success is False
            assert "diverged" in result.message.lower()
            assert result.wall_time_seconds is not None

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_fit_jax_returns_initial_params_on_failure(self) -> None:
        """Test fit_jax returns initial params when optimization fails."""
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        initial = np.array([5.0, 3.0])

        with patch("nlsq.CurveFit") as MockCurveFit:
            mock_instance = MagicMock()
            mock_instance.curve_fit.side_effect = ValueError("Bad params")
            MockCurveFit.return_value = mock_instance

            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=initial,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

            # Should return initial params
            np.testing.assert_array_equal(result.parameters, initial)


# ============================================================================
# Test ScipyNLSQAdapter.fit() exception handling
# ============================================================================


class TestScipyNLSQAdapterFitExceptionHandling:
    """Tests for ScipyNLSQAdapter.fit() exception paths (lines 293-297)."""

    @pytest.mark.unit
    def test_scipy_adapter_fit_handles_exception(self) -> None:
        """Test ScipyNLSQAdapter.fit() handles exceptions gracefully."""

        def bad_residual_fn(params: np.ndarray) -> np.ndarray:
            raise RuntimeError("Residual computation exploded")

        adapter = ScipyNLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)
        initial = np.array([1.0, 1.0])

        result = adapter.fit(
            residual_fn=bad_residual_fn,
            initial_params=initial,
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            config=config,
        )

        # Should return failure result
        assert result.success is False
        assert "exploded" in result.message.lower()
        assert result.wall_time_seconds is not None
        np.testing.assert_array_equal(result.parameters, initial)

    @pytest.mark.unit
    def test_scipy_adapter_fit_nan_residual(self) -> None:
        """Test ScipyNLSQAdapter.fit() handles NaN residuals."""

        call_count = 0

        def nan_residual_fn(params: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                return np.array([np.nan, np.nan, np.nan])
            return np.array([1.0, 1.0, 1.0])

        adapter = ScipyNLSQAdapter(parameter_names=["p1"])
        config = NLSQConfig(max_iterations=100, tolerance=1e-8)

        result = adapter.fit(
            residual_fn=nan_residual_fn,
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
            config=config,
        )

        # Scipy should handle this (may or may not succeed)
        assert result is not None


# ============================================================================
# Test ScipyNLSQAdapter covariance computation edge cases
# ============================================================================


class TestScipyAdapterCovarianceEdgeCases:
    """Tests for covariance computation in ScipyNLSQAdapter."""

    @pytest.mark.unit
    def test_scipy_adapter_singular_jacobian(self) -> None:
        """Test ScipyNLSQAdapter handles singular Jacobian (line 271-272)."""

        def residual_fn(params: np.ndarray) -> np.ndarray:
            # Function where Jacobian is singular (constant output)
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        adapter = ScipyNLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        result = adapter.fit(
            residual_fn=residual_fn,
            initial_params=np.array([1.0, 1.0]),
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            config=config,
        )

        # Should complete (covariance may be None due to singular matrix)
        assert result is not None

    @pytest.mark.unit
    def test_scipy_adapter_zero_dof(self) -> None:
        """Test ScipyNLSQAdapter handles zero degrees of freedom."""

        def residual_fn(params: np.ndarray) -> np.ndarray:
            # Same number of residuals as parameters
            a, b = params
            return np.array([a - 1.0, b - 2.0])

        adapter = ScipyNLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=50, tolerance=1e-8)

        result = adapter.fit(
            residual_fn=residual_fn,
            initial_params=np.array([0.5, 1.5]),
            bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
            config=config,
        )

        # Should handle n_dof = 0 case
        assert result is not None
        # reduced_chi2 should handle division by zero gracefully
        # (either None or a valid number based on implementation)


# ============================================================================
# Test NLSQAdapter.fit_jax with method conversion
# ============================================================================


class TestNLSQAdapterMethodConversion:
    """Test method name conversion in fit_jax."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_fit_jax_converts_dogbox_to_trf(self) -> None:
        """Test that dogbox method is converted to trf for nlsq (line 130)."""
        adapter = NLSQAdapter(parameter_names=["p1"])
        config = NLSQConfig(method="dogbox", max_iterations=10)

        def jax_fn(x: jnp.ndarray, p1: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        with patch("nlsq.CurveFit") as MockCurveFit:
            mock_instance = MagicMock()
            mock_instance.curve_fit.return_value = (np.array([1.0]), np.eye(1))
            MockCurveFit.return_value = mock_instance

            adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0]),
                bounds=(np.array([0.0]), np.array([10.0])),
                config=config,
                n_data=100,
            )

            # Check that curve_fit was called with method="trf", not "dogbox"
            call_kwargs = mock_instance.curve_fit.call_args[1]
            assert call_kwargs["method"] == "trf"


# ============================================================================
# Test NLSQAdapter.fit_jax with None covariance
# ============================================================================


class TestNLSQAdapterNoneCovariance:
    """Test handling of None covariance from nlsq."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_fit_jax_handles_none_covariance(self) -> None:
        """Test fit_jax handles None covariance from curve_fit (lines 145, 159)."""
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        with patch("nlsq.CurveFit") as MockCurveFit:
            mock_instance = MagicMock()
            # Return None for covariance
            mock_instance.curve_fit.return_value = (np.array([1.0, 1.0]), None)
            MockCurveFit.return_value = mock_instance

            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

            # Should succeed with None uncertainties
            assert result.success
            assert result.covariance is None
            assert result.uncertainties is None
