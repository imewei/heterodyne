"""Additional coverage tests for nlsq/adapter.py.

Covers:
- NLSQAdapter.fit() delegation (wraps residual into CurveFit)
- NLSQAdapter.fit_jax() exception handling (covariance warning, failure path)
- NLSQAdapter.fit_jax() returns initial params on failure
- dogbox to trf method conversion
- None covariance handling
- Model cache (hit, miss, eviction, clear)
- Convergence assessment (_assess_convergence)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.optimization.nlsq.adapter import (
    NLSQAdapter,
    _assess_convergence,
    clear_model_cache,
    get_cache_stats,
    get_or_create_fitter,
)
from heterodyne.optimization.nlsq.config import NLSQConfig

# ============================================================================
# NLSQAdapter.fit() delegation
# ============================================================================


@pytest.mark.unit
class TestNLSQAdapterFitDelegation:
    """NLSQAdapter.fit() wraps a residual_fn into CurveFit-compatible form."""

    def test_fit_converges_on_linear_residual(self) -> None:
        """fit() wraps residual_fn, calls CurveFit, returns success."""
        adapter = NLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=50, tolerance=1e-6)
        clear_model_cache()

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (
            np.array([2.0, 1.0]),
            np.eye(2) * 0.01,
        )

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit(
                residual_fn=lambda p: p - np.array([2.0, 1.0]),
                initial_params=np.array([1.0, 0.5]),
                bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
                config=config,
            )

        assert result.success
        np.testing.assert_allclose(result.parameters[0], 2.0, atol=0.1)
        np.testing.assert_allclose(result.parameters[1], 1.0, atol=0.1)
        mock_fitter.curve_fit.assert_called_once()

    def test_fit_with_jacobian_kwarg(self) -> None:
        """jacobian_fn is accepted but CurveFit doesn't use it."""
        adapter = NLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=50, tolerance=1e-6)
        clear_model_cache()

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (
            np.array([1.0, 2.0]),
            np.eye(2) * 0.01,
        )

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit(
                residual_fn=lambda p: p - np.array([1.0, 2.0]),
                initial_params=np.array([0.5, 1.5]),
                bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
                config=config,
                jacobian_fn=lambda p: np.eye(2),
            )

        assert result.success

    def test_fit_exception_returns_failure(self) -> None:
        """RuntimeError in residual_fn produces a failed result."""

        def bad_residual(params: np.ndarray) -> np.ndarray:
            raise RuntimeError("Residual exploded")

        adapter = NLSQAdapter(parameter_names=["a", "b"])
        config = NLSQConfig(max_iterations=10)
        clear_model_cache()

        result = adapter.fit(
            residual_fn=bad_residual,
            initial_params=np.array([1.0, 1.0]),
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            config=config,
        )

        assert result.success is False
        assert "exploded" in result.message.lower()


# ============================================================================
# NLSQAdapter.fit_jax() exception handling
# ============================================================================


@pytest.mark.unit
class TestNLSQAdapterFitJaxExceptionHandling:
    """Exception paths in fit_jax()."""

    @pytest.mark.requires_jax
    def test_fit_jax_covariance_with_negative_diag(self) -> None:
        """Bad covariance (negative diag) is abs'd then sqrt'd by result_builder."""
        clear_model_cache()
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        mock_fitter = MagicMock()
        bad_covariance = np.array([[-1.0, 0.0], [0.0, -1.0]])
        mock_fitter.curve_fit.return_value = (
            np.array([2.0, 3.0]),
            bad_covariance,
        )

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

        assert result.success
        # build_result_from_nlsq does sqrt(diag(abs(pcov))), so uncertainties are sqrt(1)=1
        assert result.uncertainties is not None
        np.testing.assert_allclose(result.uncertainties, [1.0, 1.0])

    @pytest.mark.requires_jax
    def test_fit_jax_optimization_failure(self) -> None:
        """RuntimeError from curve_fit produces a failed result."""
        clear_model_cache()
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.side_effect = RuntimeError("Optimization diverged")

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

        assert result.success is False
        assert "diverged" in result.message.lower()
        assert result.wall_time_seconds is not None

    @pytest.mark.requires_jax
    def test_fit_jax_returns_initial_params_on_failure(self) -> None:
        """On exception, fit_jax returns initial params."""
        clear_model_cache()
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        initial = np.array([5.0, 3.0])
        mock_fitter = MagicMock()
        mock_fitter.curve_fit.side_effect = ValueError("Bad params")

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=initial,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

        np.testing.assert_array_equal(result.parameters, initial)


# ============================================================================
# dogbox → trf method conversion
# ============================================================================


@pytest.mark.unit
class TestNLSQAdapterMethodConversion:
    """dogbox is not supported by CurveFit; must be converted to trf."""

    def test_fit_converts_dogbox_to_trf(self) -> None:
        adapter = NLSQAdapter(parameter_names=["p1"])
        config = NLSQConfig(method="dogbox", max_iterations=10)
        clear_model_cache()

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (np.array([2.0]), np.eye(1))

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            adapter.fit(
                residual_fn=lambda p: p - np.array([2.0]),
                initial_params=np.array([1.0]),
                bounds=(np.array([0.0]), np.array([10.0])),
                config=config,
            )

        call_kwargs = mock_fitter.curve_fit.call_args[1]
        assert call_kwargs["method"] == "trf"

    @pytest.mark.requires_jax
    def test_fit_jax_converts_dogbox_to_trf(self) -> None:
        adapter = NLSQAdapter(parameter_names=["p1"])
        config = NLSQConfig(method="dogbox", max_iterations=10)
        clear_model_cache()

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (np.array([2.0]), np.eye(1))

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            adapter.fit_jax(
                jax_residual_fn=lambda x, p1: jnp.zeros_like(x),
                initial_params=np.array([1.0]),
                bounds=(np.array([0.0]), np.array([10.0])),
                config=config,
                n_data=100,
            )

        call_kwargs = mock_fitter.curve_fit.call_args[1]
        assert call_kwargs["method"] == "trf"


# ============================================================================
# None covariance handling
# ============================================================================


@pytest.mark.unit
class TestNLSQAdapterNoneCovariance:
    """None covariance from curve_fit produces None uncertainties."""

    @pytest.mark.requires_jax
    def test_fit_jax_handles_none_covariance(self) -> None:
        clear_model_cache()
        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=10, tolerance=1e-4)

        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (np.array([2.0, 3.0]), None)

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            result = adapter.fit_jax(
                jax_residual_fn=jax_fn,
                initial_params=np.array([1.0, 1.0]),
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
                config=config,
                n_data=100,
            )

        assert result.success
        assert result.covariance is None
        assert result.uncertainties is None


# ============================================================================
# Model cache
# ============================================================================


@pytest.mark.unit
class TestModelCache:
    """CurveFit model caching: hit, miss, eviction, clear."""

    def test_cache_hit(self) -> None:
        clear_model_cache()
        fitter1, hit1 = get_or_create_fitter(100, 5)
        fitter2, hit2 = get_or_create_fitter(100, 5)

        assert not hit1
        assert hit2
        assert fitter1 is fitter2

        stats = get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_miss_different_shape(self) -> None:
        clear_model_cache()
        fitter1, hit1 = get_or_create_fitter(100, 5)
        fitter2, hit2 = get_or_create_fitter(200, 5)

        assert not hit1
        assert not hit2
        assert fitter1 is not fitter2

    def test_cache_miss_different_phi_angles(self) -> None:
        clear_model_cache()
        _, hit1 = get_or_create_fitter(100, 5, phi_angles=(0.0,))
        _, hit2 = get_or_create_fitter(100, 5, phi_angles=(0.0, 1.0))

        assert not hit1
        assert not hit2

    def test_cache_miss_different_scaling_mode(self) -> None:
        clear_model_cache()
        _, hit1 = get_or_create_fitter(100, 5, scaling_mode="auto")
        _, hit2 = get_or_create_fitter(100, 5, scaling_mode="individual")

        assert not hit1
        assert not hit2

    def test_cache_eviction_at_max_size(self) -> None:
        """Cache evicts oldest entry when _MODEL_CACHE_MAX_SIZE is reached."""
        clear_model_cache()
        # Fill cache to max size (64)
        for i in range(64):
            get_or_create_fitter(1000 + i, 5)

        assert get_cache_stats()["size"] == 64

        # One more should evict oldest
        get_or_create_fitter(9999, 5)
        assert get_cache_stats()["size"] == 64

    def test_clear_cache_resets_all(self) -> None:
        get_or_create_fitter(100, 5)
        clear_model_cache()

        stats = get_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ============================================================================
# Convergence assessment
# ============================================================================


@pytest.mark.unit
class TestAssessConvergence:
    """Tests for _assess_convergence heuristics."""

    def test_finite_params_succeed(self) -> None:
        success, message, reason = _assess_convergence(
            fitted_params=np.array([1.0, 2.0]),
            initial_params=np.array([0.0, 0.0]),
            reduced_chi2=1.5,
        )
        assert success
        assert reason == "tolerance"

    def test_non_finite_params_fail(self) -> None:
        success, message, reason = _assess_convergence(
            fitted_params=np.array([np.nan, 2.0]),
            initial_params=np.array([0.0, 0.0]),
            reduced_chi2=1.0,
        )
        assert not success
        assert reason == "failed"

    def test_huge_chi2_fails(self) -> None:
        success, message, reason = _assess_convergence(
            fitted_params=np.array([1.0, 2.0]),
            initial_params=np.array([0.0, 0.0]),
            reduced_chi2=1e7,
        )
        assert not success
        assert reason == "poor_fit"

    def test_no_progress_fails(self) -> None:
        params = np.array([1.0, 2.0])
        success, message, reason = _assess_convergence(
            fitted_params=params.copy(),
            initial_params=params.copy(),
            reduced_chi2=0.5,
        )
        assert not success
        assert reason == "no_progress"

    def test_none_chi2_does_not_fail(self) -> None:
        success, _, _ = _assess_convergence(
            fitted_params=np.array([1.0, 2.0]),
            initial_params=np.array([0.0, 0.0]),
            reduced_chi2=None,
        )
        assert success

    def test_inf_params_fail(self) -> None:
        success, _, reason = _assess_convergence(
            fitted_params=np.array([np.inf, 2.0]),
            initial_params=np.array([0.0, 0.0]),
            reduced_chi2=1.0,
        )
        assert not success
        assert reason == "failed"
