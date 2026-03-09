"""Unit tests for heterodyne.optimization.nlsq.adapter (dual-adapter pattern).

Tests cover:
- Model cache utilities (ModelCacheKey, get_or_create_fitter, clear_model_cache,
  get_cache_stats, eviction at capacity 64)
- NLSQAdapter: name, supports_bounds, supports_jacobian, fit()
- NLSQWrapper: name, supports_bounds, supports_jacobian, strategy routing,
  fallback chain
- Absence of ScipyNLSQAdapter
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAM_NAMES = ["p1", "p2", "p3"]
N_PARAMS = len(PARAM_NAMES)
N_DATA = 50


def _make_initial_params() -> np.ndarray:
    return np.ones(N_PARAMS, dtype=np.float64)


def _make_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(N_PARAMS, -10.0)
    upper = np.full(N_PARAMS, 10.0)
    return lower, upper


def _make_config() -> Any:
    """Return a minimal NLSQConfig."""
    from heterodyne.optimization.nlsq.config import NLSQConfig

    return NLSQConfig(max_iterations=5, tolerance=1e-4)


def _make_residual_fn(n_data: int = N_DATA) -> Any:
    """Return a trivial residual function that returns zeros."""

    def fn(params: np.ndarray) -> np.ndarray:
        return np.zeros(n_data, dtype=np.float64)

    return fn


def _make_nlsq_tuple_result(n_params: int = N_PARAMS) -> tuple[Any, Any]:
    """Minimal (popt, pcov) tuple that build_result_from_nlsq can handle."""
    popt = np.ones(n_params, dtype=np.float64)
    pcov = np.eye(n_params, dtype=np.float64) * 0.01
    return popt, pcov


# ---------------------------------------------------------------------------
# Model cache utilities
# ---------------------------------------------------------------------------


class TestModelCache:
    """Tests for get_or_create_fitter / cache helpers."""

    def setup_method(self) -> None:
        """Clear cache before each test so tests are independent."""
        from heterodyne.optimization.nlsq.adapter import clear_model_cache

        clear_model_cache()

    def teardown_method(self) -> None:
        from heterodyne.optimization.nlsq.adapter import clear_model_cache

        clear_model_cache()

    def test_model_cache_hit(self) -> None:
        """Second call with same key should return cached fitter (hit=True)."""
        mock_fitter = MagicMock()

        with patch(
            "heterodyne.optimization.nlsq.adapter.CurveFit",
            return_value=mock_fitter,
        ):
            from heterodyne.optimization.nlsq.adapter import get_or_create_fitter

            fitter1, hit1 = get_or_create_fitter(
                n_data=100,
                n_params=4,
                phi_angles=(0.0,),
                scaling_mode="auto",
            )
            fitter2, hit2 = get_or_create_fitter(
                n_data=100,
                n_params=4,
                phi_angles=(0.0,),
                scaling_mode="auto",
            )

        assert hit1 is False, "First call must be a cache miss"
        assert hit2 is True, "Second call with identical key must be a cache hit"
        assert fitter1 is fitter2, "Both calls must return the same fitter object"

    def test_model_cache_eviction(self) -> None:
        """Cache should evict oldest entry when exceeding max size (64)."""
        from heterodyne.optimization.nlsq.adapter import (
            _MODEL_CACHE_MAX_SIZE,
            _model_cache,
            get_or_create_fitter,
        )

        assert _MODEL_CACHE_MAX_SIZE == 64, "Cache max size must be 64"

        with patch(
            "heterodyne.optimization.nlsq.adapter.CurveFit",
            side_effect=lambda flength: MagicMock(),
        ):
            # Fill cache exactly to capacity
            for i in range(_MODEL_CACHE_MAX_SIZE):
                get_or_create_fitter(
                    n_data=i + 1,
                    n_params=2,
                    phi_angles=None,
                    scaling_mode="auto",
                )

            assert len(_model_cache) == _MODEL_CACHE_MAX_SIZE

            # One more entry should trigger eviction
            get_or_create_fitter(
                n_data=_MODEL_CACHE_MAX_SIZE + 1,
                n_params=2,
                phi_angles=None,
                scaling_mode="auto",
            )

        assert len(_model_cache) == _MODEL_CACHE_MAX_SIZE, (
            "Cache size must not exceed max after eviction"
        )

    def test_cache_stats(self) -> None:
        """Hits and misses must be tracked correctly by get_cache_stats."""
        with patch(
            "heterodyne.optimization.nlsq.adapter.CurveFit",
            return_value=MagicMock(),
        ):
            from heterodyne.optimization.nlsq.adapter import (
                get_cache_stats,
                get_or_create_fitter,
            )

            # Two misses (different keys), then one hit
            get_or_create_fitter(10, 2, phi_angles=None, scaling_mode="auto")
            get_or_create_fitter(20, 2, phi_angles=None, scaling_mode="auto")
            get_or_create_fitter(10, 2, phi_angles=None, scaling_mode="auto")

            stats = get_cache_stats()

        assert stats["misses"] == 2
        assert stats["hits"] == 1
        assert stats["size"] == 2

    def test_clear_model_cache(self) -> None:
        """clear_model_cache must empty the cache and reset stats."""
        with patch(
            "heterodyne.optimization.nlsq.adapter.CurveFit",
            return_value=MagicMock(),
        ):
            from heterodyne.optimization.nlsq.adapter import (
                clear_model_cache,
                get_cache_stats,
                get_or_create_fitter,
            )

            get_or_create_fitter(10, 2, phi_angles=None, scaling_mode="auto")
            get_or_create_fitter(10, 2, phi_angles=None, scaling_mode="auto")  # hit

            clear_model_cache()
            stats = get_cache_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_cache_key_includes_phi_and_scaling(self) -> None:
        """Different phi_angles / scaling_mode must produce separate cache entries."""
        with patch(
            "heterodyne.optimization.nlsq.adapter.CurveFit",
            side_effect=lambda flength: MagicMock(),
        ):
            from heterodyne.optimization.nlsq.adapter import (
                _model_cache,
                get_or_create_fitter,
            )

            get_or_create_fitter(50, 3, phi_angles=(0.0,), scaling_mode="auto")
            get_or_create_fitter(50, 3, phi_angles=(0.0, 1.0), scaling_mode="auto")
            get_or_create_fitter(50, 3, phi_angles=(0.0,), scaling_mode="individual")

        assert len(_model_cache) == 3, (
            "Each (phi_angles, scaling_mode) combination must be a distinct cache key"
        )


# ---------------------------------------------------------------------------
# NLSQAdapter
# ---------------------------------------------------------------------------


class TestNLSQAdapter:
    """Tests for NLSQAdapter (primary, JAX-traced adapter)."""

    def test_nlsq_adapter_name(self) -> None:
        """name property must return 'nlsq.CurveFit'."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
        assert adapter.name == "nlsq.CurveFit"

    def test_nlsq_adapter_supports_bounds(self) -> None:
        """supports_bounds() must return True."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
        assert adapter.supports_bounds() is True

    def test_nlsq_adapter_supports_jacobian(self) -> None:
        """supports_jacobian() must return True."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
        assert adapter.supports_jacobian() is True

    def test_nlsq_adapter_fit_returns_result(self) -> None:
        """fit() must return an NLSQResult without delegating to scipy."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter
        from heterodyne.optimization.nlsq.results import NLSQResult

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = _make_nlsq_tuple_result()

        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
                return_value=(mock_fitter, False),
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.build_result_from_nlsq",
                return_value=NLSQResult(
                    parameters=np.array([1.1, 1.2, 1.3]),
                    parameter_names=PARAM_NAMES,
                    success=True,
                    message="ok",
                ),
            ) as mock_build,
        ):
            adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
            result = adapter.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        assert isinstance(result, NLSQResult)
        mock_build.assert_called_once()

    def test_nlsq_adapter_fit_no_scipy(self) -> None:
        """fit() must not import or call scipy.optimize.least_squares."""
        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = _make_nlsq_tuple_result()

        # If scipy is called the mock below will raise
        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
                return_value=(mock_fitter, False),
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.build_result_from_nlsq",
                return_value=MagicMock(
                    spec=[
                        "parameters",
                        "parameter_names",
                        "success",
                        "message",
                        "uncertainties",
                        "covariance",
                        "final_cost",
                        "reduced_chi_squared",
                        "n_iterations",
                        "n_function_evals",
                        "convergence_reason",
                        "residuals",
                        "jacobian",
                        "wall_time_seconds",
                        "metadata",
                    ]
                ),
            ),
        ):
            from heterodyne.optimization.nlsq.adapter import NLSQAdapter

            # Ensure scipy.optimize.least_squares is never touched
            scipy_mock = MagicMock(side_effect=AssertionError("scipy called in NLSQAdapter.fit"))
            with patch.dict(
                sys.modules, {"scipy.optimize": MagicMock(least_squares=scipy_mock)}
            ):
                adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
                adapter.fit(
                    residual_fn=_make_residual_fn(),
                    initial_params=_make_initial_params(),
                    bounds=_make_bounds(),
                    config=_make_config(),
                )

        scipy_mock.assert_not_called()

    def test_nlsq_adapter_fit_convergence_nan(self) -> None:
        """fit() must return success=False when fitted params contain NaN."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter
        from heterodyne.optimization.nlsq.results import NLSQResult

        nan_params = np.full(N_PARAMS, np.nan)
        mock_fitter = MagicMock()
        mock_fitter.curve_fit.return_value = (nan_params, None)

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
            result = adapter.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        assert isinstance(result, NLSQResult)
        assert result.success is False

    def test_nlsq_adapter_fit_handles_exception(self) -> None:
        """fit() must return a failed NLSQResult when the fitter raises."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter
        from heterodyne.optimization.nlsq.results import NLSQResult

        mock_fitter = MagicMock()
        mock_fitter.curve_fit.side_effect = RuntimeError("backend crashed")

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            adapter = NLSQAdapter(parameter_names=PARAM_NAMES)
            result = adapter.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        assert isinstance(result, NLSQResult)
        assert result.success is False
        assert "backend crashed" in result.message


# ---------------------------------------------------------------------------
# NLSQWrapper
# ---------------------------------------------------------------------------


class TestNLSQWrapper:
    """Tests for NLSQWrapper (stable fallback with strategy routing)."""

    def test_nlsq_wrapper_name(self) -> None:
        """name property must return 'nlsq.NLSQWrapper'."""
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        assert wrapper.name == "nlsq.NLSQWrapper"

    def test_nlsq_wrapper_supports_bounds(self) -> None:
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        assert wrapper.supports_bounds() is True

    def test_nlsq_wrapper_supports_jacobian(self) -> None:
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        assert wrapper.supports_jacobian() is True

    def test_nlsq_wrapper_standard_strategy(self) -> None:
        """Small data should route to curve_fit (STANDARD strategy)."""
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper
        from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
        from heterodyne.optimization.nlsq.results import NLSQResult

        mock_curve_fit = MagicMock(return_value=_make_nlsq_tuple_result())

        standard_decision = StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=16.0,
            peak_memory_gb=0.001,
            reason="fits in memory",
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=standard_decision,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.curve_fit",
                mock_curve_fit,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.build_result_from_nlsq",
                return_value=NLSQResult(
                    parameters=np.array([1.1, 1.2, 1.3]),
                    parameter_names=PARAM_NAMES,
                    success=True,
                    message="ok",
                ),
            ),
        ):
            wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
            result = wrapper.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        mock_curve_fit.assert_called_once()
        assert result.success is True

    def test_nlsq_wrapper_large_strategy(self) -> None:
        """Large data should route to curve_fit_large (LARGE strategy)."""
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper
        from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
        from heterodyne.optimization.nlsq.results import NLSQResult

        mock_curve_fit_large = MagicMock(return_value=_make_nlsq_tuple_result())

        large_decision = StrategyDecision(
            strategy=NLSQStrategy.LARGE,
            threshold_gb=16.0,
            peak_memory_gb=20.0,
            reason="jacobian exceeds threshold",
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=large_decision,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.curve_fit_large",
                mock_curve_fit_large,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.build_result_from_nlsq",
                return_value=NLSQResult(
                    parameters=np.array([1.1, 1.2, 1.3]),
                    parameter_names=PARAM_NAMES,
                    success=True,
                    message="ok",
                ),
            ),
        ):
            wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
            result = wrapper.fit(
                residual_fn=_make_residual_fn(n_data=500_000),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        mock_curve_fit_large.assert_called_once()
        assert result.success is True

    def test_nlsq_wrapper_fallback_on_error(self) -> None:
        """When the first strategy fails its retries, fallback to next tier."""
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper
        from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
        from heterodyne.optimization.nlsq.results import NLSQResult

        # STREAMING fails → should fall back to LARGE then STANDARD
        streaming_decision = StrategyDecision(
            strategy=NLSQStrategy.STREAMING,
            threshold_gb=1.0,
            peak_memory_gb=100.0,
            reason="extreme scale",
        )

        mock_curve_fit = MagicMock(return_value=_make_nlsq_tuple_result())

        # AdaptiveHybridStreamingOptimizer is unavailable or raises
        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=streaming_decision,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.STREAMING_AVAILABLE",
                False,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.curve_fit_large",
                side_effect=RuntimeError("large also fails"),
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.curve_fit",
                mock_curve_fit,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.build_result_from_nlsq",
                return_value=NLSQResult(
                    parameters=np.array([1.1, 1.2, 1.3]),
                    parameter_names=PARAM_NAMES,
                    success=True,
                    message="recovered via STANDARD",
                ),
            ),
        ):
            wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, enable_recovery=True)
            result = wrapper.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        # Fallback must have reached curve_fit (STANDARD tier)
        mock_curve_fit.assert_called()
        assert result.success is True

    def test_nlsq_wrapper_all_strategies_fail_returns_failure(self) -> None:
        """When all fallback tiers fail, return a failed NLSQResult."""
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper
        from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
        from heterodyne.optimization.nlsq.results import NLSQResult

        standard_decision = StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=16.0,
            peak_memory_gb=0.001,
            reason="fits",
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=standard_decision,
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.curve_fit",
                side_effect=RuntimeError("everything failed"),
            ),
        ):
            wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, enable_recovery=False, max_retries=1)
            result = wrapper.fit(
                residual_fn=_make_residual_fn(),
                initial_params=_make_initial_params(),
                bounds=_make_bounds(),
                config=_make_config(),
            )

        assert isinstance(result, NLSQResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# Absence of ScipyNLSQAdapter
# ---------------------------------------------------------------------------


class TestNoScipyAdapter:
    """Verify ScipyNLSQAdapter no longer exists in the adapter module."""

    def test_no_scipy_adapter(self) -> None:
        """ScipyNLSQAdapter must NOT be importable from adapter module."""
        import importlib

        adapter_module = importlib.import_module("heterodyne.optimization.nlsq.adapter")
        assert not hasattr(adapter_module, "ScipyNLSQAdapter"), (
            "ScipyNLSQAdapter must be deleted from adapter.py "
            "(it delegates to scipy which is forbidden in the NLSQ path)"
        )

    def test_no_scipy_optimize_import(self) -> None:
        """adapter.py must not import scipy.optimize at module level."""
        import ast
        import importlib
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec("heterodyne.optimization.nlsq.adapter")
        assert spec is not None and spec.origin is not None
        source = pathlib.Path(spec.origin).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            # Catch: import scipy.optimize
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "scipy.optimize" not in alias.name, (
                        "adapter.py must not have a top-level 'import scipy.optimize'"
                    )
            # Catch: from scipy.optimize import ...
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "scipy.optimize" not in module, (
                    "adapter.py must not have a top-level 'from scipy.optimize import ...'"
                )
