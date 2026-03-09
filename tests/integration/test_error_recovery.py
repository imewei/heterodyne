"""Integration tests for error recovery and graceful degradation.

Tests that the NLSQ optimization pipeline handles pathological inputs,
numerical failures, and strategy exhaustion without crashing.  Each test
class covers a specific layer of the recovery stack:

- NLSQAdapter  — catches ValueError/RuntimeError, returns failed NLSQResult
- Fallback chain — cascades through strategies on MemoryError / RuntimeError
- NLSQWrapper  — tier-based cascade with enable_recovery flag
- Fitting solvers — singular matrix / constant data fallback
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.core.fitting import (
    solve_least_squares_general_jax,
    solve_least_squares_jax,
)
from heterodyne.optimization.nlsq.adapter import NLSQAdapter, NLSQWrapper
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.fallback_chain import (
    OptimizationStrategy,
    execute_optimization_with_fallback,
    get_fallback_strategy,
)
from heterodyne.optimization.nlsq.memory import NLSQStrategy
from heterodyne.optimization.nlsq.results import NLSQResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PARAM_NAMES: list[str] = ["a", "b", "c"]


def _make_config() -> NLSQConfig:
    """Return a minimal NLSQConfig for testing."""
    return NLSQConfig()


def _simple_residual(params: np.ndarray) -> np.ndarray:
    """Residual function: r = params - target."""
    target = np.array([1.0, 2.0, 3.0])
    return params - target


def _simple_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([-10.0, -10.0, -10.0])
    upper = np.array([10.0, 10.0, 10.0])
    return lower, upper


# ---------------------------------------------------------------------------
# TestNLSQAdapterErrorRecovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNLSQAdapterErrorRecovery:
    """NLSQAdapter.fit returns failure results instead of raising."""

    def test_nan_initial_params_handled(self) -> None:
        """NaN initial params produce a failure result, not an unhandled crash."""
        adapter = NLSQAdapter(parameter_names=_PARAM_NAMES)
        initial = np.array([np.nan, np.nan, np.nan])
        bounds = _simple_bounds()
        config = _make_config()

        result = adapter.fit(_simple_residual, initial, bounds, config)

        assert isinstance(result, NLSQResult)
        # NaN params are clipped to bounds, so the fit may or may not succeed,
        # but it must not raise.  If bounds clipping produces a valid start
        # the optimizer may converge; either way we get a result object.
        assert result.parameters is not None
        assert len(result.parameter_names) == len(_PARAM_NAMES)

    def test_inf_residuals_handled(self) -> None:
        """A residual function returning inf does not crash the adapter."""

        def _inf_residual(params: np.ndarray) -> np.ndarray:
            return np.full(3, np.inf)

        adapter = NLSQAdapter(parameter_names=_PARAM_NAMES)
        initial = np.array([1.0, 2.0, 3.0])
        bounds = _simple_bounds()
        config = _make_config()

        result = adapter.fit(_inf_residual, initial, bounds, config)

        assert isinstance(result, NLSQResult)
        # The optimizer likely fails to converge on infinite residuals.
        assert result.parameters is not None

    def test_zero_bounds_range_handled(self) -> None:
        """Bounds where lower == upper for a parameter are handled gracefully."""
        adapter = NLSQAdapter(parameter_names=_PARAM_NAMES)
        initial = np.array([1.0, 2.0, 3.0])
        lower = np.array([1.0, -10.0, 3.0])  # a and c are pinned
        upper = np.array([1.0, 10.0, 3.0])
        config = _make_config()

        result = adapter.fit(_simple_residual, initial, (lower, upper), config)

        assert isinstance(result, NLSQResult)
        assert result.parameters is not None
        assert len(result.parameter_names) == len(_PARAM_NAMES)


# ---------------------------------------------------------------------------
# TestFallbackChainRecovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFallbackChainRecovery:
    """Tests for get_fallback_strategy and execute_optimization_with_fallback."""

    def test_memory_error_triggers_fallback(self) -> None:
        """MemoryError from LARGE strategy triggers fallback to STREAMING."""
        error = MemoryError("unable to allocate array")
        next_strategy = get_fallback_strategy(
            OptimizationStrategy.LARGE, error=error,
        )
        assert next_strategy == OptimizationStrategy.STREAMING

    def test_runtime_error_triggers_fallback(self) -> None:
        """RuntimeError from STREAMING triggers step to LARGE."""
        error = RuntimeError("solver diverged")
        next_strategy = get_fallback_strategy(
            OptimizationStrategy.STREAMING, error=error,
        )
        assert next_strategy == OptimizationStrategy.LARGE

    def test_all_strategies_fail_raises(self) -> None:
        """When all strategies fail, RuntimeError is raised with informative msg."""
        # Build a mock model whose param_manager exposes the right API
        mock_model = MagicMock()
        mock_model.param_manager.n_varying = 3
        mock_model.param_manager.varying_names = _PARAM_NAMES
        mock_model.param_manager.get_initial_values.return_value = [1.0, 2.0, 3.0]
        mock_model.param_manager.get_bounds.return_value = (
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
        )

        # Force every _run_strategy call to raise
        with patch(
            "heterodyne.optimization.nlsq.fallback_chain._run_strategy",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="All optimization strategies failed"):
                execute_optimization_with_fallback(
                    model=mock_model,
                    c2_data=np.zeros((10, 10)),
                    phi_angle=0.0,
                    config=_make_config(),
                    start_strategy=OptimizationStrategy.STANDARD,
                )

    def test_fallback_chain_exhausted_returns_none(self) -> None:
        """get_fallback_strategy returns None when STANDARD (last tier) fails."""
        error = RuntimeError("generic error")
        next_strategy = get_fallback_strategy(
            OptimizationStrategy.STANDARD, error=error,
        )
        assert next_strategy is None


# ---------------------------------------------------------------------------
# TestNLSQWrapperRecovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNLSQWrapperRecovery:
    """NLSQWrapper returns NLSQResult with success=False when all tiers fail."""

    def test_wrapper_returns_failure_not_crash(self) -> None:
        """When all tiers fail, wrapper returns NLSQResult(success=False)."""
        wrapper = NLSQWrapper(
            parameter_names=_PARAM_NAMES,
            enable_recovery=True,
            max_retries=1,
        )
        initial = np.array([1.0, 2.0, 3.0])
        bounds = _simple_bounds()
        config = _make_config()

        # Force _call_tier to always raise so every tier is exhausted
        with patch.object(
            wrapper,
            "_call_tier",
            side_effect=RuntimeError("tier failure"),
        ):
            result = wrapper.fit(_simple_residual, initial, bounds, config)

        assert isinstance(result, NLSQResult)
        assert result.success is False
        assert "failed" in result.message.lower() or "tier" in result.message.lower()

    def test_wrapper_recovery_flag_controls_cascade(self) -> None:
        """With enable_recovery=False, only the first tier is attempted."""
        call_count = 0

        def _counting_call_tier(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        wrapper = NLSQWrapper(
            parameter_names=_PARAM_NAMES,
            enable_recovery=False,
            max_retries=1,
        )
        initial = np.array([1.0, 2.0, 3.0])
        bounds = _simple_bounds()
        config = _make_config()

        with patch.object(wrapper, "_call_tier", side_effect=_counting_call_tier):
            result = wrapper.fit(_simple_residual, initial, bounds, config)

        assert isinstance(result, NLSQResult)
        assert result.success is False
        # Only the initial tier should have been attempted (max_retries=1)
        assert call_count == 1


# ---------------------------------------------------------------------------
# TestFittingNumericalRecovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFittingNumericalRecovery:
    """JAX/NumPy solvers in core.fitting handle degenerate inputs gracefully."""

    def test_singular_covariance_fallback(self) -> None:
        """When the design matrix produces a singular Gram matrix, the solver
        returns a finite result via the SVD fallback path."""
        # All-identical rows produce a rank-1 Gram matrix
        design = np.ones((20, 3), dtype=np.float64)
        target = np.arange(20, dtype=np.float64)

        result = solve_least_squares_general_jax(design, target)

        result_np = np.asarray(result)
        assert result_np.shape == (3,)
        assert np.all(np.isfinite(result_np))

    def test_constant_data_doesnt_crash(self) -> None:
        """Fitting constant (zero-variance) data returns gracefully.

        When theory == constant the determinant of the 2x2 system is zero;
        the solver should fall back to default contrast=1, offset=0 (or 1).
        """
        n_angles = 2
        n_data = 50
        theory = np.ones((n_angles, n_data), dtype=np.float64)
        exp = np.ones((n_angles, n_data), dtype=np.float64) * 5.0

        contrast, offset = solve_least_squares_jax(theory, exp)

        contrast_np = np.asarray(contrast)
        offset_np = np.asarray(offset)
        assert contrast_np.shape == (n_angles,)
        assert offset_np.shape == (n_angles,)
        assert np.all(np.isfinite(contrast_np))
        assert np.all(np.isfinite(offset_np))

    def test_zero_theory_doesnt_crash(self) -> None:
        """All-zero theory data does not crash the batch solver."""
        n_angles = 1
        n_data = 30
        theory = np.zeros((n_angles, n_data), dtype=np.float64)
        exp = np.random.default_rng(42).normal(size=(n_angles, n_data))

        contrast, offset = solve_least_squares_jax(theory, exp)

        contrast_np = np.asarray(contrast)
        offset_np = np.asarray(offset)
        assert np.all(np.isfinite(contrast_np))
        assert np.all(np.isfinite(offset_np))
