"""Unit tests for NLSQWrapper.

Tests cover:
- Success on first attempt (simple quadratic residual via ScipyNLSQAdapter).
- Retry when the adapter fails on attempt 0 then succeeds.
- Max retries exhausted — returns best result seen so far.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.wrapper import NLSQWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAM_NAMES = ["a", "b"]

DEFAULT_BOUNDS: tuple[np.ndarray, np.ndarray] = (
    np.array([-10.0, -10.0]),
    np.array([10.0, 10.0]),
)

DEFAULT_CONFIG = NLSQConfig(max_iterations=50, verbose=0)


def _make_result(
    *,
    success: bool,
    final_cost: float | None,
    params: np.ndarray | None = None,
) -> NLSQResult:
    """Create a minimal NLSQResult for testing."""
    if params is None:
        params = np.zeros(2)
    return NLSQResult(
        parameters=params,
        parameter_names=PARAM_NAMES,
        success=success,
        message="ok" if success else "failed",
        final_cost=final_cost,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperSuccessFirstAttempt:
    """Wrapper returns immediately on first-attempt success."""

    def test_quadratic_residual_converges(self) -> None:
        """Wrapper converges on a trivial least-squares problem.

        Residual: f(x) = x - target, so the optimum is x = target.
        """
        target = np.array([3.0, -2.0])

        def residual_fn(params: np.ndarray) -> np.ndarray:
            return params - target

        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            use_jax=False,
            max_retries=3,
        )
        result = wrapper.fit(
            residual_fn=residual_fn,
            initial_params=np.array([0.0, 0.0]),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        assert result.success, f"Expected success, got: {result.message}"
        np.testing.assert_allclose(result.parameters, target, atol=1e-4)

    def test_adapter_called_once_on_success(self) -> None:
        """When the adapter succeeds immediately, fit is only called once."""
        success_result = _make_result(success=True, final_cost=0.01)
        mock_adapter = MagicMock()
        mock_adapter.fit.return_value = success_result
        mock_adapter.name = "mock"

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, max_retries=3)
        wrapper._adapter = mock_adapter

        wrapper.fit(
            residual_fn=lambda p: p,
            initial_params=np.zeros(2),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        assert mock_adapter.fit.call_count == 1


@pytest.mark.unit
class TestNLSQWrapperRetry:
    """Wrapper retries after failure and returns first success."""

    def test_retry_succeeds_on_second_attempt(self) -> None:
        """When attempt 0 fails and attempt 1 succeeds, result is attempt 1's."""
        fail_result = _make_result(success=False, final_cost=999.0)
        success_result = _make_result(
            success=True,
            final_cost=0.5,
            params=np.array([1.0, 2.0]),
        )

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = [fail_result, success_result]
        mock_adapter.name = "mock"

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, max_retries=3)
        wrapper._adapter = mock_adapter

        result = wrapper.fit(
            residual_fn=lambda p: p,
            initial_params=np.zeros(2),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        assert result.success
        assert result.final_cost == pytest.approx(0.5)
        assert mock_adapter.fit.call_count == 2

    def test_perturbed_params_passed_on_retry(self) -> None:
        """Params passed to the second attempt differ from the first."""
        fail_result = _make_result(success=False, final_cost=1.0)
        success_result = _make_result(success=True, final_cost=0.1)

        call_params: list[np.ndarray] = []

        def fake_fit(
            residual_fn: Callable,
            initial_params: np.ndarray,
            bounds: tuple[np.ndarray, np.ndarray],
            config: NLSQConfig,
            jacobian_fn: Callable | None = None,
        ) -> NLSQResult:
            call_params.append(initial_params.copy())
            return fail_result if len(call_params) == 1 else success_result

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = fake_fit
        mock_adapter.name = "mock"

        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            max_retries=3,
            perturb_scale=0.1,
        )
        wrapper._adapter = mock_adapter

        wrapper.fit(
            residual_fn=lambda p: p,
            initial_params=np.array([1.0, 1.0]),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        assert len(call_params) == 2
        # First attempt: original params (clipped).
        np.testing.assert_allclose(call_params[0], [1.0, 1.0])
        # Second attempt: perturbed — must differ.
        assert not np.allclose(call_params[0], call_params[1]), (
            "Second attempt should use perturbed parameters"
        )


@pytest.mark.unit
class TestNLSQWrapperMaxRetriesExhausted:
    """When all attempts fail, wrapper returns the best result seen."""

    def test_returns_best_when_all_fail(self) -> None:
        """Best result (lowest cost) is returned after all retries fail."""
        results = [
            _make_result(success=False, final_cost=50.0),
            _make_result(success=False, final_cost=10.0),  # best
            _make_result(success=False, final_cost=30.0),
            _make_result(success=False, final_cost=20.0),
        ]

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = results
        mock_adapter.name = "mock"

        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            max_retries=3,  # 4 total attempts
        )
        wrapper._adapter = mock_adapter

        result = wrapper.fit(
            residual_fn=lambda p: p,
            initial_params=np.zeros(2),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        # All 4 attempts should be exhausted
        assert mock_adapter.fit.call_count == 4
        # The result with cost 10.0 is the best
        assert result.final_cost == pytest.approx(10.0)
        assert not result.success

    def test_returns_result_when_cost_is_none(self) -> None:
        """A result with final_cost=None is treated as infinite cost."""
        none_cost = _make_result(success=False, final_cost=None)
        finite_cost = _make_result(success=False, final_cost=999.0)

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = [none_cost, finite_cost]
        mock_adapter.name = "mock"

        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, max_retries=1)
        wrapper._adapter = mock_adapter

        result = wrapper.fit(
            residual_fn=lambda p: p,
            initial_params=np.zeros(2),
            bounds=DEFAULT_BOUNDS,
            config=DEFAULT_CONFIG,
        )

        # finite_cost wins over None-cost
        assert result.final_cost == pytest.approx(999.0)

    def test_total_attempts_equals_max_retries_plus_one(self) -> None:
        """Total number of adapter calls equals max_retries + 1."""
        for max_retries in (0, 1, 3, 5):
            fail_result = _make_result(success=False, final_cost=1.0)
            mock_adapter = MagicMock()
            mock_adapter.fit.return_value = fail_result
            mock_adapter.name = "mock"

            wrapper = NLSQWrapper(
                parameter_names=PARAM_NAMES,
                max_retries=max_retries,
            )
            wrapper._adapter = mock_adapter

            wrapper.fit(
                residual_fn=lambda p: p,
                initial_params=np.zeros(2),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

            assert mock_adapter.fit.call_count == max_retries + 1, (
                f"Expected {max_retries + 1} calls for max_retries={max_retries}, "
                f"got {mock_adapter.fit.call_count}"
            )


@pytest.mark.unit
class TestNLSQWrapperPerturbation:
    """Unit tests for the internal _perturbed_params helper."""

    def test_attempt_zero_returns_clipped_original(self) -> None:
        """Attempt 0 must return the original params (clipped to bounds)."""
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        params = np.array([5.0, -3.0])
        lower = np.array([-10.0, -10.0])
        upper = np.array([10.0, 10.0])

        result = wrapper._perturbed_params(params, lower, upper, attempt=0)
        np.testing.assert_array_equal(result, params)

    def test_attempt_zero_clips_out_of_bounds(self) -> None:
        """Params outside bounds are clipped on attempt 0."""
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        params = np.array([15.0, -15.0])
        lower = np.array([-10.0, -10.0])
        upper = np.array([10.0, 10.0])

        result = wrapper._perturbed_params(params, lower, upper, attempt=0)
        np.testing.assert_array_equal(result, np.array([10.0, -10.0]))

    def test_perturbed_params_stay_within_bounds(self) -> None:
        """Perturbed params for any attempt must be within bounds."""
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, perturb_scale=2.0)
        params = np.array([0.0, 0.0])
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])

        for attempt in range(1, 6):
            result = wrapper._perturbed_params(params, lower, upper, attempt=attempt)
            assert np.all(result >= lower), f"attempt {attempt}: below lower bound"
            assert np.all(result <= upper), f"attempt {attempt}: above upper bound"

    def test_reproducibility_same_seed(self) -> None:
        """Same attempt index always produces the same perturbation."""
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        params = np.array([1.0, 2.0])
        lower = np.array([-10.0, -10.0])
        upper = np.array([10.0, 10.0])

        r1 = wrapper._perturbed_params(params, lower, upper, attempt=2)
        r2 = wrapper._perturbed_params(params, lower, upper, attempt=2)
        np.testing.assert_array_equal(r1, r2)


@pytest.mark.unit
class TestNLSQWrapperUpdateBest:
    """Unit tests for the static _update_best helper."""

    def test_none_best_returns_candidate(self) -> None:
        candidate = _make_result(success=True, final_cost=1.0)
        assert NLSQWrapper._update_best(None, candidate) is candidate

    def test_lower_cost_candidate_wins(self) -> None:
        best = _make_result(success=False, final_cost=5.0)
        candidate = _make_result(success=False, final_cost=2.0)
        assert NLSQWrapper._update_best(best, candidate) is candidate

    def test_higher_cost_candidate_loses(self) -> None:
        best = _make_result(success=True, final_cost=1.0)
        candidate = _make_result(success=False, final_cost=10.0)
        assert NLSQWrapper._update_best(best, candidate) is best

    def test_none_cost_treated_as_infinity(self) -> None:
        best = _make_result(success=False, final_cost=100.0)
        candidate = _make_result(success=False, final_cost=None)
        assert NLSQWrapper._update_best(best, candidate) is best
