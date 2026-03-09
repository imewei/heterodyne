"""Unit tests for NLSQWrapper (memory-aware fallback adapter).

Tests cover:
- Success on first attempt (simple residual).
- Tier fallback when one tier fails.
- All tiers exhausted returns failure result.
- Memory-based strategy selection.
- _build_tier_list() ordering and filtering.
- enable_large_dataset / enable_recovery flags.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.optimization.nlsq.adapter import NLSQWrapper
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision
from heterodyne.optimization.nlsq.results import NLSQResult

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
    params: np.ndarray | None = None,
    final_cost: float | None = None,
) -> NLSQResult:
    if params is None:
        params = np.array([1.5, -0.5])
    return NLSQResult(
        parameters=params,
        parameter_names=PARAM_NAMES,
        success=success,
        message="ok" if success else "failed",
        final_cost=final_cost,
    )


def _make_strategy_decision(strategy: NLSQStrategy) -> StrategyDecision:
    return StrategyDecision(
        strategy=strategy,
        threshold_gb=16.0,
        peak_memory_gb=1.0,
        reason="test",
    )


# ---------------------------------------------------------------------------
# Tests: success on first attempt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperSuccess:
    """Wrapper returns on first successful tier."""

    def test_simple_residual_succeeds(self) -> None:
        """STANDARD tier returns success when _call_tier succeeds."""
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, max_retries=1)

        target = np.array([3.0, -2.0])
        mock_raw = (target, np.eye(2) * 0.01)

        decision = _make_strategy_decision(NLSQStrategy.STANDARD)

        with (
            patch.object(wrapper, "_call_tier", return_value=mock_raw),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params - target

            result = wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.array([0.0, 0.0]),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        assert result.success, f"Expected success, got: {result.message}"
        np.testing.assert_allclose(result.parameters, target, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: tier fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperTierFallback:
    """Wrapper falls back to next tier when current tier fails."""

    def test_fallback_to_next_tier_on_failure(self) -> None:
        """When _call_tier raises on STREAMING, falls back to LARGE/STANDARD."""
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_large_dataset=True,
            enable_recovery=True,
            max_retries=1,
        )

        call_tiers: list[NLSQStrategy] = []

        original_call_tier = wrapper._call_tier

        def mock_call_tier(tier: NLSQStrategy, **kwargs: object) -> object:
            call_tiers.append(tier)
            if tier == NLSQStrategy.STREAMING:
                raise RuntimeError("Streaming unavailable")
            return original_call_tier(tier=tier, **kwargs)

        decision = _make_strategy_decision(NLSQStrategy.STREAMING)

        with (
            patch.object(wrapper, "_call_tier", side_effect=mock_call_tier),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):
            target = np.array([1.0, 2.0])

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params - target

            result = wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.array([0.0, 0.0]),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        # STREAMING failed, should have tried LARGE or STANDARD next
        assert NLSQStrategy.STREAMING in call_tiers
        assert len(call_tiers) >= 2
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: all tiers exhausted
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperAllTiersExhausted:
    """When every tier fails, wrapper returns a failure result."""

    def test_all_tiers_fail_returns_failure(self) -> None:
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_recovery=True,
            max_retries=1,
        )

        decision = _make_strategy_decision(NLSQStrategy.STANDARD)

        with (
            patch.object(
                wrapper,
                "_call_tier",
                side_effect=RuntimeError("always fails"),
            ),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params

            result = wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.zeros(2),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        assert result.success is False
        assert "failed" in result.message.lower()


# ---------------------------------------------------------------------------
# Tests: memory-based strategy selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperStrategySelection:
    """select_nlsq_strategy determines the initial tier."""

    def test_strategy_decision_is_used(self) -> None:
        """The wrapper uses the strategy from select_nlsq_strategy."""
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_large_dataset=True,
            enable_recovery=False,
            max_retries=1,
        )

        call_tiers: list[NLSQStrategy] = []

        def mock_call_tier(tier: NLSQStrategy, **kwargs: object) -> object:
            call_tiers.append(tier)
            raise RuntimeError("fail")

        decision = _make_strategy_decision(NLSQStrategy.LARGE)

        with (
            patch.object(wrapper, "_call_tier", side_effect=mock_call_tier),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params

            wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.zeros(2),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        # First tier attempted should be LARGE (as selected)
        assert call_tiers[0] == NLSQStrategy.LARGE


# ---------------------------------------------------------------------------
# Tests: _build_tier_list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildTierList:
    """Unit tests for _build_tier_list ordering and filtering."""

    def test_streaming_start_gives_full_cascade(self) -> None:
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_large_dataset=True,
        )
        tiers = wrapper._build_tier_list(NLSQStrategy.STREAMING)
        assert tiers == [
            NLSQStrategy.STREAMING,
            NLSQStrategy.LARGE,
            NLSQStrategy.STANDARD,
        ]

    def test_large_start_skips_streaming(self) -> None:
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_large_dataset=True,
        )
        tiers = wrapper._build_tier_list(NLSQStrategy.LARGE)
        assert tiers == [NLSQStrategy.LARGE, NLSQStrategy.STANDARD]

    def test_standard_start_is_single(self) -> None:
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES)
        tiers = wrapper._build_tier_list(NLSQStrategy.STANDARD)
        assert tiers == [NLSQStrategy.STANDARD]

    def test_enable_large_dataset_false_drops_large(self) -> None:
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_large_dataset=False,
        )
        tiers = wrapper._build_tier_list(NLSQStrategy.STREAMING)
        assert NLSQStrategy.LARGE not in tiers
        assert NLSQStrategy.STREAMING in tiers
        assert NLSQStrategy.STANDARD in tiers

    def test_enable_recovery_false_stops_after_first_tier(self) -> None:
        """With enable_recovery=False, only the first tier is attempted."""
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_recovery=False,
            max_retries=1,
        )

        call_tiers: list[NLSQStrategy] = []

        def mock_call_tier(tier: NLSQStrategy, **kwargs: object) -> object:
            call_tiers.append(tier)
            raise RuntimeError("fail")

        decision = _make_strategy_decision(NLSQStrategy.STREAMING)

        with (
            patch.object(wrapper, "_call_tier", side_effect=mock_call_tier),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params

            wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.zeros(2),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        # Only tried one tier (STREAMING), did not cascade
        assert call_tiers == [NLSQStrategy.STREAMING]


# ---------------------------------------------------------------------------
# Tests: max_retries enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNLSQWrapperMaxRetries:
    """max_retries controls per-tier retry count."""

    def test_max_retries_clamped_to_at_least_one(self) -> None:
        wrapper = NLSQWrapper(parameter_names=PARAM_NAMES, max_retries=0)
        assert wrapper._max_retries == 1

    def test_retries_exhausted_per_tier(self) -> None:
        wrapper = NLSQWrapper(
            parameter_names=PARAM_NAMES,
            enable_recovery=False,
            max_retries=3,
        )

        call_count = 0

        def mock_call_tier(tier: NLSQStrategy, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        decision = _make_strategy_decision(NLSQStrategy.STANDARD)

        with (
            patch.object(wrapper, "_call_tier", side_effect=mock_call_tier),
            patch(
                "heterodyne.optimization.nlsq.adapter.select_nlsq_strategy",
                return_value=decision,
            ),
        ):

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return params

            wrapper.fit(
                residual_fn=residual_fn,
                initial_params=np.zeros(2),
                bounds=DEFAULT_BOUNDS,
                config=DEFAULT_CONFIG,
            )

        # STANDARD tier retried max_retries=3 times
        assert call_count == 3
