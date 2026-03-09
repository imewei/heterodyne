"""Tests for memory-based NLSQ strategy selection.

Covers NLSQStrategy enum, StrategyDecision immutability, peak memory
estimation, and strategy routing for small/medium/huge datasets.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from heterodyne.optimization.nlsq.memory import (
    DEFAULT_MEMORY_FRACTION,
    FALLBACK_THRESHOLD_GB,
    NLSQStrategy,
    StrategyDecision,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    select_nlsq_strategy,
)

# ---------------------------------------------------------------------------
# NLSQStrategy enum
# ---------------------------------------------------------------------------


class TestNLSQStrategyEnum:
    def test_strategy_enum_values(self) -> None:
        assert NLSQStrategy.STANDARD.value == "standard"
        assert NLSQStrategy.LARGE.value == "large"
        assert NLSQStrategy.STREAMING.value == "streaming"

    def test_enum_has_exactly_three_members(self) -> None:
        assert len(NLSQStrategy) == 3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_memory_fraction(self) -> None:
        assert DEFAULT_MEMORY_FRACTION == 0.75

    def test_fallback_threshold_gb(self) -> None:
        assert FALLBACK_THRESHOLD_GB == 16.0


# ---------------------------------------------------------------------------
# estimate_peak_memory_gb
# ---------------------------------------------------------------------------


class TestEstimatePeakMemory:
    def test_estimate_peak_memory_small(self) -> None:
        """1000 points, 16 params should use very little memory."""
        gb = estimate_peak_memory_gb(1000, 16)
        # 1000 * 16 * 8 * 6.5 = 832_000 bytes ~ 0.00077 GB
        assert gb < 0.01
        assert gb > 0.0

    def test_estimate_peak_memory_large(self) -> None:
        """10M points, 16 params should require significant memory."""
        gb = estimate_peak_memory_gb(10_000_000, 16)
        # 10M * 16 * 8 * 6.5 = 8.32e9 bytes ~ 7.75 GB
        assert gb > 1.0
        assert gb < 20.0

    def test_zero_params_returns_zero(self) -> None:
        gb = estimate_peak_memory_gb(1_000_000, 0)
        assert gb == 0.0

    def test_scales_linearly_with_points(self) -> None:
        gb1 = estimate_peak_memory_gb(1000, 16)
        gb2 = estimate_peak_memory_gb(2000, 16)
        assert gb2 == pytest.approx(2.0 * gb1)


# ---------------------------------------------------------------------------
# detect_total_system_memory
# ---------------------------------------------------------------------------


class TestDetectSystemMemory:
    def test_returns_float_or_none(self) -> None:
        result = detect_total_system_memory()
        assert result is None or isinstance(result, float)

    def test_psutil_fallback_to_sysconf(self) -> None:
        """When psutil is unavailable, should still detect via sysconf."""
        with patch.dict("sys.modules", {"psutil": None}):
            result = detect_total_system_memory()
            # On Linux this should succeed via sysconf
            assert result is None or result > 0.0

    def test_complete_failure_returns_none(self) -> None:
        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("os.sysconf", side_effect=OSError("no sysconf")),
        ):
            result = detect_total_system_memory()
            assert result is None


# ---------------------------------------------------------------------------
# select_nlsq_strategy
# ---------------------------------------------------------------------------


class TestSelectStrategy:
    def _select_with_threshold(
        self, n_points: int, n_params: int, threshold_gb: float
    ) -> StrategyDecision:
        """Helper: patch memory detection to yield a known threshold."""
        # threshold = total_memory * fraction, so total = threshold / fraction
        total_gb = threshold_gb / DEFAULT_MEMORY_FRACTION
        with patch(
            "heterodyne.optimization.nlsq.memory.detect_total_system_memory",
            return_value=total_gb,
        ):
            return select_nlsq_strategy(n_points, n_params)

    def test_select_standard_strategy(self) -> None:
        """Small dataset should select STANDARD."""
        decision = self._select_with_threshold(1000, 16, threshold_gb=16.0)
        assert decision.strategy is NLSQStrategy.STANDARD
        assert decision.peak_memory_gb < decision.threshold_gb

    def test_select_large_strategy(self) -> None:
        """Medium dataset where peak memory exceeds threshold but index doesn't."""
        # With threshold=1.0 GB, 10M points * 16 params ~ 7.75 GB peak
        decision = self._select_with_threshold(10_000_000, 16, threshold_gb=1.0)
        assert decision.strategy is NLSQStrategy.LARGE

    def test_select_streaming_strategy(self) -> None:
        """Huge dataset where even the index array exceeds threshold."""
        # 100 billion points: index = 100e9 * 8 / 1024^3 ~ 745 GB
        decision = self._select_with_threshold(100_000_000_000, 16, threshold_gb=16.0)
        assert decision.strategy is NLSQStrategy.STREAMING

    def test_decision_contains_reason(self) -> None:
        decision = self._select_with_threshold(1000, 16, threshold_gb=16.0)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_decision_threshold_matches(self) -> None:
        decision = self._select_with_threshold(1000, 16, threshold_gb=16.0)
        assert decision.threshold_gb == pytest.approx(16.0, rel=0.01)


# ---------------------------------------------------------------------------
# StrategyDecision immutability
# ---------------------------------------------------------------------------


class TestStrategyDecisionFrozen:
    def test_strategy_decision_frozen(self) -> None:
        decision = StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=16.0,
            peak_memory_gb=0.5,
            reason="test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            decision.strategy = NLSQStrategy.LARGE  # type: ignore[misc]
