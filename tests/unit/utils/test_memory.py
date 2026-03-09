"""Tests for heterodyne/optimization/nlsq/memory.py.

Covers:
- System memory detection
- Peak memory estimation
- Strategy selection logic
- Environment variable override
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from heterodyne.optimization.nlsq.memory import (
    NLSQStrategy,
    StrategyDecision,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    select_nlsq_strategy,
)

# ============================================================================
# detect_total_system_memory
# ============================================================================


class TestDetectSystemMemory:
    """Tests for memory detection."""

    @pytest.mark.unit
    def test_returns_positive_or_none(self) -> None:
        """Should return a positive float or None."""
        result = detect_total_system_memory()
        if result is not None:
            assert result > 0

    @pytest.mark.unit
    def test_fallback_when_no_psutil(self) -> None:
        """Falls back gracefully when psutil is unavailable."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Should still return a value (via os.sysconf) or None
            result = detect_total_system_memory()
            # Just check it doesn't crash
            assert result is None or result > 0


# ============================================================================
# estimate_peak_memory_gb
# ============================================================================


class TestEstimatePeakMemory:
    """Tests for peak memory estimation."""

    @pytest.mark.unit
    def test_small_problem(self) -> None:
        """Small problem uses little memory."""
        gb = estimate_peak_memory_gb(n_points=100, n_params=5)
        assert 0 < gb < 0.001  # << 1 MB

    @pytest.mark.unit
    def test_large_problem(self) -> None:
        """Large problem uses significant memory."""
        gb = estimate_peak_memory_gb(n_points=1_000_000, n_params=14)
        assert gb > 0.1  # Should be at least 100 MB

    @pytest.mark.unit
    def test_scales_with_n_points(self) -> None:
        """Memory scales linearly with n_points."""
        gb_small = estimate_peak_memory_gb(n_points=1000, n_params=10)
        gb_large = estimate_peak_memory_gb(n_points=10000, n_params=10)
        assert gb_large == pytest.approx(gb_small * 10.0, rel=1e-10)

    @pytest.mark.unit
    def test_scales_with_n_params(self) -> None:
        """Memory scales linearly with n_params."""
        gb_small = estimate_peak_memory_gb(n_points=1000, n_params=5)
        gb_large = estimate_peak_memory_gb(n_points=1000, n_params=10)
        assert gb_large == pytest.approx(gb_small * 2.0, rel=1e-10)

    @pytest.mark.unit
    def test_overhead_factor(self) -> None:
        """Overhead factor of 6.5x is applied."""
        n_points, n_params = 1000, 10
        jacobian_gb = n_points * n_params * 8 / (1024**3)
        estimated = estimate_peak_memory_gb(n_points, n_params)
        assert estimated == pytest.approx(jacobian_gb * 6.5)


# ============================================================================
# select_nlsq_strategy
# ============================================================================


class TestSelectNLSQStrategy:
    """Tests for strategy selection."""

    @pytest.mark.unit
    def test_small_problem_standard(self) -> None:
        """Small problems use STANDARD strategy."""
        decision = select_nlsq_strategy(n_points=1000, n_params=10)
        assert decision.strategy == NLSQStrategy.STANDARD
        assert isinstance(decision, StrategyDecision)

    @pytest.mark.unit
    def test_huge_problem_large_strategy(self) -> None:
        """Large problems recommend LARGE strategy."""
        # Threshold = 0.1 GB: index (0.007 GB) fits, peak (~0.68 GB) exceeds
        with patch(
            "heterodyne.optimization.nlsq.memory.detect_total_system_memory",
            return_value=0.1 / 0.75,
        ):
            decision = select_nlsq_strategy(n_points=1_000_000, n_params=14)
            assert decision.strategy == NLSQStrategy.LARGE

    @pytest.mark.unit
    def test_decision_has_reason(self) -> None:
        """Decision always includes a human-readable reason."""
        decision = select_nlsq_strategy(n_points=100, n_params=5)
        assert len(decision.reason) > 0

    @pytest.mark.unit
    def test_decision_has_memory_estimates(self) -> None:
        """Decision includes threshold and peak memory."""
        decision = select_nlsq_strategy(n_points=100, n_params=5)
        assert decision.threshold_gb > 0
        assert decision.peak_memory_gb > 0

    @pytest.mark.unit
    def test_strategy_enum_values(self) -> None:
        """NLSQStrategy enum has expected values."""
        assert NLSQStrategy.STANDARD.value == "standard"
        assert NLSQStrategy.LARGE.value == "large"
        assert NLSQStrategy.STREAMING.value == "streaming"
