"""Unit tests for NLSQ anti-degeneracy utilities.

Covers GradientCollapseDetector, suggest_regularization,
compute_effective_lambda, and detect_hierarchical_trigger.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
    DegeneracyCheck,
    GradientCollapseDetector,
    compute_effective_lambda,
    detect_hierarchical_trigger,
    suggest_regularization,
)

# ---------------------------------------------------------------------------
# GradientCollapseDetector
# ---------------------------------------------------------------------------


class TestGradientCollapseDetector:
    """Tests for GradientCollapseDetector."""

    def test_gradient_collapse_detector_no_collapse(self) -> None:
        """Normal-magnitude Jacobians → detector never fires."""
        detector = GradientCollapseDetector(threshold=1e-8, window=3)
        jac = np.eye(4) * 1.0  # Frobenius norm = 2.0, well above threshold
        for _ in range(10):
            result = detector.update(jac)
        assert result is False

    def test_gradient_collapse_detector_collapse(self) -> None:
        """Near-zero Jacobians for `window` consecutive calls → returns True."""
        window = 4
        detector = GradientCollapseDetector(threshold=1e-8, window=window)
        near_zero_jac = np.ones((3, 3)) * 1e-12  # Frobenius norm << threshold
        result = False
        for _ in range(window):
            result = detector.update(near_zero_jac)
        assert result is True

    def test_gradient_collapse_detector_needs_full_window(self) -> None:
        """Collapse requires exactly `window` consecutive sub-threshold calls."""
        window = 5
        detector = GradientCollapseDetector(threshold=1e-8, window=window)
        near_zero_jac = np.ones((2, 2)) * 1e-12
        # window - 1 calls should not yet trigger
        for _ in range(window - 1):
            result = detector.update(near_zero_jac)
        assert result is False
        # The final call completes the window
        result = detector.update(near_zero_jac)
        assert result is True

    def test_gradient_collapse_detector_reset(self) -> None:
        """After collapse, reset() clears state; detector no longer fires."""
        window = 3
        detector = GradientCollapseDetector(threshold=1e-8, window=window)
        near_zero_jac = np.ones((2, 2)) * 1e-15
        for _ in range(window):
            detector.update(near_zero_jac)
        detector.reset()
        # After reset, even another window of near-zero Jacobians needs a full
        # fresh run — a single call cannot trigger collapse
        result = detector.update(near_zero_jac)
        assert result is False

    def test_gradient_collapse_detector_mixed_history(self) -> None:
        """One normal Jacobian in the window breaks the collapse streak."""
        window = 4
        detector = GradientCollapseDetector(threshold=1e-8, window=window)
        near_zero = np.ones((2, 2)) * 1e-15
        normal = np.eye(2) * 1.0
        # 3 near-zero, 1 normal, 3 near-zero — should NOT trigger collapse
        # because the window of 4 is interrupted by the normal call
        for _ in range(3):
            detector.update(near_zero)
        detector.update(normal)  # breaks the streak
        for _ in range(2):
            result = detector.update(near_zero)
        assert result is False

    def test_gradient_collapse_detector_invalid_threshold(self) -> None:
        """Non-positive threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            GradientCollapseDetector(threshold=0.0)

    def test_gradient_collapse_detector_invalid_window(self) -> None:
        """Window < 1 raises ValueError."""
        with pytest.raises(ValueError, match="window"):
            GradientCollapseDetector(window=0)


# ---------------------------------------------------------------------------
# suggest_regularization
# ---------------------------------------------------------------------------


class TestSuggestRegularization:
    """Tests for suggest_regularization."""

    def test_suggest_regularization_no_degeneracy(self) -> None:
        """Empty checks list → base_lambda returned unchanged."""
        base = 1e-4
        result = suggest_regularization([], base_lambda=base)
        assert result == pytest.approx(base)

    def test_suggest_regularization_all_clean(self) -> None:
        """All non-degenerate checks → base_lambda unchanged."""
        checks = [
            DegeneracyCheck(is_degenerate=False),
            DegeneracyCheck(is_degenerate=False),
        ]
        base = 5e-3
        result = suggest_regularization(checks, base_lambda=base)
        assert result == pytest.approx(base)

    def test_suggest_regularization_with_degeneracy(self) -> None:
        """One degenerate check → lambda increased above base."""
        checks = [
            DegeneracyCheck(is_degenerate=True, message="corr"),
            DegeneracyCheck(is_degenerate=False),
        ]
        base = 1e-4
        result = suggest_regularization(checks, base_lambda=base)
        assert result > base

    def test_suggest_regularization_linear_scaling(self) -> None:
        """Suggested lambda scales linearly with number of degenerate checks."""
        base = 1e-4
        checks_1 = [DegeneracyCheck(is_degenerate=True)]
        checks_2 = [DegeneracyCheck(is_degenerate=True) for _ in range(2)]
        lambda_1 = suggest_regularization(checks_1, base_lambda=base)
        lambda_2 = suggest_regularization(checks_2, base_lambda=base)
        # 1 degenerate → base * 2; 2 degenerate → base * 3
        assert lambda_1 == pytest.approx(base * 2)
        assert lambda_2 == pytest.approx(base * 3)


# ---------------------------------------------------------------------------
# compute_effective_lambda
# ---------------------------------------------------------------------------


class TestComputeEffectiveLambda:
    """Tests for compute_effective_lambda."""

    def test_compute_effective_lambda_iteration_zero(self) -> None:
        """At iteration 0 the effective lambda equals base_lambda."""
        base = 0.01
        result = compute_effective_lambda(base, iteration=0, decay_rate=0.9)
        assert result == pytest.approx(base)

    def test_compute_effective_lambda_exponential_decay(self) -> None:
        """Effective lambda follows base * decay_rate^iteration."""
        base = 1.0
        rate = 0.8
        for k in range(5):
            expected = base * (rate**k)
            result = compute_effective_lambda(base, iteration=k, decay_rate=rate)
            assert result == pytest.approx(expected, rel=1e-9)

    def test_compute_effective_lambda_decay_rate_one(self) -> None:
        """decay_rate=1 gives constant lambda regardless of iteration."""
        base = 0.05
        for k in range(10):
            result = compute_effective_lambda(base, iteration=k, decay_rate=1.0)
            assert result == pytest.approx(base)

    def test_compute_effective_lambda_invalid_decay_rate(self) -> None:
        """decay_rate outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate"):
            compute_effective_lambda(1e-3, iteration=0, decay_rate=0.0)
        with pytest.raises(ValueError, match="decay_rate"):
            compute_effective_lambda(1e-3, iteration=0, decay_rate=1.5)

    def test_compute_effective_lambda_negative_iteration(self) -> None:
        """Negative iteration raises ValueError."""
        with pytest.raises(ValueError, match="iteration"):
            compute_effective_lambda(1e-3, iteration=-1, decay_rate=0.9)


# ---------------------------------------------------------------------------
# detect_hierarchical_trigger
# ---------------------------------------------------------------------------


class TestDetectHierarchicalTrigger:
    """Tests for detect_hierarchical_trigger."""

    def test_detect_hierarchical_trigger_true(self) -> None:
        """Degenerate check + plateaued cost → True."""
        checks = [DegeneracyCheck(is_degenerate=True)]
        # Last 3 costs nearly identical (< 1% relative change)
        cost_history = [10.0, 9.0, 8.0, 7.0, 7.0001, 7.0002]
        assert detect_hierarchical_trigger(checks, cost_history) is True

    def test_detect_hierarchical_trigger_false_no_degeneracy(self) -> None:
        """No degeneracy → False regardless of cost history."""
        checks = [DegeneracyCheck(is_degenerate=False)]
        cost_history = [10.0, 9.0, 8.0, 7.0001, 7.0002, 7.0003]
        assert detect_hierarchical_trigger(checks, cost_history) is False

    def test_detect_hierarchical_trigger_false_no_plateau(self) -> None:
        """Degeneracy present but cost still decreasing → False."""
        checks = [DegeneracyCheck(is_degenerate=True)]
        # Cost dropping by ~10% per step — not a plateau
        cost_history = [100.0, 80.0, 60.0, 40.0, 20.0]
        assert detect_hierarchical_trigger(checks, cost_history) is False

    def test_detect_hierarchical_trigger_short_history(self) -> None:
        """Cost history shorter than 3 → False even with degeneracy."""
        checks = [DegeneracyCheck(is_degenerate=True)]
        assert detect_hierarchical_trigger(checks, [5.0, 5.0]) is False

    def test_detect_hierarchical_trigger_empty_checks(self) -> None:
        """Empty checks list → False."""
        assert detect_hierarchical_trigger([], [5.0, 5.0, 5.0]) is False
