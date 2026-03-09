"""Unit tests for strategies/base.py and strategies/sequential.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.optimization.nlsq.strategies.base import (
    StrategyResult,
    _LARGE_DATASET,
    _MEDIUM_DATASET,
    _SMALL_DATASET,
    select_strategy,
)
from heterodyne.optimization.nlsq.strategies.sequential import (
    AngleSubset,
    MultiAngleResult,
    SequentialStrategy,
    combine_angle_results,
    restore_fixed_parameters,
    strip_fixed_parameters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_nlsq_result(
    *,
    parameters: np.ndarray | None = None,
    parameter_names: list[str] | None = None,
    success: bool = True,
    message: str = "converged",
    covariance: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    final_cost: float | None = 1.0,
    jacobian: np.ndarray | None = None,
) -> Any:
    if parameters is None:
        parameters = np.array([1.0, 2.0])
    if parameter_names is None:
        parameter_names = [f"p{i}" for i in range(len(parameters))]
    return SimpleNamespace(
        parameters=parameters,
        parameter_names=parameter_names,
        success=success,
        message=message,
        covariance=covariance,
        residuals=residuals,
        final_cost=final_cost,
        jacobian=jacobian,
    )


def _make_strategy_result(
    *,
    parameters: np.ndarray | None = None,
    success: bool = True,
    covariance: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    final_cost: float | None = 1.0,
) -> StrategyResult:
    result = _fake_nlsq_result(
        parameters=parameters,
        success=success,
        covariance=covariance,
        residuals=residuals,
        final_cost=final_cost,
    )
    return StrategyResult(result=result, strategy_name="test")


# ===================================================================
# StrategyResult dataclass
# ===================================================================


class TestStrategyResult:
    def test_defaults(self) -> None:
        nlsq = _fake_nlsq_result()
        sr = StrategyResult(result=nlsq, strategy_name="jit")
        assert sr.strategy_name == "jit"
        assert sr.n_chunks == 1
        assert sr.peak_memory_mb == 0.0
        assert sr.metadata == {}

    def test_custom_metadata(self) -> None:
        nlsq = _fake_nlsq_result()
        sr = StrategyResult(
            result=nlsq,
            strategy_name="chunked",
            n_chunks=4,
            peak_memory_mb=512.0,
            metadata={"key": "value"},
        )
        assert sr.n_chunks == 4
        assert sr.peak_memory_mb == 512.0
        assert sr.metadata["key"] == "value"


# ===================================================================
# Dataset size thresholds
# ===================================================================


class TestThresholds:
    def test_small_threshold(self) -> None:
        assert _SMALL_DATASET == 100 * 100

    def test_medium_threshold(self) -> None:
        assert _MEDIUM_DATASET == 500 * 500

    def test_large_threshold(self) -> None:
        assert _LARGE_DATASET == 2000 * 2000


# ===================================================================
# select_strategy
# ===================================================================


class TestSelectStrategy:
    @staticmethod
    def _config(chunk_size: int | None = None) -> Any:
        return SimpleNamespace(chunk_size=chunk_size)

    def test_explicit_chunk_size(self) -> None:
        strategy = select_strategy(50, 10, self._config(chunk_size=256))
        assert strategy.__class__.__name__ == "ChunkedStrategy"

    def test_small_dataset_residual(self) -> None:
        strategy = select_strategy(50, 10, self._config())
        assert strategy.__class__.__name__ == "ResidualStrategy"

    def test_medium_dataset_jit(self) -> None:
        n_data = _SMALL_DATASET + 1
        strategy = select_strategy(n_data, 10, self._config())
        assert strategy.__class__.__name__ == "JITStrategy"

    def test_large_dataset_chunked(self) -> None:
        n_data = _MEDIUM_DATASET + 1
        strategy = select_strategy(
            n_data, 10, self._config(), available_memory_gb=8.0
        )
        assert strategy.__class__.__name__ == "ChunkedStrategy"

    def test_large_dataset_auto_memory(self) -> None:
        n_data = _MEDIUM_DATASET + 1
        with patch(
            "heterodyne.optimization.nlsq.strategies.base._estimate_available_memory",
            return_value=4.0,
        ):
            strategy = select_strategy(n_data, 10, self._config())
        assert strategy.__class__.__name__ == "ChunkedStrategy"


# ===================================================================
# strip_fixed_parameters / restore_fixed_parameters
# ===================================================================


class TestFixedParameterHelpers:
    def test_strip_removes_fixed(self) -> None:
        p = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 2.0, 0.0])
        hi = np.array([5.0, 2.0, 5.0])
        free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
        np.testing.assert_array_equal(free, [1.0, 3.0])
        np.testing.assert_array_equal(fl, [0.0, 0.0])
        np.testing.assert_array_equal(fu, [5.0, 5.0])
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_all_free(self) -> None:
        p = np.array([1.0, 2.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([10.0, 10.0])
        free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
        assert len(free) == 2
        assert mask.all()

    def test_all_fixed(self) -> None:
        p = np.array([1.0, 2.0])
        lo = np.array([1.0, 2.0])
        hi = np.array([1.0, 2.0])
        free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
        assert len(free) == 0
        assert not mask.any()

    def test_restore_round_trip(self) -> None:
        p = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 2.0, 0.0, 4.0])
        hi = np.array([5.0, 2.0, 5.0, 4.0])
        free, _, _, mask = strip_fixed_parameters(p, lo, hi)

        # Simulate optimization changing free params
        optimised_free = free * 1.5
        restored = restore_fixed_parameters(optimised_free, p, mask)

        np.testing.assert_array_equal(restored[~mask], p[~mask])
        np.testing.assert_array_almost_equal(restored[mask], free * 1.5)


# ===================================================================
# AngleSubset / MultiAngleResult dataclasses
# ===================================================================


class TestAngleSubset:
    def test_construction(self) -> None:
        c2 = np.ones((10, 10))
        subset = AngleSubset(
            phi_angle=45.0,
            angle_index=2,
            n_points=100,
            c2_data=c2,
            weights=None,
        )
        assert subset.phi_angle == 45.0
        assert subset.angle_index == 2
        assert subset.n_points == 100
        assert subset.weights is None


class TestMultiAngleResult:
    def test_construction(self) -> None:
        mar = MultiAngleResult(
            per_angle_results=[],
            n_angles_total=3,
            n_angles_success=2,
            n_angles_failed=1,
            success_rate=2 / 3,
            phi_angles=np.array([0.0, 45.0, 90.0]),
        )
        assert mar.n_angles_total == 3
        assert mar.success_rate == pytest.approx(2 / 3)


# ===================================================================
# combine_angle_results
# ===================================================================


class TestCombineAngleResults:
    def test_uniform_weighting(self) -> None:
        sr1 = _make_strategy_result(
            parameters=np.array([1.0, 2.0]),
            covariance=np.eye(2),
        )
        sr2 = _make_strategy_result(
            parameters=np.array([3.0, 4.0]),
            covariance=np.eye(2),
        )
        params, cov, cost = combine_angle_results([sr1, sr2], weighting="uniform")
        np.testing.assert_array_almost_equal(params, [2.0, 3.0])
        assert cost == pytest.approx(2.0)

    def test_inverse_variance_weighting(self) -> None:
        sr1 = _make_strategy_result(
            parameters=np.array([1.0]),
            covariance=np.array([[0.1]]),
        )
        sr2 = _make_strategy_result(
            parameters=np.array([3.0]),
            covariance=np.array([[1.0]]),
        )
        params, cov, cost = combine_angle_results(
            [sr1, sr2], weighting="inverse_variance"
        )
        # Higher weight on sr1 (lower variance)
        assert params[0] < 2.0

    def test_n_points_weighting(self) -> None:
        sr1 = _make_strategy_result(
            parameters=np.array([1.0]),
            covariance=np.array([[1.0]]),
            residuals=np.ones(100),
        )
        sr2 = _make_strategy_result(
            parameters=np.array([3.0]),
            covariance=np.array([[1.0]]),
            residuals=np.ones(900),
        )
        params, _, _ = combine_angle_results([sr1, sr2], weighting="n_points")
        # sr2 has 9x more points -> combined params closer to 3.0
        assert params[0] > 2.0

    def test_no_converged_raises(self) -> None:
        sr = _make_strategy_result(success=False)
        with pytest.raises(ValueError, match="no angles converged"):
            combine_angle_results([sr])

    def test_unknown_weighting_raises(self) -> None:
        sr = _make_strategy_result()
        with pytest.raises(ValueError, match="unknown weighting"):
            combine_angle_results([sr], weighting="fancy")

    def test_missing_covariance_fallback(self) -> None:
        """When covariance is None, identity should be used as fallback."""
        sr = _make_strategy_result(
            parameters=np.array([1.0, 2.0]),
            covariance=None,
        )
        params, cov, cost = combine_angle_results([sr], weighting="uniform")
        np.testing.assert_array_almost_equal(params, [1.0, 2.0])

    def test_missing_residuals_n_points_fallback(self) -> None:
        """When residuals is None, n_points weight should fall back to 1.0."""
        sr = _make_strategy_result(
            parameters=np.array([1.0]),
            covariance=np.array([[1.0]]),
            residuals=None,
        )
        params, _, _ = combine_angle_results([sr], weighting="n_points")
        np.testing.assert_array_almost_equal(params, [1.0])


# ===================================================================
# SequentialStrategy — construction and helpers
# ===================================================================


class TestSequentialStrategy:
    def test_construction_defaults(self) -> None:
        s = SequentialStrategy()
        assert s.name == "sequential"
        assert s._inner_name == "jit"
        assert s._min_success_rate == 0.0
        assert s._clamp_to_bounds is True

    def test_construction_custom(self) -> None:
        s = SequentialStrategy(
            inner_strategy_name="residual",
            min_success_rate=0.5,
            clamp_to_bounds=False,
        )
        assert s._inner_name == "residual"
        assert s._min_success_rate == 0.5
        assert s._clamp_to_bounds is False

    def test_repr(self) -> None:
        s = SequentialStrategy(inner_strategy_name="chunked", min_success_rate=0.8)
        r = repr(s)
        assert "chunked" in r
        assert "0.8" in r

    def test_build_subsets_no_weights(self) -> None:
        c2 = np.random.default_rng(42).random((3, 5, 5))
        phi = np.array([0.0, 45.0, 90.0])
        subsets = SequentialStrategy._build_subsets(c2, phi, weights=None)
        assert len(subsets) == 3
        assert subsets[0].phi_angle == 0.0
        assert subsets[1].angle_index == 1
        assert subsets[2].n_points == 25
        assert subsets[0].weights is None

    def test_build_subsets_per_angle_weights(self) -> None:
        c2 = np.ones((2, 4, 4))
        phi = np.array([0.0, 90.0])
        w = np.ones((2, 4, 4)) * 0.5
        subsets = SequentialStrategy._build_subsets(c2, phi, weights=w)
        np.testing.assert_array_equal(subsets[0].weights, w[0])
        np.testing.assert_array_equal(subsets[1].weights, w[1])

    def test_build_subsets_shared_weights(self) -> None:
        c2 = np.ones((2, 4, 4))
        phi = np.array([0.0, 90.0])
        w = np.ones((4, 4)) * 0.5
        subsets = SequentialStrategy._build_subsets(c2, phi, weights=w)
        # Shared weights should be the same object for all subsets
        assert subsets[0].weights is w
        assert subsets[1].weights is w

    def test_annotate_scaling_no_scaling(self) -> None:
        model = SimpleNamespace()  # no scaling attr
        sr = StrategyResult(result=_fake_nlsq_result(), strategy_name="test")
        SequentialStrategy._annotate_scaling(model, 0, sr)
        assert "contrast" not in sr.metadata

    def test_annotate_scaling_single_angle(self) -> None:
        model = SimpleNamespace(scaling=SimpleNamespace(n_angles=1))
        sr = StrategyResult(result=_fake_nlsq_result(), strategy_name="test")
        SequentialStrategy._annotate_scaling(model, 0, sr)
        assert "contrast" not in sr.metadata

    def test_annotate_scaling_multi_angle(self) -> None:
        scaling = SimpleNamespace(
            n_angles=3,
            get_for_angle=lambda i: (0.9 + i * 0.01, 0.001 * i),
        )
        model = SimpleNamespace(scaling=scaling)
        sr = StrategyResult(result=_fake_nlsq_result(), strategy_name="test")
        SequentialStrategy._annotate_scaling(model, 1, sr)
        assert sr.metadata["contrast"] == pytest.approx(0.91)
        assert sr.metadata["offset"] == pytest.approx(0.001)

    def test_annotate_scaling_exception_handled(self) -> None:
        def _raise(_i: int) -> None:
            raise RuntimeError("scaling error")

        scaling = SimpleNamespace(n_angles=3, get_for_angle=_raise)
        model = SimpleNamespace(scaling=scaling)
        sr = StrategyResult(result=_fake_nlsq_result(), strategy_name="test")
        # Should not raise
        SequentialStrategy._annotate_scaling(model, 0, sr)
        assert "contrast" not in sr.metadata

    def test_get_inner_strategy_known(self) -> None:
        for name in ("residual", "jit", "chunked"):
            s = SequentialStrategy(inner_strategy_name=name)
            inner = s._get_inner_strategy()
            assert inner is not None

    def test_get_inner_strategy_unknown_fallback(self) -> None:
        s = SequentialStrategy(inner_strategy_name="nonexistent")
        inner = s._get_inner_strategy()
        assert inner.__class__.__name__ == "JITStrategy"
