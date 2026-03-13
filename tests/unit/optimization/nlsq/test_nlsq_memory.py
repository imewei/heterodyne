"""Tests for NLSQ multi-start memory containers.

Covers MultiStartConfig defaults, SingleStartResult field structure,
and MultiStartResult best-result selection logic — without running
any actual optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.multistart import (
    MultiStartConfig,
    MultiStartResult,
    SingleStartResult,
)
from heterodyne.optimization.nlsq.results import NLSQResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nlsq_result(
    cost: float,
    success: bool = True,
    n_params: int = 3,
) -> NLSQResult:
    """Build a minimal NLSQResult for unit testing."""
    params = np.ones(n_params, dtype=np.float64)
    names = [f"p{i}" for i in range(n_params)]
    return NLSQResult(
        parameters=params,
        parameter_names=names,
        success=success,
        message="ok",
        final_cost=cost,
    )


def _make_single_start(
    cost: float,
    index: int,
    success: bool = True,
    n_params: int = 3,
) -> SingleStartResult:
    return SingleStartResult(
        result=_make_nlsq_result(cost=cost, success=success, n_params=n_params),
        start_index=index,
        initial_params=np.zeros(n_params, dtype=np.float64),
        wall_time=0.1,
    )


# ---------------------------------------------------------------------------
# MultiStartConfig
# ---------------------------------------------------------------------------


class TestMultiStartConfigDefaults:
    def test_n_starts_default(self) -> None:
        cfg = MultiStartConfig()
        assert cfg.n_starts == 10

    def test_seed_default_is_none(self) -> None:
        cfg = MultiStartConfig()
        assert cfg.seed is None

    def test_parallel_default_is_false(self) -> None:
        cfg = MultiStartConfig()
        assert cfg.parallel is False

    def test_worker_timeout_default(self) -> None:
        cfg = MultiStartConfig()
        assert cfg.worker_timeout == 1800.0

    def test_max_workers_auto_resolves(self) -> None:
        # max_workers is resolved to min(n_starts, cpu_count) in __post_init__
        cfg = MultiStartConfig(n_starts=5)
        assert cfg.max_workers is not None
        assert cfg.max_workers >= 1

    def test_n_starts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_starts"):
            MultiStartConfig(n_starts=0)

    def test_worker_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="worker_timeout"):
            MultiStartConfig(worker_timeout=-1.0)

    def test_from_dict_round_trip(self) -> None:
        cfg = MultiStartConfig.from_dict({"n_starts": 7, "seed": 99, "parallel": True})
        assert cfg.n_starts == 7
        assert cfg.seed == 99
        assert cfg.parallel is True

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = MultiStartConfig.from_dict({"n_starts": 3, "nonexistent_key": "x"})
        assert cfg.n_starts == 3


# ---------------------------------------------------------------------------
# SingleStartResult
# ---------------------------------------------------------------------------


class TestSingleStartResultCreation:
    def test_fields_accessible(self) -> None:
        nlsq = _make_nlsq_result(cost=0.5)
        init = np.array([1.0, 2.0, 3.0])
        ssr = SingleStartResult(
            result=nlsq,
            start_index=2,
            initial_params=init,
            wall_time=1.23,
        )
        assert ssr.start_index == 2
        assert ssr.wall_time == pytest.approx(1.23)
        assert ssr.result is nlsq
        np.testing.assert_array_equal(ssr.initial_params, init)

    def test_result_cost_propagates(self) -> None:
        ssr = _make_single_start(cost=3.14, index=0)
        assert ssr.result.final_cost == pytest.approx(3.14)

    def test_success_flag_preserved(self) -> None:
        ssr_ok = _make_single_start(cost=1.0, index=0, success=True)
        ssr_fail = _make_single_start(cost=9.9, index=1, success=False)
        assert ssr_ok.result.success is True
        assert ssr_fail.result.success is False


# ---------------------------------------------------------------------------
# MultiStartResult best-result selection
# ---------------------------------------------------------------------------


class TestMultiStartResultBestSelection:
    def _build_multi(
        self,
        costs: list[float],
        success_flags: list[bool] | None = None,
    ) -> MultiStartResult:
        if success_flags is None:
            success_flags = [True] * len(costs)

        starts = [
            _make_single_start(cost=c, index=i, success=s)
            for i, (c, s) in enumerate(zip(costs, success_flags, strict=True))
        ]
        best = min(starts, key=lambda ss: ss.result.final_cost or float("inf"))
        n_successful = sum(1 for ss in starts if ss.result.success)

        return MultiStartResult(
            best_result=best.result,
            all_starts=starts,
            n_successful=n_successful,
            n_total=len(starts),
            config=MultiStartConfig(n_starts=len(starts), seed=0),
            wall_time_total=0.5,
        )

    def test_best_result_is_lowest_cost(self) -> None:
        mr = self._build_multi(costs=[5.0, 1.0, 3.0])
        assert mr.best_result.final_cost == pytest.approx(1.0)

    def test_n_successful_counted_correctly(self) -> None:
        mr = self._build_multi(
            costs=[1.0, 2.0, 3.0],
            success_flags=[True, False, True],
        )
        assert mr.n_successful == 2

    def test_n_total_matches_starts(self) -> None:
        mr = self._build_multi(costs=[0.1, 0.2, 0.3, 0.4])
        assert mr.n_total == 4
        assert len(mr.all_starts) == 4

    def test_all_results_backward_compat(self) -> None:
        mr = self._build_multi(costs=[2.0, 1.5])
        results = mr.all_results
        assert len(results) == 2
        assert all(isinstance(r, NLSQResult) for r in results)

    def test_to_nlsq_result_attaches_multistart_metadata(self) -> None:
        mr = self._build_multi(costs=[3.0, 1.0, 2.0])
        result = mr.to_nlsq_result()
        assert "multistart" in result.metadata
        meta = result.metadata["multistart"]
        assert meta["n_starts"] == 3
        assert meta["n_successful"] == 3

    def test_single_start_run(self) -> None:
        mr = self._build_multi(costs=[0.42])
        assert mr.best_result.final_cost == pytest.approx(0.42)
        assert mr.n_total == 1

    def test_all_failed_uses_lowest_cost(self) -> None:
        # Even when all succeed=False, best_result should be the one with min cost
        mr = self._build_multi(
            costs=[9.0, 3.0, 7.0],
            success_flags=[False, False, False],
        )
        assert mr.n_successful == 0
        assert mr.best_result.final_cost == pytest.approx(3.0)
