"""Tests for the per-shard ``failure_mask`` exposed in ``CMCResult.metadata``.

P2-d (Gemini): callers can now identify *which* shards failed, not just
how many.  The mask is a numpy bool array of length ``num_shards``,
True at index i iff shard i was rejected by the validity gate.
"""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.core import _combine_shard_posteriors
from tests.unit.optimization.cmc.test_cmc_min_success_rate import (
    _build_failed_shard,
    _build_shard_result,
)


def _cfg() -> CMCConfig:
    return CMCConfig(
        enable="never",
        min_success_rate=0.10,
        num_warmup=1500,
        num_samples=1500,
        dense_mass=True,
        heterogeneity_abort=False,
    )


class TestFailureMask:
    def test_all_shards_succeed_mask_all_false(self) -> None:
        cfg = _cfg()
        shards = [
            _build_shard_result(convergence_passed=True, rng_seed=42) for _ in range(5)
        ]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=5, base_seed=0)
        mask = combined.metadata["failure_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (5,)
        assert not mask.any()

    def test_some_shards_fail_mask_marks_failed_indices(self) -> None:
        cfg = _cfg()
        # Alternating pattern [S, F, S, F, S] — fixed indices [1, 3] fail.
        shards = []
        for i in range(5):
            if i % 2 == 0:
                shards.append(_build_shard_result(convergence_passed=True, rng_seed=42))
            else:
                shards.append(_build_failed_shard())
        combined = _combine_shard_posteriors(shards, cfg, num_shards=5, base_seed=0)
        mask = combined.metadata["failure_mask"]
        assert mask.tolist() == [False, True, False, True, False]

    def test_all_shards_fail_mask_all_true(self) -> None:
        cfg = _cfg()
        shards = [_build_failed_shard() for _ in range(5)]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=5, base_seed=0)
        mask = combined.metadata["failure_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (5,)
        assert mask.all()

    def test_mask_dtype_and_array_kind(self) -> None:
        cfg = _cfg()
        shards = [
            _build_shard_result(convergence_passed=True, rng_seed=7) for _ in range(3)
        ]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=3, base_seed=0)
        mask = combined.metadata["failure_mask"]
        # Must be a numpy bool array — not a list, not int.
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        # Must not have been silently downcast to int.
        assert mask.dtype.kind == "b"
