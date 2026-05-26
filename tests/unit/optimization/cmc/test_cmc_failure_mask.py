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

    def test_high_divergence_shard_marked_failed(self) -> None:
        """Codex finding: a shard skipped for high divergence must appear in
        failure_mask, even though it has valid posterior samples."""
        cfg = _cfg()  # max_divergence_rate default 0.10
        good = _build_shard_result(convergence_passed=True, rng_seed=1)
        diverged = _build_shard_result(convergence_passed=True, rng_seed=2)
        # Valid samples, but divergence rate well over the gate -> excluded.
        diverged.metadata = {"divergence_rate": 0.5}
        shards = [
            good,
            diverged,
            _build_shard_result(convergence_passed=True, rng_seed=3),
        ]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=3, base_seed=0)
        mask = combined.metadata["failure_mask"]
        # The diverged shard (index 1) was dropped from consensus, so the
        # unified mask must report it as failed — not False.
        assert mask.tolist() == [False, True, False]
        assert combined.metadata["high_divergence_mask"].tolist() == [
            False,
            True,
            False,
        ]
        assert combined.metadata["n_successful_shards"] == 2

    def test_non_converged_shard_marked_failed(self) -> None:
        """A shard with valid samples but failed convergence (known, non-NaN
        diagnostics) is excluded from consensus and must be flagged."""
        cfg = _cfg()
        good = _build_shard_result(convergence_passed=True, rng_seed=4)
        # convergence_passed=False with finite r_hat (1.01) -> diagnostics are
        # KNOWN to be bad, so the raw-sample escape hatch does not apply.
        bad = _build_shard_result(convergence_passed=False, rng_seed=5)
        shards = [good, bad]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=2, base_seed=0)
        mask = combined.metadata["failure_mask"]
        assert mask.tolist() == [False, True]
        assert combined.metadata["bad_convergence_mask"].tolist() == [False, True]

    def test_failure_mask_matches_inclusion_predicate(self) -> None:
        """failure_mask is the exact complement of the shards that contributed:
        n_successful + sum(failure_mask) == n_total."""
        cfg = _cfg()
        good = _build_shard_result(convergence_passed=True, rng_seed=6)
        diverged = _build_shard_result(convergence_passed=True, rng_seed=7)
        diverged.metadata = {"divergence_rate": 0.9}
        failed = _build_failed_shard()
        non_conv = _build_shard_result(convergence_passed=False, rng_seed=8)
        shards = [good, diverged, failed, non_conv]
        combined = _combine_shard_posteriors(shards, cfg, num_shards=4, base_seed=0)
        mask = combined.metadata["failure_mask"]
        n_failed = int(mask.sum())
        assert combined.metadata["n_successful_shards"] + n_failed == len(shards)
        # Only the clean shard contributes.
        assert mask.tolist() == [False, True, True, True]
