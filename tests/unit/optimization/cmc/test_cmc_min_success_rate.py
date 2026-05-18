"""Tests for the min_success_rate gate in ``_combine_shard_posteriors``.

Regression for Codex finding C1: the combined CMCResult must NOT report
``convergence_passed=True`` when only a small fraction of the shards
contributed, even if those shards' own diagnostics (R-hat, ESS) are clean.

Prior to the fix, 1/47 shards with perfect R-hat could mark the entire
run "converged" — masking 46 timeouts.
"""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.core import _combine_shard_posteriors
from heterodyne.optimization.cmc.results import CMCResult


def _build_shard_result(
    *,
    convergence_passed: bool,
    rng_seed: int,
    n_params: int = 14,
) -> CMCResult:
    """Build a synthetic CMCResult with clean (passing) diagnostics."""
    rng = np.random.default_rng(rng_seed)
    names = [f"p{i}" for i in range(n_params)]
    # Keep cross-shard mean spread tight so the orthogonal IQR-CV
    # heterogeneity guard doesn't false-trip; this fixture exercises the
    # rate gate, not the heterogeneity guard.
    mean = rng.normal(loc=1.0, scale=0.01, size=n_params)
    std = np.full(n_params, 0.1)  # small but positive
    samples = {
        name: rng.normal(loc=mean[i], scale=std[i], size=(200,))
        for i, name in enumerate(names)
    }
    return CMCResult(
        parameter_names=names,
        posterior_mean=mean,
        posterior_std=std,
        credible_intervals={},
        convergence_passed=convergence_passed,
        r_hat=np.full(n_params, 1.01),  # clean
        ess_bulk=np.full(n_params, 800.0),  # > min_ess=400
        ess_tail=np.full(n_params, 800.0),
        bfmi=[0.5],
        samples=samples,
        map_estimate=mean.copy(),
        num_warmup=1500,
        num_samples=1500,
        num_chains=4,
        wall_time_seconds=10.0,
        metadata={"divergence_rate": 0.0},
    )


def _build_failed_shard(n_params: int = 14) -> CMCResult:
    """Build a degenerate-failed shard with no usable posterior."""
    names = [f"p{i}" for i in range(n_params)]
    return CMCResult(
        parameter_names=names,
        posterior_mean=np.zeros(n_params),
        posterior_std=np.full(n_params, np.nan),  # filtered out as not >0
        credible_intervals={},
        convergence_passed=False,
        r_hat=np.full(n_params, np.nan),
        ess_bulk=np.full(n_params, np.nan),
        ess_tail=np.full(n_params, np.nan),
        bfmi=None,
        samples=None,
        map_estimate=None,
        num_warmup=1500,
        num_samples=1500,
        num_chains=4,
        wall_time_seconds=None,
        metadata={"divergence_rate": 0.0, "failed": True},
    )


class TestMinSuccessRateGate:
    """Codex C1 fix: combined convergence requires enough surviving shards."""

    def test_single_survivor_marks_run_NOT_converged(self) -> None:
        """1/47 shards succeeding must NOT count as a converged run."""
        cfg = CMCConfig(
            enable="never",
            min_success_rate=0.90,
            num_warmup=1500,
            num_samples=1500,
            dense_mass=True,
        )
        shards: list[CMCResult] = [
            _build_shard_result(convergence_passed=True, rng_seed=0),
        ]
        shards.extend(_build_failed_shard() for _ in range(46))

        combined = _combine_shard_posteriors(
            shards, cfg, num_shards=len(shards), base_seed=0
        )

        # Diagnostics on the survivor are clean, but rate gate fails.
        assert combined.convergence_passed is False
        assert combined.metadata["n_successful_shards"] == 1
        assert combined.metadata["n_total_shards"] == 47
        assert combined.metadata["success_rate"] < cfg.min_success_rate
        assert combined.metadata["rate_passed"] is False

    def test_full_success_marks_run_converged(self) -> None:
        """When all shards succeed with clean diagnostics, run is converged."""
        cfg = CMCConfig(
            enable="never",
            min_success_rate=0.90,
            num_warmup=1500,
            num_samples=1500,
            dense_mass=True,
            # Disable the orthogonal IQR-CV heterogeneity guard so synthetic
            # cross-shard means with wide spread don't crash this fixture
            # before the rate-gate logic under test runs.
            heterogeneity_abort=False,
        )
        shards = [
            _build_shard_result(convergence_passed=True, rng_seed=i) for i in range(5)
        ]
        combined = _combine_shard_posteriors(
            shards, cfg, num_shards=len(shards), base_seed=0
        )
        assert combined.convergence_passed is True
        assert combined.metadata["rate_passed"] is True
        assert combined.metadata["success_rate"] == 1.0

    def test_low_success_rate_fails_with_homogeneous_remainder(self) -> None:
        """Rate gate must be reachable in production: 2/10 survivors with
        homogeneous (heterogeneity-passing) posteriors should still produce
        ``convergence_passed=False`` from the rate gate alone.

        Guards against the heterogeneity check (max_parameter_cv) becoming
        an inadvertent precondition that prevents the rate gate from firing.
        With production defaults (heterogeneity_abort=True, max_parameter_cv=1.0),
        the rate gate MUST still be reachable when survivors are tight.
        """
        cfg = CMCConfig(
            enable="never",
            min_success_rate=0.50,
            num_warmup=1500,
            num_samples=1500,
            dense_mass=True,
            # PRODUCTION DEFAULTS — heterogeneity guard active.
            heterogeneity_abort=True,
            max_parameter_cv=1.0,
        )
        # 2 survivors with the SAME seed → identical posterior_mean →
        # IQR-CV ≈ 0, so heterogeneity guard cannot fire and we cleanly
        # land on the rate gate.
        shards: list[CMCResult] = [
            _build_shard_result(convergence_passed=True, rng_seed=42),
            _build_shard_result(convergence_passed=True, rng_seed=42),
        ]
        shards.extend(_build_failed_shard() for _ in range(8))

        combined = _combine_shard_posteriors(shards, cfg, num_shards=10, base_seed=0)
        # Rate gate IS reachable in production: 2/10 = 0.2 < 0.5.
        assert combined.convergence_passed is False
        assert combined.metadata["success_rate"] == 0.2
        assert combined.metadata["rate_passed"] is False
        assert combined.metadata["n_successful_shards"] == 2

    def test_metadata_diagnostics_exposed(self) -> None:
        """Caller can inspect rate diagnostics from combined metadata."""
        cfg = CMCConfig(
            enable="never",
            min_success_rate=0.50,
            num_warmup=1500,
            num_samples=1500,
            dense_mass=True,
            heterogeneity_abort=False,
        )
        shards: list[CMCResult] = [
            _build_shard_result(convergence_passed=True, rng_seed=0),
            _build_shard_result(convergence_passed=True, rng_seed=1),
        ]
        shards.extend(_build_failed_shard() for _ in range(2))

        combined = _combine_shard_posteriors(shards, cfg, num_shards=4, base_seed=0)
        # 2/4 = 0.5 success rate exactly equals threshold → rate passes.
        assert combined.metadata["n_successful_shards"] == 2
        assert combined.metadata["n_total_shards"] == 4
        assert combined.metadata["success_rate"] == 0.5
        assert combined.metadata["rate_passed"] is True
