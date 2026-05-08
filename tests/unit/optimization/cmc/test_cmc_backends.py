"""Tests for CMC sampler backend types: SamplingStats and divergence constants.

Covers the ``SamplingStats`` dataclass and the module-level threshold
constants exported from :mod:`heterodyne.optimization.cmc.sampler`.
"""

from __future__ import annotations

import pytest

from heterodyne.optimization.cmc.sampler import (
    DIVERGENCE_RATE_CRITICAL,
    DIVERGENCE_RATE_HIGH,
    DIVERGENCE_RATE_TARGET,
    SamplingStats,
)

# ---------------------------------------------------------------------------
# SamplingStats construction
# ---------------------------------------------------------------------------


class TestSamplingStatsCreation:
    """SamplingStats can be constructed and exposes expected fields."""

    def _make_healthy_stats(self, **overrides) -> SamplingStats:
        defaults = {
            "num_samples": 1000,
            "num_warmup": 500,
            "num_divergences": 5,
            "divergence_rate": 0.005,
            "mean_accept_prob": 0.82,
            "max_tree_depth_fraction": 0.01,
            "wall_time_seconds": 12.4,
        }
        defaults.update(overrides)
        return SamplingStats(**defaults)

    def test_sampling_stats_creation(self) -> None:
        """SamplingStats stores all provided fields without mutation."""
        stats = self._make_healthy_stats()

        assert stats.num_samples == 1000
        assert stats.num_warmup == 500
        assert stats.num_divergences == 5
        assert pytest.approx(stats.divergence_rate) == 0.005
        assert pytest.approx(stats.mean_accept_prob) == 0.82
        assert pytest.approx(stats.max_tree_depth_fraction) == 0.01
        assert pytest.approx(stats.wall_time_seconds) == 12.4

    def test_sampling_stats_is_healthy_true(self) -> None:
        """is_healthy is True when divergence rate and accept prob are within limits."""
        stats = self._make_healthy_stats(divergence_rate=0.01, mean_accept_prob=0.80)
        assert stats.is_healthy is True

    def test_sampling_stats_unhealthy_divergence(self) -> None:
        """is_healthy is False when divergence_rate >= DIVERGENCE_RATE_HIGH."""
        stats = self._make_healthy_stats(
            divergence_rate=DIVERGENCE_RATE_HIGH,
            mean_accept_prob=0.85,
        )
        assert stats.is_healthy is False

    def test_sampling_stats_unhealthy_accept(self) -> None:
        """is_healthy is False when mean_accept_prob <= 0.6."""
        stats = self._make_healthy_stats(
            divergence_rate=0.001,
            mean_accept_prob=0.55,
        )
        assert stats.is_healthy is False

    def test_sampling_stats_frozen(self) -> None:
        """SamplingStats is immutable (frozen dataclass)."""
        stats = self._make_healthy_stats()
        with pytest.raises((AttributeError, TypeError)):
            stats.num_samples = 9999  # type: ignore[misc]

    def test_sampling_stats_zero_divergences(self) -> None:
        """A run with zero divergences is healthy provided accept prob is adequate."""
        stats = self._make_healthy_stats(num_divergences=0, divergence_rate=0.0)
        assert stats.is_healthy is True


# ---------------------------------------------------------------------------
# Divergence rate constant ordering
# ---------------------------------------------------------------------------


class TestDivergenceRateConstants:
    """Module-level divergence rate thresholds obey the expected ordering."""

    def test_target_less_than_high(self) -> None:
        assert DIVERGENCE_RATE_TARGET < DIVERGENCE_RATE_HIGH

    def test_high_less_than_critical(self) -> None:
        assert DIVERGENCE_RATE_HIGH < DIVERGENCE_RATE_CRITICAL

    def test_target_is_positive(self) -> None:
        assert DIVERGENCE_RATE_TARGET > 0.0

    def test_critical_is_below_one(self) -> None:
        assert DIVERGENCE_RATE_CRITICAL < 1.0

    def test_target_value(self) -> None:
        """TARGET should be at most 1 % — a commonly recommended threshold."""
        assert DIVERGENCE_RATE_TARGET <= 0.01

    def test_high_value(self) -> None:
        """HIGH should be no more than 10 % — warns before CRITICAL."""
        assert DIVERGENCE_RATE_HIGH <= 0.10


def test_reparam_to_physics_jax_signature() -> None:
    """reparam_to_physics_jax takes (log_at_tref, alpha, t_ref) — three scalars.

    Documents the correct call signature. The MP worker must NOT call it with
    (params_array, reparam_config) — two args of wrong types.
    """
    import inspect

    from heterodyne.optimization.cmc.reparameterization import reparam_to_physics_jax

    sig = inspect.signature(reparam_to_physics_jax)
    param_names = list(sig.parameters.keys())
    assert param_names == ["log_at_tref", "alpha", "t_ref"], (
        f"Unexpected signature: {param_names}"
    )


def test_mp_worker_model_does_not_contain_wrong_reparam_call() -> None:
    """The MP worker's _shard_model must not call reparam_to_physics_jax(params, config).

    Verifies the broken call was removed. The worker samples physics-space
    parameters directly from priors — no back-transform is needed.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing_backend as mb

    source = inspect.getsource(mb)
    # The broken call pattern: two args where first is params array
    assert "reparam_to_physics_jax(params, reparam_config)" not in source, (
        "Found broken reparam_to_physics_jax(params, reparam_config) call in "
        "multiprocessing_backend. This crashes at runtime — remove the block."
    )
