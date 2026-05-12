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


# ---------------------------------------------------------------------------
# Regression tests: ParameterSpace → worker config serialization round-trip
# (guards against "tuple index out of range" NUTS pytree crash — het_ed14fd83)
# ---------------------------------------------------------------------------


def test_parameter_space_to_config_produces_dict() -> None:
    """to_config() returns a non-empty dict with the expected top-level key."""
    from heterodyne.config.parameter_space import ParameterSpace

    space = ParameterSpace()
    cfg = space.to_config()
    assert isinstance(cfg, dict)
    assert "initial_parameters" in cfg
    ip = cfg["initial_parameters"]
    assert "parameter_names" in ip
    assert "values" in ip
    assert "active_parameters" in ip


def test_parameter_space_to_config_round_trips_via_from_config() -> None:
    """to_config() → from_config() reconstructs the same varying_names set.

    This is the critical invariant: the worker must recover the same
    varying_names as the parent so init_params and _shard_model agree.
    """
    from heterodyne.config.parameter_space import ParameterSpace

    space = ParameterSpace()
    cfg = space.to_config()
    space2 = ParameterSpace.from_config(cfg)
    assert set(space2.varying_names) == set(space.varying_names), (
        f"Round-trip mismatch: {set(space.varying_names)} → {set(space2.varying_names)}"
    )


def test_parameter_space_from_config_stamps_config_dict() -> None:
    """from_config() sets _config_dict on the returned space.

    run_shards() reads _config_dict to pass PS config to workers; the
    attribute must be present after any from_config() call.
    """
    from heterodyne.config.parameter_space import ParameterSpace

    space = ParameterSpace.from_config({})
    assert hasattr(space, "_config_dict"), (
        "ParameterSpace.from_config() must stamp _config_dict so run_shards() "
        "can pass parameter-space config to workers without a fallback warning."
    )


def test_mp_worker_init_params_restricted_to_physics_names() -> None:
    """Worker filters init_params to ALL_PARAM_NAMES before passing to NUTS.

    Scaling params (contrast, offset) appear in varying_names but are not
    latent sites in _shard_model; leaking them into init_params causes NUTS
    to crash with 'tuple index out of range' during pytree initialization.
    Verifies the guard is present in the source.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing_backend as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "_model_sites" in source, (
        "_run_shard_worker must define _model_sites = frozenset(ALL_PARAM_NAMES) "
        "to guard init_params from non-physics parameter keys."
    )
    assert "k in _model_sites" in source, (
        "init_params comprehension must filter 'k in _model_sites' so scaling "
        "params never reach NUTS."
    )


def test_mp_worker_varying_names_validation_present() -> None:
    """Worker raises ValueError early when varying_names contains non-physics params.

    Ensures the diagnostic guard (_extra_sites check) is in the source so
    any future ParameterSpace regression produces a clear error message
    instead of an opaque NUTS pytree crash.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing_backend as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "_extra_sites" in source, (
        "_run_shard_worker must check for extra (non-physics) sites in "
        "varying_names and raise ValueError with a diagnostic message."
    )


def test_failure_categories_includes_config_error() -> None:
    """run_shards failure_categories dict must track config_error.

    'tuple index out of range' from NUTS pytree mismatches was previously
    mis-classified as 'sampling', obscuring the root cause.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing_backend as mb

    source = inspect.getsource(mb.MultiprocessingBackend.run_shards)
    assert '"config_error"' in source or "'config_error'" in source, (
        "failure_categories in run_shards must include 'config_error' key so "
        "ParameterSpace misconfiguration failures are reported distinctly."
    )


def test_run_shards_uses_to_config_fallback() -> None:
    """run_shards falls back to to_config() when _config_dict is absent.

    Prevents workers from receiving empty ps_dict when parameter_space was
    created via the model constructor (not from_config).
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing_backend as mb

    source = inspect.getsource(mb.MultiprocessingBackend.run_shards)
    assert "to_config()" in source, (
        "run_shards must call parameter_space.to_config() as fallback when "
        "_config_dict is absent; otherwise workers get ps_dict={} and "
        "reconstruct an all-vary ParameterSpace that crashes NUTS."
    )
