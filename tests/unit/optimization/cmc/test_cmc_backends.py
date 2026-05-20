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

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb)
    # The broken call pattern: two args where first is params array
    assert "reparam_to_physics_jax(params, reparam_config)" not in source, (
        "Found broken reparam_to_physics_jax(params, reparam_config) call in "
        "backends/multiprocessing.py. This crashes at runtime — remove the block."
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

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "_model_sites" in source, (
        "_run_shard_worker must define _model_sites = frozenset(ALL_PARAM_NAMES) "
        "to guard init_params from non-physics parameter keys."
    )
    assert "k in _model_sites" in source, (
        "init_params comprehension must filter 'k in _model_sites' so scaling "
        "params never reach NUTS."
    )


def test_mp_worker_uses_varying_physics_names() -> None:
    """Worker must call varying_physics_names, not varying_names.

    varying_physics_names is the physics-only ParameterSpace property that
    structurally excludes contrast/offset.  Using it prevents the opaque
    'tuple index out of range' NUTS pytree crash when the caller's ParameterSpace
    has scaling params active (e.g. built from an NLSQ result).
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "varying_physics_names" in source, (
        "_run_shard_worker must call parameter_space.varying_physics_names "
        "(not varying_names) so scaling params are structurally excluded from "
        "NUTS latent sites without a manual filter."
    )


def test_parameter_space_varying_physics_names_excludes_scaling() -> None:
    """ParameterSpace.varying_physics_names never returns scaling params.

    This is the behavioral guarantee that makes the worker safe: even when
    contrast/offset are set vary=True (as NLSQ does), varying_physics_names
    returns only the 14 physics parameters.
    """
    from heterodyne.config.parameter_names import ALL_PARAM_NAMES
    from heterodyne.config.parameter_space import ParameterSpace

    space = ParameterSpace()
    space.vary["contrast"] = True
    space.vary["offset"] = True

    assert "contrast" not in space.varying_physics_names
    assert "offset" not in space.varying_physics_names
    for name in space.varying_physics_names:
        assert name in ALL_PARAM_NAMES, f"{name!r} is not a physics parameter"


def test_failure_categories_includes_config_error() -> None:
    """run_shards failure_categories dict must track config_error.

    'tuple index out of range' from NUTS pytree mismatches was previously
    mis-classified as 'sampling', obscuring the root cause.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

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

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb.MultiprocessingBackend.run_shards)
    assert "to_config()" in source, (
        "run_shards must call parameter_space.to_config() as fallback when "
        "_config_dict is absent; otherwise workers get ps_dict={} and "
        "reconstruct an all-vary ParameterSpace that crashes NUTS."
    )


def test_mp_worker_init_params_broadcast_to_num_chains() -> None:
    """Worker init_params must be broadcast to shape (num_chains,), not 0-d.

    NumPyro ≥0.21 (mcmc.py:683) does `jnp.shape(init_val)[0]` to check
    whether values are pre-batched across chains.  0-d scalars (shape=())
    cause IndexError: tuple index out of range when num_chains > 1.
    The source must call broadcast_to(..., (_num_chains,)) or equivalent.

    Regression guard for: het_0403617d — all 2 shards failed [config_error]:
    tuple index out of range.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "_num_chains" in source, (
        "_run_shard_worker must derive _num_chains from config.num_chains "
        "to broadcast init_params to shape (_num_chains,) for NumPyro ≥0.21."
    )
    assert "broadcast_to" in source, (
        "_run_shard_worker must broadcast init_params values to shape "
        "(_num_chains,); 0-d scalars trigger IndexError in NumPyro ≥0.21 "
        "when num_chains > 1."
    )


def test_mp_worker_seeds_sigma_in_init_params() -> None:
    """Worker must include sigma in init_params to avoid HalfNormal assertion.

    NumPyro ≥0.21 asserts is_prng_key(key) in HalfNormal.sample.  When sigma
    is not in init_params, init_to_median tries to sample from HalfNormal
    using a non-key trace argument — AssertionError.  Seeding sigma with
    noise_scale avoids the init_to_median path for that site.

    Regression guard for: het_0403617d secondary failure path.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert 'init_params["sigma"]' in source or "init_params['sigma']" in source, (
        "_run_shard_worker must seed init_params['sigma'] with noise_scale "
        "so init_to_median never attempts to sample HalfNormal with a "
        "non-key argument (NumPyro ≥0.21 asserts is_prng_key(key))."
    )


def test_mp_worker_clips_init_params_to_bounds() -> None:
    """Worker must clip init_params values to strictly inside parameter bounds.

    Values at the boundary edge (e.g. alpha_sample=-2.0, low=-2.0) transform
    to -inf in the unconstrained bijector, making the initial log-prob
    undefined and triggering RuntimeError: Cannot find valid initial parameters.
    Clipping by _INIT_BOUND_EPS keeps values strictly inside support.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "_INIT_BOUND_EPS" in source, (
        "_run_shard_worker must define _INIT_BOUND_EPS and clip init_params "
        "values to [low + eps, high - eps] so boundary NLSQ values (e.g. "
        "alpha_sample=-2.0 with low=-2.0) do not produce -inf in the "
        "unconstrained transform."
    )


# ---------------------------------------------------------------------------
# Regression tests: _SHARD_ARRAY_KEYS wire format and no-NLSQ init seeding
# (guards against "'>' not supported between NoneType and float" — het_457cc550)
# ---------------------------------------------------------------------------


def test_shard_array_keys_includes_element_wise_format_keys() -> None:
    """_SHARD_ARRAY_KEYS must include t1, t2, time_grid for random shards.

    The packed shared-memory pipeline only forwards keys enumerated in
    _SHARD_ARRAY_KEYS to workers.  Any key absent from this tuple is silently
    dropped (None in worker).  The element-wise wire format (random sharding
    strategy) uses t1/t2/time_grid — if these are missing, the worker falls
    through to the meshgrid path with t_jax=None and crashes with:
      TypeError: '>' not supported between instances of NoneType and float

    Regression guard for: het_457cc550 — all shards failed [sampling].
    """
    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    missing = {"t1", "t2", "time_grid"} - set(mb._SHARD_ARRAY_KEYS)
    assert not missing, (
        f"_SHARD_ARRAY_KEYS is missing {missing}. Element-wise shard keys "
        "must be listed so shared-memory pipeline forwards them to workers. "
        "Absent keys silently become None, forcing the wrong meshgrid path "
        "and crashing with TypeError: '>' not supported between NoneType and float."
    )


def test_mp_worker_seeds_physics_params_without_nlsq_warmstart() -> None:
    """Worker must seed all physics params from registry defaults when initial_values is None.

    When CMC runs without an NLSQ warm-start, initial_values is None and only
    sigma is seeded.  NumPyro ≥0.21 asserts is_prng_key(key) inside
    BetaScaled.sample() and TruncatedNormal.sample() — distributions used as
    priors for f0 and contrast.  Any site absent from init_params triggers the
    assertion when init_to_median tries to sample from these distributions.

    The fix: when initial_values is None, iterate varying_names and seed each
    param from parameter_space.values (registry defaults).  The source must
    contain a branch that reads ps_vals = parameter_space.values and populates
    init_params for the no-warm-start case.

    Regression guard for: het_457cc550 BetaScaled init_to_median failure.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb._run_shard_worker)
    assert "parameter_space.values" in source, (
        "_run_shard_worker must read parameter_space.values to seed all "
        "physics params when initial_values is None (no NLSQ warm-start). "
        "Without this, BetaScaled/TruncatedNormal priors crash in NumPyro "
        "≥0.21 via: AssertionError: is_prng_key(key)."
    )
    assert "ps_vals = parameter_space.values" in source, (
        "_run_shard_worker must assign ps_vals = parameter_space.values in the "
        "else branch (initial_values is None) so registry defaults are used to "
        "seed init_params for every varying physics parameter."
    )


def test_run_shards_shard_builder_forwards_element_wise_keys() -> None:
    """run_shards shard-dict builder must forward t1/t2/time_grid from parallel_shards.

    Three wire-format declarations must stay in sync:
      1. _SHARD_ARRAY_KEYS        — controls what shared-memory packs/unpacks.
      2. shard_data_list builder  — controls what gets extracted from parallel_shards.
      3. Worker shard_data.get()  — controls what the worker reads.

    The previous fix (e1da22f) added t1/t2/time_grid to (1) and (3) but missed
    (2).  When (2) omits a key, the SharedDataManager receives None → stores a
    zero-length sentinel → worker reads None → falls to meshgrid path with
    t_jax=None → crashes with:
      TypeError: '>' not supported between NoneType and float

    Regression guard for: het_96ff35ab — post-fix crash at 22:36 after e1da22f.
    """
    import inspect

    import heterodyne.optimization.cmc.backends.multiprocessing as mb

    source = inspect.getsource(mb.MultiprocessingBackend.run_shards)
    for key in ("t1", "t2", "time_grid"):
        assert '"t1"' in source or f"'{key}'" in source, (
            f"run_shards shard-dict builder is missing key '{key}'. "
            "Element-wise shard data (t1/t2/time_grid) from fit_cmc_sharded "
            "must be extracted in the builder so SharedDataManager actually "
            "stores them. Missing keys become zero-length sentinels → None in "
            "worker → wrong meshgrid path → TypeError."
        )
    # Verify the actual extraction pattern is present (not just the string)
    assert 'shard.get("t1")' in source, (
        'run_shards must extract t1 via shard.get("t1") in the shard-dict builder. '
        "Regression guard for het_96ff35ab (post-fix crash)."
    )


@pytest.mark.unit
class TestBugPrevention_CollectResultsGracefulDegradation:
    """Regression tests for het_c7548ee8: _collect_results must not raise when
    all shards fail — it should return [] so fit_cmc_sharded + _combine_shard_posteriors
    produce a degenerate CMCResult instead of crashing the CLI.

    Previously: _collect_results raised RuntimeError("All N shards failed"),
    which propagated all the way to the CLI with no result saved to disk.
    """

    @pytest.mark.unit
    def test_all_shards_failed_returns_empty_list_not_raises(self) -> None:
        """_collect_results returns [] when every shard failed (regression het_c7548ee8)."""
        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            MultiprocessingBackend,
        )

        backend = MultiprocessingBackend()
        config = CMCConfig()

        all_failed = [
            {
                "type": "result",
                "success": False,
                "shard_idx": i,
                "error": "Runtime timeout after 7200s",
                "error_category": "timeout",
                "duration": 7200.0,
            }
            for i in range(5)
        ]

        # Must return [] not raise RuntimeError
        result = backend._collect_results(all_failed, n_shards=5, config=config)
        assert result == [], (
            "_collect_results raised instead of returning [] when all shards failed. "
            "This is the het_c7548ee8 regression: all-timeout failure must degrade "
            "gracefully to a degenerate CMCResult, not crash the CLI."
        )

    @pytest.mark.unit
    def test_below_min_success_rate_logs_error_not_raises(self) -> None:
        """_collect_results returns partial results when success_rate < min_success_rate."""
        import numpy as np

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            MultiprocessingBackend,
        )

        backend = MultiprocessingBackend()
        config = CMCConfig()
        # Default min_success_rate is typically 0.5; create 1 success + 9 failures
        n_chains, n_samples = 4, 100
        rng = np.random.default_rng(0)
        one_good = {
            "type": "result",
            "success": True,
            "shard_idx": 0,
            "samples": {"D0_ref": rng.normal(size=n_chains * n_samples)},
            "param_names": ["D0_ref"],
            "n_chains": n_chains,
            "n_samples": n_samples,
            "extra_fields": {},
            "duration": 30.0,
            "stats": {"num_divergent": 0, "n_warmup": 50},
        }
        failures = [
            {
                "type": "result",
                "success": False,
                "shard_idx": i + 1,
                "error": "timeout",
                "error_category": "timeout",
                "duration": 7200.0,
            }
            for i in range(9)
        ]

        results = [one_good] + failures
        # Must return the 1 successful shard, not raise
        returned = backend._collect_results(results, n_shards=10, config=config)
        assert len(returned) == 1, (
            f"Expected 1 successful shard returned, got {len(returned)}. "
            "_collect_results should not raise when success_rate < min_success_rate."
        )
        assert returned[0]["success"] is True
