"""Regression: the joint multi-phi CMC path shards large data (homodyne parity).

Pins the fix for the het_97fd5bbd report, where a 3-angle / 3M-point dataset
ran a single NUTS pass over all points (``num_shards=1`` was hardcoded in
``fit_cmc_multi_phi``) instead of sharding + Consensus-MC like homodyne's
``_fit_mcmc_jax_impl``.

The per-shard NUTS runner is stubbed so the test exercises the *orchestration*
(shard decision, per-shard dispatch, consensus combine) deterministically and
fast — without paying for real sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

import heterodyne.optimization.cmc.core as cmc_core
from heterodyne.optimization.cmc.results import CMCResult


def _make_pooled_arrays(n_grid: int, angles: list[float], seed: int = 0):
    """Return flat (data, t1, t2, phi) pooled arrays for an (n_phi, n, n) stack."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.001, 0.001 * n_grid, n_grid)
    tt1, tt2 = np.meshgrid(t, t, indexing="ij")
    data, t1, t2, phi = [], [], [], []
    for a in angles:
        data.append((1.0 + 0.1 * rng.standard_normal((n_grid, n_grid))).ravel())
        t1.append(tt1.ravel())
        t2.append(tt2.ravel())
        phi.append(np.full(n_grid * n_grid, a))
    return (
        np.concatenate(data),
        np.concatenate(t1),
        np.concatenate(t2),
        np.concatenate(phi),
    )


@pytest.fixture
def stub_nuts(monkeypatch):
    """Replace the per-shard NUTS runner with a fast, consistent fake.

    Records every invocation and returns a valid CMCResult whose parameter
    vector matches the global angle set (so ``_combine_shard_posteriors``
    aligns across shards).
    """
    calls: list[dict] = []

    def _fake(**kwargs) -> CMCResult:
        calls.append(kwargs)
        space = kwargs["space"]
        n_phi = kwargs["n_phi"]
        physics = [n for n in space.varying_names if n not in ("contrast", "offset")]
        names = (
            physics
            + [f"contrast_{i}" for i in range(n_phi)]
            + [f"offset_{i}" for i in range(n_phi)]
        )
        p = len(names)
        return CMCResult(
            parameter_names=names,
            posterior_mean=np.ones(p),
            posterior_std=np.ones(p),
            credible_intervals={},
            convergence_passed=True,
            r_hat=np.ones(p),
            ess_bulk=np.full(p, 1000.0),
            num_samples=10,
            num_chains=2,
            num_shards=kwargs["result_num_shards"],
            divergences=0,
            convergence_status="converged",
        )

    monkeypatch.setattr(cmc_core, "_joint_pooled_nuts_run", _fake)
    return calls


def test_large_multi_phi_data_shards_and_combines(stub_nuts):
    """3 angles + forced small shard size → K>1 shards + consensus combine."""
    data, t1, t2, phi = _make_pooled_arrays(30, [-5.0, 5.0, 90.0])
    result = cmc_core.fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.0054,
        L=2_000_000.0,
        analysis_mode="laminar_flow",
        # backend_name="cpu" forces the in-process sequential shard path so the
        # monkeypatched runner is used (parallel workers are separate processes
        # that would not see the stub).
        cmc_config={
            "sharding": {"num_shards": 6},
            "backend_config": {"backend_name": "cpu"},
        },
        dt=0.001,
    )
    # Sharding happened: more than one per-shard NUTS run, combined result
    # reports K > 1 shards.
    assert len(stub_nuts) == 6, "expected one NUTS run per requested shard"
    assert result.num_shards == 6
    assert result.metadata.get("sharding_strategy") == "angle_balanced"
    # Every per-shard call ran the GLOBAL angle set so consensus could align.
    assert all(c["n_phi"] == 3 for c in stub_nuts)
    assert all(c["result_num_shards"] == result.num_shards for c in stub_nuts)


def test_small_data_runs_single_pass(stub_nuts):
    """Below the single-shard limit and unforced → exactly one NUTS pass."""
    data, t1, t2, phi = _make_pooled_arrays(20, [-5.0, 5.0, 90.0])
    result = cmc_core.fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.0054,
        L=2_000_000.0,
        analysis_mode="laminar_flow",
        cmc_config={},  # no forced sharding; n_total well under 100k
        dt=0.001,
    )
    assert len(stub_nuts) == 1
    assert result.num_shards == 1


def _shard_result(n_phi: int, divergences: int, num_samples: int, num_chains: int):
    """Build a per-shard CMCResult shaped like _joint_pooled_nuts_run output."""
    from heterodyne.optimization.cmc.config import CMCConfig

    cfg = CMCConfig()
    total = num_samples * num_chains
    rate = divergences / total if total else 0.0
    names = (
        [f"p{i}" for i in range(3)]
        + [f"contrast_{i}" for i in range(n_phi)]
        + [f"offset_{i}" for i in range(n_phi)]
    )
    p = len(names)
    passed = rate <= getattr(cfg, "max_divergence_rate", 0.10)
    return CMCResult(
        parameter_names=names,
        posterior_mean=np.full(p, 2.0),
        posterior_std=np.ones(p),
        credible_intervals={},
        convergence_passed=passed,
        num_samples=num_samples,
        num_chains=num_chains,
        num_shards=4,
        divergences=divergences,
        convergence_status="converged" if passed else "divergences",
        metadata={"num_divergent": divergences, "divergence_rate": rate},
    )


def test_divergent_but_acceptable_shards_are_kept_in_consensus():
    """A shard with a few divergences (rate <= max_divergence_rate) must NOT be
    dropped wholesale — otherwise large multi-shard runs collapse to all-NaN."""
    from heterodyne.optimization.cmc.config import CMCConfig
    from heterodyne.optimization.cmc.core import _combine_shard_posteriors

    cfg = CMCConfig()  # max_divergence_rate default 0.10
    n_samples, n_chains = 1000, 4  # 4000 iters → 1% rate at 40 divergences
    shards = [
        _shard_result(
            n_phi=3, divergences=0, num_samples=n_samples, num_chains=n_chains
        ),
        _shard_result(
            n_phi=3, divergences=40, num_samples=n_samples, num_chains=n_chains
        ),
        _shard_result(
            n_phi=3, divergences=10, num_samples=n_samples, num_chains=n_chains
        ),
    ]
    combined = _combine_shard_posteriors(shards, cfg, num_shards=3, base_seed=0)
    # All three shards are under the 10% rate → combined posterior is finite,
    # not the degenerate all-NaN result returned when every shard is excluded.
    assert np.all(np.isfinite(combined.posterior_mean))
    assert not combined.metadata.get("all_shards_failed", False)
    assert np.allclose(combined.posterior_mean, 2.0)


def test_single_shard_threshold_boundary(stub_nuts):
    """Pin the 100,000-point single-shard threshold (n_total > limit shards).

    n_total is constructed via the grid so it brackets the limit tightly:
    N=316 -> 316*315 = 99,540 (<= limit, single pass);
    N=317 -> 317*316 = 100,172 (> limit, sharded).
    """
    # Just below the limit -> one NUTS pass, no sharding.
    data, t1, t2, phi = _make_pooled_arrays(316, [0.0])
    below = cmc_core.fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.0054,
        L=2_000_000.0,
        analysis_mode="laminar_flow",
        cmc_config={"backend_config": {"backend_name": "cpu"}},
        dt=0.001,
    )
    assert below.num_shards == 1
    assert len(stub_nuts) == 1

    stub_nuts.clear()

    # Just above the limit -> sharded consensus.
    data, t1, t2, phi = _make_pooled_arrays(317, [0.0])
    above = cmc_core.fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.0054,
        L=2_000_000.0,
        analysis_mode="laminar_flow",
        cmc_config={"backend_config": {"backend_name": "cpu"}},
        dt=0.001,
    )
    assert above.num_shards > 1
    assert len(stub_nuts) > 1


# ---------------------------------------------------------------------------
# Parallel dispatch decision + fallback (Issue #1)
# ---------------------------------------------------------------------------


def _trivial_result(**kwargs) -> CMCResult:
    n_phi = kwargs.get("n_phi", 1)
    k = kwargs.get("result_num_shards", 1)
    p = 3 + 2 * n_phi
    return CMCResult(
        parameter_names=[f"x{i}" for i in range(p)],
        posterior_mean=np.ones(p),
        posterior_std=np.ones(p),
        credible_intervals={},
        convergence_passed=True,
        num_shards=k,
        metadata={"divergence_rate": 0.0},
    )


def _payloads(n: int) -> list[dict]:
    return [{"n_phi": 3, "result_num_shards": n, "_idx": i} for i in range(n)]


def test_dispatch_uses_parallel_when_workers_available(monkeypatch):
    from heterodyne.optimization.cmc.config import CMCConfig

    monkeypatch.setattr(cmc_core, "_estimate_n_workers", lambda: 4)
    # Pin abundant RAM so the memory-aware cap never binds — this test is about
    # CPU-based dispatch, not the OOM guard (covered separately below).
    monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 256 * 1024**3)
    monkeypatch.setattr(cmc_core, "_joint_pooled_nuts_run", _trivial_result)
    called = {"parallel": 0}

    def _fake_parallel(payloads, *, n_workers, num_chains):
        called["parallel"] += 1
        return [_trivial_result(**p) for p in payloads]

    monkeypatch.setattr(
        "heterodyne.optimization.cmc.backends.multiprocessing."
        "run_joint_pooled_shards_parallel",
        _fake_parallel,
    )
    cfg = CMCConfig.from_dict({"backend_config": {"backend_name": "auto"}})
    out = cmc_core._run_joint_shards(_payloads(6), cfg, n_shards=6)
    assert called["parallel"] == 1
    assert len(out) == 6


def test_dispatch_falls_back_to_sequential_on_parallel_error(monkeypatch):
    from heterodyne.optimization.cmc.config import CMCConfig

    monkeypatch.setattr(cmc_core, "_estimate_n_workers", lambda: 4)
    monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 256 * 1024**3)
    seq_calls = {"n": 0}

    def _counting_local(p):
        seq_calls["n"] += 1
        return _trivial_result(**p)

    monkeypatch.setattr(cmc_core, "_run_joint_pooled_shard_local", _counting_local)

    def _boom(payloads, *, n_workers, num_chains):
        raise RuntimeError("worker pool exploded")

    monkeypatch.setattr(
        "heterodyne.optimization.cmc.backends.multiprocessing."
        "run_joint_pooled_shards_parallel",
        _boom,
    )
    cfg = CMCConfig.from_dict({"backend_config": {"backend_name": "auto"}})
    out = cmc_core._run_joint_shards(_payloads(4), cfg, n_shards=4)
    assert seq_calls["n"] == 4, "must fall back to sequential per-shard runs"
    assert len(out) == 4


def test_dispatch_sequential_for_cpu_backend(monkeypatch):
    from heterodyne.optimization.cmc.config import CMCConfig

    monkeypatch.setattr(cmc_core, "_estimate_n_workers", lambda: 8)
    monkeypatch.setattr(
        cmc_core, "_run_joint_pooled_shard_local", lambda p: _trivial_result(**p)
    )

    def _must_not_call(*a, **k):
        raise AssertionError("parallel must not be used for backend_name='cpu'")

    monkeypatch.setattr(
        "heterodyne.optimization.cmc.backends.multiprocessing."
        "run_joint_pooled_shards_parallel",
        _must_not_call,
    )
    cfg = CMCConfig.from_dict({"backend_config": {"backend_name": "cpu"}})
    out = cmc_core._run_joint_shards(_payloads(5), cfg, n_shards=5)
    assert len(out) == 5


@pytest.mark.slow
def test_parallel_pooled_shards_end_to_end(monkeypatch):
    """Real spawn-pool run of the pooled model across worker processes.

    Slow (spawns JAX subprocesses + real NUTS). Verifies the parallel path
    actually builds/samples the pooled model in workers and combines, not
    just the in-process fallback. Tiny iterations keep it bounded.
    """
    # Force >1 worker so the parallel branch is taken regardless of CI cores.
    monkeypatch.setattr(cmc_core, "_estimate_n_workers", lambda: 2)
    data, t1, t2, phi = _make_pooled_arrays(12, [-5.0, 5.0, 90.0])
    result = cmc_core.fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.0054,
        L=2_000_000.0,
        analysis_mode="laminar_flow",
        cmc_config={
            "sharding": {"num_shards": 2},
            "backend_config": {"backend_name": "auto"},
            "per_shard_mcmc": {
                "num_warmup": 20,
                "num_samples": 20,
                "num_chains": 1,
                "dense_mass": False,
                "adaptive_sampling": False,
                "min_warmup": 5,
                "min_samples": 5,
            },
        },
        dt=0.001,
    )
    assert result.num_shards == 2
    assert result.posterior_mean is not None
    assert np.all(np.isfinite(result.posterior_mean))


# ---------------------------------------------------------------------------
# Bounded-index guard for the per-point grid gather (_grid_indices)
# ---------------------------------------------------------------------------


class TestGridIndicesGuard:
    """``_grid_indices`` bounds searchsorted and rejects off-grid time values.

    Pins the OOR hardening: ``np.searchsorted(grid, v)`` returns ``len(grid)``
    for ``v`` at/above the last lag, which is one past the last valid gather
    index. The guard must clip into range, snap floating-point noise to the
    nearest grid point, and raise loudly on genuinely off-grid values.
    """

    def test_exact_meshgrid_values_map_identically(self):
        grid = np.linspace(0.001, 0.064, 7)
        idx = cmc_core._grid_indices(grid, grid, axis="t1")
        assert np.array_equal(idx, np.arange(grid.size))
        assert idx.dtype == np.int32

    def test_irregular_multitau_grid_maps_exactly(self):
        grid = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
        vals = grid[[6, 0, 3, 3, 5]]
        idx = cmc_core._grid_indices(grid, vals, axis="t2")
        assert np.array_equal(idx, np.array([6, 0, 3, 3, 5], dtype=np.int32))

    def test_value_at_or_above_grid_max_is_bounded_not_oor(self):
        grid = np.linspace(0.001, 0.064, 7)
        # A value nudged just above grid[-1] makes raw searchsorted return
        # len(grid). The guard must keep every index <= len(grid) - 1.
        eps_high = np.array([grid[-1] * (1.0 + 1e-12)])
        idx = cmc_core._grid_indices(grid, eps_high, axis="t1")
        assert int(idx[0]) == grid.size - 1
        assert idx.max() < grid.size

    def test_floating_point_noise_snaps_to_nearest(self):
        grid = np.linspace(0.001, 0.064, 7)
        noisy = grid + np.full_like(grid, 1e-9)  # sub-tolerance jitter
        idx = cmc_core._grid_indices(grid, noisy, axis="t1")
        assert np.array_equal(idx, np.arange(grid.size))

    def test_off_grid_value_raises_with_offending_value(self):
        grid = np.linspace(0.001, 0.064, 7)  # spacing 0.0105
        # A lag squarely between two grid points (not a meshgrid tile).
        bad = np.array([grid[2], 0.5 * (grid[3] + grid[4]), grid[5]])
        with pytest.raises(ValueError, match=r"off the time grid"):
            cmc_core._grid_indices(grid, bad, axis="t2")

    def test_single_point_grid_returns_zeros(self):
        grid = np.array([0.001])
        idx = cmc_core._grid_indices(grid, np.array([0.001, 0.001]), axis="t1")
        assert np.array_equal(idx, np.zeros(2, dtype=np.int32))


# ---------------------------------------------------------------------------
# Memory-aware worker cap (OOM guard for concurrent shard NUTS runs)
# ---------------------------------------------------------------------------


def _sized_payloads(n: int, points_per_shard: int = 10_000) -> list[dict]:
    """Payloads carrying realistic per-shard device arrays for sizing."""
    return [
        {
            "data": np.zeros(points_per_shard, dtype=np.float64),
            "i1_indices": np.zeros(points_per_shard, dtype=np.int32),
            "i2_indices": np.zeros(points_per_shard, dtype=np.int32),
            "time_grid": np.zeros(1000, dtype=np.float64),
            "n_phi": 3,
            "result_num_shards": n,
        }
        for _ in range(n)
    ]


class TestMemoryAwareWorkerCap:
    """``_memory_aware_worker_cap`` bounds concurrency by available RAM."""

    def test_abundant_memory_does_not_cap(self, monkeypatch):
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 512 * 1024**3)
        out = cmc_core._memory_aware_worker_cap(_sized_payloads(8), 8, num_chains=4)
        assert out == 8

    def test_tight_memory_caps_below_cpu_workers(self, monkeypatch):
        # ~5 GB available, ~2 GB baseline/worker → budget 4 GB → ~2 workers.
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 5 * 1024**3)
        out = cmc_core._memory_aware_worker_cap(_sized_payloads(16), 16, num_chains=4)
        assert 1 <= out < 16

    def test_extreme_pressure_floors_at_one(self, monkeypatch):
        # Less than one worker's baseline available → must still return 1
        # (routes _run_joint_shards to the sequential in-process path).
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 256 * 1024**2)
        out = cmc_core._memory_aware_worker_cap(_sized_payloads(32), 32, num_chains=4)
        assert out == 1

    def test_unknown_memory_returns_cpu_workers(self, monkeypatch):
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: None)
        out = cmc_core._memory_aware_worker_cap(_sized_payloads(6), 6, num_chains=4)
        assert out == 6

    def test_cap_never_exceeds_cpu_workers(self, monkeypatch):
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 512 * 1024**3)
        out = cmc_core._memory_aware_worker_cap(_sized_payloads(3), 3, num_chains=4)
        assert out == 3

    def test_cap_of_one_routes_sequential(self, monkeypatch):
        """End-to-end: under memory pressure the parallel branch is skipped."""
        from heterodyne.optimization.cmc.config import CMCConfig

        monkeypatch.setattr(cmc_core, "_estimate_n_workers", lambda: 8)
        monkeypatch.setattr(cmc_core, "_available_memory_bytes", lambda: 256 * 1024**2)
        seq = {"n": 0}

        def _local(p):
            seq["n"] += 1
            return _trivial_result(**p)

        monkeypatch.setattr(cmc_core, "_run_joint_pooled_shard_local", _local)

        def _must_not_parallel(*a, **k):
            raise AssertionError("memory cap should have forced sequential dispatch")

        monkeypatch.setattr(
            "heterodyne.optimization.cmc.backends.multiprocessing."
            "run_joint_pooled_shards_parallel",
            _must_not_parallel,
        )
        cfg = CMCConfig.from_dict({"backend_config": {"backend_name": "auto"}})
        out = cmc_core._run_joint_shards(_sized_payloads(6), cfg, n_shards=6)
        assert seq["n"] == 6
        assert len(out) == 6


# ---------------------------------------------------------------------------
# #3: flat fit_mcmc_jax path skips the dense (n_phi, n_t, n_t) reconstruction
# ---------------------------------------------------------------------------


class TestFlatPathSkipsDenseReconstruction:
    """fit_mcmc_jax feeds flat pooled arrays straight into ``_fit_cmc_pooled``
    instead of densifying to (n_phi, n_t, n_t) and re-flattening."""

    @staticmethod
    def _run_capture(monkeypatch, data, t1, t2, phi):
        captured = {}

        def _capture(model, prepared, time_grid, config, nlsq_results):
            captured["prepared"] = prepared
            captured["time_grid"] = np.asarray(time_grid)
            return _trivial_result(n_phi=prepared.n_phi, result_num_shards=1)

        monkeypatch.setattr(cmc_core, "_fit_cmc_pooled", _capture)
        cmc_core.fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.0054,
            L=2_000_000.0,
            analysis_mode="laminar_flow",
            dt=0.001,
            cmc_config={},
        )
        return captured

    def test_routes_through_pooled_engine_with_conserved_values(self, monkeypatch):
        n, angles = 8, [-5.0, 5.0, 90.0]
        data, t1, t2, phi = _make_pooled_arrays(n, angles)
        prepared = self._run_capture(monkeypatch, data, t1, t2, phi)["prepared"]

        # The engine sees exactly the off-diagonal points (i1 != i2), no loss,
        # no duplication, and no dense round-trip.
        t_unique = np.unique(np.concatenate([t1, t2]))
        i1 = np.searchsorted(t_unique, t1)
        i2 = np.searchsorted(t_unique, t2)
        offdiag = data[i1 != i2]
        assert prepared.n_phi == 3
        assert prepared.n_total == offdiag.size == 3 * (n * n - n)
        assert np.allclose(np.sort(prepared.data), np.sort(offdiag))

    def test_prepared_t_values_lie_on_model_grid(self, monkeypatch):
        n, angles = 8, [0.0]
        cap = self._run_capture(monkeypatch, *_make_pooled_arrays(n, angles))
        prepared, model_t = cap["prepared"], cap["time_grid"]
        # Regrid-by-index: every pooled t1/t2 must be an exact model.t value.
        assert np.all(np.isin(prepared.t1, model_t))
        assert np.all(np.isin(prepared.t2, model_t))

    def test_incomplete_grid_raises_coverage_error(self):
        # Keep the full point count but collide two cells (duplicate cell +
        # implied gap) so the size check passes and the coverage mask fails.
        n, angles = 6, [0.0]
        data, t1, t2, phi = _make_pooled_arrays(n, angles)
        t1b, t2b, phib = t1.copy(), t2.copy(), phi.copy()
        t1b[1], t2b[1], phib[1] = t1[0], t2[0], phi[0]  # point 1 duplicates point 0
        with pytest.raises(ValueError, match="does not cover the full"):
            cmc_core.fit_mcmc_jax(
                data=data,
                t1=t1b,
                t2=t2b,
                phi=phib,
                q=0.0054,
                L=2_000_000.0,
                analysis_mode="laminar_flow",
                dt=0.001,
                cmc_config={},
            )
