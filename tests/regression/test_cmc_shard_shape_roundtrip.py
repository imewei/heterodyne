"""Regression tests for the self-describing shared-memory schema (codex C2 + gemini G3).

The Tier 2 rewrite replaces the legacy ``_SHARD_ARRAY_KEYS`` tuple with a
``_SHARD_ARRAY_SPECS`` dict of :class:`ArraySpec` objects, and stores
per-shard ``shape`` metadata in every ref dict so the worker reshapes on
load instead of silently inheriting a 1-D view.

These tests pin the schema contracts:

1. **Round-trip of rectangular 2-D ``c2_data``** — a `(M, N)` array with
   ``M != N`` must come out the other side with shape `(M, N)`, not
   ``(M*N,)``.  This is the het_7221ba99 silent-shape-loss regression guard.
2. **Element-wise path preservation** — shards built with t1/t2/time_grid
   must round-trip those 1-D arrays unchanged so the worker still
   dispatches to ``compute_c2_elementwise`` (proxy for the in-closure
   dispatch, see ``# CONTRACT:`` comment at the dispatch site).
3. **Shape/size mismatch is loud** — a hand-crafted ref dict where
   ``shape`` and ``size`` disagree must raise ``ValueError`` with a
   descriptive message, not an obscure downstream ``IndexError``.
4. **Missing ``shape`` key is loud** — a ref dict missing the ``shape``
   key (e.g. from a stale serialised payload) must raise ``KeyError``.
   This locks in the assumption that shard refs are RAM-only and never
   replayed across versions; if that ever changes, the PR that adds
   serialisation must consciously decide about schema versioning.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.cmc.backends.multiprocessing import (
    _SHARD_ARRAY_SPECS,
    ArraySpec,
    SharedDataManager,
    _load_shared_shard_data,
)

# ---------------------------------------------------------------------------
# Test 1: round-trip 2-D rectangular c2_data
# ---------------------------------------------------------------------------


class TestShardShapeRoundTrip:
    """A 2-D rectangular c2_data must survive pack→unpack with shape intact."""

    def test_rectangular_c2_data_shape_preserved(self) -> None:
        c2 = np.arange(101 * 47, dtype=np.float64).reshape(101, 47)
        shard = {
            "c2_data": c2,
            "noise_scale": 0.1,
        }
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays([shard])
            assert len(refs) == 1
            loaded = _load_shared_shard_data(refs[0])

        assert loaded["c2_data"].shape == (101, 47)
        np.testing.assert_array_equal(loaded["c2_data"], c2)

    def test_square_c2_data_shape_preserved(self) -> None:
        c2 = np.arange(50 * 50, dtype=np.float64).reshape(50, 50)
        shard = {"c2_data": c2, "noise_scale": 0.1}
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays([shard])
            loaded = _load_shared_shard_data(refs[0])
        assert loaded["c2_data"].shape == (50, 50)
        np.testing.assert_array_equal(loaded["c2_data"], c2)

    def test_multiple_shards_independent_shapes(self) -> None:
        shards = [
            {
                "c2_data": np.arange(10 * 20, dtype=np.float64).reshape(10, 20),
                "noise_scale": 0.1,
            },
            {
                "c2_data": np.arange(30 * 7, dtype=np.float64).reshape(30, 7),
                "noise_scale": 0.2,
            },
            {
                "c2_data": np.arange(5 * 5, dtype=np.float64).reshape(5, 5),
                "noise_scale": 0.3,
            },
        ]
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays(shards)
            loaded = [_load_shared_shard_data(r) for r in refs]
        assert loaded[0]["c2_data"].shape == (10, 20)
        assert loaded[1]["c2_data"].shape == (30, 7)
        assert loaded[2]["c2_data"].shape == (5, 5)
        for orig, got in zip(shards, loaded, strict=True):
            np.testing.assert_array_equal(got["c2_data"], orig["c2_data"])


# ---------------------------------------------------------------------------
# Test 2: element-wise (t1/t2/time_grid) path preservation — proxy for dispatch
# ---------------------------------------------------------------------------


class TestShardDispatch:
    """Element-wise shards must round-trip t1/t2/time_grid unchanged.

    The actual worker-dispatch (``if _shard_grid is not None``) lives
    inside a NumPyro model closure that's not directly mockable without
    significant scaffolding.  See the ``# CONTRACT:`` comment at the
    dispatch site in multiprocessing_backend.py.  This test pins the
    necessary precondition: pack/unpack must preserve the 1-D arrays
    that drive the dispatch.
    """

    def test_t1_t2_time_grid_preserved_through_roundtrip(self) -> None:
        n_pairs = 1003  # prime to catch any accidental power-of-2 reshape
        t1 = np.arange(n_pairs, dtype=np.float64)
        t2 = np.arange(n_pairs, dtype=np.float64) * 2.0
        time_grid = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        c2 = np.random.default_rng(42).normal(size=(n_pairs,)).astype(np.float64)
        shard = {
            "c2_data": c2,
            "t1": t1,
            "t2": t2,
            "time_grid": time_grid,
            "noise_scale": 0.1,
        }
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays([shard])
            loaded = _load_shared_shard_data(refs[0])
        # Element-wise path's dispatch precondition: all three keys
        # present and non-None with their original 1-D shapes.
        assert loaded["t1"] is not None
        assert loaded["t2"] is not None
        assert loaded["time_grid"] is not None
        assert loaded["t1"].shape == (n_pairs,)
        assert loaded["t2"].shape == (n_pairs,)
        assert loaded["time_grid"].shape == (256,)
        np.testing.assert_array_equal(loaded["t1"], t1)
        np.testing.assert_array_equal(loaded["t2"], t2)
        np.testing.assert_array_equal(loaded["time_grid"], time_grid)
        # c2_data is 1-D for element-wise shards (n_pairs,), not (M, N).
        assert loaded["c2_data"].shape == (n_pairs,)


# ---------------------------------------------------------------------------
# Test 3: shape/size mismatch raises ValueError with descriptive message
# ---------------------------------------------------------------------------


class TestSchemaContracts:
    """Malformed shard refs must fail loud with descriptive errors."""

    def test_shape_size_mismatch_raises_valuerror(self) -> None:
        # Pack a real (10,10) c2_data so the SHM segment is valid…
        c2 = np.arange(100, dtype=np.float64).reshape(10, 10)
        shard = {"c2_data": c2, "noise_scale": 0.1}
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays([shard])
            # …but then forge an inconsistent shape on the ref.
            refs[0]["c2_data"]["shape"] = (5, 5)  # prod=25, but size=100
            with pytest.raises(ValueError) as exc:
                _load_shared_shard_data(refs[0])
        msg = str(exc.value)
        assert "c2_data" in msg
        assert "two-time correlation matrix" in msg  # ArraySpec.description
        assert "100" in msg  # actual size
        assert "(5, 5)" in msg  # declared shape

    def test_ref_without_shape_raises_keyerror(self) -> None:
        """Shard refs must carry the ``shape`` key; legacy refs must NOT load.

        Locks in the assumption that shard refs are RAM-only and never
        serialised across versions (per design notes — checkpoints store
        CMCResult and posterior samples, not shard refs).
        """
        c2 = np.arange(12, dtype=np.float64).reshape(3, 4)
        shard = {"c2_data": c2, "noise_scale": 0.1}
        with SharedDataManager() as mgr:
            refs = mgr.create_shared_shard_arrays([shard])
            del refs[0]["c2_data"]["shape"]
            with pytest.raises(KeyError) as exc:
                _load_shared_shard_data(refs[0])
        assert "c2_data" in str(exc.value)
        assert "shape" in str(exc.value)


# ---------------------------------------------------------------------------
# ArraySpec sanity
# ---------------------------------------------------------------------------


class TestArraySpec:
    def test_specs_cover_all_known_keys(self) -> None:
        # Both element-wise and contiguous shard formats are covered.
        for key in ("c2_data", "sigma", "t", "t1", "t2", "time_grid", "weights"):
            assert key in _SHARD_ARRAY_SPECS
            spec = _SHARD_ARRAY_SPECS[key]
            assert isinstance(spec, ArraySpec)
            assert spec.expected_dtype  # non-empty
            assert spec.description  # non-empty

    def test_required_keys_disallow_none(self) -> None:
        # c2_data is the only key that must not be None — the others may
        # legitimately be absent for some shard formats.
        assert _SHARD_ARRAY_SPECS["c2_data"].allow_none is False
        assert _SHARD_ARRAY_SPECS["t1"].allow_none is True
        assert _SHARD_ARRAY_SPECS["t2"].allow_none is True
        assert _SHARD_ARRAY_SPECS["time_grid"].allow_none is True
