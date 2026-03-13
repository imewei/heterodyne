"""Integration tests for HDF5 data I/O and the XPCS loading pipeline.

Bug Prevented: Data I/O Integrity Issues
-----------------------------------------
Data loading is the first step of the analysis pipeline.  Bugs here
cascade silently into fitting and inference if not caught early:
- Corrupted or incomplete HDF5 files producing wrong shapes
- NaN / Inf leaking into the correlation matrix
- Non-monotonic time arrays breaking physics assumptions
- Checkpoint corruption going undetected after a crash
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from heterodyne.data.xpcs_loader import (
    DataValidationError,
    XPCSData,
    XPCSDataLoader,
    load_xpcs_batch,
    load_xpcs_data,
    probe_hdf5_structure,
    validate_loaded_data,
)
from heterodyne.optimization.checkpoint_manager import (
    CheckpointData,
    CheckpointManager,
)
from heterodyne.utils.path_validation import PathValidationError

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symmetric_c2(n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a valid symmetric positive-diagonal c2 matrix."""
    if rng is None:
        rng = np.random.default_rng(42)
    raw = rng.standard_normal((n, n))
    c2 = raw @ raw.T + np.eye(n) * n  # positive diagonal guaranteed
    return c2


def _write_hdf5_flat(
    path: Path,
    c2: np.ndarray,
    t: np.ndarray,
    *,
    q: float | None = None,
    phi: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a flat-layout HDF5 file for testing."""
    with h5py.File(path, "w") as f:
        f.create_dataset("c2", data=c2)
        f.create_dataset("t", data=t)
        if q is not None:
            f.create_dataset("q", data=np.array([q]))
        if phi is not None:
            f.create_dataset("phi", data=phi)
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v


def _write_hdf5_exchange(
    path: Path,
    c2: np.ndarray,
    t: np.ndarray,
    *,
    q_values: np.ndarray | None = None,
) -> None:
    """Write an APS-style /exchange/ layout HDF5 file."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("exchange")
        grp.create_dataset("twotime_corr", data=c2)
        grp.create_dataset("tau", data=t)
        if q_values is not None:
            grp.create_dataset("q_values", data=q_values)


# ---------------------------------------------------------------------------
# TestDataLoading
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDataLoading:
    """Tests for XPCS HDF5 data loading."""

    def test_load_flat_hdf5_round_trip(self, tmp_path: Path) -> None:
        """Load a flat-layout HDF5 file and verify data fidelity."""
        n = 16
        rng = np.random.default_rng(0)
        c2 = _make_symmetric_c2(n, rng=rng)
        t = np.linspace(0.0, 1.0, n)
        q_val = 0.05

        h5_path = tmp_path / "test_flat.h5"
        _write_hdf5_flat(h5_path, c2, t, q=q_val, attrs={"beamline": "APS-8ID"})

        data = load_xpcs_data(h5_path)

        np.testing.assert_array_equal(data.c2, c2.astype(np.float64))
        np.testing.assert_array_equal(data.t1, t.astype(np.float64))
        np.testing.assert_array_equal(data.t2, t.astype(np.float64))
        assert data.q == pytest.approx(q_val)
        assert data.metadata["beamline"] == "APS-8ID"

    def test_load_exchange_layout(self, tmp_path: Path) -> None:
        """Load an APS-style /exchange/ HDF5 file."""
        n = 10
        c2 = _make_symmetric_c2(n)
        t = np.arange(n, dtype=np.float64)

        h5_path = tmp_path / "exchange_test.h5"
        _write_hdf5_exchange(h5_path, c2, t)

        data = load_xpcs_data(h5_path)

        np.testing.assert_array_equal(data.c2, c2.astype(np.float64))
        np.testing.assert_array_equal(data.t1, t)
        assert data.shape == (n, n)

    def test_load_3d_multi_q(self, tmp_path: Path) -> None:
        """Load 3-D c2 with per-q-bin wavevectors."""
        n_q, n_t = 3, 8
        rng = np.random.default_rng(7)
        c2 = np.stack([_make_symmetric_c2(n_t, rng=rng) for _ in range(n_q)])
        t = np.linspace(0.0, 2.0, n_t)
        q_values = np.array([0.01, 0.02, 0.03])

        h5_path = tmp_path / "multi_q.h5"
        _write_hdf5_exchange(h5_path, c2, t, q_values=q_values)

        data = load_xpcs_data(h5_path)

        assert data.c2.shape == (n_q, n_t, n_t)
        assert data.has_multi_q
        np.testing.assert_array_almost_equal(data.q_values, q_values)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing file must raise, not return empty data."""
        missing = tmp_path / "does_not_exist.h5"
        with pytest.raises((FileNotFoundError, OSError, PathValidationError)):
            load_xpcs_data(missing)

    def test_load_missing_c2_key_raises(self, tmp_path: Path) -> None:
        """KeyError when requested c2 key is absent."""
        h5_path = tmp_path / "no_c2.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("wrong_key", data=np.ones((4, 4)))

        with pytest.raises(KeyError, match="c2"):
            load_xpcs_data(h5_path)

    def test_load_npz_round_trip(self, tmp_path: Path) -> None:
        """NPZ format round-trip preserves data."""
        n = 12
        c2 = _make_symmetric_c2(n)
        t = np.linspace(0.1, 5.0, n)

        npz_path = tmp_path / "test_data.npz"
        np.savez(npz_path, c2=c2, t=t, q=np.array([0.07]))

        data = load_xpcs_data(npz_path)

        np.testing.assert_array_almost_equal(data.c2, c2)
        np.testing.assert_array_almost_equal(data.t1, t)
        assert data.q == pytest.approx(0.07)

    def test_load_npy_round_trip(self, tmp_path: Path) -> None:
        """NPY format loads array-only data correctly."""
        n = 8
        c2 = _make_symmetric_c2(n)

        npy_path = tmp_path / "test_data.npy"
        np.save(npy_path, c2)

        data = load_xpcs_data(npy_path)

        np.testing.assert_array_almost_equal(data.c2, c2)
        assert data.n_times == n

    def test_batch_loading_skips_failures(self, tmp_path: Path) -> None:
        """Batch loader skips files that fail and continues."""
        n = 6
        c2 = _make_symmetric_c2(n)
        t = np.arange(n, dtype=np.float64)

        good = tmp_path / "good.h5"
        _write_hdf5_flat(good, c2, t)

        bad = tmp_path / "bad.h5"
        bad.write_bytes(b"not a real hdf5 file")

        results = load_xpcs_batch([good, bad])

        assert len(results) == 1
        np.testing.assert_array_equal(results[0].c2, c2.astype(np.float64))

    def test_probe_hdf5_structure(self, tmp_path: Path) -> None:
        """probe_hdf5_structure reports datasets, groups, and attributes."""
        n = 4
        c2 = _make_symmetric_c2(n)
        t = np.arange(n, dtype=np.float64)

        h5_path = tmp_path / "probe_test.h5"
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("data")
            grp.create_dataset("c2", data=c2)
            f.create_dataset("t", data=t)
            f.attrs["experiment"] = "test"

        info = probe_hdf5_structure(h5_path)

        assert info["n_datasets"] == 2
        assert info["n_groups"] == 1
        assert info["root_attrs"]["experiment"] == "test"
        dataset_paths = [d["path"] for d in info["datasets"]]
        assert "/data/c2" in dataset_paths
        assert "/t" in dataset_paths

    def test_auto_detect_format(self, tmp_path: Path) -> None:
        """XPCSDataLoader auto-detects format from extension."""
        n = 4
        c2 = _make_symmetric_c2(n)
        t = np.arange(n, dtype=np.float64)

        for ext, fmt in [(".h5", "hdf5"), (".hdf5", "hdf5"), (".npz", None)]:
            if ext in (".h5", ".hdf5"):
                path = tmp_path / f"data{ext}"
                _write_hdf5_flat(path, c2, t)
            else:
                path = tmp_path / f"data{ext}"
                np.savez(path, c2=c2, t=t)

            loader = XPCSDataLoader(path)
            if fmt is not None:
                assert loader.format == fmt
            data = loader.load()
            assert data.c2.shape == (n, n)

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        """Unknown file extension raises ValueError."""
        path = tmp_path / "data.xyz"
        path.write_text("not real data")
        with pytest.raises(ValueError, match="Unknown file format"):
            XPCSDataLoader(path)

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        """NPZ cache accelerates second load and preserves data."""
        n = 10
        c2 = _make_symmetric_c2(n)
        t = np.linspace(0.0, 1.0, n)
        q_val = 0.04

        h5_path = tmp_path / "cached.h5"
        _write_hdf5_flat(h5_path, c2, t, q=q_val)

        # First load creates cache
        data1 = load_xpcs_data(h5_path, use_cache=True)
        cache_file = h5_path.with_name(h5_path.name + ".heterodyne_cache.npz")
        assert cache_file.exists()

        # Second load reads from cache
        data2 = load_xpcs_data(h5_path, use_cache=True)

        np.testing.assert_array_equal(data1.c2, data2.c2)
        np.testing.assert_array_equal(data1.t1, data2.t1)


# ---------------------------------------------------------------------------
# TestCheckpointIO
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCheckpointIO:
    """Tests for optimization checkpoint save/load."""

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Save and load a checkpoint; values must survive the trip."""
        params = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ckpt = CheckpointData(
            parameters=params,
            cost=0.123,
            iteration=42,
            metadata={"solver": "lm"},
        )

        mgr = CheckpointManager(tmp_path / "ckpts")
        saved_path = mgr.save(ckpt)

        loaded = mgr.load(saved_path)

        np.testing.assert_array_equal(loaded.parameters, params)
        assert loaded.cost == pytest.approx(0.123)
        assert loaded.iteration == 42
        assert loaded.metadata["solver"] == "lm"

    def test_checkpoint_integrity_verification(self, tmp_path: Path) -> None:
        """Checksum validates untampered checkpoint."""
        params = np.linspace(0.0, 1.0, 14)
        ckpt = CheckpointData(parameters=params, cost=0.5, iteration=0)

        assert ckpt.verify_integrity()

        expected = CheckpointData.compute_checksum(params, 0.5)
        assert ckpt.checksum == expected

    def test_corrupted_checkpoint_detected(self, tmp_path: Path) -> None:
        """Tampered JSON file fails integrity check."""
        params = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ckpt = CheckpointData(parameters=params, cost=1.0, iteration=0)

        mgr = CheckpointManager(tmp_path / "ckpts")
        saved_path = mgr.save(ckpt)

        # Tamper with the cost value in the JSON
        raw = json.loads(saved_path.read_text(encoding="utf-8"))
        raw["cost"] = 999.0  # change cost without updating checksum
        saved_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        loaded = mgr.load(saved_path)
        assert not loaded.verify_integrity()

    def test_find_latest_valid_skips_corrupted(self, tmp_path: Path) -> None:
        """find_latest_valid returns the most recent uncorrupted checkpoint."""
        ckpt_dir = tmp_path / "ckpts"
        mgr = CheckpointManager(ckpt_dir, max_checkpoints=10)

        # Save two good checkpoints
        good1 = CheckpointData(
            parameters=np.array([1.0]),
            cost=0.1,
            iteration=0,
        )
        good2 = CheckpointData(
            parameters=np.array([2.0]),
            cost=0.2,
            iteration=1,
        )
        mgr.save(good1)
        path2 = mgr.save(good2)

        # Corrupt the latest one
        raw = json.loads(path2.read_text(encoding="utf-8"))
        raw["parameters"] = [999.0]
        path2.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        found = mgr.find_latest_valid()

        assert found is not None
        assert found.iteration == 0
        np.testing.assert_array_equal(found.parameters, np.array([1.0]))

    def test_load_latest_empty_dir(self, tmp_path: Path) -> None:
        """load_latest returns None when no checkpoints exist."""
        mgr = CheckpointManager(tmp_path / "empty_ckpts")
        assert mgr.load_latest() is None

    def test_checkpoint_retention_limit(self, tmp_path: Path) -> None:
        """Old checkpoints are pruned to respect max_checkpoints."""
        ckpt_dir = tmp_path / "retention"
        mgr = CheckpointManager(ckpt_dir, max_checkpoints=2)

        for i in range(5):
            ckpt = CheckpointData(
                parameters=np.array([float(i)]),
                cost=float(i),
                iteration=i,
            )
            mgr.save(ckpt)

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2

        # Most recent iterations survive
        latest = mgr.load_latest()
        assert latest is not None
        assert latest.iteration == 4

    def test_invalid_json_checkpoint(self, tmp_path: Path) -> None:
        """Non-JSON file raises JSONDecodeError on load."""
        ckpt_dir = tmp_path / "bad_json"
        ckpt_dir.mkdir()
        bad_path = ckpt_dir / "checkpoint_iter000000_20260101T000000.json"
        bad_path.write_text("{{{not json", encoding="utf-8")

        mgr = CheckpointManager(ckpt_dir)
        with pytest.raises(json.JSONDecodeError):
            mgr.load(bad_path)

    def test_atomic_write_leaves_no_tmp(self, tmp_path: Path) -> None:
        """After save, no .tmp files remain in the checkpoint directory."""
        ckpt_dir = tmp_path / "atomic"
        mgr = CheckpointManager(ckpt_dir)

        ckpt = CheckpointData(
            parameters=np.array([1.0, 2.0]),
            cost=0.5,
            iteration=0,
        )
        mgr.save(ckpt)

        tmp_files = list(ckpt_dir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# TestDataValidation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDataValidation:
    """Tests for XPCS data validation checks."""

    def test_valid_data_passes(self) -> None:
        """Clean synthetic data passes all checks."""
        n = 8
        c2 = _make_symmetric_c2(n)
        t = np.linspace(0.0, 1.0, n)
        data = XPCSData(c2=c2, t1=t, t2=t)

        warnings = validate_loaded_data(data)
        # May have warnings (e.g. mild asymmetry) but should not raise
        assert isinstance(warnings, list)

    def test_nan_in_c2_raises(self) -> None:
        """NaN in c2 triggers DataValidationError."""
        n = 8
        c2 = _make_symmetric_c2(n)
        c2[2, 3] = np.nan
        t = np.linspace(0.0, 1.0, n)
        data = XPCSData(c2=c2, t1=t, t2=t)

        with pytest.raises(DataValidationError, match="non-finite"):
            validate_loaded_data(data)

    def test_inf_in_t1_raises(self) -> None:
        """Inf in time array triggers DataValidationError."""
        n = 8
        c2 = _make_symmetric_c2(n)
        t = np.linspace(0.0, 1.0, n)
        t_bad = t.copy()
        t_bad[4] = np.inf
        data = XPCSData(c2=c2, t1=t_bad, t2=t)

        with pytest.raises(DataValidationError, match="non-finite"):
            validate_loaded_data(data)

    def test_shape_mismatch_raises(self) -> None:
        """t1 length mismatching c2 rows raises DataValidationError."""
        n = 8
        c2 = _make_symmetric_c2(n)
        t_short = np.linspace(0.0, 1.0, n - 2)
        data = XPCSData(c2=c2, t1=t_short, t2=np.linspace(0.0, 1.0, n))

        with pytest.raises(DataValidationError, match="does not match"):
            validate_loaded_data(data)

    def test_q_values_shape_mismatch_3d(self) -> None:
        """q_values length mismatching c2 q-axis raises error."""
        n_q, n_t = 3, 6
        rng = np.random.default_rng(11)
        c2 = np.stack([_make_symmetric_c2(n_t, rng=rng) for _ in range(n_q)])
        t = np.linspace(0.0, 1.0, n_t)
        q_wrong = np.array([0.01, 0.02])  # length 2, not 3

        data = XPCSData(c2=c2, t1=t, t2=t, q_values=q_wrong)

        with pytest.raises(DataValidationError, match="q_values length"):
            validate_loaded_data(data)

    def test_non_monotonic_time_raises(self) -> None:
        """Non-monotonic t1 triggers DataValidationError."""
        n = 8
        c2 = _make_symmetric_c2(n)
        t = np.linspace(0.0, 1.0, n)
        t_bad = t.copy()
        t_bad[3] = t_bad[2] - 0.001  # break monotonicity
        data = XPCSData(c2=c2, t1=t_bad, t2=t)

        with pytest.raises(DataValidationError, match="not strictly increasing"):
            validate_loaded_data(data)

    def test_non_positive_diagonal_raises(self) -> None:
        """Zero or negative diagonal in c2 triggers DataValidationError."""
        n = 8
        c2 = _make_symmetric_c2(n)
        c2[0, 0] = -1.0  # negative diagonal
        t = np.linspace(0.0, 1.0, n)
        data = XPCSData(c2=c2, t1=t, t2=t)

        with pytest.raises(DataValidationError, match="non-positive diagonal"):
            validate_loaded_data(data)

    def test_asymmetry_warning(self) -> None:
        """Asymmetric c2 produces a warning but does not raise."""
        n = 8
        c2 = _make_symmetric_c2(n)
        c2[0, 1] += 1e3  # break symmetry significantly
        c2[1, 0] -= 1e3
        t = np.linspace(0.0, 1.0, n)
        data = XPCSData(c2=c2, t1=t, t2=t)

        warnings = validate_loaded_data(data)
        assert any("not symmetric" in w for w in warnings)

    def test_1d_c2_raises(self) -> None:
        """1-D array for c2 is rejected."""
        data = XPCSData(
            c2=np.ones(8),
            t1=np.arange(8, dtype=np.float64),
            t2=np.arange(8, dtype=np.float64),
        )
        with pytest.raises(DataValidationError, match="2-D or 3-D"):
            validate_loaded_data(data)

    def test_validate_inside_load_pipeline(self, tmp_path: Path) -> None:
        """End-to-end: load HDF5 then validate catches NaN."""
        n = 8
        c2 = _make_symmetric_c2(n)
        c2[1, 1] = np.nan
        t = np.linspace(0.0, 1.0, n)

        h5_path = tmp_path / "bad_data.h5"
        _write_hdf5_flat(h5_path, c2, t)

        data = load_xpcs_data(h5_path)
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_loaded_data(data)

    def test_batch_load_with_validation(self, tmp_path: Path) -> None:
        """Batch loader with validate=True skips invalid files."""
        n = 6
        # Good file
        c2_good = _make_symmetric_c2(n)
        t = np.linspace(0.1, 1.0, n)
        good_path = tmp_path / "good.h5"
        _write_hdf5_flat(good_path, c2_good, t)

        # Bad file: contains NaN
        c2_bad = _make_symmetric_c2(n)
        c2_bad[0, 0] = np.nan
        bad_path = tmp_path / "bad.h5"
        _write_hdf5_flat(bad_path, c2_bad, t)

        results = load_xpcs_batch([good_path, bad_path], validate=True)

        assert len(results) == 1
        np.testing.assert_array_equal(results[0].c2, c2_good.astype(np.float64))
