"""Tests for checkpoint manager integrity enhancements."""

from __future__ import annotations

import json

import numpy as np
import pytest

from heterodyne.optimization.checkpoint_manager import CheckpointData, CheckpointManager


@pytest.fixture
def sample_data() -> CheckpointData:
    """Create a sample checkpoint for testing."""
    return CheckpointData(
        parameters=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        cost=0.5,
        iteration=10,
        metadata={"solver": "lm"},
    )


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Provide a temporary checkpoint directory."""
    return tmp_path / "checkpoints"


class TestCheckpointData:
    """Tests for CheckpointData integrity features."""

    def test_checksum_computed_on_init(self, sample_data: CheckpointData) -> None:
        assert sample_data.checksum != ""
        assert len(sample_data.checksum) == 64  # SHA-256 hex digest

    def test_version_populated(self, sample_data: CheckpointData) -> None:
        assert sample_data.version != ""

    def test_verify_integrity_passes(self, sample_data: CheckpointData) -> None:
        assert sample_data.verify_integrity() is True

    def test_verify_integrity_detects_param_corruption(
        self, sample_data: CheckpointData
    ) -> None:
        sample_data.parameters[0] = 999.0
        assert sample_data.verify_integrity() is False

    def test_verify_integrity_detects_cost_corruption(
        self, sample_data: CheckpointData
    ) -> None:
        sample_data.cost = 999.0
        assert sample_data.verify_integrity() is False

    def test_roundtrip_preserves_checksum(self, sample_data: CheckpointData) -> None:
        d = sample_data.to_dict()
        restored = CheckpointData.from_dict(d)
        assert restored.checksum == sample_data.checksum
        assert restored.verify_integrity() is True

    def test_to_dict_includes_version_and_checksum(
        self, sample_data: CheckpointData
    ) -> None:
        d = sample_data.to_dict()
        assert "version" in d
        assert "checksum" in d
        assert d["checksum"] == sample_data.checksum

    def test_from_dict_legacy_no_checksum(self) -> None:
        """Legacy checkpoints without checksum should get one computed."""
        d = {
            "parameters": [1.0, 2.0],
            "cost": 0.1,
            "iteration": 0,
            "timestamp": "2025-01-01T00:00:00",
        }
        data = CheckpointData.from_dict(d)
        assert data.verify_integrity() is True
        assert data.version == "unknown"

    def test_compute_checksum_deterministic(self) -> None:
        params = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        c1 = CheckpointData.compute_checksum(params, 0.5)
        c2 = CheckpointData.compute_checksum(params, 0.5)
        assert c1 == c2

    def test_compute_checksum_differs_for_different_data(self) -> None:
        params = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        c1 = CheckpointData.compute_checksum(params, 0.5)
        c2 = CheckpointData.compute_checksum(params, 0.6)
        assert c1 != c2


class TestCheckpointManager:
    """Tests for CheckpointManager save/load and integrity."""

    def test_save_and_load(self, checkpoint_dir, sample_data: CheckpointData) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        path = mgr.save(sample_data)
        loaded = mgr.load(path)

        np.testing.assert_array_equal(loaded.parameters, sample_data.parameters)
        assert loaded.cost == sample_data.cost
        assert loaded.verify_integrity() is True

    def test_atomic_write_no_tmp_left(
        self, checkpoint_dir, sample_data: CheckpointData
    ) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        mgr.save(sample_data)
        tmp_files = list(checkpoint_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_load_corrupted_detects_mismatch(
        self, checkpoint_dir, sample_data: CheckpointData
    ) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        path = mgr.save(sample_data)

        # Corrupt the checkpoint by modifying cost in JSON
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["cost"] = 999.0  # Change cost without updating checksum
        path.write_text(json.dumps(raw), encoding="utf-8")

        loaded = mgr.load(path)

        assert loaded.cost == 999.0
        # Checksum was not updated, so integrity should fail
        assert loaded.verify_integrity() is False

    def test_find_latest_valid_skips_corrupted(
        self, checkpoint_dir, sample_data: CheckpointData
    ) -> None:
        mgr = CheckpointManager(checkpoint_dir, max_checkpoints=10)

        # Save two valid checkpoints
        sample_data.iteration = 1
        sample_data.checksum = CheckpointData.compute_checksum(
            sample_data.parameters, sample_data.cost
        )
        mgr.save(sample_data)

        sample_data.iteration = 2
        sample_data.checksum = CheckpointData.compute_checksum(
            sample_data.parameters, sample_data.cost
        )
        path2 = mgr.save(sample_data)

        # Corrupt the latest checkpoint
        raw = json.loads(path2.read_text(encoding="utf-8"))
        raw["cost"] = 999.0
        path2.write_text(json.dumps(raw), encoding="utf-8")

        # Should return iter=1 (the older valid one)
        valid = mgr.find_latest_valid()
        assert valid is not None
        assert valid.iteration == 1
        assert valid.verify_integrity() is True

    def test_find_latest_valid_returns_none_when_all_corrupted(
        self, checkpoint_dir, sample_data: CheckpointData
    ) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        path = mgr.save(sample_data)

        # Corrupt checkpoint
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["parameters"] = [999.0]
        path.write_text(json.dumps(raw), encoding="utf-8")

        assert mgr.find_latest_valid() is None

    def test_find_latest_valid_empty_dir(self, checkpoint_dir) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        assert mgr.find_latest_valid() is None

    def test_load_latest(self, checkpoint_dir, sample_data: CheckpointData) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        mgr.save(sample_data)
        loaded = mgr.load_latest()
        assert loaded is not None
        assert loaded.verify_integrity() is True

    def test_cleanup_respects_max(self, checkpoint_dir) -> None:
        mgr = CheckpointManager(checkpoint_dir, max_checkpoints=2)
        for i in range(5):
            data = CheckpointData(
                parameters=np.array([float(i)]),
                cost=float(i),
                iteration=i,
            )
            mgr.save(data)
        assert len(mgr.list_checkpoints()) == 2
