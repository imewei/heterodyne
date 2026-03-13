"""Tests for worker pool CMC backend."""

from __future__ import annotations

from unittest.mock import patch

from heterodyne.optimization.cmc.backends.worker_pool import (
    WorkerPoolBackend,
    _estimate_physical_workers,
)


class TestWorkerPoolBackend:
    """Tests for WorkerPoolBackend."""

    def test_init_defaults(self) -> None:
        """WorkerPoolBackend initializes with default settings."""
        backend = WorkerPoolBackend()
        assert backend.n_workers >= 1
        assert backend.get_name() == "worker_pool"

    def test_init_custom_workers(self) -> None:
        """WorkerPoolBackend respects custom n_workers."""
        backend = WorkerPoolBackend(n_workers=2)
        assert backend.n_workers == 2

    def test_should_use_pool_threshold(self) -> None:
        """Pool is used when n_shards >= max(3, n_workers)."""
        assert WorkerPoolBackend.should_use_pool(n_shards=3, n_workers=2)
        assert not WorkerPoolBackend.should_use_pool(n_shards=1, n_workers=2)

    def test_should_use_pool_many_workers(self) -> None:
        """Pool requires enough shards to distribute across workers."""
        assert not WorkerPoolBackend.should_use_pool(n_shards=3, n_workers=8)
        assert WorkerPoolBackend.should_use_pool(n_shards=8, n_workers=8)


class TestEstimatePhysicalWorkers:
    """Tests for _estimate_physical_workers helper."""

    def test_uses_physical_cores(self) -> None:
        """Detects physical cores and reserves one for main process."""
        from heterodyne.device.cpu import CPUInfo

        mock_info = CPUInfo(physical_cores=8, logical_cores=16)
        with patch(
            "heterodyne.device.cpu.detect_cpu_info",
            return_value=mock_info,
        ):
            result = _estimate_physical_workers()
            assert result == 7  # 8 physical - 1 reserved

    def test_minimum_one_worker(self) -> None:
        """Always returns at least 1 worker even with single core."""
        from heterodyne.device.cpu import CPUInfo

        mock_info = CPUInfo(physical_cores=1, logical_cores=1)
        with patch(
            "heterodyne.device.cpu.detect_cpu_info",
            return_value=mock_info,
        ):
            result = _estimate_physical_workers()
            assert result >= 1

    def test_fallback_on_detection_failure(self) -> None:
        """Falls back to os.cpu_count() // 2 if detect_cpu_info fails."""
        with (
            patch(
                "heterodyne.device.cpu.detect_cpu_info",
                side_effect=RuntimeError("detection failed"),
            ),
            patch("os.cpu_count", return_value=16),
        ):
            result = _estimate_physical_workers()
            assert result == 7  # 16 // 2 - 1
