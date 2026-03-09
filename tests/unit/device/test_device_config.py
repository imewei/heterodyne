"""Unit tests for heterodyne.device.config module.

Tests cluster detection, backend recommendation, hardware configuration,
and device setup using mocks for all system-level and hardware calls.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from heterodyne.device.config import (
    CMCBackend,
    ClusterType,
    HardwareConfig,
    _recommend_backend,
    configure_optimal_device,
    detect_cluster_type,
    detect_hardware,
    get_available_memory,
    get_backend_name,
    get_device_status,
)
from heterodyne.device.cpu import CPUInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def basic_cpu() -> CPUInfo:
    return CPUInfo(physical_cores=8, logical_cores=16, numa_nodes=1)


@pytest.fixture()
def large_cpu() -> CPUInfo:
    return CPUInfo(physical_cores=32, logical_cores=64, numa_nodes=4, vendor="AMD")


@pytest.fixture()
def small_cpu() -> CPUInfo:
    return CPUInfo(physical_cores=2, logical_cores=4, numa_nodes=1)


@pytest.fixture()
def _clean_cluster_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all cluster-related env vars."""
    for key in [
        "PBS_JOBID", "PBS_NODEFILE", "PBS_ENVIRONMENT", "PBS_NCPUS", "NCPUS",
        "SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_CLUSTER_NAME", "SLURM_CPUS_PER_TASK",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# ClusterType / CMCBackend enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_cluster_type_values(self) -> None:
        assert ClusterType.STANDALONE.value == "standalone"
        assert ClusterType.PBS.value == "pbs"
        assert ClusterType.SLURM.value == "slurm"

    def test_cmc_backend_values(self) -> None:
        assert CMCBackend.PJIT.value == "pjit"
        assert CMCBackend.MULTIPROCESSING.value == "multiprocessing"
        assert CMCBackend.PBS.value == "pbs"
        assert CMCBackend.SLURM.value == "slurm"


# ---------------------------------------------------------------------------
# detect_cluster_type
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("_clean_cluster_env")
class TestDetectClusterType:
    def test_standalone_default(self) -> None:
        assert detect_cluster_type() == ClusterType.STANDALONE

    def test_pbs_jobid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PBS_JOBID", "12345")
        assert detect_cluster_type() == ClusterType.PBS

    def test_pbs_nodefile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PBS_NODEFILE", "/var/spool/pbs/nodes")
        assert detect_cluster_type() == ClusterType.PBS

    def test_pbs_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PBS_ENVIRONMENT", "PBS_BATCH")
        assert detect_cluster_type() == ClusterType.PBS

    def test_slurm_jobid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "67890")
        assert detect_cluster_type() == ClusterType.SLURM

    def test_slurm_nodelist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_NODELIST", "node[001-004]")
        assert detect_cluster_type() == ClusterType.SLURM

    def test_slurm_cluster_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "hpc_cluster")
        assert detect_cluster_type() == ClusterType.SLURM

    def test_pbs_takes_precedence_over_slurm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PBS is checked first, so if both are set PBS wins."""
        monkeypatch.setenv("PBS_JOBID", "111")
        monkeypatch.setenv("SLURM_JOB_ID", "222")
        assert detect_cluster_type() == ClusterType.PBS


# ---------------------------------------------------------------------------
# get_available_memory
# ---------------------------------------------------------------------------

class TestGetAvailableMemory:
    def test_with_psutil(self) -> None:
        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3  # 16 GB
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            mem = get_available_memory()
            assert abs(mem - 16.0) < 0.1

    def test_psutil_import_error(self) -> None:
        """When psutil is unavailable, should return 8.0 GB fallback."""
        import builtins

        real_import = builtins.__import__

        def block_psutil(name: str, *args: object, **kwargs: object) -> object:
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_psutil):
            mem = get_available_memory()
            assert mem == 8.0

    def test_psutil_oserror(self) -> None:
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = OSError("Access denied")
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            mem = get_available_memory()
            assert mem == 8.0


# ---------------------------------------------------------------------------
# _recommend_backend
# ---------------------------------------------------------------------------

class TestRecommendBackend:
    def test_pbs_cluster(self, basic_cpu: CPUInfo) -> None:
        backend, chains, max_p = _recommend_backend(
            basic_cpu, ClusterType.PBS, available_cores=8, memory_gb=32.0,
        )
        assert backend == CMCBackend.PBS
        assert chains <= 8
        assert max_p == chains

    def test_slurm_cluster(self, basic_cpu: CPUInfo) -> None:
        backend, chains, max_p = _recommend_backend(
            basic_cpu, ClusterType.SLURM, available_cores=8, memory_gb=32.0,
        )
        assert backend == CMCBackend.SLURM
        assert chains <= 8

    def test_standalone_4plus_cores_uses_pjit(self, basic_cpu: CPUInfo) -> None:
        backend, chains, max_p = _recommend_backend(
            basic_cpu, ClusterType.STANDALONE, available_cores=8, memory_gb=32.0,
        )
        assert backend == CMCBackend.PJIT
        assert chains >= 4

    def test_standalone_few_cores_uses_multiprocessing(self, small_cpu: CPUInfo) -> None:
        backend, chains, max_p = _recommend_backend(
            small_cpu, ClusterType.STANDALONE, available_cores=2, memory_gb=16.0,
        )
        assert backend == CMCBackend.MULTIPROCESSING
        assert chains >= 2  # At least 2 for diagnostics

    def test_memory_limited_chains(self, basic_cpu: CPUInfo) -> None:
        """With very little memory, chains should be capped."""
        backend, chains, max_p = _recommend_backend(
            basic_cpu, ClusterType.STANDALONE, available_cores=8, memory_gb=3.5,
        )
        # 3.5 GB / 3.0 GB per chain = 1 chain by memory
        assert max_p <= 2

    def test_pbs_memory_limited(self, basic_cpu: CPUInfo) -> None:
        backend, chains, _ = _recommend_backend(
            basic_cpu, ClusterType.PBS, available_cores=8, memory_gb=4.0,
        )
        assert chains == 1  # 4.0 / 3.0 = 1

    def test_single_core_standalone(self) -> None:
        cpu = CPUInfo(physical_cores=1, logical_cores=1)
        backend, chains, max_p = _recommend_backend(
            cpu, ClusterType.STANDALONE, available_cores=1, memory_gb=16.0,
        )
        assert backend == CMCBackend.MULTIPROCESSING
        assert chains >= 2  # min 2 for diagnostics
        assert max_p == 1

    def test_large_cluster_caps_at_8_chains(self, large_cpu: CPUInfo) -> None:
        backend, chains, max_p = _recommend_backend(
            large_cpu, ClusterType.STANDALONE, available_cores=32, memory_gb=256.0,
        )
        assert max_p <= 8


# ---------------------------------------------------------------------------
# detect_hardware
# ---------------------------------------------------------------------------

class TestDetectHardware:
    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_standalone_detection(self, basic_cpu: CPUInfo) -> None:
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=16.0),
        ):
            hw = detect_hardware()
            assert hw.cluster_type == ClusterType.STANDALONE
            assert hw.available_cores == 8
            assert hw.memory_gb == 16.0
            assert isinstance(hw.recommended_backend, CMCBackend)

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_pbs_with_ncpus(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PBS_JOBID", "42")
        monkeypatch.setenv("PBS_NCPUS", "24")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=48.0),
        ):
            hw = detect_hardware()
            assert hw.cluster_type == ClusterType.PBS
            assert hw.available_cores == 24

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_pbs_invalid_ncpus_falls_back(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PBS_JOBID", "42")
        monkeypatch.setenv("PBS_NCPUS", "not_a_number")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=16.0),
        ):
            hw = detect_hardware()
            assert hw.available_cores == basic_cpu.physical_cores

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_slurm_cpus_per_task(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=32.0),
        ):
            hw = detect_hardware()
            assert hw.cluster_type == ClusterType.SLURM
            assert hw.available_cores == 16

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_slurm_invalid_cpus_falls_back(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "bad")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=16.0),
        ):
            hw = detect_hardware()
            assert hw.available_cores == basic_cpu.physical_cores

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_slurm_no_cpus_per_task(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=16.0),
        ):
            hw = detect_hardware()
            assert hw.available_cores == basic_cpu.physical_cores

    @pytest.mark.usefixtures("_clean_cluster_env")
    def test_pbs_ncpus_fallback_var(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NCPUS (without PBS_ prefix) should also work."""
        monkeypatch.setenv("PBS_JOBID", "42")
        monkeypatch.setenv("NCPUS", "12")
        with (
            patch("heterodyne.device.config.detect_cpu_info", return_value=basic_cpu),
            patch("heterodyne.device.config.get_available_memory", return_value=32.0),
        ):
            hw = detect_hardware()
            assert hw.available_cores == 12


# ---------------------------------------------------------------------------
# get_backend_name
# ---------------------------------------------------------------------------

class TestGetBackendName:
    def test_pjit(self) -> None:
        assert get_backend_name(CMCBackend.PJIT) == "pjit"

    def test_multiprocessing(self) -> None:
        assert get_backend_name(CMCBackend.MULTIPROCESSING) == "multiprocessing"

    def test_pbs(self) -> None:
        assert get_backend_name(CMCBackend.PBS) == "pbs"

    def test_slurm(self) -> None:
        assert get_backend_name(CMCBackend.SLURM) == "slurm"


# ---------------------------------------------------------------------------
# configure_optimal_device
# ---------------------------------------------------------------------------

class TestConfigureOptimalDevice:
    @pytest.fixture(autouse=True)
    def _mock_hardware(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock detect_hardware and configure_jax_cpu for all tests."""
        self._hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.STANDALONE,
            available_cores=8,
            memory_gb=16.0,
            recommended_backend=CMCBackend.PJIT,
            recommended_chains=4,
            max_parallel_chains=4,
        )
        monkeypatch.setattr(
            "heterodyne.device.config.detect_hardware", lambda: self._hw
        )
        self._configure_calls: list[tuple[CPUInfo, int | None]] = []

        def mock_configure(cpu_info: CPUInfo, num_devices: int | None = None) -> dict[str, str]:
            self._configure_calls.append((cpu_info, num_devices))
            return {}

        monkeypatch.setattr(
            "heterodyne.device.cpu.configure_jax_cpu", mock_configure
        )

    def test_auto_mode(self) -> None:
        hw = configure_optimal_device(mode="auto")
        assert hw is self._hw
        _, num_devices = self._configure_calls[0]
        assert num_devices == 4  # max_parallel_chains

    def test_nlsq_mode(self) -> None:
        configure_optimal_device(mode="nlsq")
        _, num_devices = self._configure_calls[0]
        assert num_devices == 1

    def test_cmc_mode_default(self) -> None:
        configure_optimal_device(mode="cmc")
        _, num_devices = self._configure_calls[0]
        assert num_devices == 4  # min(4, 8)

    def test_cmc_mode_custom_chains(self) -> None:
        configure_optimal_device(mode="cmc", num_chains=6)
        _, num_devices = self._configure_calls[0]
        assert num_devices == 6

    def test_cmc_hpc_mode(self) -> None:
        configure_optimal_device(mode="cmc-hpc")
        _, num_devices = self._configure_calls[0]
        assert num_devices == 8  # min(8, 8)

    def test_cmc_hpc_custom_chains(self) -> None:
        configure_optimal_device(mode="cmc-hpc", num_chains=3)
        _, num_devices = self._configure_calls[0]
        assert num_devices == 3

    def test_auto_with_num_chains(self) -> None:
        configure_optimal_device(mode="auto", num_chains=5)
        _, num_devices = self._configure_calls[0]
        assert num_devices == 5


# ---------------------------------------------------------------------------
# get_device_status
# ---------------------------------------------------------------------------

class TestGetDeviceStatus:
    def test_returns_all_sections(self, basic_cpu: CPUInfo) -> None:
        hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.STANDALONE,
            available_cores=8,
            memory_gb=16.0,
            recommended_backend=CMCBackend.PJIT,
            recommended_chains=4,
            max_parallel_chains=4,
        )
        with patch("heterodyne.device.config.detect_hardware", return_value=hw):
            status = get_device_status()

        assert "cpu" in status
        assert "cluster" in status
        assert "cmc_recommendation" in status
        assert "environment" in status

    def test_cpu_section(self, basic_cpu: CPUInfo) -> None:
        hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.STANDALONE,
            available_cores=8,
            memory_gb=16.0,
            recommended_backend=CMCBackend.PJIT,
            recommended_chains=4,
            max_parallel_chains=4,
        )
        with patch("heterodyne.device.config.detect_hardware", return_value=hw):
            status = get_device_status()

        cpu_status = status["cpu"]
        assert isinstance(cpu_status, dict)
        assert cpu_status["physical_cores"] == 8
        assert cpu_status["logical_cores"] == 16

    def test_cluster_section(self, basic_cpu: CPUInfo) -> None:
        hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.SLURM,
            available_cores=24,
            memory_gb=64.0,
            recommended_backend=CMCBackend.SLURM,
            recommended_chains=8,
            max_parallel_chains=8,
        )
        with patch("heterodyne.device.config.detect_hardware", return_value=hw):
            status = get_device_status()

        cluster = status["cluster"]
        assert isinstance(cluster, dict)
        assert cluster["type"] == "slurm"
        assert cluster["available_cores"] == 24

    def test_cmc_recommendation_section(self, basic_cpu: CPUInfo) -> None:
        hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.STANDALONE,
            available_cores=8,
            memory_gb=16.0,
            recommended_backend=CMCBackend.PJIT,
            recommended_chains=4,
            max_parallel_chains=4,
        )
        with patch("heterodyne.device.config.detect_hardware", return_value=hw):
            status = get_device_status()

        rec = status["cmc_recommendation"]
        assert isinstance(rec, dict)
        assert rec["backend"] == "pjit"
        assert rec["chains"] == 4
        assert rec["max_parallel"] == 4


# ---------------------------------------------------------------------------
# HardwareConfig dataclass
# ---------------------------------------------------------------------------

class TestHardwareConfig:
    def test_construction(self, basic_cpu: CPUInfo) -> None:
        hw = HardwareConfig(
            cpu_info=basic_cpu,
            cluster_type=ClusterType.STANDALONE,
            available_cores=8,
            memory_gb=16.0,
            recommended_backend=CMCBackend.PJIT,
            recommended_chains=4,
            max_parallel_chains=4,
        )
        assert hw.cpu_info is basic_cpu
        assert hw.cluster_type == ClusterType.STANDALONE
        assert hw.available_cores == 8
        assert hw.memory_gb == 16.0
