"""Unit tests for heterodyne.device.cpu module.

Tests CPU detection, XLA flag configuration, NUMA awareness,
and edge cases using mocks for all system-level calls.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from heterodyne.device.cpu import (
    CPUInfo,
    _detect_fallback_cpu,
    _detect_linux_cpu,
    _detect_macos_cpu,
    _parse_lscpu,
    _safe_int,
    configure_cpu_hpc,
    configure_jax_cpu,
    detect_cpu_info,
    get_jax_cpu_flags,
    get_optimal_batch_size,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_cpu() -> CPUInfo:
    """A minimal CPUInfo for unit tests."""
    return CPUInfo(physical_cores=8, logical_cores=16)


@pytest.fixture()
def intel_cpu() -> CPUInfo:
    """An Intel CPU with NUMA and AVX-512."""
    return CPUInfo(
        physical_cores=16,
        logical_cores=32,
        numa_nodes=2,
        architecture="x86_64",
        vendor="Intel",
        model_name="Intel Xeon Gold 6248",
        has_avx=True,
        has_avx2=True,
        has_avx512=True,
        cache_sizes={"L1": 32768, "L2": 1048576, "L3": 33554432},
    )


@pytest.fixture()
def amd_cpu() -> CPUInfo:
    """An AMD CPU with NUMA and AVX2."""
    return CPUInfo(
        physical_cores=64,
        logical_cores=128,
        numa_nodes=4,
        architecture="x86_64",
        vendor="AMD",
        model_name="AMD EPYC 7742",
        has_avx=True,
        has_avx2=True,
        has_avx512=False,
    )


@pytest.fixture()
def single_core_cpu() -> CPUInfo:
    """A single-core CPU (edge case)."""
    return CPUInfo(physical_cores=1, logical_cores=1, numa_nodes=1)


@pytest.fixture()
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove threading and XLA env vars before each test."""
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NESTED",
        "OMP_MAX_ACTIVE_LEVELS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_plain_number(self) -> None:
        assert _safe_int("8") == 8

    def test_number_with_annotation(self) -> None:
        assert _safe_int("4 (2 online)") == 4

    def test_empty_string(self) -> None:
        assert _safe_int("") is None

    def test_non_numeric(self) -> None:
        assert _safe_int("abc") is None


# ---------------------------------------------------------------------------
# _parse_lscpu
# ---------------------------------------------------------------------------


class TestParseLscpu:
    LSCPU_OUTPUT = (
        "Architecture:          x86_64\n"
        "CPU(s):                32\n"
        "Core(s) per socket:    8\n"
        "Socket(s):             2\n"
        "NUMA node(s):          2\n"
        "Vendor ID:             GenuineIntel\n"
        "Model name:            Intel Xeon\n"
    )

    def test_parses_cores_and_sockets(self) -> None:
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(self.LSCPU_OUTPUT, info)
        assert result.physical_cores == 16  # 8 cores * 2 sockets
        assert result.logical_cores == 32

    def test_parses_numa_nodes(self) -> None:
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(self.LSCPU_OUTPUT, info)
        assert result.numa_nodes == 2

    def test_parses_architecture(self) -> None:
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(self.LSCPU_OUTPUT, info)
        assert result.architecture == "x86_64"

    def test_parses_intel_vendor(self) -> None:
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(self.LSCPU_OUTPUT, info)
        assert result.vendor == "Intel"

    def test_parses_amd_vendor(self) -> None:
        output = "Vendor ID:             AuthenticAMD\n"
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(output, info)
        assert result.vendor == "AMD"

    def test_parses_unknown_vendor(self) -> None:
        output = "Vendor ID:             HygonGenuine\n"
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(output, info)
        assert result.vendor == "HygonGenuine"

    def test_missing_sockets_keeps_physical_cores(self) -> None:
        output = "Core(s) per socket:    8\n"
        info = CPUInfo(physical_cores=99, logical_cores=1)
        result = _parse_lscpu(output, info)
        # sockets is None, so physical_cores should remain unchanged
        assert result.physical_cores == 99

    def test_empty_output(self) -> None:
        info = CPUInfo(physical_cores=4, logical_cores=8)
        result = _parse_lscpu("", info)
        assert result.physical_cores == 4
        assert result.logical_cores == 8

    def test_no_colon_lines_skipped(self) -> None:
        output = "=== CPU Info ===\nArchitecture:  x86_64\n"
        info = CPUInfo(physical_cores=1, logical_cores=1)
        result = _parse_lscpu(output, info)
        assert result.architecture == "x86_64"


# ---------------------------------------------------------------------------
# _detect_linux_cpu
# ---------------------------------------------------------------------------


class TestDetectLinuxCpu:
    CPUINFO_CONTENT = (
        "vendor_id\t: GenuineIntel\n"
        "model name\t: Intel Xeon Gold 6248\n"
        "flags\t\t: fpu avx avx2 avx512f\n"
    )

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=16)
    @patch("heterodyne.device.cpu.subprocess.run")
    @patch("builtins.open", mock_open(read_data=CPUINFO_CONTENT))
    def test_full_detection(
        self, mock_run: MagicMock, mock_cpu_count: MagicMock, mock_machine: MagicMock
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Architecture:          x86_64\n"
                "CPU(s):                16\n"
                "Core(s) per socket:    8\n"
                "Socket(s):             1\n"
                "NUMA node(s):          1\n"
            ),
        )
        info = _detect_linux_cpu()
        assert info.physical_cores == 8
        assert info.logical_cores == 16
        assert info.has_avx is True
        assert info.has_avx2 is True
        assert info.has_avx512 is True
        assert info.vendor == "Intel"
        assert info.model_name == "Intel Xeon Gold 6248"

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=4)
    @patch("heterodyne.device.cpu.subprocess.run", side_effect=FileNotFoundError)
    @patch("builtins.open", side_effect=OSError)
    def test_fallback_when_commands_fail(
        self,
        mock_open_fn: MagicMock,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        info = _detect_linux_cpu()
        assert info.physical_cores == 4
        assert info.logical_cores == 4
        assert info.has_avx is False

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=None)
    @patch("heterodyne.device.cpu.subprocess.run", side_effect=FileNotFoundError)
    @patch("builtins.open", side_effect=OSError)
    def test_cpu_count_none_defaults_to_1(
        self,
        mock_open_fn: MagicMock,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        info = _detect_linux_cpu()
        assert info.physical_cores == 1
        assert info.logical_cores == 1

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=8)
    @patch(
        "heterodyne.device.cpu.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="lscpu", timeout=5),
    )
    @patch("builtins.open", side_effect=OSError)
    def test_lscpu_timeout(
        self,
        mock_open_fn: MagicMock,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        info = _detect_linux_cpu()
        # Falls back to os.cpu_count
        assert info.physical_cores == 8

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=8)
    @patch("heterodyne.device.cpu.subprocess.run")
    @patch(
        "builtins.open",
        mock_open(
            read_data="vendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC\nflags\t\t: avx avx2\n"
        ),
    )
    def test_amd_vendor_detection(
        self,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        info = _detect_linux_cpu()
        assert info.vendor == "AMD"
        assert info.has_avx512 is False


# ---------------------------------------------------------------------------
# _detect_macos_cpu
# ---------------------------------------------------------------------------


class TestDetectMacosCpu:
    @patch("heterodyne.device.cpu.platform.machine", return_value="arm64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=10)
    @patch("heterodyne.device.cpu.subprocess.run")
    def test_apple_silicon(
        self,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        def sysctl_side_effect(args: list[str], **kwargs: object) -> MagicMock:
            key = args[2]
            results = {
                "hw.physicalcpu": "8",
                "hw.logicalcpu": "10",
                "machdep.cpu.brand_string": "Apple M2 Pro",
            }
            val = results.get(key)
            if val is not None:
                return MagicMock(returncode=0, stdout=val)
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = sysctl_side_effect
        info = _detect_macos_cpu()
        assert info.physical_cores == 8
        assert info.logical_cores == 10
        assert info.vendor == "Apple"
        assert info.numa_nodes == 1
        assert info.architecture == "arm64"

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=12)
    @patch("heterodyne.device.cpu.subprocess.run")
    def test_intel_mac(
        self,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        def sysctl_side_effect(args: list[str], **kwargs: object) -> MagicMock:
            key = args[2]
            results = {
                "hw.physicalcpu": "6",
                "hw.logicalcpu": "12",
                "machdep.cpu.brand_string": "Intel(R) Core(TM) i7-9750H",
                "hw.optional.avx1_0": "1",
                "hw.optional.avx2_0": "1",
                "hw.optional.avx512f": "0",
            }
            val = results.get(key)
            if val is not None:
                return MagicMock(returncode=0, stdout=val)
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = sysctl_side_effect
        info = _detect_macos_cpu()
        assert info.physical_cores == 6
        assert info.logical_cores == 12
        assert info.vendor == "Intel"
        assert info.has_avx is True
        assert info.has_avx2 is True
        assert info.has_avx512 is False

    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=4)
    @patch("heterodyne.device.cpu.subprocess.run", side_effect=FileNotFoundError)
    def test_sysctl_not_found(
        self,
        mock_run: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
    ) -> None:
        info = _detect_macos_cpu()
        # Falls back to os.cpu_count
        assert info.physical_cores == 4
        assert info.logical_cores == 4


# ---------------------------------------------------------------------------
# _detect_fallback_cpu
# ---------------------------------------------------------------------------


class TestDetectFallbackCpu:
    @patch("heterodyne.device.cpu.platform.machine", return_value="x86_64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=4)
    def test_basic(self, mock_count: MagicMock, mock_machine: MagicMock) -> None:
        info = _detect_fallback_cpu()
        assert info.physical_cores == 4
        assert info.logical_cores == 4
        assert info.architecture == "x86_64"
        assert info.numa_nodes == 1

    @patch("heterodyne.device.cpu.platform.machine", return_value="aarch64")
    @patch("heterodyne.device.cpu.os.cpu_count", return_value=None)
    def test_cpu_count_none(
        self, mock_count: MagicMock, mock_machine: MagicMock
    ) -> None:
        info = _detect_fallback_cpu()
        assert info.physical_cores == 1
        assert info.logical_cores == 1


# ---------------------------------------------------------------------------
# detect_cpu_info  (dispatcher)
# ---------------------------------------------------------------------------


class TestDetectCpuInfo:
    @patch("heterodyne.device.cpu.platform.system", return_value="Linux")
    @patch("heterodyne.device.cpu._detect_linux_cpu")
    def test_dispatches_linux(self, mock_fn: MagicMock, mock_sys: MagicMock) -> None:
        mock_fn.return_value = CPUInfo(physical_cores=8, logical_cores=16)
        info = detect_cpu_info()
        mock_fn.assert_called_once()
        assert info.physical_cores == 8

    @patch("heterodyne.device.cpu.platform.system", return_value="Darwin")
    @patch("heterodyne.device.cpu._detect_macos_cpu")
    def test_dispatches_macos(self, mock_fn: MagicMock, mock_sys: MagicMock) -> None:
        mock_fn.return_value = CPUInfo(physical_cores=10, logical_cores=10)
        info = detect_cpu_info()
        mock_fn.assert_called_once()
        assert info.physical_cores == 10

    @patch("heterodyne.device.cpu.platform.system", return_value="Windows")
    @patch("heterodyne.device.cpu._detect_fallback_cpu")
    def test_dispatches_windows(self, mock_fn: MagicMock, mock_sys: MagicMock) -> None:
        mock_fn.return_value = CPUInfo(physical_cores=4, logical_cores=8)
        _info = detect_cpu_info()
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# configure_cpu_hpc
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_env")
class TestConfigureCpuHpc:
    def test_sets_thread_counts_to_physical_cores(self, basic_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(basic_cpu, use_physical_cores_only=True)
        assert env["OMP_NUM_THREADS"] == "8"
        assert env["MKL_NUM_THREADS"] == "8"
        assert env["OPENBLAS_NUM_THREADS"] == "8"
        assert os.environ["OMP_NUM_THREADS"] == "8"

    def test_uses_logical_cores_when_requested(self, basic_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(basic_cpu, use_physical_cores_only=False)
        assert env["OMP_NUM_THREADS"] == "16"

    def test_disables_nested_parallelism(self, basic_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(basic_cpu)
        assert env["OMP_NESTED"] == "FALSE"
        assert env["OMP_MAX_ACTIVE_LEVELS"] == "1"

    def test_numa_aware_multinode(self, intel_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(intel_cpu, numa_aware=True)
        assert env["OMP_PROC_BIND"] == "close"
        assert env["OMP_PLACES"] == "cores"
        # Intel-specific
        assert env["KMP_AFFINITY"] == "granularity=fine,compact,1,0"
        assert env["KMP_BLOCKTIME"] == "0"

    def test_numa_aware_disabled(self, intel_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(intel_cpu, numa_aware=False)
        assert "OMP_PROC_BIND" not in env
        assert "KMP_AFFINITY" not in env

    def test_single_numa_node_no_binding(self, basic_cpu: CPUInfo) -> None:
        """With only one NUMA node, no NUMA-aware settings should be applied."""
        env = configure_cpu_hpc(basic_cpu, numa_aware=True)
        assert "OMP_PROC_BIND" not in env

    def test_amd_numa_no_kmp(self, amd_cpu: CPUInfo) -> None:
        """AMD CPUs should get NUMA binding but not Intel KMP_ flags."""
        env = configure_cpu_hpc(amd_cpu, numa_aware=True)
        assert env["OMP_PROC_BIND"] == "close"
        assert "KMP_AFFINITY" not in env

    def test_single_core(self, single_core_cpu: CPUInfo) -> None:
        env = configure_cpu_hpc(single_core_cpu)
        assert env["OMP_NUM_THREADS"] == "1"

    def test_auto_detects_cpu_when_none(self) -> None:
        with patch("heterodyne.device.cpu.detect_cpu_info") as mock_detect:
            mock_detect.return_value = CPUInfo(physical_cores=4, logical_cores=8)
            env = configure_cpu_hpc(cpu_info=None)
            mock_detect.assert_called_once()
            assert env["OMP_NUM_THREADS"] == "4"


# ---------------------------------------------------------------------------
# get_jax_cpu_flags
# ---------------------------------------------------------------------------


class TestGetJaxCpuFlags:
    def test_device_count_equals_physical_cores(self, basic_cpu: CPUInfo) -> None:
        flags = get_jax_cpu_flags(basic_cpu)
        assert "--xla_force_host_platform_device_count=8" in flags

    def test_custom_num_devices(self, basic_cpu: CPUInfo) -> None:
        flags = get_jax_cpu_flags(basic_cpu, num_devices=4)
        assert "--xla_force_host_platform_device_count=4" in flags

    def test_avx512_enables_fast_math(self, intel_cpu: CPUInfo) -> None:
        flags = get_jax_cpu_flags(intel_cpu)
        assert "--xla_cpu_enable_fast_math=true" in flags

    def test_avx2_enables_fast_math(self) -> None:
        cpu = CPUInfo(physical_cores=4, logical_cores=8, has_avx2=True)
        flags = get_jax_cpu_flags(cpu)
        assert "--xla_cpu_enable_fast_math=true" in flags

    def test_no_avx_no_fast_math(self) -> None:
        cpu = CPUInfo(physical_cores=4, logical_cores=8)
        flags = get_jax_cpu_flags(cpu)
        assert "fast_math" not in flags

    def test_single_device(self, single_core_cpu: CPUInfo) -> None:
        flags = get_jax_cpu_flags(single_core_cpu)
        assert "--xla_force_host_platform_device_count=1" in flags


# ---------------------------------------------------------------------------
# configure_jax_cpu
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_clean_env")
class TestConfigureJaxCpu:
    def test_sets_xla_flags(self, basic_cpu: CPUInfo) -> None:
        env = configure_jax_cpu(basic_cpu, num_devices=4)
        assert "XLA_FLAGS" in env
        assert "--xla_force_host_platform_device_count=4" in env["XLA_FLAGS"]
        assert os.environ["XLA_FLAGS"] == env["XLA_FLAGS"]

    def test_sets_jax_platforms(self, basic_cpu: CPUInfo) -> None:
        env = configure_jax_cpu(basic_cpu)
        assert env["JAX_PLATFORMS"] == "cpu"
        assert os.environ["JAX_PLATFORMS"] == "cpu"

    def test_replaces_existing_device_count(
        self, basic_cpu: CPUInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "XLA_FLAGS", "--xla_force_host_platform_device_count=2 --some_other_flag"
        )
        env = configure_jax_cpu(basic_cpu, num_devices=8)
        xla = env["XLA_FLAGS"]
        # Old device count removed, new one present
        assert "--xla_force_host_platform_device_count=8" in xla
        assert "device_count=2" not in xla
        # Other flags preserved
        assert "--some_other_flag" in xla

    def test_includes_threading_vars(self, basic_cpu: CPUInfo) -> None:
        env = configure_jax_cpu(basic_cpu)
        assert "OMP_NUM_THREADS" in env

    def test_auto_detects_cpu_when_none(self) -> None:
        with patch("heterodyne.device.cpu.detect_cpu_info") as mock_detect:
            mock_detect.return_value = CPUInfo(physical_cores=2, logical_cores=4)
            env = configure_jax_cpu(cpu_info=None)
            assert mock_detect.call_count >= 1
            assert "XLA_FLAGS" in env


# ---------------------------------------------------------------------------
# get_optimal_batch_size
# ---------------------------------------------------------------------------


class TestGetOptimalBatchSize:
    def test_returns_power_of_two(self, basic_cpu: CPUInfo) -> None:
        batch = get_optimal_batch_size(basic_cpu, data_size=1000)
        assert batch & (batch - 1) == 0  # power of 2

    def test_minimum_16(self, single_core_cpu: CPUInfo) -> None:
        batch = get_optimal_batch_size(single_core_cpu, data_size=10**9)
        assert batch >= 16

    def test_maximum_4096(self, intel_cpu: CPUInfo) -> None:
        batch = get_optimal_batch_size(intel_cpu, data_size=1, element_bytes=1)
        assert batch <= 4096

    def test_respects_l3_cache_size(self) -> None:
        cpu = CPUInfo(
            physical_cores=4,
            logical_cores=8,
            cache_sizes={"L3": 4 * 1024 * 1024},  # 4 MB L3
        )
        batch = get_optimal_batch_size(cpu, data_size=1000, element_bytes=8)
        assert 16 <= batch <= 4096

    def test_invalid_data_size(self, basic_cpu: CPUInfo) -> None:
        with pytest.raises(ValueError, match="Invalid data size"):
            get_optimal_batch_size(basic_cpu, data_size=0)
        with pytest.raises(ValueError, match="Invalid data size"):
            get_optimal_batch_size(basic_cpu, data_size=-10)

    def test_auto_detects_cpu(self) -> None:
        with patch("heterodyne.device.cpu.detect_cpu_info") as mock_detect:
            mock_detect.return_value = CPUInfo(physical_cores=4, logical_cores=8)
            batch = get_optimal_batch_size(cpu_info=None, data_size=500)
            mock_detect.assert_called_once()
            assert batch >= 16


# ---------------------------------------------------------------------------
# CPUInfo dataclass
# ---------------------------------------------------------------------------


class TestCPUInfoDataclass:
    def test_defaults(self) -> None:
        info = CPUInfo(physical_cores=4, logical_cores=8)
        assert info.numa_nodes == 1
        assert info.architecture == ""
        assert info.vendor == ""
        assert info.model_name == ""
        assert info.has_avx is False
        assert info.has_avx2 is False
        assert info.has_avx512 is False
        assert info.cache_sizes == {}

    def test_mutable_cache_sizes_default(self) -> None:
        """Each instance should get its own dict for cache_sizes."""
        a = CPUInfo(physical_cores=1, logical_cores=1)
        b = CPUInfo(physical_cores=1, logical_cores=1)
        a.cache_sizes["L3"] = 123
        assert "L3" not in b.cache_sizes
