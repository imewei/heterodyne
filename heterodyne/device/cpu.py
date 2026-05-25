"""CPU detection and HPC optimization utilities.

This module provides hardware-aware configuration for JAX workloads on CPU,
including physical core detection, NUMA topology awareness, and optimal
environment variable configuration for HPC clusters.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_HPC_CONFIGURED = False
logger = logging.getLogger(__name__)


@dataclass
class CPUInfo:
    """CPU hardware information.

    Attributes:
        physical_cores: Number of physical CPU cores.
        logical_cores: Number of logical cores (includes hyperthreading).
        numa_nodes: Number of NUMA nodes (memory domains).
        architecture: CPU architecture string (e.g., 'x86_64', 'arm64').
        vendor: CPU vendor (e.g., 'Intel', 'AMD', 'Apple').
        model_name: Full CPU model name.
        has_avx: Whether AVX instructions are available.
        has_avx2: Whether AVX2 instructions are available.
        has_avx512: Whether AVX-512 instructions are available.
        cache_sizes: Cache sizes in bytes (L1, L2, L3).
    """

    physical_cores: int
    logical_cores: int
    numa_nodes: int = 1
    architecture: str = ""
    vendor: str = ""
    model_name: str = ""
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    cache_sizes: dict[str, int] = field(default_factory=dict)


def detect_cpu_info() -> CPUInfo:
    """Detect CPU hardware information.

    Returns:
        CPUInfo dataclass with hardware details.

    Note:
        This function uses platform-specific methods:
        - Linux: lscpu, /proc/cpuinfo
        - macOS: sysctl
        - Windows: wmic (basic support)
    """
    system = platform.system()

    if system == "Linux":
        return _detect_linux_cpu()
    elif system == "Darwin":
        return _detect_macos_cpu()
    else:
        # Fallback for Windows and other platforms
        return _detect_fallback_cpu()


def _detect_linux_cpu() -> CPUInfo:
    """Detect CPU info on Linux using lscpu and /proc/cpuinfo."""
    logical_cores = os.cpu_count() or 1
    info = CPUInfo(
        physical_cores=logical_cores,
        logical_cores=logical_cores,
        architecture=platform.machine(),
    )

    # Try lscpu for detailed info
    lscpu_success = False
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            lscpu_success = True
            info = _parse_lscpu(result.stdout, info)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if lscpu_success and info.physical_cores == info.logical_cores:
        sysfs_physical_cores = _detect_linux_physical_cores_from_sysfs()
        if sysfs_physical_cores is not None:
            info.physical_cores = sysfs_physical_cores
        elif info.logical_cores > 1:
            warnings.warn(
                "Could not disambiguate Linux physical CPU core count from lscpu "
                "or sysfs; using logical CPU count for physical_cores.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Read /proc/cpuinfo for AVX flags, model name, and vendor
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()

        avx_detected = False
        for line in cpuinfo.split("\n"):
            if line.startswith("flags") and ":" in line:
                flags_set = set(line.split(":", 1)[1].split())
                info.has_avx = "avx" in flags_set
                info.has_avx2 = "avx2" in flags_set
                info.has_avx512 = any(f.startswith("avx512") for f in flags_set)
                avx_detected = True
            elif line.startswith("model name") and not info.model_name:
                info.model_name = line.split(":", 1)[1].strip()
            elif line.startswith("vendor_id") and not info.vendor:
                vendor_str = line.split(":", 1)[1].strip()
                if "Intel" in vendor_str:
                    info.vendor = "Intel"
                elif "AMD" in vendor_str:
                    info.vendor = "AMD"
                else:
                    info.vendor = vendor_str
            if avx_detected and info.model_name and info.vendor:
                break
    except OSError:
        pass

    return info


def _detect_linux_physical_cores_from_sysfs() -> int | None:
    """Detect physical cores by deduplicating Linux topology package/core IDs."""
    topology_root = Path("/sys/devices/system/cpu")
    physical_cores: set[tuple[int, int]] = set()

    for cpu_dir in topology_root.glob("cpu[0-9]*"):
        topology_dir = cpu_dir / "topology"
        try:
            package_id = int((topology_dir / "physical_package_id").read_text().strip())
            core_id = int((topology_dir / "core_id").read_text().strip())
        except (OSError, ValueError):
            continue
        physical_cores.add((package_id, core_id))

    if not physical_cores:
        return None
    return len(physical_cores)


def _safe_int(value: str) -> int | None:
    """Parse an integer from lscpu value, handling formats like '4 (2 online)'."""
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return None


def _parse_lscpu(output: str, info: CPUInfo) -> CPUInfo:
    """Parse lscpu output to extract CPU information."""
    cores_per_socket: int | None = None
    sockets: int | None = None

    for line in output.split("\n"):
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "cpu(s)":
            parsed = _safe_int(value)
            if parsed is not None:
                info.logical_cores = parsed
        elif key == "core(s) per socket":
            cores_per_socket = _safe_int(value)
        elif key == "socket(s)":
            sockets = _safe_int(value)
        elif key == "numa node(s)":
            parsed = _safe_int(value)
            if parsed is not None:
                info.numa_nodes = parsed
        elif key == "architecture":
            info.architecture = value
        elif key == "vendor id":
            if "Intel" in value:
                info.vendor = "Intel"
            elif "AMD" in value:
                info.vendor = "AMD"
            else:
                info.vendor = value

    # Compute physical cores from cores_per_socket * sockets
    if cores_per_socket is not None and sockets is not None:
        info.physical_cores = cores_per_socket * sockets

    return info


def _detect_macos_cpu() -> CPUInfo:
    """Detect CPU info on macOS using sysctl."""
    info = CPUInfo(
        physical_cores=os.cpu_count() or 1,
        logical_cores=os.cpu_count() or 1,
        architecture=platform.machine(),
    )

    def _sysctl(key: str) -> str | None:
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    # Physical cores
    val = _sysctl("hw.physicalcpu")
    if val:
        info.physical_cores = int(val)

    # Logical cores
    val = _sysctl("hw.logicalcpu")
    if val:
        info.logical_cores = int(val)

    # CPU brand
    val = _sysctl("machdep.cpu.brand_string")
    if val:
        info.model_name = val
        if "Intel" in val:
            info.vendor = "Intel"
        elif "Apple" in val:
            info.vendor = "Apple"

    # Apple Silicon detection
    if info.architecture == "arm64":
        info.vendor = "Apple"
        # Apple Silicon has unified memory, treat as single NUMA node
        info.numa_nodes = 1

    # AVX detection (Intel Macs only)
    val = _sysctl("hw.optional.avx1_0")
    if val:
        info.has_avx = val == "1"
    val = _sysctl("hw.optional.avx2_0")
    if val:
        info.has_avx2 = val == "1"
    val = _sysctl("hw.optional.avx512f")
    if val:
        info.has_avx512 = val == "1"

    return info


def _detect_fallback_cpu() -> CPUInfo:
    """Fallback CPU detection for Windows and other platforms."""
    return CPUInfo(
        physical_cores=os.cpu_count() or 1,
        logical_cores=os.cpu_count() or 1,
        architecture=platform.machine(),
    )


def configure_cpu_hpc(
    cpu_info: CPUInfo | None = None,
    use_physical_cores_only: bool = True,
    numa_aware: bool = True,
) -> dict[str, str]:
    """Configure environment variables for HPC CPU optimization.

    This function sets environment variables for optimal CPU performance
    with JAX and underlying libraries (MKL, OpenBLAS, OpenMP).

    Args:
        cpu_info: CPU information (auto-detected if None).
        use_physical_cores_only: If True, limit threads to physical cores
            (recommended for compute-bound workloads).
        numa_aware: If True, configure for NUMA-aware memory allocation.

    Returns:
        Dictionary of environment variables that were set.
    """
    global _HPC_CONFIGURED

    if cpu_info is None:
        cpu_info = detect_cpu_info()

    num_threads = (
        cpu_info.physical_cores if use_physical_cores_only else cpu_info.logical_cores
    )

    env_vars: dict[str, str] = {
        # OpenMP threading
        "OMP_NUM_THREADS": str(num_threads),
        # Intel MKL
        "MKL_NUM_THREADS": str(num_threads),
        # OpenBLAS
        "OPENBLAS_NUM_THREADS": str(num_threads),
        # BLAS (generic)
        "BLAS_NUM_THREADS": str(num_threads),
        # NumPy threading
        "NUMEXPR_NUM_THREADS": str(num_threads),
        # Avoid nested parallelism
        "OMP_NESTED": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }

    # NUMA-aware settings
    if numa_aware and cpu_info.numa_nodes > 1:
        env_vars["OMP_PROC_BIND"] = "close"
        env_vars["OMP_PLACES"] = "cores"
        # Intel-specific NUMA optimizations
        if cpu_info.vendor == "Intel":
            env_vars["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
            env_vars["KMP_BLOCKTIME"] = "0"

    if _HPC_CONFIGURED and all(
        os.environ.get(key) == value for key, value in env_vars.items()
    ):
        logger.debug("CPU HPC environment already configured; skipping re-application")
        return env_vars

    if _HPC_CONFIGURED:
        logger.debug("CPU HPC environment changed; re-applying configuration")

    # Apply to environment
    for key, value in env_vars.items():
        os.environ[key] = value

    _HPC_CONFIGURED = True

    return env_vars


def get_optimal_batch_size(
    cpu_info: CPUInfo | None = None,
    data_size: int = 1000,
    element_bytes: int = 8,
) -> int:
    """Calculate optimal batch size based on CPU cache hierarchy.

    This heuristic aims to fit working data in L3 cache while maintaining
    enough parallelism for efficient vectorization.

    Args:
        cpu_info: CPU information (auto-detected if None).
        data_size: Size of the input data dimension.
        element_bytes: Bytes per element (8 for float64, 4 for float32).

    Returns:
        Recommended batch size.
    """
    if cpu_info is None:
        cpu_info = detect_cpu_info()

    if data_size <= 0:
        raise ValueError(f"Invalid data size: {data_size}")

    # Default L3 cache assumption: 8 MB per core, shared
    l3_cache = cpu_info.cache_sizes.get("L3", 8 * 1024 * 1024 * cpu_info.physical_cores)

    # Target: use ~50% of L3 for working set
    target_bytes = l3_cache // 2

    # Estimate batch size
    batch_size = max(1, target_bytes // (data_size * element_bytes))

    # Round down to power of 2 for SIMD efficiency (avoid exceeding available resources)
    batch_size = 1 << (batch_size.bit_length() - 1) if batch_size > 0 else 1

    # Clamp to reasonable range
    return max(16, min(batch_size, 4096))


def benchmark_cpu_performance(
    cpu_info: CPUInfo | None = None,
    matrix_size: int = 1000,
) -> dict[str, float]:
    """Run a simple CPU benchmark for performance profiling.

    Args:
        cpu_info: CPU information (auto-detected if None).
        matrix_size: Size of test matrices for BLAS benchmark.

    Returns:
        Dictionary with benchmark results (GFLOPS, memory bandwidth).
    """
    import time

    import numpy as np

    if cpu_info is None:
        cpu_info = detect_cpu_info()

    results: dict[str, float] = {}

    # Matrix multiplication benchmark (BLAS performance)
    a = np.random.randn(matrix_size, matrix_size)
    b = np.random.randn(matrix_size, matrix_size)

    # Warmup
    _ = np.dot(a, b)

    # Timed run
    start = time.perf_counter()
    for _ in range(3):
        _ = np.dot(a, b)
    elapsed = time.perf_counter() - start

    # Calculate GFLOPS (2*n^3 operations for matrix mult)
    flops = 3 * 2 * (matrix_size**3)
    results["gemm_gflops"] = flops / elapsed / 1e9

    # Memory bandwidth benchmark
    array_size = 100 * 1024 * 1024 // 8  # 100 MB
    src = np.random.randn(array_size)
    dst = np.empty_like(src)

    start = time.perf_counter()
    for _ in range(10):
        np.copyto(dst, src)
    elapsed = time.perf_counter() - start

    # Calculate bandwidth in GB/s
    bytes_transferred = 10 * 2 * array_size * 8  # read + write
    results["memory_bandwidth_gbps"] = bytes_transferred / elapsed / 1e9

    results["physical_cores"] = float(cpu_info.physical_cores)
    results["logical_cores"] = float(cpu_info.logical_cores)

    return results


def get_jax_cpu_flags(
    cpu_info: CPUInfo | None = None,
    num_devices: int | None = None,
) -> str:
    """Generate XLA_FLAGS for optimal JAX CPU execution.

    Args:
        cpu_info: CPU information (auto-detected if None).
        num_devices: Number of CPU devices to expose (default: physical cores).

    Returns:
        XLA_FLAGS string to set in environment.
    """
    if cpu_info is None:
        cpu_info = detect_cpu_info()

    if num_devices is None:
        num_devices = cpu_info.physical_cores

    flags = [
        f"--xla_force_host_platform_device_count={num_devices}",
    ]

    return " ".join(flags)


def _warn_if_jax_already_imported(*, strict: bool) -> None:
    """Warn when env-based JAX configuration is likely too late to take effect."""
    if "jax" not in sys.modules:
        return

    message = (
        "JAX has already been imported; XLA_FLAGS and JAX_PLATFORMS changes "
        "made by configure_jax_cpu() may be ignored. Call device configuration "
        "before importing JAX."
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def configure_jax_cpu(
    cpu_info: CPUInfo | None = None,
    num_devices: int | None = None,
    strict: bool = False,
) -> Mapping[str, str]:
    """Configure JAX for optimal CPU execution.

    This should be called before importing JAX or at the start of a script.

    Args:
        cpu_info: CPU information (auto-detected if None).
        num_devices: Number of CPU devices (default: physical cores).
        strict: If True, raise RuntimeError when JAX was already imported.

    Returns:
        Dictionary of environment variables that were set.
    """
    _warn_if_jax_already_imported(strict=strict)

    if cpu_info is None:
        cpu_info = detect_cpu_info()

    # Configure threading
    env_vars = configure_cpu_hpc(cpu_info)

    # Configure XLA — strip any previous device-count flag before setting new one
    new_flags = get_jax_cpu_flags(cpu_info, num_devices)
    existing_flags = os.environ.get("XLA_FLAGS", "")
    cleaned = re.sub(
        r"--xla_force_host_platform_device_count=\d+", "", existing_flags
    ).strip()
    xla_flags = f"{cleaned} {new_flags}".strip() if cleaned else new_flags
    os.environ["XLA_FLAGS"] = xla_flags
    env_vars["XLA_FLAGS"] = xla_flags

    # Force CPU backend
    os.environ["JAX_PLATFORMS"] = "cpu"
    env_vars["JAX_PLATFORMS"] = "cpu"

    return env_vars
