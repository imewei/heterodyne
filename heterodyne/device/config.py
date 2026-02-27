"""Hardware configuration and CMC backend selection.

This module provides hardware detection utilities for selecting the optimal
CMC (Consensus Monte Carlo) backend based on the available resources.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from heterodyne.device.cpu import CPUInfo, detect_cpu_info


class ClusterType(Enum):
    """HPC cluster job scheduler type."""

    STANDALONE = "standalone"
    PBS = "pbs"
    SLURM = "slurm"


class CMCBackend(Enum):
    """CMC execution backend."""

    PJIT = "pjit"  # JAX pjit for single-node parallelism
    MULTIPROCESSING = "multiprocessing"  # Python multiprocessing
    PBS = "pbs"  # PBS array jobs
    SLURM = "slurm"  # Slurm array jobs


@dataclass
class HardwareConfig:
    """Hardware configuration for CMC execution.

    Attributes:
        cpu_info: Detected CPU information.
        cluster_type: Type of HPC cluster (if any).
        available_cores: Number of CPU cores available for computation.
        memory_gb: Available memory in GB.
        recommended_backend: Recommended CMC backend based on hardware.
        recommended_chains: Recommended number of MCMC chains.
        max_parallel_chains: Maximum chains that can run in parallel.
    """

    cpu_info: CPUInfo
    cluster_type: ClusterType
    available_cores: int
    memory_gb: float
    recommended_backend: CMCBackend
    recommended_chains: int
    max_parallel_chains: int


def detect_cluster_type() -> ClusterType:
    """Detect the HPC cluster scheduler type from environment.

    Returns:
        ClusterType enum indicating the detected scheduler.
    """
    # Check for PBS
    if any(
        key in os.environ
        for key in ["PBS_JOBID", "PBS_NODEFILE", "PBS_ENVIRONMENT"]
    ):
        return ClusterType.PBS

    # Check for Slurm
    if any(
        key in os.environ
        for key in ["SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_CLUSTER_NAME"]
    ):
        return ClusterType.SLURM

    return ClusterType.STANDALONE


def get_available_memory() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in GB, or a conservative estimate if detection fails.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except (ImportError, OSError):
        # Fallback: assume 8 GB available
        # OSError covers psutil.AccessDenied in restricted containers
        return 8.0


def detect_hardware() -> HardwareConfig:
    """Detect hardware configuration and recommend CMC settings.

    Returns:
        HardwareConfig with detected settings and recommendations.

    Note:
        This function considers:
        - CPU core count (physical cores preferred)
        - Available memory (for chain parallelism)
        - Cluster environment (PBS/Slurm for distributed)
        - NUMA topology (for pjit backend)
    """
    cpu_info = detect_cpu_info()
    cluster_type = detect_cluster_type()
    memory_gb = get_available_memory()

    # Determine available cores
    # In cluster jobs, respect scheduler allocation
    if cluster_type == ClusterType.PBS:
        ncpus = os.environ.get("PBS_NCPUS") or os.environ.get("NCPUS")
        if ncpus:
            try:
                available_cores = int(ncpus)
            except ValueError:
                available_cores = cpu_info.physical_cores
        else:
            available_cores = cpu_info.physical_cores
    elif cluster_type == ClusterType.SLURM:
        cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus_per_task:
            try:
                available_cores = int(cpus_per_task)
            except ValueError:
                available_cores = cpu_info.physical_cores
        else:
            available_cores = cpu_info.physical_cores
    else:
        available_cores = cpu_info.physical_cores

    # Determine recommended backend and chain count
    backend, chains, max_parallel = _recommend_backend(
        cpu_info=cpu_info,
        cluster_type=cluster_type,
        available_cores=available_cores,
        memory_gb=memory_gb,
    )

    return HardwareConfig(
        cpu_info=cpu_info,
        cluster_type=cluster_type,
        available_cores=available_cores,
        memory_gb=memory_gb,
        recommended_backend=backend,
        recommended_chains=chains,
        max_parallel_chains=max_parallel,
    )


def _recommend_backend(
    cpu_info: CPUInfo,
    cluster_type: ClusterType,
    available_cores: int,
    memory_gb: float,
) -> tuple[CMCBackend, int, int]:
    """Recommend CMC backend and chain configuration.

    Args:
        cpu_info: CPU hardware information.
        cluster_type: Detected cluster type.
        available_cores: Number of available cores.
        memory_gb: Available memory in GB.

    Returns:
        Tuple of (backend, recommended_chains, max_parallel_chains).
    """
    # Memory per chain estimate: ~2-4 GB for typical XPCS models
    memory_per_chain = 3.0
    max_chains_by_memory = max(1, int(memory_gb / memory_per_chain))

    # For cluster environments, prefer array job backends
    if cluster_type == ClusterType.PBS:
        # PBS: use array jobs for embarrassingly parallel chains
        chains = min(8, max_chains_by_memory)
        return (CMCBackend.PBS, chains, chains)

    if cluster_type == ClusterType.SLURM:
        # Slurm: similar to PBS
        chains = min(8, max_chains_by_memory)
        return (CMCBackend.SLURM, chains, chains)

    # Standalone mode: choose between pjit and multiprocessing
    if available_cores >= 4:
        # pjit is efficient for 4+ cores with NUMA awareness
        max_parallel = min(available_cores, max_chains_by_memory, 8)
        chains = max_parallel
        return (CMCBackend.PJIT, chains, max_parallel)
    else:
        # For small core counts, multiprocessing has less overhead
        max_parallel = min(available_cores, max_chains_by_memory)
        chains = max(2, max_parallel)  # At least 2 chains for diagnostics
        return (CMCBackend.MULTIPROCESSING, chains, max_parallel)


def get_backend_name(backend: CMCBackend) -> Literal["pjit", "multiprocessing", "pbs", "slurm"]:
    """Get the string name of a CMC backend for configuration.

    Args:
        backend: CMCBackend enum value.

    Returns:
        Backend name string for use in configuration.
    """
    return backend.value  # type: ignore[return-value]


def configure_optimal_device(
    mode: str = "auto",
    num_chains: int | None = None,
) -> HardwareConfig:
    """Configure device settings for optimal CMC execution.

    This function should be called before running CMC to ensure
    proper CPU threading and JAX device configuration.

    Args:
        mode: Configuration mode:
            - "auto": Automatically detect and configure
            - "cmc": Optimize for CMC (4 chains typical)
            - "cmc-hpc": Optimize for HPC CMC (8 chains)
            - "nlsq": Optimize for NLSQ (single device)
        num_chains: Override number of chains (None for auto).

    Returns:
        HardwareConfig with applied settings.
    """
    from heterodyne.device.cpu import configure_jax_cpu

    hw = detect_hardware()

    # Determine number of devices based on mode
    if mode == "nlsq":
        num_devices = 1
    elif mode == "cmc":
        num_devices = num_chains if num_chains else min(4, hw.available_cores)
    elif mode == "cmc-hpc":
        num_devices = num_chains if num_chains else min(8, hw.available_cores)
    else:  # auto
        if num_chains:
            num_devices = num_chains
        else:
            num_devices = hw.max_parallel_chains

    # Configure JAX
    configure_jax_cpu(hw.cpu_info, num_devices=num_devices)

    return hw


def get_device_status() -> dict[str, object]:
    """Get current device configuration status.

    Returns:
        Dictionary with current device settings and detected hardware.
    """
    hw = detect_hardware()

    return {
        "cpu": {
            "physical_cores": hw.cpu_info.physical_cores,
            "logical_cores": hw.cpu_info.logical_cores,
            "numa_nodes": hw.cpu_info.numa_nodes,
            "architecture": hw.cpu_info.architecture,
            "vendor": hw.cpu_info.vendor,
            "model": hw.cpu_info.model_name,
        },
        "cluster": {
            "type": hw.cluster_type.value,
            "available_cores": hw.available_cores,
            "memory_gb": round(hw.memory_gb, 1),
        },
        "cmc_recommendation": {
            "backend": hw.recommended_backend.value,
            "chains": hw.recommended_chains,
            "max_parallel": hw.max_parallel_chains,
        },
        "environment": {
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        },
    }
