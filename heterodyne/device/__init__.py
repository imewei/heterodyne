"""Device detection and HPC optimization utilities.

This module provides hardware-aware configuration for CPU-based JAX workloads,
including CPU detection, NUMA topology awareness, and optimal CMC backend
selection for MCMC sampling.

Example:
    >>> from heterodyne.device import configure_optimal_device, get_device_status
    >>> hw = configure_optimal_device(mode="cmc")
    >>> print(hw.recommended_chains)
    4
    >>> status = get_device_status()
    >>> print(status["cpu"]["physical_cores"])
    8

Note:
    This module should be imported before JAX for proper environment
    configuration. The configure_optimal_device() function sets XLA_FLAGS
    and other environment variables that must be set before JAX initializes.
"""

# Suppress JAX info logs before importing
import logging
import os

# Suppress TensorFlow/XLA logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress JAX logging
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)

# CPU detection and configuration
from heterodyne.device.cpu import (
    CPUInfo,
    benchmark_cpu_performance,
    configure_cpu_hpc,
    configure_jax_cpu,
    detect_cpu_info,
    get_jax_cpu_flags,
    get_optimal_batch_size,
)

# Hardware configuration for CMC
from heterodyne.device.config import (
    CMCBackend,
    ClusterType,
    HardwareConfig,
    configure_optimal_device,
    detect_cluster_type,
    detect_hardware,
    get_available_memory,
    get_backend_name,
    get_device_status,
)

__all__ = [
    # CPU info
    "CPUInfo",
    "detect_cpu_info",
    "configure_cpu_hpc",
    "configure_jax_cpu",
    "get_jax_cpu_flags",
    "get_optimal_batch_size",
    "benchmark_cpu_performance",
    # Hardware config
    "HardwareConfig",
    "ClusterType",
    "CMCBackend",
    "detect_hardware",
    "detect_cluster_type",
    "get_available_memory",
    "get_backend_name",
    "configure_optimal_device",
    "get_device_status",
]
