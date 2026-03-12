=================
Device Management
=================

Hardware detection, CPU/NUMA topology analysis, and optimal JAX/XLA
device configuration for CPU-bound workloads.

Hardware Configuration
======================

.. automodule:: heterodyne.device.config
   :members: HardwareConfig, detect_hardware, configure_optimal_device
   :undoc-members:
   :show-inheritance:

CPU Detection
=============

.. automodule:: heterodyne.device.cpu
   :members: CPUInfo, detect_cpu_info, configure_jax_cpu
   :undoc-members:
   :show-inheritance:
