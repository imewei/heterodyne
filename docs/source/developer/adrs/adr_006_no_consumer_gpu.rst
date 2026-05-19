.. _adr_no_consumer_gpu:

ADR-006: No GPU Acceleration on Consumer Hardware
===================================================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

Consumer-class GPUs (RTX 3090/4090, RX 7900 XTX) are available at some beamline
workstations. A quantitative assessment was performed to determine whether GPU
acceleration would provide a net benefit for the heterodyne analysis pipeline.

The two primary computations are:

1. **NLSQ Jacobian**: O(N² × n_params) per optimizer iteration, where N is the number
   of time frames and n_params = 16 (14 physical + 2 scaling). For N = 1000 frames,
   this is ~160M double-precision FLOPs per iteration.

2. **CMC NUTS leapfrog**: O(n_shard) per leapfrog step, where n_shard is the shard size.
   With typical shard sizes of 5000–20000 pairs and 20–50 leapfrog steps, each NUTS step
   costs ~200K–2M FLOPs.

**Consumer GPU float64 performance (RTX 4090)**:

The RTX 4090 provides 1:64 float64:float32 throughput ratio (82 TFLOPS float32;
~1.3 TFLOPS float64). This is too slow for the heterodyne kernel, which requires float64
throughout (see :ref:`adr_jax_cpu_only`).

**NLSQ boundary overhead**:

The primary NLSQ optimizer (``nlsq.curve_fit``) calls JAX JIT-compiled residual and
Jacobian functions, but the outer trust-region loop runs on the CPU host. At each
iteration, the Jacobian array (N² × n_params float64 values) must be transferred from
GPU to CPU. For N = 1000 and n_params = 16, this is ~128 MB per iteration. At PCIe 4.0
bandwidth (~30 GB/s), the transfer takes ~4 ms per iteration — comparable to the
computation time on a fast GPU.

**CMC multiprocessing constraint**:

The CMC backend spawns multiple worker processes, each with independent JAX device
contexts. Multiple workers sharing a single GPU would require inter-process GPU memory
management (CUDA MPS or equivalent), adding significant complexity. The virtual-device
approach (``--xla_force_host_platform_device_count=4``) that enables ``pmap`` over
4 virtual devices does not work on GPU.


Decision
--------

**Do not implement GPU acceleration on consumer-class GPUs.**  The CPU-only
configuration (``JAX_PLATFORMS=cpu``) is enforced in ``device/cpu.py`` and
remains the only supported backend.  No changes to ``pyproject.toml``, the
Makefile, or device configuration are required.


Rationale
---------

The quantitative assessment shows that consumer GPUs provide no meaningful speedup
for the heterodyne pipeline due to three compounding constraints:

1. **Float64 throughput**: Consumer GPUs have 1:64 float64:float32 ratio. Datacenter
   GPUs (A100, H100) have 1:2 ratio. Only datacenter GPUs can accelerate float64
   workloads.
2. **NLSQ boundary**: The CPU round-trip per optimizer iteration dominates GPU compute
   time for consumer-class PCIe bandwidth.
3. **CMC multiprocessing**: The spawn-based multiprocessing architecture is incompatible
   with GPU memory sharing across workers without significant refactoring.

CPU JAX with 32–64 cores provides competitive throughput for the heterodyne workload
at beamline facilities.


Future GPU Path
~~~~~~~~~~~~~~~

GPU acceleration becomes viable when **all three** conditions are met:

1. **Datacenter GPU** with :math:`\geq` 1:2 float64 ratio (A100 / H100).
2. **NLSQ library boundary eliminated** — migrate from ``nlsq.curve_fit`` to
   a pure-JAX optimizer (e.g., ``jaxopt.LevenbergMarquardt``), removing the
   per-iteration CPU round-trip.
3. **CMC refactored** to single-process ``jax.vmap``-over-chains, replacing
   spawn-based multiprocessing.

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Upgrade path
     - Estimated speedup
     - Engineering effort
   * - A100 + current code
     - 2–4x (NLSQ boundary limited)
     - Low
   * - A100/H100 + jaxopt LM rewrite
     - 10–30x (NLSQ)
     - High
   * - A100/H100 + CMC pmap refactor
     - 5–15x (CMC)
     - Medium
   * - A100/H100 + both rewrites
     - 10–30x (end-to-end)
     - High


Consequences
------------

**Positive**:

- Zero GPU dependency: installation requires no CUDA toolkit, cuDNN, or driver version
  pinning.
- Deterministic results: CPU JAX execution is deterministic given the same seed.
- Simple CI/CD: no GPU runners needed for testing.
- Full float64 throughput on any modern CPU.

**Negative / Accepted trade-offs**:

- NLSQ wall time is bounded by CPU FLOPS (~30–120 s for N = 1000).
- CMC wall time is bounded by ``N_shards × T_shard / N_workers``; typical runs take
  30–120 minutes on 32-core hardware.
- Users with A100/H100 access cannot use them without additional refactoring.


.. seealso::

   - :ref:`adr_jax_cpu_only` — foundational CPU-only decision
   - :ref:`adr_cmc_consensus` — CMC multiprocessing architecture
   - :ref:`adr_nlsq_primary` — NLSQ / CMC architectural split
