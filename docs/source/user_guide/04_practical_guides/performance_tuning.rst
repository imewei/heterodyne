.. _performance-tuning:

==================
Performance Tuning
==================

The heterodyne package runs entirely on CPU and uses JAX's XLA compiler
for JIT compilation.  This page covers the key levers for optimising
throughput and memory usage.


XLA Configuration
-----------------

The ``heterodyne-config-xla`` command-line tool configures XLA compiler
flags for optimal CPU performance on the current machine:

.. code-block:: bash

   # Auto-detect and print recommended XLA flags
   heterodyne-config-xla

   # Apply flags to the current shell session
   eval $(heterodyne-config-xla --export)

The tool detects:

* CPU instruction set extensions (AVX2, AVX-512).
* Number of physical cores.
* NUMA topology (if applicable).

It sets ``XLA_FLAGS`` to enable the appropriate LLVM target features
and intra-op parallelism.


NUMA Awareness
--------------

On multi-socket systems (common in workstations and compute nodes),
NUMA-aware thread pinning prevents cross-socket memory access penalties.
The ``heterodyne.device.cpu`` module detects the NUMA topology and
configures JAX accordingly.

For manual control:

.. code-block:: bash

   # Pin to NUMA node 0
   numactl --cpunodebind=0 --membind=0 python analysis.py

This is especially important for CMC analyses where multiple NUTS chains
run in parallel -- each chain should ideally be bound to a single NUMA
node.


Thread Count Optimisation
-------------------------

JAX and the underlying BLAS libraries use ``OMP_NUM_THREADS`` to
control parallelism.  The optimal value depends on the workload:

**NLSQ fitting**
   Typically benefits from all available cores.  Set
   ``OMP_NUM_THREADS`` equal to the number of physical cores (not
   hyperthreads):

   .. code-block:: bash

      export OMP_NUM_THREADS=16  # For a 16-core machine

**CMC sampling**
   When running multiple NUTS chains in parallel, each chain should
   use fewer threads to avoid oversubscription:

   .. code-block:: bash

      # 4 chains on a 16-core machine: 4 threads per chain
      export OMP_NUM_THREADS=4

**General rule**
   Total threads across all processes should not exceed the number of
   physical cores.  Hyperthreading typically does not help for
   numerically intensive JAX workloads.


Memory Management
-----------------

Large :math:`C_2` matrices can consume significant memory.  Key
strategies:

Frame trimming
--------------

If only a subset of the measurement time is of interest, trim the
frame range before fitting:

.. code-block:: python

   data = loader.load(frame_start=100, frame_end=500)

This reduces the :math:`C_2` matrix from :math:`N^2` to
:math:`(N_\text{trim})^2` elements.

Strategy selection
------------------

The NLSQ strategy affects peak memory:

* **JIT** -- Highest memory (full Jacobian in XLA).  Best for
  :math:`N < 500` frames.
* **Chunked** -- Moderate memory.  Good for :math:`500 < N < 2000`.
* **Sequential** -- Lowest memory.  Use for :math:`N > 2000` or
  memory-constrained environments.

.. code-block:: python

   config = NLSQConfig(strategy="chunked", chunk_size=256)

ShardGrid (CMC)
---------------

The CMC pathway avoids constructing the full :math:`N \times N` matrix
by evaluating only the grid points needed per shard.  The number of
shards controls the memory/computation trade-off:

* More shards = less memory per shard, more overhead from consensus
  combination.
* Fewer shards = more memory per shard, better statistical efficiency.

A typical starting point is ``num_shards = 4 * num_chains``.


CMC Backend Selection
---------------------

The CMC runner supports two execution modes:

**Sequential** (default)
   Shards are processed one at a time in the main process.  Simplest
   and most debuggable.  Use when memory is the bottleneck.

**Multiprocessing**
   Shards are distributed across worker processes using Python's
   ``multiprocessing`` module.  Each worker runs its NUTS chains
   independently.  Use when wall time is the bottleneck and the
   machine has enough memory for concurrent shards.

.. code-block:: python

   cmc_config = CMCConfig(
       backend="multiprocessing",
       max_workers=4,  # Number of parallel shard workers
   )


Profiling Tips
--------------

JIT compilation time
   The first call to a JIT-compiled function incurs compilation
   overhead.  Subsequent calls with the same input shapes are fast.
   If compilation is slow, check that input shapes are consistent
   (avoid triggering recompilation with different array sizes).

Memory profiling
   Use ``jax.live_arrays()`` or system tools (``htop``, ``psutil``)
   to monitor JAX array allocations during fitting.

Timing
   The ``wall_time_seconds`` field in both ``NLSQResult`` and
   ``CMCResult`` reports end-to-end fitting time, excluding data
   loading and I/O.
