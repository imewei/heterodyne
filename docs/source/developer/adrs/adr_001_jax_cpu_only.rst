.. _adr_jax_cpu_only:

ADR-001: JAX CPU-Only Backend
==============================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

Heterodyne targets **CPU-only JAX** (no GPU/TPU support). The JAX backend is used for:

1. JIT compilation of the physics kernel (``jax_backend.py``, ``physics_cmc.py``).
2. Automatic differentiation for Jacobians (NLSQ) and NUTS gradients (CMC).
3. Vectorized operations via ``jax.vmap`` over angles and time points.

The CMC backend explicitly sets:

.. code-block:: python

   # In multiprocessing.py worker initialization
   os.environ["JAX_ENABLE_X64"] = "1"
   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

This creates 4 virtual JAX devices per worker process, enabling NumPyro's ``parallel``
chain execution mode (``pmap`` over 4 devices) without requiring a real GPU.

The physics kernel evaluates the two-component correlation model over the 14-parameter
space ``(D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample, D_offset_sample,
v0, beta, v_offset, f0, f1, f2, f3, phi0)`` plus 2 per-angle scaling parameters.
At double precision, the full two-time correlation matrix is O(N²) where N is the
number of time frames.


Decision
--------

Use **JAX on CPU** as the sole numerical backend.  Enforce this via:

.. code-block:: python

   # heterodyne/__init__.py
   os.environ.setdefault("JAX_ENABLE_X64", "1")
   os.environ.setdefault("JAX_PLATFORMS", "cpu")

   # device/cpu.py
   jax.config.update("jax_platform_name", "cpu")

Float64 precision is mandatory. The transport coefficient integrals

.. math::

   J_r(t) = D_{0,\mathrm{ref}}\, t^{\alpha_\mathrm{ref}} + D_{\mathrm{offset,ref}}, \qquad
   J_s(t) = D_{0,\mathrm{sample}}\, t^{\alpha_\mathrm{sample}} + D_{\mathrm{offset,sample}}

involve subtraction of nearly equal large values at small lag times, producing
catastrophic cancellation in float32.


Rationale
---------

**Target deployment environment**

Beamline workstations (APS, ESRF, PETRA III) typically run Linux on multi-core CPUs
(16–128 cores) without GPU access. The analysis must run at the beamline during
experiments, so installation complexity must be minimal.

**Float64 requirement**

JAX defaults to float32 on GPU for performance. The 14-parameter heterodyne model requires
float64 throughout: the transport coefficient integrals involve subtraction of nearly
equal values at short lag times, producing O(1) relative errors in float32. Setting
``JAX_ENABLE_X64=1`` enables float64, but this must be done **before the first JAX
import**. The ``heterodyne/__init__.py`` and ``cli/main.py`` both call
``os.environ.setdefault("JAX_ENABLE_X64", "1")`` to ensure this.

**NLSQ library boundary**

The primary optimizer (``nlsq.curve_fit``) calls JAX JIT-compiled residual and Jacobian
functions, but the outer trust-region loop runs on the CPU host. This creates a
JAX-device-to-host round-trip per iteration. For GPU execution, this round-trip would
dominate latency (PCIe bandwidth ~16 GB/s; the Jacobian array is O(N² × 16) bytes).
The CPU round-trip is free (host memory).

**CMC multiprocessing**

The CMC backend spawns ``⌊N_cores/2⌋ − 1`` worker processes, each running NUTS on a
data shard. Each worker sets ``XLA_FLAGS=--xla_force_host_platform_device_count=4``
to create 4 virtual devices, enabling ``pmap`` over 4 NUTS chains. This requires CPU-only
JAX; GPU workers would compete for the same physical GPU.

**Determinism**

CPU JAX execution is deterministic given the same seed. GPU JAX is non-deterministic
by default due to non-associative floating-point reductions in CUDA. Heterodyne's
reproducibility guarantee (via ``CMCConfig.seed``) requires deterministic execution.


Consequences
------------

**Positive**:

- Zero-configuration installation at beamlines: ``uv sync`` is sufficient.
- Float64 precision throughout: no truncation or cancellation errors.
- Deterministic results given the same seed.
- Simple testing matrix: no CUDA version matrix.

**Negative / Accepted trade-offs**:

- No GPU acceleration. The NLSQ Jacobian computation (O(N² × n_params) per iteration)
  is the primary bottleneck for large datasets.
- CMC wall time scales with ``N_shards × N_warmup_samples``; typical runs take
  30–120 minutes for 1000-frame datasets on a 32-core workstation.
- ``XLA_FLAGS`` and ``JAX_ENABLE_X64`` must be set before first JAX import; late
  configuration is silently ignored.


Alternatives Considered
-----------------------

**A. GPU-first JAX**

Would enable faster matrix operations for the NLSQ Jacobian computation and NUTS leapfrog
steps. Rejected because: GPU not universally available at beamlines; 32-bit precision
limitation; non-deterministic results.

**B. NumPy / SciPy only**

Would eliminate the JAX dependency, simplifying installation. Rejected because: no
automatic differentiation (would require finite-difference Jacobians, ~100× slower);
no JIT compilation; cannot leverage the NumPyro ecosystem for NUTS.

**C. PyTorch backend**

PyTorch has mature CPU and GPU support with autograd. Rejected because: the NumPyro
probabilistic programming library is JAX-native, and rewriting NUTS from scratch would
be a significant engineering effort; JAX's ``jit``/``vmap``/``pmap`` composability is
a better fit for the parallelism patterns in CMC.

**D. Mixed CPU/GPU**

Support both, dispatch at runtime. Rejected for the current version because: doubles the
testing matrix; GPU code path is untested and risky; adds complexity without clear benefit
for the target deployment environment. May be reconsidered in a future version.


.. seealso::

   - :ref:`adr_no_consumer_gpu` — quantitative GPU feasibility assessment (RTX 4090)
   - :ref:`developer_architecture` — overall system design
   - :ref:`adr_nlsq_primary` — NLSQ optimizer choice
   - :ref:`adr_cmc_consensus` — CMC multiprocessing architecture
