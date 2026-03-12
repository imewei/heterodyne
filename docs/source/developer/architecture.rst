Architecture Overview
=====================

This page summarizes the package layout and the key design patterns that
shape the Heterodyne codebase.

Package Layout
--------------

.. code-block:: text

   heterodyne/
   ├── core/                        # Physics kernel -- JIT JAX
   │   ├── jax_backend.py           # Meshgrid-mode (NLSQ path)
   │   ├── physics_cmc.py           # Element-wise (CMC path)
   │   ├── physics_utils.py         # Shared primitives
   │   ├── physics_nlsq.py          # NLSQ residual/Jacobian adapters
   │   ├── models.py                # OO model interface
   │   └── heterodyne_model.py      # Unified facade
   ├── optimization/
   │   ├── nlsq/                    # Trust-region Levenberg-Marquardt (primary)
   │   └── cmc/                     # Consensus Monte Carlo / NUTS (secondary)
   ├── data/                        # HDF5 loading, validation
   ├── config/                      # YAML management, parameter registry
   ├── cli/                         # Entry points, shell completion
   ├── device/                      # CPU/NUMA detection, XLA flags
   ├── viz/                         # Visualization (PyQtGraph, Matplotlib)
   ├── io/                          # Result serialization (JSON, NPZ)
   └── utils/                       # Logging, path validation

Two-Path Integral Architecture
------------------------------

The physics kernel maintains two parallel computation paths that produce
numerically identical results but trade off memory for throughput:

**Meshgrid path** (``core/jax_backend.py``)
   Used by the NLSQ solver.  Builds a full :math:`N \times N` time-integral
   matrix via ``create_time_integral_matrix``.  This approach maximizes
   vectorization and JIT efficiency but allocates :math:`O(N^2)` memory.

**Element-wise path** (``core/physics_cmc.py``)
   Used by the CMC sampler.  Operates on a ``ShardGrid`` with
   ``precompute_shard_grid`` and :math:`O(n_\text{pairs})` cumsum lookup.
   Avoids the :math:`N \times N` allocation entirely, eliminating
   out-of-memory failures on long time series.

Both paths share low-level primitives from ``core/physics_utils.py``:
``trapezoid_cumsum``, ``create_time_integral_matrix``, ``smooth_abs``,
and rate functions.

Design Patterns
---------------

**Stateless physics functions.**
All physics kernels are pure JAX functions with no side effects.  This
guarantees JIT compatibility and makes testing straightforward.

**Immutable registries.**
Parameter defaults, bounds, and priors are stored in a
``MappingProxyType`` registry (``config/parameter_registry.py``) that
cannot be mutated after module import.

**Factory pattern.**
``create_model()`` in ``core/heterodyne_model.py`` constructs the
appropriate model variant based on configuration, hiding instantiation
details from calling code.

**Strategy pattern.**
The NLSQ optimizer selects among internal fitting strategies (e.g., direct
vs. sequential) based on data size and configuration, exposing a uniform
interface to the rest of the pipeline.

**Lazy imports.**
Top-level ``__init__.py`` uses ``__getattr__`` to defer heavy JAX imports
until first use, keeping ``import heterodyne`` lightweight for CLI and
configuration tasks.

Data Flow
---------

A typical analysis proceeds through these stages:

1. **Configuration** -- YAML is parsed and validated by ``config/``.
2. **Data loading** -- ``data/`` reads the HDF5 file, validates shapes,
   dtypes, NaN guards, and monotonicity.
3. **NLSQ warm-start** -- ``optimization/nlsq/`` runs trust-region
   Levenberg--Marquardt using the meshgrid physics path.
4. **CMC sampling** (optional) -- ``optimization/cmc/`` uses the NLSQ
   solution as an initial point for NUTS posterior sampling via the
   element-wise physics path.
5. **Diagnostics** -- ArviZ computes :math:`\hat{R}`, ESS, and BFMI.
6. **Output** -- ``io/`` serializes results to JSON and/or NPZ.

For detailed architecture documentation on individual subsystems, see the
:doc:`/architecture/index` section.
