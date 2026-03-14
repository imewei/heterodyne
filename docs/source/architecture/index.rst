Architecture Deep Dives
=======================

This section collects detailed architecture documents for each major
subsystem.  These documents describe internal design decisions, data
structures, and algorithmic details beyond what is covered in the
:doc:`/developer/architecture` overview.

Subsystem Summaries
-------------------

**Heterodyne Architecture Overview**
   Top-level system design: package structure, module responsibilities,
   and the two-path integral architecture (meshgrid vs. element-wise).

**Physical Model Architecture**
   The 14-parameter two-component correlation model: transport equations,
   velocity phase terms, sample fraction polynomials, and the unified
   ``HeterodyneModel`` facade.

**NLSQ Fitting Architecture**
   Trust-region Levenberg--Marquardt implementation: JAX-JIT residual and
   Jacobian adapters, strategy selection by data size, sequential vs.
   direct modes, and convergence diagnostics.

**CMC Fitting Architecture**
   Consensus Monte Carlo with NUTS: ``ShardGrid`` element-wise evaluation,
   reparameterized and non-reparameterized sampling paths, prior
   construction, and ArviZ post-processing.

**Data Handler Architecture**
   HDF5 loading via ``XPCSDataLoader``: shape/dtype validation, NaN guards,
   monotonicity checks, and cache management.
