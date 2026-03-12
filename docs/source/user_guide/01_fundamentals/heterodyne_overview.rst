.. _heterodyne-overview:

=========================
Heterodyne Package Overview
=========================

The **heterodyne** package fits two-time correlation matrices
:math:`C_2(t_1, t_2)` to extract physical parameters for systems with
two coherently scattering components -- typically a static or slowly
evolving *reference* component and a flowing or diffusing *sample*
component.  This page describes the scope, architecture, and design
philosophy of the package.


The Inverse Problem
===================

Given a measured :math:`C_2` matrix at one or more azimuthal angles
:math:`\phi`, the package solves for:

* **Diffusion** of the reference and sample components
  (:math:`D_0`, :math:`\alpha`, :math:`D_\text{offset}` for each).
* **Velocity** of the sample relative to the reference
  (:math:`v_0`, :math:`\beta_v`, :math:`v_\text{offset}`).
* **Fraction** time evolution describing the relative scattering
  contribution (:math:`f_0, f_1, f_2, f_3`).
* **Flow angle** offset :math:`\phi_0`.
* Per-angle **contrast** :math:`\beta` and **offset**.

In total the model has **14 physics parameters** shared across all
angles, plus **2 scaling parameters** (contrast and offset) per angle.


Three-Term Model
================

The heterodyne correlation function is the sum of three contributions:

1. **Reference self-correlation** -- Encodes the transport (diffusion)
   of the reference component alone.  Parameterised by
   :math:`D_{0,r}`, :math:`\alpha_r`, and :math:`D_\text{off,r}`.

2. **Sample self-correlation** -- Same functional form for the sample
   component, parameterised by :math:`D_{0,s}`, :math:`\alpha_s`, and
   :math:`D_\text{off,s}`.

3. **Cross-correlation (velocity phase)** -- A term carrying a
   :math:`\cos(q\, v\, \Delta t\, \cos(\phi - \phi_0))` phase that
   encodes the relative velocity between the two components.
   Parameterised by :math:`v_0`, :math:`\beta_v`, :math:`v_\text{off}`,
   and :math:`\phi_0`.

The fraction function :math:`f(t)` weights the relative contribution of
each component as a function of measurement time.


Two Optimisation Pathways
=========================

The package provides two complementary fitting strategies:

**NLSQ -- Non-Linear Least Squares**
   Fast point-estimate fitting via ``scipy.optimize.least_squares`` with
   JAX-accelerated residual and Jacobian evaluation.  Typical wall time
   is seconds to minutes.  Best for:

   * Initial exploration and parameter surveys.
   * Warm-starting the Bayesian sampler.
   * Multi-start and multi-angle joint fits.

**CMC -- Consensus Monte Carlo**
   Full Bayesian posterior estimation via NumPyro NUTS.  The dataset is
   split into shards; independent NUTS chains run on each shard, and
   posteriors are combined using inverse-variance (precision) weighting.
   Mandatory ArviZ diagnostics (R-hat, ESS, BFMI) validate convergence.


Two-Path Computational Architecture
====================================

Both optimisation pathways ultimately evaluate the same physics, but
they use different computational strategies optimised for their
respective workloads:

**Meshgrid path (NLSQ)**
   Builds the full :math:`N \times N` upper-triangle correlation matrix
   via ``create_time_integral_matrix`` and cumulative-sum operations.
   Efficient under JIT because the entire matrix is computed in a single
   vectorised pass.

**Element-wise ShardGrid path (CMC)**
   Evaluates correlation values only at the :math:`O(n_\text{pairs})`
   grid points required by each shard.  Avoids allocating the full
   :math:`N \times N` matrix, preventing out-of-memory errors when
   the number of frames is large.

Both paths call the same shared primitives in ``core/physics_utils.py``
(``trapezoid_cumsum``, rate functions, ``smooth_abs``) so the physics
is guaranteed to be identical.


What Is Not In Scope
====================

The heterodyne package focuses exclusively on the fitting of
pre-computed :math:`C_2` matrices.  The following are explicitly **out
of scope**:

* **Raw data reduction** -- Converting detector frames to :math:`C_2`
  matrices is handled by beamline-specific pipelines (e.g., *pyXPCS*,
  *Xana*, or facility tools).
* **GPU acceleration** -- The package is CPU-only by design; XLA flags
  are configured for optimal CPU/NUMA performance.
* **Single-component homodyne fitting** -- Use the companion
  *homodyne* package for systems with one scattering component.


Design Philosophy
=================

* **JAX-first** -- All physics computations use JAX arrays and are
  JIT-compiled for performance.  Automatic differentiation provides
  exact Jacobians for the NLSQ solver.
* **CPU-only** -- Deliberate choice.  The :math:`C_2` matrices from
  typical beamline experiments fit comfortably in CPU memory, and
  NUMA-aware thread pinning gives excellent per-core throughput without
  the complexity of GPU memory management.
* **Reproducible** -- Explicit JAX PRNG keys, deterministic
  initialisation, and version-locked dependencies ensure bitwise
  reproducibility.
* **Modular** -- Data loading, physics evaluation, optimisation, and
  visualisation are cleanly separated so that each layer can be tested
  and replaced independently.
