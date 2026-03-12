.. _nlsq-fitting:

=============
NLSQ Fitting
=============

The non-linear least-squares (NLSQ) pathway provides fast point-estimate
fits of the heterodyne model to measured :math:`C_2` matrices.  It is
typically the first step in any analysis and serves as the warm-start
for Bayesian inference.


Algorithm Overview
==================

The NLSQ fitter minimises:

.. math::

   \chi^2 = \sum_{i \le j}
     \bigl[ C_2^\text{model}(t_i, t_j; \boldsymbol{\theta})
          - C_2^\text{data}(t_i, t_j) \bigr]^2

using the **trust-region reflective** variant of Levenberg--Marquardt as
implemented in ``scipy.optimize.least_squares``.  JAX provides:

* JIT-compiled residual evaluation for the full upper triangle.
* Automatic-differentiation Jacobian (exact, not finite-difference).


Upper-Triangle Optimisation
===========================

Because :math:`C_2` is symmetric, only the upper triangle
(:math:`i \le j`) is fitted.  This halves the number of residual
evaluations and avoids double-counting correlations.


Strategy Selection
==================

The fitter automatically selects a computational strategy based on the
size of the :math:`C_2` matrix:

**JIT** (default for small--medium datasets)
   The entire residual + Jacobian computation is JIT-compiled into a
   single XLA program.  Fastest per-evaluation, but compilation time
   can be significant for the first call.

**Residual**
   JIT-compiles only the residual; the Jacobian is computed by
   ``scipy`` using finite differences.  Useful when memory is too
   tight for the full JIT Jacobian.

**Chunked**
   Splits the upper triangle into row-chunks and evaluates each chunk
   under JIT.  Reduces peak memory at the cost of more kernel launches.

**Sequential**
   Evaluates the model without JIT, row by row.  Slowest but has the
   smallest memory footprint.  Jacobian Frobenius norm is logged at
   DEBUG level and stored in ``metadata["jacobian_norm"]``.

Override the automatic selection via :class:`~heterodyne.optimization.nlsq.config.NLSQConfig`:

.. code-block:: python

   from heterodyne.optimization.nlsq.config import NLSQConfig

   config = NLSQConfig(strategy="chunked", chunk_size=256)


Multi-Start Optimisation
=========================

The 14-parameter landscape can have local minima.  Multi-start
optimisation mitigates this by launching multiple fits from different
initial points sampled via **Latin Hypercube Sampling** (LHS):

.. code-block:: python

   config = NLSQConfig(
       n_starts=20,           # Number of random starting points
       lhs_seed=42,           # Reproducible sampling
   )

The best result (lowest :math:`\chi^2`) is returned.


Fourier Reparameterisation
==========================

When fitting multiple azimuthal angles jointly, the per-angle contrast
and offset can be expressed as truncated Fourier series in :math:`\phi`.
This reduces the number of free scaling parameters from
:math:`2 N_\phi` to :math:`2 (2 K + 1)` where :math:`K` is the
Fourier order.

Joint multi-angle fitting is invoked via
:func:`~heterodyne.optimization.nlsq.core.fit_nlsq_multi_phi`:

.. code-block:: python

   from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

   result = fit_nlsq_multi_phi(
       model=model,
       c2_data=c2_stack,        # Shape (N_angles, N, N)
       phi_angles=phi_angles,   # List of angles in degrees
       config=config,
   )


CMA-ES Fallback
================

For strongly multi-modal problems where gradient-based NLSQ fails to
find the global minimum, the package provides a CMA-ES (Covariance
Matrix Adaptation Evolution Strategy) wrapper.  See
:doc:`../03_advanced_topics/cmaes_optimization` for details.


Basic Example
=============

.. code-block:: python

   from heterodyne.core.heterodyne_model import HeterodyneModel
   from heterodyne.optimization.nlsq.core import fit_nlsq_jax
   from heterodyne.optimization.nlsq.config import NLSQConfig

   # Build model from timestamps and wavevector
   model = HeterodyneModel(timestamps=data.timestamps, q=0.025)

   # Configure and run
   config = NLSQConfig(
       n_starts=10,
       strategy="jit",
   )

   result = fit_nlsq_jax(
       model=model,
       c2_data=data.c2,
       phi_angle=45.0,
       config=config,
   )

   # Inspect result
   print(result.summary())
   print(f"Reduced chi-squared: {result.reduced_chi_squared:.4f}")
