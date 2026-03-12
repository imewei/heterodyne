============
Optimization
============

The heterodyne optimization pipeline follows a two-stage workflow:

1. **NLSQ warm-start** -- Fast non-linear least squares via
   ``scipy.optimize.least_squares`` with JAX JIT-compiled residuals and
   Jacobians. Provides point estimates and covariance for posterior
   initialization.

2. **CMC (Bayesian MCMC)** -- Full posterior sampling via NumPyro NUTS,
   initialized from the NLSQ warm-start. Produces credible intervals,
   convergence diagnostics, and ArviZ-compatible inference data.

.. toctree::
   :maxdepth: 2

   nlsq
   cmc
