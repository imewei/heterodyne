.. _bayesian-inference:

===================
Bayesian Inference
===================

The Bayesian pathway provides full posterior distributions for all model
parameters, enabling rigorous uncertainty quantification, correlation
analysis, and multimodality detection.


Pipeline Overview
-----------------

The recommended workflow is:

1. **NLSQ warm-start** -- Run a multi-start NLSQ fit to obtain a
   point estimate and covariance.
2. **Prior construction** -- Centre priors on the NLSQ solution,
   widened by ``nlsq_prior_width_factor``.
3. **NUTS sampling** -- Run NumPyro's No-U-Turn Sampler on each data
   shard independently.
4. **Consensus combination** -- Merge shard posteriors using
   inverse-variance (precision) weighting.
5. **Diagnostics** -- Validate convergence with ArviZ.


Consensus Monte Carlo (CMC)
-----------------------------

For large :math:`C_2` matrices, running a single NUTS chain on the full
dataset is prohibitively expensive.  The **Consensus Monte Carlo** (CMC)
strategy (Scott et al., 2016) splits the data into :math:`K` shards,
runs NUTS independently on each shard, and combines the resulting
posteriors:

.. math::

   \boldsymbol{\mu}_\text{consensus}
     = \Bigl(\sum_k \boldsymbol{\Sigma}_k^{-1}\Bigr)^{-1}
       \sum_k \boldsymbol{\Sigma}_k^{-1}\, \boldsymbol{\mu}_k

This is the inverse-variance weighted mean, which is optimal under
Gaussian posteriors and a good approximation for well-behaved
problems.


Element-Wise ShardGrid Evaluation
----------------------------------

Each shard evaluates the model only at the :math:`O(n_\text{pairs})`
grid points it needs, rather than constructing the full
:math:`N \times N` correlation matrix.  The
:class:`~heterodyne.core.physics_cmc.ShardGrid` and
``precompute_shard_grid`` functions handle this via cumulative-sum
lookup, keeping memory usage proportional to the shard size.


CMCConfig
---------

The :class:`~heterodyne.optimization.cmc.config.CMCConfig` dataclass
controls every aspect of the CMC pipeline:

``target_accept_prob`` (``float``, default 0.8)
   Target acceptance probability for NUTS adaptation.  Increase to
   0.9--0.95 if you see divergent transitions; this slows sampling
   but improves exploration of difficult geometries.

``max_r_hat`` (``float``, default 1.1)
   Maximum acceptable :math:`\hat{R}` for declaring convergence.
   A stricter threshold of 1.01 is recommended for publication-quality
   results.

``nlsq_prior_width_factor`` (``float``, default 5.0)
   Multiplier applied to NLSQ uncertainties when constructing priors.
   A factor of 5 gives weakly informative priors centred on the NLSQ
   solution.

``num_warmup`` / ``num_samples`` (``int``)
   Number of warmup (adaptation) and post-warmup draws per chain.

``num_chains`` (``int``)
   Number of independent NUTS chains per shard.

``num_shards`` (``int``)
   Number of data shards for the CMC split.

Example:

.. code-block:: python

   from heterodyne.optimization.cmc.config import CMCConfig

   cmc_config = CMCConfig(
       target_accept_prob=0.9,
       max_r_hat=1.01,
       nlsq_prior_width_factor=5.0,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       num_shards=8,
   )


Sharding Strategy
-----------------

Shards are constructed by partitioning the upper triangle of the
:math:`C_2` matrix into approximately equal-sized blocks.  Each shard
contains a contiguous set of row-pairs, ensuring temporal locality
and efficient cache usage.

For parallel execution, each shard's NUTS chains can run as independent
processes (one per CPU core), coordinated by the CMC runner.


ArviZ Diagnostics
-----------------

After sampling, convergence is validated using `ArviZ
<https://python.arviz.org>`_:

:math:`\hat{R}` (R-hat)
   The Gelman--Rubin convergence diagnostic.  Compares between-chain
   and within-chain variance.

   * :math:`\hat{R} < 1.01` -- Chains have converged.
   * :math:`\hat{R} \in [1.01, 1.1]` -- Marginal; consider more
     samples.
   * :math:`\hat{R} > 1.1` -- Not converged.  Do not trust the
     posterior.

**ESS** (Effective Sample Size)
   Accounts for autocorrelation within chains.

   * ESS > 400 -- Adequate for most summaries.
   * ESS < 100 -- Posterior estimates are unreliable.

**BFMI** (Bayesian Fraction of Missing Information)
   Measures how well the sampler explores the energy distribution.

   * BFMI > 0.3 -- Acceptable.
   * BFMI < 0.3 -- The sampler may be missing important regions of
     parameter space.  Check for funnel geometries or
     reparameterisation issues.

.. code-block:: python

   # Validate convergence
   warnings = result.validate_convergence(
       r_hat_threshold=1.01,
       min_ess=400,
       min_bfmi=0.3,
   )
   if warnings:
       for w in warnings:
           print(f"DIAGNOSTIC WARNING: {w}")


Bimodal Detection
-----------------

Post-hoc bimodality detection examines the marginal posterior for each
parameter across shards.  If a parameter shows two distinct modes
(e.g., two well-separated clusters of shard means), this indicates the
likelihood surface has multiple basins.  In such cases:

* The NLSQ warm-start may have found a local minimum.
* CMA-ES global optimisation should be tried to identify all basins.
* The physical model may need additional constraints (e.g., fixing
  ``phi0`` or tightening bounds on velocity parameters).
