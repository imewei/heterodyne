.. _diagnostics:

==============================
Diagnostics and Convergence
==============================

Reliable parameter estimates require verifying that the optimiser has
converged and that the model adequately describes the data.  This page
catalogues the diagnostic tools available for both NLSQ and CMC
analyses.


NLSQ Diagnostics
=================

Jacobian Condition Number
-------------------------

The condition number of the Jacobian at the optimum indicates how
sensitive the solution is to perturbations in the data:

* :math:`\kappa < 10^4` -- Well-conditioned; parameter estimates are
  stable.
* :math:`\kappa \in [10^4, 10^8]` -- Moderately ill-conditioned;
  some parameter combinations may be poorly constrained.
* :math:`\kappa > 10^8` -- Severely ill-conditioned; the covariance
  matrix is unreliable and some parameters are effectively
  unidentifiable from the data.

The Jacobian Frobenius norm is logged at DEBUG level and stored in
``result.metadata["jacobian_norm"]`` when using the sequential strategy.

Parameter Sensitivity
---------------------

The diagonal of :math:`(\mathbf{J}^T \mathbf{J})^{-1}` gives the
variance of each parameter.  Parameters with very large variance
relative to their value are poorly constrained.  Check for:

* Large relative uncertainties (``result.validate()`` flags
  :math:`|\sigma / \theta| > 1`).
* High off-diagonal correlations in the correlation matrix
  (``result.get_correlation_matrix()`` flags :math:`|r| > 0.95`).

Residual Map
------------

The spatial structure of residuals in the :math:`C_2` matrix reveals
systematic model failures:

* **Random residuals** -- The model captures all systematic features.
* **Diagonal streaks** -- Possible frame-to-frame intensity
  fluctuations not accounted for.
* **Block structure** -- The fraction function may need a different
  functional form.

Use :func:`~heterodyne.viz.nlsq_plots.plot_residual_map` or
:func:`~heterodyne.viz.diagnostics.plot_residual_map` to visualise.


CMC Diagnostics
===============

R-hat (:math:`\hat{R}`)
------------------------

The split-:math:`\hat{R}` statistic from ArviZ compares between-chain
and within-chain variance.  It is computed automatically during CMC
analysis and stored in ``result.r_hat``.

Interpretation:

* :math:`\hat{R} < 1.01` -- Converged.
* :math:`\hat{R} \in [1.01, 1.05]` -- Probably converged; run longer
  to confirm.
* :math:`\hat{R} > 1.1` -- Not converged.  Increase ``num_warmup``
  and ``num_samples``, or investigate the model.

Effective Sample Size (ESS)
----------------------------

ESS accounts for autocorrelation within MCMC chains.  Two variants:

* **ESS bulk** -- Efficiency for estimating the posterior mean.
* **ESS tail** -- Efficiency for estimating tail quantiles (credible
  intervals).

Rules of thumb:

* ESS > 400 per parameter -- Adequate for most summaries.
* ESS > 1000 -- Recommended for publication.
* ESS < 100 -- Posterior estimates are unreliable.

BFMI (Bayesian Fraction of Missing Information)
-------------------------------------------------

BFMI measures how well the sampler's kinetic energy transitions match
the marginal energy distribution.  Stored in ``result.bfmi`` (one value
per chain).

* BFMI > 0.3 -- Acceptable.
* BFMI < 0.3 -- The sampler is not exploring the energy landscape
  efficiently.  Consider reparameterising the model or increasing
  ``target_accept_prob``.

Divergent Transitions
---------------------

Divergent transitions indicate that the sampler encountered regions
of very high curvature in the posterior.  Even a small number of
divergences can bias the posterior.  Remedies:

* Increase ``target_accept_prob`` (e.g., to 0.95).
* Tighten parameter bounds to exclude unphysical regions.
* Check for funnel-shaped posteriors (common with hierarchical
  models).


Bimodal Detection
=================

After CMC, shard-level posterior means are compared.  If a parameter
shows two or more distinct clusters of shard means (separated by more
than 2 posterior standard deviations), the parameter is flagged as
potentially bimodal.

Bimodality suggests:

* The likelihood has multiple basins (global optimisation with CMA-ES
  may help identify all modes).
* A physical degeneracy exists (e.g., :math:`\phi_0` and
  :math:`\phi_0 + 180\degree` giving equivalent fits).
* The data are insufficient to distinguish between competing
  parameter combinations.

Use :func:`~heterodyne.viz.mcmc_plots.plot_shard_comparison` to
visualise shard-to-shard agreement.


Visualisation Tools
===================

The ``heterodyne.viz`` module provides a comprehensive set of
diagnostic plots:

NLSQ diagnostics:

* :func:`~heterodyne.viz.nlsq_plots.plot_nlsq_fit` -- Data vs. model overlay.
* :func:`~heterodyne.viz.nlsq_plots.plot_residual_map` -- 2-D residual heatmap.
* :func:`~heterodyne.viz.nlsq_plots.plot_parameter_uncertainties` -- Error bars.
* :func:`~heterodyne.viz.nlsq_plots.plot_multistart_summary` -- Multi-start comparison.

CMC diagnostics:

* :func:`~heterodyne.viz.mcmc_plots.plot_trace` -- Trace plots per chain.
* :func:`~heterodyne.viz.mcmc_plots.plot_posterior` -- Marginal posteriors.
* :func:`~heterodyne.viz.mcmc_plots.plot_corner` -- Pairwise corner plot.
* :func:`~heterodyne.viz.mcmc_plots.plot_rhat_summary` -- R-hat summary bar chart.
* :func:`~heterodyne.viz.mcmc_plots.plot_energy` -- Energy diagnostic plot.
* :func:`~heterodyne.viz.mcmc_diagnostics.plot_convergence_diagnostics` -- Combined dashboard.
* :func:`~heterodyne.viz.mcmc_dashboard.plot_cmc_summary_dashboard` -- Full CMC summary.

See :doc:`../04_practical_guides/visualization` for usage examples.
