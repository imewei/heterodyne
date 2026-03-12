.. _result-interpretation:

=====================
Result Interpretation
=====================

After fitting, the package returns structured result objects that carry
fitted parameters, uncertainties, diagnostics, and metadata.  This page
explains how to read and validate those results.


NLSQ Results
============

:class:`~heterodyne.optimization.nlsq.results.NLSQResult` is returned
by ``fit_nlsq_jax`` and ``fit_nlsq_multi_phi``.

Key fields
----------

``parameters`` (``np.ndarray``)
   Fitted parameter values in the order given by ``parameter_names``.

``parameter_names`` (``list[str]``)
   Ordered list of parameter names matching ``parameters``.

``success`` (``bool``)
   Whether the optimiser converged.

``message`` (``str``)
   Termination message from the solver.

``uncertainties`` (``np.ndarray | None``)
   One-sigma uncertainties derived from the diagonal of the covariance
   matrix.  ``None`` if the covariance could not be estimated (e.g.,
   singular Jacobian).

``covariance`` (``np.ndarray | None``)
   Full parameter covariance matrix.  Use
   ``result.get_correlation_matrix()`` to obtain the normalised
   correlation matrix for detecting parameter degeneracies.

``reduced_chi_squared`` (``float | None``)
   :math:`\chi^2 / \nu` where :math:`\nu` is the number of degrees of
   freedom (data points minus free parameters).

``residuals`` (``np.ndarray | None``)
   Residual vector (data minus model) for the upper triangle.

``fitted_correlation`` (``np.ndarray | None``)
   The model :math:`C_2` matrix evaluated at the best-fit parameters.

Convenience methods
-------------------

.. code-block:: python

   # Access a single parameter by name
   d0_ref = result.get_param("D0_ref")

   # Get uncertainty for a parameter
   d0_ref_unc = result.get_uncertainty("D0_ref")

   # All parameters as a dictionary
   params = result.params_dict  # property, not a method call

   # Validate fit quality (returns list of warning strings)
   warnings = result.validate()
   for w in warnings:
       print(f"WARNING: {w}")


CMC Results
===========

:class:`~heterodyne.optimization.cmc.results.CMCResult` is returned by
the Bayesian pipeline.

Key fields
----------

``posterior_mean`` (``np.ndarray``)
   Mean of the posterior distribution for each parameter.

``posterior_std`` (``np.ndarray``)
   Standard deviation of the posterior for each parameter.

``credible_intervals`` (``dict[str, dict[str, float]]``)
   Per-parameter credible intervals.  Each entry contains
   ``lower_95``, ``upper_95``, ``lower_89``, ``upper_89``.

``convergence_passed`` (``bool``)
   Whether all convergence diagnostics passed.

``r_hat`` (``np.ndarray | None``)
   Gelman--Rubin :math:`\hat{R}` statistic for each parameter.
   Values below 1.01 indicate convergence.

``ess_bulk`` / ``ess_tail`` (``np.ndarray | None``)
   Effective sample sizes for the bulk and tails of the posterior.
   Values above 400 are generally adequate.

``bfmi`` (``list[float] | None``)
   Bayesian Fraction of Missing Information for each chain.
   Values below 0.3 suggest the sampler is struggling with the
   energy landscape.

``samples`` (``dict[str, np.ndarray] | None``)
   Raw posterior samples keyed by parameter name.  Shape is
   ``(n_chains * n_draws,)`` or ``(n_chains, n_draws)``.

Convenience methods
-------------------

.. code-block:: python

   # Summary for one parameter
   summary = result.get_param_summary("D0_ref")
   # {'mean': ..., 'std': ..., 'lower_95': ..., 'upper_95': ..., 'r_hat': ..., 'ess_bulk': ...}

   # All posterior means as dict
   means = result.params_dict()  # method call (not property)

   # Validate convergence against thresholds
   warnings = result.validate_convergence(
       r_hat_threshold=1.1,
       min_ess=100,
       min_bfmi=0.3,
   )


Goodness of Fit
================

Reduced chi-squared
-------------------

The primary goodness-of-fit metric for NLSQ is the **reduced
chi-squared**:

.. math::

   \chi^2_\text{red} = \frac{\chi^2}{\nu}

where :math:`\nu = N_\text{data} - N_\text{params}` is the number of
degrees of freedom.

* :math:`\chi^2_\text{red} \approx 1.0` -- Good fit.  The model
  explains the data within its noise level.
* :math:`\chi^2_\text{red} \gg 1` -- Poor fit.  The model is missing
  systematic features, or the data has structure not captured by the
  14-parameter model.
* :math:`\chi^2_\text{red} \ll 1` -- Possible overfit.  The error
  bars may be overestimated, or the model has too many free
  parameters relative to the information content.


Parameter Uncertainties
========================

From NLSQ (covariance matrix)
------------------------------

The covariance matrix :math:`\mathbf{C}` is estimated from the Jacobian
at the optimum:

.. math::

   \mathbf{C} \approx \bigl(\mathbf{J}^T \mathbf{J}\bigr)^{-1}\,
                       \chi^2_\text{red}

One-sigma uncertainties are :math:`\sigma_i = \sqrt{C_{ii}}`.  These
are **local** estimates valid near the optimum and assume the posterior
is approximately Gaussian.

From CMC (posterior samples)
-----------------------------

The Bayesian posterior provides the full marginal distribution for each
parameter.  Credible intervals are computed directly from the sample
quantiles and do not assume Gaussianity.  Use the 89% HDI (Highest
Density Interval) as a robust summary, or the 95% interval for
traditional reporting.

.. code-block:: python

   # Extract 95% credible interval
   ci = result.credible_intervals["D0_ref"]
   print(f"D0_ref: [{ci['lower_95']:.2e}, {ci['upper_95']:.2e}]")


Comparing NLSQ and CMC
========================

When both results are available, compare them to validate consistency:

.. code-block:: python

   from heterodyne.optimization.cmc.results import compare_cmc_nlsq

   comparison = compare_cmc_nlsq(cmc_result, nlsq_result, consistency_sigma=2.0)
   print(f"Consistent parameters: {comparison['n_consistent']}/{len(comparison['common_parameters'])}")

   for name, consistent in comparison["consistent"].items():
       if not consistent:
           dev = comparison["relative_deviations"][name]
           print(f"  {name}: {dev:.1f} sigma from CMC mean")

Parameters that disagree by more than 2 posterior standard deviations
may indicate that the NLSQ solution is trapped in a local minimum, or
that the posterior is strongly non-Gaussian.
