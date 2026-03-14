.. _visualization:

=============
Visualization
=============

The ``heterodyne.viz`` module provides a comprehensive set of plotting
functions for inspecting fits, diagnosing convergence, and generating
publication-quality figures.  All functions return Matplotlib figure
objects that can be further customised.


NLSQ Plots
-----------

plot_nlsq_fit
-------------

Overlays the best-fit model on the measured :math:`C_2` matrix.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_nlsq_fit

   fig = plot_nlsq_fit(result, c2_data, timestamps)
   fig.savefig("nlsq_fit.png", dpi=150)

plot_residual_map
-----------------

Displays the residual :math:`C_2^\text{data} - C_2^\text{model}` as a
2-D heatmap.  Systematic patterns indicate model deficiencies.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_residual_map

   fig = plot_residual_map(result)

plot_parameter_uncertainties
----------------------------

Bar chart of fitted parameters with error bars from the covariance
matrix.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_parameter_uncertainties

   fig = plot_parameter_uncertainties(result)

plot_fit_surface
----------------

3-D surface plot comparing data and model :math:`C_2` matrices.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_fit_surface

   fig = plot_fit_surface(result, c2_data, timestamps)

plot_multistart_summary
-----------------------

Compares :math:`\chi^2` and parameter values across multi-start runs.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_multistart_summary

   fig = plot_multistart_summary(multistart_results)

plot_chi_squared_landscape
--------------------------

2-D :math:`\chi^2` landscape as a function of two chosen parameters,
with the optimum marked.

.. code-block:: python

   from heterodyne.viz.nlsq_plots import plot_chi_squared_landscape

   fig = plot_chi_squared_landscape(
       model, c2_data, result,
       param_x="D0_ref", param_y="v0",
       n_grid=50,
   )


CMC Plots
---------

plot_trace
----------

Trace plots showing sampled values vs. iteration for each chain.
Essential for visual convergence assessment.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_trace

   fig = plot_trace(result, params=["D0_ref", "v0", "f0"])

plot_posterior
--------------

Marginal posterior histograms with credible intervals.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_posterior

   fig = plot_posterior(result, params=["D0_ref", "alpha_ref"])

plot_corner
-----------

Pairwise corner plot showing 2-D marginal posteriors and correlations.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_corner

   fig = plot_corner(result, params=["D0_ref", "D0_sample", "v0"])

plot_forest
-----------

Forest plot comparing posterior intervals across parameters or shards.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_forest

   fig = plot_forest(result)

plot_energy
-----------

Energy diagnostic plot showing the marginal energy distribution and
BFMI values.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_energy

   fig = plot_energy(result)

plot_rhat_summary
-----------------

Bar chart of :math:`\hat{R}` values for all parameters with threshold
lines.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_rhat_summary

   fig = plot_rhat_summary(result, threshold=1.01)

plot_shard_comparison
---------------------

Compares posterior means across CMC shards to detect bimodality.

.. code-block:: python

   from heterodyne.viz.mcmc_plots import plot_shard_comparison

   fig = plot_shard_comparison(shard_results)


Diagnostic Dashboards
---------------------

plot_convergence_diagnostics
----------------------------

Combined panel showing ESS evolution, adaptation summary, and
divergence scatter.

.. code-block:: python

   from heterodyne.viz.mcmc_diagnostics import plot_convergence_diagnostics

   fig = plot_convergence_diagnostics(result)

plot_cmc_summary_dashboard
--------------------------

Full CMC summary dashboard combining trace plots, posteriors, R-hat
summary, and shard comparison in a single multi-panel figure.

.. code-block:: python

   from heterodyne.viz.mcmc_dashboard import plot_cmc_summary_dashboard

   fig = plot_cmc_summary_dashboard(result, shard_results=shard_results)
   fig.savefig("cmc_dashboard.png", dpi=150, bbox_inches="tight")


Experimental Data Plots
------------------------

These functions visualise the raw or preprocessed data before fitting.

plot_correlation
----------------

Heatmap of the measured :math:`C_2(t_1, t_2)` matrix.

.. code-block:: python

   from heterodyne.viz.experimental_plots import plot_correlation

   fig = plot_correlation(c2_data, timestamps)

plot_g1_components
------------------

Decomposes the fitted model into its three components (reference
self-correlation, sample self-correlation, cross-correlation) and
plots each separately.

.. code-block:: python

   from heterodyne.viz.experimental_plots import plot_g1_components

   fig = plot_g1_components(model, result.parameters, timestamps)

plot_phi_dependence
-------------------

Shows how fitted parameters (especially velocity-related ones) vary
with azimuthal angle :math:`\phi`.

.. code-block:: python

   from heterodyne.viz.experimental_plots import plot_phi_dependence

   fig = plot_phi_dependence(results_by_angle, phi_angles)


ArviZ Integration
-----------------

For users who prefer ArviZ's built-in plotting, convert the CMC result
to an ``InferenceData`` object:

.. code-block:: python

   from heterodyne.optimization.cmc.results import cmc_result_to_arviz
   import arviz as az

   idata = cmc_result_to_arviz(result)

   # Use any ArviZ plot
   az.plot_trace(idata)
   az.plot_pair(idata, var_names=["D0_ref", "v0"])
   az.plot_ess(idata)
