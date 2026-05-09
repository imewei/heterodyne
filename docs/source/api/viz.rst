=============
Visualization
=============

Publication-quality plotting for MCMC diagnostics, NLSQ fit results,
experimental data overlays, interactive dashboards, and summary reports.

MCMC Plots
==========

.. automodule:: heterodyne.viz.mcmc_plots
   :members: plot_trace, plot_posterior, plot_corner
   :undoc-members:
   :show-inheritance:

ArviZ Integration
=================

Convert CMC results to ArviZ ``InferenceData`` and render standard
ArviZ diagnostic plots (trace, posterior, pair plots).

.. automodule:: heterodyne.viz.mcmc_arviz
   :members: to_inference_data, plot_arviz_trace,
             plot_arviz_posterior, plot_arviz_pair
   :undoc-members:
   :show-inheritance:

MCMC Diagnostics Plots
======================

.. automodule:: heterodyne.viz.mcmc_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

MCMC Dashboard
==============

.. automodule:: heterodyne.viz.mcmc_dashboard
   :members:
   :undoc-members:
   :show-inheritance:

NLSQ vs CMC Comparison
======================

Side-by-side overlays comparing NLSQ point estimates with CMC
posterior distributions across angles.

.. automodule:: heterodyne.viz.mcmc_comparison
   :members: plot_nlsq_vs_cmc, plot_multi_angle_comparison
   :undoc-members:
   :show-inheritance:

NLSQ Plots
===========

.. automodule:: heterodyne.viz.nlsq_plots
   :members: plot_nlsq_fit, plot_residual_map, plot_parameter_uncertainties
   :undoc-members:
   :show-inheritance:

Experimental Plots
==================

.. automodule:: heterodyne.viz.experimental_plots
   :members:
   :undoc-members:
   :show-inheritance:

Diagonal Overlay Diagnostics
=============================

Two-time correlation diagonal statistics for validating stationarity
and ergodicity of the measured correlation function.

.. automodule:: heterodyne.viz.diagnostics
   :members: DiagonalOverlayResult, compute_diagonal_overlay_stats
   :undoc-members:
   :show-inheritance:

MCMC Report
===========

.. automodule:: heterodyne.viz.mcmc_report
   :members: ReportConfig, generate_report
   :undoc-members:
   :show-inheritance:
