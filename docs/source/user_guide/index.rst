.. _user-guide:

==========
User Guide
==========

This guide covers the full workflow for analysing two-component heterodyne
XPCS data with the **heterodyne** package, from loading raw correlation
matrices to interpreting Bayesian posteriors.

.. tip::

   Choose a **learning pathway** below based on your background and goals.
   Each pathway lists pages in the recommended reading order.


Learning Pathways
=================

Path A -- New to XPCS
----------------------

Start with the physics, then walk through a complete analysis.

1. :doc:`01_fundamentals/what_is_xpcs`
2. :doc:`01_fundamentals/heterodyne_overview`
3. :doc:`02_data_and_fitting/data_loading`
4. :doc:`02_data_and_fitting/nlsq_fitting`
5. :doc:`02_data_and_fitting/result_interpretation`
6. :doc:`01_fundamentals/parameter_guide`


Path B -- Two-Component Dynamics
---------------------------------

You know X-ray scattering but want to understand the heterodyne
two-component model in depth.

1. :doc:`01_fundamentals/what_is_xpcs`
2. :doc:`01_fundamentals/heterodyne_overview`
3. :doc:`03_advanced_topics/per_angle_modes`
4. :doc:`03_advanced_topics/bayesian_inference`
5. :doc:`05_appendices/troubleshooting`
6. :doc:`01_fundamentals/parameter_guide`


Path C -- Bayesian Uncertainty Quantification
---------------------------------------------

You are comfortable with the model and want to move from point estimates
to full posteriors.

1. :doc:`02_data_and_fitting/nlsq_fitting`
2. :doc:`03_advanced_topics/bayesian_inference`
3. :doc:`03_advanced_topics/diagnostics`
4. :doc:`02_data_and_fitting/result_interpretation`
5. :doc:`04_practical_guides/visualization`


Path D -- Advanced / Performance
---------------------------------

Optimise configurations, tune performance, and explore global optimisation.

1. :doc:`04_practical_guides/configuration`
2. :doc:`04_practical_guides/performance_tuning`
3. :doc:`03_advanced_topics/cmaes_optimization`
4. :doc:`01_fundamentals/parameter_guide`
5. :doc:`03_advanced_topics/diagnostics`


Guide Contents
==============

.. toctree::
   :maxdepth: 2

   01_fundamentals/index
   02_data_and_fitting/index
   03_advanced_topics/index
   04_practical_guides/index
   05_appendices/index
