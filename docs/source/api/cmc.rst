==============
CMC (Bayesian)
==============

Bayesian posterior sampling via NumPyro NUTS, with NLSQ-derived
warm-start initialization, configurable priors, reparameterization,
and ArviZ-compatible convergence diagnostics.

Core Fitting
============

.. automodule:: heterodyne.optimization.cmc.core
   :members: fit_cmc_jax
   :undoc-members:
   :show-inheritance:

Configuration
=============

.. automodule:: heterodyne.optimization.cmc.config
   :members: CMCConfig
   :undoc-members:
   :show-inheritance:

.. note::

   Key attribute names (renamed from legacy):
   ``target_accept_prob``, ``max_r_hat``, ``nlsq_prior_width_factor``.
   The ``from_dict()`` class method handles legacy key translation.

Results
=======

.. automodule:: heterodyne.optimization.cmc.results
   :members: CMCResult
   :undoc-members:
   :show-inheritance:

NumPyro Model
=============

.. automodule:: heterodyne.optimization.cmc.model
   :members:
   :undoc-members:
   :show-inheritance:

Priors
======

.. automodule:: heterodyne.optimization.cmc.priors
   :members: build_default_priors, build_log_space_priors
   :undoc-members:
   :show-inheritance:

Sampler
=======

.. automodule:: heterodyne.optimization.cmc.sampler
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
===========

.. automodule:: heterodyne.optimization.cmc.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Reparameterization
==================

.. automodule:: heterodyne.optimization.cmc.reparameterization
   :members:
   :undoc-members:
   :show-inheritance:
