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

Scaling
=======

Parameter scaling utilities for CMC warm-start initialization and
contrast/offset estimation.

.. automodule:: heterodyne.optimization.cmc.scaling
   :members: ParameterScaling
   :undoc-members:
   :show-inheritance:

Data Preparation
================

.. automodule:: heterodyne.optimization.cmc.data_prep
   :members:
   :undoc-members:
   :show-inheritance:

CMC I/O
=======

Full shard-level and aggregate I/O pipeline for CMC results.  Supports
NPZ sample arrays, ArviZ ``InferenceData``, JSON parameter/diagnostics
files, and fitted-data arrays.

.. automodule:: heterodyne.optimization.cmc.io
   :members: save_shard_results, load_shard_results, list_shards,
             save_inference_data, load_inference_data,
             save_samples_npz, load_samples_npz, samples_to_arviz,
             save_fitted_data_npz, save_parameters_json,
             save_diagnostics_json, save_all_results
   :undoc-members:
   :show-inheritance:

CMC Plotting
============

ArviZ-backed diagnostic plots for CMC posterior samples.

.. automodule:: heterodyne.optimization.cmc.plotting
   :members: plot_trace_summary, plot_pair_plot,
             plot_posterior_predictive, plot_diagnostics_summary
   :undoc-members:
   :show-inheritance:

Backends
========

Execution backends for running NUTS chains across CPU cores, HPC
cluster nodes, and multi-device pjit configurations.  All backends
share the :class:`~heterodyne.optimization.cmc.backends.base.MCMCBackend`
protocol and are selected automatically via
:func:`~heterodyne.optimization.cmc.backends.base.select_backend` based on
the :attr:`~heterodyne.optimization.cmc.config.CMCConfig.chain_method` setting.

Backend Selection
-----------------

.. automodule:: heterodyne.optimization.cmc.backends.base
   :members: MCMCBackend, ShardPosterior, select_backend,
             combine_shard_samples, combine_shard_samples_bimodal,
             consensus_mc, robust_consensus_mc
   :undoc-members:
   :show-inheritance:

CPU Backend (Sequential / Vectorized)
--------------------------------------

.. automodule:: heterodyne.optimization.cmc.backends.cpu_backend
   :members: CPUBackend
   :undoc-members:
   :show-inheritance:

Multiprocessing Backend
-----------------------

Parallel shard evaluation across CPU cores using Python
``multiprocessing``.  Each worker process runs an independent NUTS
chain on its own shard, with full JAX persistent compilation cache
support.

.. automodule:: heterodyne.optimization.cmc.backends.multiprocessing
   :members: MultiprocessingBackend
   :undoc-members:
   :show-inheritance:

PBS/HPC Backend
---------------

Submits per-shard NUTS chains as PBS job-array tasks for use on HPC
cluster nodes (e.g. ALCF Polaris, NERSC Perlmutter).

.. automodule:: heterodyne.optimization.cmc.backends.pbs
   :members: PBSBackend, PBSConfig
   :undoc-members:
   :show-inheritance:

Pjit Backend (Multi-Device)
----------------------------

Experimental backend that shards chains across ``jax.devices()`` via
``pjit``.  Currently CPU-only because heterodyne ships no GPU support,
but the interface is device-agnostic.

.. automodule:: heterodyne.optimization.cmc.backends.pjit
   :members: PjitBackend
   :undoc-members:
   :show-inheritance:

Persistent Worker Pool
-----------------------

Long-lived worker pool that amortises per-shard JAX JIT compilation
overhead.  Recommended for large analyses with many shards.

.. automodule:: heterodyne.optimization.cmc.backends.worker_pool
   :members: PersistentWorkerPool, WorkerPoolBackend, should_use_persistent_pool
   :undoc-members:
   :show-inheritance:
