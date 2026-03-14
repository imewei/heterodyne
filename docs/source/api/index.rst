=============
API Reference
=============

Complete API documentation for the ``heterodyne`` XPCS analysis package.
All public modules, classes, and functions are documented with full type
signatures and cross-references.

Quick Navigation
================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - :doc:`core`
     - Physics constants, safe numerics, and integral primitives
   * - :doc:`heterodyne_model`
     - Main ``HeterodyneModel`` wrapper with ``ParameterManager``
   * - :doc:`jax_backend`
     - JAX/JIT meshgrid correlation computation
   * - :doc:`models`
     - Model hierarchy: base, two-component, reduced, and factory
   * - :doc:`nlsq`
     - Non-linear least squares optimization (scipy + JAX JIT)
   * - :doc:`cmc`
     - Bayesian MCMC via NumPyro (NUTS / CMC)
   * - :doc:`data`
     - XPCS data loading, validation, and preprocessing
   * - :doc:`config`
     - Configuration management, parameter registry, and parameter space
   * - :doc:`cli`
     - Command-line interface entry points and runners
   * - :doc:`device`
     - Hardware detection and CPU/NUMA configuration
   * - :doc:`io`
     - Result serialization (JSON, NPZ, MCMC diagnostics)
   * - :doc:`viz`
     - MCMC and NLSQ visualization, dashboards, and reports
   * - :doc:`utils`
     - Logging configuration and path validation utilities

Core Physics
------------

Foundational physics constants, numerical primitives, and correlation models.

.. toctree::
   :maxdepth: 2

   core
   heterodyne_model
   jax_backend
   models

Optimization
------------

NLSQ warm-start and Bayesian posterior sampling pipelines.

.. toctree::
   :maxdepth: 2

   optimization

Infrastructure
--------------

Data I/O, configuration, CLI, device management, visualization, and utilities.

.. toctree::
   :maxdepth: 2

   data
   config
   cli
   device
   io
   viz
   utils
