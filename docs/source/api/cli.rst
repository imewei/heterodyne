===
CLI
===

Command-line interface entry points, analysis commands, configuration
generation, XLA flag setup, optimization orchestration, and data
pipeline management.

Entry Points
============

The following commands are registered as console scripts:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Command
     - Short alias
     - Purpose
   * - ``heterodyne``
     - ``ht``
     - Main analysis (NLSQ/CMC)
   * - ``heterodyne-config``
     - ``ht-config``
     - Config generation/validation
   * - ``heterodyne-config-xla``
     - ``ht-config-xla``
     - XLA device configuration
   * - ``heterodyne-post-install``
     - ``ht-post-install``
     - Shell completion setup
   * - ``heterodyne-cleanup``
     - ``ht-cleanup``
     - Remove shell completion files
   * - ``heterodyne-validate``
     - ``ht-validate``
     - System validation
   * - ``hexp``
     - —
     - Plot experimental data (skip optimization)
   * - ``hsim``
     - —
     - Plot simulated C2 heatmaps from config parameters

Plotting Commands
-----------------

``hexp`` and ``hsim`` are standalone plotting entry points that bypass
optimization entirely:

.. code-block:: bash

   # Inspect experimental data for quality checking
   hexp --config config.yaml

   # Preview simulated C2 heatmaps with custom scaling
   hsim --config config.yaml --contrast 0.3 --offset-sim 1.0

These are equivalent to passing ``--plot-experimental-data`` or
``--plot-simulated-data`` to the main ``heterodyne`` command.

Main Module
===========

.. automodule:: heterodyne.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

Commands
========

.. automodule:: heterodyne.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:

Config Generator
================

.. automodule:: heterodyne.cli.config_generator
   :members:
   :undoc-members:
   :show-inheritance:

XLA Configuration
=================

.. automodule:: heterodyne.cli.xla_config
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Runner
===================

.. automodule:: heterodyne.cli.optimization_runner
   :members:
   :undoc-members:
   :show-inheritance:

Data Pipeline
=============

.. automodule:: heterodyne.cli.data_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
