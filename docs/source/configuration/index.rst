Configuration
=============

Heterodyne uses a YAML configuration file to control every aspect of an
analysis run.  The configuration is organized into the following sections:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Section
     - Purpose
   * - ``metadata``
     - Config version string and free-text description.
   * - ``analyzer_parameters``
     - Frame range, time step ``dt``, and scattering wavevector ``q``.
   * - ``experimental_data``
     - HDF5 file path and cache settings.
   * - ``initial_parameters``
     - Starting values for the 14 physics + 2 scaling parameters.
   * - ``optimization``
     - Solver selection (``nlsq`` / ``cmc``) and per-solver options.
   * - ``output``
     - Results directory, output formats, and checkpoint settings.

Generating a Default Config
---------------------------

.. code-block:: bash

   heterodyne-config --output my_config.yaml

This writes a fully commented template with sensible defaults for all
sections.  Edit the file to match your experiment before running the
analysis.

Running with a Config
---------------------

.. code-block:: bash

   heterodyne --config my_config.yaml --method nlsq

The ``--method`` flag selects the optimization backend.  When omitted, the
method specified inside the YAML ``optimization.method`` key is used.

Section Reference
-----------------

.. toctree::
   :maxdepth: 2

   templates
   options
