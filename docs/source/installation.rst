Installation
============

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Purpose
   * - Python
     - >= 3.12
     - Runtime (tested with 3.12.x and 3.13.x)
   * - JAX
     - >= 0.8.2
     - JIT-compiled numerical kernels (CPU-only)
   * - NumPy
     - >= 2.3
     - Array operations
   * - NLSQ
     - >= 0.6.10
     - Trust-region Levenberg--Marquardt solver
   * - NumPyro
     - >= 0.19
     - Bayesian MCMC / NUTS inference
   * - h5py
     - >= 3.15
     - HDF5 data I/O
   * - evosax
     - >= 0.2.0
     - CMA-ES evolutionary optimization

Install with pip
----------------

.. code-block:: bash

   pip install heterodyne

Install with uv (recommended)
------------------------------

.. code-block:: bash

   uv add heterodyne

For an editable development install:

.. code-block:: bash

   git clone https://github.com/imewei/heterodyne.git
   cd heterodyne
   uv sync

Development install
-------------------

Install with all development and documentation dependencies:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"

Or use the Makefile:

.. code-block:: bash

   make dev

Make Targets
------------

The project Makefile provides common development workflows:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Target
     - Description
   * - ``make dev``
     - Install package with development and documentation dependencies
   * - ``make test``
     - Run all tests
   * - ``make test-all``
     - Run all tests with coverage report
   * - ``make quality``
     - Run all quality checks (format + lint + type-check)
   * - ``make docs``
     - Build Sphinx documentation

Verify Installation
-------------------

After installation, verify that the package and its dependencies are correctly
configured:

.. code-block:: bash

   heterodyne-validate

This checks that JAX, NLSQ, NumPyro, and all required dependencies are
importable and reports their versions.

You can also verify from Python:

.. code-block:: python

   import heterodyne
   print(heterodyne.__version__)

   import jax
   print(jax.default_backend())  # Should print "cpu"

Shell Completion
----------------

Run the post-install script to set up shell completions and recommended
aliases:

.. code-block:: bash

   heterodyne-post-install

This installs tab completion for Bash, Zsh, and Fish shells.

**Recommended aliases** (added to your shell profile by ``heterodyne-post-install``):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Alias
     - Expands to
     - Purpose
   * - ``het``
     - ``heterodyne``
     - Main CLI
   * - ``hconfig``
     - ``heterodyne-config``
     - Config generator
   * - ``het-nlsq``
     - ``heterodyne --method nlsq``
     - NLSQ shortcut
   * - ``het-cmc``
     - ``heterodyne --method cmc``
     - CMC shortcut
   * - ``hxla``
     - ``heterodyne-config-xla``
     - XLA flag configurator
   * - ``hsetup``
     - ``heterodyne-post-install``
     - Post-install setup
   * - ``hclean``
     - ``heterodyne-cleanup``
     - Cleanup utility

Exit Codes
----------

All CLI commands use consistent exit codes:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success
   * - ``1``
     - General error (invalid arguments, missing files)
   * - ``2``
     - Configuration error (invalid YAML, missing required fields)
   * - ``3``
     - Data error (corrupt HDF5, shape mismatch, NaN detected)
   * - ``4``
     - Optimization error (convergence failure, numerical instability)
   * - ``255``
     - Unhandled exception (bug -- please report)

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``JAX_PLATFORMS``
     - Force JAX backend platform (set to ``cpu`` for heterodyne)
   * - ``JAX_ENABLE_X64``
     - Enable 64-bit floating point in JAX (``true`` recommended)
   * - ``OMP_NUM_THREADS``
     - Control OpenMP thread count for BLAS operations
   * - ``HETERODYNE_OUTPUT_DIR``
     - Override default output directory for results
   * - ``HETERODYNE_DEBUG``
     - Enable debug-level logging (``1`` to enable)

CPU Optimization
----------------

Heterodyne includes a dedicated XLA flag configurator that tunes JAX's
compiler for your specific CPU topology (core count, NUMA nodes, cache
hierarchy):

.. code-block:: bash

   heterodyne-config-xla

This detects your hardware and writes optimal XLA flags for thread allocation,
intra-op parallelism, and memory layout. The configuration is stored in
``$VIRTUAL_ENV/etc/heterodyne/xla_mode`` (or ``$CONDA_PREFIX/etc/heterodyne/xla_mode``
in conda/mamba environments, or ``~/.config/heterodyne/xla_mode`` outside any
environment) and sourced automatically on subsequent runs.

For HPC environments with many cores (36--128+), this step is strongly
recommended before running any analysis.
