Developer Guide
===============

This guide covers everything needed to contribute to the Heterodyne package:
environment setup, testing, code standards, and architecture.

Quick Reference
---------------

.. code-block:: bash

   make dev           # Install in development mode (uv sync)
   make test          # Run unit tests
   make test-all      # Run the full test suite (unit + integration + slow)
   make quality        # Lint (Ruff) + type-check (MyPy)
   make docs          # Build Sphinx documentation

Tool Chain
----------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Tool
     - Command
     - Purpose
   * - **uv**
     - ``uv sync``
     - Package manager and lockfile management.
   * - **Ruff**
     - ``uv run ruff check .``
     - Linting and import sorting.
   * - **MyPy**
     - ``uv run mypy .``
     - Static type checking with strict hints at API boundaries.
   * - **pytest**
     - ``uv run pytest``
     - Test runner with marker-based selection.
   * - **Sphinx**
     - ``make docs``
     - Documentation build (Furo theme, MyST for Markdown).

Sections
--------

.. toctree::
   :maxdepth: 2

   contributing_guide
   testing_guide
   architecture
   parameter_bounds_verification
