Testing Guide
=============

Heterodyne uses **pytest** as its test runner with marker-based selection
for granular control over which tests execute.

Test Structure
--------------

.. code-block:: text

   tests/
   ├── unit/              # Fast, isolated tests (no I/O, no JIT warmup)
   ├── integration/       # End-to-end pipeline tests
   └── conftest.py        # Shared fixtures, JAX configuration

Test Markers
------------

Tests are tagged with markers so that subsets can be selected on the command
line:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Marker
     - Description
   * - ``@pytest.mark.unit``
     - Fast, isolated unit tests.
   * - ``@pytest.mark.integration``
     - End-to-end tests that exercise multiple subsystems.
   * - ``@pytest.mark.slow``
     - Tests that take more than a few seconds (large data, many iterations).
   * - ``@pytest.mark.mcmc``
     - Tests that run MCMC / NUTS sampling.
   * - ``@pytest.mark.arviz``
     - Tests that depend on ArviZ diagnostics.
   * - ``@pytest.mark.validation``
     - Numerical validation against reference implementations.

Running Tests
-------------

.. code-block:: bash

   # All unit tests (fast)
   uv run pytest -m unit

   # Full suite including slow and integration
   make test-all

   # Quick smoke test
   make test

   # Specific marker
   uv run pytest -m "mcmc and not slow"

   # Single file
   uv run pytest tests/unit/test_physics_utils.py -v

Coverage
--------

Coverage is collected automatically in CI.  To generate a local report:

.. code-block:: bash

   uv run pytest --cov=heterodyne --cov-report=html
   open htmlcov/index.html

The coverage configuration lives in ``pyproject.toml`` under
``[tool.coverage]``.

JAX-Specific Testing Considerations
------------------------------------

**Float64 precision.**
JAX defaults to float32.  The test ``conftest.py`` enables 64-bit mode
via ``jax.config.update("jax_enable_x64", True)`` so that all tests run in
float64, matching the production pipeline.

**JIT compilation overhead.**
The first call to a JIT-compiled function incurs tracing and compilation
cost.  Tests that benchmark runtime should either:

- Call the function once as a warm-up before timing, or
- Use the ``@pytest.mark.slow`` marker so they are excluded from the fast
  suite.

**Deterministic seeds.**
All stochastic tests must set an explicit PRNG key
(``jax.random.PRNGKey(seed)``) to guarantee reproducibility across runs
and platforms.

**Pure functions.**
Physics kernels are stateless JAX functions.  Test them by passing explicit
arrays rather than relying on global state.  This also ensures they remain
JIT-compatible.

Writing New Tests
-----------------

1. Place the test in the appropriate directory (``unit/`` or
   ``integration/``).
2. Add the correct marker(s) to the test function or class.
3. Use fixtures from ``conftest.py`` for common setup (data arrays, model
   instances, default parameter vectors).
4. Assert on numerical values with ``np.testing.assert_allclose`` and
   explicit ``atol`` / ``rtol`` tolerances.
