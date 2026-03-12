Parameter Bounds Verification
=============================

This page documents the canonical parameter bounds, defaults, and the dual
prior system.  It serves as a reference for verifying that the parameter
registry and parameter space remain in sync.

Invariants
----------

- All 14 physics parameters and 2 scaling parameters have
  ``vary_default=True``.
- Units follow the Angstrom convention throughout:
  :math:`q` in :math:`\text{\AA}^{-1}`,
  :math:`D_0` in :math:`\text{\AA}^2/\text{s}^\alpha`,
  velocities in :math:`\text{\AA}/\text{s}`.

Key Parameter Defaults
----------------------

.. list-table::
   :widths: 25 15 15 15 30
   :header-rows: 1

   * - Parameter
     - Default
     - Min
     - Max
     - Notes
   * - ``D0_ref``
     - 1e4
     - 0
     - 1e6
     - ``max_bound=1e6``
   * - ``D0_sample``
     - 1e4
     - 0
     - 1e6
     - ``max_bound=1e6``
   * - ``alpha_ref``
     - 0.0
     - -2.0
     - 2.0
     - Default is 0.0, not 1.0.
   * - ``alpha_sample``
     - 0.0
     - -2.0
     - 2.0
     - Default is 0.0, not 1.0.
   * - ``v0``
     - 1e3
     - 0
     - 1e6
     - Prior: (1e3, 500).
   * - ``v_offset``
     - 0.0
     - -100
     - 100
     - Allows negative values.

Dual Prior System
-----------------

Heterodyne maintains two parallel sources of prior information.  Both
**must** stay in sync when parameter defaults or priors are updated.

**Source 1: Parameter Registry**
(``heterodyne/config/parameter_registry.py``)

The immutable ``MappingProxyType`` registry defines ``prior_mean`` and
``prior_std`` for each parameter.  These values are consumed by:

- ``cmc/priors.py:build_default_priors()``
- ``cmc/priors.py:build_log_space_priors()``

**Source 2: Parameter Space**
(``heterodyne/optimization/parameter_space.py``)

The ``_DEFAULT_PRIOR_SPECS`` dictionary defines per-parameter prior
specifications consumed by ``_default_prior()`` during ``ParameterSpace``
initialization.

Verification Procedure
~~~~~~~~~~~~~~~~~~~~~~

When modifying any parameter default, bound, or prior:

1. Update the value in ``parameter_registry.py``.
2. Update the corresponding entry in ``_DEFAULT_PRIOR_SPECS`` in
   ``parameter_space.py``.
3. Run the validation test suite:

   .. code-block:: bash

      uv run pytest -m validation -v

4. Confirm that ``build_default_priors()`` and ``build_log_space_priors()``
   produce consistent prior distributions by inspecting the test output.

Common Pitfalls
---------------

- Setting ``alpha_ref`` or ``alpha_sample`` default to 1.0 instead of 0.0.
- Forgetting to update ``_DEFAULT_PRIOR_SPECS`` after changing the registry
  (or vice versa).
- Using nanometer units instead of Angstroms (all values must be in
  :math:`\text{\AA}`).
