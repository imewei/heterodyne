.. _theory:

==================
Theory & Physics
==================

The heterodyne XPCS analysis framework implements the two-component
scattering theory developed by He *et al.* for studying nonequilibrium
dynamics under flow conditions. The theoretical foundation spans two
publications:

- **PNAS 2024** (doi:`10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_):
  Introduces the generalized two-time correlation framework, the transport
  coefficient :math:`J(t)`, and the integral formulation connecting
  microscopic velocity statistics to measurable scattering correlations.

- **PNAS 2025** (doi:`10.1073/pnas.2514216122 <https://doi.org/10.1073/pnas.2514216122>`_):
  Extends the theory to multi-component heterodyne detection, derives the
  two-component interference model, and demonstrates extraction of flow
  velocity from cross-correlation oscillations.

The key insight is that **heterodyne scattering** --- where a static
reference field interferes with scattered light from a moving sample ---
produces an oscillatory cross-correlation term whose frequency encodes the
sample velocity. Combined with the transport coefficient formalism, this
enables simultaneous measurement of diffusion, flow velocity, and
composition dynamics from a single two-time correlation measurement.

All equations reference the Supporting Information (SI) numbering from the
PNAS publications. Physical quantities use angstrom-based units throughout:
:math:`q` in :math:`\text{\AA}^{-1}`, :math:`D_0` in
:math:`\text{\AA}^2/\text{s}^\alpha`, and velocities in
:math:`\text{\AA/s}`.

.. important::

   **Numerical integration only.** The implementation always evaluates the
   transport and velocity integrals
   (:math:`\int_{t_i}^{t_j} J(t')\, dt'` and
   :math:`\int_{t_i}^{t_j} v(t')\, dt'`) numerically via
   ``trapezoid_cumsum``. Analytical antiderivatives are **never**
   substituted, even for special cases (e.g., constant :math:`J`). The
   power-law parameterization :math:`J(t) = D_0 t^\alpha + D_\mathrm{offset}`
   has no useful closed-form antiderivative for general :math:`\alpha`,
   so numerical integration is the only correct approach.

.. toctree::
   :maxdepth: 2
   :caption: Theory Topics

   transport_coefficient
   correlation_functions
   heterodyne_scattering
   classical_processes
   computational_methods
   analysis_modes
   citations
