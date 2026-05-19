.. _adr_per_angle_scaling:

ADR-005: Per-Angle Scaling Modes
==================================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

See :ref:`adr_anti_degeneracy` for the primary ADR on the anti-degeneracy system.

This ADR documents specifically the decision to expose **four distinct per-angle scaling
modes** (``auto``, ``constant``, ``individual``, ``fourier``) rather than a single strategy.

The per-angle scaling modes control how the angle-dependent speckle contrast
:math:`\beta(\phi_k)` and offset :math:`c_\mathrm{offset}(\phi_k)` are handled in NLSQ
optimization. Each mode makes a different assumption about the structure of angular
contrast variation and provides a different trade-off between flexibility and
identifiability.

The heterodyne model has 14 physical parameters. The number of per-angle scaling
parameters added depends on the mode and on :math:`n_\phi` (the number of azimuthal
angles):

- ``constant``: 0 extra params (scaling is fixed, not optimized)
- ``auto`` (``auto_averaged`` path): 2 extra params (one averaged contrast, one averaged offset)
- ``fourier``: :math:`2 \times (2K+1)` extra params (Fourier coefficients, default :math:`K=2` → 10 params)
- ``individual``: :math:`2 \times n_\phi` extra params (one pair per angle)

For typical experiments with :math:`n_\phi = 23` angles, ``individual`` mode adds 46
free parameters on top of the 14 physical parameters, creating a strongly degenerate
landscape.


Decision
--------

Expose all four modes through a single YAML configuration key:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # "auto" | "constant" | "individual" | "fourier"

The modes differ in the number of free parameters and the constraints imposed:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Mode
     - Free params (23 angles)
     - Constraint / assumption
   * - ``constant``
     - 14
     - Scaling fixed from quantile estimate; not optimized
   * - ``auto``
     - 16
     - Quantile-initialized averaging; 2 shared scaling params optimized
   * - ``fourier``
     - 24
     - Smooth angular variation; K=2 Fourier series (5 contrast + 5 offset coefficients)
   * - ``individual``
     - 60
     - No constraint; 46 per-angle params freely optimized (high degeneracy risk)

The default is ``auto``. All modes are documented, tested, and supported.


Auto-Mode Selection Logic
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``auto`` mode uses the following selection logic:

.. code-block:: python

   if n_phi >= fourier_auto_threshold:  # default: 8
       mode = "fourier"
   elif n_phi >= constant_scaling_threshold:  # default: 3
       mode = "auto_averaged"  # 2 shared scaling params
   else:
       mode = "individual"  # per-angle, but n_phi is small enough to be safe

This ensures that the most robust mode is used for the available data without
user intervention.


Rationale
---------

**Why not a single mode?**

Different experimental scenarios require different trade-offs:

- At the beamline with a single angle (``n_phi = 1``), ``individual`` mode is equivalent
  to ``constant`` (no degeneracy), but avoids a special case in the code.
- For parameter sensitivity studies, ``constant`` mode isolates the physical parameters
  by removing the scaling degree of freedom entirely.
- For datasets with smooth angular contrast variation (typical for well-aligned
  detectors), ``fourier`` mode provides the best fit quality with minimal parameters.
- For exploratory analysis or debugging, ``individual`` mode gives direct access to the
  per-angle scaling values.

**Why Fourier series for angular variation?**

The speckle contrast :math:`\beta(\phi_k)` arises from beam coherence and detector
geometry, which are smooth functions of :math:`\phi`. A truncated Fourier series with
:math:`K=2` captures the dominant even harmonics (contrast is symmetric:
:math:`\beta(\phi) = \beta(-\phi)` for any centrosymmetric detector) using only 5
coefficients instead of :math:`n_\phi`. The Fourier basis is model-agnostic: it does not
assume a specific physical mechanism for the angular variation, only smoothness.

**Why quantile initialization?**

The contrast :math:`\beta(\phi_k)` can be estimated from the data before optimization
by computing the 90th percentile of :math:`c_2` values for each angle (the maximum
correlation value, attained at short lag times). This provides a physically meaningful
starting point that is within the basin of the correct solution, reducing the number
of optimizer iterations needed.

**Why ``constant`` mode?**

``constant`` mode is not the recommended default, but it is useful for:

1. Ablation studies: comparing ``auto``/``fourier``/``individual`` against ``constant``
   shows the effect of allowing scaling to vary.
2. Highly constrained experiments where the contrast is known from an independent
   calibration measurement.
3. Fast approximate fits where the scaling is expected to be nearly constant.


Consequences
------------

**Positive**:

- Users can perform model selection by running multiple modes and comparing residuals.
- The ``auto`` mode provides a safe default for production use.
- Each mode is independently tested against known analytical solutions.
- The mode is visible in the output JSON (``result["per_angle_mode_actual"]``), enabling
  reproducibility.

**Negative / Accepted trade-offs**:

- Four modes require four code paths in ``anti_degeneracy_controller.py``, increasing
  complexity.
- ``individual`` mode is exposed despite being high-risk for degeneracy. Users who
  select it explicitly are assumed to understand the risk.
- The ``auto`` threshold parameters (``fourier_auto_threshold``,
  ``constant_scaling_threshold``) are tunable, adding configuration surface area.


Alternatives Considered
-----------------------

**A. Single mode (auto only)**

Simpler user interface. Rejected because: limits the ability to perform model selection
and may not be optimal for all experimental configurations.

**B. Hierarchical Bayesian model for contrast**

Model :math:`\beta(\phi_k) \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)` with hyperpriors.
Correct Bayesian treatment. Rejected because: this is only available in the CMC path;
for the NLSQ point estimate (which runs first), a simple parameterization is needed.
The CMC model could implement this hierarchy in a future version.

**C. Automatic model selection (AIC/BIC)**

Run all modes and select based on information criterion. Rejected because: would
quadruple the optimization time; users who want model selection can run the modes
manually.


.. seealso::

   - :ref:`theory_anti_degeneracy` — mathematical details of all modes
   - :ref:`adr_anti_degeneracy` — primary ADR on the 4-layer system
   - :mod:`heterodyne.optimization.nlsq.anti_degeneracy_controller` — implementation
   - :mod:`heterodyne.optimization.nlsq.fourier_reparam` — Fourier mode
