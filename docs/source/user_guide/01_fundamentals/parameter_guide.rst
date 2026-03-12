.. _parameter-guide:

================
Parameter Guide
================

The heterodyne model uses **14 physics parameters** (shared across all
azimuthal angles) and **2 scaling parameters** per angle.  This page
documents every parameter, its units, typical range, and physical
interpretation.

.. |AA| unicode:: U+00C5
   :ltrim:


Physics Parameters
==================

All lengths are in Angstroms (|AA|), consistent with standard
synchrotron beamline conventions (APS, ESRF, PETRA III).

Reference Component (Diffusion)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``D0_ref``
     - :math:`D_{0,r}`
     - |AA|\ :sup:`2`/s\ :sup:`alpha`
     - 1 -- 1e6
     - Reference diffusion prefactor.  Sets the overall magnitude of the
       reference component's mean-squared displacement.  Larger values
       indicate faster diffusion.
   * - ``alpha_ref``
     - :math:`\alpha_r`
     - --
     - -2 to 2
     - Reference transport exponent.  :math:`\alpha = 0` corresponds to
       normal Brownian diffusion.  Positive values indicate
       super-diffusion (e.g., ballistic motion); negative values indicate
       sub-diffusion (e.g., caging, crowding).  Default: 0.0.
   * - ``D_offset_ref``
     - :math:`D_\text{off,r}`
     - |AA|\ :sup:`2`
     - 0 -- :math:`D_0`
     - Reference diffusion baseline.  A constant contribution to the
       cumulative transport coefficient, representing a time-independent
       offset in the MSD.

Sample Component (Diffusion)
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``D0_sample``
     - :math:`D_{0,s}`
     - |AA|\ :sup:`2`/s\ :sup:`alpha`
     - 1 -- 1e6
     - Sample diffusion prefactor.  Analogous to ``D0_ref`` for the
       sample component.
   * - ``alpha_sample``
     - :math:`\alpha_s`
     - --
     - -2 to 2
     - Sample transport exponent.  Same interpretation as
       ``alpha_ref``.  Default: 0.0.
   * - ``D_offset_sample``
     - :math:`D_\text{off,s}`
     - |AA|\ :sup:`2`
     - 0 -- :math:`D_0`
     - Sample diffusion baseline.

Velocity
--------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``v0``
     - :math:`v_0`
     - |AA|/s\ :sup:`beta`
     - 0 -- 1e6
     - Velocity prefactor.  Controls the magnitude of the relative
       velocity between reference and sample components.  Default: 1e3.
   * - ``beta``
     - :math:`\beta_v`
     - --
     - -2 to 2
     - Velocity exponent.  :math:`\beta_v = 0` corresponds to constant
       velocity; non-zero values describe acceleration or deceleration
       over time.
   * - ``v_offset``
     - :math:`v_\text{off}`
     - |AA|/s
     - -100 to 100
     - Velocity offset.  A constant contribution to the velocity field.
       Can be negative, indicating flow reversal relative to the
       dominant direction.  Default: 0.0.

Fraction Evolution
-------------------

The fraction function :math:`f(t)` describes how the relative scattering
weight of the two components evolves during the measurement:

.. math::

   f(t) = f_0\, \exp(-f_1\, t)\, + f_3

where :math:`f_1` is the exponential decay rate and :math:`f_2` is a
time shift applied to :math:`t`.

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``f0``
     - :math:`f_0`
     - --
     - 0 -- 1
     - Fraction amplitude.  The initial excess weight of one component
       above the baseline :math:`f_3`.
   * - ``f1``
     - :math:`f_1`
     - 1/s
     - > 0
     - Fraction exponential rate.  Controls how quickly the component
       fraction relaxes toward the baseline.
   * - ``f2``
     - :math:`f_2`
     - s
     - --
     - Fraction time shift.  Offsets the origin of the exponential
       decay; useful when the dynamics do not begin at :math:`t=0`.
   * - ``f3``
     - :math:`f_3`
     - --
     - 0 -- 1
     - Fraction baseline.  The long-time asymptotic fraction.

Flow Geometry
-------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``phi0``
     - :math:`\phi_0`
     - degrees
     - -180 to 180
     - Flow angle offset.  The angle between the in-plane flow
       direction and the reference axis of the detector.  The velocity
       phase in the cross-correlation term is proportional to
       :math:`\cos(\phi - \phi_0)`.


Scaling Parameters (Per Angle)
==============================

Each azimuthal angle :math:`\phi_i` carries two additional parameters
that account for angle-dependent optical effects.

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Name
     - Symbol
     - Units
     - Typical Range
     - Meaning
   * - ``contrast``
     - :math:`\beta`
     - --
     - 0 -- 1
     - Speckle visibility (optical coherence factor).  Values near 1
       indicate a highly coherent beam; lower values reflect partial
       coherence, detector integration effects, or multiple-scattering
       contributions.
   * - ``offset``
     - --
     - --
     - ~1
     - Baseline correlation level.  Ideally exactly 1.0 for normalised
       :math:`g_2`, but small deviations arise from background
       subtraction imperfections or detector non-uniformity.


Parameter Count Summary
========================

For an analysis with :math:`N_\phi` azimuthal angles:

* Physics parameters: 14 (shared across all angles).
* Scaling parameters: :math:`2 \times N_\phi`.
* **Total**: :math:`14 + 2 N_\phi`.

For a typical 8-angle dataset this gives :math:`14 + 16 = 30` free
parameters.


Interpreting Fitted Values
==========================

Diffusion prefactors
   Compare :math:`D_{0,r}` and :math:`D_{0,s}` to estimate relative
   mobilities.  If :math:`D_{0,r} \ll D_{0,s}`, the reference is
   nearly static (e.g., a gel network) and the sample is mobile
   (e.g., embedded nanoparticles).

Transport exponents
   :math:`\alpha \approx 0` is normal diffusion.
   :math:`\alpha \approx 1` suggests ballistic motion on the probed
   timescale.  Persistent :math:`\alpha < 0` indicates sub-diffusive
   dynamics such as caging in a glass.

Velocity parameters
   :math:`v_0` gives the magnitude of relative flow.  The
   :math:`\phi`-dependent phase means that fitting multiple angles
   simultaneously greatly constrains :math:`v_0` and :math:`\phi_0`.
   A non-zero ``v_offset`` indicates a constant drift component
   independent of the power-law time dependence.

Fraction evolution
   A decaying :math:`f(t)` (positive :math:`f_0`, positive :math:`f_1`)
   indicates that one component progressively dominates (e.g., sedimentation
   or gelation removing scatterers from the beam volume).

Contrast
   Low contrast (:math:`\beta < 0.1`) may indicate a setup problem
   (poor coherence, wrong slit settings).  If contrast varies strongly
   with :math:`\phi`, check for anisotropic beam profiles or parasitic
   scattering.
