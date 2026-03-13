.. _analysis-modes:

===============
Analysis Modes
===============

The heterodyne package implements a 14-parameter two-component correlation
model with 2 additional per-angle scaling parameters. This section defines
each parameter, its physical role, and contrasts the heterodyne analysis
mode with the simpler homodyne alternatives.

Heterodyne Model Parameters
----------------------------

The 14 physics parameters are organized into five groups:

Reference Transport
^^^^^^^^^^^^^^^^^^^

Transport rate for the static reference component:
:math:`J_r(t) = D_{0,r}\, t^{\alpha_r} + D_{\mathrm{offset},r}`

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``D0_ref``
     - :math:`\text{\AA}^2/\text{s}^\alpha`
     - :math:`10^4`
     - Reference diffusion prefactor
   * - ``alpha_ref``
     - ---
     - 0.0
     - Reference transport exponent
   * - ``D_offset_ref``
     - :math:`\text{\AA}^2/\text{s}`
     - 0.0
     - Reference transport rate offset

Sample Transport
^^^^^^^^^^^^^^^^

Transport rate for the moving sample component:
:math:`J_s(t) = D_{0,s}\, t^{\alpha_s} + D_{\mathrm{offset},s}`

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``D0_sample``
     - :math:`\text{\AA}^2/\text{s}^\alpha`
     - :math:`10^4`
     - Sample diffusion prefactor
   * - ``alpha_sample``
     - ---
     - 0.0
     - Sample transport exponent
   * - ``D_offset_sample``
     - :math:`\text{\AA}^2/\text{s}`
     - 0.0
     - Sample transport rate offset

Velocity
^^^^^^^^

Velocity rate for the sample component:
:math:`v(t) = v_0\, t^\beta + v_\mathrm{offset}`

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``v0``
     - :math:`\text{\AA}/\text{s}^\beta`
     - :math:`10^3`
     - Velocity prefactor
   * - ``beta``
     - ---
     - 0.0
     - Velocity exponent (0 = constant velocity)
   * - ``v_offset``
     - :math:`\text{\AA/s}`
     - 0.0
     - Velocity offset (negative allowed for reversal)

Sample Fraction
^^^^^^^^^^^^^^^

Time-dependent sample fraction:

.. math::

   f_s(t) = \operatorname{clip}\!\bigl(
   f_0 \, \exp\!\bigl(f_1 (t - f_2)\bigr) + f_3,\; 0,\; 1\bigr)

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``f0``
     - ---
     - 0.5
     - Fraction amplitude
   * - ``f1``
     - :math:`\text{s}^{-1}`
     - 0.0
     - Exponential rate (0 = constant fraction)
   * - ``f2``
     - s
     - 0.0
     - Time shift
   * - ``f3``
     - ---
     - 0.0
     - Baseline offset

When :math:`f_1 = 0`, the fraction is constant:
:math:`f_s = \operatorname{clip}(f_0 + f_3, 0, 1)`.

Flow Angle
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``phi0``
     - degrees
     - 0.0
     - Flow angle offset relative to :math:`q`-vector

The total flow angle is :math:`\varphi = \varphi_\mathrm{detector} + \varphi_0`,
where :math:`\varphi_\mathrm{detector}` is the known detector geometry and
:math:`\varphi_0` is a fitted correction.

Per-Angle Scaling Parameters
-----------------------------

Two scaling parameters are fit independently for each detector angle:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Unit
     - Default
     - Description
   * - ``contrast``
     - ---
     - 0.5
     - Optical contrast :math:`\beta` (speckle contrast)
   * - ``offset``
     - ---
     - 1.0
     - Baseline offset

The full model is:

.. math::

   c_2^\mathrm{model} \;=\; \mathrm{offset} + \mathrm{contrast} \times
   \frac{C_\mathrm{ref} + C_\mathrm{sample} + C_\mathrm{cross}}{f^2}

Rate Functions
--------------

The two rate functions entering the correlation model are:

**Transport rate** (per component):

.. math::

   J(t) \;=\; D_0\, t^\alpha + D_\mathrm{offset}

This is evaluated by ``compute_transport_rate`` and floored at zero to
ensure non-negative transport.

**Velocity rate**:

.. math::

   v(t) \;=\; v_0\, t^\beta + v_\mathrm{offset}

This is evaluated by ``compute_velocity_rate`` and is *not* floored,
because the velocity integral enters a cosine and is naturally bounded.

The velocity enters the phase factor as:

.. math::

   \mathrm{phase}(t_1, t_2) \;=\; q \cos(\varphi) \int_{t_1}^{t_2} v(t')\, dt'

Both integrals are always evaluated numerically via ``trapezoid_cumsum``.
No analytical antiderivatives are used.

Comparison with Homodyne Modes
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Mode
     - Parameters
     - Description
   * - Homodyne static
     - 3
     - Single component: :math:`g_2 = 1 + \beta \exp(-2q^2 \int J\, dt)`.
       Parameters: :math:`D_0`, :math:`\alpha`, :math:`D_\mathrm{offset}`.
   * - Homodyne laminar flow
     - 7
     - Single component with shear: adds a :math:`\mathrm{sinc}^2` shear
       broadening term and 4 flow parameters.
   * - **Heterodyne**
     - **14 + 2/angle**
     - Two components (reference + sample) with relative velocity. The
       cross-correlation oscillation encodes velocity information that is
       invisible in homodyne detection.

The fundamental advantage of heterodyne analysis is access to the **phase**
of the field correlation through interference with a reference. Homodyne
detection measures only :math:`|c_1|^2`, losing all velocity phase
information. The cost is a larger parameter space (14 vs. 3--7) and the
requirement that a coherent reference scattering component be present in
the measurement geometry.
