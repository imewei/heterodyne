.. _correlation-functions:

=======================
Correlation Functions
=======================

The heterodyne scattering framework defines a hierarchy of correlation
functions that connect microscopic dynamics to the experimentally measured
intensity correlations. These follow from PNAS 2024 Eqs. 1--3 and SI
Eqs. S-1 through S-20.

First-Order Two-Time Correlation
--------------------------------

The first-order (field) correlation function between times :math:`t_1` and
:math:`t_2` is defined as (PNAS 2024 Eq. 1):

.. math::

   c_1(q, t_1, t_2) =
   \frac{\langle E^*(q, t_1)\, E(q, t_2) \rangle}
        {\bigl[\langle |E(q, t_1)|^2 \rangle\,
                \langle |E(q, t_2)|^2 \rangle\bigr]^{1/2}}

where :math:`E(q, t)` is the scattered electric field at wavevector
:math:`q` and time :math:`t`, and angle brackets denote ensemble averaging.

Second-Order Correlation and the Siegert Relation
-------------------------------------------------

The second-order (intensity) correlation function is (PNAS 2024 Eq. 2):

.. math::

   c_2(q, t_1, t_2) =
   \frac{\langle I(q, t_1)\, I(q, t_2) \rangle}
        {\langle I(q, t_1) \rangle\, \langle I(q, t_2) \rangle}

For Gaussian scattered fields, the Siegert relation connects the two
orders:

.. math::

   c_2(q, t_1, t_2) = 1 + \beta\, |c_1(q, t_1, t_2)|^2

where :math:`\beta` is the optical contrast (speckle contrast), determined
by the coherence properties of the beamline optics. In practice,
:math:`\beta \in (0, 1]`.

One-Time Correlation (Equilibrium)
----------------------------------

At thermal equilibrium, the system is time-translation invariant and the
correlation functions depend only on the lag time
:math:`\tau = t_2 - t_1`:

.. math::

   g_2(q, \tau) = 1 + \beta\, |g_1(q, \tau)|^2

where :math:`g_1(q, \tau) = c_1(q, t, t+\tau)` and
:math:`g_2(q, \tau) = c_2(q, t, t+\tau)` are the standard XPCS
autocorrelation functions. The two-time formulation reduces to the
one-time formulation in this limit.

Factorization of the Field Correlation
--------------------------------------

The first-order correlation factorizes into internal (diffusive) and
external (advective) contributions (PNAS 2024 Eq. 7):

.. math::

   c_1(q, t_1, t_2) = c_{1,\mathrm{in}}(q, t_1, t_2)
                     \cdot c_{1,\mathrm{ex}}(q, t_1, t_2)

This factorization holds when the internal (thermal) fluctuations are
statistically independent of the external (flow-driven) displacement.

Internal Correlation
^^^^^^^^^^^^^^^^^^^^

The internal contribution arises from thermal diffusion and is expressed
through the transport coefficient integral (PNAS 2024 Eq. 8):

.. math::

   c_{1,\mathrm{in}}(q, t_1, t_2) \;=\;
   \exp\!\left(-\frac{q^2}{2}
   \int_{t_1}^{t_2} J(t')\, dt'\right)

where :math:`J(t)` is the transport coefficient. The factor of
:math:`\frac{1}{2}` arises because the integral gives the full position
variance growth, while :math:`c_1` involves the *one-dimensional*
projection along :math:`q`.

.. important::

   The implementation **always computes the integral numerically** via
   ``trapezoid_cumsum``, even when :math:`J(t)` is constant. It never
   substitutes analytical antiderivatives. This ensures correctness for
   the general power-law parameterization
   :math:`J(t) = D_0 t^\alpha + D_\mathrm{offset}`, where no useful
   closed-form antiderivative exists.

External Correlation
^^^^^^^^^^^^^^^^^^^^

The external contribution encodes the deterministic (mean) flow velocity
(PNAS 2024 Eq. 9):

.. math::

   c_{1,\mathrm{ex}}(q, t_1, t_2) \;=\;
   \exp\!\left(i\, q \int_{t_1}^{t_2}
   \langle v(t')\rangle\, dt'\right)

where :math:`\langle v(t)\rangle` is the expected velocity at time
:math:`t`. This term produces a phase shift proportional to the mean
displacement, leading to the oscillatory interference pattern that is the
hallmark of heterodyne detection.

When the field correlation enters the Siegert relation, only the modulus
:math:`|c_1|^2` matters for homodyne detection, eliminating the phase
information. Heterodyne detection preserves the phase through interference
with a reference beam.
