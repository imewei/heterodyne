.. _heterodyne-scattering:

===========================
Heterodyne Scattering Model
===========================

This section presents the core theoretical model implemented by the
heterodyne package: a two-component interference scattering model derived
from PNAS 2025 SI Section F (Eqs. S-77 through S-98). The heterodyne
geometry --- where scattered light from a moving sample interferes with
that from a static reference --- produces characteristic oscillations in
the cross-correlation whose frequency directly encodes the sample velocity.

Multi-Component Scattered Field
-------------------------------

For a system of :math:`N` scattering components, the total scattered field
at wavevector :math:`q` and time :math:`t` is (Eq. S-77):

.. math::

   E(q, t) = \sum_{n=1}^{N} x_n(t)\, E_n(q, t)

where :math:`x_n(t)` is the field amplitude fraction of the
:math:`n`-th component and :math:`E_n(q, t)` is its scattered field.
Each component has its own transport coefficient :math:`J_n(t)` and
mean velocity :math:`\langle v_n(t)\rangle`.

Two Key Assumptions
^^^^^^^^^^^^^^^^^^^

The derivation proceeds under two physical assumptions (Eq. S-84):

1. **Uniform scattering contrast**: all components scatter with the same
   contrast factor, so that intensity fractions are determined solely by
   composition :math:`x_n(t)`.

2. **No cross-composition spatial correlation**: the positions of
   particles in different components are statistically independent, so
   cross-component field correlations vanish.

Under these assumptions, the multi-component intensity correlation takes a
compact form.

General N-Component Correlation
-------------------------------

The second-order two-time correlation for an :math:`N`-component system is
(Eq. S-94):

.. math::

   c_2(q, t_1, t_2) \;=\; 1 + \frac{\beta}{f^2}
   \sum_{n=1}^{N} \sum_{m=1}^{N}
   x_n(t_1)\, x_n(t_2)\, x_m(t_1)\, x_m(t_2)
   \;\cdot\; A_{nm}(t_1, t_2)

where the cross-correlation amplitude :math:`A_{nm}` is:

.. math::

   A_{nm}(t_1, t_2)
   \;=\; \exp\!\left(-\frac{q^2}{2}\int_{t_1}^{t_2}
     \left[J_n(t') + J_m(t')\right] dt'\right)
   \cos\!\left(q\cos(\varphi)\int_{t_1}^{t_2}
     \left[\langle v_n(t')\rangle
     - \langle v_m(t')\rangle\right] dt'\right)

where :math:`\beta` is the optical contrast, :math:`f` is a normalization
factor, and :math:`\varphi_n` is the angle between the velocity of
component :math:`n` and the scattering vector :math:`q`.

.. note::

   The cosine term arises from the real part of the product
   :math:`c_1^{\mathrm{(ex)},n} \times c_1^{\mathrm{(ex)},m*}`, which
   reduces the complex exponentials to a single cosine of the velocity
   difference. Same-component terms (:math:`n = m`) have zero velocity
   difference, so their cosine factor is unity.

Two-Component Simplification
-----------------------------

The heterodyne package implements the :math:`N = 2` specialization with a
**reference** component (:math:`r`) and a **sample** component (:math:`s`):

- **Reference** (:math:`r`): static scatterer with transport
  :math:`J_r(t)` and zero mean velocity
- **Sample** (:math:`s`): moving scatterer with transport :math:`J_s(t)`,
  mean velocity :math:`\langle v(t)\rangle`, and flow angle
  :math:`\varphi`

The sample fraction is :math:`x_s(t) \in [0, 1]` and the reference
fraction is :math:`x_r(t) = 1 - x_s(t)`.

Two-Time Form (Eq. S-95)
^^^^^^^^^^^^^^^^^^^^^^^^^

The full two-time correlation is:

.. math::

   c_2(q, t_1, t_2) \;=\; 1 + \frac{\beta}{f^2}
   \left[
     \bigl[x_r(t_1)\, x_r(t_2)\bigr]^2\, A_{rr}
     \;+\; \bigl[x_s(t_1)\, x_s(t_2)\bigr]^2\, A_{ss}
     \;+\; 2\, x_r(t_1)\, x_r(t_2)\, x_s(t_1)\, x_s(t_2)\, A_{rs}
   \right]

with:

.. math::

   A_{rr} &= \exp\!\left(-q^2 \int_{t_1}^{t_2} J_r(t')\, dt'\right), \\
   A_{ss} &= \exp\!\left(-q^2 \int_{t_1}^{t_2} J_s(t')\, dt'\right), \\
   A_{rs} &= \exp\!\left(-\frac{q^2}{2}
              \int_{t_1}^{t_2}\left[J_r(t') + J_s(t')\right] dt'\right)
              \cos\!\left(q\cos(\varphi)
              \int_{t_1}^{t_2}\langle v(t')\rangle\, dt'\right)

This expression contains three distinct physical contributions, described
below.

Three-Term Structure
^^^^^^^^^^^^^^^^^^^^

**Reference self-correlation** (first term):

.. math::

   C_\mathrm{ref} \;=\; \bigl[x_r(t_1)\, x_r(t_2)\bigr]^2\;
   \exp\!\left(-q^2 \int_{t_1}^{t_2} J_r(t')\, dt'\right)

Describes the decorrelation of the static reference scattering due to its
own internal dynamics (thermal diffusion). This term decays monotonically
with lag time.

**Sample self-correlation** (second term):

.. math::

   C_\mathrm{sample} \;=\; \bigl[x_s(t_1)\, x_s(t_2)\bigr]^2\;
   \exp\!\left(-q^2 \int_{t_1}^{t_2} J_s(t')\, dt'\right)

Describes the decorrelation of the sample scattering. Like the reference
term, this decays monotonically but typically faster due to flow-enhanced
transport.

**Cross-correlation** (third term):

.. math::

   C_\mathrm{cross} \;=\; 2\, x_r(t_1)\, x_r(t_2)\, x_s(t_1)\, x_s(t_2)\;
   \exp\!\left(-\frac{q^2}{2} \int_{t_1}^{t_2}
   \left[J_s(t') + J_r(t')\right] dt'\right)\;
   \cos\!\left(q \cos(\varphi) \int_{t_1}^{t_2}
   \langle v(t')\rangle\, dt'\right)

This is the signature heterodyne term. The cosine factor produces
oscillations whose frequency is proportional to
:math:`q \cos(\varphi) \cdot \langle v \rangle` --- the projection of the
sample velocity onto the scattering vector. The oscillation amplitude is
modulated by the geometric mean of the transport decays from both
components, and is maximized when the reference and sample fractions are
balanced (:math:`x_s \approx 0.5`).

Equilibrium One-Time Form (Eq. S-98)
-------------------------------------

At equilibrium, where the composition fractions, transport coefficients,
and velocity are all time-independent, the two-time correlation reduces to
a function of lag time :math:`\tau = t_2 - t_1` only. Denoting the
equilibrium sample fraction :math:`x \equiv I_s / (I_s + I_r)`:

.. math::

   g_2(q, \tau) \;=\; 1 + \beta
   \left[
     (1 - x)^2\,
       \exp\!\left(-q^2 \int_0^{\tau} J_r(t')\, dt'\right)
     + x^2\,
       \exp\!\left(-q^2 \int_0^{\tau} J_s(t')\, dt'\right)
     + 2\, x(1 - x)\,
       \exp\!\left(-\frac{q^2}{2}
         \int_0^{\tau} \left[J_r(t') + J_s(t')\right] dt'\right)
       \cos\!\left(q \cos(\varphi)
         \int_0^{\tau} \langle v(t')\rangle\, dt'\right)
   \right]

This is the one-time specialization of Eq. S-95, obtained by setting
:math:`t_1 = 0` and :math:`t_2 = \tau` with time-independent fractions.

.. important::

   The implementation **always evaluates the integrals numerically** via
   ``trapezoid_cumsum``, even in equilibrium. It never substitutes
   analytical antiderivatives (e.g., :math:`\int J\, dt = 2D\tau` for
   constant :math:`J`). This avoids silent approximation errors for the
   general power-law parameterization :math:`J(t) = D_0 t^\alpha +
   D_\mathrm{offset}`, which has no useful closed-form antiderivative.

Normalization
-------------

The normalization factor :math:`f^2` in the correlation expression ensures
that :math:`c_2(q, t, t) = 1 + \beta` on the diagonal. For the
two-component system:

.. math::

   f^2(t_1, t_2) \;=\; \left[x_s^2(t_1) + x_r^2(t_1)\right]
                  \cdot \left[x_s^2(t_2) + x_r^2(t_2)\right]

This normalization accounts for the fact that the total scattered
intensity is not simply the sum of individual intensities when the
component fractions are time-dependent.

Angle Dependence
----------------

The flow angle :math:`\varphi` controls the projection of velocity onto
the scattering direction:

- :math:`\varphi = 0`: maximum velocity sensitivity
  (:math:`\cos(\varphi) = 1`)
- :math:`\varphi = 90^\circ`: zero velocity sensitivity
  (:math:`\cos(\varphi) = 0`), reducing to a purely diffusive model

By measuring at multiple detector angles :math:`\varphi`, the full
velocity vector can be reconstructed. The implementation supports
simultaneous multi-angle fitting where the 14 physics parameters are
shared across angles while per-angle contrast and offset are independently
varied.
