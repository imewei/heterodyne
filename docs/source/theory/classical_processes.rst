.. _classical-processes:

==============================
Classical Stochastic Processes
==============================

The generalized transport coefficient framework subsumes several classical
stochastic models as special cases. This section catalogs these limits,
following PNAS 2024 SI Section 4 (Eqs. S-108 through S-127), and shows
how each maps onto the power-law parameterization used by the heterodyne
package.

Wiener Process (Standard Diffusion)
------------------------------------

The Wiener process describes free Brownian motion with a constant
diffusion coefficient :math:`D`. The transport coefficient is:

.. math::

   J(t) \;=\; 2D

and the internal field correlation is:

.. math::

   c_{1,\mathrm{in}}(q, t_1, t_2) \;=\;
   \exp\!\left(-\frac{q^2}{2} \int_{t_1}^{t_2} J(t')\, dt'\right)

This is the simplest case: the position variance grows linearly in time,
:math:`\mathrm{Var}\!\left[x(t)\right] = \int_0^t J(t')\, dt'`, and the
correlation decays monotonically with the integrated transport.

**Power-law mapping**: :math:`\alpha = 0`, :math:`D_0 = 2D`,
:math:`D_\mathrm{offset} = 0`.

Ornstein-Uhlenbeck Process
---------------------------

The Ornstein-Uhlenbeck (OU) process models diffusion in a harmonic
potential with restoring rate :math:`\gamma`:

.. math::

   J(t) \;=\; 2D\left(1 - e^{-\gamma t}\right)^2

At short times (:math:`t \ll 1/\gamma`), the particle is ballistic and
:math:`J(t) \approx 2D\gamma^2 t^2`. At long times
(:math:`t \gg 1/\gamma`), the velocity decorrelates and
:math:`J(t) \to 2D`, recovering Fickian diffusion.

The OU process is not exactly representable by the power-law
parameterization, but for time windows much longer than :math:`1/\gamma`
it is well approximated by :math:`\alpha = 0`.

Brownian Oscillator
-------------------

The Brownian oscillator describes a particle in a harmonic potential with
natural frequency :math:`\omega_0` and damping rate :math:`\gamma`. It
exhibits two regimes:

**Overdamped** (:math:`\gamma > 2\omega_0`):

.. math::

   J(t) = 2D \left(1 - e^{-\gamma t/2}
   \left[\cosh(\Omega t)
   + \frac{\gamma}{2\Omega}\sinh(\Omega t)\right]\right)^2

where :math:`\Omega = \sqrt{(\gamma/2)^2 - \omega_0^2}`.

**Underdamped** (:math:`\gamma < 2\omega_0`):

.. math::

   J(t) = 2D \left(1 - e^{-\gamma t/2}
   \left[\cos(\omega t)
   + \frac{\gamma}{2\omega}\sin(\omega t)\right]\right)^2

where :math:`\omega = \sqrt{\omega_0^2 - (\gamma/2)^2}`.

In both cases, :math:`J(t) \to 2D` at long times. The underdamped case
exhibits oscillatory transients in :math:`J(t)` that can produce
non-monotonic correlation decay at short lag times.

Advection-Diffusion
--------------------

The advection-diffusion model combines Brownian diffusion with a constant
drift velocity :math:`v_0`. Using the internal/external factorization:

.. math::

   c_1(q, t_1, t_2) \;=\;
   \exp\!\left(-\frac{q^2}{2} \int_{t_1}^{t_2} J(t')\, dt'\right)\;
   \exp\!\left(i\, q \int_{t_1}^{t_2} v(t')\, dt'\right)

The first factor is the diffusive decay (transport integral); the second is
the advective phase shift (velocity integral). In a homodyne measurement,
the phase vanishes under the modulus squared:

.. math::

   g_2(q, \tau) \;=\; 1 + \beta\,
   \exp\!\left(-q^2 \int_0^{\tau} J(t')\, dt'\right)

so the velocity is invisible. In a heterodyne measurement, the velocity
produces observable oscillations in the cross-correlation (see
:ref:`heterodyne-scattering`).

**Power-law mapping**: :math:`\alpha = 0`, :math:`D_0 = 2D`,
:math:`D_\mathrm{offset} = 0`, with velocity :math:`v(t) = v_0` (i.e.,
:math:`\beta_v = 0`, :math:`v_\mathrm{offset} = v_0`).

Power-Law Transport
-------------------

The general power-law parameterization used in the heterodyne package
(PNAS 2024 SI Eq. S-105):

.. math::

   J(t) \;=\; D_0\, t^\alpha + D_\mathrm{offset}

This phenomenological form captures the leading-order behavior of a wide
class of transport processes:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameters
     - Model
     - Physical scenario
   * - :math:`\alpha = 0,\; D_\mathrm{offset} = 0`
     - Wiener
     - Free diffusion
   * - :math:`\alpha < 0`
     - Subdiffusive
     - Viscoelastic confinement, crowded environments
   * - :math:`0 < \alpha < 1`
     - Weakly superdiffusive
     - Persistent random walks
   * - :math:`\alpha = 1`
     - Ballistic
     - Uniform acceleration
   * - :math:`D_\mathrm{offset} \neq 0`
     - Mixed
     - Baseline diffusion with time-dependent correction

The offset :math:`D_\mathrm{offset}` allows the model to accommodate
processes that have a finite transport rate at :math:`t = 0` (e.g., a
Wiener component superimposed on a power-law anomalous process).

Summary of Limits
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 25 20

   * - Process
     - :math:`J(t)`
     - Short-time
     - Long-time
   * - Wiener
     - :math:`2D`
     - :math:`2D`
     - :math:`2D`
   * - OU
     - :math:`2D(1-e^{-\gamma t})^2`
     - :math:`\sim t^2`
     - :math:`2D`
   * - Oscillator
     - (see above)
     - :math:`\sim t^2`
     - :math:`2D`
   * - Advection-diffusion
     - :math:`2D`
     - :math:`2D`
     - :math:`2D`
   * - Power-law
     - :math:`D_0 t^\alpha + D_\mathrm{offset}`
     - :math:`D_\mathrm{offset}`
     - :math:`\sim t^\alpha`
