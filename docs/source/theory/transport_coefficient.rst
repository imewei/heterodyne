.. _transport-coefficient:

==========================
Transport Coefficient J(t)
==========================

The transport coefficient :math:`J(t)` is the central quantity connecting
microscopic particle dynamics to measurable scattering correlations. It
appears as the time derivative of the position variance and encodes the
full history of velocity correlations.

Definition
----------

Following PNAS 2024 SI Eq. S-38, the transport coefficient is defined as
the time derivative of the position variance:

.. math::

   J(t) \;=\; \frac{d}{dt}\,\mathrm{Var}\!\left[x(t)\right]
   \;=\; 2\,\mathrm{Cov}\!\left[x(t),\, v(t)\right]

where :math:`x(t)` is particle displacement and :math:`v(t)` is the
instantaneous velocity. The factor of 2 arises from the chain rule applied
to the variance of a Gaussian process.

Green-Kubo Formula
------------------

The transport coefficient admits a Green-Kubo [GreenKubo]_ integral
representation (PNAS 2024 SI Eq. S-38):

.. math::

   J(t) \;=\; 2 \int_0^t \mathrm{Cov}\!\left[v(t),\, v(t')\right] dt'

This connects :math:`J(t)` to the velocity autocorrelation function. When
the velocity process is stationary, the integrand depends only on
:math:`|t - t'|`, recovering the classical Green-Kubo relation for the
diffusion coefficient.

Physical Meaning
----------------

The transport coefficient has a direct physical interpretation:

- **Position variance growth**: :math:`\mathrm{Var}\!\left[x(t)\right] = \int_0^t J(t')\, dt'`
- **Instantaneous diffusivity**: :math:`J(t)/2` gives the time-dependent diffusion coefficient
- **Long-time limit**: for an equilibrium Wiener process, :math:`J(t) \to 2D` as :math:`t \to \infty`

The distinction between :math:`J(t)` and the standard diffusion coefficient
:math:`D` is crucial for nonequilibrium systems where the transport
properties evolve in time.

Power-Law Parameterization
--------------------------

For practical fitting, the transport coefficient is parameterized as a
power law with offset (PNAS 2024 SI Eq. S-105):

.. math::

   J(t) \;=\; D_0 \, t^\alpha + D_\mathrm{offset}

where:

- :math:`D_0` is the transport prefactor in :math:`\text{\AA}^2/\text{s}^\alpha`
- :math:`\alpha` is the transport exponent (dimensionless)
- :math:`D_\mathrm{offset}` is a constant rate offset in :math:`\text{\AA}^2/\text{s}`

The exponent :math:`\alpha` classifies the transport regime:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Exponent
     - Regime
     - Physical example
   * - :math:`\alpha = 0`
     - Diffusive (Wiener)
     - Free Brownian motion, :math:`J = D_0 + D_\mathrm{offset}`
   * - :math:`\alpha < 0`
     - Subdiffusive
     - Confined motion, viscoelastic media
   * - :math:`\alpha > 0`
     - Superdiffusive
     - Active transport, persistent motion
   * - :math:`\alpha = 1`
     - Ballistic
     - Constant acceleration

Connection to Diffusion
------------------------

Classical diffusion models are special cases of the :math:`J(t)` formalism:

**Wiener process** (standard Brownian motion):

.. math::

   J(t) \;=\; 2D \qquad \Longrightarrow \qquad
   \mathrm{Var}\!\left[x(t)\right] = \int_0^t J(t')\, dt'

For constant :math:`J`, the variance grows linearly, but the
implementation always computes this integral numerically.

**Ornstein-Uhlenbeck process** (diffusion with restoring force):

.. math::

   J(t) \;=\; 2D\left(1 - e^{-\gamma t}\right)^2

where :math:`\gamma` is the relaxation rate. At equilibrium
(:math:`t \gg 1/\gamma`), :math:`J(t) \to 2D`, recovering Fickian
diffusion.

Connection to Macroscopic Rheology
----------------------------------

The transport coefficient bridges microscopic scattering measurements to
macroscopic material properties. From PNAS 2024 SI Eq. S-134:

.. math::

   J(t) \;\approx\; \frac{k_B T}{\pi r} \left|\dot{\gamma}(t)\right|

where :math:`k_B T` is the thermal energy, :math:`r` is the particle
radius, and :math:`\dot{\gamma}(t)` is the local shear rate. This
generalized Stokes-Einstein relation connects the measured :math:`J(t)`
to the time-dependent rheological response of the material under flow.
