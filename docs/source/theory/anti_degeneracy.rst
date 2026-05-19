.. _theory_anti_degeneracy:

Anti-Degeneracy Defense System
================================

This page describes the mathematical basis and operational details of the four-layer
anti-degeneracy defense system implemented in heterodyne's NLSQ optimizer. The system
is coordinated by
:class:`~heterodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`.

For the architectural decision to implement this system, see :ref:`adr_anti_degeneracy`.
For the per-angle scaling mode decision, see :ref:`adr_per_angle_scaling`.


The Degeneracy Problem
----------------------

The heterodyne two-component correlation model evaluates

.. math::
   :label: c2_model

   c_2(\phi_k, t_1, t_2;\, \theta)
   = c_\mathrm{offset}(\phi_k)
   + \beta(\phi_k)\,
     \exp\!\Bigl(-q^2 \bigl[\mathcal{J}_r(t_1,t_2) + f_s(t_1,t_2)\,\mathcal{J}_s(t_1,t_2)\bigr]\Bigr)
     \cos\!\Bigl(q\cos\phi_k \int_{t_1}^{t_2} v(t')\,dt'\Bigr)

where:

- :math:`\mathcal{J}_r(t_1,t_2) = \int_{t_1}^{t_2} J_r(t')\,dt'` is the reference transport integral,
  :math:`J_r(t) = D_{0,\mathrm{ref}}\,t^{\alpha_\mathrm{ref}} + D_{\mathrm{offset,ref}}`
- :math:`\mathcal{J}_s(t_1,t_2) = \int_{t_1}^{t_2} J_s(t')\,dt'` is the sample transport integral,
  :math:`J_s(t) = D_{0,\mathrm{sample}}\,t^{\alpha_\mathrm{sample}} + D_{\mathrm{offset,sample}}`
- :math:`f_s(t_1,t_2)` is the sample fraction averaged over :math:`[t_1, t_2]`
- :math:`v(t) = v_0\,t^\beta + v_\mathrm{offset}` is the velocity field
- :math:`\beta(\phi_k)` and :math:`c_\mathrm{offset}(\phi_k)` are per-angle scaling parameters

The model has 14 physical parameters and :math:`2n_\phi` per-angle scaling parameters.
If both sets are freely optimized simultaneously, the loss landscape has a **flat direction**:
any multiplicative increase in :math:`D_{0,\mathrm{ref}}` (or :math:`D_{0,\mathrm{sample}}`)
can be exactly compensated by dividing all :math:`\beta(\phi_k)` by the same factor,
leaving all residuals unchanged. The NLSQ optimizer cannot distinguish these solutions.

Beyond the diffusion-contrast flat direction, the 14-parameter space contains additional
known degeneracies:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter pair
     - Type
     - Mechanism
   * - :math:`D_{0,\mathrm{ref}}` / :math:`D_{0,\mathrm{sample}}`
     - Trading
     - Two diffusion prefactors; identical effect for :math:`f_s = 0.5`
   * - :math:`f_0` / :math:`f_3`
     - Trading
     - Fraction amplitude vs. baseline; sum is nearly constant
   * - :math:`v_0` / :math:`v_\mathrm{offset}`
     - Trading
     - Velocity magnitude vs. constant; degenerate at long lag times
   * - :math:`\alpha_\mathrm{ref}` / :math:`D_{0,\mathrm{ref}}`
     - Compensation
     - Exponent changes the lag-time scaling; prefactor rescales the overall magnitude
   * - :math:`\alpha_\mathrm{sample}` / :math:`D_{0,\mathrm{sample}}`
     - Compensation
     - Same as above for the sample component


Layer 1: Fourier / Constant Reparameterization
-----------------------------------------------

**Implementation**: :mod:`heterodyne.optimization.nlsq.fourier_reparam`

The first layer removes the per-angle scaling flat direction by constraining the
:math:`2n_\phi` parameters :math:`\{(\beta(\phi_k), c_\mathrm{offset}(\phi_k))\}` to a
lower-dimensional manifold.

**Fourier mode** (``per_angle_mode = "fourier"`` or ``auto`` with :math:`n_\phi \geq 8`):

The angular contrast and offset are expressed as truncated Fourier series:

.. math::

   \beta(\phi_k) = \sum_{m=0}^{K} a_m \cos(m\phi_k) + \sum_{m=1}^{K} b_m \sin(m\phi_k), \qquad
   c_\mathrm{offset}(\phi_k) = \sum_{m=0}^{K} c_m \cos(m\phi_k) + \sum_{m=1}^{K} d_m \sin(m\phi_k)

With :math:`K=2` (the default), each series has :math:`2K+1 = 5` coefficients, for a
total of 10 parameters replacing :math:`2n_\phi`. For :math:`n_\phi \geq 8`, this is a
reduction; for :math:`n_\phi = 3`, Fourier and individual modes have the same count.

The Fourier coefficients are initialized from the data using a discrete Fourier transform
of the quantile-estimated per-angle contrasts. This provides a physically meaningful
starting point close to the true solution.

.. note::

   The Fourier basis is model-agnostic. In homodyne, the sinc-shear model motivates
   even-harmonic dominance (beam coherence is symmetric). In heterodyne, the cosine
   velocity-phase term also produces even-harmonic dominance in :math:`\beta(\phi)`,
   so the same Fourier basis is appropriate.

**Auto-averaged mode** (``auto`` with :math:`3 \leq n_\phi < 8`):

A single pair :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})` replaces all per-angle
values. The shared contrast and offset are initialized from the median of the
quantile-estimated per-angle values:

.. math::

   \bar{\beta} = \mathrm{median}_k\,[\hat{\beta}(\phi_k)], \qquad
   \bar{c}_\mathrm{offset} = \mathrm{median}_k\,[\hat{c}_\mathrm{offset}(\phi_k)]

This adds only 2 parameters to the 14-parameter physical model.

**Individual mode** (``per_angle_mode = "individual"`` or ``auto`` with :math:`n_\phi < 3`):

All :math:`2n_\phi` per-angle parameters are freely optimized. For small :math:`n_\phi`
(1–2 angles), this is not degenerate because the angular coverage is insufficient to
create a flat direction. For :math:`n_\phi \geq 3`, this mode is high-risk and should
be used only for ablation studies.

**Constant mode** (``per_angle_mode = "constant"``):

Per-angle scaling parameters are fixed at quantile estimates and not optimized. The 14
physical parameters are optimized in isolation. This is useful for:

- Ablation studies (testing whether optimized scaling improves the fit)
- Cases where independent calibration of contrast is available
- Fast approximate fits


Layer 2: Hierarchical Optimization
------------------------------------

**Implementation**: :mod:`heterodyne.optimization.nlsq.hierarchical`

The second layer separates the optimization into two stages to avoid gradient
entanglement between physical parameters and per-angle scaling.

**Stage 1 — Physical-only optimization**:

Fix the per-angle scaling at quantile estimates and optimize only the 14 physical
parameters:

.. math::

   \hat{\theta}_\mathrm{phys} = \arg\min_{\theta_\mathrm{phys}} \sum_{k,i,j}
   \bigl[c_2^\mathrm{data}(\phi_k, t_i, t_j)
   - c_2^\mathrm{model}(\phi_k, t_i, t_j;\, \theta_\mathrm{phys},\, \bar{\beta},\, \bar{c}_\mathrm{offset})\bigr]^2

**Stage 2 — Joint optimization**:

Starting from :math:`(\hat{\theta}_\mathrm{phys}, \bar{\beta}, \bar{c}_\mathrm{offset})`,
optimize all parameters jointly:

.. math::

   (\hat{\theta}_\mathrm{phys}, \hat{\beta}, \hat{c}_\mathrm{offset})
   = \arg\min_{\theta_\mathrm{phys},\, \beta,\, c_\mathrm{offset}} \sum_{k,i,j}
   \bigl[c_2^\mathrm{data} - c_2^\mathrm{model}\bigr]^2

The hierarchical approach ensures that Stage 1 explores the physical parameter space
without interference from scaling, providing a physically meaningful initialization for
Stage 2.

**Activation condition**: Hierarchical optimization is active when
``hierarchical_enable = True`` (default) and the controller is in any mode other than
``fixed_constant``.


Layer 3: Adaptive CV-Based Regularization
------------------------------------------

**Implementation**: :mod:`heterodyne.optimization.nlsq.adaptive_regularization`

The third layer adds a regularization penalty based on the per-angle residual
coefficient of variation (CV):

.. math::

   \mathrm{CV}(\phi_k) = \frac{\sigma_k}{\mu_k}

where :math:`\sigma_k` and :math:`\mu_k` are the standard deviation and mean of the
residuals :math:`r_{ij}(\phi_k) = c_2^\mathrm{data}(\phi_k, t_i, t_j) - c_2^\mathrm{model}(\phi_k, t_i, t_j)`
for all :math:`(t_i, t_j)` pairs at angle :math:`\phi_k`.

A physically correct fit has small, roughly uniform residuals across angles (low CV).
A degenerate fit has large residuals with high angular variance (high CV). The augmented
loss is:

.. math::

   L(\theta) = \sum_{k,i,j} r_{ij}^2(\phi_k)
   + \lambda \sum_k \max\!\bigl(0,\, \mathrm{CV}(\phi_k) - \mathrm{CV}_\mathrm{target}\bigr)^2

The regularization weight :math:`\lambda` is adapted dynamically: it is increased when
the current CV exceeds ``regularization_max_cv``, and decreased when the fit is already
within the target. This prevents the regularization from dominating the physics residual
after convergence.

**Activation condition**: Layer 3 is active when ``regularization_mode != "disabled"``
(default: ``"relative"``).


Layer 4: Gradient Collapse Monitoring
---------------------------------------

**Implementation**: :mod:`heterodyne.optimization.nlsq.gradient_monitor`

The fourth layer monitors the Jacobian during optimization. Gradient collapse — where the
gradient with respect to physical parameters becomes negligible compared to the gradient
with respect to per-angle scaling — is a diagnostic symptom of the degenerate manifold.

The monitored ratio is:

.. math::

   \rho = \frac{\|\nabla_{\theta_\mathrm{phys}} L\|}{\|\nabla_{\beta, c_\mathrm{offset}} L\|}

When :math:`\rho < \rho_\mathrm{threshold}` (default: 0.01) for
``gradient_consecutive_triggers`` consecutive iterations (default: 5), the controller
triggers the configured response action:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Response mode
     - Action
   * - ``"warn"``
     - Log a warning; continue optimization
   * - ``"hierarchical"``
     - Trigger a hierarchical restart (Stage 1 from current physical params)
   * - ``"reset"``
     - Reset per-angle scaling to quantile estimates; restart joint optimization
   * - ``"abort"``
     - Raise an error and abort the optimization

The default response is ``"hierarchical"``.

**Activation condition**: Layer 4 is active when ``gradient_monitoring_enable = True``
(default) and the optimizer is running in any mode other than ``constant``.


Layer 5: Absent — Shear-Sensitivity Weighting
-----------------------------------------------

Homodyne implements a fifth layer that upweights data points near :math:`\phi = 0`
(parallel to flow), where the laminar-flow sinc term

.. math::

   \operatorname{sinc}^2\!\bigl(q\cos\phi_k \cdot \dot{\gamma}_0 \cdot \Delta t / 2\bigr)

is most sensitive to the shear rate :math:`\dot{\gamma}_0`. This breaks the
:math:`(D_0, \dot{\gamma}_0)` degeneracy specific to the homodyne laminar-flow model.

**Heterodyne does not include Layer 5.** The heterodyne velocity-phase model has no sinc
term. The flow contribution enters as:

.. math::

   \cos\!\Bigl(q\cos\phi_k \int_{t_1}^{t_2} v(t')\,dt'\Bigr)

The oscillation amplitude is :math:`q|\cos\phi_k|`, which is naturally largest near
:math:`\phi_k = 0` and vanishes near :math:`\phi_k = \pi/2`. This intrinsic angular
sensitivity provides sufficient differentiation between the diffusion exponential and
the velocity cosine term without requiring explicit upweighting.

Adding Layer 5 to heterodyne would double-weight the :math:`\phi \approx 0` data region,
biasing the fit and increasing the effective noise floor. This is a **narrow physics
exemption** documented in spec §2 and tracked as a D3 divergence from homodyne.

The ``use_shear_weighting`` property of
:class:`~heterodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`
always returns ``False``.


Per-Angle Mode Summary
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 10 10 42

   * - Mode
     - Layer 1
     - Layer 2
     - Layer 3
     - Layer 4
     - Notes
   * - ``constant``
     - Quantile
     - Disabled
     - Optional
     - Optional
     - Scaling fixed; no per-angle optimization
   * - ``auto``
     - Averaged
     - Active
     - Active
     - Active
     - 2 shared scaling params; recommended default
   * - ``fourier``
     - Fourier
     - Active
     - Active
     - Active
     - K=2 Fourier coefficients; best for smooth variation
   * - ``individual``
     - Per-angle
     - Optional
     - Optional
     - Optional
     - :math:`2n_\phi` free params; degeneracy risk for :math:`n_\phi \geq 3`


Configuration Reference
-----------------------

All anti-degeneracy settings are nested under ``optimization.nlsq.anti_degeneracy``:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         enable: true                        # Master switch
         per_angle_mode: "auto"              # "auto" | "constant" | "individual" | "fourier"
         fourier_order: 2                    # K in the Fourier series (2K+1 coefficients)
         fourier_auto_threshold: 8           # n_phi threshold for auto -> fourier
         constant_scaling_threshold: 3       # n_phi threshold for auto -> auto_averaged

         # Layer 2: Hierarchical
         hierarchical_enable: true
         hierarchical_max_outer_iterations: 5
         hierarchical_outer_tolerance: 1.0e-4
         hierarchical_physical_max_iterations: 100
         hierarchical_per_angle_max_iterations: 50

         # Layer 3: Adaptive CV regularization
         regularization_mode: "relative"     # "relative" | "absolute" | "disabled"
         regularization_lambda: 1.0
         regularization_target_cv: 0.10
         regularization_target_contribution: 0.10
         regularization_max_cv: 0.20

         # Layer 4: Gradient monitoring
         gradient_monitoring_enable: true
         gradient_ratio_threshold: 0.01
         gradient_consecutive_triggers: 5
         gradient_response_mode: "hierarchical"  # "warn"|"hierarchical"|"reset"|"abort"


Diagnostic Output
-----------------

The controller logs each layer's activation at ``INFO`` level, framed with separator
lines for visibility:

.. code-block:: text

   ============================================================
   ANTI-DEGENERACY: Auto-selected 'auto_averaged' mode
     Reason: n_phi (7) >= constant_scaling_threshold (3)
     Behavior: Quantile estimates -> AVERAGED -> OPTIMIZED
     Parameters: 14 physical + 2 averaged scaling = 16 total
   ============================================================
   ANTI-DEGENERACY: Layer 4 - Gradient Collapse Monitor
     Enabled: True
     Ratio threshold: 0.01
     Response mode: hierarchical
   ============================================================

Layer 4 triggers are logged at ``WARNING`` level:

.. code-block:: text

   WARNING: Gradient collapse detected (ratio=0.003 < threshold=0.01, 5 consecutive)
   Triggering hierarchical restart from current physical parameters.

Post-fit, any identified degenerate parameter pairs are logged at ``WARNING`` level,
enabling downstream diagnosis.


.. seealso::

   - :ref:`adr_anti_degeneracy` — architectural decision record
   - :ref:`adr_per_angle_scaling` — per-angle scaling mode decision
   - :mod:`heterodyne.optimization.nlsq.anti_degeneracy_controller` — orchestrator
   - :mod:`heterodyne.optimization.nlsq.fourier_reparam` — Layer 1 implementation
   - :mod:`heterodyne.optimization.nlsq.hierarchical` — Layer 2 implementation
   - :mod:`heterodyne.optimization.nlsq.adaptive_regularization` — Layer 3 implementation
   - :mod:`heterodyne.optimization.nlsq.gradient_monitor` — Layer 4 implementation
