.. _adr_anti_degeneracy:

ADR-003: Anti-Degeneracy Layer for Velocity-Phase Model
=========================================================

:Status: Accepted
:Date: 2026-05-19
:Deciders: Core team

Context
-------

The heterodyne two-component model fits the two-time correlation function at multiple
azimuthal angles simultaneously:

.. math::

   c_2(\phi_k, t_1, t_2;\, \theta)
   = c_\mathrm{offset}(\phi_k)
   + \beta(\phi_k)\,
     \exp\!\Bigl(-q^2 \bigl[\mathcal{J}_r(t_1,t_2) + f_s\,\mathcal{J}_s(t_1,t_2)\bigr]\Bigr)
     \cos\!\Bigl(q\cos\phi_k \int_{t_1}^{t_2} v(t')\,dt'\Bigr)

Each angle :math:`\phi_k` has its own contrast :math:`\beta(\phi_k)` and offset
:math:`c_\mathrm{offset}(\phi_k)` arising from detector geometry and beam coherence.

If these angle-specific parameters are treated as free variables in the optimization (one
pair per angle, :math:`2 \times n_\phi` extra free parameters), the NLSQ landscape has a
**degenerate flat direction**: any increase in :math:`D_{0,\mathrm{ref}}` or
:math:`D_{0,\mathrm{sample}}` can be compensated by increasing all :math:`\beta(\phi_k)`
by a corresponding factor, without changing the residual.

This degeneracy causes optimizers to return unphysical solutions with
:math:`D_{0,\mathrm{ref}} \approx 0` and inflated contrasts, or vice versa. It is not a
numerical artifact — it is a genuine identifiability issue rooted in the functional form
of the model.

The 14-parameter heterodyne model has additional degeneracies beyond the
diffusion-contrast flat direction:

- :math:`D_{0,\mathrm{ref}}` / :math:`D_{0,\mathrm{sample}}` trading (two diffusion prefactors)
- :math:`f_0` / :math:`f_3` trading (fraction amplitude vs. baseline)
- :math:`v_0` / :math:`v_\mathrm{offset}` trading (velocity magnitude vs. constant offset)
- :math:`\alpha_\mathrm{ref}` / :math:`D_{0,\mathrm{ref}}` compensation (exponent-prefactor pair)
- :math:`\alpha_\mathrm{sample}` / :math:`D_{0,\mathrm{sample}}` compensation


Decision
--------

Heterodyne implements a **four-layer anti-degeneracy defense system** coordinated by
:class:`~heterodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`:

**Layer 1: Fourier/Constant Reparameterization**
   Transforms the parameter space to remove the flat direction. In ``auto`` mode with
   sufficient angles: the per-angle contrasts and offsets are expressed as a truncated
   Fourier series over :math:`\phi`, replacing :math:`2n_\phi` independent values with
   :math:`2K+1` Fourier coefficients (default :math:`K=2`). With fewer angles, a single
   averaged pair :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})` is used, initialized from
   quantile estimates and jointly optimized with the 14 physical parameters.

**Layer 2: Hierarchical Optimization**
   First stage: fix scaling parameters at quantile estimates, optimize only the 14 physical
   parameters. Second stage: jointly optimize all parameters from the Layer-1 solution.
   This avoids the flat landscape during the physically important first phase.

**Layer 3: Adaptive CV-based Regularization**
   Penalizes solutions where the per-angle residual coefficient of variation (CV) is
   anomalously large, which signals that the optimizer is "fitting noise" in the
   degeneracy direction rather than the physics signal.

**Layer 4: Gradient Collapse Monitoring**
   Tracks the ratio :math:`\|\nabla_\mathrm{physical}\| / \|\nabla_\mathrm{per\text{-}angle}\|`
   during optimization. When this ratio falls below a threshold for several consecutive
   iterations (indicating gradient collapse), triggers a hierarchical restart.

**Layer 5: Shear-Sensitivity Weighting — INTENTIONALLY ABSENT**
   Homodyne implements a fifth layer that upweights data points from angles near
   :math:`\phi = 0` (parallel to flow), where the :math:`\operatorname{sinc}` shear
   term is most sensitive to :math:`\dot{\gamma}_0`. This breaks the
   :math:`(D_0, \dot{\gamma}_0)` degeneracy specific to laminar-flow XPCS.

   Heterodyne does **not** include Layer 5. The heterodyne velocity-phase model has no
   :math:`\operatorname{sinc}` term: the flow contribution enters as
   :math:`\cos(q\cos\phi_k \int v\,dt')`, not as a sinc attenuation. The
   :math:`(D_0, v_0)` degeneracy is broken by the oscillatory angular dependence of the
   cosine term across multiple :math:`\phi_k` values, without requiring explicit
   shear-sensitivity upweighting. This is a **narrow physics exemption** (spec §2).

The four layers are controlled by a single YAML setting:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # "auto" activates all 4 layers

See :ref:`theory_anti_degeneracy` for the mathematical details of each mode.


Rationale
---------

**Why reparameterization (Layer 1)?**

Direct optimization of :math:`2n_\phi` per-angle parameters creates a flat manifold in
the loss landscape. Reparameterizing as Fourier coefficients exploits the physical prior
that angular contrast variation is smooth (it arises from detector geometry), reducing
the effective parameter count and eliminating the flat direction. The Fourier basis is
model-agnostic: it captures smooth :math:`\phi`-variation of any origin.

**Why hierarchical (Layer 2)?**

In the joint landscape, gradient descent for the 14 physical parameters is dominated
by the per-angle scaling gradient. Fixing the scaling in the first stage forces the
optimizer to explore the physically meaningful sub-space first, then fine-tune scaling
in the second stage from a physically reasonable starting point.

**Why CV regularization (Layer 3)?**

A degenerate solution has large per-angle residuals with high variance across angles
(high CV). A physical solution has small, roughly uniform residuals (low CV). The CV
penalty provides a soft constraint that guides the optimizer away from the degenerate
manifold without requiring hard constraints that might exclude physical solutions.

**Why gradient monitoring (Layer 4)?**

Gradient collapse (ratio < threshold for N consecutive iterations) is a reliable
symptom of landing on the degenerate manifold. The response (hierarchical restart) is
cheaper than abandoning the optimization and re-running from scratch.

**Why not Layer 5 in heterodyne?**

The shear-sensitivity weighting in homodyne upweights the sinc-dominated regime to
break the :math:`(D_0, \dot{\gamma}_0)` degeneracy that is specific to laminar-flow
XPCS. Heterodyne's velocity-phase cosine term provides a different angular sensitivity
structure: the amplitude of the cosine oscillation is :math:`q|\cos\phi_k|`, which is
naturally largest near :math:`\phi_k = 0` and vanishes near :math:`\phi_k = \pi/2`.
This intrinsic angular sensitivity breaks the diffusion-velocity degeneracy without
explicit upweighting. Adding Layer 5 would double-weight the :math:`\phi \approx 0`
region, biasing the residuals and increasing the effective noise floor.


Consequences
------------

**Positive**:

- No degenerate solutions in production use of ``auto`` mode.
- Clear diagnostic logging when each layer activates (visible at ``INFO`` level).
- All four layers are independently tested and can be individually disabled via config.
- ``constant`` mode allows ablation studies (disabling the optimization of scaling).

**Negative / Accepted trade-offs**:

- ``auto`` mode adds 2 parameters (16 vs 14 physical-only) and 2 optimization stages.
- The 4-layer system adds code complexity: 4 modules, each with its own config class.
- New contributors must read this ADR and :ref:`theory_anti_degeneracy` to understand
  why the optimizer is structured this way.
- The deliberate absence of Layer 5 must be documented to prevent well-intentioned
  re-addition by future contributors unfamiliar with homodyne's design history.

**Operational advice**:

- Always use ``auto`` mode unless there is a specific reason to use another.
- If fitted :math:`D_{0,\mathrm{ref}} \approx 0` or :math:`D_{0,\mathrm{sample}} \approx 0`
  and contrasts are large: degeneracy was not fully resolved; switch to ``auto`` if not
  already, or verify initial parameter guesses.
- Individual degenerate parameter pairs are logged at ``WARNING`` level during
  post-fit diagnosis.


Alternatives Considered
-----------------------

**A. Single-mode only (auto)**

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

**D. Port homodyne Layer 5 without modification**

Port shear-sensitivity weighting directly to heterodyne. Rejected because: heterodyne
has no sinc term; the weighting would apply to the cosine oscillation amplitude, which
already provides natural angular sensitivity. Explicit upweighting of :math:`\phi \approx 0`
would bias the fit toward data from that angular region and increase effective noise.


.. seealso::

   - :ref:`theory_anti_degeneracy` — mathematical details of all modes
   - :ref:`adr_per_angle_scaling` — per-angle scaling mode decision
   - :mod:`heterodyne.optimization.nlsq.anti_degeneracy_controller` — implementation
   - :mod:`heterodyne.optimization.nlsq.fourier_reparam` — Fourier mode
