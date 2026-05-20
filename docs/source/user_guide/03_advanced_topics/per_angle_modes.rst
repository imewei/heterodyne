.. _per-angle-modes:

===================
Per-Angle Scaling
===================

Each azimuthal angle :math:`\phi_i` in a heterodyne XPCS measurement
has its own speckle contrast and baseline offset.  This page explains
how the package handles these per-angle parameters and how Fourier
reparameterisation reduces the parameter count for joint multi-angle
fits.


Per-Angle Parameters
--------------------

For :math:`N_\phi` azimuthal angles the model has:

* **14 shared physics parameters** -- identical across all angles.
* **2 scaling parameters per angle**:
  ``contrast_i`` (:math:`\beta_i`) and ``offset_i``.

Total free parameters: :math:`14 + 2 N_\phi`.

For a typical 8-angle dataset: :math:`14 + 16 = 30`.


Per-Angle Modes
---------------

Heterodyne's NLSQ optimizer supports four per-angle scaling modes, with
identical semantics to the homodyne reference at
`homodyne anti-degeneracy theory
<https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html>`_.
For :math:`n_\phi` angles and Fourier truncation order :math:`K`:

.. list-table::
   :header-rows: 1
   :widths: 16 24 22 38

   * - Mode
     - Optimizer params
     - β(φ), o(φ) status
     - Behaviour
   * - ``"constant"``
     - 14 (physics only)
     - Frozen per angle
     - β,o pre-estimated by quantile and held fixed in the residual.
       The optimizer never sees β, o as free variables.
   * - ``"auto"`` (default)
     - 16 = 14 + 2
     - Averaged, then optimized
     - Per-angle quantile estimates averaged to one β̄, ō and
       optimized jointly with physics.  Selected automatically when
       ``n_phi >= constant_scaling_threshold`` (default 3).
   * - ``"fourier"`` (K=2)
     - :math:`14 + 2(2K+1)` = 24
     - Optimized as Fourier coeffs
     - β(φ), o(φ) modeled as truncated Fourier series in φ.
       The :math:`2K+1` coefficients per scalar are optimized jointly.
   * - ``"individual"``
     - :math:`14 + 2 n_\phi`
     - Per-angle, fully free
     - Each angle gets its own free β_k, o_k.  Use with caution:
       prone to scaling absorption for many angles.

.. note::

   Heterodyne uses 14 physics parameters (homodyne uses 7) — the
   per-angle parameter counts above add the same per-angle blocks on
   top of the larger physics block.

.. deprecated:: 0.7
   ``per_angle_mode="independent"`` is a deprecation alias for
   ``"individual"`` (homodyne's canonical name).  It will be removed
   in heterodyne v1.0.

Layer 5 (shear weighting) — intentionally excluded
--------------------------------------------------

Homodyne's anti-degeneracy theory defines a fifth defense layer that
re-weights the residual by the shear-sensitivity sinc term in the g₂
formula.  Heterodyne uses a velocity-phase physics model: the g₂
expression contains no shear sinc term and there is nothing for L5 to
re-weight.  The exclusion is enforced at the code level —
:class:`~heterodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`
has no ``shear_weighter`` field, and
:meth:`~heterodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.use_shear_weighting`
always returns ``False``.


Fourier Reparameterisation
---------------------------

When the per-angle contrast and offset vary smoothly with :math:`\phi`,
they can be represented as truncated Fourier series:

.. math::

   \beta(\phi) = a_0 + \sum_{k=1}^{K}
     \bigl[ a_k \cos(k\phi) + b_k \sin(k\phi) \bigr]

This reduces the number of free scaling parameters from
:math:`2 N_\phi` to :math:`2 (2K + 1)`.  For :math:`K = 1` and
8 angles, the count drops from 16 to 6.

The :class:`~heterodyne.optimization.nlsq.fourier_reparam.FourierReparamConfig`
controls this behaviour:

.. code-block:: python

   from heterodyne.optimization.nlsq.fourier_reparam import FourierReparamConfig

   fourier_config = FourierReparamConfig(
       per_angle_mode="fourier",   # "independent" | "fourier" | "auto"
       fourier_order=1,            # Truncation order K
       fourier_auto_threshold=6,   # Switch to Fourier if N_angles >= threshold
   )

Modes:

``"independent"``
   No reparameterisation; each angle has free contrast and offset.

``"fourier"``
   Express contrast and offset as Fourier series of order :math:`K`.

``"auto"``
   Use ``"fourier"`` when :math:`N_\phi \ge` ``fourier_auto_threshold``,
   otherwise ``"independent"``.


Joint Multi-Angle Fitting
--------------------------

The :func:`~heterodyne.optimization.nlsq.core.fit_nlsq_multi_phi`
function performs a joint fit across all angles simultaneously.  When
Fourier mode is active, the optimiser vector is structured as:

.. code-block:: text

   [ physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs ]

This is transparent to the user -- the result object reports per-angle
contrast and offset values reconstructed from the Fourier coefficients.

.. code-block:: python

   from heterodyne.optimization.nlsq.core import fit_nlsq_multi_phi

   result = fit_nlsq_multi_phi(
       model=model,
       c2_data=c2_stack,
       phi_angles=phi_angles,
       config=nlsq_config,
   )

   # Per-angle values are in the result
   for name, val in result.params_dict.items():
       if name.startswith("contrast") or name.startswith("offset"):
           print(f"  {name}: {val:.4f}")


When to Use Fourier Mode
-------------------------

* **Many angles (>= 6)** -- Fourier mode significantly reduces the
  parameter count and regularises the fit.
* **Smooth angular variation** -- If contrast varies smoothly with
  :math:`\phi` (as expected from beam geometry), Fourier mode is
  well-justified physically.
* **Noisy individual fits** -- If per-angle fits show erratic contrast
  values, Fourier mode pools information across angles for more stable
  estimates.

Fourier mode is **not** recommended when:

* Contrast varies discontinuously (e.g., due to detector gaps or
  beamstop shadows).
* Only 2--3 angles are available (too few data points for meaningful
  Fourier coefficients).
