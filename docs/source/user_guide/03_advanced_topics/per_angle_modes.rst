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
====================

For :math:`N_\phi` azimuthal angles the model has:

* **14 shared physics parameters** -- identical across all angles.
* **2 scaling parameters per angle**:
  ``contrast_i`` (:math:`\beta_i`) and ``offset_i``.

Total free parameters: :math:`14 + 2 N_\phi`.

For a typical 8-angle dataset: :math:`14 + 16 = 30`.


ScalingConfig and PerAngleScaling
==================================

The :class:`~heterodyne.core.scaling_utils.ScalingConfig` dataclass
controls scaling behaviour:

.. code-block:: python

   from heterodyne.core.scaling_utils import ScalingConfig

   config = ScalingConfig(
       n_angles=8,
       mode="individual",  # Each angle gets independent contrast/offset
   )

Available scaling modes:

``"constant"``
   A single contrast and offset shared across all angles.
   Free scaling parameters: 2.

``"individual"``
   Independent contrast and offset for every angle.
   Free scaling parameters: :math:`2 N_\phi`.

``"auto"``
   Automatically selects between ``"constant"`` and ``"individual"``
   based on the number of angles and data quality.

``"constant_averaged"``
   Fits individual values, then averages them post-hoc.  Useful for
   diagnostics (checking angle-to-angle consistency) while reporting
   a single representative value.

The :class:`~heterodyne.core.scaling_utils.PerAngleScaling` manager
tracks which scaling parameters are varying in the optimiser and
provides ``expand`` / ``compress`` operations to convert between the
flat optimiser vector and the per-angle representation.


Fourier Reparameterisation
===========================

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
==========================

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
=========================

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
