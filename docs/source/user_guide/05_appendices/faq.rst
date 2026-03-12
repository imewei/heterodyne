.. _faq:

==========================
Frequently Asked Questions
==========================


What is the difference between homodyne and heterodyne?
========================================================

**Homodyne** scattering involves a single scattering component.  The
standard Siegert relation directly yields the dynamics of that component
from the intensity autocorrelation.

**Heterodyne** scattering involves *two* coherently scattering
components (e.g., a static reference and a flowing sample).  The
measured correlation contains three terms: two self-correlations and a
cross-correlation carrying a velocity-dependent phase.  This package
fits the heterodyne three-term model.

If your system has only one component, use the companion *homodyne*
package instead.


Why is heterodyne CPU-only?
============================

The :math:`C_2` matrices from typical beamline XPCS experiments have
dimensions of hundreds to a few thousand frames.  At this scale:

* CPU memory is more than sufficient (a 2000 x 2000 float64 matrix
  is ~30 MB).
* JAX's XLA compiler produces efficient native code for modern
  CPUs with AVX2/AVX-512 instructions.
* NUMA-aware thread pinning gives excellent per-core throughput.
* GPU memory management adds complexity without a proportional
  speedup for these matrix sizes.

The CMC sharding strategy further reduces per-evaluation memory,
making GPU unnecessary even for very long measurements.


How many parameters does the model have?
=========================================

The model has **14 physics parameters** (shared across all angles):

* 3 for reference diffusion (``D0_ref``, ``alpha_ref``,
  ``D_offset_ref``).
* 3 for sample diffusion (``D0_sample``, ``alpha_sample``,
  ``D_offset_sample``).
* 3 for velocity (``v0``, ``beta``, ``v_offset``).
* 4 for fraction evolution (``f0``, ``f1``, ``f2``, ``f3``).
* 1 for flow angle (``phi0``).

Plus **2 scaling parameters per angle** (``contrast``, ``offset``).

Total for :math:`N_\phi` angles: :math:`14 + 2 N_\phi`.

See :doc:`../01_fundamentals/parameter_guide` for full details.


When should I use CMC vs. NLSQ?
=================================

Use **NLSQ** when:

* You need a quick point estimate.
* You are exploring the parameter space or testing different model
  configurations.
* You need a warm-start for CMC.
* Computation time is limited.

Use **CMC** when:

* You need rigorous uncertainty quantification (credible intervals,
  full posteriors).
* You want to detect parameter correlations and multimodality.
* You are preparing results for publication.
* You suspect the NLSQ solution may be a local minimum.

The recommended workflow is always NLSQ first (for warm-start), then
CMC for final results.


What units does heterodyne use?
================================

All lengths are in **Angstroms** (1 |AA| = 10\ :sup:`-10` m):

* Wavevector *q*: |AA|\ :sup:`-1`
* Diffusion prefactor :math:`D_0`: |AA|\ :sup:`2`/s\ :sup:`alpha`
* Diffusion offset :math:`D_\text{off}`: |AA|\ :sup:`2`
* Velocity :math:`v_0`: |AA|/s\ :sup:`beta`
* X-ray wavelength: |AA| (e.g., 1.55 |AA| for 8 keV)

This matches the standard parameterisation used at synchrotron
beamlines (APS, ESRF, PETRA III).  To convert from nanometres, multiply
by 10.

Angles (``phi0``, ``phi_angles``) are in **degrees**.

Times (timestamps, :math:`f_1`, :math:`f_2`) are in **seconds**.


Can I fix some parameters and fit the rest?
============================================

Yes.  In the YAML configuration, set ``vary: false`` and provide a
fixed ``value``:

.. code-block:: yaml

   parameter_space:
     phi0:
       vary: false
       value: 45.0

Or programmatically via the ``ParameterManager``.  Fixed parameters
are excluded from the optimiser vector, reducing the dimensionality
of the problem.


How do I handle data with different numbers of frames per angle?
=================================================================

The joint multi-angle fitter (``fit_nlsq_multi_phi``) requires all
angles to have the same :math:`C_2` dimensions.  If your angles have
different frame counts, either:

1. **Trim** all angles to the minimum common frame count.
2. **Pad** shorter datasets with NaN and mask them during fitting.
3. **Fit angles independently** and combine results post-hoc.

Option 1 is simplest and recommended unless the frame count difference
is large.


.. |AA| unicode:: U+00C5
   :ltrim:
