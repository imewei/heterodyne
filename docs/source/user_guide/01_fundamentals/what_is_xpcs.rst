.. _what-is-xpcs:

==============
What Is XPCS?
==============

X-ray Photon Correlation Spectroscopy (XPCS) is a technique that uses
the temporal correlations of coherent X-ray speckle patterns to probe
nanoscale dynamics in condensed-matter systems.  This page provides the
conceptual foundation needed to understand the rest of the user guide.


Coherent X-rays and Speckle Patterns
=====================================

When a partially coherent X-ray beam (wavelength on the order of 1 |AA|)
illuminates a disordered sample, interference among the scattered waves
produces a granular intensity distribution on the detector known as a
**speckle pattern**.  Each speckle encodes the instantaneous spatial
arrangement of scatterers at the probed wavevector *q*.

Key properties:

* The speckle size is set by the coherence area of the beam, not by the
  sample structure.
* As particles move, the speckle pattern evolves -- fast dynamics produce
  rapid intensity fluctuations, slow dynamics produce slowly drifting
  speckles.
* Recording a time series of 2-D detector frames gives direct access to
  the dynamics at every *q* pixel simultaneously.


From Speckles to Dynamics
==========================

The fundamental observable is the **intensity--intensity autocorrelation
function**:

.. math::

   g_2(\mathbf{q}, \tau)
   = \frac{\langle I(\mathbf{q}, t)\, I(\mathbf{q}, t+\tau) \rangle_t}
          {\langle I(\mathbf{q}, t) \rangle_t^2}

For ergodic, stationary systems this depends only on the lag time
:math:`\tau`.  In non-stationary samples the full **two-time correlation
function** must be retained:

.. math::

   C_2(\mathbf{q};\, t_1, t_2)
   = \frac{\langle I(\mathbf{q}, t_1)\, I(\mathbf{q}, t_2) \rangle}
          {\langle I(\mathbf{q}, t_1) \rangle
           \langle I(\mathbf{q}, t_2) \rangle}

The heterodyne package works with :math:`C_2(t_1, t_2)` matrices directly,
making it suitable for systems that age, flow, or otherwise evolve during
the measurement.


The Siegert Relation
====================

Under Gaussian statistics the measured intensity correlation is related
to the *field* (amplitude) autocorrelation :math:`g_1` through the
**Siegert relation**:

.. math::

   g_2(\mathbf{q}, \tau)
   = 1 + \beta\, |g_1(\mathbf{q}, \tau)|^2

where :math:`\beta` is the **speckle contrast** (optical coherence
factor, :math:`0 < \beta \le 1`).  The physics enters through
:math:`g_1`, which in turn depends on the transport coefficient and
velocity fields of the scatterers.


Transport Coefficient
=====================

The field correlation for a single-component diffusive system is:

.. math::

   g_1(q, \tau) = \exp\!\bigl[-q^2\, J(\tau)\bigr]

where :math:`J(\tau)` is the cumulative transport coefficient (integrated
mean-squared displacement):

.. math::

   J(\tau) = D_0\, \frac{\tau^{1+\alpha}}{1+\alpha} + D_\text{offset}\, \tau

* :math:`D_0` (|AA|\ :sup:`2`/s\ :sup:`alpha`) -- diffusion prefactor.
* :math:`\alpha` (dimensionless) -- transport exponent.
  :math:`\alpha = 0` is normal Brownian diffusion;
  :math:`\alpha > 0` is super-diffusive;
  :math:`\alpha < 0` is sub-diffusive.
* :math:`D_\text{offset}` (|AA|\ :sup:`2`) -- constant baseline.


Homodyne vs. Heterodyne Scattering
====================================

**Homodyne scattering** arises when a single scattering component
dominates the signal.  The standard Siegert relation applies directly,
and :math:`g_2 - 1 \propto |g_1|^2` gives the dynamics of that one
component.

**Heterodyne scattering** occurs when *two* distinct scattering
populations contribute coherently to the same detector pixel -- for
example a static reference component and a flowing sample component.
The measured correlation then contains three terms:

1. Reference self-correlation.
2. Sample self-correlation.
3. A **cross-correlation** that carries a velocity-dependent phase.

The heterodyne cross-term is sensitive to *directed* motion (flow
velocity) in addition to diffusion, and its phase depends on the angle
between the scattering vector and the flow direction.  Extracting all
three contributions from the measured :math:`C_2` is the central inverse
problem solved by this package.


Why XPCS?
==========

Compared to alternative probes of nanoscale dynamics (DLS, DDM, particle
tracking), XPCS offers several distinct advantages:

* **No dilution required** -- measurements are made on concentrated, opaque,
  or otherwise optically inaccessible samples.
* **Spatial selectivity** -- the wavevector *q* selects a specific length
  scale (from sub-nm to hundreds of nm).
* **Azimuthal resolution** -- anisotropic dynamics (e.g., flow) are
  resolved as a function of the in-plane angle :math:`\phi`.
* **Sub-|AA| wavelength** -- hard X-rays penetrate thick or absorbing
  materials that block visible light.
* **Non-invasive** -- the sample is probed in situ under realistic
  conditions (temperature, pressure, confinement).

These properties make XPCS the method of choice for studying dynamics
in colloidal gels, metallic glasses, cement hydration, biological
membranes, and other complex systems.


.. |AA| unicode:: U+00C5
   :ltrim:
