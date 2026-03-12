.. _glossary:

========
Glossary
========

.. glossary::
   :sorted:

   XPCS
      X-ray Photon Correlation Spectroscopy.  A technique that extracts
      dynamics from the temporal correlations of coherent X-ray speckle
      patterns.

   speckle
      A granular intensity pattern produced by the interference of
      coherent waves scattered from a disordered sample.  The speckle
      size depends on the coherence of the beam, not on the sample
      structure.

   coherent scattering
      Scattering in which the phase relationships between scattered
      waves are preserved, enabling interference and speckle formation.
      Requires a beam with sufficient transverse and longitudinal
      coherence.

   two-time correlation
      The matrix :math:`C_2(t_1, t_2)` measuring the normalised
      intensity--intensity correlation between frames at times
      :math:`t_1` and :math:`t_2`.  For stationary systems it depends
      only on the lag :math:`\tau = |t_2 - t_1|`; for non-stationary
      systems the full matrix must be retained.

   Siegert relation
      The relation :math:`g_2 = 1 + \beta |g_1|^2` connecting the
      intensity autocorrelation (:math:`g_2`) to the field
      autocorrelation (:math:`g_1`) under Gaussian statistics.
      :math:`\beta` is the speckle contrast.

   transport coefficient
      The cumulative function :math:`J(\tau)` describing the integrated
      mean-squared displacement growth.  For anomalous diffusion:
      :math:`J(\tau) = D_0 \tau^{1+\alpha}/(1+\alpha) + D_\text{off}\,\tau`.

   heterodyne scattering
      Scattering from a system with two coherently contributing
      components (e.g., a static reference and a mobile sample).  The
      measured correlation contains self-correlation terms from each
      component plus a cross-correlation term carrying a velocity phase.

   homodyne scattering
      Scattering dominated by a single component.  The standard Siegert
      relation applies directly, and only diffusion (no velocity phase)
      needs to be modelled.

   NLSQ
      Non-Linear Least Squares.  An optimisation method that minimises
      the sum of squared residuals between data and model.  In the
      heterodyne package, implemented via ``scipy.optimize.least_squares``
      with JAX-computed residuals and Jacobians.

   CMC
      Consensus Monte Carlo.  A divide-and-combine Bayesian strategy
      that splits the dataset into shards, runs MCMC independently on
      each, and merges posteriors via inverse-variance weighting.

   NUTS
      No-U-Turn Sampler.  An adaptive variant of Hamiltonian Monte
      Carlo (HMC) that automatically tunes the trajectory length.
      Implemented in the package via NumPyro.

   ArviZ
      A Python library for exploratory analysis of Bayesian models,
      providing diagnostics (R-hat, ESS, BFMI), summary statistics,
      and visualisation.  Used as the standard diagnostic backend in
      the heterodyne package.

   R-hat
      The Gelman--Rubin convergence diagnostic
      (:math:`\hat{R}`).  Compares between-chain and within-chain
      variance.  Values below 1.01 indicate convergence; values above
      1.1 indicate the chains have not mixed.

   ESS
      Effective Sample Size.  The number of independent draws equivalent
      to the autocorrelated MCMC chain.  ESS > 400 is generally
      adequate for posterior summaries.

   BFMI
      Bayesian Fraction of Missing Information.  Measures how well the
      MCMC sampler explores the energy distribution.  Values below 0.3
      suggest poor exploration.

   per-angle scaling
      The practice of fitting independent speckle contrast
      (:math:`\beta_i`) and baseline offset values for each azimuthal
      angle :math:`\phi_i`, accounting for angle-dependent optical
      effects.

   Fourier reparameterisation
      Expressing the per-angle contrast and offset as truncated Fourier
      series in :math:`\phi`, reducing the number of free scaling
      parameters from :math:`2 N_\phi` to :math:`2(2K+1)` where
      :math:`K` is the Fourier order.

   ShardGrid
      A data structure in ``core/physics_cmc.py`` that stores only the
      :math:`(t_1, t_2)` pairs needed by a single CMC shard, avoiding
      allocation of the full :math:`N \times N` correlation matrix.

   CMA-ES
      Covariance Matrix Adaptation Evolution Strategy.  A derivative-free
      global optimisation algorithm that maintains and adapts a
      multivariate normal search distribution.  Used in the package as
      a fallback when gradient-based NLSQ fails to find the global
      minimum.

   speckle contrast
      The optical coherence factor :math:`\beta` (:math:`0 < \beta \le 1`)
      quantifying the visibility of the speckle pattern.  Depends on
      beam coherence, detector pixel size, and scattering geometry.

   reduced chi-squared
      :math:`\chi^2_\text{red} = \chi^2 / \nu` where :math:`\nu` is
      the number of degrees of freedom.  A value near 1.0 indicates a
      good fit; much greater than 1 indicates poor fit; much less than
      1 suggests overfitting or overestimated errors.

   wavevector
      The scattering vector :math:`\mathbf{q}` whose magnitude
      :math:`q = (4\pi/\lambda)\sin(\theta/2)` selects the probed
      length scale.  Units: inverse Angstroms (1/|AA|).

.. |AA| unicode:: U+00C5
   :ltrim:
