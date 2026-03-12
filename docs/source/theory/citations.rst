.. _citations:

==========
References
==========

Primary Publications
--------------------

The theoretical framework implemented by this package is developed in the
following two papers:

.. [He2024] H. He, W. Chen, *et al.*,
   "Generalized two-time correlation for study of nonequilibrium dynamics
   with X-ray photon correlation spectroscopy,"
   *Proc. Natl. Acad. Sci. U.S.A.* **121** (48), e2401162121 (2024).
   doi:`10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

   Introduces the transport coefficient :math:`J(t)`, the two-time
   correlation framework, and the connection between microscopic velocity
   statistics and measurable scattering correlations. Derives the integral
   formulation and the power-law parameterization.

.. [He2025] H. He, W. Chen, *et al.*,
   "Multi-component heterodyne XPCS for extracting flow dynamics under
   nonequilibrium conditions,"
   *Proc. Natl. Acad. Sci. U.S.A.* **122**, e2514216122 (2025).
   doi:`10.1073/pnas.2514216122 <https://doi.org/10.1073/pnas.2514216122>`_

   Extends the theory to multi-component heterodyne detection. Derives
   the N-component correlation (SI Eq. S-94), the two-component
   specialization (SI Eqs. S-95/S-98), and demonstrates velocity
   extraction from cross-correlation oscillations.

BibTeX Entries
--------------

.. code-block:: bibtex

   @article{He2024,
     author  = {He, Hongrui and Chen, Wei and others},
     title   = {Generalized two-time correlation for study of
                nonequilibrium dynamics with {X}-ray photon
                correlation spectroscopy},
     journal = {Proceedings of the National Academy of Sciences},
     volume  = {121},
     number  = {48},
     pages   = {e2401162121},
     year    = {2024},
     doi     = {10.1073/pnas.2401162121},
   }

   @article{He2025,
     author  = {He, Hongrui and Chen, Wei and others},
     title   = {Multi-component heterodyne {XPCS} for extracting
                flow dynamics under nonequilibrium conditions},
     journal = {Proceedings of the National Academy of Sciences},
     volume  = {122},
     pages   = {e2514216122},
     year    = {2025},
     doi     = {10.1073/pnas.2514216122},
   }

Related References
------------------

.. [SiegertRelation] A. J. F. Siegert,
   "On the fluctuations in signals returned by many independently
   moving scatterers,"
   MIT Radiation Laboratory Report No. 465 (1943).

   Original derivation of the Siegert relation connecting first- and
   second-order correlation functions for Gaussian random fields.

.. [GreenKubo] M. S. Green,
   "Markoff random processes and the statistical mechanics of
   time-dependent phenomena. II. Irreversible processes in fluids,"
   *J. Chem. Phys.* **22**, 398 (1954).
   R. Kubo,
   "Statistical-mechanical theory of irreversible processes. I.
   General theory and simple applications to magnetic and conduction
   problems,"
   *J. Phys. Soc. Japan* **12**, 570 (1957).

   The Green-Kubo relations connecting transport coefficients to
   time-integrated velocity autocorrelation functions.

.. [BerneXPCS] B. J. Berne and R. Pecora,
   *Dynamic Light Scattering: With Applications to Chemistry,
   Biology, and Physics* (Wiley, 1976).

   Standard reference for the theory of dynamic light scattering,
   including homodyne and heterodyne detection geometries.

.. [SuttonXPCS] M. Sutton,
   "A review of X-ray intensity fluctuation spectroscopy,"
   *C. R. Physique* **9**, 657 (2008).

   Comprehensive review of XPCS methodology, coherence requirements,
   and data analysis procedures.
