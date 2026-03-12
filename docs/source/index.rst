.. Heterodyne documentation master file

.. image:: https://img.shields.io/badge/python-3.12%2B-blue
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/JAX-CPU--only-orange
   :alt: JAX CPU-only

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

.. image:: https://img.shields.io/badge/DOI-10.1073%2Fpnas.2401162121-blue
   :target: https://doi.org/10.1073/pnas.2401162121
   :alt: PNAS 2024

.. image:: https://img.shields.io/badge/DOI-10.1073%2Fpnas.2514216122-blue
   :target: https://doi.org/10.1073/pnas.2514216122
   :alt: PNAS 2025

Heterodyne
==========

CPU-optimized JAX-based heterodyne scattering analysis for XPCS under nonequilibrium conditions.

Heterodyne provides a complete pipeline for fitting two-component heterodyne
correlation functions from X-ray Photon Correlation Spectroscopy (XPCS) experiments.
It combines JIT-compiled physics kernels with dual inference engines (NLSQ and
MCMC/NUTS) to extract transport coefficients, velocity fields, and sample
fractions from nonequilibrium soft matter systems.

At a Glance
-----------

.. list-table::
   :widths: 30 70
   :stub-columns: 1

   * - **Python**
     - 3.12+
   * - **JAX**
     - >= 0.8.2 (CPU-only)
   * - **NLSQ**
     - >= 0.6.10 (trust-region Levenberg--Marquardt)
   * - **Bayesian**
     - NumPyro / NUTS / CMC
   * - **Analysis mode**
     - Two-component heterodyne (14 physics + 2 scaling parameters)
   * - **Input**
     - HDF5 via ``XPCSDataLoader``
   * - **Output**
     - JSON + NPZ
   * - **License**
     - MIT

Key Features
------------

- **JIT-compiled heterodyne physics** -- all correlation kernels run through
  JAX's XLA compiler for maximum throughput on multi-core CPUs.
- **Two-component velocity phase model** -- simultaneously resolves reference
  and sample dynamics with distinct transport exponents.
- **14-parameter transport + fraction + velocity model** -- diffusion
  prefactors, transport exponents, velocity amplitude and offset, flow angle,
  and time-dependent sample fractions.
- **NLSQ + CMC/NUTS dual inference** -- fast deterministic warm-start via
  trust-region Levenberg--Marquardt, followed by fully Bayesian posterior
  sampling.
- **Element-wise ShardGrid CMC** -- O(n_pairs) cumsum lookup prevents N x N
  matrix allocation, eliminating out-of-memory failures on long time series.
- **Fourier reparameterization for multi-angle fits** -- joint optimization
  across azimuthal angles with configurable Fourier order.
- **CPU/NUMA-aware HPC backend** -- automatic thread pinning and XLA flag
  tuning for 36--128-core cluster nodes.
- **Strict data integrity** -- runtime validation of shapes, dtypes, NaN
  guards, and monotonicity checks at every I/O boundary.

Quick Start
-----------

**1. Install**

.. code-block:: bash

   uv add heterodyne

**2. Generate a configuration file**

.. code-block:: bash

   heterodyne-config --output my_config.yaml

**3. Run an analysis**

.. code-block:: bash

   heterodyne --config my_config.yaml --method nlsq

**Python API**

.. code-block:: python

   from heterodyne.data.xpcs_loader import XPCSDataLoader
   from heterodyne.optimization.nlsq.core import fit_nlsq_jax

   loader = XPCSDataLoader("experiment.hdf5")
   data = loader.load()

   result = fit_nlsq_jax(
       data=data,
       method="trf",
   )

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/01_fundamentals/index
   user_guide/02_data_and_fitting/index
   user_guide/03_advanced_topics/index
   user_guide/04_practical_guides/index
   user_guide/05_appendices/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Physics

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   configuration/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture Deep Dives

   architecture/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   installation
   quickstart
   contributing

Citation
--------

If you use Heterodyne in your research, please cite both papers:

**He et al., PNAS 2024** -- Transport coefficient approach for characterizing
nonequilibrium dynamics in soft matter:

.. code-block:: bibtex

   @article{He2024transport,
     title   = {Transport coefficient approach for characterizing
                nonequilibrium dynamics in soft matter},
     author  = {He, Hongrui and Chen, Wei and others},
     journal = {Proceedings of the National Academy of Sciences},
     year    = {2024},
     doi     = {10.1073/pnas.2401162121},
   }

**He et al., PNAS 2025** -- Bridging microscopic dynamics and rheology in the
yielding of charged colloidal suspensions:

.. code-block:: bibtex

   @article{He2025bridging,
     title   = {Bridging microscopic dynamics and rheology in the
                yielding of charged colloidal suspensions},
     author  = {He, Hongrui and Chen, Wei and others},
     journal = {Proceedings of the National Academy of Sciences},
     year    = {2025},
     doi     = {10.1073/pnas.2514216122},
   }

Community & Support
-------------------

- **Source code:** https://github.com/imewei/heterodyne
- **Issue tracker:** https://github.com/imewei/heterodyne/issues
- **Documentation:** https://heterodyne.readthedocs.io

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
