Configuration Options Reference
===============================

This page documents every configuration key accepted by the Heterodyne
analysis pipeline, organized by YAML section.

metadata
--------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``config_version``
     - ``str``
     - ``"2.0.1"``
     - Semantic version of the configuration schema.
   * - ``description``
     - ``str``
     - ``""``
     - Free-text description stored in output metadata.

analyzer_parameters
-------------------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``dt``
     - ``float``
     - *required*
     - Frame-to-frame time step in seconds.
   * - ``start_frame``
     - ``int``
     - ``0``
     - Index of the first frame (inclusive).
   * - ``end_frame``
     - ``int``
     - *required*
     - Index of the last frame (exclusive).
   * - ``scattering.wavevector_q``
     - ``float``
     - *required*
     - Scattering wavevector :math:`q` in :math:`\text{\AA}^{-1}`.

experimental_data
-----------------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``file_path``
     - ``str``
     - *required*
     - Path to the HDF5 experiment file.
   * - ``cache_compression``
     - ``bool``
     - ``true``
     - Compress cached intermediate arrays to reduce disk usage.

initial_parameters
------------------

The ``parameter_names`` list enumerates which of the 16 parameters to include
in the fit.  The parameter registry defines default starting values, bounds,
and priors for every parameter.

Physics Parameters (14)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 12 12 12 42
   :header-rows: 1

   * - Parameter
     - Default
     - Min Bound
     - Max Bound
     - Description
   * - ``D0_ref``
     - 1e4
     - 0
     - 1e6
     - Reference diffusion prefactor (:math:`\text{\AA}^2/\text{s}^\alpha`).
   * - ``alpha_ref``
     - 0.0
     - -2.0
     - 2.0
     - Reference transport exponent.
   * - ``D_offset_ref``
     - 0.0
     - -1e4
     - 1e4
     - Reference diffusion offset.
   * - ``D0_sample``
     - 1e4
     - 0
     - 1e6
     - Sample diffusion prefactor (:math:`\text{\AA}^2/\text{s}^\alpha`).
   * - ``alpha_sample``
     - 0.0
     - -2.0
     - 2.0
     - Sample transport exponent.
   * - ``D_offset_sample``
     - 0.0
     - -1e4
     - 1e4
     - Sample diffusion offset.
   * - ``v0``
     - 1e3
     - 0
     - 1e6
     - Velocity amplitude (:math:`\text{\AA}/\text{s}`).
   * - ``beta``
     - 1.0
     - 0
     - 2.0
     - Velocity exponent.
   * - ``v_offset``
     - 0.0
     - -100
     - 100
     - Velocity offset (:math:`\text{\AA}/\text{s}`).
   * - ``f0``
     - 0.5
     - 0
     - 1.0
     - Sample fraction coefficient 0.
   * - ``f1``
     - 0.0
     - -1.0
     - 1.0
     - Sample fraction coefficient 1.
   * - ``f2``
     - 0.0
     - -1.0
     - 1.0
     - Sample fraction coefficient 2.
   * - ``f3``
     - 0.0
     - -1.0
     - 1.0
     - Sample fraction coefficient 3.
   * - ``phi0``
     - 0.0
     - :math:`-\pi`
     - :math:`\pi`
     - Flow angle (radians).

Scaling Parameters (2)
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 12 12 12 42
   :header-rows: 1

   * - Parameter
     - Default
     - Min Bound
     - Max Bound
     - Description
   * - ``contrast``
     - 1.0
     - 0
     - 10.0
     - Speckle contrast scaling factor.
   * - ``offset``
     - 0.0
     - -1.0
     - 1.0
     - Baseline offset.

.. note::

   All 16 parameters have ``vary_default=True`` in the parameter registry.
   To fix a parameter during fitting, override its ``vary`` flag in the
   configuration or the Python API.

optimization
------------

Top-Level
~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``method``
     - ``str``
     - ``"nlsq"``
     - Optimization backend: ``"nlsq"`` or ``"cmc"``.

optimization.nlsq
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``max_iterations``
     - ``int``
     - ``100``
     - Maximum Levenberg--Marquardt iterations.
   * - ``tolerance``
     - ``float``
     - ``1e-8``
     - Convergence tolerance on the cost function.

optimization.cmc
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``num_warmup``
     - ``int``
     - ``500``
     - NUTS warmup (adaptation) steps per chain.
   * - ``num_samples``
     - ``int``
     - ``1000``
     - Post-warmup posterior draws per chain.
   * - ``num_chains``
     - ``int``
     - ``4``
     - Number of independent MCMC chains.
   * - ``target_accept_prob``
     - ``float``
     - ``0.8``
     - Target acceptance probability for NUTS step-size adaptation.
   * - ``max_r_hat``
     - ``float``
     - ``1.01``
     - Maximum Gelman--Rubin :math:`\hat{R}` for convergence.

output
------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``directory``
     - ``str``
     - ``"./results"``
     - Directory for result files and checkpoints.
   * - ``formats``
     - ``list[str]``
     - ``["json", "npz"]``
     - Output format(s): ``"json"``, ``"npz"``, or both.
