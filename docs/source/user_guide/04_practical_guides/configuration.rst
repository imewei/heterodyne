.. _configuration:

=============
Configuration
=============

The heterodyne package uses YAML configuration files to define analysis
parameters, experimental metadata, and optimisation settings.  This
page documents each section of the configuration file.


Configuration File Structure
-----------------------------

A complete YAML configuration contains the following top-level sections:

.. code-block:: yaml

   metadata:
     ...
   analysis_mode:
     ...
   analyzer_parameters:
     ...
   experimental_data:
     ...
   initial_parameters:
     ...
   parameter_space:
     ...
   optimization:
     ...
   output:
     ...


Section Reference
-----------------

metadata
--------

General information about the analysis run.

.. code-block:: yaml

   metadata:
     name: "gel_dynamics_run42"
     description: "Two-component fit for colloidal gel at T=25C"
     operator: "J. Smith"
     date: "2026-03-12"

analysis_mode
-------------

Controls what type of analysis to run.

.. code-block:: yaml

   analysis_mode:
     mode: "nlsq"          # "nlsq", "cmc", or "nlsq+cmc"
     per_angle: true        # Fit each angle independently or jointly
     fourier_mode: "auto"   # "independent", "fourier", or "auto"

analyzer_parameters
-------------------

Physical constants and measurement geometry.

.. code-block:: yaml

   analyzer_parameters:
     wavelength: 1.55       # X-ray wavelength in Angstroms (8 keV)
     q_value: 0.025         # Scattering wavevector in 1/Angstroms
     detector_distance: 5.0 # Sample-to-detector distance in metres

experimental_data
-----------------

Paths to the input data files and angle definitions.

.. code-block:: yaml

   experimental_data:
     data_dir: "./data/run42/"
     file_pattern: "c2_phi{angle:.1f}.h5"
     phi_angles: [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
     format: "auto"         # "auto", "aps_u", "aps_legacy", "npz", "mat"
     use_cache: true         # Enable NPZ caching

initial_parameters
------------------

Starting values for the optimiser.  Any parameter not listed uses the
default from the parameter registry.

.. code-block:: yaml

   initial_parameters:
     D0_ref: 1000.0
     D0_sample: 5000.0
     v0: 500.0
     f0: 0.5
     f3: 0.3

parameter_space
---------------

Bounds and fix/free status for each parameter.

.. code-block:: yaml

   parameter_space:
     D0_ref:
       min: 1.0
       max: 1.0e6
       vary: true
     alpha_ref:
       min: -2.0
       max: 2.0
       vary: true
     phi0:
       min: -180.0
       max: 180.0
       vary: false           # Fix flow angle
       value: 45.0           # Fixed value

optimization
------------

Settings for the NLSQ and CMC optimisers.

.. code-block:: yaml

   optimization:
     nlsq:
       strategy: "jit"
       n_starts: 20
       lhs_seed: 42
       ftol: 1.0e-10
       xtol: 1.0e-10
       max_nfev: 5000
     cmc:
       target_accept_prob: 0.9
       max_r_hat: 1.01
       nlsq_prior_width_factor: 5.0
       num_warmup: 1000
       num_samples: 2000
       num_chains: 4
       num_shards: 8
     cmaes:
       sigma0: 0.25
       population_size: 64
       max_generations: 500
       seed: 42

output
------

Controls for result storage and reporting.

.. code-block:: yaml

   output:
     output_dir: "./results/run42/"
     save_residuals: true
     save_posterior_samples: true
     checkpoint: true
     checkpoint_interval: 100
     log_level: "INFO"


Loading Configuration
---------------------

.. code-block:: python

   import yaml
   from pathlib import Path

   config_path = Path("analysis_config.yaml")
   with config_path.open() as f:
       config = yaml.safe_load(f)

The configuration dictionary is passed to the analysis runner, which
constructs the appropriate ``NLSQConfig``, ``CMCConfig``, and model
objects.


Environment Variables
---------------------

Some settings can also be controlled via environment variables,
which take precedence over YAML values:

``HETERODYNE_LOG_LEVEL``
   Logging verbosity (``DEBUG``, ``INFO``, ``WARNING``).

``OMP_NUM_THREADS``
   Number of OpenMP threads for BLAS operations.

``XLA_FLAGS``
   Custom XLA compiler flags.  Normally configured automatically by
   ``heterodyne-config-xla``.

See :doc:`performance_tuning` for details on XLA and threading
configuration.
