Configuration Templates
=======================

This page provides a complete YAML template that covers every configuration
section.  Copy it as a starting point and adjust values to match your
experiment.

Full Template
-------------

.. code-block:: yaml

   # ---------------------------------------------------------------
   # Heterodyne XPCS Analysis -- Configuration Template
   # ---------------------------------------------------------------

   metadata:
     config_version: "2.0.1"
     description: "Two-component heterodyne analysis"

   analyzer_parameters:
     dt: 0.1                     # Frame-to-frame time step (seconds)
     start_frame: 0              # First frame index (inclusive)
     end_frame: 1000             # Last frame index (exclusive)
     scattering:
       wavevector_q: 0.0054      # Scattering wavevector q (A^-1)

   experimental_data:
     file_path: "./data/experiment.hdf"
     cache_compression: true     # Compress cached intermediate arrays

   initial_parameters:
     parameter_names:
       - D0_ref                  # Reference diffusion prefactor (A^2/s^alpha)
       - alpha_ref               # Reference transport exponent
       - D_offset_ref            # Reference diffusion offset
       - D0_sample               # Sample diffusion prefactor (A^2/s^alpha)
       - alpha_sample            # Sample transport exponent
       - D_offset_sample         # Sample diffusion offset
       - v0                      # Velocity amplitude (A/s)
       - beta                    # Velocity exponent
       - v_offset                # Velocity offset (A/s)
       - f0                      # Sample fraction coefficient 0
       - f1                      # Sample fraction coefficient 1
       - f2                      # Sample fraction coefficient 2
       - f3                      # Sample fraction coefficient 3
       - phi0                    # Flow angle (radians)

   optimization:
     method: "nlsq"              # "nlsq" or "cmc"

     nlsq:
       max_iterations: 100       # Maximum L-M iterations
       tolerance: 1.0e-8         # Convergence tolerance (cost function)

     cmc:
       num_warmup: 500           # NUTS warmup (adaptation) steps
       num_samples: 1000         # Post-warmup posterior draws per chain
       num_chains: 4             # Independent MCMC chains
       target_accept_prob: 0.8   # Target acceptance probability for NUTS
       max_r_hat: 1.01           # Convergence threshold (Gelman-Rubin R-hat)

   output:
     directory: "./results"
     formats: ["json", "npz"]    # One or both of json, npz

Minimal Template
----------------

The minimal configuration requires only the data path and frame range.
All other values fall back to defaults defined in the parameter registry.

.. code-block:: yaml

   metadata:
     config_version: "2.0.1"

   analyzer_parameters:
     dt: 0.1
     start_frame: 0
     end_frame: 1000
     scattering:
       wavevector_q: 0.0054

   experimental_data:
     file_path: "./data/experiment.hdf"

   optimization:
     method: "nlsq"

CMC-Only Template
-----------------

When running Bayesian inference without a preceding NLSQ warm-start, ensure
that ``initial_parameters`` provides reasonable starting values so the sampler
starts in a region of non-negligible posterior density.

.. code-block:: yaml

   metadata:
     config_version: "2.0.1"
     description: "Bayesian CMC analysis (no NLSQ warm-start)"

   analyzer_parameters:
     dt: 0.1
     start_frame: 0
     end_frame: 1000
     scattering:
       wavevector_q: 0.0054

   experimental_data:
     file_path: "./data/experiment.hdf"

   optimization:
     method: "cmc"
     cmc:
       num_warmup: 1000
       num_samples: 2000
       num_chains: 4
       target_accept_prob: 0.85
       max_r_hat: 1.01

   output:
     directory: "./results/cmc_only"
     formats: ["json", "npz"]
