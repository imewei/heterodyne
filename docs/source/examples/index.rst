Examples
========

This section provides worked examples demonstrating common analysis
workflows with Heterodyne.

End-to-End NLSQ Workflow
-------------------------

The standard workflow loads an HDF5 dataset, runs NLSQ fitting, and
saves the results:

.. code-block:: python

   from heterodyne.data.xpcs_loader import XPCSDataLoader
   from heterodyne.optimization.nlsq.core import fit_nlsq_jax

   # 1. Load experimental data
   loader = XPCSDataLoader("experiment.hdf5")
   data = loader.load()

   # 2. Run NLSQ fitting (trust-region Levenberg-Marquardt)
   result = fit_nlsq_jax(
       data=data,
       method="trf",
       max_iterations=100,
       tolerance=1e-8,
   )

   # 3. Inspect results
   print(f"Converged: {result.success}")
   print(f"Cost: {result.cost:.6e}")
   for name, value in zip(result.parameter_names, result.x, strict=True):
       print(f"  {name}: {value:.6f}")

Or from the command line:

.. code-block:: bash

   heterodyne --config my_config.yaml --method nlsq

Bayesian CMC Analysis
---------------------

For full posterior uncertainty quantification, use the CMC (Consensus
Monte Carlo) backend.  A preceding NLSQ warm-start provides a good
initial point for the sampler:

.. code-block:: python

   from heterodyne.data.xpcs_loader import XPCSDataLoader
   from heterodyne.optimization.nlsq.core import fit_nlsq_jax
   from heterodyne.optimization.cmc.runner import run_cmc

   # 1. Load data
   loader = XPCSDataLoader("experiment.hdf5")
   data = loader.load()

   # 2. NLSQ warm-start
   nlsq_result = fit_nlsq_jax(data=data, method="trf")

   # 3. CMC posterior sampling
   cmc_result = run_cmc(
       data=data,
       init_params=nlsq_result.x,
       num_warmup=500,
       num_samples=1000,
       num_chains=4,
       target_accept_prob=0.8,
   )

   # 4. Diagnostics (ArviZ)
   import arviz as az

   idata = cmc_result.to_arviz()
   summary = az.summary(idata, var_names=cmc_result.parameter_names)
   print(summary)

   # Check convergence
   assert summary["r_hat"].max() < 1.01, "Chains have not converged"

From the command line, the full NLSQ-then-CMC pipeline runs as:

.. code-block:: bash

   heterodyne --config my_config.yaml --method nlsq
   heterodyne --config my_config.yaml --method cmc

Multi-Angle Joint Fitting
--------------------------

When data is collected at multiple azimuthal angles, the Fourier
reparameterization mode performs a joint fit across all angles:

.. code-block:: python

   from heterodyne.core.fourier_reparam import (
       FourierReparamConfig,
       fit_nlsq_multi_phi,
   )

   config = FourierReparamConfig(
       per_angle_mode="fourier",  # "independent", "fourier", or "auto"
       fourier_order=2,
   )

   result = fit_nlsq_multi_phi(
       data_list=multi_angle_data,   # List of datasets, one per angle
       phi_values=phi_angles,        # Array of azimuthal angles
       config=config,
   )

   # The result parameter vector contains:
   # [physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]

In ``"auto"`` mode, the fitter selects between independent and Fourier
reparameterization based on the ``fourier_auto_threshold`` setting.

Further Reading
---------------

- :doc:`/user_guide/01_fundamentals/index` -- Conceptual introduction to
  heterodyne XPCS analysis.
- :doc:`/configuration/index` -- Full YAML configuration reference.
- :doc:`/api/index` -- Python API documentation.
- :doc:`/developer/architecture` -- Package architecture and design patterns.
