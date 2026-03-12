Quick Start
===========

This guide walks through a complete heterodyne XPCS analysis in five minutes.

1. Install
----------

.. code-block:: bash

   uv add heterodyne

See :doc:`installation` for alternative installation methods and environment
setup.

2. Generate a Configuration File
---------------------------------

.. code-block:: bash

   heterodyne-config --output my_config.yaml

This creates a YAML configuration file with sensible defaults. The key
sections are:

.. code-block:: yaml

   analysis_mode: heterodyne

   experimental_data:
     data_file: /path/to/experiment.hdf5

   analyzer_parameters:
     dt: 0.001              # Frame interval (seconds)
     start_frame: 0         # First frame to include
     end_frame: -1          # Last frame (-1 = all)
     wavevector_q: 0.015    # Wavevector in Angstrom^-1

   optimization:
     method: nlsq           # "nlsq" or "cmc"

Edit ``data_file`` and ``wavevector_q`` to match your experiment.

3. Load Data
------------

.. code-block:: python

   from heterodyne.data.xpcs_loader import XPCSDataLoader

   loader = XPCSDataLoader("experiment.hdf5")
   data = loader.load()

``XPCSDataLoader`` reads HDF5 files produced by standard XPCS beamline
pipelines and validates shapes, dtypes, and monotonicity at load time.

4. Run NLSQ Fitting
--------------------

.. code-block:: python

   from heterodyne.optimization.nlsq.core import fit_nlsq_jax

   result = fit_nlsq_jax(
       data=data,
       method="trf",
   )

   print(result.parameters)
   print(result.cost)

The trust-region reflective (``trf``) solver provides fast, deterministic
parameter estimates suitable as warm-start values for subsequent Bayesian
inference.

5. CLI Usage
------------

Run the full pipeline from the command line:

.. code-block:: bash

   heterodyne --config my_config.yaml --method nlsq

For Bayesian inference via NUTS/CMC:

.. code-block:: bash

   heterodyne --config my_config.yaml --method cmc

Results are written to JSON (parameter summaries) and NPZ (posterior samples,
diagnostics) files in the output directory.

Parameter Interpretation
------------------------

The 14 physics parameters and 2 scaling parameters describe a two-component
heterodyne correlation model. Units follow the standard beamline convention
(all lengths in Angstroms).

**Diffusion parameters**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Range
     - Interpretation
   * - ``D0_ref``
     - 1 -- 1e6
     - Reference diffusion prefactor (Angstrom^2 / s^alpha)
   * - ``D0_sample``
     - 1 -- 1e6
     - Sample diffusion prefactor (Angstrom^2 / s^alpha)
   * - ``alpha_ref``
     - -2 -- 2
     - Reference transport exponent (0 = normal diffusion)
   * - ``alpha_sample``
     - -2 -- 2
     - Sample transport exponent (0 = normal diffusion)

**Velocity parameters**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Range
     - Interpretation
   * - ``v0``
     - > 0
     - Velocity prefactor (Angstrom / s^beta)
   * - ``v_offset``
     - -100 -- 100
     - Velocity offset (Angstrom / s)
   * - ``phi0``
     - 0 -- 360
     - Flow angle offset (degrees)
   * - ``beta``
     - -2 -- 2
     - Velocity time exponent

**Sample fraction parameters**

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Parameter
     - Interpretation
   * - ``f0``, ``f1``, ``f2``, ``f3``
     - Coefficients controlling the time-dependent sample fraction evolution

**Scaling parameters (per-angle)**

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Parameter
     - Interpretation
   * - ``contrast``
     - Amplitude scaling of the correlation function
   * - ``offset``
     - Additive baseline offset

Next Steps
----------

- :doc:`installation` -- detailed environment setup and HPC configuration
- Theory & Physics -- derivation of the two-component heterodyne model
- Configuration -- full reference for all YAML options
- API Reference -- complete module and function documentation
