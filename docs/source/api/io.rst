=============
I/O Utilities
=============

Serialization of NLSQ and MCMC results to JSON, NPZ, and diagnostic
output formats.  The :mod:`heterodyne.optimization.cmc.io` module
provides additional shard-level CMC I/O; see :doc:`cmc`.

JSON Utilities
==============

JSON-safe serialization helpers for JAX arrays, NumPy scalars, and
custom result types.

.. automodule:: heterodyne.io.json_utils
   :members: json_safe, json_serializer
   :undoc-members:
   :show-inheritance:

NLSQ Writers
============

.. automodule:: heterodyne.io.nlsq_writers
   :members: save_nlsq_json_files, save_nlsq_npz_file
   :undoc-members:
   :show-inheritance:

MCMC Writers
============

.. automodule:: heterodyne.io.mcmc_writers
   :members: save_mcmc_results, save_mcmc_diagnostics
   :undoc-members:
   :show-inheritance:
