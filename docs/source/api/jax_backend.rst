===========
JAX Backend
===========

GPU/CPU-accelerated correlation computation via the NLSQ meshgrid path.
Functions in this module build full N x N integral matrices using
cumulative trapezoidal sums and are designed for JIT compilation.

.. automodule:: heterodyne.core.jax_backend
   :members: compute_c2_heterodyne, compute_g1_transport,
             compute_transport_integral_matrix, compute_velocity_integral_matrix
   :undoc-members:
   :show-inheritance:
