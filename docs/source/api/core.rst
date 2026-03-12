============
Core Physics
============

Foundational physics constants, safe numerical primitives, and integral
building blocks shared by both the NLSQ meshgrid path and the CMC
element-wise path.

Physics Constants and Validation
================================

.. automodule:: heterodyne.core.physics
   :members: PhysicsConstants, PARAMETER_BOUNDS, ValidationResult
   :undoc-members:
   :show-inheritance:

Numerical Primitives
====================

Safe arithmetic wrappers and integral utilities used throughout the
correlation computation pipeline.

.. automodule:: heterodyne.core.physics_utils
   :members: safe_exp, safe_log, safe_power, safe_divide, safe_sqrt,
             trapezoid_cumsum, create_time_integral_matrix, smooth_abs,
             compute_transport_rate, compute_velocity_rate
   :undoc-members:
   :show-inheritance:

Physics Factors
===============

Pre-computed physics factor tables for correlation model evaluation.

.. automodule:: heterodyne.core.physics_factors
   :members:
   :undoc-members:
   :show-inheritance:
