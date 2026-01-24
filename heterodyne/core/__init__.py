"""Core physics engine for heterodyne XPCS analysis."""

from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.core.models import HeterodyneModelBase, TwoComponentModel
from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_g1_transport,
    compute_velocity_integral_matrix,
)
from heterodyne.core.theory import (
    compute_transport_coefficient,
    compute_fraction,
    compute_time_integral_matrix,
)
from heterodyne.core.physics import PhysicsConstants, PARAMETER_BOUNDS
from heterodyne.core.physics_factors import create_physics_factors

__all__ = [
    # Main model
    "HeterodyneModel",
    "HeterodyneModelBase",
    "TwoComponentModel",
    # JAX functions
    "compute_c2_heterodyne",
    "compute_g1_transport",
    "compute_velocity_integral_matrix",
    # Theory functions
    "compute_transport_coefficient",
    "compute_fraction",
    "compute_time_integral_matrix",
    # Physics constants
    "PhysicsConstants",
    "PARAMETER_BOUNDS",
    "create_physics_factors",
]
