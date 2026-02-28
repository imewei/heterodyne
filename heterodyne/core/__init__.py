"""Core physics engine for heterodyne XPCS analysis."""

from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_g1_transport,
    compute_transport_integral_matrix,
    compute_velocity_integral_matrix,
)
from heterodyne.core.models import HeterodyneModelBase, TwoComponentModel
from heterodyne.core.physics import PARAMETER_BOUNDS, PhysicsConstants
from heterodyne.core.physics_factors import create_physics_factors
from heterodyne.core.theory import (
    compute_fraction,
    compute_time_integral_matrix,
    compute_transport_coefficient,
)
from heterodyne.core.theory import (
    compute_transport_integral_matrix as compute_transport_integral_matrix_theory,
)

__all__ = [
    # Main model
    "HeterodyneModel",
    "HeterodyneModelBase",
    "TwoComponentModel",
    # JAX functions
    "compute_c2_heterodyne",
    "compute_g1_transport",
    "compute_transport_integral_matrix",
    "compute_velocity_integral_matrix",
    # Theory functions
    "compute_transport_coefficient",
    "compute_transport_integral_matrix_theory",
    "compute_fraction",
    "compute_time_integral_matrix",
    # Physics constants
    "PhysicsConstants",
    "PARAMETER_BOUNDS",
    "create_physics_factors",
]
