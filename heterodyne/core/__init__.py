"""Core physics engine for heterodyne XPCS analysis."""

from heterodyne.core.diagonal_correction import (
    apply_diagonal_correction,
    compute_diagonal_mask,
    compute_weights_excluding_diagonal,
    estimate_diagonal_excess,
)
from heterodyne.core.fitting import ScaledFittingEngine
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_g1_transport,
    compute_transport_integral_matrix,
    compute_velocity_integral_matrix,
)
from heterodyne.core.models import HeterodyneModelBase, TwoComponentModel
from heterodyne.core.physics import PARAMETER_BOUNDS, PhysicsConstants
from heterodyne.core.physics_cmc import (
    ShardGrid,
    compute_c2_elementwise,
    compute_log_likelihood,
    compute_log_likelihood_elementwise,
    compute_posterior_predictive,
    compute_sharded_log_likelihood,
    compute_sharded_log_likelihood_elementwise,
    precompute_shard_grid,
    precompute_shard_grid_from_matrix,
    prepare_shards_elementwise,
)
from heterodyne.core.physics_factors import create_physics_factors
from heterodyne.core.physics_nlsq import (
    compute_flat_residuals,
    compute_nlsq_jacobian,
    make_residual_fn,
    make_varying_residual_fn,
)
from heterodyne.core.physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_time_integral_matrix,
    safe_divide,
    safe_exp,
    safe_log,
    safe_power,
    safe_sinc,
    safe_sqrt,
    smooth_abs,
    symmetrize,
    trapezoid_cumsum,
)
from heterodyne.core.scaling_utils import PerAngleScaling, ScalingConfig
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
    # Fitting engine
    "ScaledFittingEngine",
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
    # NLSQ-optimized
    "compute_flat_residuals",
    "make_residual_fn",
    "make_varying_residual_fn",
    "compute_nlsq_jacobian",
    # CMC-optimized (element-wise + meshgrid)
    "ShardGrid",
    "precompute_shard_grid",
    "precompute_shard_grid_from_matrix",
    "compute_c2_elementwise",
    "compute_log_likelihood",
    "compute_log_likelihood_elementwise",
    "compute_sharded_log_likelihood",
    "compute_sharded_log_likelihood_elementwise",
    "compute_posterior_predictive",
    "prepare_shards_elementwise",
    # Diagonal correction
    "compute_diagonal_mask",
    "apply_diagonal_correction",
    "estimate_diagonal_excess",
    "compute_weights_excluding_diagonal",
    # Per-angle scaling
    "PerAngleScaling",
    "ScalingConfig",
    # Physics utilities (shared primitives)
    "trapezoid_cumsum",
    "create_time_integral_matrix",
    "smooth_abs",
    "compute_transport_rate",
    "compute_velocity_rate",
    "safe_sinc",
    "safe_exp",
    "safe_power",
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "symmetrize",
    # Physics constants
    "PhysicsConstants",
    "PARAMETER_BOUNDS",
    "create_physics_factors",
]
