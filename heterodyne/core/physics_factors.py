"""Pre-computed physics factors for efficient correlation computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    pass


@dataclass
class PhysicsFactors:
    """Pre-computed factors that don't depend on fit parameters.
    
    These are computed once from experimental setup and reused across
    all optimization iterations for efficiency.
    """
    
    # Time arrays
    t: jnp.ndarray  # Time array, shape (N,)
    t_grid_1: jnp.ndarray  # Meshgrid t1, shape (N, N)
    t_grid_2: jnp.ndarray  # Meshgrid t2, shape (N, N)
    
    # Scattering
    q: float  # Wavevector magnitude
    q_squared: float  # q²
    
    # Temporal
    dt: float  # Time step
    n_times: int  # Number of time points
    
    # Geometry
    phi_angle: float  # Detector phi angle (degrees)
    
    def __post_init__(self) -> None:
        """Validate factors."""
        if self.q <= 0:
            raise ValueError(f"q must be positive, got {self.q}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
    
    @property
    def time_extent(self) -> float:
        """Total time span."""
        return float(self.t[-1] - self.t[0])
    
    def get_q_cosine(self, phi0: float = 0.0) -> float:
        """Get q * cos(phi_total) for cross-term phase.
        
        Args:
            phi0: Additional angle from fit parameters
            
        Returns:
            q * cos(phi_angle + phi0)
        """
        total_phi_rad = jnp.deg2rad(self.phi_angle + phi0)
        return self.q * float(jnp.cos(total_phi_rad))


def create_physics_factors(
    n_times: int,
    dt: float,
    q: float,
    phi_angle: float = 0.0,
    t_start: float = 0.0,
) -> PhysicsFactors:
    """Create physics factors from experimental parameters.
    
    Args:
        n_times: Number of time points
        dt: Time step
        q: Scattering wavevector magnitude
        phi_angle: Detector phi angle (degrees)
        t_start: Starting time (default 0)
        
    Returns:
        PhysicsFactors instance
    """
    # Create time array
    t = jnp.arange(n_times) * dt + t_start
    
    # Create meshgrids for 2D correlation
    t_grid_1, t_grid_2 = jnp.meshgrid(t, t, indexing="ij")
    
    return PhysicsFactors(
        t=t,
        t_grid_1=t_grid_1,
        t_grid_2=t_grid_2,
        q=float(q),
        q_squared=float(q * q),
        dt=float(dt),
        n_times=n_times,
        phi_angle=float(phi_angle),
    )


def create_physics_factors_from_config(config: dict) -> PhysicsFactors:
    """Create physics factors from configuration dictionary.
    
    Args:
        config: Configuration with 'temporal' and 'scattering' sections
        
    Returns:
        PhysicsFactors instance
    """
    temporal = config.get("temporal", {})
    scattering = config.get("scattering", {})
    
    return create_physics_factors(
        n_times=int(temporal.get("time_length", 1000)),
        dt=float(temporal.get("dt", 1.0)),
        q=float(scattering.get("wavevector_q", 0.01)),
        phi_angle=0.0,  # Set per-fit
        t_start=float(temporal.get("t_start", 0.0)),
    )


@dataclass
class CachedMatrices:
    """Cached matrices that depend only on time grid.
    
    These are expensive to recompute and don't change during fitting.
    """
    
    # Time difference matrix: |t1 - t2|
    time_diff: jnp.ndarray
    
    # Age matrix: (t1 + t2) / 2
    mean_time: jnp.ndarray
    
    # Indices for upper/lower triangular
    triu_indices: tuple[jnp.ndarray, jnp.ndarray]
    tril_indices: tuple[jnp.ndarray, jnp.ndarray]


def create_cached_matrices(factors: PhysicsFactors) -> CachedMatrices:
    """Create cached matrices from physics factors.
    
    Args:
        factors: PhysicsFactors instance
        
    Returns:
        CachedMatrices instance
    """
    t1 = factors.t_grid_1
    t2 = factors.t_grid_2
    n = factors.n_times
    
    return CachedMatrices(
        time_diff=jnp.abs(t1 - t2),
        mean_time=(t1 + t2) / 2,
        triu_indices=jnp.triu_indices(n),
        tril_indices=jnp.tril_indices(n),
    )
