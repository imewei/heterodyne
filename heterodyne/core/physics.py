"""Physical constants and parameter bounds for heterodyne model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical constants for XPCS scattering analysis.
    
    All values in SI base units unless otherwise noted.
    """
    
    # Boltzmann constant (J/K)
    k_B: ClassVar[float] = 1.380649e-23
    
    # Planck constant (J·s)
    h: ClassVar[float] = 6.62607015e-34
    
    # Speed of light (m/s)
    c: ClassVar[float] = 299792458.0
    
    # X-ray wavelengths (nm) - common energies
    WAVELENGTH_8KEV: ClassVar[float] = 0.155  # nm
    WAVELENGTH_10KEV: ClassVar[float] = 0.124  # nm
    WAVELENGTH_12KEV: ClassVar[float] = 0.103  # nm


# Default parameter bounds for heterodyne model
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    # Reference transport
    "D0_ref": (0.0, 1e6),
    "alpha_ref": (-2.0, 2.0),
    "D_offset_ref": (-1e5, 1e5),
    # Sample transport
    "D0_sample": (0.0, 1e6),
    "alpha_sample": (-2.0, 2.0),
    "D_offset_sample": (-1e5, 1e5),
    # Velocity
    "v0": (-1e4, 1e4),
    "beta": (-2.0, 2.0),
    "v_offset": (-1e4, 1e4),
    # Fraction
    "f0": (0.0, 1.0),
    "f1": (-10.0, 10.0),
    "f2": (-1e4, 1e4),
    "f3": (0.0, 1.0),
    # Angle
    "phi0": (-360.0, 360.0),
}


def get_default_bounds_array() -> tuple[np.ndarray, np.ndarray]:
    """Get default bounds as arrays in canonical parameter order.
    
    Returns:
        (lower_bounds, upper_bounds) each of shape (14,)
    """
    from heterodyne.config.parameter_names import ALL_PARAM_NAMES
    
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in ALL_PARAM_NAMES])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in ALL_PARAM_NAMES])
    return lower, upper


@dataclass(frozen=True)
class TransportPhysics:
    """Physical interpretation of transport parameters.
    
    Transport coefficient: J(t) = D0 * t^alpha + offset
    
    Physical regimes based on alpha:
    - alpha = 1.0: Normal (Brownian) diffusion
    - alpha < 1.0: Subdiffusion (crowded/constrained)
    - alpha > 1.0: Superdiffusion (active/directed)
    - alpha = 2.0: Ballistic motion
    """
    
    # Alpha value regimes
    NORMAL_DIFFUSION: ClassVar[float] = 1.0
    BALLISTIC: ClassVar[float] = 2.0
    
    @staticmethod
    def interpret_alpha(alpha: float) -> str:
        """Interpret alpha value physically.
        
        Args:
            alpha: Diffusion exponent
            
        Returns:
            Physical interpretation string
        """
        if abs(alpha - 1.0) < 0.05:
            return "normal diffusion"
        elif alpha < 0.5:
            return "strongly subdiffusive"
        elif alpha < 1.0:
            return "subdiffusive"
        elif alpha < 1.5:
            return "weakly superdiffusive"
        elif alpha < 2.0:
            return "superdiffusive"
        else:
            return "ballistic/directed"
    
    @staticmethod
    def diffusion_coefficient(D0: float, alpha: float, t: float = 1.0) -> float:
        """Compute effective diffusion coefficient at time t.
        
        For J(t) = D0 * t^alpha, the effective D is:
        D_eff = dJ/dt = D0 * alpha * t^(alpha-1)
        
        Args:
            D0: Transport prefactor
            alpha: Transport exponent
            t: Time point (default 1.0)
            
        Returns:
            Effective diffusion coefficient
        """
        if t <= 0:
            return 0.0
        return D0 * alpha * (t ** (alpha - 1))
