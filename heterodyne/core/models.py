"""Model class hierarchy for heterodyne correlation analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.core.jax_backend import compute_c2_heterodyne

if TYPE_CHECKING:
    pass


class HeterodyneModelBase(ABC):
    """Abstract base class for heterodyne models."""
    
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of model parameters."""
        ...
    
    @property
    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in order."""
        ...
    
    @abstractmethod
    def compute_correlation(
        self,
        params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi: float,
    ) -> jnp.ndarray:
        """Compute model correlation matrix.
        
        Args:
            params: Parameter array
            t: Time array
            q: Wavevector
            dt: Time step
            phi: Angle
            
        Returns:
            Correlation matrix
        """
        ...
    
    @abstractmethod
    def get_default_params(self) -> np.ndarray:
        """Get default parameter values."""
        ...


@dataclass
class TwoComponentModel(HeterodyneModelBase):
    """Two-component heterodyne correlation model.
    
    Implements the 14-parameter model:
    - Reference transport (3): D0_ref, alpha_ref, D_offset_ref
    - Sample transport (3): D0_sample, alpha_sample, D_offset_sample
    - Velocity (3): v0, beta, v_offset
    - Fraction (4): f0, f1, f2, f3
    - Angle (1): phi0
    """
    
    _defaults: dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Set default parameter values."""
        if not self._defaults:
            self._defaults = {
                "D0_ref": 1.0,
                "alpha_ref": 1.0,
                "D_offset_ref": 0.0,
                "D0_sample": 1.0,
                "alpha_sample": 1.0,
                "D_offset_sample": 0.0,
                "v0": 0.0,
                "beta": 0.0,
                "v_offset": 0.0,
                "f0": 0.5,
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
                "phi0": 0.0,
            }
    
    @property
    def n_params(self) -> int:
        """Number of parameters (14)."""
        return 14
    
    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in canonical order."""
        return ALL_PARAM_NAMES
    
    def compute_correlation(
        self,
        params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi: float,
    ) -> jnp.ndarray:
        """Compute two-time heterodyne correlation.
        
        Args:
            params: Parameter array, shape (14,)
            t: Time array
            q: Scattering wavevector
            dt: Time step
            phi: Detector phi angle
            
        Returns:
            Correlation matrix c2(t1, t2), shape (N, N)
        """
        return compute_c2_heterodyne(params, t, q, dt, phi)
    
    def get_default_params(self) -> np.ndarray:
        """Get default parameter values as array."""
        return np.array([self._defaults[name] for name in ALL_PARAM_NAMES])
    
    def params_to_dict(self, params: np.ndarray | jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary.
        
        Args:
            params: Parameter array, shape (14,)
            
        Returns:
            Dict mapping names to values
        """
        return {name: float(params[i]) for i, name in enumerate(ALL_PARAM_NAMES)}
    
    def dict_to_params(self, param_dict: dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array.
        
        Args:
            param_dict: Dict with parameter names as keys
            
        Returns:
            Parameter array, shape (14,)
        """
        return np.array([param_dict.get(name, self._defaults[name]) for name in ALL_PARAM_NAMES])
    
    def compute_g1_reference(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
        q: float,
    ) -> jnp.ndarray:
        """Compute reference g1 correlation only.
        
        Args:
            params: Full parameter array
            t: Time array
            q: Wavevector
            
        Returns:
            g1_ref array
        """
        D0, alpha, offset = params[0], params[1], params[2]
        t_safe = jnp.maximum(t, 1e-10)
        J = D0 * jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0) + offset
        J = jnp.maximum(J, 0.0)
        return jnp.exp(-q * q * J)

    def compute_g1_sample(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
        q: float,
    ) -> jnp.ndarray:
        """Compute sample g1 correlation only.
        
        Args:
            params: Full parameter array
            t: Time array
            q: Wavevector
            
        Returns:
            g1_sample array
        """
        D0, alpha, offset = params[3], params[4], params[5]
        t_safe = jnp.maximum(t, 1e-10)
        J = D0 * jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0) + offset
        J = jnp.maximum(J, 0.0)
        return jnp.exp(-q * q * J)
    
    def compute_fraction(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute sample fraction only.
        
        Args:
            params: Full parameter array
            t: Time array
            
        Returns:
            f_sample array in [0, 1]
        """
        f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
        exponent = jnp.clip(f1 * (t - f2), -100, 100)
        return jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)


# Default model instance
DEFAULT_MODEL = TwoComponentModel()
