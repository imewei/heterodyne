"""Parameter space definition with prior distributions for Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

if TYPE_CHECKING:
    import jax.numpy as jnp


class PriorType(Enum):
    """Available prior distribution types."""
    
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    HALFNORMAL = "halfnormal"
    EXPONENTIAL = "exponential"


@dataclass
class PriorDistribution:
    """Prior distribution specification for a parameter."""
    
    prior_type: PriorType
    params: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def uniform(cls, low: float, high: float) -> PriorDistribution:
        """Create uniform prior."""
        return cls(PriorType.UNIFORM, {"low": low, "high": high})
    
    @classmethod
    def normal(cls, loc: float, scale: float) -> PriorDistribution:
        """Create normal (Gaussian) prior."""
        return cls(PriorType.NORMAL, {"loc": loc, "scale": scale})
    
    @classmethod
    def lognormal(cls, loc: float, scale: float) -> PriorDistribution:
        """Create log-normal prior (for positive parameters)."""
        return cls(PriorType.LOGNORMAL, {"loc": loc, "scale": scale})
    
    @classmethod
    def halfnormal(cls, scale: float) -> PriorDistribution:
        """Create half-normal prior (for positive parameters)."""
        return cls(PriorType.HALFNORMAL, {"scale": scale})
    
    def to_numpyro(self, name: str) -> Any:
        """Convert to NumPyro distribution.
        
        Args:
            name: Parameter name for the distribution
            
        Returns:
            NumPyro distribution object
        """
        import numpyro.distributions as dist
        
        if self.prior_type == PriorType.UNIFORM:
            return dist.Uniform(self.params["low"], self.params["high"])
        elif self.prior_type == PriorType.NORMAL:
            return dist.Normal(self.params["loc"], self.params["scale"])
        elif self.prior_type == PriorType.LOGNORMAL:
            return dist.LogNormal(self.params["loc"], self.params["scale"])
        elif self.prior_type == PriorType.HALFNORMAL:
            return dist.HalfNormal(self.params["scale"])
        elif self.prior_type == PriorType.EXPONENTIAL:
            return dist.Exponential(self.params.get("rate", 1.0))
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")


@dataclass
class ParameterSpace:
    """Complete parameter space for heterodyne model optimization.
    
    Manages parameter values, bounds, vary flags, and priors.
    """
    
    values: dict[str, float] = field(default_factory=dict)
    vary: dict[str, bool] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    priors: dict[str, PriorDistribution] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize with defaults from registry."""
        for name in ALL_PARAM_NAMES:
            info = DEFAULT_REGISTRY[name]
            if name not in self.values:
                self.values[name] = info.default
            if name not in self.vary:
                self.vary[name] = info.vary_default
            if name not in self.bounds:
                self.bounds[name] = (info.min_bound, info.max_bound)
            if name not in self.priors:
                # Default: uniform prior within bounds
                self.priors[name] = PriorDistribution.uniform(
                    info.min_bound, info.max_bound
                )
    
    @property
    def n_total(self) -> int:
        """Total number of parameters."""
        return len(ALL_PARAM_NAMES)
    
    @property
    def n_varying(self) -> int:
        """Number of parameters that vary in optimization."""
        return sum(1 for v in self.vary.values() if v)
    
    @property
    def varying_names(self) -> list[str]:
        """Names of parameters that vary."""
        return [name for name in ALL_PARAM_NAMES if self.vary.get(name, False)]
    
    @property
    def fixed_names(self) -> list[str]:
        """Names of parameters that are fixed."""
        return [name for name in ALL_PARAM_NAMES if not self.vary.get(name, True)]
    
    def get_initial_array(self) -> np.ndarray:
        """Get initial values as numpy array in canonical order.
        
        Returns:
            Array of shape (14,) with parameter values
        """
        return np.array([self.values[name] for name in ALL_PARAM_NAMES])
    
    def get_bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds as numpy arrays.
        
        Returns:
            (lower_bounds, upper_bounds) each of shape (14,)
        """
        lower = np.array([self.bounds[name][0] for name in ALL_PARAM_NAMES])
        upper = np.array([self.bounds[name][1] for name in ALL_PARAM_NAMES])
        return lower, upper
    
    def get_vary_mask(self) -> np.ndarray:
        """Get boolean mask for varying parameters.
        
        Returns:
            Boolean array of shape (14,)
        """
        return np.array([self.vary[name] for name in ALL_PARAM_NAMES])
    
    def array_to_dict(self, arr: np.ndarray | jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary.
        
        Args:
            arr: Array of shape (14,)
            
        Returns:
            Dict mapping parameter names to values
        """
        return {name: float(arr[i]) for i, name in enumerate(ALL_PARAM_NAMES)}
    
    def update_from_dict(self, params: dict[str, float]) -> None:
        """Update parameter values from dictionary.
        
        Args:
            params: Dict with parameter names as keys
        """
        for name, value in params.items():
            if name in self.values:
                self.values[name] = value
    
    def validate(self) -> list[str]:
        """Validate parameter space configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for name in ALL_PARAM_NAMES:
            value = self.values.get(name)
            bounds = self.bounds.get(name)
            
            if value is None:
                errors.append(f"Missing value for {name}")
                continue
            
            if bounds is None:
                errors.append(f"Missing bounds for {name}")
                continue
            
            low, high = bounds
            if not (low <= value <= high):
                errors.append(
                    f"{name}={value} outside bounds [{low}, {high}]"
                )
        
        return errors
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParameterSpace:
        """Create ParameterSpace from configuration dictionary.
        
        Args:
            config: Config dict with 'parameters' section
            
        Returns:
            Configured ParameterSpace
        """
        space = cls()
        
        params_config = config.get("parameters", {})
        
        # Process each parameter group
        group_map = {
            "reference": ["D0_ref", "alpha_ref", "D_offset_ref"],
            "sample": ["D0_sample", "alpha_sample", "D_offset_sample"],
            "velocity": ["v0", "beta", "v_offset"],
            "fraction": ["f0", "f1", "f2", "f3"],
            "angle": ["phi0"],
        }
        
        # Map config names to internal names
        config_to_internal = {
            "D0": lambda g: f"D0_{g}" if g in ("ref", "sample") else "D0_sample",
            "alpha": lambda g: f"alpha_{g}" if g in ("ref", "sample") else "alpha_sample",
            "D_offset": lambda g: f"D_offset_{g}" if g in ("ref", "sample") else "D_offset_sample",
        }
        
        for group_name, param_names in group_map.items():
            group_config = params_config.get(group_name, {})
            
            for param_name in param_names:
                # Try to find matching config key
                config_key = None
                for ck in group_config:
                    if ck in param_name or param_name.startswith(ck):
                        config_key = ck
                        break
                
                # Direct match
                if param_name in group_config:
                    config_key = param_name
                
                if config_key and config_key in group_config:
                    pconfig = group_config[config_key]
                    if isinstance(pconfig, dict):
                        if "value" in pconfig:
                            space.values[param_name] = pconfig["value"]
                        if "min" in pconfig and "max" in pconfig:
                            space.bounds[param_name] = (pconfig["min"], pconfig["max"])
                        if "vary" in pconfig:
                            space.vary[param_name] = pconfig["vary"]
        
        return space
