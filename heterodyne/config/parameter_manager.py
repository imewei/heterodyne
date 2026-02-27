"""Parameter manager for heterodyne model optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_GROUPS
from heterodyne.config.parameter_space import ParameterSpace
from heterodyne.config.physics_validators import validate_time_integral_safety

if TYPE_CHECKING:
    import jax.numpy as jnp


@dataclass
class ParameterManager:
    """Manages parameter values, constraints, and transformations.
    
    Provides the bridge between configuration and optimization by:
    - Managing which parameters vary vs are fixed
    - Handling parameter transformations (e.g., bounded -> unbounded)
    - Constructing full parameter arrays from varying subsets
    - Validating parameter values against physics constraints
    """

    space: ParameterSpace = field(default_factory=ParameterSpace)

    @property
    def n_params(self) -> int:
        """Total number of model parameters (14)."""
        return len(ALL_PARAM_NAMES)

    @property
    def n_varying(self) -> int:
        """Number of parameters that vary in optimization."""
        return self.space.n_varying

    @property
    def varying_names(self) -> list[str]:
        """Names of varying parameters."""
        return self.space.varying_names

    @property
    def varying_indices(self) -> list[int]:
        """Indices of varying parameters in full array."""
        return [i for i, name in enumerate(ALL_PARAM_NAMES) if self.space.vary[name]]

    @property
    def fixed_indices(self) -> list[int]:
        """Indices of fixed parameters in full array."""
        return [i for i, name in enumerate(ALL_PARAM_NAMES) if not self.space.vary[name]]

    def get_initial_values(self) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Returns:
            Array of shape (n_varying,) with initial values for varying params
        """
        full = self.space.get_initial_array()
        return full[self.varying_indices]

    def get_full_values(self) -> np.ndarray:
        """Get all 14 parameter values.
        
        Returns:
            Array of shape (14,)
        """
        return self.space.get_initial_array()

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for varying parameters.
        
        Returns:
            (lower, upper) each of shape (n_varying,)
        """
        lower_full, upper_full = self.space.get_bounds_arrays()
        idx = self.varying_indices
        return lower_full[idx], upper_full[idx]

    def expand_varying_to_full(
        self,
        varying_params: np.ndarray | jnp.ndarray,
    ) -> np.ndarray:
        """Expand varying parameters to full 14-parameter array.
        
        Fixed parameters are filled from stored values.
        
        Args:
            varying_params: Array of shape (n_varying,)
            
        Returns:
            Array of shape (14,)
        """
        full = self.space.get_initial_array().copy()
        for i, idx in enumerate(self.varying_indices):
            full[idx] = float(varying_params[i])
        return full

    def extract_varying(self, full_params: np.ndarray | jnp.ndarray) -> np.ndarray:
        """Extract varying parameters from full array.
        
        Args:
            full_params: Array of shape (14,)
            
        Returns:
            Array of shape (n_varying,)
        """
        return np.array([full_params[i] for i in self.varying_indices])

    def update_values(self, params: np.ndarray | dict[str, float]) -> None:
        """Update stored parameter values.
        
        Args:
            params: Either array of shape (14,) or dict with param names
        """
        if isinstance(params, dict):
            self.space.update_from_dict(params)
        else:
            params_dict = self.space.array_to_dict(np.asarray(params))
            self.space.update_from_dict(params_dict)

    def get_parameter_dict(self) -> dict[str, float]:
        """Get current parameter values as dictionary."""
        return dict(self.space.values)

    def set_vary(self, name: str, vary: bool) -> None:
        """Set whether a parameter varies in optimization.
        
        Args:
            name: Parameter name
            vary: Whether to vary this parameter
        """
        if name not in ALL_PARAM_NAMES:
            raise ValueError(f"Unknown parameter: {name}")
        self.space.vary[name] = vary

    def set_bounds(self, name: str, lower: float, upper: float) -> None:
        """Set bounds for a parameter.
        
        Args:
            name: Parameter name
            lower: Lower bound
            upper: Upper bound
        """
        if name not in ALL_PARAM_NAMES:
            raise ValueError(f"Unknown parameter: {name}")
        self.space.bounds[name] = (lower, upper)

    def validate_physics(self, params: np.ndarray | None = None) -> list[str]:
        """Validate parameters against physics constraints.
        
        Args:
            params: Full parameter array, or None to use stored values
            
        Returns:
            List of violation messages (empty if valid)
        """
        if params is None:
            params = self.get_full_values()

        violations = []
        param_dict = self.space.array_to_dict(params)

        # Diffusion coefficients must be non-negative
        for prefix in ("D0_ref", "D0_sample"):
            if param_dict[prefix] < 0:
                violations.append(f"{prefix} must be non-negative")

        # Fraction parameters f0, f3 should be in [0, 1]
        for name in ("f0", "f3"):
            val = param_dict[name]
            if not (0 <= val <= 1):
                violations.append(f"{name}={val:.3f} should be in [0, 1]")

        # Alpha exponents typically in [-2, 2]
        for name in ("alpha_ref", "alpha_sample", "beta"):
            val = param_dict[name]
            if abs(val) > 2:
                violations.append(f"{name}={val:.3f} has unusual magnitude (>2)")

        # Time integral safety for alpha exponents
        for alpha_name in ("alpha_ref", "alpha_sample"):
            result = validate_time_integral_safety(
                alpha=param_dict[alpha_name],
                t_min=0.0,
                t_max=1e6,
            )
            violations.extend(result.errors)
            violations.extend(result.warnings)

        return violations

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParameterManager:
        """Create ParameterManager from configuration dictionary.
        
        Args:
            config: Full configuration dict
            
        Returns:
            Configured ParameterManager
        """
        space = ParameterSpace.from_config(config)
        return cls(space=space)

    def get_group_values(self, group: str) -> dict[str, float]:
        """Get parameter values for a specific group.
        
        Args:
            group: Group name ('reference', 'sample', 'velocity', 'fraction', 'angle')
            
        Returns:
            Dict mapping parameter names to values
        """
        if group not in PARAM_GROUPS:
            raise ValueError(f"Unknown group: {group}")
        return {name: self.space.values[name] for name in PARAM_GROUPS[group]}
