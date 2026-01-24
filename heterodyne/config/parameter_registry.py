"""Parameter registry with metadata and bounds for heterodyne model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_GROUPS

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata for a single model parameter."""
    
    name: str
    default: float
    min_bound: float
    max_bound: float
    description: str
    unit: str = ""
    group: str = ""
    vary_default: bool = True
    
    def validate_value(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.min_bound <= value <= self.max_bound
    
    def clip_value(self, value: float) -> float:
        """Clip value to bounds."""
        return np.clip(value, self.min_bound, self.max_bound)


@dataclass
class ParameterRegistry:
    """Registry of all heterodyne model parameters with metadata."""
    
    _parameters: dict[str, ParameterInfo] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize with default heterodyne parameters."""
        if not self._parameters:
            self._parameters = _create_default_registry()
    
    def __getitem__(self, name: str) -> ParameterInfo:
        """Get parameter info by name."""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[name]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names in canonical order."""
        for name in ALL_PARAM_NAMES:
            if name in self._parameters:
                yield name
    
    def __len__(self) -> int:
        return len(self._parameters)
    
    def get_defaults(self) -> dict[str, float]:
        """Get default values for all parameters."""
        return {name: self._parameters[name].default for name in self}
    
    def get_bounds(self) -> tuple[list[float], list[float]]:
        """Get (lower_bounds, upper_bounds) as lists."""
        lower = [self._parameters[name].min_bound for name in self]
        upper = [self._parameters[name].max_bound for name in self]
        return lower, upper
    
    def get_group(self, group_name: str) -> list[ParameterInfo]:
        """Get all parameters in a group."""
        if group_name not in PARAM_GROUPS:
            raise KeyError(f"Unknown group: {group_name}")
        return [self._parameters[name] for name in PARAM_GROUPS[group_name]]
    
    def get_varying_indices(self, vary_flags: dict[str, bool]) -> list[int]:
        """Get indices of parameters that vary in optimization."""
        indices = []
        for i, name in enumerate(self):
            if vary_flags.get(name, self._parameters[name].vary_default):
                indices.append(i)
        return indices


def _create_default_registry() -> dict[str, ParameterInfo]:
    """Create default parameter registry for heterodyne model."""
    params: dict[str, ParameterInfo] = {}
    
    # Reference transport: J_r(t) = D0_ref * t^alpha_ref + D_offset_ref
    params["D0_ref"] = ParameterInfo(
        name="D0_ref",
        default=1.0,
        min_bound=0.0,
        max_bound=1e6,
        description="Reference diffusion coefficient prefactor",
        unit="nm²/s^α",
        group="reference",
        vary_default=True,
    )
    params["alpha_ref"] = ParameterInfo(
        name="alpha_ref",
        default=1.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Reference diffusion exponent (1=normal, <1=subdiffusive)",
        unit="",
        group="reference",
        vary_default=True,
    )
    params["D_offset_ref"] = ParameterInfo(
        name="D_offset_ref",
        default=0.0,
        min_bound=-1e5,
        max_bound=1e5,
        description="Reference transport offset",
        unit="nm²",
        group="reference",
        vary_default=False,
    )
    
    # Sample transport: J_s(t) = D0_sample * t^alpha_sample + D_offset_sample
    params["D0_sample"] = ParameterInfo(
        name="D0_sample",
        default=1.0,
        min_bound=0.0,
        max_bound=1e6,
        description="Sample diffusion coefficient prefactor",
        unit="nm²/s^α",
        group="sample",
        vary_default=True,
    )
    params["alpha_sample"] = ParameterInfo(
        name="alpha_sample",
        default=1.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Sample diffusion exponent",
        unit="",
        group="sample",
        vary_default=True,
    )
    params["D_offset_sample"] = ParameterInfo(
        name="D_offset_sample",
        default=0.0,
        min_bound=-1e5,
        max_bound=1e5,
        description="Sample transport offset",
        unit="nm²",
        group="sample",
        vary_default=False,
    )
    
    # Velocity: v(t) = v0 * t^beta + v_offset
    params["v0"] = ParameterInfo(
        name="v0",
        default=0.0,
        min_bound=-1e4,
        max_bound=1e4,
        description="Velocity prefactor",
        unit="nm/s^β",
        group="velocity",
        vary_default=True,
    )
    params["beta"] = ParameterInfo(
        name="beta",
        default=0.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Velocity exponent",
        unit="",
        group="velocity",
        vary_default=False,
    )
    params["v_offset"] = ParameterInfo(
        name="v_offset",
        default=0.0,
        min_bound=-1e4,
        max_bound=1e4,
        description="Velocity offset",
        unit="nm/s",
        group="velocity",
        vary_default=False,
    )
    
    # Fraction: f_s(t) = f0 * exp(f1 * (t - f2)) + f3
    params["f0"] = ParameterInfo(
        name="f0",
        default=0.5,
        min_bound=0.0,
        max_bound=1.0,
        description="Fraction amplitude",
        unit="",
        group="fraction",
        vary_default=True,
    )
    params["f1"] = ParameterInfo(
        name="f1",
        default=0.0,
        min_bound=-10.0,
        max_bound=10.0,
        description="Fraction exponential rate",
        unit="1/s",
        group="fraction",
        vary_default=False,
    )
    params["f2"] = ParameterInfo(
        name="f2",
        default=0.0,
        min_bound=-1e4,
        max_bound=1e4,
        description="Fraction time shift",
        unit="s",
        group="fraction",
        vary_default=False,
    )
    params["f3"] = ParameterInfo(
        name="f3",
        default=0.0,
        min_bound=0.0,
        max_bound=1.0,
        description="Fraction baseline offset",
        unit="",
        group="fraction",
        vary_default=False,
    )
    
    # Flow angle
    params["phi0"] = ParameterInfo(
        name="phi0",
        default=0.0,
        min_bound=-360.0,
        max_bound=360.0,
        description="Flow angle relative to q-vector",
        unit="degrees",
        group="angle",
        vary_default=True,
    )
    
    return params


# Module-level default registry instance
DEFAULT_REGISTRY = ParameterRegistry()
