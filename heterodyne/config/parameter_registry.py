"""Parameter registry with metadata and bounds for heterodyne model.

All length units use Å (angstroms) for consistency with practical XPCS convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.config.parameter_names import (
    ALL_PARAM_NAMES_WITH_SCALING,
    PARAM_GROUPS,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata for a single model parameter.

    Attributes:
        name: Parameter name matching canonical order in parameter_names.py.
        default: Default value (must be within [min_bound, max_bound]).
        min_bound: Lower bound for optimization.
        max_bound: Upper bound for optimization.
        description: Human-readable description.
        unit: Physical unit string (e.g. "Å²/s^α").
        group: Parameter group name.
        vary_default: Whether this parameter varies by default in optimization.
        log_space: If True, MCMC samplers should reparameterize in log-space.
        prior_mean: Center of the default Bayesian prior (None = midpoint of bounds).
        prior_std: Width of the default Bayesian prior (None = half-range of bounds).
        is_scaling: If True, this parameter participates in per-angle expansion.
        is_physical: If True, this is a physical model parameter (not scaling).
        is_flow: If True, this parameter is related to flow/velocity.
    """

    name: str
    default: float
    min_bound: float
    max_bound: float
    description: str
    unit: str = ""
    group: str = ""
    vary_default: bool = True
    log_space: bool = False
    prior_mean: float | None = None
    prior_std: float | None = None
    is_scaling: bool = False
    is_physical: bool = True
    is_flow: bool = False

    def validate_value(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.min_bound <= value <= self.max_bound

    def clip_value(self, value: float) -> float:
        """Clip value to bounds."""
        return float(np.clip(value, self.min_bound, self.max_bound))


@dataclass
class ParameterRegistry:
    """Registry of all heterodyne model parameters with metadata."""

    _parameters: Mapping[str, ParameterInfo] = field(
        default_factory=lambda: MappingProxyType(_create_default_registry())
    )

    def __post_init__(self) -> None:
        """Ensure parameters are immutable."""
        if not isinstance(self._parameters, MappingProxyType):
            object.__setattr__(
                self, "_parameters", MappingProxyType(dict(self._parameters))
            )

    def __getitem__(self, name: str) -> ParameterInfo:
        """Get parameter info by name."""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names in canonical order (14 physics + 2 scaling)."""
        for name in ALL_PARAM_NAMES_WITH_SCALING:
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

    def get_log_space_names(self) -> list[str]:
        """Get names of parameters that should be sampled in log-space."""
        return [
            name
            for name in self
            if self._parameters[name].log_space
        ]

    def get_scaling_names(self) -> list[str]:
        """Get names of per-angle scaling parameters."""
        return [
            name
            for name in self._parameters
            if self._parameters[name].is_scaling
        ]


def _create_default_registry() -> dict[str, ParameterInfo]:
    """Create default parameter registry for heterodyne model.

    Physical bounds are authoritative values from XPCS convention.
    All length units in Å (angstroms).
    """
    params: dict[str, ParameterInfo] = {}

    # Reference transport: J_r(t) = D0_ref * t^alpha_ref + D_offset_ref
    params["D0_ref"] = ParameterInfo(
        name="D0_ref",
        default=1e4,
        min_bound=100.0,
        max_bound=1e6,
        description="Reference diffusion coefficient prefactor",
        unit="Å²/s^α",
        group="reference",
        vary_default=True,
        log_space=True,
        prior_mean=1e4,
        prior_std=5e3,
        is_physical=True,
    )
    params["alpha_ref"] = ParameterInfo(
        name="alpha_ref",
        default=0.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Reference transport exponent (1=diffusive, <1=subdiffusive)",
        unit="",
        group="reference",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1.0,
        is_physical=True,
    )
    params["D_offset_ref"] = ParameterInfo(
        name="D_offset_ref",
        default=0.0,
        min_bound=-1e5,
        max_bound=1e5,
        description="Reference transport rate offset (intentionally wide; clamped at runtime)",
        unit="Å²",
        group="reference",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1e3,
        is_physical=True,
    )

    # Sample transport: J_s(t) = D0_sample * t^alpha_sample + D_offset_sample
    params["D0_sample"] = ParameterInfo(
        name="D0_sample",
        default=1e4,
        min_bound=100.0,
        max_bound=1e6,
        description="Sample diffusion coefficient prefactor",
        unit="Å²/s^α",
        group="sample",
        vary_default=True,
        log_space=True,
        prior_mean=1e4,
        prior_std=5e3,
        is_physical=True,
    )
    params["alpha_sample"] = ParameterInfo(
        name="alpha_sample",
        default=0.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Sample transport exponent (1=diffusive, <1=subdiffusive)",
        unit="",
        group="sample",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1.0,
        is_physical=True,
    )
    params["D_offset_sample"] = ParameterInfo(
        name="D_offset_sample",
        default=0.0,
        min_bound=-1e5,
        max_bound=1e5,
        description="Sample transport rate offset (intentionally wide; clamped at runtime)",
        unit="Å²",
        group="sample",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1e3,
        is_physical=True,
    )

    # Velocity: v(t) = v0 * t^beta + v_offset
    params["v0"] = ParameterInfo(
        name="v0",
        default=1e3,
        min_bound=1e-6,
        max_bound=1e4,
        description="Velocity prefactor (non-negative magnitude)",
        unit="Å/s^β",
        group="velocity",
        vary_default=True,
        log_space=True,
        prior_mean=1e3,
        prior_std=500.0,
        is_physical=True,
        is_flow=True,
    )
    params["beta"] = ParameterInfo(
        name="beta",
        default=0.0,
        min_bound=-2.0,
        max_bound=2.0,
        description="Velocity exponent (0=constant, <0=deceleration)",
        unit="",
        group="velocity",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1.0,
        is_physical=True,
        is_flow=True,
    )
    params["v_offset"] = ParameterInfo(
        name="v_offset",
        default=0.0,
        min_bound=-100.0,
        max_bound=100.0,
        description="Velocity offset (allows negative for direction reversal)",
        unit="Å/s",
        group="velocity",
        vary_default=True,
        prior_mean=0.0,
        prior_std=25.0,
        is_physical=True,
        is_flow=True,
    )

    # Fraction: f_s(t) = f0 * exp(f1 * (t - f2)) + f3
    params["f0"] = ParameterInfo(
        name="f0",
        default=0.5,
        min_bound=0.0,
        max_bound=1.0,
        description="Sample fraction amplitude (field amplitude, not intensity fraction)",
        unit="",
        group="fraction",
        vary_default=True,
        prior_mean=0.5,
        prior_std=0.25,
        is_physical=True,
    )
    params["f1"] = ParameterInfo(
        name="f1",
        default=0.0,
        min_bound=-10.0,
        max_bound=10.0,
        description="Fraction exponential rate",
        unit="1/s",
        group="fraction",
        vary_default=True,
        prior_mean=0.0,
        prior_std=5.0,
        is_physical=True,
    )
    params["f2"] = ParameterInfo(
        name="f2",
        default=0.0,
        min_bound=-1e4,
        max_bound=1e4,
        description="Fraction time shift",
        unit="s",
        group="fraction",
        vary_default=True,
        prior_mean=0.0,
        prior_std=1e3,
        is_physical=True,
    )
    params["f3"] = ParameterInfo(
        name="f3",
        default=0.0,
        min_bound=0.0,
        max_bound=1.0,
        description="Fraction baseline offset",
        unit="",
        group="fraction",
        vary_default=True,
        prior_mean=0.0,
        prior_std=0.5,
        is_physical=True,
    )

    # Flow angle
    params["phi0"] = ParameterInfo(
        name="phi0",
        default=0.0,
        min_bound=-10.0,
        max_bound=10.0,
        description="Flow angle relative to q-vector (tightened per XPCS convention)",
        unit="degrees",
        group="angle",
        vary_default=True,
        prior_mean=0.0,
        prior_std=5.0,
        is_physical=True,
        is_flow=True,
    )

    # Scaling parameters: not part of the 14-element physics array passed to
    # JIT kernels, but stored in the unified registry for consistent lookup
    # via DEFAULT_REGISTRY["contrast"] / DEFAULT_REGISTRY["offset"].
    params["contrast"] = ParameterInfo(
        name="contrast",
        default=0.5,
        min_bound=0.0,
        max_bound=1.0,
        description="Optical contrast (per-angle scaling factor)",
        unit="",
        group="scaling",
        vary_default=True,
        prior_mean=0.5,
        prior_std=0.25,
        is_scaling=True,
        is_physical=False,
    )
    params["offset"] = ParameterInfo(
        name="offset",
        default=1.0,
        min_bound=0.5,
        max_bound=1.5,
        description="Baseline offset (per-angle)",
        unit="",
        group="scaling",
        vary_default=True,
        prior_mean=1.0,
        prior_std=0.25,
        is_scaling=True,
        is_physical=False,
    )

    return params


# Module-level default registry instance
DEFAULT_REGISTRY = ParameterRegistry()

# Convenience alias: scaling-only view of the registry for code that needs
# to distinguish scaling from physics parameters.
SCALING_PARAMS: Mapping[str, ParameterInfo] = MappingProxyType({
    name: DEFAULT_REGISTRY[name]
    for name in ("contrast", "offset")
})
