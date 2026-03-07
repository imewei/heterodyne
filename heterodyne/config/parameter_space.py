"""Parameter space definition with prior distributions for Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.config.parameter_names import (
    ALL_PARAM_NAMES,
    ALL_PARAM_NAMES_WITH_SCALING,
    SCALING_PARAMS,
)
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

if TYPE_CHECKING:
    import jax.numpy as jnp


class PriorType(Enum):
    """Available prior distribution types."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
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

    @classmethod
    def truncated_normal(
        cls, loc: float, scale: float, low: float, high: float,
    ) -> PriorDistribution:
        """Create truncated normal prior (bounded Gaussian)."""
        return cls(
            PriorType.TRUNCATED_NORMAL,
            {"loc": loc, "scale": scale, "low": low, "high": high},
        )

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
        elif self.prior_type == PriorType.TRUNCATED_NORMAL:
            return dist.TruncatedNormal(
                loc=self.params["loc"],
                scale=self.params["scale"],
                low=self.params["low"],
                high=self.params["high"],
            )
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
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            info = DEFAULT_REGISTRY[name]
            if name not in self.values:
                self.values[name] = info.default
            if name not in self.vary:
                self.vary[name] = info.vary_default
            if name not in self.bounds:
                self.bounds[name] = (info.min_bound, info.max_bound)
            if name not in self.priors:
                self.priors[name] = _default_prior(name, info)

    @property
    def n_total(self) -> int:
        """Total number of parameters."""
        return len(ALL_PARAM_NAMES)

    @property
    def n_varying(self) -> int:
        """Number of parameters that vary in optimization."""
        return len(self.varying_names)

    @property
    def varying_names(self) -> list[str]:
        """Names of parameters that vary (physics + scaling)."""
        return [name for name in ALL_PARAM_NAMES_WITH_SCALING if self.vary.get(name, False)]

    @property
    def fixed_names(self) -> list[str]:
        """Names of parameters that are fixed."""
        return [name for name in ALL_PARAM_NAMES_WITH_SCALING if not self.vary.get(name, False)]

    @property
    def varying_physics_names(self) -> list[str]:
        """Names of varying physics parameters (excludes scaling)."""
        return [name for name in ALL_PARAM_NAMES if self.vary.get(name, False)]

    @property
    def scaling_values(self) -> dict[str, float]:
        """Get contrast and offset values."""
        return {name: self.values[name] for name in SCALING_PARAMS}

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

        Raises:
            ValueError: If a key doesn't match any known parameter
        """
        for name, value in params.items():
            if name not in self.values:
                raise ValueError(
                    f"Unknown parameter '{name}'. "
                    f"Valid parameters: {list(ALL_PARAM_NAMES)}"
                )
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
            "scaling": ["contrast", "offset"],
        }

        for group_name, param_names in group_map.items():
            group_config = params_config.get(group_name, {})

            # Check for unknown keys in this group
            known_params = set(param_names)
            for ck in group_config:
                if ck not in known_params:
                    raise ValueError(
                        f"Unknown parameter key '{ck}' in group '{group_name}'. "
                        f"Valid keys: {param_names}"
                    )

            for param_name in param_names:
                # Direct key match only — no substring matching
                if param_name not in group_config:
                    continue

                pconfig = group_config[param_name]
                if isinstance(pconfig, dict):
                    if "value" in pconfig:
                        space.values[param_name] = pconfig["value"]
                    if "min" in pconfig and "max" in pconfig:
                        space.bounds[param_name] = (pconfig["min"], pconfig["max"])
                    if "vary" in pconfig:
                        space.vary[param_name] = pconfig["vary"]
                    if "prior" in pconfig:
                        prior_type_str = pconfig["prior"]
                        prior_params = pconfig.get("prior_params", {})
                        space.priors[param_name] = _build_prior(
                            param_name, prior_type_str, prior_params, space.bounds[param_name]
                        )

        return space


# Default TruncatedNormal prior specifications: (loc, scale)
# All parameters use TruncatedNormal priors truncated to their registry bounds.
_DEFAULT_PRIOR_SPECS: dict[str, tuple[float, float]] = {
    "D0_ref": (1e4, 5e3),
    "alpha_ref": (0.0, 1.0),
    "D_offset_ref": (0.0, 1e3),
    "D0_sample": (1e4, 5e3),
    "alpha_sample": (0.0, 1.0),
    "D_offset_sample": (0.0, 1e3),
    "v0": (1e3, 500.0),
    "beta": (0.0, 1.0),
    "v_offset": (0.0, 25.0),
    "f0": (0.5, 0.25),
    "f1": (0.0, 5.0),
    "f2": (0.0, 1e3),
    "f3": (0.0, 0.5),
    "phi0": (0.0, 5.0),
    "contrast": (0.5, 0.25),
    "offset": (1.0, 0.25),
}


def _default_prior(
    name: str,
    info: Any,
) -> PriorDistribution:
    """Build the default TruncatedNormal prior for a parameter.

    Args:
        name: Parameter name.
        info: ParameterInfo from the registry.

    Returns:
        TruncatedNormal prior distribution.
    """
    if name in _DEFAULT_PRIOR_SPECS:
        loc, scale = _DEFAULT_PRIOR_SPECS[name]
        return PriorDistribution.truncated_normal(
            loc=loc, scale=scale,
            low=info.min_bound, high=info.max_bound,
        )
    # Fallback for any unspecified parameter
    return PriorDistribution.uniform(info.min_bound, info.max_bound)


def _build_prior(
    name: str,
    prior_type_str: str,
    prior_params: dict[str, float],
    bounds: tuple[float, float],
) -> PriorDistribution:
    """Build a PriorDistribution from config strings.

    Args:
        name: Parameter name (for error messages)
        prior_type_str: One of "uniform", "normal", "lognormal",
            "halfnormal", "exponential"
        prior_params: Distribution-specific parameters
            (e.g. {"loc": 0, "scale": 1} for normal)
        bounds: (low, high) bounds, used as fallback for uniform

    Returns:
        Configured PriorDistribution
    """
    try:
        prior_type = PriorType(prior_type_str)
    except ValueError:
        valid = [pt.value for pt in PriorType]
        raise ValueError(
            f"Unknown prior type '{prior_type_str}' for parameter '{name}'. "
            f"Valid types: {valid}"
        ) from None

    if prior_type == PriorType.UNIFORM:
        low = prior_params.get("low", bounds[0])
        high = prior_params.get("high", bounds[1])
        return PriorDistribution.uniform(low, high)
    elif prior_type == PriorType.NORMAL:
        loc = prior_params.get("loc", (bounds[0] + bounds[1]) / 2)
        scale = prior_params.get("scale", (bounds[1] - bounds[0]) / 4)
        return PriorDistribution.normal(loc, scale)
    elif prior_type == PriorType.TRUNCATED_NORMAL:
        loc = prior_params.get("loc", (bounds[0] + bounds[1]) / 2)
        scale = prior_params.get("scale", (bounds[1] - bounds[0]) / 4)
        low = prior_params.get("low", bounds[0])
        high = prior_params.get("high", bounds[1])
        return PriorDistribution.truncated_normal(loc, scale, low, high)
    elif prior_type == PriorType.LOGNORMAL:
        loc = prior_params.get("loc", 0.0)
        scale = prior_params.get("scale", 1.0)
        return PriorDistribution.lognormal(loc, scale)
    elif prior_type == PriorType.HALFNORMAL:
        scale = prior_params.get("scale", 1.0)
        return PriorDistribution.halfnormal(scale)
    elif prior_type == PriorType.EXPONENTIAL:
        return PriorDistribution(PriorType.EXPONENTIAL, prior_params)
    else:
        raise ValueError(f"Unhandled prior type: {prior_type}")
