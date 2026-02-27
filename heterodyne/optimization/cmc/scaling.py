"""Smooth bounded parameter scaling for heterodyne CMC.

Replaces jnp.clip() (zero gradient at bounds) with tanh-based smooth
bounding that is differentiable everywhere, allowing NUTS to adapt its
mass matrix near parameter boundaries.

Adapted from homodyne/optimization/cmc/scaling.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from heterodyne.config.parameter_space import ParameterSpace


@dataclass
class ParameterScaling:
    """Scaling specification for a single parameter.

    Defines the mapping between z-space (standard normal for MCMC)
    and original physics space with smooth bounding.

    Attributes:
        name: Parameter name.
        center: NLSQ best-fit value (center of prior).
        scale: Prior width (NLSQ uncertainty × width_factor).
        low: Lower bound in physics space.
        high: Upper bound in physics space.
    """

    name: str
    center: float
    scale: float
    low: float
    high: float

    def to_normalized(self, value: float | jnp.ndarray) -> float | jnp.ndarray:
        """Transform from physics space to z-space (normalized).

        z = (value - center) / scale

        Args:
            value: Physics-space value.

        Returns:
            Normalized z-space value.
        """
        return (value - self.center) / self.scale

    def to_original(self, z_value: jnp.ndarray) -> jnp.ndarray:
        """Transform from z-space to bounded original (physics) space.

        raw = center + scale * z
        result = smooth_bound(raw, low, high)

        Args:
            z_value: Normalized z-space value.

        Returns:
            Bounded physics-space value.
        """
        raw = self.center + self.scale * z_value
        return smooth_bound(raw, self.low, self.high)


def smooth_bound(
    raw: jnp.ndarray,
    low: float,
    high: float,
) -> jnp.ndarray:
    """Smooth bounding using tanh transform.

    Maps (-inf, +inf) → (low, high) via:
      mid + half * tanh((raw - mid) / half)

    This is differentiable everywhere, unlike jnp.clip() which has
    zero gradient at bounds and kills NUTS adaptation.

    Args:
        raw: Unbounded input value.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Bounded value in (low, high).
    """
    mid = jnp.float64((low + high) / 2.0)
    half = jnp.float64((high - low) / 2.0)
    # Guard degenerate bounds (low == high) to avoid 0/0 → NaN
    return jnp.where(half > 0, mid + half * jnp.tanh((raw - mid) / half), mid)  # type: ignore[no-any-return]


def smooth_bound_inverse(
    value: float,
    low: float,
    high: float,
) -> float:
    """Inverse of smooth_bound for initialization.

    Recovers the raw (unbounded) value from a bounded value:
      raw = mid + half * arctanh((value - mid) / half)

    Args:
        value: Bounded value in (low, high).
        low: Lower bound.
        high: Upper bound.

    Returns:
        Unbounded raw value.
    """
    mid = (low + high) / 2.0
    half = (high - low) / 2.0

    # Clamp to avoid arctanh(±1) = ±inf
    normalized = (value - mid) / half
    normalized = float(np.clip(normalized, -0.999, 0.999))

    return mid + half * float(np.arctanh(normalized))


def compute_scaling_factors(
    space: ParameterSpace,
    nlsq_values: dict[str, float] | None = None,
    nlsq_uncertainties: dict[str, float] | None = None,
    width_factor: float = 2.0,
) -> dict[str, ParameterScaling]:
    """Build ParameterScaling for each varying parameter.

    Uses NLSQ values as centers and NLSQ uncertainties × width_factor
    as scale. Falls back to bounds midpoint and range/6 when NLSQ
    results are unavailable.

    Args:
        space: Parameter space with bounds and varying flags.
        nlsq_values: NLSQ best-fit values by name.
        nlsq_uncertainties: NLSQ uncertainties by name.
        width_factor: Multiplier on NLSQ uncertainty for prior width.

    Returns:
        Dict mapping parameter name to ParameterScaling.
    """
    scalings: dict[str, ParameterScaling] = {}

    for name in space.varying_names:
        low, high = space.bounds[name]

        # Center: NLSQ value or bounds midpoint
        if nlsq_values is not None and name in nlsq_values:
            center = nlsq_values[name]
        else:
            center = (low + high) / 2.0

        # Scale: NLSQ uncertainty × width_factor or bounds range / 6
        if (
            nlsq_uncertainties is not None
            and name in nlsq_uncertainties
            and nlsq_uncertainties[name] > 0
        ):
            scale = nlsq_uncertainties[name] * width_factor
        else:
            scale = (high - low) / 6.0

        # Ensure minimum scale to avoid division by zero
        scale = max(scale, 1e-10)

        scalings[name] = ParameterScaling(
            name=name,
            center=center,
            scale=scale,
            low=low,
            high=high,
        )

    return scalings
