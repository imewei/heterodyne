"""Reference-time reparameterization for heterodyne CMC.

Breaks banana-shaped posteriors for correlated power-law pairs
(D0/alpha) by sampling at a reference time t_ref where the product
D(t_ref) = D0 * t_ref^alpha is well-constrained by data.

Adapted from homodyne/optimization/cmc/reparameterization.py for
heterodyne's 3 power-law pairs:
  - D0_ref / alpha_ref
  - D0_sample / alpha_sample
  - v0 / beta
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Power-law pairs: (prefactor_name, exponent_name)
POWER_LAW_PAIRS: tuple[tuple[str, str], ...] = (
    ("D0_ref", "alpha_ref"),
    ("D0_sample", "alpha_sample"),
    ("v0", "beta"),
)


@dataclass(frozen=True)
class ReparamConfig:
    """Configuration for reference-time reparameterization.

    Attributes:
        enable_d_ref: Reparameterize D0_ref/alpha_ref pair.
        enable_d_sample: Reparameterize D0_sample/alpha_sample pair.
        enable_v_ref: Reparameterize v0/beta pair.
        t_ref: Reference time (geometric mean of dt and t_max).
    """

    enable_d_ref: bool = True
    enable_d_sample: bool = True
    enable_v_ref: bool = True
    t_ref: float = 1.0

    @property
    def enabled_pairs(self) -> list[tuple[str, str]]:
        """Return list of enabled (prefactor, exponent) pairs."""
        flags = [self.enable_d_ref, self.enable_d_sample, self.enable_v_ref]
        return [pair for pair, flag in zip(POWER_LAW_PAIRS, flags, strict=True) if flag]

    def is_reparameterized(self, name: str) -> bool:
        """Check if a parameter participates in reparameterization."""
        for prefactor, exponent in self.enabled_pairs:
            if name in (prefactor, exponent):
                return True
        return False

    def get_reparam_name(self, prefactor: str) -> str:
        """Get the reparameterized log-space name for a prefactor."""
        return f"log_{prefactor}_at_tref"


def compute_t_ref(
    dt: float,
    t_max: float,
    fallback_value: float | None = None,
) -> float:
    """Compute reference time as geometric mean of dt and t_max.

    t_ref = sqrt(dt * t_max)

    This places t_ref in the middle of the logarithmic time range,
    where the correlation function is most sensitive to the transport
    parameters.

    Args:
        dt: Time step (minimum lag time).
        t_max: Maximum lag time.
        fallback_value: Value to use if dt or t_max are invalid.

    Returns:
        Reference time.
    """
    if dt <= 0 or t_max <= 0:
        if fallback_value is not None:
            return fallback_value
        raise ValueError(f"dt and t_max must be positive, got dt={dt}, t_max={t_max}")

    t_ref = math.sqrt(dt * t_max)

    if t_ref <= 0 or not math.isfinite(t_ref):
        if fallback_value is not None:
            return fallback_value
        raise ValueError(f"Invalid t_ref={t_ref} from dt={dt}, t_max={t_max}")

    return t_ref


def transform_nlsq_to_reparam_space(
    nlsq_values: dict[str, float],
    nlsq_uncertainties: dict[str, float],
    t_ref: float,
    config: ReparamConfig | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Transform NLSQ point estimates to reparameterized space.

    For each enabled power-law pair (A0, alpha):
      log_A_at_tref = log(A0) + alpha * log(t_ref)

    Delta-method propagation for uncertainty:
      Var(log_A_at_tref) ≈ (sigma_A0/A0)² + (log(t_ref) * sigma_alpha)²

    Non-reparameterized parameters pass through unchanged.

    Args:
        nlsq_values: NLSQ best-fit values by parameter name.
        nlsq_uncertainties: NLSQ uncertainties by parameter name.
        t_ref: Reference time.
        config: Reparameterization config. Defaults to all pairs enabled.

    Returns:
        Tuple of (transformed_values, transformed_uncertainties).
    """
    if config is None:
        config = ReparamConfig(t_ref=t_ref)

    transformed_values: dict[str, float] = {}
    transformed_uncertainties: dict[str, float] = {}

    log_tref = math.log(t_ref) if t_ref > 0 else 0.0

    # Track which params are consumed by reparameterization
    consumed: set[str] = set()

    for prefactor, exponent in config.enabled_pairs:
        if prefactor not in nlsq_values or exponent not in nlsq_values:
            continue

        a0 = nlsq_values[prefactor]
        alpha = nlsq_values[exponent]

        # Forward transform: log(A0 * t_ref^alpha) = log(A0) + alpha * log(t_ref)
        if a0 <= 0:
            logger.warning("Negative prefactor %s=%s, clamping to 1e-10", prefactor, a0)
            a0 = max(abs(a0), 1e-10)
            log_a0 = math.log(a0)
        else:
            log_a0 = math.log(a0)

        log_at_tref = log_a0 + alpha * log_tref

        # Delta-method uncertainty propagation
        sigma_a0 = nlsq_uncertainties.get(prefactor, 0.0)
        sigma_alpha = nlsq_uncertainties.get(exponent, 0.0)

        # Var ≈ (σ_A0/A0)² + (log(t_ref) · σ_alpha)²
        rel_sigma_a0 = sigma_a0 / max(abs(a0), 1e-30)
        sigma_log_at_tref = math.sqrt(
            rel_sigma_a0**2 + (log_tref * sigma_alpha) ** 2
        )

        reparam_name = config.get_reparam_name(prefactor)
        transformed_values[reparam_name] = log_at_tref
        transformed_uncertainties[reparam_name] = max(sigma_log_at_tref, 1e-6)

        # Exponent passes through (still sampled directly)
        transformed_values[exponent] = alpha
        transformed_uncertainties[exponent] = max(sigma_alpha, 1e-6)

        consumed.add(prefactor)
        consumed.add(exponent)

    # Pass through non-reparameterized parameters
    for name, value in nlsq_values.items():
        if name not in consumed:
            transformed_values[name] = value
            transformed_uncertainties[name] = nlsq_uncertainties.get(name, 0.0)

    return transformed_values, transformed_uncertainties


def transform_to_sampling_space(
    params: dict[str, float],
    config: ReparamConfig,
) -> dict[str, float]:
    """Transform physics-space parameters to sampling (reparam) space.

    Used for initializing MCMC chains from NLSQ results.

    Args:
        params: Physics-space parameter values.
        config: Reparameterization config.

    Returns:
        Sampling-space parameter values.
    """
    result: dict[str, float] = {}
    log_tref = math.log(config.t_ref) if config.t_ref > 0 else 0.0
    consumed: set[str] = set()

    for prefactor, exponent in config.enabled_pairs:
        if prefactor not in params or exponent not in params:
            continue

        a0 = params[prefactor]
        alpha = params[exponent]

        if a0 <= 0:
            logger.warning("Negative prefactor %s=%s, clamping to 1e-10", prefactor, a0)
            a0 = max(abs(a0), 1e-10)
            log_a0 = math.log(a0)
        else:
            log_a0 = math.log(a0)

        reparam_name = config.get_reparam_name(prefactor)
        result[reparam_name] = log_a0 + alpha * log_tref
        result[exponent] = alpha
        consumed.update([prefactor, exponent])

    for name, value in params.items():
        if name not in consumed:
            result[name] = value

    return result


def reparam_to_physics_jax(
    log_at_tref: jnp.ndarray,
    alpha: jnp.ndarray,
    t_ref: float,
) -> jnp.ndarray:
    """Back-transform reparameterized values to physics space (JAX).

    A0 = exp(log_at_tref - alpha * log(t_ref))

    Args:
        log_at_tref: Log of the quantity at t_ref.
        alpha: Power-law exponent.
        t_ref: Reference time.

    Returns:
        A0 (prefactor in physics space).
    """
    log_tref = jnp.log(jnp.float64(t_ref))
    return jnp.exp(log_at_tref - alpha * log_tref)


def transform_to_physics_space(
    samples: dict[str, np.ndarray],
    config: ReparamConfig,
) -> dict[str, np.ndarray]:
    """Transform sampling-space posterior samples to physics space.

    Vectorized over sample dimension. For each enabled pair, computes:
      A0 = exp(log_at_tref - alpha * log(t_ref))

    Non-reparameterized parameters pass through.

    Args:
        samples: Dict of posterior samples keyed by sampling-space names.
        config: Reparameterization config.

    Returns:
        Dict of physics-space samples.
    """
    result: dict[str, np.ndarray] = {}
    log_tref = np.log(config.t_ref) if config.t_ref > 0 else 0.0
    consumed: set[str] = set()

    for prefactor, exponent in config.enabled_pairs:
        reparam_name = config.get_reparam_name(prefactor)

        if reparam_name not in samples or exponent not in samples:
            continue

        log_at_tref = samples[reparam_name]
        alpha = samples[exponent]

        # Back-transform: A0 = exp(log_at_tref - alpha * log(t_ref))
        a0 = np.exp(log_at_tref - alpha * log_tref)

        result[prefactor] = a0
        result[exponent] = alpha
        consumed.add(reparam_name)
        consumed.add(exponent)

    # Pass through non-consumed parameters
    for name, values in samples.items():
        if name not in consumed:
            result[name] = values

    return result
