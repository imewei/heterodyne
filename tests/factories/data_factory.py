"""Factory functions that generate synthetic test data for heterodyne tests.

All functions return pure NumPy arrays and plain Python dicts to keep
factories free of JAX/NumPyro dependencies.
"""

from __future__ import annotations

import numpy as np


def make_tau_array(
    n: int = 64,
    tau_min: float = 1e-6,
    tau_max: float = 1.0,
) -> np.ndarray:
    """Return a log-spaced array of delay times.

    Args:
        n: Number of delay points.
        tau_min: Minimum delay time (seconds).
        tau_max: Maximum delay time (seconds).

    Returns:
        1-D array of shape ``(n,)`` with log-spaced values in
        ``[tau_min, tau_max]``.
    """
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n)


def make_g2_synthetic(
    tau: np.ndarray,
    contrast: float = 0.3,
    offset: float = 1.0,
    decay_rate: float = 1e3,
) -> np.ndarray:
    """Compute a simple single-exponential g2 correlation function.

    Uses the model ``g2(τ) = offset + contrast * exp(-2 * decay_rate * τ)``,
    which corresponds to a Siegert-relation approximation for a
    single-component diffusive system.

    Args:
        tau: 1-D array of delay times (seconds).
        contrast: Optical contrast (amplitude of the decay, 0–1).
        offset: Long-lag baseline (typically 1.0 for normalised g2).
        decay_rate: Effective decay rate in units of s⁻¹.

    Returns:
        1-D array of g2 values with the same shape as ``tau``.
    """
    tau = np.asarray(tau, dtype=float)
    return offset + contrast * np.exp(-2.0 * decay_rate * tau)


def make_correlation_data(
    n_tau: int = 64,
    n_angles: int = 3,
    contrast: float = 0.3,
    offset: float = 1.0,
    decay_rate: float = 1e3,
    noise_level: float = 0.002,
    seed: int = 0,
) -> dict:
    """Build a synthetic multi-angle correlation dataset.

    Each angle uses the same exponential decay model with a small amount
    of Gaussian noise added to simulate real detector statistics.

    Args:
        n_tau: Number of delay-time points.
        n_angles: Number of scattering angles.
        contrast: Optical contrast shared across angles.
        offset: Baseline offset shared across angles.
        decay_rate: Exponential decay rate in s⁻¹.
        noise_level: Standard deviation of additive Gaussian noise.
        seed: NumPy random seed for reproducibility.

    Returns:
        Dictionary with keys:

        ``"tau"``
            1-D array of shape ``(n_tau,)``.
        ``"g2"``
            2-D array of shape ``(n_angles, n_tau)`` with noisy g2 values.
        ``"angles"``
            List of ``n_angles`` float phi-angle values in degrees,
            evenly spaced from 0 to 90.
    """
    rng = np.random.default_rng(seed)
    tau = make_tau_array(n=n_tau)
    g2_clean = make_g2_synthetic(
        tau, contrast=contrast, offset=offset, decay_rate=decay_rate
    )
    noise = rng.normal(scale=noise_level, size=(n_angles, n_tau))
    g2 = np.broadcast_to(g2_clean, (n_angles, n_tau)).copy() + noise
    angles = list(np.linspace(0.0, 90.0, n_angles))
    return {
        "tau": tau,
        "g2": g2,
        "angles": angles,
    }


def make_param_dict(overrides: dict | None = None) -> dict[str, float]:
    """Return a dictionary of all 16 model parameters with registry defaults.

    Reads default values from :data:`heterodyne.config.parameter_registry.DEFAULT_REGISTRY`
    so that the factory stays in sync with the authoritative registry.

    Args:
        overrides: Optional mapping of parameter names to replacement values.
            Any key present here overrides the registry default.

    Returns:
        Dictionary of ``{param_name: default_value}`` for all 16 parameters
        (14 physics + 2 scaling), with ``overrides`` applied.
    """
    from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

    params: dict[str, float] = {
        name: DEFAULT_REGISTRY[name].default for name in DEFAULT_REGISTRY
    }
    if overrides:
        params.update(overrides)
    return params
