"""Shared fixtures for heterodyne test suite.

This module provides reusable pytest fixtures for testing the heterodyne package:
- JAX configuration verification
- Model fixtures with default and custom parameters
- Synthetic test data generation
- NLSQ and CMC configuration presets
"""

from __future__ import annotations

import os

# CLAUDE.md Rule 8 — JAX_ENABLE_X64 must be set BEFORE the first JAX import.
# The conftest is the test suite's first JAX touchpoint (line below); without
# this env var, tests that explicitly request ``dtype=jnp.float64`` get
# truncated to float32 and emit a ``UserWarning``.  Setting it here is
# load-order-independent — it works whether or not the test under collection
# imports heterodyne first.
os.environ.setdefault("JAX_ENABLE_X64", "True")

from typing import TYPE_CHECKING  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# Mirror the env var into the live JAX config so the change takes effect
# even when JAX has already been imported by a plugin earlier in the boot
# sequence (pytest plugins, IDE harness, etc.).
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from heterodyne import CMCConfig, HeterodyneModel, NLSQConfig


# ============================================================================
# Pytest Markers — declared canonically in pyproject.toml [tool.pytest.ini_options]
# `markers = [...]`. Avoid re-registering in pytest_configure here so the two
# lists cannot drift.
# ============================================================================


# ============================================================================
# JAX Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def ensure_jax_float64() -> bool:
    """Session-scoped tripwire that verifies JAX x64 mode is enabled.

    Autouse ensures the contract (CLAUDE.md rule 8) is asserted exactly
    once per pytest session, before any test runs. Fires loudly if a
    future refactor breaks the early JAX_ENABLE_X64 setup at the top of
    this conftest, instead of letting tests silently truncate to float32.

    Returns:
        True if JAX x64 is enabled

    Raises:
        pytest.fail: If x64 mode is not enabled
    """
    # Check environment variable
    env_x64 = os.environ.get("JAX_ENABLE_X64", "").lower() in ("true", "1")

    # Check JAX config
    config_x64 = jax.config.jax_enable_x64

    if not (env_x64 or config_x64):
        pytest.fail(
            "JAX x64 mode is not enabled. Set JAX_ENABLE_X64=True or "
            "import heterodyne before JAX to enable float64 precision."
        )

    return True


@pytest.fixture
def jax_rng_key() -> jax.Array:
    """Provide a reproducible JAX random key.

    Returns:
        JAX PRNGKey with seed 42
    """
    return jax.random.PRNGKey(42)


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def base_model_config() -> dict:
    """Base configuration dictionary for HeterodyneModel.

    Returns:
        Configuration dict with temporal, scattering, and parameters sections
    """
    return {
        "analyzer_parameters": {
            "dt": 1.0,
            "start_frame": 1,
            "end_frame": 50,
            "scattering": {
                "wavevector_q": 0.01,
            },
        },
        "parameters": {},
    }


@pytest.fixture
def heterodyne_model(base_model_config: dict) -> HeterodyneModel:
    """Create a standard HeterodyneModel for testing.

    Uses default parameter values with 50 time points.

    Returns:
        Configured HeterodyneModel instance
    """
    from heterodyne import HeterodyneModel

    return HeterodyneModel.from_config(base_model_config)


@pytest.fixture
def small_model_config() -> dict:
    """Configuration for a minimal model (fast tests).

    Returns:
        Configuration with only 20 time points
    """
    return {
        "analyzer_parameters": {
            "dt": 1.0,
            "start_frame": 1,
            "end_frame": 20,
            "scattering": {
                "wavevector_q": 0.01,
            },
        },
        "parameters": {},
    }


@pytest.fixture
def small_heterodyne_model(small_model_config: dict) -> HeterodyneModel:
    """Create a minimal HeterodyneModel for fast tests.

    Returns:
        Small HeterodyneModel with 20 time points
    """
    from heterodyne import HeterodyneModel

    return HeterodyneModel.from_config(small_model_config)


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================


@pytest.fixture
def synthetic_c2_clean(heterodyne_model: HeterodyneModel) -> np.ndarray:
    """Generate clean synthetic C2 correlation data.

    Computes correlation using the model's default parameters.

    Args:
        heterodyne_model: The model fixture

    Returns:
        Clean correlation matrix without noise
    """
    c2 = heterodyne_model.compute_correlation(phi_angle=0.0)
    return np.asarray(c2)


@pytest.fixture
def synthetic_c2_data(
    heterodyne_model: HeterodyneModel,
    jax_rng_key: jax.Array,
) -> np.ndarray:
    """Generate synthetic C2 data with realistic noise.

    Adds Gaussian noise scaled to ~1% of signal magnitude.

    Args:
        heterodyne_model: The model fixture
        jax_rng_key: Reproducible RNG key

    Returns:
        Noisy correlation matrix
    """
    c2_clean = heterodyne_model.compute_correlation(phi_angle=0.0)
    c2_clean = jnp.asarray(c2_clean)

    # Add small Gaussian noise (1% of typical signal)
    noise_scale = 0.01 * jnp.max(jnp.abs(c2_clean))
    noise = jax.random.normal(jax_rng_key, shape=c2_clean.shape) * noise_scale

    c2_noisy = c2_clean + noise
    return np.asarray(c2_noisy)


@pytest.fixture
def small_c2_data(
    small_heterodyne_model: HeterodyneModel,
    jax_rng_key: jax.Array,
) -> np.ndarray:
    """Generate small synthetic C2 data for fast tests.

    Returns:
        20x20 noisy correlation matrix
    """
    c2_clean = small_heterodyne_model.compute_correlation(phi_angle=0.0)
    c2_clean = jnp.asarray(c2_clean)

    noise_scale = 0.01 * jnp.max(jnp.abs(c2_clean))
    noise = jax.random.normal(jax_rng_key, shape=c2_clean.shape) * noise_scale

    return np.asarray(c2_clean + noise)


# ============================================================================
# NLSQ Configuration Fixtures
# ============================================================================


@pytest.fixture
def nlsq_config() -> NLSQConfig:
    """Standard NLSQ configuration.

    Returns:
        Default NLSQConfig suitable for integration tests
    """
    from heterodyne import NLSQConfig

    return NLSQConfig(
        max_iterations=50,
        tolerance=1e-6,
        method="trf",
        verbose=0,
    )


@pytest.fixture
def fast_nlsq_config() -> NLSQConfig:
    """Fast NLSQ configuration for quick tests.

    Returns:
        NLSQConfig with minimal iterations for speed
    """
    from heterodyne import NLSQConfig

    return NLSQConfig(
        max_iterations=10,
        tolerance=1e-4,
        method="trf",
        verbose=0,
    )


# ============================================================================
# CMC Configuration Fixtures
# ============================================================================


@pytest.fixture
def cmc_config_1chain() -> CMCConfig:
    """CMC configuration with single chain (fastest).

    Returns:
        CMCConfig with 1 chain, minimal samples
    """
    from heterodyne import CMCConfig

    return CMCConfig(
        num_chains=1,
        num_warmup=100,
        num_samples=100,
        # fast_warmup opts out of the Rule-12 dense-mass warmup floor (1500
        # steps) for CI; min_warmup/min_samples adjusted to match the
        # reduced num_* so adaptive_sampling validation also passes.
        fast_warmup=True,
        min_warmup=50,
        min_samples=50,
        seed=42,
        use_nlsq_warmstart=True,
    )


@pytest.fixture
def cmc_config_2chains() -> CMCConfig:
    """CMC configuration with 2 chains.

    Returns:
        CMCConfig with 2 chains for testing multi-chain behavior
    """
    from heterodyne import CMCConfig

    return CMCConfig(
        num_chains=2,
        num_warmup=100,
        num_samples=100,
        fast_warmup=True,
        min_warmup=50,
        min_samples=50,
        seed=42,
        use_nlsq_warmstart=True,
    )


@pytest.fixture
def cmc_config_4chains() -> CMCConfig:
    """CMC configuration with 4 chains (standard).

    Returns:
        CMCConfig with 4 chains for full convergence testing
    """
    from heterodyne import CMCConfig

    return CMCConfig(
        num_chains=4,
        num_warmup=200,
        num_samples=200,
        fast_warmup=True,
        min_warmup=50,
        min_samples=50,
        seed=42,
        use_nlsq_warmstart=True,
    )
