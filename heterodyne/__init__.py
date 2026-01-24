# Enable float64 for JAX before any imports - required for scientific computing precision
# This must be set before JAX is imported
# ruff: noqa: E402, I001
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

# Now import JAX and configure
import jax
jax.config.update("jax_enable_x64", True)

"""Heterodyne: CPU-optimized JAX-based heterodyne scattering analysis for XPCS.

This package provides tools for analyzing two-component heterodyne XPCS
(X-ray Photon Correlation Spectroscopy) data using a 14-parameter model.

Main Components:
- core: Physics engine with JAX-accelerated correlation computation
- config: Configuration management and parameter handling
- data: Data loading and validation
- optimization: NLSQ and CMC (Bayesian) fitting
- device: CPU detection and HPC optimization
- viz: Visualization utilities
- cli: Command-line interface

Example:
    >>> from heterodyne import HeterodyneModel, fit_nlsq_jax
    >>> model = HeterodyneModel.from_config(config)
    >>> result = fit_nlsq_jax(model, c2_data, phi_angle=45.0)

For optimal CPU performance, configure the device before JAX is imported:
    >>> from heterodyne.device import configure_optimal_device
    >>> hw = configure_optimal_device(mode="cmc")  # Before importing JAX
    >>> import jax  # Now JAX uses optimal settings
"""

try:
    from heterodyne._version import __version__, __version_tuple__
except ImportError:
    # Fallback for editable installs before first build
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")

# Core model
from heterodyne.core.heterodyne_model import HeterodyneModel
from heterodyne.core.models import TwoComponentModel

# Configuration
from heterodyne.config.manager import ConfigManager, load_xpcs_config
from heterodyne.config.parameter_manager import ParameterManager

# Data loading
from heterodyne.data.xpcs_loader import XPCSDataLoader, load_xpcs_data

# Optimization
from heterodyne.optimization.nlsq import fit_nlsq_jax, NLSQConfig, NLSQResult
from heterodyne.optimization.cmc import fit_cmc_jax, CMCConfig, CMCResult

# Device configuration (lazy import to avoid forcing JAX import)
# Use: from heterodyne.device import configure_optimal_device


def get_device_config():
    """Get the device configuration module.

    Returns:
        The heterodyne.device module.

    Example:
        >>> device = get_device_config()
        >>> hw = device.configure_optimal_device(mode="cmc")
    """
    from heterodyne import device
    return device


__all__ = [
    # Version
    "__version__",
    "__version_tuple__",
    # Core
    "HeterodyneModel",
    "TwoComponentModel",
    # Config
    "ConfigManager",
    "load_xpcs_config",
    "ParameterManager",
    # Data
    "XPCSDataLoader",
    "load_xpcs_data",
    # Optimization
    "fit_nlsq_jax",
    "NLSQConfig",
    "NLSQResult",
    "fit_cmc_jax",
    "CMCConfig",
    "CMCResult",
    # Device (lazy import helper)
    "get_device_config",
]
