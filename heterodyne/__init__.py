# Enable float64 for JAX before any imports - required for scientific computing precision
# This must be set before JAX is imported
# ruff: noqa: E402, I001
from __future__ import annotations

import os
import types
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

# Configure XLA flags for multi-core CPU parallelism and stability
_xla_flags = (
    "--xla_force_host_platform_device_count=4 --xla_disable_hlo_passes=constant_folding"
)
_existing_flags = os.environ.get("XLA_FLAGS", "")
if _existing_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_flags} {_xla_flags}"
else:
    os.environ["XLA_FLAGS"] = _xla_flags

# Now import JAX and configure
import jax

jax.config.update("jax_enable_x64", True)

# Filter NumPyro deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpyro")

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


# ---------------------------------------------------------------------------
# Lazy import system -- avoids 3-6s JAX startup cost on every import
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "HeterodyneModel": ("heterodyne.core.heterodyne_model", "HeterodyneModel"),
    "TwoComponentModel": ("heterodyne.core.models", "TwoComponentModel"),
    "ConfigManager": ("heterodyne.config.manager", "ConfigManager"),
    "load_xpcs_config": ("heterodyne.config.manager", "load_xpcs_config"),
    "ParameterManager": ("heterodyne.config.parameter_manager", "ParameterManager"),
    "XPCSDataLoader": ("heterodyne.data.xpcs_loader", "XPCSDataLoader"),
    "load_xpcs_data": ("heterodyne.data.xpcs_loader", "load_xpcs_data"),
    "fit_nlsq_jax": ("heterodyne.optimization.nlsq", "fit_nlsq_jax"),
    "NLSQConfig": ("heterodyne.optimization.nlsq", "NLSQConfig"),
    "NLSQResult": ("heterodyne.optimization.nlsq", "NLSQResult"),
    "fit_cmc_jax": ("heterodyne.optimization.cmc", "fit_cmc_jax"),
    "CMCConfig": ("heterodyne.optimization.cmc", "CMCConfig"),
    "CMCResult": ("heterodyne.optimization.cmc", "CMCResult"),
}

# Module availability flags -- populated on first access
_MODULE_FLAGS: dict[str, tuple[str, str]] = {
    "HAS_CORE": ("heterodyne.core", "Core physics engine"),
    "HAS_DATA": ("heterodyne.data", "Data loading"),
    "HAS_CONFIG": ("heterodyne.config", "Configuration management"),
    "HAS_OPTIMIZATION": ("heterodyne.optimization", "Optimization"),
    "HAS_DEVICE": ("heterodyne.device", "Device configuration"),
    "HAS_VIZ": ("heterodyne.viz", "Visualization"),
    "HAS_CLI": ("heterodyne.cli", "Command-line interface"),
}


def __getattr__(name: str) -> object:
    """Lazy import hook for deferred module loading."""
    # Check lazy imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        # Cache in module namespace for subsequent access
        globals()[name] = attr
        return attr

    # Check module availability flags
    if name in _MODULE_FLAGS:
        module_path, description = _MODULE_FLAGS[name]
        import importlib

        try:
            importlib.import_module(module_path)
            result = True
        except ImportError:
            result = False
        globals()[name] = result
        return result

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def get_device_config() -> types.ModuleType:
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
    # Module availability flags
    "HAS_CORE",
    "HAS_DATA",
    "HAS_CONFIG",
    "HAS_OPTIMIZATION",
    "HAS_DEVICE",
    "HAS_VIZ",
    "HAS_CLI",
]
