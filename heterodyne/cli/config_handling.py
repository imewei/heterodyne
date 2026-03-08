"""Configuration loading and CLI override merging for heterodyne CLI."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heterodyne.config.manager import ConfigManager
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def load_and_merge_config(
    yaml_path: Path | str,
    cli_args: argparse.Namespace,
) -> ConfigManager:
    """Load YAML configuration and merge CLI overrides.

    Args:
        yaml_path: Path to the YAML configuration file.
        cli_args: Parsed command-line arguments that may override
            config file values.

    Returns:
        Fully configured ConfigManager instance.
    """
    logger.info("Loading configuration from %s", yaml_path)
    config_manager = ConfigManager.from_yaml(yaml_path)
    apply_cli_overrides(config_manager, cli_args)
    return config_manager


def apply_cli_overrides(
    config_manager: ConfigManager,
    args: argparse.Namespace,
) -> None:
    """Apply CLI argument overrides to the configuration.

    Modifies the ConfigManager in-place based on CLI arguments that
    take precedence over YAML file values.

    Args:
        config_manager: Configuration manager to modify.
        args: Parsed CLI arguments with potential overrides.
    """
    # Override output directory
    if hasattr(args, "output") and args.output is not None:
        logger.debug("CLI override: output_dir=%s", args.output)

    # Override NLSQ settings
    if hasattr(args, "multistart") and args.multistart:
        nlsq_config = config_manager.nlsq_config
        nlsq_config["multistart"] = True
        if hasattr(args, "multistart_n"):
            nlsq_config["multistart_n"] = args.multistart_n
        logger.debug("CLI override: multistart enabled (n=%s)",
                     getattr(args, "multistart_n", "default"))

    # Override CMC settings
    if hasattr(args, "num_samples") and args.num_samples is not None:
        config_manager.cmc_config["num_samples"] = args.num_samples
        logger.debug("CLI override: num_samples=%d", args.num_samples)

    if hasattr(args, "num_chains") and args.num_chains is not None:
        config_manager.cmc_config["num_chains"] = args.num_chains
        logger.debug("CLI override: num_chains=%d", args.num_chains)


def _configure_device(args: argparse.Namespace) -> dict[str, Any]:
    """Configure the compute device based on CLI arguments and hardware detection.

    Attempts to use ``heterodyne.device.config.configure_optimal_device()``
    for hardware-aware setup.  Falls back to basic CPU defaults when the
    device module is unavailable.

    Args:
        args: Parsed CLI arguments.  Recognised attributes:
            ``device`` (str | None) and ``n_threads`` (int | None).

    Returns:
        Dict with keys ``device_type``, ``n_threads``, ``numa_nodes``,
        and ``device_configured`` (bool).
    """
    device_override: str | None = getattr(args, "device", None)
    n_threads_override: int | None = getattr(args, "n_threads", None)

    result: dict[str, Any] = {
        "device_type": "cpu",
        "n_threads": 1,
        "numa_nodes": 1,
        "device_configured": False,
    }

    try:
        from heterodyne.device.config import configure_optimal_device

        hw = configure_optimal_device()
        result["device_type"] = "cpu"
        result["n_threads"] = hw.available_cores
        result["device_configured"] = True
    except (ImportError, AttributeError) as exc:
        logger.debug("Device auto-configuration unavailable: %s", exc)

    # CLI overrides take precedence
    if device_override is not None:
        result["device_type"] = device_override
    if n_threads_override is not None:
        result["n_threads"] = n_threads_override

    logger.info(
        "Device configured: type=%s, n_threads=%d, auto=%s",
        result["device_type"],
        result["n_threads"],
        result["device_configured"],
    )
    return result


def _build_mcmc_runtime_kwargs(
    config_manager: ConfigManager,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assemble MCMC runtime keyword arguments.

    Priority order (highest to lowest): CLI args, ``config_manager.cmc_config``,
    built-in defaults.

    Args:
        config_manager: Active configuration manager.
        args: Parsed CLI arguments with potential MCMC overrides.

    Returns:
        Validated dict of MCMC runtime kwargs.

    Raises:
        ValueError: If any assembled value fails validation.
    """
    defaults: dict[str, Any] = {
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 4,
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
    }

    # Layer 1: config file values
    cmc_cfg = config_manager.get_cmc_config()
    for key in defaults:
        if key in cmc_cfg:
            defaults[key] = cmc_cfg[key]

    # Layer 2: CLI overrides
    for key in list(defaults):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    # Validate
    if defaults["num_warmup"] <= 0:
        msg = f"num_warmup must be > 0, got {defaults['num_warmup']}"
        raise ValueError(msg)
    if defaults["num_samples"] <= 0:
        msg = f"num_samples must be > 0, got {defaults['num_samples']}"
        raise ValueError(msg)
    if defaults["num_chains"] <= 0:
        msg = f"num_chains must be > 0, got {defaults['num_chains']}"
        raise ValueError(msg)
    if not (0 < defaults["target_accept_prob"] < 1):
        msg = (
            f"target_accept_prob must be in (0, 1), "
            f"got {defaults['target_accept_prob']}"
        )
        raise ValueError(msg)

    logger.debug("Assembled MCMC runtime kwargs: %s", defaults)
    return defaults


def _get_default_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build a minimal valid configuration when no config file is provided.

    Uses CLI arguments for data path, wavevector *q*, and time step *dt*
    when available, otherwise applies sensible defaults.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Config dict with required sections: ``experimental_data``,
        ``temporal``, ``scattering``, and ``parameters``.
    """
    data_path: str = getattr(args, "data_path", "")
    dt: float = getattr(args, "dt", None) or 1.0
    q: float = getattr(args, "q", None) or 0.01

    config: dict[str, Any] = {
        "experimental_data": {
            "data_path": data_path,
        },
        "temporal": {
            "dt": dt,
            "time_length": 1000,
        },
        "scattering": {
            "wavevector_q": q,
        },
        "parameters": {
            "reference": {},
            "sample": {},
            "velocity": {},
            "fraction": {},
            "angle": {},
            "scaling": {},
        },
    }

    logger.debug("Built default config (no YAML): dt=%s, q=%s", dt, q)
    return config


def _get_package_version() -> str:
    """Return the heterodyne package version string.

    Resolution order:

    1. ``heterodyne._version.__version__`` (editable / dev install).
    2. ``importlib.metadata.version("heterodyne")`` (installed package).
    3. ``"unknown"`` as last resort.
    """
    try:
        from heterodyne._version import __version__  # type: ignore[import-untyped]

        return str(__version__)
    except (ImportError, AttributeError):
        pass

    try:
        return importlib.metadata.version("heterodyne")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
