"""Configuration loading and CLI override merging for heterodyne CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

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
