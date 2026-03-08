"""Data loading and validation pipeline for heterodyne CLI."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from heterodyne.data.validation import validate_xpcs_data
from heterodyne.data.xpcs_loader import load_xpcs_data
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.data.xpcs_loader import XPCSData

logger = get_logger(__name__)


def load_and_validate_data(config_manager: ConfigManager) -> XPCSData:
    """Load and validate XPCS experimental data.

    Args:
        config_manager: Configuration with data file path.

    Returns:
        Validated XPCSData object.

    Raises:
        SystemExit: If data validation fails with errors.
    """
    logger.info("Loading data from %s", config_manager.data_file_path)
    data = load_xpcs_data(config_manager.data_file_path)

    validation = validate_xpcs_data(data)
    if not validation.is_valid:
        for err in validation.errors:
            logger.error("Data validation error: %s", err)
        raise SystemExit(1)

    for warn in validation.warnings:
        logger.warning("Data validation warning: %s", warn)

    return data


def resolve_phi_angles(
    args: argparse.Namespace,
    config_manager: ConfigManager,
) -> list[float]:
    """Determine phi angles from CLI args or configuration.

    Priority: CLI --phi > config file > default [0.0].

    Args:
        args: Parsed CLI arguments (may have .phi attribute).
        config_manager: Configuration manager.

    Returns:
        List of phi angles in degrees.
    """
    phi_angles = getattr(args, "phi", None)
    if phi_angles is None:
        phi_angles = config_manager.phi_angles
    if phi_angles is None:
        phi_angles = [0.0]

    logger.info("Analyzing phi angles: %s", phi_angles)
    return phi_angles


def prepare_cmc_data(
    data: Any,
    phi_angles: list[float],
) -> dict[str, Any]:
    """Prepare data for CMC analysis.

    Extracts and organizes correlation data for each phi angle.

    Args:
        data: XPCSData object with correlation matrices.
        phi_angles: List of phi angles to process.

    Returns:
        Dictionary with prepared data keyed by purpose.
    """
    import numpy as np

    c2 = np.asarray(data.c2)
    prepared: dict[str, Any] = {
        "c2_data": c2,
        "phi_angles": phi_angles,
        "n_angles": len(phi_angles),
        "is_multi_angle": c2.ndim == 3,
    }

    logger.debug(
        "Prepared CMC data: %d angles, c2 shape=%s",
        len(phi_angles), c2.shape,
    )
    return prepared
