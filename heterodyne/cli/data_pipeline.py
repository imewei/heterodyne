"""Data loading and validation pipeline for heterodyne CLI."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.data.validation import validate_xpcs_data
from heterodyne.data.xpcs_loader import (
    XPCSData,
    _apply_diagonal_correction,
    load_xpcs_data,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager

logger = get_logger(__name__)

# Common azimuthal angles used in XPCS experiments (degrees).
COMMON_XPCS_ANGLES: list[int] = [0, 30, 45, 60, 90, 120, 135, 150, 180]


def _exclude_t0_from_analysis(data: XPCSData) -> XPCSData:
    """Exclude the first time point (t=0) from analysis data.

    At t=0 the two-time correlation function has a singularity that causes
    D(t) -> infinity, which breaks numerical fitting.  This function slices
    out the first time point from c2, t1, t2, and uncertainties to prevent
    the singularity from propagating into downstream optimizers.

    Args:
        data: Validated XPCSData with at least 2 time points.

    Returns:
        New XPCSData with the first time point removed.
    """
    original_n = data.t1.shape[0]
    if original_n <= 1:
        logger.warning("Cannot exclude t=0: data has only %d time point(s)", original_n)
        return data

    logger.warning(
        "Excluding t=0 time point to prevent D(t)->inf singularity (c2 %s -> %s)",
        data.c2.shape,
        (
            (*data.c2.shape[:-2], data.c2.shape[-2] - 1, data.c2.shape[-1] - 1)
            if data.c2.ndim >= 2
            else "(?)"
        ),
    )

    # Slice c2: remove first row and column from the time dimensions.
    if data.c2.ndim == 3:
        c2_new = data.c2[:, 1:, 1:]
    else:
        c2_new = data.c2[1:, 1:]

    t1_new = data.t1[1:]
    t2_new = data.t2[1:]

    uncertainties_new = data.uncertainties
    if data.uncertainties is not None:
        if data.uncertainties.ndim == data.c2.ndim:
            # Same shape as c2 — slice identically.
            if data.uncertainties.ndim == 3:
                uncertainties_new = data.uncertainties[:, 1:, 1:]
            else:
                uncertainties_new = data.uncertainties[1:, 1:]
        elif data.uncertainties.ndim == 1:
            # Per-time-point uncertainties.
            uncertainties_new = data.uncertainties[1:]

    return XPCSData(
        c2=c2_new,
        t1=t1_new,
        t2=t2_new,
        q=data.q,
        phi_angles=data.phi_angles,
        uncertainties=uncertainties_new,
        q_values=data.q_values,
        metadata=data.metadata,
    )


def load_and_validate_data(config_manager: ConfigManager) -> XPCSData:
    """Load and validate XPCS experimental data.

    Args:
        config_manager: Configuration with data file path.

    Returns:
        Validated XPCSData object.

    Raises:
        SystemExit: If data validation fails with errors.
    """
    # Extract frame range from analyzer_parameters (1-indexed, inclusive)
    start_frame = config_manager.start_frame
    end_frame = config_manager.end_frame
    frame_range: tuple[int, int] | None = None
    if start_frame > 1 or end_frame < 100_000:
        frame_range = (start_frame, end_frame)
        logger.info(
            "Loading data from %s (frames %d–%d)",
            config_manager.data_file_path,
            start_frame,
            end_frame,
        )
    else:
        logger.info("Loading data from %s", config_manager.data_file_path)

    # Build template variables for cache filename substitution
    template_vars: dict[str, str] | None = None
    cache_template = config_manager.cache_filename_template
    if cache_template:
        template_vars = {
            "wavevector_q": f"{config_manager.wavevector_q:.4f}",
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
        }

    data = load_xpcs_data(
        config_manager.data_file_path,
        use_cache=True,
        frame_range=frame_range,
        cache_dir=config_manager.cache_file_path,
        cache_template=cache_template,
        template_vars=template_vars,
        cache_compression=config_manager.cache_compression,
    )

    validation = validate_xpcs_data(data)
    if not validation.is_valid:
        for err in validation.errors:
            logger.error("Data validation error: %s", err)
        raise SystemExit(1)

    for warn in validation.warnings:
        logger.warning("Data validation warning: %s", warn)

    data = _exclude_t0_from_analysis(data)

    # Mandatory diagonal correction: APS two-time XPCS data has inflated
    # diagonal elements (detector shot-noise artifact).  Interpolate from
    # nearest off-diagonal neighbors to bring the diagonal into the
    # physically correct range.  This matches homodyne's mandatory
    # correction in load_experimental_data().
    corrected_c2 = _apply_diagonal_correction(data.c2, width=1, method="interpolate")
    data = XPCSData(
        c2=corrected_c2,
        t1=data.t1,
        t2=data.t2,
        q=data.q,
        phi_angles=data.phi_angles,
        q_values=data.q_values,
        metadata=data.metadata,
    )
    logger.info("Applied mandatory diagonal correction to C2 data")

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

    # Normalize angles to [-180, 180] range.
    phi_angles = [((a + 180.0) % 360.0) - 180.0 for a in phi_angles]

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
    c2 = np.asarray(data.c2)
    prepared: dict[str, Any] = {
        "c2_data": c2,
        "phi_angles": phi_angles,
        "n_angles": len(phi_angles),
        "is_multi_angle": c2.ndim == 3,
    }

    logger.debug(
        "Prepared CMC data: %d angles, c2 shape=%s",
        len(phi_angles),
        c2.shape,
    )
    return prepared
