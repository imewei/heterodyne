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
    logger.info(
        "Loaded XPCS data: c2 shape=%s, %d phi angles",
        data.c2.shape,
        len(data.phi_angles) if data.phi_angles is not None else 0,
    )

    return data


def _apply_phi_filtering(
    data_phi_angles: np.ndarray,
    phi_cfg: dict[str, Any],
) -> list[float] | None:
    """Apply phi_filtering config to select angles from data.

    Returns matched data angles as a list, or None if no match.
    """
    target_ranges = phi_cfg.get("target_ranges", [])
    if not target_ranges:
        return None

    # Inline normalization to [-180, 180] to keep types unambiguous.
    _arr = np.asarray(data_phi_angles, dtype=float)
    normalized: np.ndarray = np.where((_arr % 360) > 180, (_arr % 360) - 360, _arr % 360)
    tol = float(phi_cfg.get("tolerance", 5.0))
    selected_mask = np.zeros(len(normalized), dtype=bool)

    def _norm_scalar(a: float) -> float:
        v = a % 360.0
        return v - 360.0 if v > 180.0 else v

    for rng in target_ranges:
        if isinstance(rng, dict):
            lo = _norm_scalar(float(rng.get("min_angle", -10.0)))
            hi = _norm_scalar(float(rng.get("max_angle", 10.0)))
        elif isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo = _norm_scalar(float(rng[0]))
            hi = _norm_scalar(float(rng[1]))
        elif isinstance(rng, (int, float)):
            center = _norm_scalar(float(rng))
            lo, hi = center - tol, center + tol
        else:
            continue

        if lo <= hi:
            selected_mask |= (normalized >= lo) & (normalized <= hi)
        else:
            # Wrap-around range (e.g. [170, -170])
            selected_mask |= (normalized >= lo) | (normalized <= hi)

    if not np.any(selected_mask):
        return None

    return [float(a) for a in data_phi_angles[selected_mask]]


def resolve_phi_angles(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    data_phi_angles: np.ndarray | None = None,
) -> list[float]:
    """Determine phi angles from CLI args or configuration.

    Priority: CLI --phi > config scattering.phi_angles > phi_filtering > default [0.0].

    Args:
        args: Parsed CLI arguments (may have .phi attribute).
        config_manager: Configuration manager.
        data_phi_angles: Actual phi angles present in the loaded data (for
            phi_filtering). When provided, phi_filtering.target_ranges is applied
            against these angles instead of falling back to [0.0].

    Returns:
        List of phi angles in degrees.
    """
    phi_angles: list[float] | None = getattr(args, "phi", None)
    if phi_angles is not None:
        logger.debug("Phi angles from CLI: %s", phi_angles)
    else:
        phi_angles = config_manager.phi_angles
        if phi_angles is not None:
            logger.debug("Phi angles from config scattering.phi_angles: %s", phi_angles)
        elif data_phi_angles is not None:
            phi_cfg: dict[str, Any] = config_manager._config.get("phi_filtering", {})  # type: ignore[attr-defined]
            if phi_cfg.get("enabled", False):
                filtered = _apply_phi_filtering(np.asarray(data_phi_angles, dtype=float), phi_cfg)
                if filtered is not None:
                    phi_angles = filtered
                    logger.debug("Phi angles from phi_filtering: %s", phi_angles)
                else:
                    fallback = phi_cfg.get("fallback_to_all_angles", True)
                    if fallback:
                        phi_angles = [float(a) for a in data_phi_angles]
                        logger.debug(
                            "phi_filtering matched nothing; falling back to all %d angles",
                            len(phi_angles),
                        )
                    else:
                        phi_angles = [0.0]
                        logger.debug("phi_filtering matched nothing; defaulting to [0.0]")
            else:
                phi_angles = [0.0]
                logger.debug("phi_filtering disabled; defaulting to [0.0]")
        else:
            phi_angles = [0.0]
            logger.debug("No phi angle source; defaulting to [0.0]")

    logger.debug("Raw phi angles before normalization: %s", phi_angles)
    # Normalize angles to [-180, 180] range.
    phi_angles = [((a + 180.0) % 360.0) - 180.0 for a in phi_angles]
    logger.debug("Normalized phi angles: %s", phi_angles)

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
