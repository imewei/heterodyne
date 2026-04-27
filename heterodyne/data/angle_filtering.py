"""Angle-specific filtering utilities for multi-phi XPCS data."""

from __future__ import annotations

from typing import Any

import numpy as np

from heterodyne.data.types import AngleRange
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def filter_by_angle_range(
    c2_3d: np.ndarray,
    phi_angles: np.ndarray,
    angle_range: AngleRange,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter 3D correlation data to angles within a range.

    Args:
        c2_3d: Correlation data, shape (n_phi, n_t, n_t).
        phi_angles: 1D array of phi angles in degrees, length n_phi.
        angle_range: AngleRange specifying (phi_min, phi_max) inclusive.

    Returns:
        Tuple of (filtered_c2, filtered_phi_angles). filtered_c2 has
        shape (n_selected, n_t, n_t).

    Raises:
        ValueError: If c2_3d is not 3D, lengths mismatch, or no angles
            fall within the range.
    """
    if c2_3d.ndim != 3:
        raise ValueError(
            f"c2_3d must be 3D, got {c2_3d.ndim}D with shape {c2_3d.shape}"
        )

    if len(phi_angles) != c2_3d.shape[0]:
        raise ValueError(
            f"phi_angles length ({len(phi_angles)}) must match "
            f"c2_3d first axis ({c2_3d.shape[0]})"
        )

    if angle_range.phi_min > angle_range.phi_max:
        raise ValueError(
            f"phi_min ({angle_range.phi_min}) must be <= phi_max ({angle_range.phi_max})"
        )

    mask = (phi_angles >= angle_range.phi_min) & (phi_angles <= angle_range.phi_max)
    n_selected = int(np.sum(mask))

    if n_selected == 0:
        raise ValueError(
            f"No angles in [{angle_range.phi_min}, {angle_range.phi_max}]. "
            f"Available: {phi_angles.tolist()}"
        )

    logger.debug(
        "Angle range [%.1f, %.1f]: selected %d/%d angles",
        angle_range.phi_min,
        angle_range.phi_max,
        n_selected,
        len(phi_angles),
    )

    return c2_3d[mask], phi_angles[mask]


def select_angles(
    phi_angles: np.ndarray,
    indices: np.ndarray | list[int],
) -> np.ndarray:
    """Select a subset of phi angles by index.

    Args:
        phi_angles: 1D array of phi angles.
        indices: Integer indices into phi_angles.

    Returns:
        Subset of phi_angles at the given indices.

    Raises:
        IndexError: If any index is out of bounds.
    """
    indices_arr = np.asarray(indices, dtype=int)

    if indices_arr.size > 0:
        if np.any(indices_arr < 0) or np.any(indices_arr >= len(phi_angles)):
            out_of_bounds = indices_arr[
                (indices_arr < 0) | (indices_arr >= len(phi_angles))
            ]
            raise IndexError(
                f"Indices {out_of_bounds.tolist()} out of bounds "
                f"for phi_angles of length {len(phi_angles)}"
            )

    return phi_angles[indices_arr]  # type: ignore[no-any-return]


def find_nearest_angle(
    phi_angles: np.ndarray,
    target: float,
) -> int:
    """Find the index of the angle nearest to a target value.

    Args:
        phi_angles: 1D array of available phi angles in degrees.
        target: Target angle in degrees.

    Returns:
        Index of the nearest angle in phi_angles.

    Raises:
        ValueError: If phi_angles is empty.
    """
    if phi_angles.size == 0:
        raise ValueError("phi_angles is empty")

    diffs = np.abs(phi_angles - target)
    idx = int(np.argmin(diffs))

    logger.debug(
        "Nearest angle to %.2f: index %d (%.2f, delta=%.2f)",
        target,
        idx,
        float(phi_angles[idx]),
        float(diffs[idx]),
    )

    return idx


def normalize_angle_to_symmetric_range(
    angle: float | np.ndarray,
) -> float | np.ndarray:
    """Normalize angle(s) to [-180, 180] range.

    Args:
        angle: Angle(s) in degrees.

    Returns:
        Normalized angle(s) in [-180, 180].
    """
    angle_array = np.asarray(angle)
    normalized = angle_array % 360
    normalized = np.where(normalized > 180, normalized - 360, normalized)
    if np.isscalar(angle):
        return float(normalized)
    return normalized  # type: ignore[return-value]


def apply_angle_filtering_for_plot(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering for plotting, extracting config from data dict.

    This is a convenience wrapper that normalizes angles and applies
    range-based filtering using the ``phi_filtering`` section of the
    config stored in *data*.

    Args:
        phi_angles: Array of phi angles in degrees.
        c2_exp: Experimental correlation data, shape ``(n_phi, n_t1, n_t2)``.
        data: Data dictionary; if it contains a ``config`` key with a
            ``phi_filtering`` section, range filtering is applied.

    Returns:
        Tuple of ``(filtered_indices, filtered_phi_angles, filtered_c2_exp)``.
        Returns all angles unchanged when no filtering config is found
        or when filtering yields no matches.
    """
    phi_angles = np.asarray(phi_angles, dtype=float)
    all_indices = list(range(len(phi_angles)))

    # Extract filtering config
    config = data.get("config", {})
    if isinstance(config, dict):
        phi_cfg = config.get("phi_filtering", {})
    else:
        # ConfigManager or similar — try attribute access
        phi_cfg = getattr(config, "phi_filtering", {}) or {}

    if not phi_cfg or not phi_cfg.get("enabled", False):
        logger.debug("Phi filtering not enabled for plot, using all %d angles", len(phi_angles))
        return all_indices, phi_angles, c2_exp

    target_ranges = phi_cfg.get("target_ranges", [])
    if not target_ranges:
        return all_indices, phi_angles, c2_exp

    # Normalize angles to [-180, 180]
    normalized = normalize_angle_to_symmetric_range(phi_angles)

    # Default tolerance used when target_ranges entries are scalar angles
    tol = float(phi_cfg.get("tolerance", 5.0))

    # Apply OR logic: angle selected if it falls within ANY target range
    selected_mask = np.zeros(len(normalized), dtype=bool)
    for rng in target_ranges:
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo = float(normalize_angle_to_symmetric_range(rng[0]))
            hi = float(normalize_angle_to_symmetric_range(rng[1]))
        elif isinstance(rng, dict):
            # Dict format: {min_angle: X, max_angle: Y} (from YAML config)
            lo = float(normalize_angle_to_symmetric_range(rng.get("min_angle", -10.0)))
            hi = float(normalize_angle_to_symmetric_range(rng.get("max_angle", 10.0)))
        elif isinstance(rng, (int, float, np.floating, np.integer)):
            # Single target angle: expand to [center-tol, center+tol]
            center = float(normalize_angle_to_symmetric_range(float(rng)))
            lo, hi = center - tol, center + tol
        else:
            continue
        if lo <= hi:
            selected_mask |= (normalized >= lo) & (normalized <= hi)
        else:
            # Wrap-around range (e.g. [170, -170])
            selected_mask |= (normalized >= lo) | (normalized <= hi)

    if not np.any(selected_mask):
        logger.warning(
            "Angle filtering matched no angles; returning all %d angles",
            len(phi_angles),
        )
        return all_indices, phi_angles, c2_exp

    indices = list(np.where(selected_mask)[0])
    filtered_phi = phi_angles[selected_mask]
    filtered_c2 = c2_exp[selected_mask] if c2_exp.ndim == 3 else c2_exp

    logger.info(
        "Angle filtering for plot: %d/%d angles selected: %s",
        len(indices),
        len(phi_angles),
        filtered_phi.tolist(),
    )
    return indices, filtered_phi, filtered_c2


def compute_angle_quality(
    c2_3d: np.ndarray,
    phi_angles: np.ndarray,
) -> dict[str, Any]:
    """Compute per-angle quality metrics for multi-phi data.

    For each angle slice, computes signal-to-noise ratio (SNR),
    mean, and standard deviation from finite off-diagonal elements.

    Args:
        c2_3d: Correlation data, shape (n_phi, n_t, n_t).
        phi_angles: 1D array of phi angles, length n_phi.

    Returns:
        Dictionary with keys:
            - ``phi_angles``: Input angles (for reference).
            - ``snr``: Per-angle SNR (mean / std of off-diagonal finite values).
            - ``mean``: Per-angle mean of off-diagonal finite values.
            - ``std``: Per-angle std of off-diagonal finite values.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if c2_3d.ndim != 3:
        raise ValueError(f"c2_3d must be 3D, got {c2_3d.ndim}D")

    if len(phi_angles) != c2_3d.shape[0]:
        raise ValueError(
            f"phi_angles length ({len(phi_angles)}) must match "
            f"c2_3d first axis ({c2_3d.shape[0]})"
        )

    n_phi = c2_3d.shape[0]
    n_t = c2_3d.shape[1]

    # Off-diagonal mask (shared across all slices)
    off_diag = ~np.eye(n_t, dtype=bool)

    snr_arr = np.zeros(n_phi)
    mean_arr = np.zeros(n_phi)
    std_arr = np.zeros(n_phi)

    for i in range(n_phi):
        slice_data = c2_3d[i]
        off_diag_vals = slice_data[off_diag]
        finite_vals = off_diag_vals[np.isfinite(off_diag_vals)]

        if finite_vals.size == 0:
            snr_arr[i] = 0.0
            mean_arr[i] = np.nan
            std_arr[i] = np.nan
            continue

        m = float(np.mean(finite_vals))
        s = float(np.std(finite_vals))

        mean_arr[i] = m
        std_arr[i] = s
        snr_arr[i] = abs(m) / s if s > 0 else 0.0

    return {
        "phi_angles": phi_angles.copy(),
        "snr": snr_arr,
        "mean": mean_arr,
        "std": std_arr,
    }
