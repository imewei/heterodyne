"""Per-angle quantile-based β,o estimator for homodyne `constant`-mode parity.

This module provides the data-prep step for ``per_angle_mode="constant"``:
β(φ) and o(φ) are estimated once from quantile analysis of g2 per angle
and held FROZEN through the NLSQ fit.  See
https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html
for the homodyne reference.

Distinct from :func:`heterodyne.core.scaling_utils.compute_averaged_scaling`,
which averages the per-angle vectors into a single scalar pair before
returning — used by the homodyne ``auto`` (averaged) mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerAngleScaling:
    """Per-angle contrast/offset estimates for the FROZEN constant mode."""

    contrast_per_angle: np.ndarray  # shape (n_phi,)
    offset_per_angle: np.ndarray  # shape (n_phi,)


def compute_per_angle_quantile_scaling(
    *,
    g2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> PerAngleScaling:
    """Estimate per-angle β(φ), o(φ) from g2 quantiles.

    For each angle i, the offset estimate is the ``q_low``-quantile of g2
    at that angle (baseline floor) and the contrast estimate is
    ``q_high - q_low`` (the dynamic range above the floor).  Both are
    clipped to the supplied bounds.

    Angles with no observations default to the midpoint of the bounds.

    Parameters
    ----------
    g2 : np.ndarray
        Flat array of g2 observations across all angles, length N.
    phi_indices : np.ndarray
        Integer array of length N giving the angle index ``[0, n_phi)``
        for each observation.
    n_phi : int
        Number of distinct phi angles.
    contrast_bounds, offset_bounds : (float, float)
        Hard bounds; estimates outside are clipped (not rejected).
    q_low, q_high : float
        Quantile probabilities (defaults: 5 % and 95 %).

    Returns
    -------
    PerAngleScaling
        Frozen dataclass with two ``(n_phi,)`` float64 arrays.
    """
    g2_arr = np.asarray(g2, dtype=np.float64)
    phi_arr = np.asarray(phi_indices, dtype=np.int32)

    contrast = np.full(
        n_phi, 0.5 * (contrast_bounds[0] + contrast_bounds[1]), dtype=np.float64
    )
    offset = np.full(
        n_phi, 0.5 * (offset_bounds[0] + offset_bounds[1]), dtype=np.float64
    )

    for i in range(n_phi):
        mask = phi_arr == i
        vals = g2_arr[mask]
        if vals.size == 0:
            continue
        ql = float(np.nanquantile(vals, q_low))
        qh = float(np.nanquantile(vals, q_high))
        contrast[i] = float(np.clip(qh - ql, contrast_bounds[0], contrast_bounds[1]))
        offset[i] = float(np.clip(ql, offset_bounds[0], offset_bounds[1]))

    return PerAngleScaling(
        contrast_per_angle=contrast,
        offset_per_angle=offset,
    )
