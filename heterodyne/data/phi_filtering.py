"""Phi angle filtering for heterodyne XPCS data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.data.xpcs_loader import XPCSData

logger = get_logger(__name__)


@dataclass
class PhiFilterResult:
    """Result of phi angle filtering.

    Note on ``selected_indices``: For ``select_angles`` and
    ``select_angle_range``, these are indices into the *original*
    phi_angles array. For ``average_symmetric_angles``, they are
    output-space indices (0, 1, 2, ...) because symmetric averaging
    produces a new, smaller set of angles with no 1-to-1 mapping
    back to the input.
    """

    # Filtered correlation data
    c2: np.ndarray

    # Selected phi angles
    phi_angles: np.ndarray

    # Indices of selected angles (see class docstring for semantics)
    selected_indices: np.ndarray

    # Number of angles selected
    n_angles: int


class PhiAngleFilter:
    """Filter correlation data by phi angles.

    Heterodyne data typically has multiple phi angles (detector positions).
    This filter selects specific angles or ranges.
    """

    def __init__(
        self,
        phi_angles: np.ndarray | None = None,
    ) -> None:
        """Initialize filter.

        Args:
            phi_angles: Available phi angles in data
        """
        self._phi_angles = phi_angles

    def set_available_angles(self, phi_angles: np.ndarray) -> None:
        """Set available phi angles.

        Args:
            phi_angles: Array of available angles (degrees)
        """
        self._phi_angles = np.asarray(phi_angles)

    def select_angles(
        self,
        c2: np.ndarray,
        target_angles: list[float] | np.ndarray | None = None,
        angle_tolerance: float = 5.0,
    ) -> PhiFilterResult:
        """Select specific phi angles from data.

        Args:
            c2: Correlation data, shape (n_phi, n_t, n_t) or (n_t, n_t)
            target_angles: Angles to select (None for all)
            angle_tolerance: Tolerance for angle matching (degrees)

        Returns:
            PhiFilterResult with filtered data
        """
        if c2.ndim == 2:
            # Single angle data
            return PhiFilterResult(
                c2=c2,
                phi_angles=np.array([0.0])
                if self._phi_angles is None
                else self._phi_angles[:1],
                selected_indices=np.array([0]),
                n_angles=1,
            )

        if self._phi_angles is None:
            raise ValueError("Phi angles not set. Call set_available_angles first.")

        if target_angles is None:
            # Return all angles
            return PhiFilterResult(
                c2=c2,
                phi_angles=self._phi_angles,
                selected_indices=np.arange(len(self._phi_angles)),
                n_angles=len(self._phi_angles),
            )

        target_angles = np.asarray(target_angles)

        # Find closest matching angles
        selected_indices = []
        matched_angles = []

        for target in target_angles:
            # Find closest angle
            diffs = np.abs(self._phi_angles - target)
            min_idx = np.argmin(diffs)

            if diffs[min_idx] <= angle_tolerance:
                selected_indices.append(min_idx)
                matched_angles.append(self._phi_angles[min_idx])
            else:
                logger.warning(
                    "No angle within %.1f° of target %.1f°. Closest: %.1f°",
                    angle_tolerance,
                    target,
                    float(self._phi_angles[min_idx]),
                )

        # Deduplicate indices while preserving order
        seen = set()
        deduped_indices = []
        deduped_angles = []
        for idx, angle in zip(selected_indices, matched_angles, strict=True):
            if idx not in seen:
                seen.add(idx)
                deduped_indices.append(idx)
                deduped_angles.append(angle)
        selected_indices = np.array(deduped_indices)
        matched_angles = np.array(deduped_angles)

        if len(selected_indices) == 0:
            raise ValueError(f"No angles matched targets {target_angles}")

        return PhiFilterResult(
            c2=c2[selected_indices],
            phi_angles=matched_angles,
            selected_indices=selected_indices,
            n_angles=len(selected_indices),
        )

    def select_angle_range(
        self,
        c2: np.ndarray,
        phi_min: float,
        phi_max: float,
    ) -> PhiFilterResult:
        """Select angles within a range.

        Args:
            c2: Correlation data
            phi_min: Minimum angle (degrees)
            phi_max: Maximum angle (degrees)

        Returns:
            PhiFilterResult with filtered data
        """
        if c2.ndim == 2:
            return PhiFilterResult(
                c2=c2,
                phi_angles=np.array([0.0]),
                selected_indices=np.array([0]),
                n_angles=1,
            )

        if self._phi_angles is None:
            raise ValueError("Phi angles not set.")

        if phi_min > phi_max:
            raise ValueError(f"phi_min ({phi_min}) must be <= phi_max ({phi_max})")

        mask = (self._phi_angles >= phi_min) & (self._phi_angles <= phi_max)
        selected_indices = np.where(mask)[0]

        if len(selected_indices) == 0:
            raise ValueError(f"No angles in range [{phi_min}, {phi_max}]")

        return PhiFilterResult(
            c2=c2[selected_indices],
            phi_angles=self._phi_angles[mask],
            selected_indices=selected_indices,
            n_angles=len(selected_indices),
        )

    def average_symmetric_angles(
        self,
        c2: np.ndarray,
        symmetry_center: float = 0.0,
    ) -> PhiFilterResult:
        """Average correlation at symmetric phi angles.

        For angles symmetric about a center (e.g., +45° and -45°),
        average their correlations to improve statistics.

        Args:
            c2: Correlation data
            symmetry_center: Center of symmetry (degrees)

        Returns:
            PhiFilterResult with averaged data
        """
        if c2.ndim == 3 and self._phi_angles is None:
            raise ValueError(
                "phi_angles must be provided for 3D correlation data "
                f"(shape {c2.shape}) to perform symmetric averaging"
            )
        if c2.ndim == 2:
            return PhiFilterResult(
                c2=c2,
                phi_angles=np.array([symmetry_center]),
                selected_indices=np.array([0]),
                n_angles=1,
            )

        # Find symmetric pairs
        relative_angles = self._phi_angles - symmetry_center
        unique_abs_angles = np.unique(np.abs(relative_angles))

        averaged_c2 = []
        averaged_phi = []

        for abs_angle in unique_abs_angles:
            # Find indices of +angle and -angle
            mask = np.isclose(np.abs(relative_angles), abs_angle, atol=0.1)
            if np.sum(mask) > 0:
                averaged_c2.append(np.mean(c2[mask], axis=0))
                averaged_phi.append(symmetry_center + abs_angle)

        # Note: selected_indices are in output-space (post-averaging), not
        # indices into the original phi_angles array. After symmetric averaging,
        # the output has fewer angles than the input, so these indices simply
        # enumerate the new averaged result (0, 1, 2, ...).
        return PhiFilterResult(
            c2=np.array(averaged_c2),
            phi_angles=np.array(averaged_phi),
            selected_indices=np.arange(len(averaged_phi)),
            n_angles=len(averaged_phi),
        )


def filter_by_phi(
    data: XPCSData,
    target_angles: list[float] | None = None,
    angle_tolerance: float = 5.0,
) -> PhiFilterResult:
    """Convenience function to filter data by phi angles.

    Args:
        data: XPCSData with multi-phi correlation
        target_angles: Specific angles to select
        angle_tolerance: Matching tolerance

    Returns:
        PhiFilterResult
    """
    filter = PhiAngleFilter(phi_angles=data.phi_angles)
    return filter.select_angles(
        data.c2,
        target_angles=target_angles,
        angle_tolerance=angle_tolerance,
    )
