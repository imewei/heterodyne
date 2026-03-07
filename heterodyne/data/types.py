"""Core type definitions for XPCS data handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class DataSlice(NamedTuple):
    """A single slice of XPCS correlation data at a specific angle.

    Attributes:
        c2: Two-time correlation matrix, shape (n_t, n_t).
        t1: Time array for first axis.
        t2: Time array for second axis.
        phi_angle: Azimuthal angle in degrees.
        weights: Optional per-element weights, same shape as c2.
    """

    c2: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    phi_angle: float
    weights: np.ndarray | None


class QRange(NamedTuple):
    """Wavevector range for data selection.

    Attributes:
        q_min: Minimum wavevector value (inverse angstroms).
        q_max: Maximum wavevector value (inverse angstroms).
    """

    q_min: float
    q_max: float


class AngleRange(NamedTuple):
    """Azimuthal angle range for data selection.

    Attributes:
        phi_min: Minimum angle in degrees.
        phi_max: Maximum angle in degrees.
    """

    phi_min: float
    phi_max: float


@dataclass
class FilterResult:
    """Result of a data filtering operation.

    Attributes:
        data: Filtered data array.
        mask: Boolean mask indicating retained elements (True = kept).
        n_removed: Number of elements removed by the filter.
        reason: Human-readable description of the filtering applied.
    """

    data: np.ndarray
    mask: np.ndarray
    n_removed: int
    reason: str
