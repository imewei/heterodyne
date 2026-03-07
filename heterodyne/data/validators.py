"""Input validation utilities for XPCS data arrays.

Complements the higher-level ``validation.py`` module by providing
fine-grained, composable checks on individual arrays. Each function
returns a list of error strings; an empty list indicates valid input.
"""

from __future__ import annotations

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def validate_correlation_shape(
    c2: np.ndarray,
    expected_shape: tuple[int, ...] | None = None,
) -> list[str]:
    """Validate shape of a correlation matrix.

    Checks that ``c2`` is 2D (single angle) or 3D (multi-angle batch),
    and optionally matches an expected shape.

    Args:
        c2: Correlation array to validate.
        expected_shape: If provided, the exact expected shape.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if c2.ndim not in (2, 3):
        errors.append(
            f"Correlation data must be 2D or 3D, got {c2.ndim}D "
            f"with shape {c2.shape}"
        )
        return errors

    # For 2D: (n_t, n_t); for 3D: (n_phi, n_t, n_t)
    if c2.ndim == 2 and c2.shape[0] != c2.shape[1]:
        errors.append(
            f"2D correlation matrix must be square, got shape {c2.shape}"
        )

    if c2.ndim == 3 and c2.shape[1] != c2.shape[2]:
        errors.append(
            f"3D correlation slices must be square, "
            f"got shape {c2.shape} (axes 1,2 differ)"
        )

    if expected_shape is not None and c2.shape != expected_shape:
        errors.append(
            f"Shape mismatch: got {c2.shape}, expected {expected_shape}"
        )

    return errors


def validate_time_arrays(
    t1: np.ndarray,
    t2: np.ndarray,
) -> list[str]:
    """Validate time arrays for monotonicity and matching lengths.

    Args:
        t1: First time axis array.
        t2: Second time axis array.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if t1.ndim != 1:
        errors.append(f"t1 must be 1D, got {t1.ndim}D with shape {t1.shape}")

    if t2.ndim != 1:
        errors.append(f"t2 must be 1D, got {t2.ndim}D with shape {t2.shape}")

    if t1.ndim == 1 and t2.ndim == 1 and len(t1) != len(t2):
        errors.append(
            f"Time array lengths must match: len(t1)={len(t1)}, len(t2)={len(t2)}"
        )

    # Monotonicity checks (only for 1D arrays)
    if t1.ndim == 1 and len(t1) >= 2:
        if not np.all(np.diff(t1) > 0):
            errors.append("t1 is not strictly monotonically increasing")

    if t2.ndim == 1 and len(t2) >= 2:
        if not np.all(np.diff(t2) > 0):
            errors.append("t2 is not strictly monotonically increasing")

    return errors


def validate_q_range(
    q: np.ndarray,
    q_min: float,
    q_max: float,
) -> list[str]:
    """Validate that wavevector values fall within the specified range.

    Args:
        q: Array of wavevector values.
        q_min: Minimum allowed wavevector.
        q_max: Maximum allowed wavevector.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if q_min > q_max:
        errors.append(f"q_min ({q_min}) must be <= q_max ({q_max})")
        return errors

    if q.size == 0:
        errors.append("Wavevector array is empty")
        return errors

    actual_min = float(np.min(q))
    actual_max = float(np.max(q))

    if actual_min < q_min:
        errors.append(
            f"Wavevector values below q_min: min(q)={actual_min:.4g} < {q_min:.4g}"
        )

    if actual_max > q_max:
        errors.append(
            f"Wavevector values above q_max: max(q)={actual_max:.4g} > {q_max:.4g}"
        )

    return errors


def validate_weights(
    weights: np.ndarray,
    data_shape: tuple[int, ...],
) -> list[str]:
    """Validate weight array for non-negativity and shape compatibility.

    Args:
        weights: Weight array to validate.
        data_shape: Expected shape (must match weights shape).

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if weights.shape != data_shape:
        errors.append(
            f"Weights shape {weights.shape} does not match "
            f"data shape {data_shape}"
        )

    if np.any(weights < 0):
        n_negative = int(np.sum(weights < 0))
        errors.append(
            f"Weights must be non-negative: found {n_negative} negative value(s)"
        )

    if np.any(np.isnan(weights)):
        n_nan = int(np.sum(np.isnan(weights)))
        errors.append(f"Weights contain {n_nan} NaN value(s)")

    if np.any(np.isinf(weights)):
        n_inf = int(np.sum(np.isinf(weights)))
        errors.append(f"Weights contain {n_inf} infinite value(s)")

    return errors


def validate_no_nan(
    arr: np.ndarray,
    name: str,
) -> list[str]:
    """Check that an array contains no NaN or Inf values.

    Args:
        arr: Array to check.
        name: Descriptive name for error messages.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    nan_count = int(np.sum(np.isnan(arr)))
    if nan_count > 0:
        pct = 100.0 * nan_count / arr.size
        errors.append(
            f"'{name}' contains {nan_count} NaN value(s) ({pct:.2f}% of {arr.size} elements)"
        )

    inf_count = int(np.sum(np.isinf(arr)))
    if inf_count > 0:
        pct = 100.0 * inf_count / arr.size
        errors.append(
            f"'{name}' contains {inf_count} Inf value(s) ({pct:.2f}% of {arr.size} elements)"
        )

    return errors
