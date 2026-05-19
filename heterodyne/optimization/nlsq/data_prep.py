"""Data preparation for NLSQ fitting.

Converts correlation matrices and weights into the flat arrays that
scipy.optimize.least_squares expects, and constructs appropriate
weight arrays from data statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class PreparedData:
    """Container for prepared optimization data.

    Attributes:
        xdata: Flattened independent variable data
        ydata: Flattened dependent variable data (observations)
        n_data: Total number of data points
        n_phi: Number of unique phi angles
        phi_unique: Unique phi angle values
    """

    xdata: np.ndarray
    ydata: np.ndarray
    n_data: int
    n_phi: int
    phi_unique: np.ndarray


@dataclass
class ExpandedParameters:
    """Container for expanded per-angle parameters.

    Attributes:
        params: Expanded parameter array
        bounds: Expanded bounds tuple (lower, upper)
        n_params: Total number of parameters
        n_physical: Number of physical (non-scaling) parameters
        n_angles: Number of phi angles
    """

    params: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None
    n_params: int
    n_physical: int
    n_angles: int


def expand_per_angle_parameters(
    compact_params: np.ndarray,
    compact_bounds: tuple[np.ndarray, np.ndarray] | None,
    n_angles: int,
    n_physical: int,
    logger: Any = None,
) -> ExpandedParameters:
    """Expand compact parameters to per-angle format.

    When per_angle_scaling=True with N angles, parameters are structured as:
    - N contrast parameters (one per angle)
    - N offset parameters (one per angle)
    - n_physical physical parameters

    Args:
        compact_params: Compact parameter array (n_physical + 2 elements)
        compact_bounds: Compact bounds tuple or None
        n_angles: Number of phi angles
        n_physical: Number of physical parameters
        logger: Optional logger for diagnostics

    Returns:
        ExpandedParameters with per-angle parameters and bounds

    Raises:
        ValueError: If parameter count doesn't match expected
    """
    expected_compact = n_physical + 2
    if len(compact_params) != expected_compact:
        raise ValueError(
            f"Expected {expected_compact} compact parameters "
            f"(2 scaling + {n_physical} physical), got {len(compact_params)}"
        )

    contrast = compact_params[0]
    offset = compact_params[1]
    physical = compact_params[2:]

    # Expand: [contrast_0..N, offset_0..N, physical...]
    expanded_params = np.concatenate(
        [
            np.full(n_angles, contrast),
            np.full(n_angles, offset),
            physical,
        ]
    )

    expanded_bounds: tuple[np.ndarray, np.ndarray] | None = None
    if compact_bounds is not None:
        lower_compact, upper_compact = compact_bounds
        contrast_lo, offset_lo = lower_compact[0], lower_compact[1]
        contrast_hi, offset_hi = upper_compact[0], upper_compact[1]
        physical_lo = lower_compact[2:]
        physical_hi = upper_compact[2:]

        expanded_lower = np.concatenate(
            [
                np.full(n_angles, contrast_lo),
                np.full(n_angles, offset_lo),
                physical_lo,
            ]
        )
        expanded_upper = np.concatenate(
            [
                np.full(n_angles, contrast_hi),
                np.full(n_angles, offset_hi),
                physical_hi,
            ]
        )
        expanded_bounds = (expanded_lower, expanded_upper)

    if logger:
        logger.info(
            "Expanded %d compact params to %d per-angle params (%d angles):",
            expected_compact,
            len(expanded_params),
            n_angles,
        )
        logger.info(
            "    - Contrast per angle: %d (indices 0 to %d)", n_angles, n_angles - 1
        )
        logger.info(
            "    - Offset per angle: %d (indices %d to %d)",
            n_angles,
            n_angles,
            2 * n_angles - 1,
        )
        logger.info(
            "    - Physical: %d (indices %d to %d)",
            n_physical,
            2 * n_angles,
            2 * n_angles + n_physical - 1,
        )

    return ExpandedParameters(
        params=expanded_params,
        bounds=expanded_bounds,
        n_params=len(expanded_params),
        n_physical=n_physical,
        n_angles=n_angles,
    )


def validate_bounds(
    bounds: tuple[np.ndarray, np.ndarray] | None,
    n_params: int,
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate parameter bounds.

    Args:
        bounds: Bounds tuple (lower, upper) or None
        n_params: Expected number of parameters
        logger: Optional logger for diagnostics

    Returns:
        Validated bounds or None

    Raises:
        ValueError: If bounds are invalid
    """
    if bounds is None:
        return None

    lower, upper = bounds

    if len(lower) != n_params or len(upper) != n_params:
        raise ValueError(
            f"Bounds dimension mismatch: expected {n_params}, "
            f"got lower={len(lower)}, upper={len(upper)}"
        )

    invalid_indices = np.where(lower > upper)[0]
    if len(invalid_indices) > 0:
        raise ValueError(
            f"Invalid bounds at indices {invalid_indices}: "
            f"lower > upper. Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}"
        )

    return (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def validate_initial_params(
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any = None,
) -> np.ndarray:
    """Validate and clip initial parameters to bounds.

    Args:
        params: Initial parameter guess, shape (n_params,)
        bounds: Parameter bounds (lower, upper) or None
        logger: Optional logger for diagnostics

    Returns:
        Validated (and possibly clipped) parameters
    """
    params = np.asarray(params, dtype=np.float64)

    if bounds is None:
        return params

    lower, upper = bounds
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    clipped: np.ndarray = np.clip(params, lower, upper)

    if not np.allclose(params, clipped):
        n_clipped = int(np.sum(~np.isclose(params, clipped)))
        if logger:
            logger.warning("Clipped %d initial parameters to bounds", n_clipped)

    return clipped


def convert_bounds_to_nlsq_format(
    bounds: tuple[np.ndarray, np.ndarray] | tuple[list, list] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert bounds to NLSQ-compatible format.

    NLSQ expects bounds as (lower_array, upper_array) with float64 dtype.

    Args:
        bounds: Input bounds in any supported format, or None

    Returns:
        Bounds as (lower_array, upper_array) with float64 dtype, or None
    """
    if bounds is None:
        return None

    lower, upper = bounds

    # Convert to numpy arrays with float64 dtype
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    return (lower, upper)


def build_parameter_labels(
    per_angle_scaling: bool,
    n_phi: int,
    physical_param_names: list[str],
) -> list[str]:
    """Build human-readable parameter labels.

    Args:
        per_angle_scaling: Whether per-angle scaling is enabled
        n_phi: Number of phi angles
        physical_param_names: Names of physical parameters

    Returns:
        List of parameter label strings
    """
    labels: list[str] = []
    if per_angle_scaling:
        labels.extend([f"contrast[{i}]" for i in range(n_phi)])
        labels.extend([f"offset[{i}]" for i in range(n_phi)])
    else:
        labels.extend(["contrast", "offset"])
    labels.extend(physical_param_names)
    return labels


def classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    atol: float = 1e-9,
) -> list[str]:
    """Classify parameter status relative to bounds.

    Args:
        values: Parameter values
        lower: Lower bounds or None
        upper: Upper bounds or None
        atol: Absolute tolerance for bound comparison

    Returns:
        List of status strings: 'active', 'at_lower_bound', or 'at_upper_bound'
    """
    if lower is None or upper is None:
        return ["active"] * len(values)

    statuses: list[str] = []
    for value, lo, hi in zip(values, lower, upper, strict=False):
        if np.isclose(value, lo, atol=atol * (1.0 + abs(lo))):
            statuses.append("at_lower_bound")
        elif np.isclose(value, hi, atol=atol * (1.0 + abs(hi))):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def flatten_upper_triangle(
    matrix: np.ndarray,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Flatten the upper triangle of a symmetric matrix.

    For a two-time correlation matrix C2(t1, t2), only the upper triangle
    (t2 >= t1) contains independent data. This extracts those elements
    in row-major order for residual computation.

    Args:
        matrix: Square matrix of shape (N, N)
        include_diagonal: Whether to include diagonal elements

    Returns:
        1D array of upper-triangle values
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

    n = matrix.shape[0]
    if include_diagonal:
        indices = np.triu_indices(n, k=0)
    else:
        indices = np.triu_indices(n, k=1)

    return matrix[indices]


def unflatten_upper_triangle(
    flat: np.ndarray,
    n: int,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Reconstruct symmetric matrix from upper-triangle values.

    Args:
        flat: 1D array of upper-triangle values
        n: Matrix size
        include_diagonal: Whether flat includes diagonal

    Returns:
        Symmetric matrix of shape (n, n)
    """
    matrix = np.zeros((n, n))
    if include_diagonal:
        indices = np.triu_indices(n, k=0)
    else:
        indices = np.triu_indices(n, k=1)

    expected_len = len(indices[0])
    if len(flat) != expected_len:
        raise ValueError(
            f"Expected {expected_len} values for n={n} "
            f"(include_diagonal={include_diagonal}), got {len(flat)}"
        )

    matrix[indices] = flat
    # Mirror to lower triangle
    matrix = matrix + matrix.T
    if include_diagonal:
        np.fill_diagonal(matrix, np.diag(matrix) / 2)

    return matrix


def compute_weights(
    c2_data: np.ndarray,
    method: str = "uniform",
    sigma: np.ndarray | None = None,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    """Compute weight array for NLSQ fitting.

    Args:
        c2_data: Correlation data, shape (N, N)
        method: Weight method:
            - 'uniform': Equal weights (1.0)
            - 'inverse_variance': 1/sigma² from provided sigma
            - 'data_amplitude': 1/|data| for heteroscedastic data
        sigma: Standard deviation array for 'inverse_variance' method
        exclude_diagonal: Zero out diagonal weights (diagonal often noisy)

    Returns:
        Weight array of shape (N, N) where weights = 1/sigma²
    """
    if method == "uniform":
        weights = np.ones_like(c2_data)

    elif method == "inverse_variance":
        if sigma is None:
            raise ValueError("sigma required for inverse_variance weighting")
        if sigma.shape != c2_data.shape:
            raise ValueError(
                f"sigma shape {sigma.shape} doesn't match data shape {c2_data.shape}"
            )
        # Clamp sigma to avoid division by zero
        sigma_safe = np.maximum(np.abs(sigma), 1e-30)
        weights = 1.0 / (sigma_safe**2)

    elif method == "data_amplitude":
        amplitude = np.maximum(np.abs(c2_data), 1e-30)
        weights = 1.0 / amplitude

    else:
        raise ValueError(f"Unknown weight method: {method!r}")

    if exclude_diagonal:
        np.fill_diagonal(weights, 0.0)

    return weights


def prepare_fit_data(
    c2_data: np.ndarray,
    weights: np.ndarray | None = None,
    use_upper_triangle: bool = True,
    exclude_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare correlation data and weights for least-squares fitting.

    Flattens data and weights into 1D arrays suitable for
    scipy.optimize.least_squares, optionally using only the
    upper triangle of the symmetric matrix.

    Args:
        c2_data: Correlation matrix, shape (N, N)
        weights: Optional weight matrix, shape (N, N). Defaults to uniform.
        use_upper_triangle: Use only upper triangle (recommended for symmetry)
        exclude_diagonal: Exclude diagonal from fit

    Returns:
        Tuple of:
        - data_flat: 1D flattened data
        - weights_flat: 1D flattened weights (sqrt for residual scaling)
        - n_data: Number of data points
    """
    if weights is None:
        weights = np.ones_like(c2_data)

    if exclude_diagonal:
        weights = weights.copy()
        np.fill_diagonal(weights, 0.0)

    if use_upper_triangle:
        data_flat = flatten_upper_triangle(c2_data)
        weights_flat = flatten_upper_triangle(weights)
    else:
        data_flat = c2_data.ravel()
        weights_flat = weights.ravel()

    # Convert weights to sqrt for residual scaling:
    # residual_i = sqrt(w_i) * (model_i - data_i)
    # so that sum(residual²) = sum(w * (model - data)²)
    sqrt_weights = np.sqrt(np.maximum(weights_flat, 0.0))

    n_data = int(np.sum(sqrt_weights > 0))

    logger.debug(
        "Prepared fit data: %d points (%d non-zero weight) from (%d, %d) matrix",
        len(data_flat),
        n_data,
        c2_data.shape[0],
        c2_data.shape[1],
    )

    return data_flat, sqrt_weights, n_data


def compute_degrees_of_freedom(
    n_data: int,
    n_params: int,
) -> int:
    """Compute degrees of freedom for chi-squared calculation.

    Args:
        n_data: Number of data points with non-zero weight
        n_params: Number of varying parameters

    Returns:
        Degrees of freedom (n_data - n_params), minimum 1
    """
    dof = max(n_data - n_params, 1)
    if n_data <= n_params:
        logger.warning(
            "Underdetermined system: %d data points, %d parameters", n_data, n_params
        )
    return dof
