"""Diagonal correction for two-time correlation matrices.

In XPCS two-time correlation matrices c2(t1, t2), the diagonal (t1 = t2)
has different statistics than off-diagonal elements due to photon shot noise
and detector artifacts. The diagonal excess arises because the same speckle
pattern is correlated with itself at zero lag, introducing a delta-function
contribution that biases the correlation.

This module provides utilities for:
- Masking diagonal and near-diagonal elements
- Correcting diagonal artifacts via interpolation, NaN masking, or symmetry
- Estimating the magnitude of diagonal excess
- Computing weight arrays that exclude the diagonal band
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def compute_diagonal_mask(n_times: int, width: int = 1) -> jnp.ndarray:
    """Compute a boolean mask for the diagonal band of a two-time matrix.

    Returns True for elements within ``width`` of the main diagonal,
    i.e., where |i - j| < width.

    Args:
        n_times: Size of the square matrix (number of time points).
        width: Half-width of the diagonal band. ``width=1`` masks only
            the main diagonal; ``width=2`` masks the diagonal and its
            immediate neighbors.

    Returns:
        Boolean array of shape (n_times, n_times). True on the diagonal
        band, False elsewhere.

    Raises:
        ValueError: If ``width < 1``.
    """
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)
    idx = jnp.arange(n_times)
    return jnp.abs(idx[:, None] - idx[None, :]) < width


def apply_diagonal_correction(
    c2: jnp.ndarray,
    width: int = 1,
    method: str = "interpolate",
) -> jnp.ndarray:
    """Correct diagonal artifacts in a two-time correlation matrix.

    The diagonal of c2 (and optionally near-diagonal elements within
    ``width``) is replaced according to the chosen method.

    Args:
        c2: Two-time correlation matrix, shape (N, N). Must be square.
        width: Half-width of the diagonal band to correct. ``width=1``
            corrects only the main diagonal.
        method: Correction strategy. One of:

            - ``"interpolate"``: Replace diagonal elements with the mean
              of their nearest off-diagonal neighbors. At boundaries,
              available neighbors are reused (clamped indexing).
            - ``"mask"``: Set diagonal elements to NaN for exclusion in
              downstream fitting.
            - ``"mirror"``: Replace using matrix symmetry, c2[i,j] from
              c2[j,i]. For the exact diagonal (i=j) this is a no-op;
              useful when ``width > 1`` to fill near-diagonal from the
              transposed side.

    Returns:
        Corrected correlation matrix, shape (N, N).

    Raises:
        ValueError: If ``method`` is not one of the supported strategies,
            or if ``width < 1``.
    """
    valid_methods = ("interpolate", "mask", "mirror")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)

    if method == "interpolate":
        return _apply_interpolation(c2, width)
    if method == "mask":
        return _apply_nan_mask(c2, width)
    # method == "mirror"
    return _apply_mirror(c2, width)


def _apply_interpolation(c2: jnp.ndarray, width: int) -> jnp.ndarray:
    """Replace diagonal band with interpolated values from neighbors.

    For width=1, each diagonal element c2[i,i] is replaced with the mean
    of its 4 nearest off-diagonal neighbors: (i-1,i), (i+1,i), (i,i-1),
    (i,i+1). Boundary elements use clamped indices.

    For width>1, each masked element (i,j) with |i-j| < width is replaced
    with the average of the two nearest elements along the perpendicular
    direction to the diagonal at distance ``width`` from the diagonal.
    """
    n = c2.shape[0]

    if width == 1:
        # Fast path: only the main diagonal
        i_idx = jnp.arange(n)
        i_prev = jnp.maximum(i_idx - 1, 0)
        i_next = jnp.minimum(i_idx + 1, n - 1)

        # Average of 4 nearest off-diagonal neighbors
        neighbor_avg = (
            c2[i_prev, i_idx]
            + c2[i_next, i_idx]
            + c2[i_idx, i_prev]
            + c2[i_idx, i_next]
        ) / 4.0

        diag_mask = jnp.eye(n, dtype=jnp.bool_)
        replacement = jnp.diag(neighbor_avg)
        return jnp.where(diag_mask, replacement, c2)

    # General case: width > 1
    idx_i, idx_j = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    diff = idx_i - idx_j
    mask = jnp.abs(diff) < width

    # For each element, find the two nearest off-band neighbors along
    # the row axis (perpendicular to diagonal).
    # Shift needed to reach the band edge from element (i, j):
    shift = width - jnp.abs(diff)
    i_above = jnp.clip(idx_i - shift, 0, n - 1)
    i_below = jnp.clip(idx_i + shift, 0, n - 1)

    interpolated = (c2[i_above, idx_j] + c2[i_below, idx_j]) / 2.0
    return jnp.where(mask, interpolated, c2)


def _apply_nan_mask(c2: jnp.ndarray, width: int) -> jnp.ndarray:
    """Set diagonal band to NaN."""
    mask = compute_diagonal_mask(c2.shape[0], width)
    return jnp.where(mask, jnp.nan, c2)


def _apply_mirror(c2: jnp.ndarray, width: int) -> jnp.ndarray:
    """Replace diagonal band using matrix symmetry c2[i,j] = c2[j,i].

    For the exact diagonal (i == j) this is a no-op since c2[i,i] = c2[i,i].
    For near-diagonal elements (0 < |i-j| < width), the transposed value
    provides an independent estimate from the opposite side of the diagonal.
    """
    mask = compute_diagonal_mask(c2.shape[0], width)
    return jnp.where(mask, c2.T, c2)


def estimate_diagonal_excess(
    c2: jnp.ndarray,
    width: int = 1,
) -> dict[str, Any]:
    """Estimate the statistical excess of diagonal vs off-diagonal elements.

    Computes summary statistics quantifying the diagonal artifact in c2.
    The "excess" is the difference between the mean diagonal value and
    the mean off-diagonal value, normalized by the off-diagonal standard
    deviation.

    Args:
        c2: Two-time correlation matrix, shape (N, N).
        width: Half-width of the diagonal band. Elements with |i - j| < width
            are considered "diagonal."

    Returns:
        Dictionary with keys:

        - ``"mean_diagonal"``: Mean of elements on the diagonal band.
        - ``"mean_off_diagonal"``: Mean of elements outside the diagonal band.
        - ``"mean_excess"``: Difference (diagonal mean - off-diagonal mean).
        - ``"std_diagonal"``: Standard deviation of diagonal band elements.
        - ``"std_off_diagonal"``: Standard deviation of off-diagonal elements.
        - ``"std_ratio"``: Ratio std_diagonal / std_off_diagonal.
        - ``"excess_sigma"``: Excess normalized by off-diagonal std
          (mean_excess / std_off_diagonal). A large value (>> 1) indicates
          a significant diagonal artifact.

    Raises:
        ValueError: If ``width < 1``.
    """
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)

    mask = compute_diagonal_mask(c2.shape[0], width)
    off_mask = ~mask

    diag_vals = c2[mask]
    off_vals = c2[off_mask]

    mean_diag = jnp.mean(diag_vals)
    mean_off = jnp.mean(off_vals)
    std_diag = jnp.std(diag_vals)
    std_off = jnp.std(off_vals)

    mean_excess = mean_diag - mean_off
    # Protect against division by zero when off-diagonal is constant
    std_ratio = std_diag / jnp.maximum(std_off, 1e-30)
    excess_sigma = mean_excess / jnp.maximum(std_off, 1e-30)

    return {
        "mean_diagonal": mean_diag,
        "mean_off_diagonal": mean_off,
        "mean_excess": mean_excess,
        "std_diagonal": std_diag,
        "std_off_diagonal": std_off,
        "std_ratio": std_ratio,
        "excess_sigma": excess_sigma,
    }


def compute_weights_excluding_diagonal(
    shape: tuple[int, int],
    width: int = 1,
) -> jnp.ndarray:
    """Compute a weight array with zeros on the diagonal band.

    Returns an array of ones with zeros where |i - j| < width. This is
    useful for constructing weight matrices that exclude the diagonal
    artifact when fitting correlation models.

    Args:
        shape: Shape of the output array (n_rows, n_cols). Typically
            square (N, N) for a two-time correlation matrix, but
            rectangular shapes are supported.
        width: Half-width of the diagonal band to zero out.

    Returns:
        Float array of shape ``shape`` with 1.0 off-diagonal and 0.0
        on the diagonal band.

    Raises:
        ValueError: If ``width < 1``.
    """
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)
    n_rows, n_cols = shape
    idx_i = jnp.arange(n_rows)
    idx_j = jnp.arange(n_cols)
    dist = jnp.abs(idx_i[:, None] - idx_j[None, :])
    return jnp.where(dist < width, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Batch diagonal correction + backend abstraction (Task 1.5)
# ---------------------------------------------------------------------------


def _is_jax_array(array: Any) -> bool:
    """Check if *array* is a JAX array.

    Handles both the modern ``jax.Array`` type and the legacy
    ``jaxlib.xla_extension.ArrayImpl`` so that the check works across
    JAX versions. Returns ``False`` if JAX is not importable.
    """
    try:
        import jax  # noqa: F811

        if isinstance(array, jax.Array):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        from jaxlib.xla_extension import ArrayImpl

        if isinstance(array, ArrayImpl):
            return True
    except Exception:  # noqa: BLE001
        pass

    return False


def _resolve_backend(array: Any) -> str:
    """Detect whether *array* is backed by JAX or NumPy.

    Returns:
        ``"jax"`` or ``"numpy"``.
    """
    backend = "jax" if _is_jax_array(array) else "numpy"
    logger.debug("Detected backend: %s for array type %s", backend, type(array).__name__)
    return backend


def _apply_statistical_correction(
    c2: Any,
    width: int,
    estimator: str = "mean",
) -> Any:
    """Replace diagonal band with statistical estimates from nearby off-diagonal elements.

    For each element on the diagonal band (|i - j| < *width*), a replacement
    value is computed from a window of off-diagonal elements at distances
    *width* to *width* + 3 from the diagonal.

    Args:
        c2: Two-time correlation matrix, shape (N, N). May be a JAX or
            NumPy array.
        width: Half-width of the diagonal band to correct.
        estimator: One of ``"mean"``, ``"median"``, or ``"trimmed_mean"``.
            For ``"trimmed_mean"`` the top and bottom 10 % of values in the
            window are removed before averaging.

    Returns:
        Corrected matrix of the same type and shape as *c2*.

    Raises:
        ValueError: If *estimator* is not recognised or *width* < 1.
    """
    valid_estimators = ("mean", "median", "trimmed_mean")
    if estimator not in valid_estimators:
        msg = f"estimator must be one of {valid_estimators}, got {estimator!r}"
        raise ValueError(msg)
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)

    backend = _resolve_backend(c2)

    if backend == "jax":
        xp = jnp
    else:
        import numpy as np

        xp = np  # type: ignore[assignment]

    n = c2.shape[0]
    idx_i, idx_j = xp.meshgrid(xp.arange(n), xp.arange(n), indexing="ij")
    diff = idx_i - idx_j
    abs_diff = xp.abs(diff)
    band_mask = abs_diff < width

    # Window: elements at distance [width, width+3) from the diagonal
    window_lo = width
    window_hi = width + 3

    # For each (i, j) on the band, collect window elements along the row.
    # We gather elements at offsets that place them in the window band.
    # Strategy: for each row i, the window elements are those columns j'
    # where window_lo <= |i - j'| < window_hi.  We compute the row-wise
    # statistic for the window, then broadcast back to all band elements
    # in that row.

    # Build per-row statistics from the window band
    row_stats = xp.zeros(n, dtype=c2.dtype)
    col_idx = xp.arange(n)

    for i in range(n):
        dists = xp.abs(col_idx - i)
        window_mask = (dists >= window_lo) & (dists < window_hi)
        window_vals = c2[i][window_mask]

        if window_vals.size == 0:
            # Fallback: widen the window if nothing available at boundary
            window_mask = dists >= window_lo
            window_vals = c2[i][window_mask]

        if window_vals.size == 0:
            # Edge case: matrix too small — keep original value
            row_stats = row_stats.at[i].set(c2[i, i]) if backend == "jax" else row_stats
            if backend == "numpy":
                row_stats[i] = c2[i, i]
            continue

        if estimator == "mean":
            stat = xp.mean(window_vals)
        elif estimator == "median":
            stat = xp.median(window_vals)
        else:
            # trimmed_mean: remove top/bottom 10%
            k = max(1, int(round(0.1 * window_vals.size)))
            sorted_vals = xp.sort(window_vals)
            trimmed = sorted_vals[k : window_vals.size - k]
            if trimmed.size == 0:
                trimmed = sorted_vals
            stat = xp.mean(trimmed)

        if backend == "jax":
            row_stats = row_stats.at[i].set(stat)
        else:
            row_stats[i] = stat

    # Broadcast row statistics into a full matrix and apply where band_mask is True
    replacement = xp.broadcast_to(row_stats[:, None], (n, n))
    result = xp.where(band_mask, replacement, c2)
    return result


def apply_diagonal_correction_batch(
    c2_batch: Any,
    width: int = 1,
    method: str = "interpolate",
) -> Any:
    """Apply diagonal correction to a batch of correlation matrices.

    Args:
        c2_batch: Array of shape ``(n_batch, N, N)`` or ``(N, N)``. May be
            a JAX or NumPy array.
        width: Half-width of the diagonal band to correct.
        method: Correction strategy — one of ``"interpolate"``, ``"mask"``,
            ``"mirror"``, or ``"statistical"``.

    Returns:
        Corrected array with the same shape and type as *c2_batch*.

    Raises:
        ValueError: If the input is not 2-D or 3-D, or if *method* is
            unknown.
    """
    ndim = c2_batch.ndim
    if ndim not in (2, 3):
        msg = f"c2_batch must be 2-D or 3-D, got {ndim}-D"
        raise ValueError(msg)

    valid_methods = ("interpolate", "mask", "mirror", "statistical")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)

    # Single matrix — delegate directly
    if ndim == 2:
        if method == "statistical":
            return _apply_statistical_correction(c2_batch, width)
        return apply_diagonal_correction(c2_batch, width, method)

    # Batch dimension present
    backend = _resolve_backend(c2_batch)

    if backend == "jax":
        import jax

        if method == "statistical":

            def _correct_one(c2: jnp.ndarray) -> jnp.ndarray:
                return _apply_statistical_correction(c2, width)

            return jax.vmap(_correct_one)(c2_batch)

        def _correct_one_standard(c2: jnp.ndarray) -> jnp.ndarray:
            return apply_diagonal_correction(c2, width, method)

        return jax.vmap(_correct_one_standard)(c2_batch)

    # NumPy path: loop over batch dimension
    import numpy as np

    results = np.empty_like(c2_batch)
    for k in range(c2_batch.shape[0]):
        if method == "statistical":
            results[k] = _apply_statistical_correction(c2_batch[k], width)
        else:
            results[k] = apply_diagonal_correction(c2_batch[k], width, method)
    return results


def get_diagonal_correction_methods() -> list[str]:
    """Return the list of supported diagonal correction methods."""
    return ["interpolate", "mask", "mirror", "statistical"]


def get_available_backends() -> list[str]:
    """Return the list of available array backends.

    Always includes ``"numpy"``. Includes ``"jax"`` if JAX is importable.
    """
    backends = ["numpy"]
    try:
        import jax  # noqa: F401, F811

        backends.append("jax")
    except ImportError:
        pass
    return backends
