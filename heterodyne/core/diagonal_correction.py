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

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

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
    idx = jnp.arange(n_times, dtype=jnp.int32)
    return jnp.abs(idx[:, None] - idx[None, :]) < width


def apply_diagonal_correction(
    c2: Any,
    width: int = 1,
    method: str = "basic",
    backend: str | None = None,
    **config: Any,
) -> Any:
    """Correct diagonal artifacts in a two-time correlation matrix.

    The diagonal of c2 (and optionally near-diagonal elements within
    ``width``) is replaced according to the chosen method.

    Args:
        c2: Two-time correlation matrix, shape (N, N). Must be square.
        width: Half-width of the diagonal band to correct. ``width=1``
            corrects only the main diagonal.
        method: Correction strategy. One of:

            - ``"basic"``: Homodyne-compatible adjacent side-band averaging.
              ``"interpolate"`` is accepted as a legacy heterodyne alias.
            - ``"interpolation"``: Homodyne-compatible linear interpolation
              from neighboring off-diagonal values.
            - ``"mask"``: Set diagonal elements to NaN for exclusion in
              downstream fitting.
            - ``"mirror"``: Replace using matrix symmetry, c2[i,j] from
              c2[j,i]. For the exact diagonal (i=j) this is a no-op;
              useful when ``width > 1`` to fill near-diagonal from the
              transposed side.
            - ``"statistical"``: Replace diagonal-band elements using a
              row-wise off-diagonal statistic.

    Returns:
        Corrected correlation matrix, shape (N, N).

    Raises:
        ValueError: If ``method`` is not one of the supported strategies,
            or if ``width < 1``.
    """
    valid_methods = (
        "basic",
        "interpolate",
        "interpolation",
        "mask",
        "mirror",
        "statistical",
    )
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)
    if width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)

    resolved_backend = _resolve_backend(c2) if backend in (None, "auto") else backend
    if resolved_backend == "numpy":
        return _apply_standard_correction_numpy(np.asarray(c2), width, method, config)

    if method in ("basic", "interpolate"):
        return _apply_interpolation(c2, width)
    if method in ("interpolation", "statistical"):
        logger.warning(
            "JAX backend only supports 'basic' diagonal correction, got %r. "
            "Using 'basic' method.",
            method,
        )
        return _apply_interpolation(c2, width)
    if method == "mask":
        return _apply_nan_mask(c2, width)
    # method == "mirror"
    return _apply_mirror(c2, width)


def _apply_interpolation(c2: jnp.ndarray, width: int) -> jnp.ndarray:
    """Replace diagonal band with interpolated values from neighbors.

    For width=1, the first upper and lower off-diagonal bands are averaged
    into a side band, and each diagonal element is replaced with the adjacent
    side-band average. Boundary elements use the single available side-band
    value.

    For width>1, each masked element (i,j) with |i-j| < width is replaced
    with the average of the two nearest elements along the perpendicular
    direction to the diagonal at distance ``width`` from the diagonal.
    """
    n = c2.shape[0]

    if width == 1:
        if n <= 1:
            return c2

        # Homodyne-compatible fast path: average the two first off-diagonal
        # side bands, then average adjacent side-band values onto the diagonal.
        idx_upper = jnp.arange(n - 1)
        idx_lower = jnp.arange(1, n)
        side_band = 0.5 * (c2[idx_upper, idx_lower] + c2[idx_lower, idx_upper])
        diag_val = jnp.zeros(n, dtype=c2.dtype)
        diag_val = diag_val.at[:-1].add(side_band)
        diag_val = diag_val.at[1:].add(side_band)
        norm = jnp.ones(n, dtype=c2.dtype)
        norm = norm.at[1:-1].set(2)
        return c2.at[jnp.diag_indices(n)].set(diag_val / norm)

    # General case: width > 1.
    # Use broadcasting instead of meshgrid to avoid allocating two N×N
    # index arrays — saves ~4 MB at N=1000 and avoids XLA materializing
    # unnecessary intermediates.
    idx = jnp.arange(n, dtype=jnp.int32)
    diff = idx[:, None] - idx[None, :]  # (N, N) via broadcasting
    abs_diff = jnp.abs(diff)
    mask = abs_diff < width

    # For each element, find the two nearest off-band neighbors along
    # the row axis (perpendicular to diagonal).
    # Shift needed to reach the band edge from element (i, j):
    shift = width - abs_diff
    i_above = jnp.clip(idx[:, None] - shift, 0, n - 1)
    i_below = jnp.clip(idx[:, None] + shift, 0, n - 1)

    interpolated = (c2[i_above, idx[None, :]] + c2[i_below, idx[None, :]]) / 2.0
    return jnp.where(mask, interpolated, c2)


def _apply_standard_correction_numpy(
    c2: np.ndarray, width: int, method: str, config: dict[str, Any] | None = None
) -> np.ndarray:
    """NumPy implementation for non-statistical methods."""
    config = config or {}
    if method in ("basic", "interpolate"):
        return _apply_basic_correction_numpy(c2, width)
    if method == "interpolation":
        return _apply_linear_interpolation_correction_numpy(c2, config)
    if method == "statistical":
        return _apply_homodyne_statistical_correction_numpy(c2, config)

    idx_i, idx_j = np.meshgrid(
        np.arange(c2.shape[0]), np.arange(c2.shape[0]), indexing="ij"
    )
    mask = np.abs(idx_i - idx_j) < width
    if method == "mask":
        result = c2.copy().astype(np.float64)
        result[mask] = np.nan
        return result

    return np.where(mask, c2.T, c2)


def _apply_basic_correction_numpy(c2: np.ndarray, width: int) -> np.ndarray:
    """Homodyne basic diagonal correction for NumPy arrays."""
    result = c2.copy()
    n = result.shape[0]
    if n <= 1:
        return result

    idx_upper = np.arange(n - 1)
    idx_lower = np.arange(1, n)
    side_band = 0.5 * (result[idx_upper, idx_lower] + result[idx_lower, idx_upper])
    diag_val = np.zeros(n, dtype=result.dtype)
    diag_val[:-1] += side_band
    diag_val[1:] += side_band
    norm = np.ones(n, dtype=result.dtype)
    norm[1:-1] = 2.0
    np.fill_diagonal(result, diag_val / norm)

    if width > 1:
        idx_i, idx_j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = (np.abs(idx_i - idx_j) < width) & (idx_i != idx_j)
        diff = idx_i - idx_j
        shift = width - np.abs(diff)
        i_above = np.clip(idx_i - shift, 0, n - 1)
        i_below = np.clip(idx_i + shift, 0, n - 1)
        interpolated = (result[i_above, idx_j] + result[i_below, idx_j]) / 2.0
        result = np.where(mask, interpolated, result)

    return result


def _apply_homodyne_statistical_correction_numpy(
    c2: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Homodyne statistical diagonal correction for NumPy arrays."""
    result = c2.copy()
    n = c2.shape[0]
    window_size = int(config.get("window_size", 3))
    estimator = str(config.get("estimator", "median"))
    trim_fraction = float(config.get("trim_fraction", 0.2))

    for i in range(n):
        neighbors: list[float] = []
        for offset in range(1, min(window_size + 1, n)):
            if i - offset >= 0:
                neighbors.append(float(c2[i - offset, i]))
                neighbors.append(float(c2[i, i - offset]))
            if i + offset < n:
                neighbors.append(float(c2[i + offset, i]))
                neighbors.append(float(c2[i, i + offset]))

        if not neighbors:
            continue

        neighbors_arr = np.array(neighbors, dtype=np.float64)
        if estimator == "median":
            result[i, i] = np.nanmedian(neighbors_arr)
        elif estimator == "mean":
            result[i, i] = np.nanmean(neighbors_arr)
        elif estimator == "trimmed_mean":
            finite_neighbors = neighbors_arr[np.isfinite(neighbors_arr)]
            if finite_neighbors.size == 0:
                result[i, i] = np.nan
            else:
                try:
                    from scipy import stats

                    result[i, i] = stats.trim_mean(finite_neighbors, trim_fraction)
                except ImportError:
                    result[i, i] = np.nanmedian(neighbors_arr)
        else:
            logger.warning("Unknown estimator %r, using median", estimator)
            result[i, i] = np.nanmedian(neighbors_arr)

    return result


def _apply_linear_interpolation_correction_numpy(
    c2: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Homodyne interpolation diagonal correction for NumPy arrays."""
    result = c2.copy()
    n = c2.shape[0]
    interp_method = str(config.get("interpolation_method", "linear"))

    for i in range(n):
        if 0 < i < n - 1:
            y_points = [c2[i - 1, i], c2[i + 1, i]]
            if interp_method == "cubic":
                raise NotImplementedError(
                    "Cubic diagonal correction is not yet implemented. "
                    "Use method='linear'."
                )
            result[i, i] = np.nanmean(y_points)
        elif i == 0 and n > 1:
            result[i, i] = c2[0, 1]
        elif i == n - 1 and n > 1:
            result[i, i] = c2[n - 2, n - 1]

    return result


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
    std_ratio = std_diag / jnp.where(std_off > 1e-30, std_off, 1e-30)
    excess_sigma = mean_excess / jnp.where(std_off > 1e-30, std_off, 1e-30)

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
    idx_i = jnp.arange(n_rows, dtype=jnp.int32)
    idx_j = jnp.arange(n_cols, dtype=jnp.int32)
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
        from jaxlib.xla_extension import ArrayImpl  # type: ignore[import-not-found]

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
    logger.debug(
        "Detected backend: %s for array type %s", backend, type(array).__name__
    )
    return backend


def _apply_statistical_correction_numpy(
    c2: Any,
    width: int,
    estimator: str = "mean",
) -> Any:
    """Vectorized NumPy path for statistical diagonal correction.

    Replaces the diagonal band (|i - j| < *width*) with per-row statistics
    computed from a window of off-diagonal elements at distances
    [*width*, *width* + 3) from the diagonal.

    Args:
        c2: Two-time correlation matrix, shape (N, N). Must be a NumPy array.
        width: Half-width of the diagonal band to correct.
        estimator: One of ``"mean"``, ``"median"``, or ``"trimmed_mean"``.

    Returns:
        Corrected NumPy matrix of the same shape as *c2*.
    """
    import numpy as np

    n = c2.shape[0]
    col_idx = np.arange(n, dtype=np.int32)
    # dist[i, j] = |i - j| — distance of column j from row i's diagonal
    dist = np.abs(col_idx[None, :] - col_idx[:, None])  # (N, N)

    band_mask = dist < width
    window_lo = width
    window_hi = width + 3

    # Primary window: [width, width+3)
    window_mask = (dist >= window_lo) & (dist < window_hi)  # (N, N)
    # Fallback window (for boundary rows): everything at distance >= width
    fallback_mask = dist >= window_lo  # (N, N)

    # Determine which rows have no primary window elements
    has_primary = np.any(window_mask, axis=1)  # (N,)
    # Use fallback where primary is empty
    effective_mask = np.where(
        has_primary[:, None], window_mask, fallback_mask
    )  # (N, N)

    # Determine which rows have *any* window elements
    has_any = np.any(effective_mask, axis=1)  # (N,)

    # Replace non-window elements with NaN for masked reduction
    c2_masked = np.where(effective_mask, c2, np.nan)  # (N, N)

    if estimator == "mean":
        with np.errstate(all="ignore"):
            row_stats = np.nanmean(c2_masked, axis=1)  # (N,)
    elif estimator == "median":
        with np.errstate(all="ignore"):
            row_stats = np.nanmedian(c2_masked, axis=1)  # (N,)
    else:
        # trimmed_mean: sort each row, trim top/bottom 10% of window elements
        # Count valid elements per row
        counts = np.sum(effective_mask, axis=1)  # (N,)
        k = np.maximum(1, np.round(0.1 * counts).astype(int))  # (N,)

        # Sort along axis=1 — NaN sorts to end
        sorted_vals = np.sort(c2_masked, axis=1)  # (N, N)

        # Build a column index array and mask out the trimmed tails
        col_range = np.arange(n)[None, :]  # (1, N)
        trim_mask = (col_range >= k[:, None]) & (col_range < (counts - k)[:, None])
        # If trimming leaves nothing, fall back to the full window
        has_trimmed = np.any(trim_mask, axis=1)
        trim_mask = np.where(
            has_trimmed[:, None],
            trim_mask,
            col_range < counts[:, None],
        )

        trimmed_vals = np.where(trim_mask, sorted_vals, np.nan)
        with np.errstate(all="ignore"):
            row_stats = np.nanmean(trimmed_vals, axis=1)

    # Rows with no window elements at all: keep original diagonal value
    row_stats = np.where(has_any, row_stats, np.diag(c2))

    # Broadcast row statistics and apply
    replacement = np.broadcast_to(row_stats[:, None], (n, n))
    return np.where(band_mask, replacement, c2)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _apply_statistical_correction_jax(
    c2: jnp.ndarray,
    width: int,
    estimator: str = "mean",
) -> jnp.ndarray:
    """Vectorized JAX path for statistical diagonal correction.

    Fully compatible with ``jit`` and ``vmap`` — no Python loops or
    boolean indexing. Uses ``jnp.where``-based masked reductions.

    Args:
        c2: Two-time correlation matrix, shape (N, N). Must be a JAX array.
        width: Half-width of the diagonal band to correct.
        estimator: One of ``"mean"``, ``"median"``, or ``"trimmed_mean"``.

    Returns:
        Corrected JAX matrix of the same shape as *c2*.
    """
    n = c2.shape[0]
    col_idx = jnp.arange(n, dtype=jnp.int32)
    dist = jnp.abs(col_idx[None, :] - col_idx[:, None])  # (N, N)

    band_mask = dist < width
    window_lo = width
    window_hi = width + 3

    # Primary window mask and fallback (everything at distance >= width)
    window_mask = (dist >= window_lo) & (dist < window_hi)  # (N, N)
    fallback_mask = dist >= window_lo  # (N, N)

    # Use fallback where primary window is empty for a row
    has_primary = jnp.any(window_mask, axis=1)  # (N,)
    effective_mask = jnp.where(has_primary[:, None], window_mask, fallback_mask)

    # Whether each row has any window elements at all
    has_any = jnp.any(effective_mask, axis=1)  # (N,)

    # Masked c2: set non-window positions to NaN
    c2_masked = jnp.where(effective_mask, c2, jnp.nan)  # (N, N)

    if estimator == "mean":
        row_stats = jnp.nanmean(c2_masked, axis=1)  # (N,)
    elif estimator == "median":
        row_stats = jnp.nanmedian(c2_masked, axis=1)  # (N,)
    else:
        # trimmed_mean: sort each row, trim top/bottom 10% of valid elements
        counts = jnp.sum(effective_mask, axis=1)  # (N,)
        k = jnp.maximum(1, jnp.round(0.1 * counts).astype(jnp.int32))  # (N,)

        # Sort along axis=1 — NaN sorts to end
        sorted_vals = jnp.sort(c2_masked, axis=1)  # (N, N)

        # Build column index and mask out trimmed tails
        col_range = jnp.arange(n)[None, :]  # (1, N)
        trim_mask = (col_range >= k[:, None]) & (col_range < (counts - k)[:, None])
        # If trimming leaves nothing, fall back to full window
        has_trimmed = jnp.any(trim_mask, axis=1)
        trim_mask = jnp.where(
            has_trimmed[:, None],
            trim_mask,
            col_range < counts[:, None],
        )

        trimmed_vals = jnp.where(trim_mask, sorted_vals, jnp.nan)
        row_stats = jnp.nanmean(trimmed_vals, axis=1)

    # Rows with no window elements: keep original diagonal value
    row_stats = jnp.where(has_any, row_stats, jnp.diag(c2))

    # Broadcast row statistics and apply
    replacement = jnp.broadcast_to(row_stats[:, None], (n, n))
    return jnp.where(band_mask, replacement, c2)


def _apply_statistical_correction(
    c2: Any,
    width: int,
    estimator: str = "mean",
    *,
    backend: str | None = None,
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
        backend: Pre-resolved backend string (``"jax"`` or ``"numpy"``).
            If ``None``, detected from *c2*.

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

    if backend is None:
        backend = _resolve_backend(c2)

    if backend == "jax":
        return _apply_statistical_correction_jax(c2, width, estimator)
    return _apply_statistical_correction_numpy(c2, width, estimator)


@functools.lru_cache(maxsize=8)
def _get_batch_statistical_fn(width: int) -> Any:
    """Return a cached jit(vmap) function for batch statistical correction."""

    def _correct_one(c2: jnp.ndarray) -> jnp.ndarray:
        return _apply_statistical_correction_jax(c2, width)  # type: ignore[no-any-return]

    return jax.jit(jax.vmap(_correct_one))


@functools.lru_cache(maxsize=8)
def _get_batch_standard_fn(width: int, method: str) -> Any:
    """Return a cached jit(vmap) function for batch standard correction."""

    def _correct_one(c2: jnp.ndarray) -> jnp.ndarray:
        return apply_diagonal_correction(c2, width, method)

    return jax.jit(jax.vmap(_correct_one))


def apply_diagonal_correction_batch(
    c2_batch: Any,
    width: int = 1,
    method: str = "basic",
    backend: str | None = None,
    **config: Any,
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

    valid_methods = (
        "basic",
        "interpolate",
        "interpolation",
        "mask",
        "mirror",
        "statistical",
    )
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got {method!r}"
        raise ValueError(msg)

    # Resolve backend once for the entire batch
    resolved_backend = (
        _resolve_backend(c2_batch) if backend in (None, "auto") else backend
    )

    # Single matrix — delegate directly
    if ndim == 2:
        return apply_diagonal_correction(
            c2_batch,
            width=width,
            method=method,
            backend=resolved_backend,
            **config,
        )

    # Batch dimension present
    if resolved_backend == "jax":
        if method not in ("basic", "interpolate"):
            logger.warning(
                "JAX backend only supports 'basic' diagonal correction, got %r. "
                "Using 'basic' method.",
                method,
            )
            method = "basic"
        return _get_batch_standard_fn(width, method)(c2_batch)

    # NumPy path: loop over batch dimension
    import numpy as np

    results = np.empty_like(c2_batch)
    for k in range(c2_batch.shape[0]):
        results[k] = apply_diagonal_correction(
            c2_batch[k],
            width=width,
            method=method,
            backend="numpy",
            **config,
        )
    return results


def get_diagonal_correction_methods() -> list[str]:
    """Return the list of supported diagonal correction methods."""
    return ["basic", "statistical", "interpolation"]


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
