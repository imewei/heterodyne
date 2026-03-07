"""Datashader-based visualization backend for large XPCS datasets.

Provides fast rasterized rendering of two-time correlation matrices,
residual heatmaps, and multi-angle grids.  Falls back to matplotlib
when datashader is not installed.

Full datashader functionality requires::

    pip install datashader

Without it, all functions still work via the matplotlib fallback path,
though rendering of very large matrices will be slower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = get_logger(__name__)

# Optional datashader import --------------------------------------------------
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    HAS_DATASHADER = True
except ImportError:  # pragma: no cover
    HAS_DATASHADER = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_correlation_heatmap(
    c2: np.ndarray,
    times: np.ndarray,
    width: int = 800,
    height: int = 800,
) -> PILImage.Image | np.ndarray:
    """Render a two-time correlation heatmap.

    Uses datashader for fast rasterization of large matrices.  If
    datashader is not installed, falls back to matplotlib.

    Requires ``pip install datashader`` for full functionality.

    Args:
        c2: Two-time correlation matrix, shape ``(n_t, n_t)``.
        times: 1-D time array of length *n_t*.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        PIL Image (datashader path) or numpy RGB array (matplotlib path).
    """
    if HAS_DATASHADER:
        return _ds_heatmap(c2, times, width, height, cmap="viridis", symmetric=False)
    return _mpl_heatmap(c2, times, width, height, cmap="viridis", symmetric=False)


def render_residual_heatmap(
    residuals: np.ndarray,
    times: np.ndarray,
    width: int = 800,
    height: int = 800,
) -> PILImage.Image | np.ndarray:
    """Render a residual heatmap with a symmetric colormap centered at 0.

    Uses datashader for fast rasterization of large matrices.  If
    datashader is not installed, falls back to matplotlib.

    Requires ``pip install datashader`` for full functionality.

    Args:
        residuals: Residual matrix, shape ``(n_t, n_t)``.
        times: 1-D time array of length *n_t*.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        PIL Image (datashader path) or numpy RGB array (matplotlib path).
    """
    if HAS_DATASHADER:
        return _ds_heatmap(
            residuals, times, width, height, cmap="RdBu_r", symmetric=True
        )
    return _mpl_heatmap(
        residuals, times, width, height, cmap="RdBu_r", symmetric=True
    )


def render_multi_angle_grid(
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    times: np.ndarray,
    ncols: int = 3,
) -> PILImage.Image | np.ndarray:
    """Render a grid of correlation heatmaps, one per azimuthal angle.

    Uses datashader for fast rasterization of large matrices.  If
    datashader is not installed, falls back to matplotlib.

    Requires ``pip install datashader`` for full functionality.

    Args:
        c2_data: 3-D array of shape ``(n_phi, n_t, n_t)``.
        phi_angles: 1-D array of phi angles in degrees, length *n_phi*.
        times: 1-D time array of length *n_t*.
        ncols: Number of columns in the subplot grid.

    Returns:
        PIL Image (datashader path) or numpy RGB array (matplotlib path).

    Raises:
        ValueError: If *c2_data* is not 3-D or dimension mismatch.
    """
    if c2_data.ndim != 3:
        msg = f"c2_data must be 3-D (n_phi, n_t, n_t), got ndim={c2_data.ndim}"
        raise ValueError(msg)

    n_phi = c2_data.shape[0]
    if phi_angles.shape != (n_phi,):
        msg = (
            f"phi_angles length ({phi_angles.shape[0]}) must match "
            f"first axis of c2_data ({n_phi})"
        )
        raise ValueError(msg)

    nrows = max(1, int(np.ceil(n_phi / ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    vmin = float(np.nanmin(c2_data))
    vmax = float(np.nanmax(c2_data))

    for idx, (ax, phi) in enumerate(
        zip(axes.flat, phi_angles, strict=False)
    ):
        if idx >= n_phi:
            break
        im = ax.imshow(
            c2_data[idx],
            extent=[times[0], times[-1], times[-1], times[0]],
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"phi = {phi:.1f} deg")
        ax.set_xlabel("t2")
        ax.set_ylabel("t1")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax in axes.flat[n_phi:]:
        ax.set_visible(False)

    fig.tight_layout()

    # Convert figure to numpy RGB array
    fig.canvas.draw()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
    rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    return rgb.copy()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ds_heatmap(
    matrix: np.ndarray,
    times: np.ndarray,
    width: int,
    height: int,
    cmap: str,
    symmetric: bool,
) -> PILImage.Image:
    """Render via datashader (requires datashader + pandas)."""
    t1_grid, t2_grid = np.meshgrid(times, times, indexing="ij")
    df = pd.DataFrame(
        {
            "t1": t1_grid.ravel(),
            "t2": t2_grid.ravel(),
            "value": matrix.ravel(),
        }
    )

    # Drop NaN rows so datashader doesn't choke
    df = df.dropna(subset=["value"])

    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(float(times[0]), float(times[-1])),
        y_range=(float(times[0]), float(times[-1])),
    )
    agg = canvas.points(df, "t2", "t1", agg=ds.mean("value"))

    if symmetric:
        vmax = float(np.nanmax(np.abs(matrix)))
        span = (-vmax, vmax)
    else:
        span = None  # auto

    img = tf.shade(agg, cmap=cmap, span=span)
    return img.to_pil()


def _mpl_heatmap(
    matrix: np.ndarray,
    times: np.ndarray,
    width: int,
    height: int,
    cmap: str,
    symmetric: bool,
) -> np.ndarray:
    """Fallback rendering via matplotlib."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    kwargs: dict[str, object] = {
        "extent": [times[0], times[-1], times[-1], times[0]],
        "aspect": "auto",
        "cmap": cmap,
    }
    if symmetric:
        vmax = float(np.nanmax(np.abs(matrix)))
        kwargs["vmin"] = -vmax
        kwargs["vmax"] = vmax

    im = ax.imshow(matrix, **kwargs)  # type: ignore[arg-type]
    ax.set_xlabel("t2")
    ax.set_ylabel("t1")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    fig.canvas.draw()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
    rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    return rgb.copy()
