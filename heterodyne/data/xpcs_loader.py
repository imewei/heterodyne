"""XPCS data loading from HDF5, NPZ, and MAT files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from heterodyne.utils.logging import get_logger
from heterodyne.utils.path_validation import validate_file_exists

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# NPZ cache suffix appended to original filename
_CACHE_SUFFIX = ".heterodyne_cache.npz"

# Keys used inside NPZ cache files
_CACHE_KEY_MTIME = "source_mtime"
_CACHE_KEY_C2 = "c2"
_CACHE_KEY_T = "t"
_CACHE_KEY_Q = "q"
_CACHE_KEY_Q_VALUES = "q_values"
_CACHE_KEY_PHI = "phi"


@dataclass
class XPCSData:
    """Container for loaded XPCS data."""

    # Two-time correlation matrix c2(t1, t2)
    c2: np.ndarray

    # Time arrays
    t1: np.ndarray
    t2: np.ndarray

    # Optional metadata
    q: float | None = None
    phi_angles: np.ndarray | None = None
    uncertainties: np.ndarray | None = None

    # Multi-q support: q values for each q-bin when c2 has shape (n_q, n_t, n_t)
    q_values: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of correlation data."""
        return self.c2.shape

    @property
    def n_times(self) -> int:
        """Number of time points."""
        if self.c2.ndim == 3:
            return self.c2.shape[1]
        return self.c2.shape[0]

    @property
    def has_multi_phi(self) -> bool:
        """Whether data has multiple phi angles."""
        return self.c2.ndim == 3

    @property
    def has_multi_q(self) -> bool:
        """Whether data contains multiple q-bins (q_values is set)."""
        return self.q_values is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class DataValidationError(ValueError):
    """Raised when loaded XPCS data fails validation checks."""


def validate_loaded_data(data: XPCSData) -> list[str]:
    """Validate an XPCSData container and return a list of warning strings.

    Performs the following checks:

    1. **NaN / Inf** - c2, t1, t2 must be finite.
    2. **Shape consistency** - t1 and t2 lengths must match the time
       dimensions of c2.  If q_values is set, its length must match
       ``c2.shape[0]``.
    3. **Symmetry** - For 2-D c2 the matrix should be approximately symmetric
       (max |c2 - c2.T| / max(|c2|) < 1e-6).  A warning is issued but no
       exception is raised.
    4. **Positive diagonal** - All diagonal elements of c2 (or each slice for
       3-D) must be positive.
    5. **Time monotonicity** - t1 and t2 must be strictly increasing.

    Args:
        data: Loaded XPCSData to validate.

    Returns:
        List of human-readable warning strings.  An empty list means all
        checks passed.

    Raises:
        DataValidationError: If any hard constraint is violated (NaN/Inf,
            shape mismatch, non-positive diagonal, non-monotonic time).
    """
    warnings: list[str] = []

    # 1. Finiteness
    for name, arr in (("c2", data.c2), ("t1", data.t1), ("t2", data.t2)):
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise DataValidationError(
                f"{name} contains {n_bad} non-finite value(s) (NaN or Inf)"
            )

    # 2. Shape consistency
    n_t1 = data.t1.shape[0]
    n_t2 = data.t2.shape[0]

    if data.c2.ndim == 2:
        expected_rows, expected_cols = data.c2.shape
        if n_t1 != expected_rows:
            raise DataValidationError(
                f"t1 length {n_t1} does not match c2 row count {expected_rows}"
            )
        if n_t2 != expected_cols:
            raise DataValidationError(
                f"t2 length {n_t2} does not match c2 column count {expected_cols}"
            )
    elif data.c2.ndim == 3:
        n_q, n_rows, n_cols = data.c2.shape
        if n_t1 != n_rows:
            raise DataValidationError(
                f"t1 length {n_t1} does not match c2 time-axis size {n_rows}"
            )
        if n_t2 != n_cols:
            raise DataValidationError(
                f"t2 length {n_t2} does not match c2 time-axis size {n_cols}"
            )
        if data.q_values is not None and data.q_values.shape[0] != n_q:
            raise DataValidationError(
                f"q_values length {data.q_values.shape[0]} does not match "
                f"c2 q-axis size {n_q}"
            )
    else:
        raise DataValidationError(
            f"c2 must be 2-D or 3-D, got {data.c2.ndim}-D"
        )

    # 3. Symmetry (soft check, 2-D only)
    if data.c2.ndim == 2:
        max_abs = np.max(np.abs(data.c2))
        if max_abs > 0:
            asymmetry = np.max(np.abs(data.c2 - data.c2.T)) / max_abs
            if asymmetry > 1e-6:
                warnings.append(
                    f"c2 is not symmetric: max relative asymmetry = {asymmetry:.3e}"
                )

    # 4. Positive diagonal
    if data.c2.ndim == 2:
        diag = np.diag(data.c2)
        if np.any(diag <= 0):
            n_bad = int(np.sum(diag <= 0))
            raise DataValidationError(
                f"c2 has {n_bad} non-positive diagonal element(s)"
            )
    else:
        for qi in range(data.c2.shape[0]):
            diag = np.diag(data.c2[qi])
            if np.any(diag <= 0):
                n_bad = int(np.sum(diag <= 0))
                raise DataValidationError(
                    f"c2[{qi}] has {n_bad} non-positive diagonal element(s)"
                )

    # 5. Time monotonicity
    for name, arr in (("t1", data.t1), ("t2", data.t2)):
        if arr.shape[0] > 1 and not np.all(np.diff(arr) > 0):
            raise DataValidationError(
                f"{name} is not strictly increasing"
            )

    if warnings:
        for w in warnings:
            logger.warning("Data validation warning: %s", w)
    else:
        logger.debug("Data validation passed")

    return warnings


# ---------------------------------------------------------------------------
# Half-matrix reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_from_half_matrix(data: np.ndarray) -> np.ndarray:
    """Reconstruct a full symmetric matrix from a triangular half-matrix.

    Many XPCS analysis codes store only the upper or lower triangle of the
    two-time correlation matrix c2(t1, t2) to save disk space.  This function
    detects which triangle is populated (by comparing the L1 norms of the
    strict upper and lower halves) and reconstructs the full symmetric matrix
    via ``M + M.T - diag(M)``.

    The input may be 2-D (N, N) or 3-D (n_q, N, N).  For 3-D arrays the
    reconstruction is applied independently to each q-slice.

    Args:
        data: Array of shape (N, N) or (n_q, N, N) containing a triangular
            half-matrix with zeros (or negligible values) in the unused
            triangle.

    Returns:
        Full symmetric array of the same shape.

    Raises:
        ValueError: If ``data`` is not 2-D or 3-D, or if the last two
            dimensions are not square.
    """
    if data.ndim == 2:
        return _reconstruct_2d(data)
    if data.ndim == 3:
        result = np.empty_like(data)
        for qi in range(data.shape[0]):
            result[qi] = _reconstruct_2d(data[qi])
        return result
    raise ValueError(
        f"_reconstruct_from_half_matrix expects 2-D or 3-D input, "
        f"got {data.ndim}-D"
    )


def _reconstruct_2d(m: np.ndarray) -> np.ndarray:
    """Reconstruct a full symmetric matrix from a 2-D triangular half."""
    n_rows, n_cols = m.shape
    if n_rows != n_cols:
        raise ValueError(
            f"Matrix must be square for half-matrix reconstruction, "
            f"got shape ({n_rows}, {n_cols})"
        )
    # Determine which triangle is populated by comparing their L1 norms.
    # In both cases the formula M + M.T - diag(M) produces the full matrix.
    upper_norm = np.sum(np.abs(np.triu(m, k=1)))
    lower_norm = np.sum(np.abs(np.tril(m, k=-1)))
    if upper_norm >= lower_norm:
        logger.debug("Half-matrix reconstruction: upper triangle -> full matrix")
    else:
        logger.debug("Half-matrix reconstruction: lower triangle -> full matrix")
    return m + m.T - np.diag(np.diag(m))


# ---------------------------------------------------------------------------
# Diagonal correction (NumPy wrapper around the JAX implementation)
# ---------------------------------------------------------------------------


def _apply_diagonal_correction(
    c2: np.ndarray,
    width: int = 1,
    method: str = "interpolate",
) -> np.ndarray:
    """Apply diagonal artifact correction to a two-time correlation matrix.

    Implements the same correction strategies as
    :mod:`heterodyne.core.diagonal_correction` but operates on NumPy arrays
    directly to avoid pulling in the ``heterodyne.core`` package initialiser.
    This correction targets the two-time correlation c2, **not** the one-time
    g2.

    For 3-D input of shape (n_q, N, N) the correction is applied
    independently to each q-slice.

    Args:
        c2: Two-time correlation matrix, shape (N, N) or (n_q, N, N).
        width: Half-width of the diagonal band to correct.  ``width=1``
            corrects only the main diagonal.
        method: One of ``"interpolate"``, ``"mask"``, ``"mirror"``.

    Returns:
        Corrected array of the same shape as ``c2``, as a NumPy array.

    Raises:
        ValueError: If ``method`` is not one of the supported strategies,
            ``width < 1``, or ``c2`` is not 2-D or 3-D.
    """
    valid_methods = ("interpolate", "mask", "mirror")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    if c2.ndim == 2:
        return _diag_correct_2d(c2, width, method)
    if c2.ndim == 3:
        slices = [_diag_correct_2d(c2[qi], width, method) for qi in range(c2.shape[0])]
        return np.stack(slices, axis=0)
    raise ValueError(
        f"_apply_diagonal_correction expects 2-D or 3-D input, got {c2.ndim}-D"
    )


def _diag_correct_2d(m: np.ndarray, width: int, method: str) -> np.ndarray:
    """Apply diagonal correction to a single (N, N) matrix."""
    n = m.shape[0]
    idx_i, idx_j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = np.abs(idx_i - idx_j) < width

    if method == "interpolate":
        if width == 1:
            # Fast path: only the main diagonal
            i_idx = np.arange(n)
            i_prev = np.maximum(i_idx - 1, 0)
            i_next = np.minimum(i_idx + 1, n - 1)
            neighbor_avg = (
                m[i_prev, i_idx]
                + m[i_next, i_idx]
                + m[i_idx, i_prev]
                + m[i_idx, i_next]
            ) / 4.0
            result = m.copy()
            result[i_idx, i_idx] = neighbor_avg
            return result
        # General case
        diff = idx_i - idx_j
        shift = width - np.abs(diff)
        i_above = np.clip(idx_i - shift, 0, n - 1)
        i_below = np.clip(idx_i + shift, 0, n - 1)
        interpolated = (m[i_above, idx_j] + m[i_below, idx_j]) / 2.0
        return np.where(mask, interpolated, m)

    if method == "mask":
        result = m.copy().astype(np.float64)
        result[mask] = np.nan
        return result

    # method == "mirror": c2[i,j] = c2[j,i] for band elements
    return np.where(mask, m.T, m)


# ---------------------------------------------------------------------------
# HDF5 structure probing
# ---------------------------------------------------------------------------


def probe_hdf5_structure(file_path: Path | str) -> dict[str, Any]:
    """Inspect and report the structure of an HDF5 file.

    Recursively walks the HDF5 tree and collects dataset names, shapes,
    dtypes, and top-level attributes.  Useful for discovering key names
    before loading.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Dictionary with the following keys:

        - ``"datasets"``: list of dicts, each with ``"path"``, ``"shape"``,
          ``"dtype"`` for every dataset in the file.
        - ``"groups"``: list of str paths for all groups.
        - ``"root_attrs"``: dict of attributes on the root ``/`` group.
        - ``"n_datasets"``: total dataset count.
        - ``"n_groups"``: total group count.

    Raises:
        heterodyne.utils.path_validation.PathValidationError: If the file
            does not exist or is not readable.
    """
    resolved = validate_file_exists(file_path, "HDF5 file")

    datasets: list[dict[str, Any]] = []
    groups: list[str] = []

    def _visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(
                {
                    "path": f"/{name}",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )
        elif isinstance(obj, h5py.Group):
            groups.append(f"/{name}")

    with h5py.File(resolved, "r") as f:
        f.visititems(_visitor)
        root_attrs: dict[str, Any] = {}
        for key in f.attrs:
            value = f.attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            root_attrs[key] = value

    logger.info(
        "HDF5 structure: %d datasets, %d groups in %s",
        len(datasets),
        len(groups),
        resolved.name,
    )

    return {
        "datasets": datasets,
        "groups": groups,
        "root_attrs": root_attrs,
        "n_datasets": len(datasets),
        "n_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------


def load_xpcs_batch(
    file_paths: list[Path | str],
    c2_key: str = "c2",
    time_key: str = "t",
    format: str | None = None,
    use_cache: bool = False,
    validate: bool = False,
    apply_diag_correction: bool = False,
    diag_correction_width: int = 1,
    diag_correction_method: str = "interpolate",
    frame_range: tuple[int, int] | None = None,
    select_q: float | None = None,
    q_tolerance: float | None = None,
) -> list[XPCSData]:
    """Load multiple XPCS data files and return them as a list.

    Each file is loaded independently using :class:`XPCSDataLoader`.  Files
    that fail to load are logged as errors and skipped; the remaining
    successfully loaded datasets are returned in input order.

    Args:
        file_paths: Sequence of paths to data files.
        c2_key: Key for correlation data in each file.
        time_key: Key for time array in each file.
        format: File format (auto-detected per file if None).
        use_cache: If True, enable NPZ caching for each file.
        validate: If True, run :func:`validate_loaded_data` on each result.
        apply_diag_correction: If True, apply diagonal correction to each c2.
        diag_correction_width: Band width passed to diagonal correction.
        diag_correction_method: Method passed to diagonal correction.
        frame_range: Optional ``(start, end)`` with 1-based inclusive frame
            indices applied uniformly to every file.
        select_q: Target wavevector in Å⁻¹ applied uniformly to every file.
        q_tolerance: Maximum absolute deviation from ``select_q`` in Å⁻¹.

    Returns:
        List of :class:`XPCSData` objects, one per successfully loaded file.
        Failed files are omitted.
    """
    results: list[XPCSData] = []
    n_total = len(file_paths)

    for idx, fp in enumerate(file_paths):
        try:
            loader = XPCSDataLoader(fp, format=format)
            data = loader.load(
                c2_key=c2_key,
                time_key=time_key,
                use_cache=use_cache,
                frame_range=frame_range,
                select_q=select_q,
                q_tolerance=q_tolerance,
            )
            if apply_diag_correction:
                data.c2 = _apply_diagonal_correction(
                    data.c2,
                    width=diag_correction_width,
                    method=diag_correction_method,
                )
            if validate:
                validate_loaded_data(data)
            results.append(data)
            logger.debug(
                "Batch load [%d/%d]: OK %s",
                idx + 1,
                n_total,
                Path(fp).name,
            )
        except Exception as exc:
            logger.error(
                "Batch load [%d/%d]: FAILED %s - %s",
                idx + 1,
                n_total,
                Path(fp).name,
                exc,
            )

    logger.info(
        "Batch load complete: %d/%d files loaded successfully",
        len(results),
        n_total,
    )
    return results


# ---------------------------------------------------------------------------
# NPZ cache helpers
# ---------------------------------------------------------------------------


def _cache_path_for(source: Path) -> Path:
    """Return the NPZ cache path collocated with *source*."""
    return source.with_name(source.name + _CACHE_SUFFIX)


def _source_mtime(source: Path) -> float:
    """Return the modification time of *source* as a float."""
    return source.stat().st_mtime


def _cache_is_valid(source: Path, cache: Path) -> bool:
    """Return True if *cache* exists and was built from the current *source*."""
    if not cache.exists():
        return False
    try:
        stored = np.load(cache, allow_pickle=False)
        cached_mtime = float(stored[_CACHE_KEY_MTIME])
        return cached_mtime == _source_mtime(source)
    except Exception:
        return False


def _write_cache(cache_path: Path, data: XPCSData, source_mtime: float) -> None:
    """Persist *data* to an NPZ cache file (allow_pickle=False on read-back)."""
    arrays: dict[str, np.ndarray] = {
        _CACHE_KEY_MTIME: np.array(source_mtime, dtype=np.float64),
        _CACHE_KEY_C2: data.c2,
        _CACHE_KEY_T: data.t1,
    }
    if data.q is not None:
        arrays[_CACHE_KEY_Q] = np.array(data.q, dtype=np.float64)
    if data.q_values is not None:
        arrays[_CACHE_KEY_Q_VALUES] = data.q_values
    if data.phi_angles is not None:
        arrays[_CACHE_KEY_PHI] = data.phi_angles

    np.savez(cache_path, **arrays)
    logger.debug("Cache written: %s", cache_path)


def _read_cache(cache_path: Path) -> XPCSData:
    """Load XPCSData from an NPZ cache file.

    Uses allow_pickle=False to prevent code execution from untrusted files.
    """
    stored = np.load(cache_path, allow_pickle=False)
    c2 = np.asarray(stored[_CACHE_KEY_C2], dtype=np.float64)
    t = np.asarray(stored[_CACHE_KEY_T], dtype=np.float64)
    q = float(stored[_CACHE_KEY_Q]) if _CACHE_KEY_Q in stored else None
    q_values = (
        np.asarray(stored[_CACHE_KEY_Q_VALUES], dtype=np.float64)
        if _CACHE_KEY_Q_VALUES in stored
        else None
    )
    phi = (
        np.asarray(stored[_CACHE_KEY_PHI], dtype=np.float64)
        if _CACHE_KEY_PHI in stored
        else None
    )
    logger.debug("Cache hit: %s", cache_path)
    return XPCSData(c2=c2, t1=t, t2=t, q=q, q_values=q_values, phi_angles=phi)


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class XPCSDataLoader:
    """Loader for XPCS correlation data from various file formats."""

    def __init__(
        self,
        file_path: Path | str,
        format: str | None = None,
    ) -> None:
        """Initialize loader.

        Args:
            file_path: Path to data file.
            format: File format ('hdf5', 'npz', 'mat', 'npy'), or None to
                auto-detect from extension.
        """
        self.file_path = validate_file_exists(file_path, "XPCS data file")
        self.format = format or self._detect_format()

    def _detect_format(self) -> str:
        """Detect file format from extension."""
        suffix = self.file_path.suffix.lower()
        format_map = {
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".hdf": "hdf5",
            ".npz": "npz",
            ".npy": "npy",
            ".mat": "mat",
        }
        if suffix not in format_map:
            raise ValueError(f"Unknown file format: {suffix}")
        return format_map[suffix]

    def load(
        self,
        c2_key: str = "c2",
        time_key: str = "t",
        q_key: str | None = "q",
        phi_key: str | None = "phi",
        use_cache: bool = False,
        frame_range: tuple[int, int] | None = None,
        select_q: float | None = None,
        q_tolerance: float | None = None,
    ) -> XPCSData:
        """Load XPCS data from file.

        Args:
            c2_key: Key/path for correlation data.
            time_key: Key/path for time array.
            q_key: Optional key for scalar wavevector.
            phi_key: Optional key for phi angles.
            use_cache: If True and the format supports caching (hdf5, mat),
                attempt to load from a collocated NPZ cache first.  The cache
                is invalidated automatically when the source file's mtime
                changes.
            frame_range: Optional ``(start, end)`` with **1-based** indexing
                (matching homodyne convention).  If provided, only frames
                ``start`` through ``end`` (inclusive) are retained after
                loading.  Validation is performed before slicing.
            select_q: Target wavevector in Å⁻¹.  When set and the loaded data
                has ``q_values`` (multi-q 3-D), the q-bin(s) closest to this
                value are selected.  If ``q_tolerance`` is also given, all
                bins within that tolerance are kept; otherwise only the single
                nearest bin is kept.
            q_tolerance: Maximum absolute deviation from ``select_q`` in Å⁻¹.
                Ignored when ``select_q`` is ``None``.

        Returns:
            XPCSData container.
        """
        logger.info("Loading XPCS data from %s", self.file_path)

        if use_cache and self.format in ("hdf5", "mat"):
            data = self._load_with_cache(c2_key, time_key, q_key, phi_key)
        elif self.format == "hdf5":
            data = self._load_hdf5(c2_key, time_key, q_key, phi_key)
        elif self.format == "npz":
            data = self._load_npz(c2_key, time_key, q_key, phi_key)
        elif self.format == "npy":
            data = self._load_npy()
        elif self.format == "mat":
            data = self._load_mat(c2_key, time_key, q_key, phi_key)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        if frame_range is not None:
            data = self._apply_frame_slicing(data, frame_range)

        if select_q is not None and data.q_values is not None:
            data = self._apply_q_selection(data, select_q, q_tolerance)

        return data

    # ------------------------------------------------------------------
    # Post-load transforms
    # ------------------------------------------------------------------

    def _apply_frame_slicing(
        self,
        data: XPCSData,
        frame_range: tuple[int, int],
    ) -> XPCSData:
        """Slice an XPCSData to a sub-range of frames.

        Uses **1-based, inclusive** indexing on both ends, consistent with
        the homodyne convention.  For a 2-D c2 the slice is
        ``c2[start_0:end, start_0:end]``; for a 3-D c2 the time axes are
        sliced as ``c2[:, start_0:end, start_0:end]``.

        Args:
            data: Loaded XPCSData to slice.
            frame_range: ``(start, end)`` with 1-based inclusive indices.

        Returns:
            New XPCSData with sliced arrays.

        Raises:
            ValueError: If ``start`` < 1, ``end`` > n_frames, or
                ``start`` > ``end``.
        """
        start, end = frame_range
        n_frames = data.n_times

        start_0 = start - 1  # convert to 0-based
        if start_0 < 0:
            raise ValueError(
                f"frame_range start must be >= 1 (1-based), got {start}"
            )
        if end > n_frames:
            raise ValueError(
                f"frame_range end {end} exceeds n_frames {n_frames}"
            )
        if start > end:
            raise ValueError(
                f"frame_range start {start} must be <= end {end}"
            )

        logger.info(
            "Frame slicing: frames %d–%d (0-based %d:%d), %d → %d frames",
            start,
            end,
            start_0,
            end,
            n_frames,
            end - start_0,
        )

        if data.c2.ndim == 2:
            c2_sliced = data.c2[start_0:end, start_0:end]
        else:
            c2_sliced = data.c2[:, start_0:end, start_0:end]

        t1_sliced = data.t1[start_0:end]
        t2_sliced = data.t2[start_0:end]

        return XPCSData(
            c2=c2_sliced,
            t1=t1_sliced,
            t2=t2_sliced,
            q=data.q,
            q_values=data.q_values,
            phi_angles=data.phi_angles,
            uncertainties=data.uncertainties,
            metadata=data.metadata,
        )

    def _apply_q_selection(
        self,
        data: XPCSData,
        select_q: float,
        q_tolerance: float | None,
    ) -> XPCSData:
        """Select q-bin(s) from a multi-q XPCSData by proximity to a target.

        When ``q_tolerance`` is ``None``, only the single nearest q-bin is
        kept.  When ``q_tolerance`` is given, all bins within that absolute
        deviation are kept.

        If the result is a single q-bin the data is reduced to 2-D: ``c2``
        drops to ``(n_t, n_t)``, ``q`` is set to the selected q-value, and
        ``q_values`` is cleared.  If multiple bins are selected the data
        remains 3-D with updated ``q_values``.

        Args:
            data: Multi-q XPCSData (``q_values`` must not be ``None``).
            select_q: Target wavevector in Å⁻¹.
            q_tolerance: Maximum absolute deviation in Å⁻¹.  ``None`` means
                select only the single closest bin.

        Returns:
            Sliced XPCSData.

        Raises:
            ValueError: If no q-bin lies within the requested tolerance.
        """
        q_values = data.q_values
        assert q_values is not None  # guaranteed by caller

        indices, selected_q = select_optimal_wavevector(q_values, select_q, q_tolerance)

        if indices.size == 0:
            raise ValueError(
                f"No q-bin within tolerance {q_tolerance} Å⁻¹ of target "
                f"{select_q:.6g} Å⁻¹.  Available q: {q_values.tolist()}"
            )

        deviation = float(np.abs(selected_q - select_q).min())
        logger.info(
            "Q selection: target=%.6g Å⁻¹, selected %d bin(s) "
            "(min deviation=%.4g Å⁻¹)",
            select_q,
            indices.size,
            deviation,
        )

        c2_sel = data.c2[indices]  # (n_sel, n_t, n_t) or (1, n_t, n_t)
        phi_sel: np.ndarray | None = None
        if data.phi_angles is not None:
            phi_sel = data.phi_angles[indices]

        if indices.size == 1:
            # Reduce to single-q 2-D
            return XPCSData(
                c2=c2_sel[0],
                t1=data.t1,
                t2=data.t2,
                q=float(selected_q[0]),
                q_values=None,
                phi_angles=phi_sel,
                uncertainties=data.uncertainties,
                metadata=data.metadata,
            )

        return XPCSData(
            c2=c2_sel,
            t1=data.t1,
            t2=data.t2,
            q=None,
            q_values=selected_q,
            phi_angles=phi_sel,
            uncertainties=data.uncertainties,
            metadata=data.metadata,
        )

    # ------------------------------------------------------------------
    # NPZ caching
    # ------------------------------------------------------------------

    def _load_with_cache(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from NPZ cache if valid, otherwise load from source and cache.

        Cache files sit alongside the original with the suffix
        ``<filename>.heterodyne_cache.npz``.  Validity is determined by
        comparing the stored mtime with the current source mtime; no content
        hashing is performed.  The cache write is non-fatal: if the filesystem
        is read-only or quota is exceeded, the warning is logged and loading
        continues normally from the source.

        Args:
            c2_key: Key for correlation data.
            time_key: Key for time array.
            q_key: Optional key for scalar wavevector.
            phi_key: Optional key for phi angles.

        Returns:
            XPCSData loaded from cache or from source.
        """
        cache = _cache_path_for(self.file_path)
        mtime = _source_mtime(self.file_path)

        if _cache_is_valid(self.file_path, cache):
            try:
                return _read_cache(cache)
            except Exception as exc:
                logger.warning(
                    "Cache read failed for %s (%s), reloading from source",
                    cache.name,
                    exc,
                )

        # Load from original source
        if self.format == "hdf5":
            data = self._load_hdf5(c2_key, time_key, q_key, phi_key)
        else:
            data = self._load_mat(c2_key, time_key, q_key, phi_key)

        # Write cache; failure is non-fatal
        try:
            _write_cache(cache, data, mtime)
        except Exception as exc:
            logger.warning("Could not write cache %s: %s", cache.name, exc)

        return data

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _detect_hdf5_format(self, f: h5py.File) -> str:
        """Detect the HDF5 layout convention used in an open file.

        Checks for well-known dataset paths to identify the format:

        - ``"aps_u"`` - APS-U twotime format: has
          ``xpcs/twotime/correlation_map`` and
          ``xpcs/qmap/dynamic_v_list_dim0``.
        - ``"aps_old"`` - APS legacy format: has ``xpcs/dqlist`` and
          ``exchange/C2T_all``.
        - ``"exchange"`` - APS-style ``/exchange/`` group (existing
          heterodyne convention).
        - ``"flat"`` - Default: datasets at root level.

        Args:
            f: Open ``h5py.File`` object.

        Returns:
            One of ``"aps_u"``, ``"aps_old"``, ``"exchange"``, ``"flat"``.
        """
        if (
            "xpcs/twotime/correlation_map" in f
            and "xpcs/qmap/dynamic_v_list_dim0" in f
        ):
            logger.debug("HDF5 format detected: aps_u")
            return "aps_u"
        if "xpcs/dqlist" in f and "exchange/C2T_all" in f:
            logger.debug("HDF5 format detected: aps_old")
            return "aps_old"
        if "exchange" in f:
            logger.debug("HDF5 format detected: exchange")
            return "exchange"
        logger.debug("HDF5 format detected: flat")
        return "flat"

    def _load_hdf5_aps_u(self, f: h5py.File) -> XPCSData:
        """Load from APS-U twotime HDF5 format.

        Layout expectations:

        - ``xpcs/qmap/dynamic_v_list_dim0`` - 1-D array of q-values (Å⁻¹).
        - ``xpcs/qmap/dynamic_v_list_dim1`` - 1-D array of phi-values.
        - ``xpcs/twotime/processed_bins`` - 1-based bin indices (int array).
        - ``xpcs/twotime/correlation_map`` - group whose datasets are
          sorted half-matrices keyed as ``c2_00001``, ``c2_00002``, etc.

        The bin index encodes ``(q_idx, phi_idx)`` via::

            bin_idx = bin - 1          # 0-based
            q_idx   = bin_idx // n_phi
            phi_idx = bin_idx % n_phi

        Args:
            f: Open ``h5py.File`` object.

        Returns:
            XPCSData with a 3-D ``c2`` of shape ``(n_valid_bins, n_t, n_t)``,
            ``q_values`` set to the per-bin q-values, and ``phi_angles`` set
            to the per-bin phi-values.
        """
        q_all = np.asarray(f["xpcs/qmap/dynamic_v_list_dim0"], dtype=np.float64).ravel()
        phi_all = np.asarray(f["xpcs/qmap/dynamic_v_list_dim1"], dtype=np.float64).ravel()
        n_phi = phi_all.shape[0]

        bins_raw = np.asarray(f["xpcs/twotime/processed_bins"]).ravel()
        # Convert 1-based bin indices to 0-based
        bins_0 = bins_raw.astype(np.int64) - 1

        corr_group = f["xpcs/twotime/correlation_map"]
        sorted_keys = sorted(corr_group.keys())

        matrices: list[np.ndarray] = []
        q_sel: list[float] = []
        phi_sel: list[float] = []

        for key, bin_idx in zip(sorted_keys, bins_0, strict=True):
            if bin_idx < 0:
                logger.debug("APS-U loader: skipping invalid bin index %d (key %s)", bin_idx, key)
                continue
            q_idx = int(bin_idx) // n_phi
            phi_idx = int(bin_idx) % n_phi
            if q_idx >= q_all.shape[0] or phi_idx >= phi_all.shape[0]:
                logger.warning(
                    "APS-U loader: index out of range (q_idx=%d/%d, phi_idx=%d/%d) "
                    "for key %s, skipping",
                    q_idx,
                    q_all.shape[0],
                    phi_idx,
                    phi_all.shape[0],
                    key,
                )
                continue
            half = np.asarray(corr_group[key], dtype=np.float64)
            full = _reconstruct_from_half_matrix(half)
            matrices.append(full)
            q_sel.append(float(q_all[q_idx]))
            phi_sel.append(float(phi_all[phi_idx]))

        if not matrices:
            raise ValueError("APS-U HDF5 file contains no valid correlation matrices")

        c2 = np.stack(matrices, axis=0)  # (n_bins, n_t, n_t)
        n_frames = c2.shape[1]

        t: np.ndarray
        for t_candidate in ("xpcs/twotime/t", "xpcs/t", "exchange/t"):
            if t_candidate in f:
                t = np.asarray(f[t_candidate], dtype=np.float64).ravel()
                logger.debug("APS-U loader: time from '%s'", t_candidate)
                break
        else:
            t = np.arange(n_frames, dtype=np.float64)
            logger.warning("APS-U loader: time dataset not found, using frame indices")

        metadata: dict[str, Any] = {}
        for attr_key in f.attrs:
            value = f.attrs[attr_key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[attr_key] = value

        logger.info(
            "APS-U loader: loaded %d bins, %d frames, q in [%.4g, %.4g] Å⁻¹",
            c2.shape[0],
            n_frames,
            min(q_sel),
            max(q_sel),
        )
        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=None,
            q_values=np.array(q_sel, dtype=np.float64),
            phi_angles=np.array(phi_sel, dtype=np.float64),
            metadata=metadata,
        )

    def _load_hdf5_aps_old(self, f: h5py.File) -> XPCSData:
        """Load from APS legacy twotime HDF5 format.

        Layout expectations:

        - ``xpcs/dqlist`` - shape (1, N) q-values (Å⁻¹); squeezed to (N,).
        - ``xpcs/dphilist`` - shape (1, N) phi-values; squeezed to (N,).
        - ``exchange/C2T_all`` - group of half-matrix datasets, one per q-bin.

        Args:
            f: Open ``h5py.File`` object.

        Returns:
            XPCSData with a 3-D ``c2`` of shape ``(n_q, n_t, n_t)``,
            ``q_values``, and ``phi_angles``.
        """
        q_raw = np.asarray(f["xpcs/dqlist"], dtype=np.float64)
        q_values = q_raw.squeeze()
        if q_values.ndim == 0:
            q_values = q_values.reshape(1)

        phi_raw = np.asarray(f["xpcs/dphilist"], dtype=np.float64)
        phi_values = phi_raw.squeeze()
        if phi_values.ndim == 0:
            phi_values = phi_values.reshape(1)

        c2t_group = f["exchange/C2T_all"]
        sorted_keys = sorted(c2t_group.keys())

        matrices: list[np.ndarray] = []
        for key in sorted_keys:
            half = np.asarray(c2t_group[key], dtype=np.float64)
            full = _reconstruct_from_half_matrix(half)
            matrices.append(full)

        if not matrices:
            raise ValueError("APS old HDF5 file: exchange/C2T_all group is empty")

        c2 = np.stack(matrices, axis=0)  # (n_q, n_t, n_t)
        n_frames = c2.shape[1]

        t: np.ndarray
        for t_candidate in ("exchange/t", "exchange/tau", "xpcs/t"):
            if t_candidate in f:
                t = np.asarray(f[t_candidate], dtype=np.float64).ravel()
                logger.debug("APS old loader: time from '%s'", t_candidate)
                break
        else:
            t = np.arange(n_frames, dtype=np.float64)
            logger.warning("APS old loader: time dataset not found, using frame indices")

        metadata: dict[str, Any] = {}
        for attr_key in f.attrs:
            value = f.attrs[attr_key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[attr_key] = value

        logger.info(
            "APS old loader: loaded %d q-bins, %d frames",
            c2.shape[0],
            n_frames,
        )
        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=None,
            q_values=q_values,
            phi_angles=phi_values,
            metadata=metadata,
        )

    def _load_hdf5(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from HDF5 file.

        Supports four layout conventions detected automatically via
        :meth:`_detect_hdf5_format`:

        1. **APS-U** - ``xpcs/twotime/correlation_map`` with q/phi maps.
        2. **APS old** - ``xpcs/dqlist`` + ``exchange/C2T_all``.
        3. **Exchange group** - APS-style ``/exchange/`` group containing
           ``twotime_corr``, ``tau``, ``q_val`` (or ``q_values``), and
           optionally ``phi``.
        4. **Flat** - datasets at root level (``/c2``, ``/t``, etc.).
        """
        with h5py.File(self.file_path, "r") as f:
            fmt = self._detect_hdf5_format(f)

            if fmt == "aps_u":
                return self._load_hdf5_aps_u(f)

            if fmt == "aps_old":
                return self._load_hdf5_aps_old(f)

            if fmt == "exchange":
                return self._load_hdf5_exchange(f)

            # --- Flat layout ---
            if c2_key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Key '{c2_key}' not found. Available: {available}"
                )
            c2 = np.asarray(f[c2_key], dtype=np.float64)

            if time_key in f:
                t = np.asarray(f[time_key], dtype=np.float64)
            else:
                logger.warning(
                    "Time key '%s' not found, using indices", time_key
                )
                n_t = c2.shape[-2] if c2.ndim == 3 else c2.shape[0]
                t = np.arange(n_t, dtype=np.float64)

            q: float | None = None
            q_values: np.ndarray | None = None
            if q_key and q_key in f:
                q_arr = np.asarray(f[q_key], dtype=np.float64).ravel()
                if c2.ndim == 3 and q_arr.shape[0] == c2.shape[0]:
                    q_values = q_arr
                else:
                    q = float(q_arr[0])

            phi: np.ndarray | None = None
            if phi_key and phi_key in f:
                phi = np.asarray(f[phi_key], dtype=np.float64)

            metadata: dict[str, Any] = {}
            for key in f.attrs:
                value = f.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif isinstance(value, np.generic):
                    value = value.item()
                metadata[key] = value

        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            q_values=q_values,
            phi_angles=phi,
            metadata=metadata,
        )

    def _load_hdf5_exchange(self, f: h5py.File) -> XPCSData:
        """Load from an APS-style ``/exchange/`` HDF5 group.

        Expected datasets inside ``/exchange/``:

        - ``twotime_corr`` (required) - two-time correlation, shape (N, N)
          or (n_q, N, N).
        - ``tau`` or ``t`` (required) - time axis.
        - ``q_val`` or ``q_values`` (optional) - per-q-bin wavevectors.
        - ``phi`` or ``phi_angles`` (optional) - azimuthal angles.
        """
        exch = f["exchange"]

        c2: np.ndarray | None = None
        for c2_candidate in ("twotime_corr", "twotime", "c2", "corr"):
            if c2_candidate in exch:
                c2 = np.asarray(exch[c2_candidate], dtype=np.float64)
                logger.debug(
                    "Exchange group: loaded c2 from 'exchange/%s'", c2_candidate
                )
                break
        if c2 is None:
            available = list(exch.keys())
            raise KeyError(
                f"No two-time correlation dataset found in /exchange/. "
                f"Available keys: {available}"
            )

        t: np.ndarray | None = None
        for t_candidate in ("tau", "t", "times", "delay_time"):
            if t_candidate in exch:
                t = np.asarray(exch[t_candidate], dtype=np.float64).ravel()
                logger.debug(
                    "Exchange group: loaded time from 'exchange/%s'", t_candidate
                )
                break
        if t is None:
            n_t = c2.shape[-2] if c2.ndim >= 2 else c2.shape[0]
            t = np.arange(n_t, dtype=np.float64)
            logger.warning("Exchange group: time array not found, using indices")

        q: float | None = None
        q_values: np.ndarray | None = None
        for q_candidate in ("q_val", "q_values", "q", "qval"):
            if q_candidate in exch:
                q_arr = np.asarray(exch[q_candidate], dtype=np.float64).ravel()
                if c2.ndim == 3 and q_arr.shape[0] == c2.shape[0]:
                    q_values = q_arr
                else:
                    q = float(q_arr[0])
                break

        phi: np.ndarray | None = None
        for phi_candidate in ("phi", "phi_angles", "azimuth"):
            if phi_candidate in exch:
                phi = np.asarray(exch[phi_candidate], dtype=np.float64)
                break

        metadata: dict[str, Any] = {}
        for key in f.attrs:
            value = f.attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[key] = value

        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            q_values=q_values,
            phi_angles=phi,
            metadata=metadata,
        )

    def _load_npz(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from NPZ file.

        Uses allow_pickle=False to prevent code execution from untrusted files.
        """
        try:
            npz_file = np.load(self.file_path, allow_pickle=False)
        except ValueError as exc:
            raise ValueError(
                f"NPZ file {self.file_path} contains objects that require "
                "deserialization, which is not allowed for security"
            ) from exc

        with npz_file as data:
            if c2_key not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Key '{c2_key}' not found. Available: {available}"
                )
            c2 = np.asarray(data[c2_key], dtype=np.float64)

            if time_key in data:
                t = np.asarray(data[time_key], dtype=np.float64)
            else:
                n_t = c2.shape[-2] if c2.ndim == 3 else c2.shape[0]
                t = np.arange(n_t, dtype=np.float64)

            q: float | None = None
            q_values: np.ndarray | None = None
            if q_key and q_key in data:
                q_arr = np.asarray(data[q_key], dtype=np.float64).ravel()
                if c2.ndim == 3 and q_arr.shape[0] == c2.shape[0]:
                    q_values = q_arr
                else:
                    q = float(q_arr[0])

            phi: np.ndarray | None = None
            if phi_key and phi_key in data:
                phi = np.asarray(data[phi_key], dtype=np.float64)

        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            q_values=q_values,
            phi_angles=phi,
        )

    def _load_npy(self) -> XPCSData:
        """Load from NPY file (just the array)."""
        c2 = np.load(self.file_path, allow_pickle=False).astype(np.float64)
        if c2.ndim < 2:
            raise ValueError(
                f"Expected 2D or 3D array from {self.file_path}, got {c2.ndim}D"
            )
        t = np.arange(c2.shape[-2] if c2.ndim == 3 else c2.shape[0])
        return XPCSData(c2=c2, t1=t, t2=t)

    def _load_mat(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from MATLAB .mat file."""
        from scipy.io import loadmat

        data = loadmat(self.file_path)

        if c2_key not in data:
            available = [k for k in data.keys() if not k.startswith("__")]
            raise KeyError(
                f"Key '{c2_key}' not found. Available: {available}"
            )

        c2 = np.asarray(data[c2_key], dtype=np.float64)

        if time_key in data:
            t = np.asarray(data[time_key], dtype=np.float64).ravel()
        else:
            n_t = c2.shape[-2] if c2.ndim == 3 else c2.shape[0]
            t = np.arange(n_t, dtype=np.float64)

        q: float | None = None
        q_values: np.ndarray | None = None
        if q_key and q_key in data:
            q_arr = np.asarray(data[q_key], dtype=np.float64).ravel()
            if c2.ndim == 3 and q_arr.shape[0] == c2.shape[0]:
                q_values = q_arr
            else:
                q = float(q_arr[0])

        phi: np.ndarray | None = None
        if phi_key and phi_key in data:
            phi = np.asarray(data[phi_key], dtype=np.float64).ravel()

        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            q_values=q_values,
            phi_angles=phi,
        )


# ---------------------------------------------------------------------------
# Wavevector selection
# ---------------------------------------------------------------------------


def select_optimal_wavevector(
    q_values: np.ndarray,
    target_q: float,
    tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select q-bin indices closest to a target wavevector.

    Args:
        q_values: Available q-values array (Å⁻¹).
        target_q: Target wavevector in Å⁻¹.
        tolerance: Maximum absolute deviation from ``target_q`` in Å⁻¹.  If
            ``None``, only the single closest q-value is selected.

    Returns:
        Tuple of ``(selected_indices, selected_q_values)`` where both are
        1-D NumPy arrays.  ``selected_indices`` contains integer indices into
        ``q_values``; ``selected_q_values`` contains the corresponding
        q-values.

    Raises:
        ValueError: If ``q_values`` is empty.
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.size == 0:
        raise ValueError("q_values is empty")

    deviations = np.abs(q_values - target_q)

    if tolerance is None:
        closest_idx = int(np.argmin(deviations))
        indices = np.array([closest_idx], dtype=np.intp)
    else:
        indices = np.where(deviations <= tolerance)[0].astype(np.intp)

    return indices, q_values[indices]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def load_xpcs_data(
    file_path: Path | str,
    c2_key: str = "c2",
    time_key: str = "t",
    format: str | None = None,
    use_cache: bool = False,
    frame_range: tuple[int, int] | None = None,
    select_q: float | None = None,
    q_tolerance: float | None = None,
) -> XPCSData:
    """Convenience function to load XPCS data.

    Args:
        file_path: Path to data file.
        c2_key: Key for correlation data.
        time_key: Key for time array.
        format: File format (auto-detected if None).
        use_cache: Enable NPZ caching to avoid re-reading large source files.
        frame_range: Optional ``(start, end)`` with 1-based inclusive frame
            indices.  See :meth:`XPCSDataLoader.load` for details.
        select_q: Target wavevector in Å⁻¹ for q-bin selection.  Applied only
            when the loaded data has multiple q-bins.
        q_tolerance: Maximum absolute deviation from ``select_q`` in Å⁻¹.
            ``None`` selects only the single closest bin.

    Returns:
        XPCSData container.
    """
    loader = XPCSDataLoader(file_path, format=format)
    return loader.load(
        c2_key=c2_key,
        time_key=time_key,
        use_cache=use_cache,
        frame_range=frame_range,
        select_q=select_q,
        q_tolerance=q_tolerance,
    )
