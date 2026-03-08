"""Preprocessing pipeline for XPCS correlation data."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations."""

    c2: np.ndarray
    applied_steps: list[str] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)


def _remove_outliers_2d(
    c2_slice: np.ndarray,
    n_sigma: float,
    replace_with: str,
) -> np.ndarray:
    """Remove outliers from a single 2D correlation matrix.

    For square matrices, statistics are computed from off-diagonal elements
    to avoid biasing by the (typically larger) diagonal values.

    Args:
        c2_slice: 2D correlation array.
        n_sigma: Number of standard deviations for outlier threshold.
        replace_with: Replacement strategy ('median', 'nan', 'clip').

    Returns:
        Array with outliers replaced.
    """
    if c2_slice.ndim == 2 and c2_slice.shape[0] == c2_slice.shape[1]:
        off_diag_mask = ~np.eye(c2_slice.shape[0], dtype=bool)
        off_diag = c2_slice[off_diag_mask]
        mean = np.nanmean(off_diag)
        std = np.nanstd(off_diag)
    else:
        mean = np.nanmean(c2_slice)
        std = np.nanstd(c2_slice)
    outlier_mask = np.abs(c2_slice - mean) > n_sigma * std

    result = c2_slice.copy()
    if replace_with == "median":
        result[outlier_mask] = np.nanmedian(c2_slice)
    elif replace_with == "nan":
        result[outlier_mask] = np.nan
    elif replace_with == "clip":
        result = np.clip(result, mean - n_sigma * std, mean + n_sigma * std)
    return result


class PreprocessingPipeline:
    """Pipeline for preprocessing XPCS correlation data.

    Supports operations like:
    - Baseline subtraction
    - Normalization
    - Outlier removal
    - Smoothing
    """

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self._steps: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []

    def add_step(
        self,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
    ) -> PreprocessingPipeline:
        """Add a preprocessing step.

        Args:
            name: Step name for logging
            func: Function that transforms c2 array

        Returns:
            Self for chaining
        """
        self._steps.append((name, func))
        return self

    def normalize_diagonal(self) -> PreprocessingPipeline:
        """Add diagonal normalization step.

        Normalizes c2 so that diagonal values are 1.
        """
        def _normalize(c2: np.ndarray) -> np.ndarray:
            if c2.ndim == 3:
                # Batch of matrices: normalize each slice
                result = np.empty_like(c2)
                for i in range(c2.shape[0]):
                    result[i] = _normalize(c2[i])
                return result
            if c2.ndim != 2:
                return c2
            if c2.shape[0] != c2.shape[1]:
                raise ValueError(f"Expected square matrix, got shape {c2.shape}")
            diag = np.diag(c2)
            # Avoid division by zero
            diag_safe = np.where(np.abs(diag) > 1e-10, diag, 1.0)
            # Normalize using outer product of sqrt(diag)
            norm = np.sqrt(np.outer(diag_safe, diag_safe))
            return np.asarray(c2 / norm)

        return self.add_step("normalize_diagonal", _normalize)

    def subtract_baseline(self, baseline: float = 1.0) -> PreprocessingPipeline:
        """Add baseline subtraction step.

        Args:
            baseline: Baseline value to subtract
        """
        def _subtract(c2: np.ndarray) -> np.ndarray:
            return c2 - baseline

        return self.add_step(f"subtract_baseline({baseline})", _subtract)

    def clip_values(
        self,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> PreprocessingPipeline:
        """Add value clipping step.

        Args:
            min_val: Minimum value
            max_val: Maximum value
        """
        def _clip(c2: np.ndarray) -> np.ndarray:
            return np.clip(c2, min_val, max_val)

        return self.add_step(f"clip({min_val}, {max_val})", _clip)

    def remove_outliers(
        self,
        n_sigma: float = 5.0,
        replace_with: str = "median",
    ) -> PreprocessingPipeline:
        """Add outlier removal step.

        Args:
            n_sigma: Number of standard deviations for outlier threshold
            replace_with: Replacement strategy ('median', 'nan', 'clip')
        """
        def _remove_outliers(c2: np.ndarray) -> np.ndarray:
            if c2.ndim == 3:
                # Batch: process each phi-slice independently
                result = np.empty_like(c2)
                total_outliers = 0
                for i in range(c2.shape[0]):
                    before = c2[i]
                    after = _remove_outliers_2d(before, n_sigma, replace_with)
                    total_outliers += np.sum(before != after)
                    result[i] = after
                if total_outliers > 0:
                    logger.info("Removed %d outliers (%.2f%%)", total_outliers, 100*total_outliers/c2.size)
                return result

            result = _remove_outliers_2d(c2, n_sigma, replace_with)
            n_outliers = np.sum(c2 != result)
            if n_outliers > 0:
                logger.info("Removed %d outliers (%.2f%%)", n_outliers, 100*n_outliers/c2.size)
            return result

        return self.add_step(f"remove_outliers({n_sigma}σ)", _remove_outliers)

    def symmetrize(self) -> PreprocessingPipeline:
        """Add symmetrization step for correlation matrices.

        Makes c2(t1, t2) = c2(t2, t1). Handles both 2D and 3D (batch) data.
        """
        def _symmetrize(c2: np.ndarray) -> np.ndarray:
            if c2.ndim == 2:
                return np.asarray(np.nanmean(np.stack([c2, c2.T]), axis=0))
            elif c2.ndim == 3:
                # Batch of matrices: symmetrize each slice
                transposed = np.transpose(c2, (0, 2, 1))
                return np.asarray(np.nanmean(np.stack([c2, transposed]), axis=0))
            return c2

        return self.add_step("symmetrize", _symmetrize)

    def crop_time(
        self,
        t_start: int = 0,
        t_end: int | None = None,
    ) -> PreprocessingPipeline:
        """Add time cropping step.

        Args:
            t_start: Starting index
            t_end: Ending index (exclusive), None for end
        """
        if t_start < 0:
            raise ValueError(f"Crop bounds must be non-negative, got start={t_start}")
        if t_end is not None:
            if t_end < 0:
                raise ValueError(f"Crop bounds must be non-negative, got end={t_end}")
            if t_start >= t_end:
                raise ValueError(f"start ({t_start}) must be less than end ({t_end})")

        def _crop(c2: np.ndarray) -> np.ndarray:
            if c2.ndim == 2:
                return c2[t_start:t_end, t_start:t_end]
            elif c2.ndim == 3:
                return c2[:, t_start:t_end, t_start:t_end]
            return c2

        return self.add_step(f"crop_time({t_start}:{t_end})", _crop)

    def process(self, c2: np.ndarray) -> PreprocessingResult:
        """Apply all preprocessing steps.

        Args:
            c2: Input correlation array

        Returns:
            PreprocessingResult with processed data
        """
        result = c2.copy()
        applied = []

        for name, func in self._steps:
            logger.debug("Applying: %s", name)
            result = func(result)
            applied.append(name)

        # Compute statistics
        statistics = {
            "min": float(np.nanmin(result)),
            "max": float(np.nanmax(result)),
            "mean": float(np.nanmean(result)),
            "std": float(np.nanstd(result)),
            "nan_count": int(np.sum(np.isnan(result))),
        }

        return PreprocessingResult(
            c2=result,
            applied_steps=applied,
            statistics=statistics,
        )


def preprocess_correlation(
    c2: np.ndarray,
    normalize: bool = True,
    remove_outliers: bool = True,
    symmetrize: bool = True,
) -> PreprocessingResult:
    """Convenience function for standard preprocessing.

    Args:
        c2: Input correlation array
        normalize: Whether to normalize diagonal to 1
        remove_outliers: Whether to remove outliers
        symmetrize: Whether to symmetrize

    Returns:
        PreprocessingResult
    """
    pipeline = PreprocessingPipeline()

    if remove_outliers:
        pipeline.remove_outliers(n_sigma=5.0)

    if symmetrize:
        pipeline.symmetrize()

    if normalize:
        pipeline.normalize_diagonal()

    return pipeline.process(c2)


# ---------------------------------------------------------------------------
# Extended preprocessing: enums, provenance, normalization, noise reduction
# ---------------------------------------------------------------------------

class PreprocessingStage(Enum):
    """Stages in the preprocessing pipeline."""

    LOAD_RAW = "load_raw"
    VALIDATE_INPUT = "validate_input"
    NORMALIZE = "normalize"
    CORRECT_BASELINE = "correct_baseline"
    REDUCE_NOISE = "reduce_noise"
    TRANSFORM = "transform"
    VALIDATE_OUTPUT = "validate_output"


class NormalizationMethod(Enum):
    """Available normalization methods."""

    DIAGONAL = "diagonal"
    BASELINE = "baseline"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    PHYSICS_BASED = "physics_based"
    NONE = "none"


class NoiseReductionMethod(Enum):
    """Available noise reduction methods."""

    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_SMOOTH = "gaussian_smooth"
    WAVELET = "wavelet"
    NONE = "none"


@dataclass
class TransformationRecord:
    """Record of a single preprocessing transformation for provenance tracking.

    Attributes:
        stage: Which preprocessing stage this belongs to.
        method: Name of the method applied.
        parameters: Parameters used for the transformation.
        timestamp: ISO-format timestamp of when the transformation was applied.
        input_hash: SHA-256 hash of the input array.
        output_hash: SHA-256 hash of the output array.
    """

    stage: PreprocessingStage
    method: str
    parameters: dict[str, Any]
    timestamp: str
    input_hash: str
    output_hash: str


@dataclass
class PreprocessingProvenance:
    """Audit trail for preprocessing operations.

    Tracks every transformation applied to data, including hashes of
    input/output arrays for reproducibility verification.

    Attributes:
        records: List of transformation records.
        source_file: Path to the source data file, if applicable.
        created_at: ISO-format creation timestamp.
    """

    records: list[TransformationRecord] = field(default_factory=list)
    source_file: str | None = None
    created_at: str = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        """Set created_at to current UTC time if not provided."""
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def add_record(self, record: TransformationRecord) -> None:
        """Append a transformation record to the provenance trail.

        Args:
            record: The transformation record to add.
        """
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "source_file": self.source_file,
            "created_at": self.created_at,
            "records": [
                {
                    "stage": rec.stage.value,
                    "method": rec.method,
                    "parameters": rec.parameters,
                    "timestamp": rec.timestamp,
                    "input_hash": rec.input_hash,
                    "output_hash": rec.output_hash,
                }
                for rec in self.records
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessingProvenance:
        """Deserialize provenance from a dictionary.

        Args:
            d: Dictionary as produced by ``to_dict``.

        Returns:
            Reconstructed PreprocessingProvenance instance.
        """
        records = [
            TransformationRecord(
                stage=PreprocessingStage(rec["stage"]),
                method=rec["method"],
                parameters=rec["parameters"],
                timestamp=rec["timestamp"],
                input_hash=rec["input_hash"],
                output_hash=rec["output_hash"],
            )
            for rec in d.get("records", [])
        ]
        return cls(
            records=records,
            source_file=d.get("source_file"),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Array hashing utility
# ---------------------------------------------------------------------------


def _compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of array bytes for provenance tracking.

    Args:
        arr: NumPy array to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------


def normalize_zscore(c2: np.ndarray) -> np.ndarray:
    """Z-score normalization: (c2 - mean) / std.

    Handles zero standard deviation by returning a zero-centred array.

    Args:
        c2: Input correlation array.

    Returns:
        Z-score normalized array.
    """
    mean = np.nanmean(c2)
    std = np.nanstd(c2)
    if std == 0.0:
        logger.warning("Z-score normalization: std is zero, returning (c2 - mean)")
        return np.asarray(c2 - mean)
    return np.asarray((c2 - mean) / std)


def normalize_minmax(c2: np.ndarray) -> np.ndarray:
    """Min-max normalization: (c2 - min) / (max - min).

    Handles degenerate case (max == min) by returning zeros.

    Args:
        c2: Input correlation array.

    Returns:
        Array scaled to [0, 1].
    """
    c_min = np.nanmin(c2)
    c_max = np.nanmax(c2)
    denom = c_max - c_min
    if denom == 0.0:
        logger.warning("Min-max normalization: range is zero, returning zeros")
        return np.zeros_like(c2)
    return np.asarray((c2 - c_min) / denom)


def normalize_robust(c2: np.ndarray) -> np.ndarray:
    """Robust normalization: (c2 - median) / IQR.

    Uses the interquartile range (Q3 - Q1) as the scale factor.
    Handles IQR == 0 by returning (c2 - median).

    Args:
        c2: Input correlation array.

    Returns:
        Robustly normalized array.
    """
    flat = c2[~np.isnan(c2)] if np.any(np.isnan(c2)) else c2.ravel()
    median = np.nanmedian(c2)
    q1 = np.percentile(flat, 25)
    q3 = np.percentile(flat, 75)
    iqr = q3 - q1
    if iqr == 0.0:
        logger.warning("Robust normalization: IQR is zero, returning (c2 - median)")
        return np.asarray(c2 - median)
    return np.asarray((c2 - median) / iqr)


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------


def apply_baseline_correction(
    c2: np.ndarray,
    baseline: np.ndarray | float | None = None,
    method: str = "subtract",
) -> np.ndarray:
    """Apply baseline correction to a correlation matrix.

    Args:
        c2: Input correlation array (2D or 3D).
        baseline: Baseline value(s). If *None*, the baseline is estimated
            from the last 10% of off-diagonal elements (far from the
            diagonal, corresponding to long time delays).
        method: Correction strategy — ``"subtract"``, ``"divide"``, or
            ``"polynomial"``.

    Returns:
        Baseline-corrected array.

    Raises:
        ValueError: If *method* is not one of the supported strategies.
    """
    if method not in ("subtract", "divide", "polynomial"):
        raise ValueError(
            f"Unknown baseline correction method '{method}'. "
            "Supported: 'subtract', 'divide', 'polynomial'."
        )

    if method == "polynomial":
        return _baseline_polynomial(c2)

    if baseline is None:
        baseline = _estimate_baseline(c2)

    if method == "subtract":
        return np.asarray(c2 - baseline)

    # method == "divide"
    safe_baseline: np.ndarray | float
    if isinstance(baseline, np.ndarray):
        safe_baseline = np.where(np.abs(baseline) > 1e-15, baseline, 1.0)
    else:
        safe_baseline = float(baseline) if abs(float(baseline)) > 1e-15 else 1.0
    return np.asarray(c2 / safe_baseline)


def _estimate_baseline(c2: np.ndarray) -> float:
    """Estimate baseline from far-off-diagonal elements.

    Takes elements whose |i - j| distance is in the top 10% of the
    possible range, i.e. the longest time-delay elements.

    Args:
        c2: 2D or 3D correlation array.

    Returns:
        Scalar baseline estimate.
    """
    if c2.ndim == 3:
        # Average over batch dimension first
        mat = np.nanmean(c2, axis=0)
    else:
        mat = c2

    n = mat.shape[0]
    if n < 2:
        return float(np.nanmean(mat))

    threshold = int(0.9 * n)
    rows, cols = np.indices(mat.shape)
    far_mask = np.abs(rows - cols) >= threshold
    far_elements = mat[far_mask]

    if far_elements.size == 0:
        return float(np.nanmean(mat))

    return float(np.nanmean(far_elements))


def _baseline_polynomial(c2: np.ndarray, degree: int = 2) -> np.ndarray:
    """Remove polynomial baseline trend from a correlation matrix.

    Fits a polynomial of the given degree to the mean of each anti-diagonal
    (constant time-delay) and subtracts the trend.

    Args:
        c2: 2D or 3D correlation array.
        degree: Polynomial degree for the fit.

    Returns:
        Baseline-corrected array.
    """
    if c2.ndim == 3:
        result = np.empty_like(c2)
        for i in range(c2.shape[0]):
            result[i] = _baseline_polynomial(c2[i], degree=degree)
        return result

    n = c2.shape[0]
    if n < degree + 1:
        return c2.copy()

    # Compute mean along each anti-diagonal offset
    offsets = np.arange(n)
    diag_means = np.array(
        [float(np.nanmean(np.diag(c2, k))) for k in range(n)]
    )

    # Fit polynomial to anti-diagonal means
    valid = ~np.isnan(diag_means)
    if np.sum(valid) < degree + 1:
        return c2.copy()

    # Fit using numpy.polynomial (lowest-degree-first coefficients)
    coeffs = np.polynomial.polynomial.polyfit(
        offsets[valid], diag_means[valid], degree,
    )

    # Evaluate polynomial on unique lag values (1D), then index into result
    baseline_1d = np.polynomial.polynomial.polyval(offsets, coeffs)

    # Build baseline matrix by indexing: baseline[i,j] = baseline_1d[|i-j|]
    rows, cols = np.indices(c2.shape)
    baseline_matrix = baseline_1d[np.abs(rows - cols)]

    return np.asarray(c2 - baseline_matrix)


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------


def apply_noise_reduction(
    c2: np.ndarray,
    method: NoiseReductionMethod = NoiseReductionMethod.GAUSSIAN_SMOOTH,
    **kwargs: Any,
) -> np.ndarray:
    """Apply noise reduction to a correlation matrix.

    Args:
        c2: Input correlation array (2D or 3D).
        method: Noise reduction method to apply.
        **kwargs: Method-specific parameters.
            - ``kernel_size`` (int): Kernel size for median filter (default 3).
            - ``sigma`` (float): Standard deviation for Gaussian smooth (default 1.0).

    Returns:
        Noise-reduced array.
    """
    if method is NoiseReductionMethod.NONE:
        return c2.copy()

    if method is NoiseReductionMethod.MEDIAN_FILTER:
        return _noise_median_filter(c2, kernel_size=kwargs.get("kernel_size", 3))

    if method is NoiseReductionMethod.GAUSSIAN_SMOOTH:
        return _noise_gaussian_smooth(c2, sigma=kwargs.get("sigma", 1.0))

    if method is NoiseReductionMethod.WAVELET:
        logger.warning(
            "Wavelet noise reduction is not yet implemented; "
            "returning data unchanged."
        )
        return c2.copy()

    msg = f"Unsupported noise reduction method: {method}"  # type: ignore[unreachable]
    raise ValueError(msg)


def _noise_median_filter(c2: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter for noise reduction.

    Args:
        c2: Input array.
        kernel_size: Size of the median filter kernel.

    Returns:
        Filtered array.
    """
    from scipy.ndimage import median_filter

    return np.asarray(median_filter(c2, size=kernel_size))


def _noise_gaussian_smooth(c2: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing for noise reduction.

    Args:
        c2: Input array.
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        Smoothed array.
    """
    from scipy.ndimage import gaussian_filter

    return np.asarray(gaussian_filter(c2, sigma=sigma))


# ---------------------------------------------------------------------------
# Chunked processing
# ---------------------------------------------------------------------------


def process_chunked(
    c2: np.ndarray,
    pipeline: PreprocessingPipeline,
    chunk_size: int = 100,
) -> PreprocessingResult:
    """Process data in chunks along the first (batch) axis.

    For 3D data ``(n_phi, n_t, n_t)``, slices are processed in groups of
    *chunk_size* along axis 0 and concatenated.  For 2D data the pipeline
    is applied directly.

    Args:
        c2: Input correlation array (2D or 3D).
        pipeline: Configured :class:`PreprocessingPipeline`.
        chunk_size: Number of slices per chunk (along axis 0).

    Returns:
        Combined :class:`PreprocessingResult`.
    """
    if c2.ndim != 3:
        return pipeline.process(c2)

    n_phi = c2.shape[0]
    if n_phi <= chunk_size:
        return pipeline.process(c2)

    all_c2: list[np.ndarray] = []
    all_steps: list[str] = []
    combined_stats: dict[str, float] = {}

    for start in range(0, n_phi, chunk_size):
        end = min(start + chunk_size, n_phi)
        logger.debug("Processing chunk [%d:%d] of %d", start, end, n_phi)
        chunk_result = pipeline.process(c2[start:end])
        all_c2.append(chunk_result.c2)

        # Keep the step names from the first chunk (they are all the same)
        if not all_steps:
            all_steps = chunk_result.applied_steps

    merged = np.concatenate(all_c2, axis=0)

    # Recompute statistics over the full merged result
    combined_stats = {
        "min": float(np.nanmin(merged)),
        "max": float(np.nanmax(merged)),
        "mean": float(np.nanmean(merged)),
        "std": float(np.nanstd(merged)),
        "nan_count": int(np.sum(np.isnan(merged))),
    }

    return PreprocessingResult(
        c2=merged,
        applied_steps=all_steps,
        statistics=combined_stats,
    )


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def preprocess_xpcs_data(
    c2: np.ndarray,
    normalize_method: NormalizationMethod = NormalizationMethod.DIAGONAL,
    noise_reduction: NoiseReductionMethod = NoiseReductionMethod.NONE,
    remove_outliers: bool = True,
    symmetrize: bool = True,
    baseline_correction: bool = False,
    **kwargs: Any,
) -> PreprocessingResult:
    """Convenience function for comprehensive XPCS preprocessing.

    Builds a :class:`PreprocessingPipeline` with the requested steps and
    applies it to *c2*.  The processing order is:

    1. Outlier removal (optional)
    2. Symmetrization (optional)
    3. Baseline correction (optional)
    4. Normalization (configurable method)
    5. Noise reduction (configurable method)

    Args:
        c2: Input correlation array.
        normalize_method: Which normalization to apply.
        noise_reduction: Which noise reduction to apply.
        remove_outliers: Whether to remove outliers before other steps.
        symmetrize: Whether to symmetrize the correlation matrix.
        baseline_correction: Whether to apply baseline correction.
        **kwargs: Extra keyword arguments forwarded to noise reduction
            (e.g. ``kernel_size``, ``sigma``).

    Returns:
        :class:`PreprocessingResult` with processed data, step list,
        and summary statistics.
    """
    pipeline = PreprocessingPipeline()

    # 1. Outlier removal
    if remove_outliers:
        pipeline.remove_outliers(n_sigma=5.0)

    # 2. Symmetrize
    if symmetrize:
        pipeline.symmetrize()

    # 3. Baseline correction
    if baseline_correction:
        pipeline.add_step(
            "baseline_correction",
            lambda arr: apply_baseline_correction(arr),
        )

    # 4. Normalization
    if normalize_method is NormalizationMethod.DIAGONAL:
        pipeline.normalize_diagonal()
    elif normalize_method is NormalizationMethod.ZSCORE:
        pipeline.add_step("normalize_zscore", normalize_zscore)
    elif normalize_method is NormalizationMethod.MINMAX:
        pipeline.add_step("normalize_minmax", normalize_minmax)
    elif normalize_method is NormalizationMethod.ROBUST:
        pipeline.add_step("normalize_robust", normalize_robust)
    elif normalize_method is NormalizationMethod.BASELINE:
        pipeline.add_step(
            "normalize_baseline",
            lambda arr: apply_baseline_correction(arr, method="divide"),
        )
    elif normalize_method is NormalizationMethod.PHYSICS_BASED:
        # Physics-based normalization is equivalent to diagonal for XPCS
        pipeline.normalize_diagonal()
    elif normalize_method is NormalizationMethod.NONE:
        pass  # No normalization

    # 5. Noise reduction
    if noise_reduction is not NoiseReductionMethod.NONE:
        nr_method = noise_reduction
        nr_kwargs = {
            k: v for k, v in kwargs.items() if k in ("kernel_size", "sigma")
        }
        pipeline.add_step(
            f"noise_reduction({nr_method.value})",
            lambda arr, _m=nr_method, _kw=nr_kwargs: apply_noise_reduction(  # type: ignore[misc]
                arr, method=_m, **_kw
            ),
        )

    return pipeline.process(c2)
