"""Preprocessing pipeline for XPCS correlation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

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
            if c2.ndim != 2:
                return c2
            diag = np.diag(c2)
            # Avoid division by zero
            diag_safe = np.where(np.abs(diag) > 1e-10, diag, 1.0)
            # Normalize using outer product of sqrt(diag)
            norm = np.sqrt(np.outer(diag_safe, diag_safe))
            return c2 / norm
        
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
            mean = np.nanmean(c2)
            std = np.nanstd(c2)
            outlier_mask = np.abs(c2 - mean) > n_sigma * std
            
            result = c2.copy()
            if replace_with == "median":
                result[outlier_mask] = np.nanmedian(c2)
            elif replace_with == "nan":
                result[outlier_mask] = np.nan
            elif replace_with == "clip":
                result = np.clip(result, mean - n_sigma * std, mean + n_sigma * std)
            
            n_outliers = np.sum(outlier_mask)
            if n_outliers > 0:
                logger.info(f"Removed {n_outliers} outliers ({100*n_outliers/c2.size:.2f}%)")
            
            return result
        
        return self.add_step(f"remove_outliers({n_sigma}σ)", _remove_outliers)
    
    def symmetrize(self) -> PreprocessingPipeline:
        """Add symmetrization step for 2D correlation.
        
        Makes c2(t1, t2) = c2(t2, t1).
        """
        def _symmetrize(c2: np.ndarray) -> np.ndarray:
            if c2.ndim != 2:
                return c2
            return (c2 + c2.T) / 2
        
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
            logger.debug(f"Applying: {name}")
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
