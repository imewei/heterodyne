"""Data loading and validation for XPCS experiments."""

from heterodyne.data.angle_filtering import (
    compute_angle_quality,
    filter_by_angle_range,
    find_nearest_angle,
    select_angles,
)
from heterodyne.data.config import DataConfig
from heterodyne.data.filtering_utils import (
    apply_q_range_filter,
    apply_sigma_clip,
    apply_time_window,
    compute_data_mask,
)
from heterodyne.data.memory_manager import (
    AdaptiveChunker,
    ChunkInfo,
    MemoryManager,
    MemoryMapManager,
    MemoryPressureLevel,
    MemoryPressureMonitor,
)
from heterodyne.data.optimization import (
    DatasetSizeCategory,
    categorize_dataset,
    create_loading_plan,
    process_chunks_parallel,
)
from heterodyne.data.performance_engine import (
    PerformanceEngine,
    TieredCache,
    TieredCacheConfig,
)
from heterodyne.data.phi_filtering import PhiAngleFilter, filter_by_phi
from heterodyne.data.preprocessing import PreprocessingPipeline, preprocess_correlation
from heterodyne.data.quality_controller import (
    QualityController,
    QualityLevel,
    QualityMetric,
    QualityReport,
)
from heterodyne.data.types import AngleRange, DataSlice, FilterResult, QRange
from heterodyne.data.validation import DataQualityReport, validate_xpcs_data
from heterodyne.data.validators import (
    validate_correlation_shape,
    validate_no_nan,
    validate_q_range,
    validate_time_arrays,
    validate_weights,
)
from heterodyne.data.xpcs_loader import (
    XPCSDataLoader,
    load_xpcs_data,
    select_optimal_wavevector,
)

__all__ = [
    # Types
    "DataSlice",
    "QRange",
    "AngleRange",
    "FilterResult",
    # Config
    "DataConfig",
    # Validators
    "validate_correlation_shape",
    "validate_time_arrays",
    "validate_q_range",
    "validate_weights",
    "validate_no_nan",
    # Filtering utilities
    "apply_time_window",
    "apply_q_range_filter",
    "apply_sigma_clip",
    "compute_data_mask",
    # Angle filtering
    "filter_by_angle_range",
    "select_angles",
    "find_nearest_angle",
    "compute_angle_quality",
    # Quality controller
    "QualityLevel",
    "QualityMetric",
    "QualityReport",
    "QualityController",
    # Loader
    "XPCSDataLoader",
    "load_xpcs_data",
    "select_optimal_wavevector",
    # Validation
    "validate_xpcs_data",
    "DataQualityReport",
    # Preprocessing
    "PreprocessingPipeline",
    "preprocess_correlation",
    "PhiAngleFilter",
    "filter_by_phi",
    # Memory management
    "MemoryManager",
    "MemoryMapManager",
    "AdaptiveChunker",
    "ChunkInfo",
    "MemoryPressureLevel",
    "MemoryPressureMonitor",
    # Performance
    "PerformanceEngine",
    "TieredCache",
    "TieredCacheConfig",
    # Optimization
    "DatasetSizeCategory",
    "categorize_dataset",
    "create_loading_plan",
    "process_chunks_parallel",
]
