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
from heterodyne.data.xpcs_loader import XPCSDataLoader, load_xpcs_data

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
    # Existing exports
    "XPCSDataLoader",
    "load_xpcs_data",
    "validate_xpcs_data",
    "DataQualityReport",
    "PreprocessingPipeline",
    "preprocess_correlation",
    "PhiAngleFilter",
    "filter_by_phi",
]
