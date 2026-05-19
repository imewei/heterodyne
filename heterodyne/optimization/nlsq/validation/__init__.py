"""NLSQ validation for heterodyne model optimization.

Provides pre-fit input validation and post-fit quality checks:

- ``InputValidator``: pre-fit checks for data, bounds, and initial parameters
- ``validate_array_dimensions``, ``validate_no_nan_inf``,
  ``validate_bounds_consistency``, ``validate_initial_params``:
  functional input validators (homodyne parity)
- ``ResultValidator``: homodyne-parity result validator (bool API)
- ``validate_optimized_params``, ``validate_covariance``,
  ``validate_result_consistency``: functional result validators (homodyne parity)
- ``BoundsValidator``: checks parameters against physical bounds
- ``ConvergenceValidator``: assesses convergence quality
- ``FitQualityConfig`` / ``FitQualityReport`` / ``validate_fit_quality``:
  homodyne-parity post-fit quality validator with configurable thresholds
- ``classify_fit_quality``: legacy 3-bin fit-quality classifier
- ``ValidationIssue`` / ``ValidationReport`` / ``ValidationSeverity``:
  structured report types (heterodyne-native)
"""

from __future__ import annotations

from heterodyne.optimization.nlsq.validation.bounds import BoundsValidator
from heterodyne.optimization.nlsq.validation.convergence import ConvergenceValidator
from heterodyne.optimization.nlsq.validation.fit_quality import (
    FitQualityConfig,
    FitQualityReport,
    FitQualityValidator,
    classify_fit_quality,
    validate_fit_quality,
)
from heterodyne.optimization.nlsq.validation.input_validator import (
    InputValidator,
    validate_array_dimensions,
    validate_bounds_consistency,
    validate_initial_params,
    validate_no_nan_inf,
)
from heterodyne.optimization.nlsq.validation.result import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from heterodyne.optimization.nlsq.validation.result_validator import (
    ResultValidator,
    validate_covariance,
    validate_optimized_params,
    validate_result_consistency,
)

__all__ = [
    # Input validation — class
    "InputValidator",
    # Input validation — functional (homodyne parity)
    "validate_array_dimensions",
    "validate_bounds_consistency",
    "validate_initial_params",
    "validate_no_nan_inf",
    # Result validation — class (homodyne parity)
    "ResultValidator",
    # Result validation — functional (homodyne parity)
    "validate_covariance",
    "validate_optimized_params",
    "validate_result_consistency",
    # Heterodyne-native validators
    "BoundsValidator",
    "ConvergenceValidator",
    # Fit quality (homodyne parity)
    "FitQualityConfig",
    "FitQualityReport",
    "FitQualityValidator",
    "classify_fit_quality",
    "validate_fit_quality",
    # Structured report types (heterodyne-native)
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
]
