"""NLSQ validation for heterodyne model optimization.

Provides pre-fit input validation and post-fit quality checks:

- ``InputValidator``: pre-fit checks for data, bounds, and initial parameters
- ``ResultValidator``: validates ``NLSQResult`` quality metrics
- ``BoundsValidator``: checks parameters against physical bounds
- ``ConvergenceValidator``: assesses convergence quality
- ``FitQualityConfig`` / ``FitQualityReport`` / ``validate_fit_quality``:
  homodyne-parity post-fit quality validator with configurable thresholds
- ``classify_fit_quality``: legacy 3-bin fit-quality classifier
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
from heterodyne.optimization.nlsq.validation.input_validator import InputValidator
from heterodyne.optimization.nlsq.validation.result import (
    ResultValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    "BoundsValidator",
    "ConvergenceValidator",
    "FitQualityConfig",
    "FitQualityReport",
    "FitQualityValidator",
    "InputValidator",
    "ResultValidator",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "classify_fit_quality",
    "validate_fit_quality",
]
