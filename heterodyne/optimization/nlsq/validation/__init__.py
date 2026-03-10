"""NLSQ result validation for heterodyne model optimization.

Provides post-fit quality checks:
- ResultValidator: validates NLSQResult quality metrics
- BoundsValidator: checks parameters against physical bounds
- ConvergenceValidator: assesses convergence quality
"""

from __future__ import annotations

from heterodyne.optimization.nlsq.validation.bounds import BoundsValidator
from heterodyne.optimization.nlsq.validation.convergence import ConvergenceValidator
from heterodyne.optimization.nlsq.validation.fit_quality import classify_fit_quality
from heterodyne.optimization.nlsq.validation.result import (
    ResultValidator,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    "ResultValidator",
    "ValidationReport",
    "ValidationSeverity",
    "BoundsValidator",
    "ConvergenceValidator",
    "classify_fit_quality",
]
