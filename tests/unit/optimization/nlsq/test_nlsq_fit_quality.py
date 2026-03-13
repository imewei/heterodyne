"""Tests for NLSQ fit quality validation."""

from __future__ import annotations

import numpy as np

from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.optimization.nlsq.validation.fit_quality import FitQualityValidator


class TestFitQualityValidator:
    """Tests for FitQualityValidator."""

    def _make_result(self, **kwargs) -> NLSQResult:
        defaults: dict[str, object] = {
            "parameters": np.array([1000.0, 0.5]),
            "parameter_names": ["D0_ref", "alpha_ref"],
            "success": True,
            "message": "converged",
            "reduced_chi_squared": 1.5,
        }
        defaults.update(kwargs)
        return NLSQResult(**defaults)

    def test_good_fit_passes(self) -> None:
        """Normal chi-squared passes."""
        validator = FitQualityValidator()
        report = validator.validate(self._make_result())
        assert report.is_valid

    def test_high_chi_squared_warns(self) -> None:
        """Very high chi-squared triggers warning."""
        validator = FitQualityValidator(chi2_warn=5.0)
        result = self._make_result(reduced_chi_squared=8.0)
        report = validator.validate(result)
        assert len(report.warnings) > 0

    def test_extreme_chi_squared_fails(self) -> None:
        """Extreme chi-squared triggers error."""
        validator = FitQualityValidator(chi2_fail=50.0)
        result = self._make_result(reduced_chi_squared=100.0)
        report = validator.validate(result)
        assert not report.is_valid

    def test_param_at_bound_warns(self) -> None:
        """Parameter at bound edge triggers warning."""
        validator = FitQualityValidator()
        result = self._make_result(
            parameters=np.array([100.5, 0.5]),  # D0_ref near lower bound 100
        )
        report = validator.validate(result)
        assert any("bound" in w.message.lower() for w in report.warnings)
