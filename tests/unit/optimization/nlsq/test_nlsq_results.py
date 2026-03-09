"""Tests for NLSQ results module.

Tests NLSQResult class methods and properties.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from heterodyne.optimization.nlsq.results import NLSQResult

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_result() -> NLSQResult:
    """Create simple successful result."""
    return NLSQResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        parameter_names=["a", "b", "c"],
        success=True,
        message="Converged",
    )


@pytest.fixture
def full_result() -> NLSQResult:
    """Create result with all optional fields."""
    params = np.array([1.0, 2.0, 3.0])
    uncertainties = np.array([0.1, 0.2, 0.3])
    covariance = np.array([
        [0.01, 0.005, 0.002],
        [0.005, 0.04, 0.01],
        [0.002, 0.01, 0.09],
    ])

    return NLSQResult(
        parameters=params,
        parameter_names=["a", "b", "c"],
        success=True,
        message="Converged",
        uncertainties=uncertainties,
        covariance=covariance,
        final_cost=0.123,
        reduced_chi_squared=1.05,
        n_iterations=50,
        n_function_evals=200,
        convergence_reason="ftol reached",
        residuals=np.array([0.01, -0.02, 0.01, -0.01]),
        wall_time_seconds=1.5,
        metadata={"method": "nlsq"},
    )


@pytest.fixture
def failed_result() -> NLSQResult:
    """Create failed result."""
    return NLSQResult(
        parameters=np.array([0.0, 0.0, 0.0]),
        parameter_names=["a", "b", "c"],
        success=False,
        message="Did not converge",
        n_iterations=100,
    )


# ============================================================================
# Property Tests
# ============================================================================


class TestNLSQResultProperties:
    """Tests for NLSQResult properties."""

    @pytest.mark.unit
    def test_n_params(self, simple_result: NLSQResult) -> None:
        """n_params returns correct count."""
        assert simple_result.n_params == 3

    @pytest.mark.unit
    def test_params_dict(self, simple_result: NLSQResult) -> None:
        """params_dict returns correct dictionary."""
        expected = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert simple_result.params_dict == expected

    @pytest.mark.unit
    def test_params_dict_values_are_float(self, simple_result: NLSQResult) -> None:
        """params_dict values are Python floats, not numpy."""
        for val in simple_result.params_dict.values():
            assert isinstance(val, float)


# ============================================================================
# get_param Tests
# ============================================================================


class TestGetParam:
    """Tests for get_param method."""

    @pytest.mark.unit
    def test_get_existing_param(self, simple_result: NLSQResult) -> None:
        """Get value for existing parameter."""
        assert simple_result.get_param("a") == 1.0
        assert simple_result.get_param("b") == 2.0
        assert simple_result.get_param("c") == 3.0

    @pytest.mark.unit
    def test_get_nonexistent_param_raises(self, simple_result: NLSQResult) -> None:
        """Raises KeyError for nonexistent parameter."""
        with pytest.raises(KeyError) as excinfo:
            simple_result.get_param("nonexistent")
        assert "nonexistent" in str(excinfo.value)

    @pytest.mark.unit
    def test_get_param_returns_float(self, simple_result: NLSQResult) -> None:
        """get_param returns Python float."""
        result = simple_result.get_param("a")
        assert isinstance(result, float)


# ============================================================================
# get_uncertainty Tests
# ============================================================================


class TestGetUncertainty:
    """Tests for get_uncertainty method."""

    @pytest.mark.unit
    def test_get_existing_uncertainty(self, full_result: NLSQResult) -> None:
        """Get uncertainty for existing parameter."""
        assert full_result.get_uncertainty("a") == pytest.approx(0.1)
        assert full_result.get_uncertainty("b") == pytest.approx(0.2)
        assert full_result.get_uncertainty("c") == pytest.approx(0.3)

    @pytest.mark.unit
    def test_get_uncertainty_no_uncertainties_returns_none(
        self, simple_result: NLSQResult
    ) -> None:
        """Returns None when uncertainties not available."""
        assert simple_result.get_uncertainty("a") is None

    @pytest.mark.unit
    def test_get_uncertainty_nonexistent_param_returns_none(
        self, full_result: NLSQResult
    ) -> None:
        """Returns None for nonexistent parameter."""
        assert full_result.get_uncertainty("nonexistent") is None

    @pytest.mark.unit
    def test_get_uncertainty_returns_float(self, full_result: NLSQResult) -> None:
        """get_uncertainty returns Python float."""
        result = full_result.get_uncertainty("a")
        assert isinstance(result, float)


# ============================================================================
# get_correlation_matrix Tests
# ============================================================================


class TestGetCorrelationMatrix:
    """Tests for get_correlation_matrix method."""

    @pytest.mark.unit
    def test_correlation_from_covariance(self, full_result: NLSQResult) -> None:
        """Computes correlation matrix from covariance."""
        corr = full_result.get_correlation_matrix()
        assert corr is not None
        assert corr.shape == (3, 3)

        # Diagonal should be 1
        assert_allclose(np.diag(corr), [1.0, 1.0, 1.0], rtol=1e-10)

        # Off-diagonal should be in [-1, 1]
        assert np.all(np.abs(corr) <= 1.0)

    @pytest.mark.unit
    def test_correlation_no_covariance_returns_none(
        self, simple_result: NLSQResult
    ) -> None:
        """Returns None when covariance not available."""
        assert simple_result.get_correlation_matrix() is None

    @pytest.mark.unit
    def test_correlation_is_symmetric(self, full_result: NLSQResult) -> None:
        """Correlation matrix is symmetric."""
        corr = full_result.get_correlation_matrix()
        assert corr is not None
        assert_allclose(corr, corr.T)

    @pytest.mark.unit
    def test_correlation_handles_zero_variance(self) -> None:
        """Handles zero variance (constant parameter) gracefully."""
        result = NLSQResult(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            success=True,
            message="OK",
            covariance=np.array([[0.01, 0.0], [0.0, 0.0]]),  # Zero variance for b
        )
        corr = result.get_correlation_matrix()
        assert corr is not None
        # Should handle without NaN
        assert not np.any(np.isnan(corr))


# ============================================================================
# validate Tests
# ============================================================================


class TestValidate:
    """Tests for validate method."""

    @pytest.mark.unit
    def test_successful_fit_no_warnings(self, simple_result: NLSQResult) -> None:
        """Successful fit with good metrics has no warnings."""
        result = NLSQResult(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            success=True,
            message="Converged",
            reduced_chi_squared=1.0,
        )
        warnings = result.validate()
        assert len(warnings) == 0

    @pytest.mark.unit
    def test_failed_fit_warning(self, failed_result: NLSQResult) -> None:
        """Failed fit generates warning."""
        warnings = failed_result.validate()
        assert any("failed" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_high_chi_squared_warning(self) -> None:
        """High reduced chi-squared generates warning."""
        result = NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["a"],
            success=True,
            message="OK",
            reduced_chi_squared=5.0,  # > 2.0
        )
        warnings = result.validate()
        assert any("χ²" in w and "> 2" in w for w in warnings)

    @pytest.mark.unit
    def test_low_chi_squared_overfit_warning(self) -> None:
        """Very low reduced chi-squared generates overfit warning."""
        result = NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["a"],
            success=True,
            message="OK",
            reduced_chi_squared=0.1,  # < 0.5
        )
        warnings = result.validate()
        assert any("overfit" in w.lower() for w in warnings)

    @pytest.mark.unit
    def test_large_uncertainty_warning(self) -> None:
        """Large relative uncertainty generates warning."""
        result = NLSQResult(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            success=True,
            message="OK",
            uncertainties=np.array([2.0, 0.1]),  # a has 200% uncertainty
        )
        warnings = result.validate()
        assert any("Large uncertainty" in w and "a" in w for w in warnings)

    @pytest.mark.unit
    def test_high_correlation_warning(self) -> None:
        """Highly correlated parameters generate warning."""
        result = NLSQResult(
            parameters=np.array([1.0, 2.0]),
            parameter_names=["a", "b"],
            success=True,
            message="OK",
            covariance=np.array([[1.0, 0.99], [0.99, 1.0]]),  # r = 0.99
        )
        warnings = result.validate()
        assert any("correlated" in w.lower() for w in warnings)


# ============================================================================
# summary Tests
# ============================================================================


class TestSummary:
    """Tests for summary method."""

    @pytest.mark.unit
    def test_summary_returns_string(self, simple_result: NLSQResult) -> None:
        """summary() returns a string."""
        summary = simple_result.summary()
        assert isinstance(summary, str)

    @pytest.mark.unit
    def test_summary_contains_title(self, simple_result: NLSQResult) -> None:
        """Summary contains title."""
        summary = simple_result.summary()
        assert "NLSQ Fit Result" in summary

    @pytest.mark.unit
    def test_summary_contains_success_status(self, simple_result: NLSQResult) -> None:
        """Summary contains success status."""
        summary = simple_result.summary()
        assert "Success: True" in summary

    @pytest.mark.unit
    def test_summary_contains_message(self, simple_result: NLSQResult) -> None:
        """Summary contains message."""
        summary = simple_result.summary()
        assert "Converged" in summary

    @pytest.mark.unit
    def test_summary_contains_parameter_names(self, simple_result: NLSQResult) -> None:
        """Summary contains parameter names."""
        summary = simple_result.summary()
        assert "a" in summary
        assert "b" in summary
        assert "c" in summary

    @pytest.mark.unit
    def test_summary_contains_parameter_values(self, simple_result: NLSQResult) -> None:
        """Summary contains parameter values."""
        summary = simple_result.summary()
        # Parameters are formatted in scientific notation
        assert "1.0000e+00" in summary or "1.00" in summary

    @pytest.mark.unit
    def test_summary_with_uncertainties(self, full_result: NLSQResult) -> None:
        """Summary shows uncertainties when available."""
        summary = full_result.summary()
        assert "±" in summary

    @pytest.mark.unit
    def test_summary_contains_statistics(self, full_result: NLSQResult) -> None:
        """Summary contains statistics section."""
        summary = full_result.summary()
        assert "Statistics" in summary
        assert "Final cost" in summary
        assert "Reduced χ²" in summary
        assert "Iterations" in summary

    @pytest.mark.unit
    def test_summary_contains_wall_time(self, full_result: NLSQResult) -> None:
        """Summary contains wall time when available."""
        summary = full_result.summary()
        assert "Wall time" in summary

    @pytest.mark.unit
    def test_summary_failed_result(self, failed_result: NLSQResult) -> None:
        """Summary handles failed result."""
        summary = failed_result.summary()
        assert "Success: False" in summary
        assert "Did not converge" in summary


# ============================================================================
# Edge Cases and Integration
# ============================================================================


class TestNLSQResultEdgeCases:
    """Edge case tests for NLSQResult."""

    @pytest.mark.unit
    def test_single_parameter(self) -> None:
        """Single parameter result works correctly."""
        result = NLSQResult(
            parameters=np.array([42.0]),
            parameter_names=["single"],
            success=True,
            message="OK",
        )
        assert result.n_params == 1
        assert result.get_param("single") == 42.0
        assert "single" in result.summary()

    @pytest.mark.unit
    def test_many_parameters(self) -> None:
        """Many parameters work correctly."""
        n = 14
        params = np.arange(n, dtype=float)
        names = [f"p{i}" for i in range(n)]

        result = NLSQResult(
            parameters=params,
            parameter_names=names,
            success=True,
            message="OK",
        )
        assert result.n_params == n
        for i in range(n):
            assert result.get_param(f"p{i}") == float(i)

    @pytest.mark.unit
    def test_zero_values(self) -> None:
        """Zero parameter values handled correctly."""
        result = NLSQResult(
            parameters=np.array([0.0, 0.0]),
            parameter_names=["a", "b"],
            success=True,
            message="OK",
            uncertainties=np.array([0.1, 0.1]),
        )
        # Validate should not warn about large relative uncertainty for zero values
        warnings = result.validate()
        # Zero in numerator means ratio is 0, not infinity
        assert not any("Large uncertainty" in w for w in warnings)

    @pytest.mark.unit
    def test_metadata_preserved(self) -> None:
        """Metadata is preserved."""
        metadata = {"custom_key": "custom_value", "number": 42}
        result = NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["a"],
            success=True,
            message="OK",
            metadata=metadata,
        )
        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["number"] == 42
