"""Tests for build_result_from_nlsq in result_builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from heterodyne.optimization.nlsq.result_builder import build_result_from_nlsq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAMES = ["a", "b", "c"]
_POPT = np.array([1.0, 2.0, 3.0])
_PCOV = np.diag([0.01, 0.04, 0.09])
_N_DATA = 50


# ---------------------------------------------------------------------------
# test_build_from_nlsq_tuple
# ---------------------------------------------------------------------------


class TestBuildFromNlsqTuple:
    """(popt, pcov) input -> valid NLSQResult."""

    def test_basic(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, _PCOV), _NAMES, n_data=_N_DATA
        )

        assert result.success is True
        np.testing.assert_array_equal(result.parameters, _POPT)
        assert result.parameter_names == _NAMES
        assert result.covariance is not None
        np.testing.assert_array_equal(result.covariance, _PCOV)
        assert result.uncertainties is not None
        np.testing.assert_allclose(
            result.uncertainties, np.sqrt(np.diag(_PCOV))
        )

    def test_wall_time_propagated(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, _PCOV), _NAMES, n_data=_N_DATA, wall_time=2.5
        )
        assert result.wall_time_seconds == 2.5


# ---------------------------------------------------------------------------
# test_build_from_nlsq_triple
# ---------------------------------------------------------------------------


class TestBuildFromNlsqTriple:
    """(popt, pcov, info) input -> valid NLSQResult with info merged."""

    def test_info_dict_merged(self) -> None:
        info = {"nfev": 42, "message": "converged OK"}
        result = build_result_from_nlsq(
            (_POPT, _PCOV, info), _NAMES, n_data=_N_DATA
        )

        assert result.success is True
        assert result.metadata["nfev"] == 42
        assert result.metadata["message"] == "converged OK"

    def test_non_dict_info_wrapped(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, _PCOV, "some_string"), _NAMES, n_data=_N_DATA
        )

        assert "raw_info" in result.metadata
        assert result.metadata["raw_info"] == "some_string"


# ---------------------------------------------------------------------------
# test_build_from_nlsq_dict
# ---------------------------------------------------------------------------


class TestBuildFromNlsqDict:
    """dict with 'x', 'pcov' keys -> valid NLSQResult."""

    def test_basic(self) -> None:
        d: dict[str, Any] = {
            "x": _POPT,
            "pcov": _PCOV,
            "success": True,
            "message": "streaming done",
            "streaming_diagnostics": {"epochs": 10},
        }
        result = build_result_from_nlsq(d, _NAMES, n_data=_N_DATA)

        assert result.success is True
        np.testing.assert_array_equal(result.parameters, _POPT)
        assert result.covariance is not None
        assert result.metadata["streaming_diagnostics"] == {"epochs": 10}

    def test_popt_key_fallback(self) -> None:
        d: dict[str, Any] = {"popt": _POPT, "pcov": _PCOV}
        result = build_result_from_nlsq(d, _NAMES, n_data=_N_DATA)
        np.testing.assert_array_equal(result.parameters, _POPT)

    def test_missing_pcov_in_dict(self) -> None:
        d: dict[str, Any] = {"x": _POPT}
        result = build_result_from_nlsq(d, _NAMES, n_data=_N_DATA)
        assert result.covariance is None
        assert result.uncertainties is None


# ---------------------------------------------------------------------------
# test_build_from_nlsq_object
# ---------------------------------------------------------------------------


class TestBuildFromNlsqObject:
    """object with .x, .pcov attrs -> valid NLSQResult."""

    def test_basic(self) -> None:
        obj = SimpleNamespace(x=_POPT, pcov=_PCOV, message="ok", nfev=100)
        result = build_result_from_nlsq(obj, _NAMES, n_data=_N_DATA)

        assert result.success is True
        np.testing.assert_array_equal(result.parameters, _POPT)
        assert result.covariance is not None
        assert result.metadata["nfev"] == 100

    def test_popt_attribute_fallback(self) -> None:
        obj = SimpleNamespace(popt=_POPT, pcov=_PCOV)
        result = build_result_from_nlsq(obj, _NAMES, n_data=_N_DATA)
        np.testing.assert_array_equal(result.parameters, _POPT)

    def test_no_pcov_attribute(self) -> None:
        obj = SimpleNamespace(x=_POPT)
        result = build_result_from_nlsq(obj, _NAMES, n_data=_N_DATA)
        assert result.covariance is None
        assert result.uncertainties is None


# ---------------------------------------------------------------------------
# test_build_from_nlsq_none_pcov
# ---------------------------------------------------------------------------


class TestBuildFromNlsqNonePcov:
    """pcov=None -> uncertainties=None."""

    def test_tuple_none_pcov(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, None), _NAMES, n_data=_N_DATA
        )
        assert result.covariance is None
        assert result.uncertainties is None

    def test_triple_none_pcov(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, None, {}), _NAMES, n_data=_N_DATA
        )
        assert result.covariance is None
        assert result.uncertainties is None


# ---------------------------------------------------------------------------
# test_build_from_nlsq_preserves_metadata
# ---------------------------------------------------------------------------


class TestBuildFromNlsqPreservesMetadata:
    """metadata dict propagated."""

    def test_metadata_passed_through(self) -> None:
        meta = {"strategy": "sequential", "attempt": 3}
        result = build_result_from_nlsq(
            (_POPT, _PCOV), _NAMES, n_data=_N_DATA, metadata=meta
        )
        assert result.metadata["strategy"] == "sequential"
        assert result.metadata["attempt"] == 3

    def test_metadata_merged_with_info(self) -> None:
        info = {"nfev": 42}
        meta = {"strategy": "direct"}
        result = build_result_from_nlsq(
            (_POPT, _PCOV, info), _NAMES, n_data=_N_DATA, metadata=meta
        )
        # Both sources present
        assert result.metadata["strategy"] == "direct"
        assert result.metadata["nfev"] == 42

    def test_default_metadata_empty(self) -> None:
        result = build_result_from_nlsq(
            (_POPT, _PCOV), _NAMES, n_data=_N_DATA
        )
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# test_build_from_nlsq_chi_squared
# ---------------------------------------------------------------------------


class TestBuildFromNlsqChiSquared:
    """Reduced chi-squared computed correctly."""

    def test_chi_squared_from_pcov(self) -> None:
        # With pcov provided directly (no residuals), reduced chi² is None
        result = build_result_from_nlsq(
            (_POPT, _PCOV), _NAMES, n_data=_N_DATA
        )
        # Without residuals, no chi-squared can be computed
        assert result.reduced_chi_squared is None

    def test_chi_squared_from_dict_with_residuals(self) -> None:
        residuals = np.array([0.1, -0.1, 0.05, 0.02])
        d: dict[str, Any] = {
            "x": _POPT,
            "pcov": _PCOV,
            "fun": residuals,
        }
        result = build_result_from_nlsq(d, _NAMES, n_data=_N_DATA)
        expected_cost = float(np.sum(residuals**2))
        dof = _N_DATA - len(_NAMES)
        expected_chi2 = expected_cost / dof
        assert result.reduced_chi_squared == pytest.approx(expected_chi2)
        assert result.final_cost == pytest.approx(expected_cost)

    def test_chi_squared_from_object_with_fun(self) -> None:
        residuals = np.array([0.1, -0.2])
        obj = SimpleNamespace(x=_POPT, pcov=_PCOV, fun=residuals)
        result = build_result_from_nlsq(obj, _NAMES, n_data=_N_DATA)
        expected_cost = float(np.sum(residuals**2))
        dof = _N_DATA - len(_NAMES)
        assert result.reduced_chi_squared == pytest.approx(expected_cost / dof)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestBuildFromNlsqErrors:
    """Edge cases and error handling."""

    def test_unrecognized_format_raises(self) -> None:
        with pytest.raises(TypeError, match="Unrecognized"):
            build_result_from_nlsq(42, _NAMES, n_data=_N_DATA)

    def test_bad_tuple_length_raises(self) -> None:
        with pytest.raises(TypeError, match="tuple length"):
            build_result_from_nlsq(
                (_POPT, _PCOV, {}, "extra"), _NAMES, n_data=_N_DATA
            )
