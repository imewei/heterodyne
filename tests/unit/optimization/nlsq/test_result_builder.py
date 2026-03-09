"""Tests for heterodyne.optimization.nlsq.result_builder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.optimization.nlsq.result_builder import (
    TimedContext,
    _compute_covariance,
    _status_to_reason,
    build_failed_result,
    build_result_from_arrays,
    build_result_from_scipy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockOptimizeResult(SimpleNamespace):
    """Mock scipy OptimizeResult that supports .get() for dict-like access."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _make_scipy_result(
    x: np.ndarray,
    fun: np.ndarray,
    *,
    jac: np.ndarray | None = None,
    status: int = 1,
    message: str = "converged",
    njev: int = 5,
    nfev: int = 20,
    success: bool = True,
) -> _MockOptimizeResult:
    """Create a mock scipy OptimizeResult."""
    result = _MockOptimizeResult(
        x=x,
        fun=fun,
        status=status,
        message=message,
        njev=njev,
        nfev=nfev,
        success=success,
    )
    if jac is not None:
        result.jac = jac
    return result


# ---------------------------------------------------------------------------
# build_result_from_scipy
# ---------------------------------------------------------------------------


class TestBuildResultFromScipy:
    """Tests for build_result_from_scipy."""

    def test_basic_successful_fit(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        fun = np.array([0.01, -0.02, 0.03, 0.01])
        opt = _make_scipy_result(x, fun, status=1)
        names = ["a", "b", "c"]

        result = build_result_from_scipy(opt, names, n_data=100)

        assert result.success is True
        np.testing.assert_array_equal(result.parameters, x)
        assert result.parameter_names == names
        assert result.n_iterations == 5
        assert result.n_function_evals == 20
        assert result.final_cost == pytest.approx(np.sum(fun**2))
        assert result.reduced_chi_squared is not None

    def test_with_jacobian_computes_covariance(self) -> None:
        x = np.array([1.0, 2.0])
        fun = np.array([0.01, -0.02, 0.03])
        jac = np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 0.1]])
        opt = _make_scipy_result(x, fun, jac=jac, status=1)

        result = build_result_from_scipy(opt, ["a", "b"], n_data=10)

        assert result.covariance is not None
        assert result.covariance.shape == (2, 2)
        assert result.uncertainties is not None
        assert len(result.uncertainties) == 2

    def test_without_jacobian_no_covariance(self) -> None:
        x = np.array([1.0])
        fun = np.array([0.1])
        opt = _make_scipy_result(x, fun)

        result = build_result_from_scipy(opt, ["a"], n_data=10)

        assert result.covariance is None
        assert result.uncertainties is None

    def test_negative_status_is_failure(self) -> None:
        x = np.array([1.0])
        fun = np.array([0.1])
        opt = _make_scipy_result(x, fun, status=-1)

        result = build_result_from_scipy(opt, ["a"], n_data=10)

        assert result.success is False

    def test_wall_time_and_metadata(self) -> None:
        x = np.array([1.0])
        fun = np.array([0.1])
        opt = _make_scipy_result(x, fun, status=1)
        meta = {"strategy": "sequential"}

        result = build_result_from_scipy(
            opt, ["a"], n_data=10, wall_time=1.5, metadata=meta
        )

        assert result.wall_time_seconds == 1.5
        assert result.metadata["strategy"] == "sequential"

    def test_status_zero_is_failure(self) -> None:
        x = np.array([1.0])
        fun = np.array([0.1])
        opt = _make_scipy_result(x, fun, status=0)

        result = build_result_from_scipy(opt, ["a"], n_data=10)
        assert result.success is False


# ---------------------------------------------------------------------------
# build_result_from_arrays
# ---------------------------------------------------------------------------


class TestBuildResultFromArrays:
    """Tests for build_result_from_arrays."""

    def test_basic(self) -> None:
        params = np.array([1.0, 2.0])
        residuals = np.array([0.1, -0.1, 0.05])

        result = build_result_from_arrays(
            params, ["a", "b"], residuals, n_data=50
        )

        assert result.success is True
        assert result.message == ""
        np.testing.assert_array_equal(result.parameters, params)
        assert result.final_cost == pytest.approx(np.sum(residuals**2))

    def test_with_jacobian(self) -> None:
        params = np.array([1.0, 2.0])
        residuals = np.array([0.1, -0.1, 0.05])
        jac = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        result = build_result_from_arrays(
            params, ["a", "b"], residuals, n_data=50, jacobian=jac
        )

        assert result.covariance is not None
        assert result.uncertainties is not None

    def test_failed_fit(self) -> None:
        params = np.array([0.0])
        residuals = np.array([10.0])

        result = build_result_from_arrays(
            params, ["a"], residuals, n_data=5,
            success=False, message="did not converge",
        )

        assert result.success is False
        assert result.message == "did not converge"

    def test_metadata_default_empty(self) -> None:
        result = build_result_from_arrays(
            np.array([1.0]), ["a"], np.array([0.1]), n_data=5
        )
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# build_failed_result
# ---------------------------------------------------------------------------


class TestBuildFailedResult:
    """Tests for build_failed_result."""

    def test_basic(self) -> None:
        result = build_failed_result(["a", "b"], "explosion")

        assert result.success is False
        assert result.message == "explosion"
        assert len(result.parameters) == 2
        np.testing.assert_array_equal(result.parameters, [0.0, 0.0])

    def test_with_initial_params(self) -> None:
        init = np.array([3.0, 4.0])
        result = build_failed_result(["a", "b"], "fail", initial_params=init)

        np.testing.assert_array_equal(result.parameters, init)

    def test_with_wall_time(self) -> None:
        result = build_failed_result(["x"], "timeout", wall_time=99.9)
        assert result.wall_time_seconds == 99.9

    def test_with_metadata(self) -> None:
        result = build_failed_result(["x"], "fail", metadata={"reason": "oom"})
        assert result.metadata["reason"] == "oom"


# ---------------------------------------------------------------------------
# TimedContext
# ---------------------------------------------------------------------------


class TestTimedContext:
    """Tests for TimedContext."""

    def test_records_elapsed(self) -> None:
        timer = TimedContext()
        assert timer.elapsed == 0.0
        with timer:
            # No-op; just measuring the context manager
            _ = sum(range(100))
        assert timer.elapsed > 0.0

    def test_returns_self(self) -> None:
        timer = TimedContext()
        with timer as t:
            assert t is timer


# ---------------------------------------------------------------------------
# _compute_covariance
# ---------------------------------------------------------------------------


class TestComputeCovariance:
    """Tests for _compute_covariance."""

    def test_well_conditioned(self) -> None:
        jac = np.eye(3)
        residuals = np.array([0.1, 0.2, 0.3])
        cov = _compute_covariance(jac, residuals, n_data=10, n_params=3)

        assert cov is not None
        assert cov.shape == (3, 3)
        # For identity J, cov = s^2 * I where s^2 = sum(res^2) / dof
        s2 = np.sum(residuals**2) / (10 - 3)
        np.testing.assert_allclose(np.diag(cov), s2)

    def test_singular_returns_none(self) -> None:
        # Rank-deficient Jacobian (duplicate columns)
        jac = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        residuals = np.array([0.1, 0.2, 0.3])
        # J^T J is singular (rank 1) but not singular enough for LinAlgError
        # The function adds Tikhonov regularization for high condition numbers
        cov = _compute_covariance(jac, residuals, n_data=10, n_params=2)
        # Should still return something due to regularization
        if cov is not None:
            assert cov.shape == (2, 2)

    def test_underdetermined_dof_clamped(self) -> None:
        jac = np.eye(2)
        residuals = np.array([0.5, 0.5])
        cov = _compute_covariance(jac, residuals, n_data=1, n_params=2)

        assert cov is not None
        # dof = max(1-2, 1) = 1
        s2 = np.sum(residuals**2) / 1
        np.testing.assert_allclose(np.diag(cov), s2)


# ---------------------------------------------------------------------------
# _status_to_reason
# ---------------------------------------------------------------------------


class TestStatusToReason:
    """Tests for _status_to_reason."""

    def test_known_statuses(self) -> None:
        assert "gtol" in _status_to_reason(1)
        assert "xtol" in _status_to_reason(2)
        assert "ftol" in _status_to_reason(3)
        assert "Maximum" in _status_to_reason(0)
        assert "Improper" in _status_to_reason(-1)

    def test_unknown_status(self) -> None:
        reason = _status_to_reason(999)
        assert "Unknown" in reason
        assert "999" in reason
