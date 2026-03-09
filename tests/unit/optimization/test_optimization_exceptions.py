"""Tests for optimization exception hierarchy."""

from __future__ import annotations

import pytest

from heterodyne.optimization.exceptions import (
    BoundsError,
    ConvergenceError,
    DegeneracyError,
    NumericalError,
    OptimizationError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Verify inheritance and basic behavior of the exception classes."""

    @pytest.mark.unit
    def test_all_inherit_from_optimization_error(self) -> None:
        """Every leaf exception is an OptimizationError."""
        for exc_cls in (
            ConvergenceError,
            NumericalError,
            BoundsError,
            DegeneracyError,
            ValidationError,
        ):
            exc = exc_cls("test message")
            assert isinstance(exc, OptimizationError)
            assert isinstance(exc, Exception)

    @pytest.mark.unit
    def test_messages_preserved(self) -> None:
        """Exception message is accessible via str()."""
        msg = "optimizer diverged after 100 steps"
        exc = ConvergenceError(msg)
        assert msg in str(exc)

    @pytest.mark.unit
    def test_catch_broad(self) -> None:
        """Catching OptimizationError catches all subtypes."""
        with pytest.raises(OptimizationError):
            raise NumericalError("NaN detected")

    @pytest.mark.unit
    def test_catch_specific(self) -> None:
        """Catching a specific subtype does not catch siblings."""
        with pytest.raises(BoundsError):
            raise BoundsError("D0_ref = -1 outside [0, 1e5]")

        with pytest.raises(OptimizationError):
            raise DegeneracyError("singular Jacobian")

    @pytest.mark.unit
    def test_no_cross_catch(self) -> None:
        """BoundsError does not catch NumericalError."""
        with pytest.raises(NumericalError):
            raise NumericalError("Inf encountered")
