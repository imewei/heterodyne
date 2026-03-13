"""Numerical validation utilities for NLSQ optimization.

Provides runtime checks for gradients, parameters, and loss values
to detect NaN/Inf before they propagate through the computation graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.exceptions import (
    BoundsError,
    NLSQNumericalError,
    NumericalError,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    import jax.numpy as jnp

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Module-level validation functions (numpy-only, no JAX dependency)
# ------------------------------------------------------------------


def validate_array(
    data: Any,
    *,
    name: str = "array",
) -> np.ndarray:
    """Validate an array for NaN/Inf values.

    Args:
        data: Array-like input.
        name: Descriptive name for error messages.

    Returns:
        Validated float64 numpy array.

    Raises:
        NumericalError: If any element is NaN or Inf.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return arr
    if np.any(np.isnan(arr)):
        raise NumericalError(f"NaN detected in {name}")
    if np.any(np.isinf(arr)):
        raise NumericalError(f"Inf detected in {name}")
    return arr


def validate_parameters(
    values: Any,
    *,
    names: list[str],
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Validate parameter values for NaN/Inf and optional bounds.

    Args:
        values: Parameter values (array-like).
        names: Parameter names (must match length of values).
        bounds: Optional per-parameter (lower, upper) bounds.

    Returns:
        Validated float64 numpy array.

    Raises:
        ValueError: If lengths don't match.
        NumericalError: If any value is NaN/Inf.
        BoundsError: If any value violates its bounds.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) != len(names):
        raise ValueError(
            f"Parameter values length ({len(arr)}) != names length ({len(names)})"
        )
    if bounds is not None and len(bounds) != len(arr):
        raise ValueError(f"bounds length ({len(bounds)}) != values length ({len(arr)})")

    for i, name in enumerate(names):
        v = arr[i]
        if np.isnan(v):
            raise NumericalError(f"NaN detected in parameter {name}")
        if np.isinf(v):
            raise NumericalError(f"Inf detected in parameter {name}")
        if bounds is not None:
            lo, hi = bounds[i]
            if v < lo or v > hi:
                raise BoundsError(
                    f"Parameter {name} = {v} violates bounds [{lo}, {hi}]"
                )

    return arr


def safe_compute(
    fn: Callable[..., np.ndarray],
    *args: Any,
    fallback: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Call *fn* and validate the result for NaN/Inf.

    Args:
        fn: Callable returning an ndarray.
        *args: Positional arguments forwarded to *fn*.
        fallback: If provided and result has NaN/Inf, return this instead
            of raising.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The result of *fn* if finite, or *fallback* if provided.

    Raises:
        NumericalError: If result has NaN/Inf and no fallback is given.
    """
    result = fn(*args, **kwargs)
    arr = np.asarray(result)
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        if fallback is not None:
            return fallback
        raise NumericalError("NaN/Inf detected in computation result")
    return result


class NumericalValidator:
    """Runtime numerical validation for optimization tensors.

    Checks arrays for NaN/Inf values and optional bound violations
    at critical points in the optimization loop.

    Args:
        enable_validation: Whether validation checks are active.
            Can be toggled at runtime via :meth:`enable`/:meth:`disable`.
        bounds: Optional dict mapping parameter names to ``(lower, upper)``
            tuples for bound-violation checks.
    """

    def __init__(
        self,
        enable_validation: bool = True,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._enabled = enable_validation
        self._bounds: dict[str, tuple[float, float]] = bounds or {}

    # ------------------------------------------------------------------
    # Public validation methods
    # ------------------------------------------------------------------

    def validate_gradients(self, gradients: jnp.ndarray | np.ndarray) -> None:
        """Check that all gradient values are finite.

        Args:
            gradients: Gradient array to validate.

        Raises:
            NLSQNumericalError: If any gradient element is NaN or Inf.
        """
        if not self._enabled:
            return

        import jax.numpy as jnp

        if not jnp.all(jnp.isfinite(gradients)):
            non_finite_mask = ~jnp.isfinite(gradients)
            invalid_count = int(jnp.sum(non_finite_mask))
            logger.error(
                "Non-finite gradients detected: %d / %d elements",
                invalid_count,
                gradients.size,
            )
            raise NLSQNumericalError(
                f"Non-finite gradients detected: {invalid_count} / {gradients.size} elements",
                detection_point="gradient",
                invalid_values=gradients,
            )

    def validate_parameters(
        self,
        parameters: jnp.ndarray | np.ndarray | dict[str, Any],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Check parameters for finite values and optional bound violations.

        Args:
            parameters: Parameter array or dict to validate.
            bounds: Optional per-call bounds override. Falls back to
                instance bounds if not provided.

        Raises:
            NLSQNumericalError: If any parameter is NaN/Inf or violates bounds.
        """
        if not self._enabled:
            return

        import jax.numpy as jnp

        active_bounds = bounds if bounds is not None else self._bounds

        # Dict path: validate each named parameter
        if isinstance(parameters, dict):
            for name, value in parameters.items():
                arr = jnp.asarray(value)
                if not jnp.all(jnp.isfinite(arr)):
                    logger.error("Non-finite parameter detected: %s", name)
                    raise NLSQNumericalError(
                        f"Non-finite parameter detected: {name}",
                        detection_point="parameter",
                        invalid_values={name: value},
                    )
                if name in active_bounds:
                    lo, hi = active_bounds[name]
                    if jnp.any(arr < lo) or jnp.any(arr > hi):
                        logger.error(
                            "Parameter %s violates bounds [%g, %g]", name, lo, hi
                        )
                        raise NLSQNumericalError(
                            f"Parameter {name} violates bounds [{lo}, {hi}]",
                            detection_point="parameter_bounds",
                            invalid_values={name: value, "bounds": (lo, hi)},
                        )
            return

        # Array path
        arr = jnp.asarray(parameters)
        if not jnp.all(jnp.isfinite(arr)):
            non_finite_mask = ~jnp.isfinite(arr)
            invalid_count = int(jnp.sum(non_finite_mask))
            logger.error(
                "Non-finite parameters detected: %d / %d elements",
                invalid_count,
                arr.size,
            )
            raise NLSQNumericalError(
                f"Non-finite parameters detected: {invalid_count} / {arr.size} elements",
                detection_point="parameter",
                invalid_values=parameters,
            )

    def validate_loss(self, loss_value: jnp.ndarray | float) -> None:
        """Check that the loss value is a finite scalar.

        Args:
            loss_value: Scalar loss value to validate.

        Raises:
            NLSQNumericalError: If loss is NaN, Inf, or non-scalar.
        """
        if not self._enabled:
            return

        import jax.numpy as jnp

        loss_arr = jnp.asarray(loss_value)
        if loss_arr.ndim != 0:
            raise NLSQNumericalError(
                f"Loss must be scalar, got shape {loss_arr.shape}",
                detection_point="loss",
                invalid_values=loss_value,
            )
        if not jnp.isfinite(loss_arr):
            logger.error("Non-finite loss detected: %s", loss_value)
            raise NLSQNumericalError(
                f"Non-finite loss detected: {loss_value}",
                detection_point="loss",
                invalid_values=loss_value,
            )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        """Replace the current bounds dictionary.

        Args:
            bounds: New parameter bounds mapping.
        """
        self._bounds = dict(bounds)

    def disable(self) -> None:
        """Disable all validation checks (zero overhead)."""
        self._enabled = False

    def enable(self) -> None:
        """Re-enable validation checks."""
        self._enabled = True
