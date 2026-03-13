"""Exception hierarchy for optimization errors.

Provides structured error types for convergence failures, numerical
instabilities, and checkpoint I/O issues during NLSQ fitting and CMC
sampling. The ``NLSQOptimizationError`` family covers NLSQ-specific
failures; the ``OptimizationError`` family covers both NLSQ and CMC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class NLSQOptimizationError(Exception):
    """Base exception for all NLSQ optimization errors.

    Attributes:
        message: Human-readable error description.
        error_context: Structured dict of contextual information for
            diagnostics and recovery decisions.
    """

    def __init__(
        self,
        message: str,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_context: dict[str, Any] = error_context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = self.message
        if self.error_context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.error_context.items())
            return f"{base} [{ctx_str}]"
        return base


class NLSQConvergenceError(NLSQOptimizationError):
    """Raised when NLSQ optimization fails to converge.

    Attributes:
        iteration_count: Number of iterations completed before failure.
        final_loss: Loss value at the point of failure.
        parameters: Parameter values at the point of failure.
    """

    def __init__(
        self,
        message: str,
        iteration_count: int = 0,
        final_loss: float = float("nan"),
        parameters: Any = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.iteration_count = iteration_count
        self.final_loss = final_loss
        self.parameters = parameters
        ctx = error_context or {}
        ctx.setdefault("iteration_count", iteration_count)
        ctx.setdefault("final_loss", final_loss)
        super().__init__(message, error_context=ctx)


class NLSQNumericalError(NLSQOptimizationError):
    """Raised when NaN/Inf values are detected during optimization.

    Attributes:
        detection_point: Where the invalid value was detected
            (e.g. ``"gradient"``, ``"parameter"``, ``"loss"``).
        invalid_values: The problematic values for diagnostics.
    """

    def __init__(
        self,
        message: str,
        detection_point: str = "unknown",
        invalid_values: Any = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.detection_point = detection_point
        self.invalid_values = invalid_values
        ctx = error_context or {}
        ctx.setdefault("detection_point", detection_point)
        super().__init__(message, error_context=ctx)


# ---------------------------------------------------------------------------
# Generic optimization exception hierarchy (covers both NLSQ and CMC)
# ---------------------------------------------------------------------------


class OptimizationError(Exception):
    """Generic base for all optimization errors (NLSQ and CMC).

    Attributes:
        message: Human-readable error description.
        error_context: Structured dict of contextual information for
            diagnostics and recovery decisions.
    """

    def __init__(
        self,
        message: str,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_context: dict[str, Any] = error_context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = self.message
        if self.error_context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.error_context.items())
            return f"{base} [{ctx_str}]"
        return base


class ConvergenceError(OptimizationError):
    """Generic convergence failure (not NLSQ-specific).

    Attributes:
        iteration_count: Number of iterations completed before failure.
        final_loss: Loss value at the point of failure.
    """

    def __init__(
        self,
        message: str,
        iteration_count: int = 0,
        final_loss: float = float("nan"),
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.iteration_count = iteration_count
        self.final_loss = final_loss
        ctx = error_context or {}
        ctx.setdefault("iteration_count", iteration_count)
        ctx.setdefault("final_loss", final_loss)
        super().__init__(message, error_context=ctx)


class NumericalError(OptimizationError):
    """Generic numerical instability (NaN/Inf) detected.

    Attributes:
        detection_point: Where the invalid value was detected
            (e.g. ``"gradient"``, ``"parameter"``, ``"loss"``).
        invalid_values: The problematic values for diagnostics.
    """

    def __init__(
        self,
        message: str,
        detection_point: str = "unknown",
        invalid_values: Any = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.detection_point = detection_point
        self.invalid_values = invalid_values
        ctx = error_context or {}
        ctx.setdefault("detection_point", detection_point)
        super().__init__(message, error_context=ctx)


class BoundsError(OptimizationError):
    """A parameter has hit its optimization bounds.

    Attributes:
        parameter_name: Name of the parameter that hit a bound.
        value: The parameter value at the bound.
        bound: The bound value that was hit.
        bound_type: Either ``"lower"`` or ``"upper"``.
    """

    def __init__(
        self,
        message: str,
        parameter_name: str = "unknown",
        value: float = float("nan"),
        bound: float = float("nan"),
        bound_type: str = "unknown",
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.parameter_name = parameter_name
        self.value = value
        self.bound = bound
        self.bound_type = bound_type
        ctx = error_context or {}
        ctx.setdefault("parameter_name", parameter_name)
        ctx.setdefault("value", value)
        ctx.setdefault("bound", bound)
        ctx.setdefault("bound_type", bound_type)
        super().__init__(message, error_context=ctx)


class DegeneracyError(OptimizationError):
    """Parameter degeneracy detected during optimization.

    Attributes:
        affected_params: Names of the degenerate parameters.
        correlation: Absolute correlation coefficient between the
            affected parameters, or ``None`` if not computed.
    """

    def __init__(
        self,
        message: str,
        affected_params: list[str] | None = None,
        correlation: float | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.affected_params: list[str] = affected_params or []
        self.correlation = correlation
        ctx = error_context or {}
        ctx.setdefault("affected_params", self.affected_params)
        ctx.setdefault("correlation", correlation)
        super().__init__(message, error_context=ctx)


class ValidationError(OptimizationError):
    """Result validation failure.

    Attributes:
        validation_type: Category of the validation that failed
            (e.g. ``"bounds"``, ``"convergence"``, ``"chi_squared"``).
        details: Human-readable description of why validation failed.
    """

    def __init__(
        self,
        message: str,
        validation_type: str = "unknown",
        details: str = "",
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.validation_type = validation_type
        self.details = details
        ctx = error_context or {}
        ctx.setdefault("validation_type", validation_type)
        ctx.setdefault("details", details)
        super().__init__(message, error_context=ctx)


class StreamingError(OptimizationError):
    """Streaming or chunked optimization failure.

    Attributes:
        chunk_index: Zero-based index of the chunk that failed, or
            ``None`` if the failure occurred outside a chunk.
        total_chunks: Total number of chunks in the streaming job, or
            ``None`` if not known.
    """

    def __init__(
        self,
        message: str,
        chunk_index: int | None = None,
        total_chunks: int | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        ctx = error_context or {}
        ctx.setdefault("chunk_index", chunk_index)
        ctx.setdefault("total_chunks", total_chunks)
        super().__init__(message, error_context=ctx)


class ShardingError(OptimizationError):
    """CMC shard failure.

    Attributes:
        shard_index: Zero-based index of the shard that failed, or
            ``None`` if the failure occurred outside a specific shard.
        total_shards: Total number of shards, or ``None`` if not known.
        backend: Name of the CMC backend (e.g. ``"cpu"``, ``"gpu"``),
            or ``None`` if not determined.
    """

    def __init__(
        self,
        message: str,
        shard_index: int | None = None,
        total_shards: int | None = None,
        backend: str | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.shard_index = shard_index
        self.total_shards = total_shards
        self.backend = backend
        ctx = error_context or {}
        ctx.setdefault("shard_index", shard_index)
        ctx.setdefault("total_shards", total_shards)
        ctx.setdefault("backend", backend)
        super().__init__(message, error_context=ctx)


class BackendError(OptimizationError):
    """CMC backend execution failure.

    Attributes:
        backend_name: Name of the backend that failed
            (e.g. ``"cpu"``, ``"gpu"``).
        worker_id: Identifier of the worker that failed, or ``None``
            if the failure occurred outside a specific worker.
    """

    def __init__(
        self,
        message: str,
        backend_name: str = "unknown",
        worker_id: int | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.worker_id = worker_id
        ctx = error_context or {}
        ctx.setdefault("backend_name", backend_name)
        ctx.setdefault("worker_id", worker_id)
        super().__init__(message, error_context=ctx)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (existing code may use these short names).
# The NLSQ-prefixed versions remain the canonical NLSQ exceptions.
# ---------------------------------------------------------------------------

__all__ = [
    # NLSQ-specific (canonical)
    "NLSQOptimizationError",
    "NLSQConvergenceError",
    "NLSQNumericalError",
    # Generic (NLSQ + CMC)
    "OptimizationError",
    "ConvergenceError",
    "NumericalError",
    "BoundsError",
    "DegeneracyError",
    "ValidationError",
    "StreamingError",
    "ShardingError",
    "BackendError",
]
