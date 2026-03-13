"""Multi-start optimization with Latin Hypercube Sampling.

Supports parallel execution via ProcessPoolExecutor with automatic
fallback to sequential when JAX functions cannot be serialized for
inter-process communication.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

logger = get_logger(__name__)

_WORKER_TIMEOUT = 1800.0  # 30 minutes default
_MAX_POINTS_FOR_PARALLEL = 500_000


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiStartConfig:
    """Configuration for multi-start optimization.

    Attributes:
        n_starts: Number of starting points (including the user-provided one).
        seed: Random seed for Latin Hypercube Sampling reproducibility.
        parallel: Whether to attempt parallel execution via ProcessPoolExecutor.
            Defaults to False because JAX closures cannot be sent across
            process boundaries.
        max_workers: Maximum number of worker processes.  Defaults to
            ``min(n_starts, os.cpu_count() or 4)``.
        worker_timeout: Per-worker timeout in seconds.
        max_data_points_for_parallel: Auto-disable parallel when
            ``n_data * n_starts`` exceeds this threshold.
    """

    n_starts: int = 10
    seed: int | None = None
    parallel: bool = (
        False  # Default False — JAX closures cannot cross process boundaries
    )
    max_workers: int | None = None
    worker_timeout: float = _WORKER_TIMEOUT
    max_data_points_for_parallel: int = _MAX_POINTS_FOR_PARALLEL

    def __post_init__(self) -> None:
        if self.n_starts < 1:
            raise ValueError(f"n_starts must be >= 1, got {self.n_starts}")
        if self.worker_timeout <= 0:
            raise ValueError(f"worker_timeout must be > 0, got {self.worker_timeout}")
        if self.max_workers is None:
            self.max_workers = min(self.n_starts, os.cpu_count() or 4)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiStartConfig:
        """Create from dictionary, ignoring unknown keys.

        Args:
            d: Dictionary of configuration values.

        Returns:
            Populated ``MultiStartConfig`` instance.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SingleStartResult:
    """Result from a single optimization start.

    Attributes:
        result: The ``NLSQResult`` returned by the adapter.
        start_index: Zero-based index of this start within the run.
        initial_params: Starting parameter vector used for this run.
        wall_time: Wall-clock time in seconds for this single start.
    """

    result: NLSQResult
    start_index: int
    initial_params: np.ndarray
    wall_time: float


@dataclass
class MultiStartResult:
    """Aggregated result from a multi-start optimization run.

    Attributes:
        best_result: The ``NLSQResult`` with the lowest final cost.
        all_starts: Ordered list of ``SingleStartResult`` for every start.
        n_successful: Number of starts that reported ``success=True``.
        n_total: Total number of starts attempted.
        config: The ``MultiStartConfig`` used for this run.
        wall_time_total: Total elapsed wall-clock time in seconds.
    """

    best_result: NLSQResult
    all_starts: list[SingleStartResult]
    n_successful: int
    n_total: int
    config: MultiStartConfig
    wall_time_total: float = 0.0

    @property
    def all_results(self) -> list[NLSQResult]:
        """Backward-compatible accessor returning all ``NLSQResult`` objects."""
        return [s.result for s in self.all_starts]

    def to_nlsq_result(self) -> NLSQResult:
        """Convert the best result to an ``NLSQResult`` with multistart metadata.

        Attaches a ``"multistart"`` key to ``result.metadata`` containing
        summary statistics for the full multi-start run.

        Returns:
            The best ``NLSQResult`` with metadata populated.
        """
        best_index = next(
            (s.start_index for s in self.all_starts if s.result is self.best_result),
            0,
        )
        self.best_result.metadata["multistart"] = {
            "n_starts": self.n_total,
            "n_successful": self.n_successful,
            "wall_time_total": self.wall_time_total,
            "best_start_index": best_index,
        }
        return self.best_result


# ---------------------------------------------------------------------------
# Module-level worker function (required for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _multistart_worker_sequential(
    adapter: NLSQAdapterBase,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    config: NLSQConfig,
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> NLSQResult:
    """Run a single optimization start inside a worker process.

    This function is defined at module level so that ProcessPoolExecutor
    can submit it across process boundaries.  Note: JAX-based residual
    functions cannot be sent across process boundaries, so this function
    is only callable in practice when a serialization-safe residual is
    supplied.  The ``MultiStartOptimizer.fit()`` method always falls back
    to sequential execution for JAX residuals.

    Args:
        adapter: Adapter instance carrying solver configuration.
        residual_fn: Residual callable (must be serialization-safe).
        initial_params: Starting parameter vector.
        bounds: ``(lower, upper)`` bound arrays.
        config: NLSQ solver configuration.
        jacobian_fn: Optional analytic Jacobian (must be serialization-safe).

    Returns:
        ``NLSQResult`` for this start.
    """
    return adapter.fit(
        residual_fn=residual_fn,
        initial_params=initial_params,
        bounds=bounds,
        config=config,
        jacobian_fn=jacobian_fn,
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class MultiStartOptimizer:
    """Multi-start optimizer using Latin Hypercube Sampling.

    Runs optimization from multiple starting points to improve the
    chances of finding the global minimum.  The first starting point is
    always the user-supplied ``initial_params``; the remaining
    ``n_starts - 1`` points are drawn via Latin Hypercube Sampling within
    the supplied bounds.

    Parallel execution is available via ``MultiStartConfig.parallel=True``
    but is disabled by default because JAX residual closures cannot be
    transmitted across process boundaries.  When parallel is requested,
    the optimizer logs a warning and falls back to sequential execution.
    """

    def __init__(
        self,
        adapter: NLSQAdapterBase,
        config: MultiStartConfig | None = None,
        *,
        # Legacy keyword arguments for backward compatibility
        n_starts: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the multi-start optimizer.

        Args:
            adapter: NLSQ adapter used to run each individual fit.
            config: Full ``MultiStartConfig``.  When provided, ``n_starts``
                and ``seed`` keyword arguments are ignored.
            n_starts: Legacy argument — number of starting points.
            seed: Legacy argument — random seed.
        """
        self._adapter = adapter
        if config is not None:
            self._config = config
        else:
            self._config = MultiStartConfig(
                n_starts=n_starts if n_starts is not None else 10,
                seed=seed,
            )
        self._rng = np.random.default_rng(self._config.seed)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def generate_starting_points(
        self,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Generate starting points using Latin Hypercube Sampling.

        The first point is always ``initial_params``.  The remaining
        ``n_starts - 1`` points are drawn via LHS within ``bounds``.

        Args:
            initial_params: User-provided initial values (used as first point).
            bounds: ``(lower, upper)`` bound arrays of shape ``(n_params,)``.

        Returns:
            Array of shape ``(n_starts, n_params)``.
        """
        lower, upper = bounds
        n_params = len(initial_params)

        # First point is user-provided
        starting_points = [initial_params.copy()]

        # Generate LHS points for remaining starts
        n_lhs = self._config.n_starts - 1
        if n_lhs > 0:
            lhs_points = self._latin_hypercube_sample(n_lhs, n_params, lower, upper)
            starting_points.extend(lhs_points)

        return np.array(starting_points)

    # ------------------------------------------------------------------
    # Sampling internals
    # ------------------------------------------------------------------

    def _latin_hypercube_sample(
        self,
        n_samples: int,
        n_dims: int,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate Latin Hypercube samples.

        Divides each dimension into ``n_samples`` equal strata, places one
        sample per stratum with a uniform random offset, then shuffles the
        stratum assignment independently across dimensions.

        Args:
            n_samples: Number of samples to generate.
            n_dims: Dimensionality of the parameter space.
            lower: Lower bounds, shape ``(n_dims,)``.
            upper: Upper bounds, shape ``(n_dims,)``.

        Returns:
            List of ``n_samples`` arrays each of shape ``(n_dims,)``.
        """
        # Proper Latin Hypercube Sampling:
        # Divide each dimension into n_samples equal strata,
        # place one sample per stratum, then shuffle across dimensions.
        result = np.zeros((n_samples, n_dims))
        for d in range(n_dims):
            # Create one sample per stratum with random offset within stratum
            perm = self._rng.permutation(n_samples)
            for i in range(n_samples):
                stratum = perm[i]
                low_edge = lower[d] + stratum * (upper[d] - lower[d]) / n_samples
                high_edge = lower[d] + (stratum + 1) * (upper[d] - lower[d]) / n_samples
                result[i, d] = low_edge + self._rng.random() * (high_edge - low_edge)

        return [result[i] for i in range(n_samples)]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> MultiStartResult:
        """Run multi-start optimization.

        Generates starting points via LHS, then runs the adapter from each
        point sequentially.  If ``MultiStartConfig.parallel=True``, a warning
        is emitted and execution falls back to sequential because JAX residual
        closures cannot be transmitted across process boundaries.

        Args:
            residual_fn: Residual function mapping parameters to residual vector.
            initial_params: Initial guess; always used as the first start.
            bounds: ``(lower, upper)`` bound arrays.
            config: Per-run NLSQ solver configuration.
            jacobian_fn: Optional analytic Jacobian.

        Returns:
            ``MultiStartResult`` with the best result and per-start details.
        """
        t_start = time.perf_counter()
        starting_points = self.generate_starting_points(initial_params, bounds)
        n_points = len(starting_points)

        logger.info(
            "Multi-start: %d starts, parallel=%s",
            n_points,
            self._config.parallel,
        )

        # JAX closures cannot cross process boundaries — always sequential.
        if self._config.parallel:
            logger.warning(
                "Parallel multi-start requested but JAX closures cannot be "
                "transmitted across process boundaries. "
                "Falling back to sequential execution."
            )

        all_starts = self._run_sequential(
            starting_points=starting_points,
            residual_fn=residual_fn,
            bounds=bounds,
            config=config,
            jacobian_fn=jacobian_fn,
        )

        # Identify best result
        best: NLSQResult | None = None
        best_cost = np.inf
        n_successful = 0

        for s in all_starts:
            if s.result.success:
                n_successful += 1
            cost = s.result.final_cost if s.result.final_cost is not None else np.inf
            if cost < best_cost:
                best_cost = cost
                best = s.result

        # Fallback: if nothing succeeded, take the run with the lowest cost
        if best is None:
            best = min(
                (s.result for s in all_starts),
                key=lambda r: r.final_cost if r.final_cost is not None else np.inf,
            )

        wall_total = time.perf_counter() - t_start

        logger.info(
            "Multi-start complete: %d/%d successful, best cost=%.4e, total time=%.1fs",
            n_successful,
            n_points,
            best_cost,
            wall_total,
        )

        return MultiStartResult(
            best_result=best,
            all_starts=all_starts,
            n_successful=n_successful,
            n_total=n_points,
            config=self._config,
            wall_time_total=wall_total,
        )

    # ------------------------------------------------------------------
    # Execution backends
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        starting_points: np.ndarray,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> list[SingleStartResult]:
        """Execute all starts sequentially, logging progress after each.

        Args:
            starting_points: Array of shape ``(n_starts, n_params)``.
            residual_fn: Residual callable.
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ solver configuration.
            jacobian_fn: Optional analytic Jacobian.

        Returns:
            Ordered list of ``SingleStartResult`` for every start.
        """
        all_starts: list[SingleStartResult] = []
        best_cost = np.inf
        n_total = len(starting_points)

        for i, start in enumerate(starting_points):
            logger.debug("Start %d/%d", i + 1, n_total)
            t0 = time.perf_counter()

            result = self._adapter.fit(
                residual_fn=residual_fn,
                initial_params=start,
                bounds=bounds,
                config=config,
                jacobian_fn=jacobian_fn,
            )

            wall = time.perf_counter() - t0
            all_starts.append(
                SingleStartResult(
                    result=result,
                    start_index=i,
                    initial_params=start,
                    wall_time=wall,
                )
            )

            cost = result.final_cost if result.final_cost is not None else np.inf
            if cost < best_cost:
                best_cost = cost
                logger.debug(
                    "  Start %d/%d: new best cost=%.4e (%.2fs)",
                    i + 1,
                    n_total,
                    best_cost,
                    wall,
                )
            else:
                logger.debug(
                    "  Start %d/%d: cost=%.4e (%.2fs)",
                    i + 1,
                    n_total,
                    cost,
                    wall,
                )

        return all_starts

    def _run_parallel(
        self,
        starting_points: np.ndarray,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> list[SingleStartResult]:
        """Execute starts in parallel via ProcessPoolExecutor.

        This method requires that ``residual_fn`` (and ``jacobian_fn``) are
        safe to transmit across process boundaries.  If any worker raises an
        error or times out, the failed starts are logged and excluded from the
        results.  If all workers fail, raises ``RuntimeError``.

        Note: This method is present for completeness and future use.  It is
        not invoked by ``fit()`` for JAX-based residuals.  Use
        ``MultiStartConfig.parallel=True`` only when supplying a residual
        function that can be transmitted across process boundaries.

        Args:
            starting_points: Array of shape ``(n_starts, n_params)``.
            residual_fn: Residual callable (must be transmittable across
                process boundaries).
            bounds: ``(lower, upper)`` bound arrays.
            config: NLSQ solver configuration.
            jacobian_fn: Optional analytic Jacobian (must be transmittable
                across process boundaries).

        Returns:
            Ordered list of ``SingleStartResult`` for every completed start.

        Raises:
            RuntimeError: If every worker fails or times out.
        """
        n_total = len(starting_points)
        max_workers = self._config.max_workers or min(n_total, os.cpu_count() or 4)
        timeout = self._config.worker_timeout

        # Map from future -> (start_index, initial_params, t0)
        future_meta: dict[Any, tuple[int, np.ndarray, float]] = {}
        results_map: dict[int, SingleStartResult] = {}

        logger.info(
            "Parallel multi-start: %d starts, %d workers, timeout=%.0fs",
            n_total,
            max_workers,
            timeout,
        )

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, start in enumerate(starting_points):
                    t0 = time.perf_counter()
                    future = executor.submit(
                        _multistart_worker_sequential,
                        self._adapter,
                        residual_fn,
                        start,
                        bounds,
                        config,
                        jacobian_fn,
                    )
                    future_meta[future] = (i, start, t0)

                best_cost = np.inf
                for future in as_completed(future_meta, timeout=timeout):
                    idx, start, t0 = future_meta[future]
                    wall = time.perf_counter() - t0
                    try:
                        result: NLSQResult = future.result(timeout=0)
                        cost = (
                            result.final_cost
                            if result.final_cost is not None
                            else np.inf
                        )
                        if cost < best_cost:
                            best_cost = cost
                            logger.debug(
                                "  Worker %d/%d done: new best cost=%.4e (%.2fs)",
                                idx + 1,
                                n_total,
                                best_cost,
                                wall,
                            )
                        else:
                            logger.debug(
                                "  Worker %d/%d done: cost=%.4e (%.2fs)",
                                idx + 1,
                                n_total,
                                cost,
                                wall,
                            )
                        results_map[idx] = SingleStartResult(
                            result=result,
                            start_index=idx,
                            initial_params=start,
                            wall_time=wall,
                        )
                    except FuturesTimeoutError:
                        logger.warning(
                            "Worker %d/%d timed out after %.0fs — skipping",
                            idx + 1,
                            n_total,
                            timeout,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Worker %d/%d failed: %s — skipping",
                            idx + 1,
                            n_total,
                            exc,
                        )

        except FuturesTimeoutError:
            logger.warning(
                "Global as_completed timeout reached after %.0fs — "
                "proceeding with %d completed workers",
                timeout,
                len(results_map),
            )

        if not results_map:
            raise RuntimeError(
                "All parallel workers failed or timed out. "
                "Consider using sequential execution."
            )

        # Return in original start order
        return [results_map[i] for i in sorted(results_map)]


# ---------------------------------------------------------------------------
# Standalone LHS utilities
# ---------------------------------------------------------------------------


def check_zero_volume_bounds(
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
) -> list[int]:
    """Identify parameter dimensions with zero sampling volume.

    A dimension has zero volume when its lower bound equals its upper bound,
    meaning the parameter is effectively fixed.  LHS sampling should skip
    these dimensions and keep all starts at the fixed value.

    Args:
        bounds_lower: Lower bound array of shape ``(n_params,)``.
        bounds_upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        Sorted list of dimension indices where
        ``bounds_lower[i] == bounds_upper[i]``.

    Raises:
        ValueError: If arrays have different lengths.
    """
    lower = np.asarray(bounds_lower, dtype=np.float64)
    upper = np.asarray(bounds_upper, dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError(
            f"bounds_lower and bounds_upper must have the same shape, "
            f"got {lower.shape} vs {upper.shape}"
        )
    fixed_dims = [int(i) for i in np.where(lower == upper)[0]]
    if fixed_dims:
        logger.debug(
            "check_zero_volume_bounds: %d fixed dimension(s) detected: %s",
            len(fixed_dims),
            fixed_dims,
        )
    return fixed_dims


def generate_lhs_starts(
    n_starts: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Generate Latin Hypercube starting points, excluding fixed dimensions.

    Fixed dimensions (where ``bounds_lower[i] == bounds_upper[i]``) are
    detected via :func:`check_zero_volume_bounds` and excluded from LHS
    sampling.  All starts receive the fixed value for those dimensions.

    Args:
        n_starts: Number of starting points to generate.
        bounds_lower: Lower bound array of shape ``(n_params,)``.
        bounds_upper: Upper bound array of shape ``(n_params,)``.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(n_starts, n_params)`` containing starting points.

    Raises:
        ValueError: If ``n_starts < 1`` or bound arrays have mismatched shapes.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")

    lower = np.asarray(bounds_lower, dtype=np.float64)
    upper = np.asarray(bounds_upper, dtype=np.float64)
    n_params = len(lower)

    fixed_dims = check_zero_volume_bounds(lower, upper)
    fixed_set = set(fixed_dims)
    free_dims = [i for i in range(n_params) if i not in fixed_set]

    # Initialise output; fixed dimensions get their constant value immediately
    starts = np.empty((n_starts, n_params), dtype=np.float64)
    for i in fixed_dims:
        starts[:, i] = lower[i]

    if not free_dims:
        logger.warning(
            "generate_lhs_starts: all %d dimensions are fixed — "
            "all starts are identical",
            n_params,
        )
        return starts

    # LHS over free dimensions only
    rng = np.random.default_rng(seed)
    n_free = len(free_dims)
    lhs_block = np.zeros((n_starts, n_free), dtype=np.float64)

    for col_idx, dim in enumerate(free_dims):
        lo = lower[dim]
        hi = upper[dim]
        perm = rng.permutation(n_starts).astype(np.float64)
        u = rng.random(n_starts)
        lhs_block[:, col_idx] = lo + (perm + u) * (hi - lo) / n_starts

    free_dims_arr = np.array(free_dims)
    starts[:, free_dims_arr] = lhs_block

    logger.debug(
        "generate_lhs_starts: generated %d starts across %d free / %d fixed dims "
        "(seed=%d)",
        n_starts,
        n_free,
        len(fixed_dims),
        seed,
    )
    return starts
