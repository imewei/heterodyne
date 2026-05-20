"""Multi-start optimization with Latin Hypercube Sampling.

Supports parallel execution via ProcessPoolExecutor with automatic
fallback to sequential when JAX functions cannot be serialized for
inter-process communication.

NOTE: Subsampling is explicitly NOT supported per project requirements.
Numerical precision and reproducibility take priority over computational speed.
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

    from numpy.typing import NDArray

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
        enable: Whether to use multi-start optimization. Default: False.
        n_starts: Number of starting points (including the user-provided one).
        seed: Random seed for Latin Hypercube Sampling reproducibility.
        sampling_strategy: Method for generating starting points.
            ``"latin_hypercube"`` or ``"random"``.
        custom_starts: User-provided custom starting points to include
            alongside generated starts.
        n_workers: Number of parallel workers. 0 = auto (cpu_count). Default: 0.
        use_screening: Whether to pre-screen starting points by cost.
        screen_keep_fraction: Fraction of starts to keep after screening.
        refine_top_k: Refine only the top-k starts after screening.
        refinement_ftol: Function tolerance for refinement phase.
        degeneracy_threshold: Relative chi-squared threshold for degeneracy
            detection across basins.
        parallel: Whether to attempt parallel execution via ProcessPoolExecutor.
            Defaults to False because JAX closures cannot be sent across
            process boundaries.
        max_workers: Maximum number of worker processes.  Defaults to
            ``min(n_starts, os.cpu_count() or 4)``.
        worker_timeout: Per-worker timeout in seconds.
        max_data_points_for_parallel: Auto-disable parallel when
            ``n_data * n_starts`` exceeds this threshold.
    """

    enable: bool = False
    n_starts: int = 10
    seed: int | None = None
    sampling_strategy: str = "latin_hypercube"
    custom_starts: list[list[float]] | None = None
    n_workers: int = 0
    use_screening: bool = True
    screen_keep_fraction: float = 0.5
    refine_top_k: int = 3
    refinement_ftol: float = 1e-12
    degeneracy_threshold: float = 0.1
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

    @classmethod
    def from_nlsq_config(cls, nlsq_config: Any) -> MultiStartConfig:
        """Create MultiStartConfig from NLSQConfig.

        Parameters
        ----------
        nlsq_config : NLSQConfig
            NLSQ configuration object.

        Returns
        -------
        MultiStartConfig
            Multi-start configuration.
        """
        custom_starts = getattr(nlsq_config, "multi_start_custom_starts", None)
        return cls(
            enable=getattr(nlsq_config, "enable_multi_start", False),
            n_starts=getattr(nlsq_config, "multi_start_n_starts", 10),
            seed=getattr(nlsq_config, "multi_start_seed", None),
            sampling_strategy=getattr(
                nlsq_config, "multi_start_sampling_strategy", "latin_hypercube"
            ),
            custom_starts=custom_starts,
            n_workers=getattr(nlsq_config, "multi_start_n_workers", 0),
            use_screening=getattr(nlsq_config, "multi_start_use_screening", True),
            screen_keep_fraction=getattr(
                nlsq_config, "multi_start_screen_keep_fraction", 0.5
            ),
            refine_top_k=getattr(nlsq_config, "multi_start_refine_top_k", 3),
            refinement_ftol=getattr(nlsq_config, "multi_start_refinement_ftol", 1e-12),
            degeneracy_threshold=getattr(
                nlsq_config, "multi_start_degeneracy_threshold", 0.1
            ),
        )

    def to_nlsq_global_config(self) -> Any:
        """Convert to NLSQ's GlobalOptimizationConfig.

        Returns
        -------
        GlobalOptimizationConfig
            NLSQ global optimization configuration.

        Raises
        ------
        ImportError
            If NLSQ GlobalOptimizationConfig is not available.

        Notes
        -----
        Maps heterodyne's multi-start configuration to NLSQ's
        GlobalOptimizationConfig:

        - sampling_strategy -> sampler (lhs, sobol, halton)
        - use_screening -> elimination_rounds (0 if disabled)
        - screen_keep_fraction -> elimination_fraction (inverted)
        """
        try:
            from nlsq.global_optimization import GlobalOptimizationConfig
        except ImportError as e:
            raise ImportError(
                "NLSQ GlobalOptimizationConfig not available. "
                "Please install NLSQ >= 0.4.0: pip install nlsq>=0.4.0"
            ) from e

        sampler_map = {
            "latin_hypercube": "lhs",
            "lhs": "lhs",
            "sobol": "sobol",
            "halton": "halton",
            "random": "lhs",
        }
        sampler_str = sampler_map.get(self.sampling_strategy, "lhs")
        elimination_fraction = 1.0 - self.screen_keep_fraction
        elimination_rounds = 3 if self.use_screening else 0

        return GlobalOptimizationConfig(
            n_starts=self.n_starts,
            sampler=sampler_str,  # type: ignore[arg-type]
            elimination_rounds=elimination_rounds,
            elimination_fraction=elimination_fraction,
        )


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
        strategy_used: Strategy that was used (always ``"full"``).
        n_unique_basins: Number of distinct local minima found.
        degeneracy_detected: Whether parameter degeneracy was detected.
        total_wall_time: Total execution time in seconds (homodyne-parity alias).
        wall_time_total: Total elapsed wall-clock time in seconds.
        screening_costs: Initial costs from screening phase.
        basin_labels: Cluster labels for each result.
    """

    best_result: NLSQResult
    all_starts: list[SingleStartResult]
    n_successful: int
    n_total: int
    config: MultiStartConfig
    strategy_used: str = "full"
    n_unique_basins: int = 1
    degeneracy_detected: bool = False
    wall_time_total: float = 0.0
    screening_costs: NDArray[np.float64] | None = None
    basin_labels: NDArray[np.int64] | None = None

    @property
    def best(self) -> NLSQResult:
        """Homodyne-parity alias: best result by cost."""
        return self.best_result

    @property
    def total_wall_time(self) -> float:
        """Homodyne-parity alias for :attr:`wall_time_total`.

        Was a duplicate dataclass field; now a derived property so
        callers using either name always see the same value (previously
        ``total_wall_time`` was always 0.0 because ``fit()`` only set
        ``wall_time_total``).
        """
        return self.wall_time_total

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
            "strategy_used": self.strategy_used,
            "n_unique_basins": self.n_unique_basins,
            "degeneracy_detected": self.degeneracy_detected,
        }
        return self.best_result

    def to_optimization_result(self) -> Any:
        """Convert MultiStartResult to a result dict for CLI compatibility.

        Returns a dictionary containing the best solution with multi-start
        metadata, compatible with downstream reporting.

        Returns
        -------
        dict[str, Any]
            Result dict with best parameters and multi-start diagnostics.
        """
        best = self.best_result
        multistart_diagnostics: dict[str, Any] = {
            "strategy_used": self.strategy_used,
            "n_starts": self.n_total,
            "n_successful": self.n_successful,
            "n_unique_basins": self.n_unique_basins,
            "degeneracy_detected": self.degeneracy_detected,
            "total_wall_time": self.total_wall_time,
        }
        return {
            "best_result": best,
            "multistart_diagnostics": multistart_diagnostics,
        }


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

        # JAX closures cannot cross process boundaries — the production path
        # is always sequential. ``_force_parallel`` is the explicit test-only
        # opt-in for exercising ``_run_parallel`` with a process-safe
        # residual function (no JAX tracers, no live config managers).
        force_parallel = bool(getattr(self, "_force_parallel", False))
        use_parallel = force_parallel and self._config.parallel

        logger.info(
            "Multi-start: %d starts, parallel=%s%s",
            n_points,
            use_parallel,
            " (force-flag set)" if force_parallel else "",
        )

        if self._config.parallel and not force_parallel:
            logger.warning(
                "Parallel multi-start requested but JAX closures cannot be "
                "transmitted across process boundaries. "
                "Falling back to sequential execution. "
                "(Set ``optimizer._force_parallel = True`` to exercise the "
                "parallel path explicitly — test-only.)"
            )

        if use_parallel:
            all_starts = self._run_parallel(
                starting_points=starting_points,
                residual_fn=residual_fn,
                bounds=bounds,
                config=config,
                jacobian_fn=jacobian_fn,
            )
        else:
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


def validate_n_starts_for_lhs(
    n_starts: int,
    n_params: int,
    warn: bool = True,
) -> int:
    """Validate n_starts for Latin Hypercube Sampling coverage.

    For LHS to provide meaningful coverage, n_starts should be at least
    n_params.  Very large n_starts relative to parameter space may produce
    redundant samples.

    Parameters
    ----------
    n_starts : int
        Requested number of starting points.
    n_params : int
        Number of parameters (dimensions).
    warn : bool
        Whether to emit warnings for suboptimal settings.

    Returns
    -------
    int
        Validated n_starts (unchanged if valid).
    """
    if n_starts < n_params and warn:
        logger.warning(
            "n_starts (%d) < n_params (%d): "
            "LHS coverage may be inadequate. Consider n_starts >= %d.",
            n_starts,
            n_params,
            n_params,
        )
    max_meaningful = n_params * 1000
    if n_starts > max_meaningful and warn:
        logger.warning(
            "n_starts (%d) is very large for %d parameters. "
            "This may produce redundant samples with diminishing returns. "
            "Consider n_starts <= %d.",
            n_starts,
            n_params,
            max_meaningful,
        )
    return n_starts


def generate_random_starts(
    bounds: NDArray[np.float64],
    n_starts: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate starting points via random uniform sampling.

    Parameters
    ----------
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array.
    n_starts : int
        Number of starting points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    """
    rng = np.random.default_rng(seed)
    n_params = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    samples = rng.uniform(lower, upper, size=(n_starts, n_params))
    logger.debug(
        "Generated %d random starting points for %d parameters", n_starts, n_params
    )
    return samples


def include_custom_starts(
    generated_starts: NDArray[np.float64],
    custom_starts: list[list[float]] | NDArray[np.float64] | None,
    bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Include user-provided custom starting points alongside generated starts.

    Custom starting points are prepended to the generated starts so they are
    always included (not filtered by screening).

    Parameters
    ----------
    generated_starts : NDArray[np.float64]
        Starting points generated by LHS or random sampling.
    custom_starts : list[list[float]] | NDArray[np.float64] | None
        User-provided custom starting points.
    bounds : NDArray[np.float64]
        Parameter bounds as (n_params, 2) array, used for validation.

    Returns
    -------
    NDArray[np.float64]
        Combined starting points with custom starts first.
    """
    if custom_starts is None or len(custom_starts) == 0:
        return generated_starts

    custom_array = np.asarray(custom_starts, dtype=np.float64)
    n_params = bounds.shape[0]
    if custom_array.ndim == 1:
        custom_array = custom_array.reshape(1, -1)

    if custom_array.shape[1] != n_params:
        logger.warning(
            "Custom starts have wrong dimension: %d != %d. Ignoring custom starts.",
            custom_array.shape[1],
            n_params,
        )
        return generated_starts

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    n_custom = len(custom_array)
    valid_mask = np.all((custom_array >= lower) & (custom_array <= upper), axis=1)
    n_valid = int(np.sum(valid_mask))

    if n_valid < n_custom:
        logger.warning(
            "%d custom starting point(s) are outside bounds and will be skipped.",
            n_custom - n_valid,
        )
        custom_array = custom_array[valid_mask]

    if len(custom_array) == 0:
        return generated_starts

    logger.info("Including %d custom starting point(s)", len(custom_array))
    return np.vstack([custom_array, generated_starts])


def screen_starts(
    cost_func: Callable[[NDArray[np.float64]], float],
    starts: NDArray[np.float64],
    keep_fraction: float = 0.5,
    min_keep: int = 3,
    n_workers: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pre-filter starting points by initial cost.

    Parameters
    ----------
    cost_func : Callable
        Function that computes cost (chi-squared) for a parameter vector.
    starts : NDArray[np.float64]
        Starting points as (n_starts, n_params) array.
    keep_fraction : float
        Fraction of starting points to keep (0, 1].
    min_keep : int
        Minimum number of starting points to keep.
    n_workers : int
        Number of parallel workers for cost evaluation. 0 = auto
        (cpu_count - 1).

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Filtered starting points and their initial costs (all costs, not just
        kept ones).
    """
    from concurrent.futures import ThreadPoolExecutor

    n_total_starts = len(starts)
    n_keep = max(min_keep, int(n_total_starts * keep_fraction))
    n_keep = min(n_keep, n_total_starts)

    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    if n_total_starts >= 4 and n_workers > 1:
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                costs = np.array(list(executor.map(cost_func, starts)))
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "Parallel screening failed, falling back to sequential: %s", exc
            )
            costs = np.array([cost_func(start) for start in starts])
    else:
        costs = np.array([cost_func(start) for start in starts])

    sorted_indices = np.argsort(costs)
    keep_indices = sorted_indices[:n_keep]
    filtered_starts = starts[keep_indices]
    filtered_costs = costs[keep_indices]

    logger.info(
        "Screening: kept %d/%d starts (best cost: %.4g, worst kept: %.4g)",
        n_keep,
        n_total_starts,
        filtered_costs[0],
        filtered_costs[-1],
    )
    return filtered_starts, costs


def get_n_workers(config: MultiStartConfig, n_starts: int) -> int:
    """Determine number of parallel workers.

    Parameters
    ----------
    config : MultiStartConfig
        Multi-start configuration.
    n_starts : int
        Number of starting points.

    Returns
    -------
    int
        Number of workers to use.
    """
    if config.n_workers > 0:
        n_workers = config.n_workers
    else:
        n_workers = os.cpu_count() or 4
    n_workers = min(n_workers, n_starts)
    logger.debug("Using %d parallel workers for %d starts", n_workers, n_starts)
    return n_workers


def detect_degeneracy(
    results: list[SingleStartResult],
    chi_sq_threshold: float = 0.1,
    param_threshold: float = 0.2,
) -> tuple[bool, int, NDArray[np.int64] | None]:
    """Detect parameter degeneracy from multiple optimization results.

    Parameters
    ----------
    results : list[SingleStartResult]
        List of optimization results (heterodyne ``SingleStartResult``
        wrapping an inner ``NLSQResult``).
    chi_sq_threshold : float
        Maximum relative chi-squared difference to consider similar.
    param_threshold : float
        Maximum relative parameter distance to consider same basin.

    Returns
    -------
    tuple[bool, int, NDArray[np.int64] | None]
        (degeneracy_detected, n_unique_basins, basin_labels)
    """
    successful = [r for r in results if r.result.success]
    if len(successful) < 2:
        return False, 1, None

    # Sort by cost
    successful.sort(
        key=lambda r: r.result.final_cost if r.result.final_cost is not None else np.inf
    )
    best_cost = (
        successful[0].result.final_cost
        if successful[0].result.final_cost is not None
        else 0.0
    )

    basins: list[list[SingleStartResult]] = []
    basin_assignments: list[int] = []

    for r in successful:
        r_cost = r.result.final_cost if r.result.final_cost is not None else np.inf
        cost_diff = abs(r_cost - best_cost) / (abs(best_cost) + 1e-10)
        if cost_diff > chi_sq_threshold:
            basin_assignments.append(-1)
            continue

        r_params = r.result.parameters
        found_basin = False
        for basin_idx, basin in enumerate(basins):
            center_params = basin[0].result.parameters
            if center_params is None:
                continue
            param_dist = np.linalg.norm(r_params - center_params) / (
                np.linalg.norm(center_params) + 1e-10
            )
            if param_dist < param_threshold:
                basin.append(r)
                basin_assignments.append(basin_idx)
                found_basin = True
                break

        if not found_basin:
            basins.append([r])
            basin_assignments.append(len(basins) - 1)

    n_unique_basins = len(basins)
    degeneracy_detected = n_unique_basins > 1
    labels = np.array(basin_assignments, dtype=np.int64)

    if degeneracy_detected:
        logger.warning(
            "Parameter degeneracy detected: %d distinct basins "
            "with similar chi-squared values",
            n_unique_basins,
        )

    return degeneracy_detected, n_unique_basins, labels
