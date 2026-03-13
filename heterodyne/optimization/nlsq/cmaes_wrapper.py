"""CMA-ES optimization wrapper for heterodyne parameter fitting.

Provides a derivative-free global optimizer as an alternative to
gradient-based NLSQ methods.  Useful when the cost landscape is
multi-modal or the Jacobian is unreliable.

The ``cma`` package is an optional dependency.  If it is not installed,
importing this module succeeds but :meth:`CMAESWrapper.fit` raises
:class:`ImportError` at call time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.result_builder import build_result_from_arrays
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)

try:
    import cma as _cma  # noqa: F401

    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False

#: Public availability flag consumed by ``core.py``.
CMAES_AVAILABLE: bool = _HAS_CMA


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimization.

    Attributes:
        sigma0: Initial step-size (standard deviation) for the search
            distribution.  A good default is ~1/4 of the expected parameter
            range.
        popsize: Population size.  ``None`` lets cma choose automatically.
        maxiter: Maximum number of CMA-ES generations.
        tolx: Termination tolerance on parameter changes.
        tolfun: Termination tolerance on cost function changes.
        seed: Random seed for reproducibility.
    """

    sigma0: float = 0.5
    popsize: int | None = None
    maxiter: int = 1000
    tolx: float = 1e-11
    tolfun: float = 1e-11
    seed: int = 42
    diagonal_filtering: str = "none"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1")
        if self.popsize is not None and self.popsize < 2:
            raise ValueError("popsize must be >= 2 when specified")
        if self.diagonal_filtering not in ("remove", "none"):
            raise ValueError(
                f"diagonal_filtering must be 'remove' or 'none', "
                f"got {self.diagonal_filtering!r}"
            )


class CMAESWrapper:
    """Wrapper around ``cma.fmin2`` for bounded parameter optimization.

    Adapts CMA-ES to the heterodyne fitting interface, returning
    :class:`~heterodyne.optimization.nlsq.results.NLSQResult`-compatible
    output via the result builder.

    Parameters
    ----------
    config : CMAESConfig
        CMA-ES hyperparameters.
    parameter_names : list[str]
        Ordered parameter names (used in the returned result).
    """

    def __init__(
        self,
        config: CMAESConfig | None = None,
        parameter_names: list[str] | None = None,
    ) -> None:
        self._config = config or CMAESConfig()
        self._parameter_names = parameter_names or []

    def fit(
        self,
        objective_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        *,
        residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        n_data: int | None = None,
        parameter_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NLSQResult:
        """Run CMA-ES optimization.

        Args:
            objective_fn: Scalar objective ``f(x) -> float`` to minimize.
            x0: Initial guess, shape ``(n_params,)``.
            bounds: ``(lower, upper)`` arrays, each shape ``(n_params,)``.
            residual_fn: Optional residual function for building a full
                :class:`NLSQResult` with covariance information.  If ``None``
                a synthetic residual vector is constructed from the final cost.
            n_data: Number of data points (for reduced chi-squared).  Defaults
                to 1 if not provided.
            parameter_names: Override parameter names for this call.
            metadata: Extra metadata attached to the result.

        Returns:
            An :class:`NLSQResult` populated from the CMA-ES solution.

        Raises:
            ImportError: If the ``cma`` package is not installed.
        """
        if not _HAS_CMA:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install it with: uv add cma"
            )
        import cma  # noqa: F811 — guarded re-import

        names = parameter_names or self._parameter_names
        if not names:
            names = [f"p{i}" for i in range(len(x0))]

        x0 = np.asarray(x0, dtype=np.float64)
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)

        opts: dict[str, Any] = {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxiter": self._config.maxiter,
            "tolx": self._config.tolx,
            "tolfun": self._config.tolfun,
            "seed": self._config.seed,
            "verbose": -9,  # suppress cma stdout
        }
        if self._config.popsize is not None:
            opts["popsize"] = self._config.popsize
        if self._config.diagonal_filtering == "remove":
            opts["CMA_diagonal"] = True

        logger.info(
            "Starting CMA-ES: sigma0=%.3g, maxiter=%d, n_params=%d",
            self._config.sigma0,
            self._config.maxiter,
            len(x0),
        )

        t0 = time.perf_counter()
        es = cma.CMAEvolutionStrategy(x0.tolist(), self._config.sigma0, opts)
        es.optimize(objective_fn)
        wall_time = time.perf_counter() - t0

        best_x = np.asarray(es.result.xbest, dtype=np.float64)
        best_cost = float(es.result.fbest)
        n_evals = int(es.result.evaluations)
        n_iters = int(es.result.iterations)

        logger.info(
            "CMA-ES finished: cost=%.6e, evals=%d, iters=%d, wall=%.2fs",
            best_cost,
            n_evals,
            n_iters,
            wall_time,
        )

        # Build residual vector for NLSQResult
        if residual_fn is not None:
            residuals = np.asarray(residual_fn(best_x), dtype=np.float64)
        else:
            # Synthetic scalar residual from cost
            residuals = np.array([np.sqrt(max(best_cost, 0.0))], dtype=np.float64)

        effective_n_data = n_data if n_data is not None else max(len(residuals), 1)

        stop_reason = "; ".join(f"{k}={v}" for k, v in es.stop().items())
        success = best_cost < np.inf and not np.isnan(best_cost)

        result_metadata = {"optimizer": "CMA-ES", "stop_conditions": es.stop()}
        if metadata:
            result_metadata.update(metadata)

        return build_result_from_arrays(
            parameters=best_x,
            parameter_names=names,
            residuals=residuals,
            n_data=effective_n_data,
            success=success,
            message=stop_reason if stop_reason else "CMA-ES completed",
            n_iterations=n_iters,
            n_function_evals=n_evals,
            wall_time=wall_time,
            metadata=result_metadata,
        )


# ---------------------------------------------------------------------------
# CMAESResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CMAESResult:
    """Detailed result container for a CMA-ES optimization run.

    Attributes:
        best_params: Mapping of parameter name to fitted value.
        best_cost: Scalar cost (objective value) at the solution.
        n_iterations: Number of CMA-ES generations completed.
        n_evaluations: Total number of objective function evaluations.
        converged: Whether CMA-ES stopped due to a convergence criterion.
        final_sigma: Step-size (standard deviation) at termination.
        history: Cost value recorded at the end of each generation.
    """

    best_params: dict[str, float]
    best_cost: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    final_sigma: float
    history: list[float]


# ---------------------------------------------------------------------------
# Coordinate transformation utilities
# ---------------------------------------------------------------------------


def normalize_to_unit_cube(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Map parameter vector from physical bounds to the unit hypercube [0, 1].

    Args:
        x: Parameter vector of shape ``(n_params,)``.
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        Transformed array of shape ``(n_params,)`` with values in [0, 1].
        Dimensions where ``lower == upper`` are mapped to 0.0.

    Raises:
        ValueError: If arrays have mismatched shapes.
    """
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (x.shape == lower.shape == upper.shape):
        raise ValueError(
            f"x, lower, and upper must have the same shape; "
            f"got {x.shape}, {lower.shape}, {upper.shape}"
        )
    span = upper - lower
    # Avoid division by zero for fixed dimensions
    safe_span = np.where(span > 0, span, 1.0)
    x_norm = np.where(span > 0, (x - lower) / safe_span, 0.0)
    return x_norm


def denormalize_from_unit_cube(
    x_norm: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Map parameter vector from the unit hypercube [0, 1] to physical bounds.

    This is the inverse of :func:`normalize_to_unit_cube`.

    Args:
        x_norm: Normalized parameter vector of shape ``(n_params,)``
            with values in [0, 1].
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        De-normalized array of shape ``(n_params,)`` in physical units.

    Raises:
        ValueError: If arrays have mismatched shapes.
    """
    x_norm = np.asarray(x_norm, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (x_norm.shape == lower.shape == upper.shape):
        raise ValueError(
            f"x_norm, lower, and upper must have the same shape; "
            f"got {x_norm.shape}, {lower.shape}, {upper.shape}"
        )
    return lower + x_norm * (upper - lower)


def compute_adaptive_cmaes_params(
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
    """Compute adaptive population size and max generations from bound ranges.

    Scales popsize from 50 to 200 and max_generations from 200 to 500 based
    on the ratio between the largest and smallest parameter ranges.  This
    ensures the search budget grows with the difficulty implied by
    heterogeneous parameter scales (e.g. D0 in 1e6 vs alpha near 1).

    Args:
        bounds: ``(lower, upper)`` arrays, each shape ``(n_params,)``.

    Returns:
        ``(popsize, max_generations)`` tuple.
    """
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    spans = upper - lower
    # Filter out fixed dimensions (span <= 0)
    active_spans = spans[spans > 0]
    if len(active_spans) < 2:
        return 50, 200

    scale_ratio = float(active_spans.max() / active_spans.min())

    # Clamp the ratio to [1, 1e6] for interpolation
    log_ratio = np.log10(max(scale_ratio, 1.0))
    # log_ratio in [0, 6], map linearly to popsize in [50, 200]
    t = min(log_ratio / 6.0, 1.0)
    popsize = int(50 + t * 150)

    # max_generations: [200, 500]
    max_generations = int(200 + t * 300)

    logger.debug(
        "Adaptive CMA-ES: scale_ratio=%.1f, popsize=%d, max_generations=%d",
        scale_ratio,
        popsize,
        max_generations,
    )
    return popsize, max_generations


def build_anti_degeneracy_objective(
    base_objective: Callable[[np.ndarray], float],
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_names: list[str],
    penalty_weight: float = 0.01,
) -> Callable[[np.ndarray], float]:
    """Wrap an objective with penalty terms for known degenerate parameter pairs.

    For each known degenerate pair (e.g. v0/v_offset, D0_ref/D0_sample),
    a soft penalty encourages separation of their normalized values.
    This discourages the optimizer from settling in degenerate valleys.

    Args:
        base_objective: Original scalar objective ``f(x) -> float``.
        bounds: ``(lower, upper)`` arrays.
        parameter_names: Parameter names matching the indices in ``x``.
        penalty_weight: Strength of the penalty terms relative to the
            base cost.

    Returns:
        Wrapped objective function with degeneracy penalties.
    """
    from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
        _KNOWN_DEGENERATE_PAIRS,
    )

    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    spans = upper - lower
    safe_spans = np.where(spans > 0, spans, 1.0)

    # Build index pairs for known degeneracies that are present in this fit
    name_to_idx = {n: i for i, n in enumerate(parameter_names)}
    active_pairs: list[tuple[int, int]] = []
    for name_a, name_b, _desc in _KNOWN_DEGENERATE_PAIRS:
        if name_a in name_to_idx and name_b in name_to_idx:
            active_pairs.append((name_to_idx[name_a], name_to_idx[name_b]))

    if not active_pairs:
        return base_objective

    def penalized_objective(x: np.ndarray) -> float:
        cost = base_objective(x)
        x_norm = (x - lower) / safe_spans
        penalty = 0.0
        for idx_a, idx_b in active_pairs:
            diff = abs(x_norm[idx_a] - x_norm[idx_b])
            # Penalize when normalized values are too close
            penalty += 1.0 / (diff + 0.01)
        return cost + penalty_weight * penalty

    return penalized_objective


def fit_with_cmaes(
    objective_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_names: list[str] | None = None,
    *,
    config: CMAESConfig | None = None,
    residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_data: int | None = None,
    anti_degeneracy: bool = False,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Convenience function to run CMA-ES optimization.

    Applies adaptive population sizing and max_generations based on the
    parameter scale ratio when the user hasn't explicitly set these values.

    Args:
        objective_fn: Scalar objective ``f(x) -> float`` to minimize.
        initial_params: Initial guess, shape ``(n_params,)``.
        bounds: ``(lower, upper)`` arrays.
        parameter_names: Ordered parameter names.
        config: CMA-ES configuration; ``None`` for defaults.
        residual_fn: Optional residual function for covariance estimation.
        n_data: Number of data points (for reduced chi-squared).
        anti_degeneracy: Apply degeneracy-penalty weighting to objective.
        metadata: Extra metadata attached to the result.

    Returns:
        An :class:`NLSQResult` from the CMA-ES solution.
    """
    cfg = config or CMAESConfig()

    # Adaptive population sizing and max_generations (Fix #2, #3)
    if cfg.popsize is None or cfg.maxiter == CMAESConfig.maxiter:
        adaptive_pop, adaptive_gen = compute_adaptive_cmaes_params(bounds)
        if cfg.popsize is None:
            cfg = CMAESConfig(
                sigma0=cfg.sigma0,
                popsize=adaptive_pop,
                maxiter=cfg.maxiter
                if cfg.maxiter != CMAESConfig.maxiter
                else adaptive_gen,
                tolx=cfg.tolx,
                tolfun=cfg.tolfun,
                seed=cfg.seed,
                diagonal_filtering=cfg.diagonal_filtering,
            )
        elif cfg.maxiter == CMAESConfig.maxiter:
            cfg = CMAESConfig(
                sigma0=cfg.sigma0,
                popsize=cfg.popsize,
                maxiter=adaptive_gen,
                tolx=cfg.tolx,
                tolfun=cfg.tolfun,
                seed=cfg.seed,
                diagonal_filtering=cfg.diagonal_filtering,
            )

    # Anti-degeneracy objective wrapping (Fix #6)
    effective_objective = objective_fn
    names = parameter_names or [f"p{i}" for i in range(len(initial_params))]
    if anti_degeneracy:
        effective_objective = build_anti_degeneracy_objective(
            objective_fn,
            bounds,
            names,
        )

    wrapper = CMAESWrapper(config=cfg, parameter_names=names)
    return wrapper.fit(
        objective_fn=effective_objective,
        x0=initial_params,
        bounds=bounds,
        residual_fn=residual_fn,
        n_data=n_data,
        parameter_names=names,
        metadata=metadata,
    )


def adjust_covariance_for_bounds(
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Scale a covariance matrix to account for parameter bounds ranges.

    When parameters have very different dynamic ranges, a unit-cube
    covariance should be scaled to physical space.  The adjustment is:

        cov_adjusted[i, j] = cov[i, j] * (upper[i] - lower[i])
                                        * (upper[j] - lower[j])

    Args:
        cov: Covariance matrix of shape ``(n_params, n_params)``.
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        Adjusted covariance matrix of shape ``(n_params, n_params)``.

    Raises:
        ValueError: If array shapes are inconsistent.
    """
    cov = np.asarray(cov, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    n = len(lower)
    if cov.shape != (n, n):
        raise ValueError(
            f"cov must be square with side length matching bounds ({n}), "
            f"got shape {cov.shape}"
        )
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError(
            f"lower and upper must have shape ({n},), "
            f"got {lower.shape} and {upper.shape}"
        )

    spans = upper - lower  # shape (n,)
    # Outer product of spans gives the scaling matrix
    scale = np.outer(spans, spans)  # shape (n, n)
    result = cov * scale

    # Warn if the result is not positive semi-definite (can happen if
    # the input cov has numerical noise from CMA-ES)
    min_eig = float(np.linalg.eigvalsh(result).min())
    if min_eig < 0:
        logger.warning(
            "adjust_covariance_for_bounds: result is not PSD "
            "(min eigenvalue=%.2e). Downstream consumers may need "
            "to apply a PSD projection.",
            min_eig,
        )

    return result
