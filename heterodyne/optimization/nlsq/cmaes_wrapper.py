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

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1")
        if self.popsize is not None and self.popsize < 2:
            raise ValueError("popsize must be >= 2 when specified")


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
                "CMA-ES requires the 'cma' package. Install it with: "
                "uv add cma"
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

        stop_reason = "; ".join(
            f"{k}={v}" for k, v in es.stop().items()
        )
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
