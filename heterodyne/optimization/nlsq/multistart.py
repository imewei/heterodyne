"""Multi-start optimization with Latin Hypercube Sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

logger = get_logger(__name__)


@dataclass
class MultiStartResult:
    """Result of multi-start optimization."""

    best_result: NLSQResult
    all_results: list[NLSQResult]
    n_successful: int
    n_total: int


class MultiStartOptimizer:
    """Multi-start optimizer using Latin Hypercube Sampling.

    Runs optimization from multiple starting points to improve
    chances of finding the global minimum.
    """

    def __init__(
        self,
        adapter: NLSQAdapterBase,
        n_starts: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize multi-start optimizer.

        Args:
            adapter: NLSQ adapter to use for each optimization
            n_starts: Number of starting points
            seed: Random seed for reproducibility
        """
        self._adapter = adapter
        self._n_starts = n_starts
        self._rng = np.random.default_rng(seed)

    def generate_starting_points(
        self,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Generate starting points using Latin Hypercube Sampling.

        Args:
            initial_params: User-provided initial values (used as first point)
            bounds: (lower, upper) bound arrays

        Returns:
            Array of shape (n_starts, n_params)
        """
        lower, upper = bounds
        n_params = len(initial_params)

        # First point is user-provided
        starting_points = [initial_params.copy()]

        # Generate LHS points for remaining starts
        n_lhs = self._n_starts - 1
        if n_lhs > 0:
            lhs_points = self._latin_hypercube_sample(n_lhs, n_params, lower, upper)
            starting_points.extend(lhs_points)

        return np.array(starting_points)

    def _latin_hypercube_sample(
        self,
        n_samples: int,
        n_dims: int,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate Latin Hypercube samples.

        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            lower: Lower bounds
            upper: Upper bounds

        Returns:
            List of sample arrays
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

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> MultiStartResult:
        """Run multi-start optimization.

        Args:
            residual_fn: Residual function
            initial_params: Initial guess
            bounds: Parameter bounds
            config: Optimization config
            jacobian_fn: Optional Jacobian

        Returns:
            MultiStartResult with best and all results
        """
        starting_points = self.generate_starting_points(initial_params, bounds)

        logger.info(f"Running multi-start optimization with {len(starting_points)} starts")

        all_results: list[NLSQResult] = []
        best_result: NLSQResult | None = None
        best_cost = np.inf

        for i, start in enumerate(starting_points):
            logger.debug(f"Starting point {i+1}/{len(starting_points)}")

            result = self._adapter.fit(
                residual_fn=residual_fn,
                initial_params=start,
                bounds=bounds,
                config=config,
                jacobian_fn=jacobian_fn,
            )

            all_results.append(result)

            if result.success and result.final_cost is not None:
                if result.final_cost < best_cost:
                    best_cost = result.final_cost
                    best_result = result
                    logger.debug(f"  New best cost: {best_cost:.6e}")

        n_successful = sum(1 for r in all_results if r.success)

        if best_result is None:
            logger.warning("All optimization runs failed, using result with lowest cost")
            best_result = min(
                all_results,
                key=lambda r: r.final_cost if r.final_cost is not None else np.inf,
            )

        logger.info(
            f"Multi-start complete: {n_successful}/{len(starting_points)} successful, "
            f"best cost = {best_cost:.6e}"
        )

        return MultiStartResult(
            best_result=best_result,
            all_results=all_results,
            n_successful=n_successful,
            n_total=len(starting_points),
        )
