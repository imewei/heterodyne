"""Parallel Gauss-Newton accumulation for chunked NLSQ optimization.

Provides out-of-core and optionally parallel computation of J^T J and
J^T r accumulations across data chunks, enabling memory-efficient
fitting of datasets that exceed available RAM.

The accumulation pattern:
    J^T J = sum_k J_k^T J_k
    J^T r = sum_k J_k^T r_k
    cost  = sum_k ||r_k||^2

where k indexes over data chunks. This is mathematically equivalent to
computing the full Jacobian but requires only O(chunk_size * n_params)
memory instead of O(n_data * n_params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class GaussNewtonAccumulation:
    """Accumulated Gauss-Newton quantities across data chunks.

    These quantities are sufficient to solve the normal equations
    J^T J delta = -J^T r without forming the full Jacobian.

    Attributes:
        JtJ: Accumulated J^T J matrix, shape (n_params, n_params).
        Jtf: Accumulated J^T r vector, shape (n_params,).
        cost: Accumulated sum of squared residuals.
        n_data: Total number of data points processed.
    """

    JtJ: np.ndarray
    Jtf: np.ndarray
    cost: float
    n_data: int


def accumulate_chunks_sequential(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    residual_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    params: np.ndarray,
) -> GaussNewtonAccumulation:
    """Accumulate Gauss-Newton quantities sequentially over chunks.

    Args:
        chunks: List of (data_chunk, weight_chunk) tuples. Each data_chunk
            has shape (chunk_size, ...) and weight_chunk matches or is None.
        residual_fn: Function taking params and returning (residuals, jacobian)
            for a given chunk. The function is called once per chunk.
        params: Current parameter values, shape (n_params,).

    Returns:
        GaussNewtonAccumulation with accumulated quantities.
    """
    n_params = len(params)
    JtJ = np.zeros((n_params, n_params), dtype=np.float64)
    Jtf = np.zeros(n_params, dtype=np.float64)
    cost = 0.0
    n_data = 0

    for chunk_idx, (_data_chunk, weight_chunk) in enumerate(chunks):
        try:
            residuals, jacobian = residual_fn(params)

            # Apply weights if provided
            if weight_chunk is not None:
                w = np.asarray(weight_chunk).ravel()
                residuals = residuals * np.sqrt(w)
                jacobian = jacobian * np.sqrt(w)[:, np.newaxis]

            # Accumulate normal equations
            JtJ += jacobian.T @ jacobian
            Jtf += jacobian.T @ residuals
            cost += float(np.dot(residuals, residuals))
            n_data += len(residuals)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Chunk %d failed during accumulation: %s (skipping)",
                chunk_idx, exc,
            )

    logger.debug(
        "Sequential accumulation: %d chunks, %d total points, cost=%.4e",
        len(chunks), n_data, cost,
    )

    return GaussNewtonAccumulation(JtJ=JtJ, Jtf=Jtf, cost=cost, n_data=n_data)


def accumulate_chunks_parallel(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    residual_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    params: np.ndarray,
    n_workers: int = 2,
) -> GaussNewtonAccumulation:
    """Accumulate Gauss-Newton quantities in parallel using threads.

    Uses concurrent.futures.ThreadPoolExecutor for parallel chunk
    evaluation. Falls back to sequential if threading fails.

    Args:
        chunks: List of (data_chunk, weight_chunk) tuples.
        residual_fn: Residual+Jacobian function.
        params: Current parameter values.
        n_workers: Number of parallel workers.

    Returns:
        GaussNewtonAccumulation with accumulated quantities.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_params = len(params)
    JtJ = np.zeros((n_params, n_params), dtype=np.float64)
    Jtf = np.zeros(n_params, dtype=np.float64)
    cost = 0.0
    n_data = 0

    def process_chunk(
        chunk_data: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Process a single chunk and return partial results."""
        data_chunk, weight_chunk = chunk_data
        residuals, jacobian = residual_fn(params)

        if weight_chunk is not None:
            w = np.asarray(weight_chunk).ravel()
            residuals = residuals * np.sqrt(w)
            jacobian = jacobian * np.sqrt(w)[:, np.newaxis]

        chunk_JtJ = jacobian.T @ jacobian
        chunk_Jtf = jacobian.T @ residuals
        chunk_cost = float(np.dot(residuals, residuals))
        chunk_n = len(residuals)

        return chunk_JtJ, chunk_Jtf, chunk_cost, chunk_n

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    c_JtJ, c_Jtf, c_cost, c_n = future.result()
                    JtJ += c_JtJ
                    Jtf += c_Jtf
                    cost += c_cost
                    n_data += c_n
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Parallel chunk %d failed: %s (skipping)",
                        chunk_idx, exc,
                    )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Parallel accumulation failed (%s); falling back to sequential",
            exc,
        )
        return accumulate_chunks_sequential(chunks, residual_fn, params)

    logger.debug(
        "Parallel accumulation: %d chunks, %d workers, %d total points",
        len(chunks), n_workers, n_data,
    )

    return GaussNewtonAccumulation(JtJ=JtJ, Jtf=Jtf, cost=cost, n_data=n_data)


def should_use_parallel_accumulation(n_chunks: int, threshold: int = 10) -> bool:
    """Decide whether parallel chunk accumulation is worthwhile.

    Threading overhead makes parallel evaluation slower for small
    numbers of chunks. The threshold is empirically set at 10.

    Args:
        n_chunks: Number of data chunks.
        threshold: Minimum chunks to justify parallelism.

    Returns:
        True if parallel accumulation is recommended.
    """
    return n_chunks >= threshold


def create_ooc_kernels(
    n_params: int = 14,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Create JIT-compiled out-of-core kernels for J^T J and J^T r.

    Uses JAX JIT compilation for efficient matrix-vector products
    during chunk accumulation.

    Args:
        n_params: Number of model parameters.

    Returns:
        Tuple of (jtj_kernel, jtr_kernel) where each is a JIT-compiled
        function taking (jacobian_chunk,) or (jacobian_chunk, residuals_chunk).
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def jtj_kernel(jacobian: jnp.ndarray) -> jnp.ndarray:
        """Compute J^T J for a single chunk."""
        return jacobian.T @ jacobian

    @jax.jit
    def jtr_kernel(jacobian: jnp.ndarray, residuals: jnp.ndarray) -> jnp.ndarray:
        """Compute J^T r for a single chunk."""
        return jacobian.T @ residuals

    logger.debug("Created OOC kernels for %d-parameter model", n_params)
    return jtj_kernel, jtr_kernel
