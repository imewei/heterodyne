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

Also provides ``OOCSharedArrays`` and ``OOCComputePool`` for parallelizing
per-chunk JIT compute across persistent workers using shared memory.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.shared_memory
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_MIN_CHUNKS_FOR_PARALLEL = 10
_MIN_CHUNKS_FOR_PARALLEL_COMPUTE = 10


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
                chunk_idx,
                exc,
            )

    logger.debug(
        "Sequential accumulation: %d chunks, %d total points, cost=%.4e",
        len(chunks),
        n_data,
        cost,
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
    from concurrent.futures import ThreadPoolExecutor

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
                        chunk_idx,
                        exc,
                    )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Parallel accumulation failed (%s); falling back to sequential",
            exc,
        )
        return accumulate_chunks_sequential(chunks, residual_fn, params)

    logger.debug(
        "Parallel accumulation: %d chunks, %d workers, %d total points",
        len(chunks),
        n_workers,
        n_data,
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


def should_use_parallel_compute(n_chunks: int) -> bool:
    """Determine if parallel chunk COMPUTE is worthwhile.

    Parameters
    ----------
    n_chunks : int
        Number of chunks in the out-of-core iteration.

    Returns
    -------
    bool
        True if n_chunks >= threshold for parallel compute.
    """
    return n_chunks >= _MIN_CHUNKS_FOR_PARALLEL_COMPUTE


# ---------------------------------------------------------------------------
# OOC shared memory and compute pool
# ---------------------------------------------------------------------------

# Worker-process globals (set by _ooc_worker_init)
_w_phi: np.ndarray | None = None
_w_t1: np.ndarray | None = None
_w_t2: np.ndarray | None = None
_w_g2: np.ndarray | None = None
_w_sigma: np.ndarray | None = None
_w_chunk_boundaries: list[tuple[int, int]] = []
_w_compute_accumulators: Callable[..., Any] | None = None
_w_compute_chi2: Callable[..., Any] | None = None
_w_shm_handles: list[multiprocessing.shared_memory.SharedMemory] = []


def _ooc_worker_cleanup() -> None:
    """Close shared memory handles on worker exit."""
    for shm in _w_shm_handles:
        try:
            shm.close()
        except (OSError, ValueError):
            pass


def _ooc_worker_init(
    shm_refs: dict[str, dict[str, Any]],
    physics_config: dict[str, Any],
    chunk_boundaries: list[tuple[int, int]],
    threads_per_worker: int = 1,
) -> None:
    """Initialise OOC worker: attach shared memory and JIT kernels.

    Parameters
    ----------
    shm_refs : dict
        Picklable shared-memory references from ``OOCSharedArrays.get_refs()``.
    physics_config : dict
        Physics constants needed to build JIT kernels.
    chunk_boundaries : list of (start, end) tuples
        Index ranges for each chunk.
    threads_per_worker : int
        OMP/MKL thread count per worker.
    """
    import atexit
    import os

    global _w_phi, _w_t1, _w_t2, _w_g2, _w_sigma  # noqa: PLW0603
    global _w_chunk_boundaries, _w_compute_accumulators, _w_compute_chi2  # noqa: PLW0603
    global _w_shm_handles  # noqa: PLW0603

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

    import jax

    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(__import__("pathlib").Path.home() / ".cache" / "heterodyne" / "jax_cache"),
    )
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Attach to shared memory arrays (zero-copy)
    arrays: dict[str, np.ndarray | None] = {}
    for name, ref in shm_refs.items():
        shm = multiprocessing.shared_memory.SharedMemory(name=ref["shm_name"])
        _w_shm_handles.append(shm)
        arrays[name] = np.ndarray(ref["shape"], dtype=ref["dtype"], buffer=shm.buf)

    _w_phi = arrays.get("phi")
    _w_t1 = arrays.get("t1")
    _w_t2 = arrays.get("t2")
    _w_g2 = arrays.get("g2")
    _w_sigma = arrays.get("sigma")
    _w_chunk_boundaries = chunk_boundaries

    # Build JIT kernels from physics_config
    n_params: int = physics_config.get("n_params", 14)
    _w_compute_accumulators, _w_compute_chi2 = create_ooc_kernels(n_params)

    atexit.register(_ooc_worker_cleanup)


def _ooc_compute_chunk(
    args: tuple[np.ndarray, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute J^T J, J^T r, and chi2 for a single chunk.

    Parameters
    ----------
    args : (params_np, chunk_id)
        params_np: current parameter values as numpy array.
        chunk_id: index into _w_chunk_boundaries.

    Returns
    -------
    (JtJ, Jtr, chi2) as numpy arrays / float.
    """
    import jax.numpy as jnp

    params_np, chunk_id = args
    start, end = _w_chunk_boundaries[chunk_id]

    phi_c = _w_phi[start:end]  # type: ignore[index]
    t1_c = _w_t1[start:end]  # type: ignore[index]
    t2_c = _w_t2[start:end]  # type: ignore[index]
    g2_c = _w_g2[start:end]  # type: ignore[index]
    sigma_c = _w_sigma[start:end] if _w_sigma is not None else 1.0  # type: ignore[index]

    p = jnp.asarray(params_np)
    JtJ, Jtr, chi2 = _w_compute_accumulators(  # type: ignore[misc]
        p,
        jnp.asarray(phi_c),
        jnp.asarray(t1_c),
        jnp.asarray(t2_c),
        jnp.asarray(g2_c),
        jnp.asarray(sigma_c) if isinstance(sigma_c, np.ndarray) else sigma_c,
    )
    return np.asarray(JtJ), np.asarray(Jtr), float(chi2)


def _ooc_compute_chi2_chunk(
    args: tuple[np.ndarray, int],
) -> float:
    """Compute chi2 for a single chunk (no Jacobian).

    Parameters
    ----------
    args : (params_np, chunk_id)

    Returns
    -------
    chi2 as float.
    """
    import jax.numpy as jnp

    params_np, chunk_id = args
    start, end = _w_chunk_boundaries[chunk_id]

    phi_c = _w_phi[start:end]  # type: ignore[index]
    t1_c = _w_t1[start:end]  # type: ignore[index]
    t2_c = _w_t2[start:end]  # type: ignore[index]
    g2_c = _w_g2[start:end]  # type: ignore[index]
    sigma_c = _w_sigma[start:end] if _w_sigma is not None else 1.0  # type: ignore[index]

    p = jnp.asarray(params_np)
    chi2 = _w_compute_chi2(  # type: ignore[misc]
        p,
        jnp.asarray(phi_c),
        jnp.asarray(t1_c),
        jnp.asarray(t2_c),
        jnp.asarray(g2_c),
        jnp.asarray(sigma_c) if isinstance(sigma_c, np.ndarray) else sigma_c,
    )
    return float(chi2)


class OOCSharedArrays:
    """Shared memory manager for OOC flat data arrays.

    Parameters
    ----------
    phi_flat, t1_flat, t2_flat, g2_flat : np.ndarray
        Flat data arrays for the OOC iteration.
    sigma_flat : np.ndarray or None
        Per-point uncertainty weights.
    chunk_boundaries : list of (start, end) tuples
        Index boundaries for each chunk.
    """

    def __init__(
        self,
        phi_flat: np.ndarray,
        t1_flat: np.ndarray,
        t2_flat: np.ndarray,
        g2_flat: np.ndarray,
        sigma_flat: np.ndarray | None,
        chunk_boundaries: list[tuple[int, int]],
    ) -> None:
        self._shm_blocks: list[multiprocessing.shared_memory.SharedMemory] = []
        self._refs: dict[str, dict[str, Any]] = {}
        self._chunk_boundaries = chunk_boundaries

        self._create_shm("phi", phi_flat)
        self._create_shm("t1", t1_flat)
        self._create_shm("t2", t2_flat)
        self._create_shm("g2", g2_flat)
        if sigma_flat is not None:
            self._create_shm("sigma", sigma_flat)

    def _create_shm(self, name: str, arr: np.ndarray) -> None:
        """Allocate shared memory and copy array into it."""
        arr = np.ascontiguousarray(arr)
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=arr.nbytes)
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        buf[:] = arr
        self._shm_blocks.append(shm)
        self._refs[name] = {
            "shm_name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }

    def get_refs(self) -> dict[str, dict[str, Any]]:
        """Get picklable shared memory references."""
        return self._refs

    def cleanup(self) -> None:
        """Close and unlink all shared memory blocks."""
        for shm in self._shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except (OSError, ValueError):
                pass
        self._shm_blocks.clear()

    def __enter__(self) -> OOCSharedArrays:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


class OOCComputePool:
    """Persistent process pool for parallel OOC chunk computation.

    Workers share flat data arrays via shared memory and cache JIT kernels.
    The pool persists across L-M iterations (JIT compile once, reuse).

    Parameters
    ----------
    n_workers : int
        Number of parallel compute workers.
    shared_arrays : OOCSharedArrays
        Shared memory manager with data arrays.
    physics_config : dict
        Physics constants for JIT kernel creation.
    chunk_boundaries : list of (start, end)
        Index boundaries for each chunk.
    threads_per_worker : int
        OMP/MKL threads per worker process.
    """

    def __init__(
        self,
        n_workers: int,
        shared_arrays: OOCSharedArrays,
        physics_config: dict[str, Any],
        chunk_boundaries: list[tuple[int, int]],
        threads_per_worker: int = 1,
    ) -> None:
        self._n_workers = n_workers
        self._n_chunks = len(chunk_boundaries)
        self._shutdown = False

        ctx = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_ooc_worker_init,
            initargs=(
                shared_arrays.get_refs(),
                physics_config,
                chunk_boundaries,
                threads_per_worker,
            ),
        )
        logger.info(
            "OOCComputePool started: %d workers, %d chunks",
            n_workers,
            self._n_chunks,
        )

    def compute_accumulators(
        self, params: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Dispatch all chunks to workers and collect (JtJ, Jtr, chi2) tuples.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values.

        Returns
        -------
        list of (JtJ, Jtr, chi2) tuples, one per chunk.
        """
        futures = [
            self._executor.submit(_ooc_compute_chunk, (params, chunk_id))
            for chunk_id in range(self._n_chunks)
        ]
        results: list[tuple[np.ndarray, np.ndarray, float]] = []
        for future in as_completed(futures):
            results.append(future.result(timeout=300))
        return results

    def compute_chi2(self, params: np.ndarray, stride: int = 1) -> float:
        """Dispatch chi2-only computation across workers (no Jacobian).

        Parameters
        ----------
        params : np.ndarray
            Current parameter values.
        stride : int
            Chunk stride for subsampling (1 = all chunks).

        Returns
        -------
        float
            Estimated total chi2 (scaled if stride > 1).
        """
        chunk_ids = list(range(0, self._n_chunks, stride))
        futures = [
            self._executor.submit(_ooc_compute_chi2_chunk, (params, cid))
            for cid in chunk_ids
        ]
        total_chi2 = 0.0
        for future in as_completed(futures):
            total_chi2 += future.result(timeout=300)

        if stride > 1 and len(chunk_ids) > 0:
            total_chi2 *= self._n_chunks / len(chunk_ids)
        return total_chi2

    def shutdown(self) -> None:
        """Shut down the pool. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("OOCComputePool shut down")

    def __enter__(self) -> OOCComputePool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
