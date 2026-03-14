"""Multiprocessing backend for CMC sharded MCMC execution.

This module provides parallel NUTS execution using Python's multiprocessing
module for CPU-based parallelism across CMC shards.  Each shard runs as a
separate spawned process with its own JAX initialization, avoiding JAX
shared-state issues across forked processes.

Key design decisions:
- ``mp_context="spawn"`` (not fork): JAX cannot be safely shared across
  fork.  Spawned workers re-initialize JAX from scratch.
- All NumPyro imports inside worker functions: spawn safety requires that
  no JAX/NumPyro state exists at import time in the child process.
- Shared memory for common data: ``SharedDataManager`` places config,
  parameter-space state, and per-shard arrays in shared memory once,
  avoiding redundant pickle overhead through spawn.
- LPT scheduling: shards dispatched highest-cost-first to minimize
  tail latency on identical parallel workers.
- Heartbeat thread inside each worker: emits liveness pings so the parent
  can detect frozen processes and apply ``heartbeat_timeout``.
- Adaptive polling: poll interval grows when no shard has completed
  recently, shrinking CPU overhead during long-running shards.

Optimizations carried over from homodyne v2.22.2:
- Batch PRNG key generation: pre-generate all shard keys in one JAX call.
- Per-shard shared memory (packed format): 4 segments total regardless
  of shard count, avoiding fd exhaustion.
- deque for pending shards: O(1) popleft instead of O(n) list.pop(0).
- Persistent compilation cache via ``jax.config.update`` (env var alone
  insufficient in JAX 0.8+, ``min_compile_time`` lowered to 0).

This backend is selected when ``config.num_chains >= 3``, or when
``config.backend_name == "multiprocessing"``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from heterodyne.optimization.cmc.backends.base import BackendCapabilities, CMCBackend
from heterodyne.utils.logging import get_logger, log_exception, with_context

if TYPE_CHECKING:
    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of heterodyne model parameters (14-parameter two-component model).
_N_PARAMS_HETERODYNE: int = 14

#: Memory constants for estimation.
_BYTES_PER_FLOAT64: int = 8
_CPU_MEMORY_OVERHEAD_FACTOR: float = 8.0  # Conservative: 14 params x gradient bufs
_BYTES_PER_GB: float = 1024.0**3

#: Keys for per-shard numpy arrays stored in packed shared memory.
#: ``None``-valued arrays are stored as zero-length sentinels.
_SHARD_ARRAY_KEYS: tuple[str, ...] = (
    "c2_data",
    "sigma",
    "t",
    "weights",
)

# ---------------------------------------------------------------------------
# SharedDataManager
# ---------------------------------------------------------------------------


class SharedDataManager:
    """Manages shared memory blocks for data common to all CMC shards.

    Uses ``multiprocessing.shared_memory`` to share config dicts, parameter-
    space state, initial values, and per-shard arrays across spawned worker
    processes, avoiding redundant serialisation per shard.

    Serialization note: uses ``pickle`` internally for trusted internal dicts
    only (``CMCConfig.to_dict()``, parameter-space dict).  This matches the
    existing multiprocessing behaviour which also serialises all process
    arguments.  External/untrusted data is never serialised here.

    Must be used as a context manager or ``cleanup()`` called in a
    ``finally`` block to avoid leaked shared memory segments on Linux.

    Attributes:
        _shared_blocks: All allocated ``SharedMemory`` segments.
        _refs: Named references returned to callers.
    """

    def __init__(self) -> None:
        self._shared_blocks: list[mp.shared_memory.SharedMemory] = []
        self._refs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_shared_bytes(self, name: str, data: bytes) -> dict[str, Any]:
        """Store raw bytes in a shared memory segment.

        Args:
            name: Logical name for this block (used for bookkeeping only).
            data: Bytes to copy into shared memory.

        Returns:
            Reference dict with ``shm_name``, ``size``, and ``type`` keys.
        """
        shm = mp.shared_memory.SharedMemory(create=True, size=max(1, len(data)))
        shm.buf[: len(data)] = data
        self._shared_blocks.append(shm)
        ref: dict[str, Any] = {
            "shm_name": shm.name,
            "size": len(data),
            "type": "bytes",
        }
        self._refs[name] = ref
        return ref

    def create_shared_array(self, name: str, array: np.ndarray) -> dict[str, Any]:
        """Store a numpy array in a shared memory segment.

        Args:
            name: Logical name for this block.
            array: Array to copy into shared memory (contiguous float64).

        Returns:
            Reference dict with ``shm_name``, ``shape``, ``dtype``, and
            ``type`` keys.
        """
        arr = np.ascontiguousarray(array)
        shm = mp.shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shared_arr[:] = arr
        self._shared_blocks.append(shm)
        ref: dict[str, Any] = {
            "shm_name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "type": "array",
        }
        self._refs[name] = ref
        return ref

    def create_shared_dict(self, name: str, d: dict[str, Any]) -> dict[str, Any]:
        """Serialise a trusted internal dict into shared memory.

        Only used for ``CMCConfig.to_dict()`` and parameter-space dicts.
        External/untrusted data is never passed here.

        Args:
            name: Logical name for this block.
            d: Dict to serialise into shared memory.

        Returns:
            Reference dict (same as :meth:`create_shared_bytes`).
        """
        import pickle as _pkl  # noqa: S403 — trusted internal data only  # nosec B403

        return self.create_shared_bytes(name, _pkl.dumps(d))

    def create_shared_shard_arrays(
        self,
        shard_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Place per-shard numpy arrays into packed shared memory.

        Instead of creating one ``SharedMemory`` segment per array per shard
        (``n_shards * 4`` = many file descriptors), this concatenates all
        shard arrays for each key into a single shared memory block.  Only
        ``len(_SHARD_ARRAY_KEYS)`` segments are created regardless of shard
        count.

        Args:
            shard_data_list: List of shard data dicts, each containing numpy
                arrays keyed by ``_SHARD_ARRAY_KEYS`` plus a scalar
                ``noise_scale``.  Arrays for ``sigma`` and ``weights`` may be
                ``None``, in which case a zero-length sentinel is stored.

        Returns:
            List of lightweight shard references (shm names + offsets).
            Each ref dict is small enough to serialise cheaply through spawn.
        """
        n_shards = len(shard_data_list)
        key_meta: dict[str, dict[str, Any]] = {}

        for key in _SHARD_ARRAY_KEYS:
            arrays: list[np.ndarray] = []
            sizes: list[int] = []
            dtypes: list[str] = []

            for sd in shard_data_list:
                raw = sd.get(key)
                if raw is None:
                    arr = np.empty(0, dtype=np.float64)
                else:
                    arr = np.ascontiguousarray(np.asarray(raw).ravel())
                arrays.append(arr)
                sizes.append(arr.shape[0])
                dtypes.append(str(arr.dtype))

            # Use dtype from first non-empty array, or float64 fallback
            reference_dtype = next(
                (dtypes[i] for i in range(n_shards) if sizes[i] > 0),
                "float64",
            )
            cast_arrays = [
                a.astype(reference_dtype) if a.size > 0 else a for a in arrays
            ]
            combined = (
                np.concatenate(cast_arrays)
                if any(a.size > 0 for a in cast_arrays)
                else np.empty(0, dtype=reference_dtype)
            )

            shm = mp.shared_memory.SharedMemory(
                create=True, size=max(1, combined.nbytes)
            )
            shared_arr = np.ndarray(
                combined.shape, dtype=combined.dtype, buffer=shm.buf
            )
            if combined.size > 0:
                shared_arr[:] = combined
            self._shared_blocks.append(shm)

            # Prefix-sum offsets for per-shard slicing
            offsets: list[int] = [0]
            for s in sizes[:-1]:
                offsets.append(offsets[-1] + s)

            key_meta[key] = {
                "shm_name": shm.name,
                "dtype": reference_dtype,
                "offsets": offsets,
                "sizes": sizes,
            }

        shard_refs: list[dict[str, Any]] = []
        for i in range(n_shards):
            ref: dict[str, Any] = {
                "noise_scale": shard_data_list[i].get("noise_scale", 0.1),
            }
            for key in _SHARD_ARRAY_KEYS:
                meta = key_meta[key]
                ref[key] = {
                    "shm_name": meta["shm_name"],
                    "dtype": meta["dtype"],
                    "offset": meta["offsets"][i],
                    "size": meta["sizes"][i],
                }
            shard_refs.append(ref)

        return shard_refs

    def cleanup(self) -> None:
        """Release all shared memory blocks.

        Idempotent — safe to call more than once.  Must be called in a
        ``finally`` block to avoid leaked segments.
        """
        for shm in self._shared_blocks:
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, OSError):
                pass
        self._shared_blocks.clear()
        self._refs.clear()

    def __enter__(self) -> SharedDataManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Shared-memory reconstruction helpers (called inside worker processes)
# ---------------------------------------------------------------------------


def _load_shared_bytes(ref: dict[str, Any]) -> bytes:
    """Reconstruct raw bytes from a shared memory reference."""
    shm = mp.shared_memory.SharedMemory(name=ref["shm_name"], create=False)
    try:
        data = bytes(shm.buf[: ref["size"]])
    finally:
        shm.close()
    return data


def _load_shared_dict(ref: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a trusted internal dict from a shared memory reference.

    Only called inside worker processes for ``CMCConfig`` and parameter-space
    dicts that were serialised by the parent process — never for external data.
    """
    import pickle as _pkl  # noqa: S403 — trusted internal data only  # nosec B403

    return _pkl.loads(_load_shared_bytes(ref))  # noqa: S301  # nosec B301


def _load_shared_array(ref: dict[str, Any]) -> np.ndarray:
    """Reconstruct a numpy array from a shared memory reference (copying)."""
    shm = mp.shared_memory.SharedMemory(name=ref["shm_name"], create=False)
    try:
        arr = np.ndarray(
            ref["shape"], dtype=np.dtype(ref["dtype"]), buffer=shm.buf
        ).copy()
    finally:
        shm.close()
    return arr


def _load_shared_shard_data(shard_ref: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct per-shard arrays from packed shared memory.

    Each array key maps to a single concatenated ``SharedMemory`` block
    shared across all shards.  The per-shard ref carries ``offset``
    (element index) and ``size`` (element count) to slice this shard's
    portion.  Sentinel entries with ``size == 0`` are returned as ``None``.

    Args:
        shard_ref: Lightweight shard reference created by
            :meth:`SharedDataManager.create_shared_shard_arrays`.

    Returns:
        Shard data dict with numpy arrays (copied from shared memory)
        and scalar ``noise_scale``.  Arrays that were originally ``None``
        are returned as ``None``.
    """
    shard_data: dict[str, Any] = {"noise_scale": shard_ref["noise_scale"]}

    for key in _SHARD_ARRAY_KEYS:
        arr_ref = shard_ref[key]
        size = arr_ref["size"]
        if size == 0:
            shard_data[key] = None
            continue

        shm = mp.shared_memory.SharedMemory(name=arr_ref["shm_name"], create=False)
        try:
            dtype = np.dtype(arr_ref["dtype"])
            offset = arr_ref["offset"]
            total_elements = len(shm.buf) // dtype.itemsize
            full_arr = np.ndarray((total_elements,), dtype=dtype, buffer=shm.buf)
            arr = full_arr[offset : offset + size].copy()
        finally:
            shm.close()
        shard_data[key] = arr

    return shard_data


# ---------------------------------------------------------------------------
# PRNG key helpers
# ---------------------------------------------------------------------------


def _generate_shard_keys(n_shards: int, seed: int = 42) -> list[tuple[int, ...]]:
    """Pre-generate all shard PRNG keys in a single JAX call.

    Amortises JAX compilation overhead across all shards by generating
    keys in the parent process before spawning workers.

    Args:
        n_shards: Number of shards to generate keys for.
        seed: Base seed for PRNG key generation.

    Returns:
        List of raw ``uint32`` tuples that can be passed through spawn
        and reconstructed via ``jax.random.wrap_key_data`` in workers.
    """
    import jax
    import jax.numpy as jnp

    base_key = jax.random.PRNGKey(seed)
    all_keys = jax.random.split(base_key, n_shards + 1)
    shard_keys = all_keys[1:]

    key_tuples: list[tuple[int, ...]] = []
    for key in shard_keys:
        raw = jax.random.key_data(key).flatten().astype(jnp.uint32)
        key_tuples.append(tuple(int(x) for x in raw))

    return key_tuples


# ---------------------------------------------------------------------------
# LPT scheduling
# ---------------------------------------------------------------------------


def _compute_lpt_schedule(
    shard_data_list: list[dict[str, Any]],
) -> deque[int]:
    """Order shard indices by descending estimated cost (LPT heuristic).

    Cost = ``n_points * (1 + normalised_noise)``, where noise is linearly
    scaled to ``[0, 1]`` across shards.  Dispatching the most expensive
    shards first minimises tail latency on identical parallel workers.

    Args:
        shard_data_list: Shard dicts with ``"c2_data"`` (array or ``None``)
            and ``"noise_scale"`` (float).

    Returns:
        Shard indices sorted by descending cost as a ``deque`` for O(1)
        ``popleft``.
    """
    n_shards = len(shard_data_list)
    sizes: list[int] = []
    for i in range(n_shards):
        c2 = shard_data_list[i].get("c2_data")
        sizes.append(len(c2) if c2 is not None else 1)

    noises = [
        float(shard_data_list[i].get("noise_scale", 0.1)) for i in range(n_shards)
    ]

    max_noise = max(noises) if noises else 1.0
    min_noise = min(noises) if noises else 0.0
    noise_range = max_noise - min_noise

    if noise_range > 0.0:
        costs = [
            sizes[i] * (1.0 + (noises[i] - min_noise) / noise_range)
            for i in range(n_shards)
        ]
    else:
        costs = [float(s) for s in sizes]

    return deque(sorted(range(n_shards), key=lambda i: costs[i], reverse=True))


# ---------------------------------------------------------------------------
# Worker-process helpers
# ---------------------------------------------------------------------------


def _get_physical_cores() -> int:
    """Return physical core count, falling back to ``os.cpu_count() // 2``."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical is not None:
            return physical
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 1) // 2)


def _compute_threads_per_worker(total_threads: int, workers: int) -> int:
    """Derive a conservative per-worker thread budget to avoid oversubscription.

    Uses physical cores (not logical) as the safe pool.

    Args:
        total_threads: Total logical thread count available.
        workers: Number of concurrent worker processes.

    Returns:
        Number of threads to allocate per worker (minimum 1).
    """
    physical_cores = _get_physical_cores()
    safe_pool = max(1, min(total_threads, physical_cores))
    worker_count = max(1, workers)
    return max(1, safe_pool // worker_count)


def _estimate_shard_time(
    n_data: int,
    n_params: int,
    n_samples: int,
) -> float:
    """Rough estimate of per-shard wall-clock time in seconds.

    Uses a simple linear model calibrated on 14-parameter NUTS runs:
    approximately 0.5 ms per data point per sample, times a log factor
    for parameter count.

    Args:
        n_data: Number of data points in the shard.
        n_params: Number of varying model parameters.
        n_samples: Total MCMC draws (warmup + samples).

    Returns:
        Estimated duration in seconds (lower bound; actual cost varies).
    """
    import math

    base_s_per_point_per_sample = 5e-4
    param_factor = math.log1p(n_params) / math.log1p(_N_PARAMS_HETERODYNE)
    return base_s_per_point_per_sample * n_data * n_samples * param_factor


def _validate_worker_result(result: dict[str, Any]) -> None:
    """Validate that a worker result dict is internally consistent.

    Checks for non-finite sample values and shape consistency across
    parameters.  Raises ``ValueError`` on failure so the caller can mark
    the shard as failed rather than propagating corrupt data into the
    consensus step.

    Args:
        result: Worker result dict (as returned by ``_run_shard_worker``).

    Raises:
        ValueError: If samples contain NaN/Inf or shapes are inconsistent.
    """
    samples: dict[str, np.ndarray] = result.get("samples", {})
    if not samples:
        raise ValueError("Worker result contains no samples dict")

    expected_shape: tuple[int, ...] | None = None
    for name, arr in samples.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"Sample array for '{name}' is not a numpy array "
                f"(got {type(arr).__name__})"
            )
        nan_count = int(np.sum(~np.isfinite(arr)))
        if nan_count > 0:
            raise ValueError(
                f"Sample array for '{name}' contains {nan_count} non-finite values"
            )
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: '{name}' has shape {arr.shape}, "
                f"expected {expected_shape}"
            )


def _init_worker_jax(threads_per_worker: int, num_chains: int) -> None:
    """Per-worker JAX initialisation called before any JAX/NumPyro imports.

    Configures XLA/OpenMP environment variables and calls
    ``jax.config.update`` to enable float64 and the persistent
    compilation cache.

    Args:
        threads_per_worker: Number of OpenMP/MKL threads to allocate.
        num_chains: Number of MCMC chains (sets XLA virtual device count).
    """
    import re as _re

    # Thread pinning — avoid oversubscription across concurrent workers.
    # CRITICAL: clear OMP_PROC_BIND / OMP_PLACES to prevent all workers
    # competing for the same physical cores (massive contention on NUMA).
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_per_worker)
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)

    # Enable float64 BEFORE importing JAX.  Spawned workers start fresh
    # processes and do not inherit the parent's jax.config state.
    os.environ["JAX_ENABLE_X64"] = "true"

    # Persistent compilation cache so later workers reuse compiled XLA
    # programs from the first worker (JAX 0.8+: env var alone insufficient).
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(os.path.expanduser("~/.cache/heterodyne/jax_cache"))),
    )
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    # Set XLA virtual device count to num_chains so parallel chain_method
    # works correctly with multiple virtual CPU devices.
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    _xla_flags = _re.sub(r"--xla_force_host_platform_device_count=\d+", "", _xla_flags)
    os.environ["XLA_FLAGS"] = (
        _xla_flags.strip() + f" --xla_force_host_platform_device_count={num_chains}"
    )

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# ---------------------------------------------------------------------------
# Core shard worker (runs entirely inside the spawned child process)
# ---------------------------------------------------------------------------


def _run_shard_worker(
    shard_idx: int,
    shard_data: dict[str, Any],
    config_dict: dict[str, Any],
    initial_values: dict[str, Any] | None,
    result_queue: mp.Queue | None,
    rng_key_tuple: tuple[int, ...] | None,
) -> dict[str, Any]:
    """Run NUTS on a single data shard inside a spawned worker process.

    All imports of JAX and NumPyro occur here, after ``_init_worker_jax``
    has already set environment variables.

    Emits periodic heartbeat messages to ``result_queue`` via a background
    thread so the parent can detect frozen workers and apply
    ``heartbeat_timeout``.

    Args:
        shard_idx: Shard index, used for logging and result identification.
        shard_data: Dict with ``c2_data``, ``sigma``, ``t``, ``weights``,
            ``noise_scale``, ``q``, ``dt``, ``phi_angle``, ``contrast``,
            ``offset``, optionally ``reparam_config_dict``,
            ``parameter_space_dict``, and ``n_phi``.
        config_dict: Serialised ``CMCConfig`` (from ``CMCConfig.to_dict()``).
        initial_values: Optional NLSQ warm-start values (parameter name to
            float).
        result_queue: Multiprocessing queue for heartbeat and result
            delivery.  May be ``None`` in unit-test contexts.
        rng_key_tuple: Pre-generated PRNG key as raw ``uint32`` tuple.
            Falls back to ``PRNGKey(42 + shard_idx)`` when ``None``.

    Returns:
        Result dict with keys: ``type``, ``success``, ``shard_idx``,
        ``samples``, ``n_chains``, ``n_samples``, ``param_names``,
        ``extra_fields``, ``duration``, ``stats``.  On failure: ``error``,
        ``error_category``, ``traceback``.
    """
    # All JAX/NumPyro imports are deferred to here for spawn safety.
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer import initialization as numpyro_init

    from heterodyne.config.parameter_names import ALL_PARAM_NAMES
    from heterodyne.core.jax_backend import compute_c2_heterodyne
    from heterodyne.optimization.cmc.config import CMCConfig

    start_time = time.perf_counter()
    worker_logger = get_logger(
        __name__,
        context={"run": config_dict.get("run_id"), "shard": shard_idx},
    )

    n_points = (
        len(shard_data["c2_data"]) if shard_data.get("c2_data") is not None else 0
    )
    worker_logger.info(
        "Shard %d starting: %d points",
        shard_idx,
        n_points,
    )

    # ------------------------------------------------------------------
    # Heartbeat thread — emits liveness pings to the parent queue
    # ------------------------------------------------------------------
    stop_hb = threading.Event()
    heartbeat_interval = 30.0

    def _heartbeat_loop() -> None:
        while True:
            # Wait for stop signal or timeout
            if stop_hb.wait(timeout=heartbeat_interval):
                break
            payload: dict[str, Any] = {
                "type": "heartbeat",
                "shard_idx": shard_idx,
                "elapsed": time.perf_counter() - start_time,
            }
            if result_queue is not None:
                try:
                    result_queue.put_nowait(payload)
                except Exception:  # noqa: BLE001 — best-effort heartbeat
                    pass

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

    try:
        config = CMCConfig.from_dict(config_dict)

        # Reconstruct PRNG key from raw uint32 tuple
        if rng_key_tuple is not None:
            rng_key = jax.random.wrap_key_data(
                jnp.array(rng_key_tuple, dtype=jnp.uint32)
            )
        else:
            rng_key = jax.random.PRNGKey(42 + shard_idx)

        # Convert shard arrays to JAX
        c2_jax = jnp.asarray(shard_data["c2_data"])
        t_raw = shard_data.get("t")
        t_jax: jnp.ndarray | None = jnp.asarray(t_raw) if t_raw is not None else None

        sigma_raw = shard_data.get("sigma")
        if sigma_raw is not None:
            sigma_jax: jnp.ndarray | float = jnp.asarray(sigma_raw)
        else:
            # Estimate sigma from data MAD as a fallback
            median_val = float(jnp.median(c2_jax))
            mad = float(jnp.median(jnp.abs(c2_jax - median_val)))
            sigma_jax = max(mad * 1.4826, 1e-6)

        q_val: float = float(shard_data.get("q", 1.0))
        dt_val: float = float(shard_data.get("dt", 1e-3))
        phi_angle: float = float(shard_data.get("phi_angle", 0.0))
        contrast: float = float(shard_data.get("contrast", 1.0))
        offset: float = float(shard_data.get("offset", 1.0))

        # Reconstruct reparameterisation config if serialised
        reparam_config = None
        reparam_config_dict = shard_data.get("reparam_config_dict")
        if reparam_config_dict is not None:
            from heterodyne.optimization.cmc.reparameterization import ReparamConfig

            reparam_config = ReparamConfig(**reparam_config_dict)

        # Reconstruct parameter space
        from heterodyne.config.parameter_space import ParameterSpace

        ps_dict: dict[str, Any] = shard_data.get("parameter_space_dict") or {}
        try:
            parameter_space = ParameterSpace.from_config(config_dict=ps_dict)
        except Exception as exc:  # noqa: BLE001 — fall back to defaults
            logger.warning(
                "ParameterSpace.from_config failed (%s); falling back to defaults. "
                "Worker priors/bounds may differ from parent config.",
                exc,
            )
            parameter_space = ParameterSpace.default()

        # ------------------------------------------------------------------
        # Build NumPyro model (inline closure over shard-local arrays)
        # ------------------------------------------------------------------
        import numpyro
        import numpyro.distributions as dist

        varying_names = parameter_space.varying_names
        fixed_values = parameter_space.get_initial_array()

        # Warm-start init params for NumPyro
        init_params: dict[str, jnp.ndarray] | None = None
        if initial_values is not None:
            init_params = {
                k: jnp.asarray(v)
                for k, v in initial_values.items()
                if k in varying_names
            }

        def _shard_model() -> None:
            """NumPyro model for one CMC shard (14-parameter heterodyne)."""
            params = jnp.asarray(fixed_values)
            for i, name in enumerate(ALL_PARAM_NAMES):
                if name in varying_names:
                    prior = parameter_space.priors[name]
                    param = numpyro.sample(name, prior.to_numpyro(name))
                    params = params.at[i].set(param)

            # Apply reparameterised-to-physics transform when configured
            if reparam_config is not None:
                from heterodyne.optimization.cmc.reparameterization import (
                    reparam_to_physics_jax,
                )

                params = reparam_to_physics_jax(params, reparam_config)

            # Compute 14-parameter heterodyne c2 prediction.
            # t_jax, sigma_jax, c2_jax are closure-captured from the outer
            # function scope; ruff F821 cannot resolve closures statically.
            c2_model = compute_c2_heterodyne(
                params,
                t_jax,  # noqa: F821 — closure variable
                q_val,
                dt_val,
                phi_angle,
                contrast,
                offset,
            )
            numpyro.sample(  # noqa: F821 — closure variables sigma_jax, c2_jax
                "obs",
                dist.Normal(c2_model, sigma_jax),  # noqa: F821
                obs=c2_jax,  # noqa: F821
            )

        # ------------------------------------------------------------------
        # Configure NUTS / MCMC
        # ------------------------------------------------------------------
        _init_map: dict[str, Any] = {
            "init_to_median": numpyro_init.init_to_median,
            "init_to_sample": numpyro_init.init_to_sample,
            "init_to_value": numpyro_init.init_to_value,
        }
        init_strategy_name = getattr(config, "init_strategy", "init_to_median")
        init_factory = _init_map.get(init_strategy_name, numpyro_init.init_to_median)

        kernel = NUTS(
            _shard_model,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            init_strategy=init_factory(),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            chain_method="sequential",  # single process = sequential chains
            progress_bar=False,
        )

        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))

        samples_raw: dict[str, Any] = mcmc.get_samples()
        samples_np: dict[str, np.ndarray] = {
            k: np.array(v) for k, v in samples_raw.items()
        }

        extra_raw: dict[str, Any] = mcmc.get_extra_fields()
        extra_np: dict[str, np.ndarray] = {k: np.array(v) for k, v in extra_raw.items()}

        diverging = extra_np.get("diverging")
        num_divergent = int(np.sum(diverging)) if diverging is not None else 0

        duration = time.perf_counter() - start_time

        # Free large JAX arrays before serialisation to reduce peak memory
        del c2_jax, sigma_jax, t_jax, extra_raw, samples_raw
        mcmc = None  # type: ignore[assignment]

        divergence_str = f", divergences: {num_divergent}" if num_divergent > 0 else ""
        worker_logger.info(
            "Shard %d completed in %.2fs: %d samples/chain x %d chains%s",
            shard_idx,
            duration,
            config.num_samples,
            config.num_chains,
            divergence_str,
        )
        if num_divergent > 0:
            worker_logger.warning(
                "Shard %d had %d divergent transitions", shard_idx, num_divergent
            )

        return {
            "type": "result",
            "success": True,
            "shard_idx": shard_idx,
            "samples": samples_np,
            "param_names": list(samples_np.keys()),
            "n_chains": config.num_chains,
            "n_samples": config.num_samples,
            "extra_fields": extra_np,
            "duration": duration,
            "stats": {
                "num_divergent": num_divergent,
                "n_warmup": config.num_warmup,
                "n_samples": config.num_samples,
            },
        }

    except Exception as exc:  # noqa: BLE001 — top-level worker; must convert any crash to result dict
        import traceback as _tb

        duration = time.perf_counter() - start_time
        error_str = str(exc).lower()
        if "nan" in error_str or "inf" in error_str or "singular" in error_str:
            error_category = "numerical"
        elif "convergence" in error_str or "diverge" in error_str:
            error_category = "convergence"
        elif "memory" in error_str:
            error_category = "memory_error"
        else:
            error_category = "sampling"

        log_exception(
            worker_logger,
            exc,
            context={
                "shard_idx": shard_idx,
                "duration_s": round(duration, 2),
                "error_category": error_category,
                "n_points": n_points,
            },
        )

        return {
            "type": "result",
            "success": False,
            "shard_idx": shard_idx,
            "error": str(exc),
            "error_category": error_category,
            "traceback": _tb.format_exc(),
            "duration": duration,
        }

    finally:
        stop_hb.set()
        hb_thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Top-level worker entry point (module-level for spawn pickling)
# ---------------------------------------------------------------------------


def _run_shard_worker_with_queue(
    shard_idx: int,
    shard_ref: dict[str, Any],
    config_ref: dict[str, Any],
    shared_kwargs_ref: dict[str, Any],
    initial_values_ref: dict[str, Any] | None,
    ps_ref: dict[str, Any],
    threads_per_worker: int,
    result_queue: mp.Queue,
    rng_key_tuple: tuple[int, ...] | None = None,
) -> None:
    """Entry point for each per-shard spawned process.

    Reconstructs all shared data from shared memory, calls
    :func:`_run_shard_worker`, and puts the result dict on ``result_queue``.
    Wraps the entire body in try/except so initialisation crashes are
    captured and reported to the parent rather than silently lost.

    This function must be defined at module level so that Python's spawn
    mechanism can pickle it when creating child processes.

    Args:
        shard_idx: Shard index.
        shard_ref: Packed shared-memory reference for per-shard arrays.
        config_ref: Shared-memory reference for ``CMCConfig`` dict.
        shared_kwargs_ref: Shared-memory reference for shared scalar kwargs.
        initial_values_ref: Shared-memory reference for NLSQ warm-start
            values, or ``None``.
        ps_ref: Shared-memory reference for parameter-space dict.
        threads_per_worker: Per-worker thread budget (sets OMP/MKL env).
        result_queue: Multiprocessing queue for result delivery.
        rng_key_tuple: Pre-generated PRNG key raw ``uint32`` tuple.
    """
    try:
        _init_worker_jax(
            threads_per_worker=threads_per_worker,
            num_chains=int(os.environ.get("HETERODYNE_CMC_NUM_CHAINS", "4")),
        )

        shard_data = _load_shared_shard_data(shard_ref)
        config_dict = _load_shared_dict(config_ref)
        shared_kwargs = _load_shared_dict(shared_kwargs_ref)
        initial_values: dict[str, Any] | None = (
            _load_shared_dict(initial_values_ref)
            if initial_values_ref is not None
            else None
        )
        ps_dict = _load_shared_dict(ps_ref)

        # Merge shared scalars and parameter-space dict into shard_data
        shard_data.update(shared_kwargs)
        shard_data["parameter_space_dict"] = ps_dict

        result = _run_shard_worker(
            shard_idx=shard_idx,
            shard_data=shard_data,
            config_dict=config_dict,
            initial_values=initial_values,
            result_queue=result_queue,
            rng_key_tuple=rng_key_tuple,
        )

    except Exception as exc:  # noqa: BLE001 — top-level worker; must convert any crash to result dict
        import traceback as _tb

        result = {
            "type": "result",
            "success": False,
            "shard_idx": shard_idx,
            "error": f"Worker initialisation failed: {exc}",
            "error_category": "init_crash",
            "traceback": _tb.format_exc(),
            "duration": 0.0,
        }

    # Best-effort delivery: drop the result if the queue is full or closed.
    try:
        result_queue.put_nowait(result)
    except Exception:  # noqa: BLE001 — best-effort delivery
        pass


# ---------------------------------------------------------------------------
# LPTScheduler
# ---------------------------------------------------------------------------


class LPTScheduler:
    """Longest Processing Time scheduler for load balancing across cores.

    Assigns shards to workers based on estimated computation time using the
    LPT (Longest Processing Time first) heuristic.  The highest-cost shards
    are dispatched first so that the remaining shards finishing last are the
    cheapest, minimising overall tail latency.

    This is a simple greedy scheduler; it does not account for real-time
    feedback about actual execution durations.

    Attributes:
        n_workers: Number of parallel workers.
        _shard_order: Deque of shard indices sorted by descending cost.
    """

    def __init__(
        self,
        shard_costs: list[float],
        n_workers: int,
    ) -> None:
        """Initialise the LPT scheduler.

        Args:
            shard_costs: Estimated cost (positive float) per shard.  Higher
                is more expensive.
            n_workers: Number of parallel workers.
        """
        if not shard_costs:
            raise ValueError("shard_costs must be non-empty")
        self.n_workers = max(1, n_workers)
        self._shard_order: deque[int] = deque(
            sorted(range(len(shard_costs)), key=lambda i: shard_costs[i], reverse=True)
        )

    @classmethod
    def from_shard_data(
        cls,
        shard_data_list: list[dict[str, Any]],
        n_workers: int,
        n_params: int = _N_PARAMS_HETERODYNE,
        n_samples: int = 1000,
    ) -> LPTScheduler:
        """Build an :class:`LPTScheduler` from raw shard data dicts.

        Cost is estimated via :func:`_estimate_shard_time` for each shard.

        Args:
            shard_data_list: Shard dicts with ``"c2_data"`` and
                ``"noise_scale"`` keys.
            n_workers: Number of parallel workers.
            n_params: Number of model parameters (default: 14).
            n_samples: Expected total MCMC draws per shard.

        Returns:
            Configured :class:`LPTScheduler`.
        """
        costs: list[float] = []
        for sd in shard_data_list:
            c2 = sd.get("c2_data")
            n_data = len(c2) if c2 is not None else 1
            costs.append(_estimate_shard_time(n_data, n_params, n_samples))
        return cls(shard_costs=costs, n_workers=n_workers)

    def next_shard(self) -> int | None:
        """Pop and return the next shard index to dispatch.

        Returns:
            Next shard index (highest remaining cost), or ``None`` when
            all shards have been dispatched.
        """
        if not self._shard_order:
            return None
        return self._shard_order.popleft()

    def remaining(self) -> int:
        """Return the number of shards not yet dispatched."""
        return len(self._shard_order)

    def as_deque(self) -> deque[int]:
        """Return the internal order deque (consumed by dispatch loop)."""
        return self._shard_order


# ---------------------------------------------------------------------------
# MultiprocessingBackend
# ---------------------------------------------------------------------------


class MultiprocessingBackend(CMCBackend):
    """CMC backend that parallelises NUTS across shards via spawned processes.

    Each shard runs as an independent Python process so that JAX is
    initialised fresh per shard — avoiding the shared-state issues that
    arise when forking a process that already has a JAX runtime loaded.

    Shared data (config, parameter space, initial values, per-shard arrays)
    is placed in ``SharedDataManager`` once in the parent and accessed via
    ``_load_shared_*`` in each child, minimising serialisation overhead
    through spawn.

    The :meth:`run` method provides the standard single-shard
    :class:`CMCBackend` contract (sequential chain execution, no subprocess
    overhead).  For multi-shard CMC, use :meth:`run_shards`, which
    orchestrates the full parallel dispatch loop.

    Attributes:
        n_workers: Number of concurrent worker processes.
        spawn_method: Multiprocessing start method (always ``"spawn"``).
        _shared_mgr: Active :class:`SharedDataManager` during
            :meth:`run_shards`; ``None`` otherwise.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        spawn_method: str = "spawn",
    ) -> None:
        """Initialise the multiprocessing backend.

        Args:
            n_workers: Number of worker processes.  Defaults to the
                estimated physical core count, capped to avoid
                oversubscription.
            spawn_method: Process start method.  Must be ``"spawn"`` for
                JAX safety.  ``"fork"`` is explicitly unsupported.

        Raises:
            ValueError: If ``spawn_method="fork"`` is requested.
        """
        if spawn_method == "fork":
            raise ValueError(
                "MultiprocessingBackend does not support spawn_method='fork'. "
                "JAX cannot be safely shared across forked processes. "
                "Use spawn_method='spawn' (default)."
            )

        if n_workers is None:
            n_workers = max(1, _get_physical_cores())
        else:
            n_workers = min(n_workers, max(1, _get_physical_cores()))

        self.n_workers: int = max(1, n_workers)
        self.spawn_method: str = spawn_method
        self._shared_mgr: SharedDataManager | None = None

        logger.debug(
            "MultiprocessingBackend: n_workers=%d, spawn_method=%s",
            self.n_workers,
            self.spawn_method,
        )

    # ------------------------------------------------------------------
    # CMCBackend abstract methods
    # ------------------------------------------------------------------

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run NUTS sampling for a single shard (standard CMCBackend contract).

        For multi-shard CMC, call :meth:`run_shards` instead.  This method
        provides API parity with :class:`CPUBackend` and :class:`PjitBackend`
        using sequential chain execution and no subprocess overhead.

        Args:
            model: NumPyro model function.
            config: CMC configuration.
            rng_key: JAX PRNG key.
            init_params: Optional per-chain initial values.

        Returns:
            Dictionary of posterior samples from all chains.

        Raises:
            RuntimeError: If MCMC sampling fails.
        """
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer import initialization as numpyro_init

        logger.info(
            "MultiprocessingBackend.run: single-shard sequential mode, "
            "%d chains (%d warmup, %d samples)",
            config.num_chains,
            config.num_warmup,
            config.num_samples,
        )

        _init_map: dict[str, Any] = {
            "init_to_median": numpyro_init.init_to_median,
            "init_to_sample": numpyro_init.init_to_sample,
            "init_to_value": numpyro_init.init_to_value,
        }
        init_factory = _init_map.get(
            getattr(config, "init_strategy", "init_to_median"),
            numpyro_init.init_to_median,
        )

        kernel = NUTS(
            model,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            dense_mass=config.dense_mass,
            init_strategy=init_factory(),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            chain_method="sequential",
            progress_bar=True,
        )
        mcmc.run(rng_key, init_params=init_params, extra_fields=("energy",))
        samples = mcmc.get_samples()
        logger.info("MultiprocessingBackend.run: sampling complete")
        return dict(samples)

    def get_capabilities(self) -> BackendCapabilities:
        """Return multiprocessing backend capabilities.

        Returns:
            :class:`BackendCapabilities` indicating sharding support,
            parallel shards equal to ``n_workers``.
        """
        return BackendCapabilities(
            supports_sharding=True,
            supports_parallel_chains=True,
            max_parallel_shards=self.n_workers,
        )

    def validate_resources(self) -> None:
        """Check that CPU resources and multiprocessing are available.

        Raises:
            RuntimeError: If no JAX CPU device is found or if the
                ``multiprocessing`` module cannot create a spawn context.
        """
        import jax

        devices = jax.devices("cpu")
        if not devices:
            raise RuntimeError("MultiprocessingBackend: no JAX CPU devices found.")

        try:
            mp.get_context(self.spawn_method)
        except ValueError as exc:
            raise RuntimeError(
                f"MultiprocessingBackend: cannot create '{self.spawn_method}' "
                f"multiprocessing context: {exc}"
            ) from exc

        logger.debug(
            "MultiprocessingBackend.validate_resources: %d CPU device(s), n_workers=%d",
            len(devices),
            self.n_workers,
        )

    def estimate_memory(
        self,
        n_data: int,
        n_params: int,
        n_chains: int,
    ) -> float:
        """Estimate peak memory for all concurrent workers combined.

        Conservative upper bound: each worker holds one chain's live state
        (params + momentum + gradients) plus the data buffer.  Workers run
        chains sequentially so ``n_chains`` does not multiply within a
        single worker.

        Args:
            n_data: Number of data points per shard.
            n_params: Number of model parameters.
            n_chains: Number of MCMC chains per shard (not used for per-worker
                estimate; included for API uniformity).

        Returns:
            Estimated peak memory in gigabytes.
        """
        state_bytes = 3 * n_params * _BYTES_PER_FLOAT64
        data_bytes = n_data * _BYTES_PER_FLOAT64
        per_worker_bytes = (state_bytes + data_bytes) * _CPU_MEMORY_OVERHEAD_FACTOR
        total_bytes = per_worker_bytes * self.n_workers
        return total_bytes / _BYTES_PER_GB

    def cleanup(self) -> None:
        """Release shared memory and any other resources.

        Idempotent — safe to call multiple times.
        """
        if self._shared_mgr is not None:
            self._shared_mgr.cleanup()
            self._shared_mgr = None
        logger.debug("MultiprocessingBackend.cleanup: complete")

    # ------------------------------------------------------------------
    # CMC sharded execution
    # ------------------------------------------------------------------

    def run_shards(
        self,
        shards: list[dict[str, Any]],
        config: CMCConfig,
        initial_values: dict[str, Any] | None = None,
        parameter_space: Any | None = None,
        progress_bar: bool = True,
    ) -> list[dict[str, Any]]:
        """Run NUTS in parallel across all CMC shards.

        Orchestrates the full parallel dispatch loop:

        1. Allocate shared memory for config, parameter space, initial
           values, and per-shard arrays.
        2. Pre-generate all PRNG keys in the parent process.
        3. Dispatch shards to worker processes in LPT order.
        4. Drain the result queue with adaptive polling.
        5. Enforce per-shard and heartbeat timeouts.
        6. Validate and return successful shard results.

        Args:
            shards: List of shard dicts.  Each must contain at minimum
                ``c2_data`` (numpy array).  Optional keys: ``sigma``,
                ``t``, ``weights``, ``noise_scale``, ``q``, ``dt``,
                ``phi_angle``, ``contrast``, ``offset``,
                ``reparam_config_dict``.
            config: CMC configuration with NUTS hyperparameters and
                timeout settings.
            initial_values: Optional NLSQ warm-start values shared
                across all shards.
            parameter_space: Optional :class:`ParameterSpace` instance.
                Its internal config dict is serialised into shared memory.
            progress_bar: Whether to show a tqdm progress bar.

        Returns:
            List of validated successful result dicts, one per succeeded
            shard.  Each dict contains ``shard_idx``, ``samples``,
            ``n_chains``, ``n_samples``, ``param_names``, ``extra_fields``,
            ``duration``, and ``stats``.

        Raises:
            ValueError: If ``shards`` is empty.
            RuntimeError: If all shards fail, or if the success rate falls
                below ``config.min_success_rate``.
        """
        if not shards:
            raise ValueError("run_shards: shards list must be non-empty")

        n_shards = len(shards)
        actual_workers = min(self.n_workers, n_shards)
        total_threads = mp.cpu_count() or 1
        threads_per_worker = _compute_threads_per_worker(total_threads, actual_workers)

        run_logger = with_context(
            logger,
            run=getattr(config, "run_id", None),
            backend="multiprocessing",
        )
        run_logger.info(
            "run_shards: %d shards, %d workers, %d threads/worker",
            n_shards,
            actual_workers,
            threads_per_worker,
        )
        run_logger.info(
            "Per-shard timeout: %ds, heartbeat timeout: %ds",
            config.per_shard_timeout,
            config.heartbeat_timeout,
        )

        # ---------------------------------------------------------- #
        # Serialise shared data into shared memory
        # ---------------------------------------------------------- #
        config_dict = config.to_dict()

        if parameter_space is not None and hasattr(parameter_space, "_config_dict"):
            ps_dict: dict[str, Any] = parameter_space._config_dict
        else:
            ps_dict = {}
            run_logger.warning(
                "ParameterSpace._config_dict not available; workers will use "
                "default parameter bounds (may produce unconstrained proposals)"
            )

        # Shared scalars extracted from the first shard (same for all shards
        # in a homogeneous CMC split).
        _first = shards[0]
        shared_kwargs: dict[str, Any] = {
            "q": _first.get("q", 1.0),
            "dt": _first.get("dt", 1e-3),
            "phi_angle": _first.get("phi_angle", 0.0),
            "contrast": _first.get("contrast", 1.0),
            "offset": _first.get("offset", 1.0),
            "n_phi": _first.get("n_phi", 1),
            "reparam_config_dict": _first.get("reparam_config_dict"),
        }

        # Build per-shard numpy dicts for shared memory packing
        shard_data_list: list[dict[str, Any]] = []
        for shard in shards:
            c2 = shard.get("c2_data")
            shard_data_list.append(
                {
                    "c2_data": np.asarray(c2) if c2 is not None else None,
                    "sigma": (
                        np.asarray(shard["sigma"])
                        if shard.get("sigma") is not None
                        else None
                    ),
                    "t": (
                        np.asarray(shard["t"]) if shard.get("t") is not None else None
                    ),
                    "weights": (
                        np.asarray(shard["weights"])
                        if shard.get("weights") is not None
                        else None
                    ),
                    "noise_scale": float(shard.get("noise_scale", 0.1)),
                }
            )

        shared_mgr = SharedDataManager()
        self._shared_mgr = shared_mgr

        try:
            shared_config_ref = shared_mgr.create_shared_dict("config", config_dict)
            shared_ps_ref = shared_mgr.create_shared_dict("ps", ps_dict)
            shared_kwargs_ref = shared_mgr.create_shared_dict("kwargs", shared_kwargs)
            shared_iv_ref: dict[str, Any] | None = None
            if initial_values is not None:
                shared_iv_ref = shared_mgr.create_shared_dict(
                    "init_vals", initial_values
                )
            shared_shard_refs = shared_mgr.create_shared_shard_arrays(shard_data_list)
        except Exception:  # noqa: BLE001 — cleanup-and-reraise; must run shared_mgr.cleanup() on any failure
            shared_mgr.cleanup()
            self._shared_mgr = None
            raise

        # Free numpy copies after they are copied into shared memory
        del shard_data_list

        # Sentinel variables (defined before try so finally never NameErrors)
        _saved_env: dict[str, str | None] = {}
        active_processes: dict[int, tuple[mp.Process, float]] = {}
        pbar = None

        try:
            run_logger.debug(
                "Shared memory allocated: %d blocks",
                len(shared_mgr._shared_blocks),
            )

            # Pre-generate PRNG keys in parent (batch optimisation)
            seed = config.seed if config.seed is not None else 42
            shard_keys = _generate_shard_keys(n_shards, seed=seed)
            run_logger.debug("Pre-generated %d PRNG keys (seed=%d)", n_shards, seed)

            ctx = mp.get_context(self.spawn_method)
            result_queue: mp.Queue = ctx.Queue()

            # Temporarily override env for spawned workers to prevent
            # thread oversubscription inherited from the parent process.
            _worker_env_overrides: dict[str, str] = {
                "OMP_NUM_THREADS": str(threads_per_worker),
                "MKL_NUM_THREADS": str(threads_per_worker),
                "OPENBLAS_NUM_THREADS": str(threads_per_worker),
                "VECLIB_MAXIMUM_THREADS": str(threads_per_worker),
                # Pass num_chains so workers set XLA device count dynamically
                "HETERODYNE_CMC_NUM_CHAINS": str(config.num_chains),
            }
            _worker_env_clear = ["OMP_PROC_BIND", "OMP_PLACES"]

            for key in _worker_env_clear:
                _saved_env[key] = os.environ.pop(key, None)
            for key, val in _worker_env_overrides.items():
                _saved_env[key] = os.environ.get(key)
                os.environ[key] = val

            # LPT scheduling: dispatch highest-cost shards first
            pending_shards = _compute_lpt_schedule(
                [
                    {
                        "c2_data": shards[i].get("c2_data"),
                        "noise_scale": float(shards[i].get("noise_scale", 0.1)),
                    }
                    for i in range(n_shards)
                ]
            )
            if n_shards > 1:
                run_logger.debug("LPT dispatch order: %s", list(pending_shards))

            results: list[dict[str, Any]] = []
            completed_count = 0
            recorded_shards: set[int] = set()
            last_heartbeat: dict[int, float] = {}
            success_count = 0

            # Early-abort: if >50% of first N shards fail, terminate early
            early_abort_threshold = 0.5
            early_abort_sample_size = min(10, n_shards)
            failure_categories: dict[str, int] = {
                "timeout": 0,
                "heartbeat_timeout": 0,
                "crash": 0,
                "numerical": 0,
                "convergence": 0,
                "memory_error": 0,
                "sampling": 0,
                "init_crash": 0,
                "unknown": 0,
            }
            early_abort_triggered = False

            pbar = tqdm(
                total=n_shards,
                desc="CMC shards",
                disable=not progress_bar,
                unit="shard",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            pbar.set_postfix_str("starting...")
            pbar.refresh()

            start_time = time.time()
            poll_interval_min = 0.5
            poll_interval_max = 5.0
            poll_interval = poll_interval_min
            last_completion_time = start_time
            status_log_interval = 300.0  # parent status log every 5 minutes
            last_status_log = start_time
            shards_launched = 0
            per_shard_timeout = config.per_shard_timeout

            while completed_count < n_shards:
                # -------------------------------------------------- #
                # Drain the result queue
                # -------------------------------------------------- #
                while True:
                    try:
                        message: dict[str, Any] = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as _qexc:  # noqa: BLE001 — best-effort queue drain; any IPC error breaks the loop
                        run_logger.warning("Queue read error: %s", _qexc)
                        break

                    msg_type = message.get("type")
                    msg_shard_idx = message.get("shard_idx")

                    if msg_type == "heartbeat" and msg_shard_idx is not None:
                        last_heartbeat[msg_shard_idx] = time.time()
                        continue

                    if msg_type == "result" or message.get("success") is not None:
                        if (
                            msg_shard_idx is not None
                            and msg_shard_idx in recorded_shards
                        ):
                            run_logger.debug(
                                "Ignoring duplicate result for shard %d",
                                msg_shard_idx,
                            )
                            continue

                        results.append(message)
                        if msg_shard_idx is not None:
                            recorded_shards.add(msg_shard_idx)
                        completed_count += 1
                        pbar.update(1)
                        last_completion_time = time.time()
                        poll_interval = poll_interval_min

                        if message.get("success"):
                            success_count += 1
                            pbar.set_postfix(
                                shard=message.get("shard_idx", "?"),
                                time=f"{message.get('duration', 0):.1f}s",
                            )
                        else:
                            category = message.get("error_category", "unknown")
                            if category in failure_categories:
                                failure_categories[category] += 1
                            else:
                                failure_categories["unknown"] += 1
                            pbar.set_postfix(
                                shard=message.get("shard_idx", "?"),
                                status="failed",
                            )

                        # Early-abort check after first N completions
                        if (
                            not early_abort_triggered
                            and completed_count >= early_abort_sample_size
                            and completed_count <= early_abort_sample_size + 2
                        ):
                            total_failures = sum(failure_categories.values())
                            failure_rate = total_failures / completed_count
                            if failure_rate > early_abort_threshold:
                                early_abort_triggered = True
                                run_logger.error(
                                    "EARLY ABORT: %.1f%% failure rate in first "
                                    "%d shards exceeds %.0f%% threshold. "
                                    "Failure breakdown: %s",
                                    failure_rate * 100,
                                    completed_count,
                                    early_abort_threshold * 100,
                                    failure_categories,
                                )
                                pending_shards.clear()
                                for _idx, (_proc, _) in list(active_processes.items()):
                                    run_logger.info(
                                        "Terminating shard %d (early abort)",
                                        _idx,
                                    )
                                    _proc.terminate()
                                    _proc.join(timeout=2)
                                    if _proc.is_alive():
                                        _proc.kill()
                                        _proc.join(timeout=1)
                                    active_processes.pop(_idx, None)

                        if msg_shard_idx in active_processes:
                            _proc, _ = active_processes.pop(msg_shard_idx)
                            if _proc.is_alive():
                                _proc.join(timeout=1)
                        continue

                    if run_logger.isEnabledFor(logging.DEBUG):
                        run_logger.debug(
                            "Ignoring unexpected queue message: %s", message
                        )

                # -------------------------------------------------- #
                # Launch new processes up to capacity
                # -------------------------------------------------- #
                while len(active_processes) < actual_workers and pending_shards:
                    next_shard_idx = pending_shards.popleft()

                    process = ctx.Process(
                        target=_run_shard_worker_with_queue,
                        args=(
                            next_shard_idx,
                            shared_shard_refs[next_shard_idx],
                            shared_config_ref,
                            shared_kwargs_ref,
                            shared_iv_ref,
                            shared_ps_ref,
                            threads_per_worker,
                            result_queue,
                            shard_keys[next_shard_idx],
                        ),
                    )
                    process.start()
                    _now = time.time()
                    active_processes[next_shard_idx] = (process, _now)
                    last_heartbeat[next_shard_idx] = _now
                    shards_launched += 1

                # -------------------------------------------------- #
                # Check process health (timeout / heartbeat / crash)
                # -------------------------------------------------- #
                for _idx, (_process, _proc_start) in list(active_processes.items()):
                    if _idx in recorded_shards:
                        del active_processes[_idx]
                        continue

                    _now = time.time()
                    _elapsed = _now - _proc_start
                    _last_active = last_heartbeat.get(_idx, _proc_start)
                    _inactive = _now - _last_active

                    if not _process.is_alive():
                        _process.join(timeout=1)
                        _exit_code = _process.exitcode
                        del active_processes[_idx]

                        if _idx not in recorded_shards:
                            if _exit_code is not None and _exit_code < 0:
                                import signal as _signal

                                try:
                                    _sig_name = _signal.Signals(-_exit_code).name
                                except ValueError:
                                    _sig_name = str(-_exit_code)
                                _err = (
                                    f"Process killed by signal {_sig_name} "
                                    f"(exit_code={_exit_code})"
                                )
                            elif _exit_code is not None and _exit_code > 0:
                                _err = (
                                    f"Process exited with error "
                                    f"(exit_code={_exit_code})"
                                )
                            else:
                                _err = "Process exited without returning a result"

                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": _idx,
                                    "error": _err,
                                    "error_category": "crash",
                                    "duration": _elapsed,
                                }
                            )
                            recorded_shards.add(_idx)
                            failure_categories["crash"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=_idx, status="no-result")

                    elif _elapsed > per_shard_timeout:
                        run_logger.warning(
                            "Shard %d exceeded runtime limit: %.0fs "
                            "(limit: %ds), terminating (pid=%s)",
                            _idx,
                            _elapsed,
                            per_shard_timeout,
                            _process.pid,
                        )
                        _process.terminate()
                        _process.join(timeout=5)
                        if _process.is_alive():
                            _process.kill()
                            _process.join(timeout=2)

                        del active_processes[_idx]
                        if _idx not in recorded_shards:
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": _idx,
                                    "error": (
                                        f"Runtime timeout after {_elapsed:.0f}s "
                                        f"(limit: {per_shard_timeout}s)"
                                    ),
                                    "error_category": "timeout",
                                    "duration": _elapsed,
                                }
                            )
                            recorded_shards.add(_idx)
                            failure_categories["timeout"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=_idx, status="timeout")

                    elif _inactive > config.heartbeat_timeout:
                        run_logger.warning(
                            "Shard %d unresponsive for %.0fs "
                            "(heartbeat timeout: %ds), terminating (pid=%s)",
                            _idx,
                            _inactive,
                            config.heartbeat_timeout,
                            _process.pid,
                        )
                        _process.terminate()
                        _process.join(timeout=5)
                        if _process.is_alive():
                            _process.kill()
                            _process.join(timeout=2)

                        del active_processes[_idx]
                        if _idx not in recorded_shards:
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": _idx,
                                    "error": (
                                        f"Unresponsive after {_inactive:.0f}s "
                                        f"(heartbeat timeout: "
                                        f"{config.heartbeat_timeout}s)"
                                    ),
                                    "error_category": "heartbeat_timeout",
                                    "duration": _elapsed,
                                }
                            )
                            recorded_shards.add(_idx)
                            failure_categories["heartbeat_timeout"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=_idx, status="frozen")

                # -------------------------------------------------- #
                # Progress bar refresh
                # -------------------------------------------------- #
                if completed_count < n_shards:
                    _elapsed_total = time.time() - start_time
                    _mins, _secs = divmod(int(_elapsed_total), 60)
                    _hrs, _mins = divmod(_mins, 60)
                    if _hrs > 0:
                        pbar.set_postfix_str(
                            f"active={len(active_processes)} "
                            f"elapsed={_hrs}h{_mins:02d}m"
                        )
                    else:
                        pbar.set_postfix_str(
                            f"active={len(active_processes)} "
                            f"elapsed={_mins}m{_secs:02d}s"
                        )

                    _ts = time.time()
                    if _ts - last_status_log >= status_log_interval:
                        _active_hb = {
                            k: f"{_ts - last_heartbeat.get(k, _ts):.0f}s"
                            for k in active_processes
                        }
                        run_logger.info(
                            "CMC status: %d/%d complete; active=%d; "
                            "launched=%d; heartbeats=%s",
                            completed_count,
                            n_shards,
                            len(active_processes),
                            shards_launched,
                            _active_hb,
                        )
                        last_status_log = _ts

                    # Adaptive poll: grow interval during slow periods
                    _since_completion = time.time() - last_completion_time
                    if _since_completion > 30.0:
                        poll_interval = min(poll_interval * 1.1, poll_interval_max)

                    time.sleep(poll_interval)

                # Orphan detection: mark stragglers when no activity remains
                if (
                    not active_processes
                    and not pending_shards
                    and completed_count < n_shards
                ):
                    _missing = set(range(n_shards)) - recorded_shards
                    for _idx in sorted(_missing):
                        results.append(
                            {
                                "success": False,
                                "shard_idx": _idx,
                                "error": "Shard exited without emitting a result",
                                "error_category": "crash",
                                "duration": None,
                            }
                        )
                        recorded_shards.add(_idx)
                        completed_count += 1
                        pbar.update(1)
                        pbar.set_postfix(shard=_idx, status="no-result")

        except KeyboardInterrupt:
            run_logger.warning("Interrupted — terminating all active processes")
            for _idx, (_process, _) in active_processes.items():
                run_logger.debug("Terminating shard %d (pid=%s)", _idx, _process.pid)
                _process.terminate()
                _process.join(timeout=2)
            raise

        finally:
            if pbar is not None:
                pbar.close()
            for _idx, (_process, _) in list(active_processes.items()):
                if _process.is_alive():
                    run_logger.warning("Cleaning up orphan process for shard %d", _idx)
                    _process.terminate()
                    _process.join(timeout=2)

            # Restore parent environment to pre-spawn state
            for _key, _val in _saved_env.items():
                if _val is None:
                    os.environ.pop(_key, None)
                else:
                    os.environ[_key] = _val

            shared_mgr.cleanup()
            self._shared_mgr = None

        # ---------------------------------------------------------- #
        # Collect and validate results
        # ---------------------------------------------------------- #
        return self._collect_results(results, n_shards, config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_shared_data(
        self,
        shards: list[dict[str, Any]],
        config_dict: dict[str, Any],
        shared_kwargs: dict[str, Any],
        initial_values: dict[str, Any] | None,
        ps_dict: dict[str, Any],
    ) -> tuple[
        SharedDataManager,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        """Create all shared memory allocations for worker data.

        Called by :meth:`run_shards` before spawning processes.

        Args:
            shards: Raw shard dicts.
            config_dict: Serialised CMCConfig.
            shared_kwargs: Shared scalar kwargs (q, dt, phi_angle, …).
            initial_values: Optional warm-start values.
            ps_dict: Parameter-space config dict.

        Returns:
            Tuple of ``(mgr, config_ref, kwargs_ref, ps_ref, iv_ref,
            shard_refs)``.
        """
        shard_data_list: list[dict[str, Any]] = []
        for shard in shards:
            c2 = shard.get("c2_data")
            shard_data_list.append(
                {
                    "c2_data": np.asarray(c2) if c2 is not None else None,
                    "sigma": (
                        np.asarray(shard["sigma"])
                        if shard.get("sigma") is not None
                        else None
                    ),
                    "t": (
                        np.asarray(shard["t"]) if shard.get("t") is not None else None
                    ),
                    "weights": (
                        np.asarray(shard["weights"])
                        if shard.get("weights") is not None
                        else None
                    ),
                    "noise_scale": float(shard.get("noise_scale", 0.1)),
                }
            )

        mgr = SharedDataManager()
        try:
            config_ref = mgr.create_shared_dict("config", config_dict)
            kwargs_ref = mgr.create_shared_dict("kwargs", shared_kwargs)
            ps_ref = mgr.create_shared_dict("ps", ps_dict)
            iv_ref: dict[str, Any] | None = None
            if initial_values is not None:
                iv_ref = mgr.create_shared_dict("init_vals", initial_values)
            shard_refs = mgr.create_shared_shard_arrays(shard_data_list)
        except Exception:  # noqa: BLE001 — cleanup-and-reraise; must run mgr.cleanup() on any failure
            mgr.cleanup()
            raise

        return mgr, config_ref, kwargs_ref, ps_ref, iv_ref, shard_refs

    def _create_worker_configs(
        self,
        shards: list[dict[str, Any]],
        config: CMCConfig,
    ) -> list[dict[str, Any]]:
        """Build per-shard lightweight dicts for LPT cost estimation.

        Args:
            shards: Raw shard dicts.
            config: CMC configuration (unused; reserved for future use).

        Returns:
            List of dicts with ``c2_data`` and ``noise_scale`` keys
            (sufficient for :func:`_compute_lpt_schedule`).
        """
        return [
            {
                "c2_data": shards[i].get("c2_data"),
                "noise_scale": float(shards[i].get("noise_scale", 0.1)),
            }
            for i in range(len(shards))
        ]

    def _collect_results(
        self,
        results: list[dict[str, Any]],
        n_shards: int,
        config: CMCConfig,
    ) -> list[dict[str, Any]]:
        """Gather and validate worker results.

        Filters failed shards, validates sample integrity, and checks
        success rate against config thresholds.

        Args:
            results: Raw result dicts from worker processes.
            n_shards: Total shard count (for rate computation).
            config: CMC configuration (carries success-rate thresholds).

        Returns:
            Validated successful result dicts.

        Raises:
            RuntimeError: If all shards fail.
        """
        successful: list[dict[str, Any]] = []
        for res in results:
            if res.get("success"):
                try:
                    _validate_worker_result(res)
                    successful.append(res)
                except ValueError as _ve:
                    logger.warning(
                        "Shard %s validation failed: %s",
                        res.get("shard_idx", "?"),
                        _ve,
                    )
            else:
                _err_cat = res.get("error_category", "unknown")
                logger.warning(
                    "Shard %s failed [%s]: %s",
                    res.get("shard_idx", "?"),
                    _err_cat,
                    res.get("error", "unknown"),
                )
                if res.get("traceback"):
                    logger.debug(
                        "Shard %s traceback:\n%s",
                        res.get("shard_idx", "?"),
                        res["traceback"],
                    )

        if not successful:
            error_categories_summary: dict[str, int] = {}
            for res in results:
                if not res.get("success"):
                    _cat = res.get("error_category", "unknown")
                    error_categories_summary[_cat] = (
                        error_categories_summary.get(_cat, 0) + 1
                    )
            raise RuntimeError(
                f"All {n_shards} shards failed. "
                f"Error categories: {error_categories_summary}"
            )

        success_rate = len(successful) / n_shards
        if success_rate < config.min_success_rate_warning:
            logger.error(
                "Success rate %.1f%% below minimum threshold %.1f%% "
                "— analysis may be unreliable",
                success_rate * 100,
                config.min_success_rate_warning * 100,
            )
        elif success_rate < config.min_success_rate:
            logger.warning(
                "Success rate %.1f%% below recommended threshold %.1f%% "
                "— consider investigating failed shards",
                success_rate * 100,
                config.min_success_rate * 100,
            )

        valid_durations = [
            res["duration"] for res in successful if res.get("duration") is not None
        ]
        if valid_durations:
            _sorted = sorted(valid_durations)
            logger.debug(
                "Shard timing: n=%d, min=%.1fs, max=%.1fs, median=%.1fs",
                len(valid_durations),
                min(valid_durations),
                max(valid_durations),
                _sorted[len(_sorted) // 2],
            )

        logger.info(
            "run_shards complete: %d/%d shards succeeded",
            len(successful),
            n_shards,
        )
        return successful

    def _handle_worker_failure(
        self,
        shard_idx: int,
        error: Exception,
    ) -> dict[str, Any]:
        """Build a failure result dict for a shard that raised an exception.

        Args:
            shard_idx: Shard index.
            error: Exception caught from the worker.

        Returns:
            Failure result dict with ``success=False``, ``shard_idx``,
            ``error``, ``error_category``, and ``duration`` keys.
        """
        import traceback as _tb

        error_str = str(error).lower()
        if "nan" in error_str or "inf" in error_str:
            category = "numerical"
        elif "memory" in error_str:
            category = "memory_error"
        elif "convergence" in error_str:
            category = "convergence"
        else:
            category = "sampling"

        return {
            "type": "result",
            "success": False,
            "shard_idx": shard_idx,
            "error": str(error),
            "error_category": category,
            "traceback": _tb.format_exc(),
            "duration": 0.0,
        }

    def is_available(self) -> bool:
        """Check whether this backend can run on the current platform.

        Returns:
            ``True`` if the spawn multiprocessing context is available.
        """
        try:
            mp.get_context(self.spawn_method)
            return True
        except (ValueError, OSError):
            return False

    def __repr__(self) -> str:
        return (
            f"MultiprocessingBackend("
            f"n_workers={self.n_workers}, "
            f"spawn_method={self.spawn_method!r})"
        )
