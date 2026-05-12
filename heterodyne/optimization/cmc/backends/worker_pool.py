"""Worker pool backend for multi-shard CMC execution.

Spawns persistent workers that each run MCMC on assigned shards.
Amortizes JAX/NumPyro initialization overhead across tasks.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import multiprocessing.process
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    import jax.numpy as jnp

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)


def _estimate_physical_workers() -> int:
    """Estimate optimal worker count from physical core topology.

    Uses ``detect_cpu_info()`` for accurate physical core detection
    (lscpu on Linux, sysctl on macOS), reserving one core for the
    main process. Falls back to ``os.cpu_count() // 2`` if detection
    fails or returns suspicious values.

    Returns:
        Number of worker processes (>= 1).
    """
    try:
        from heterodyne.device.cpu import detect_cpu_info

        info = detect_cpu_info()
        physical = info.physical_cores
        # Sanity: physical should be <= logical and >= 1
        if 1 <= physical <= info.logical_cores:
            return max(1, physical - 1)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: assume hyperthreading (2 threads/core)
    logical = os.cpu_count() or 2
    physical_estimate = max(1, logical // 2)
    return max(1, physical_estimate - 1)


def _run_shard_worker(
    model_fn: Callable[..., Any],
    config_dict: dict[str, Any],
    shard_data: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Per-shard MCMC worker function.

    Runs in a subprocess. Imports NumPyro locally to avoid
    fork-safety issues with JAX.

    Args:
        model_fn: NumPyro model factory (must be picklable).
        config_dict: Serialized CMCConfig.
        shard_data: Data for this shard.
        seed: Random seed for this shard.

    Returns:
        Dictionary of posterior samples.
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    # Ensure CPU backend in spawned worker process
    jax.config.update("jax_platform_name", "cpu")

    rng_key = jax.random.PRNGKey(seed)

    kernel = NUTS(
        model_fn,
        target_accept_prob=config_dict["target_accept_prob"],
        max_tree_depth=config_dict.get("max_tree_depth", 10),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=config_dict["num_warmup"],
        num_samples=config_dict["num_samples"],
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )

    mcmc.run(rng_key)
    return dict(mcmc.get_samples())


class WorkerPoolBackend:
    """Persistent worker pool for multi-shard CMC execution.

    Distributes MCMC shards across a pool of worker processes.
    Each worker runs one chain per shard, and results are combined.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        """Initialize with optional worker count.

        Args:
            n_workers: Number of workers. Defaults to physical_cores - 1
                (reserving one core for the main process). Uses
                ``detect_cpu_info()`` for accurate physical core detection;
                falls back to ``os.cpu_count() // 2`` if detection fails.
        """
        if n_workers is None:
            n_workers = _estimate_physical_workers()
        self._n_workers = max(1, n_workers)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def get_name(self) -> str:
        return "worker_pool"

    @staticmethod
    def should_use_pool(n_shards: int, n_workers: int) -> bool:
        """Check if pool execution is beneficial.

        Homodyne parity: pool is used whenever there are at least 3 shards,
        regardless of the worker count.  The pool's startup cost is amortised
        across shards, not workers, so the gate only needs to reject trivially
        small shard counts.

        Args:
            n_shards: Number of data shards.
            n_workers: Available workers (unused; kept for API stability).

        Returns:
            True when parallelism would be beneficial.
        """
        _ = n_workers
        return n_shards >= 3

    def run(
        self,
        model: Callable[..., Any],
        config: CMCConfig,
        rng_key: jnp.ndarray,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run MCMC via worker pool.

        Args:
            model: NumPyro model function.
            config: CMC configuration.
            rng_key: JAX PRNG key (used to generate per-shard seeds).
            init_params: Optional initial values (not used in pool mode).

        Returns:
            Combined posterior samples from all workers.
        """
        import jax
        import jax.numpy as jnp

        n_chains = config.num_chains

        logger.info(
            "WorkerPoolBackend: dispatching %d chains across %d workers",
            n_chains,
            self._n_workers,
        )

        config_dict = {
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "target_accept_prob": config.target_accept_prob,
            "max_tree_depth": config.max_tree_depth,
        }

        # Generate deterministic seeds per chain
        seeds = [
            int(
                jax.random.randint(
                    jax.random.fold_in(rng_key, i),
                    (),
                    0,
                    2**31 - 1,
                )
            )
            for i in range(n_chains)
        ]

        with ProcessPoolExecutor(
            max_workers=self._n_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(_run_shard_worker, model, config_dict, {}, seed)
                for seed in seeds
            ]

            all_samples: dict[str, list[Any]] = {}
            for future in futures:
                samples = future.result()
                for name, arr in samples.items():
                    all_samples.setdefault(name, []).append(arr)

        combined = {name: jnp.concatenate(arrs) for name, arrs in all_samples.items()}

        logger.info("WorkerPoolBackend: combined %d chains", n_chains)
        return combined


# ---------------------------------------------------------------------------
# PersistentWorkerPool — homodyne CMC parity (queue-based event loop)
# ---------------------------------------------------------------------------
#
# ``WorkerPoolBackend`` above wraps :class:`ProcessPoolExecutor` and only
# amortises the worker process startup cost.  The class below mirrors
# homodyne's :class:`WorkerPool` architecture: each worker runs an event
# loop with per-worker task queues, a ready-signal handshake, a bounded
# shared result queue, round-robin task submit, and a defensive
# MemoryError drain so an OOM in one worker cannot deadlock the parent.


def should_use_persistent_pool(n_shards: int, n_workers: int) -> bool:
    """Homodyne-parity gate: persistent pool helps when there are >=3 shards."""
    _ = n_workers
    return n_shards >= 3


class PersistentWorkerPool:
    """Persistent process pool for CMC shard dispatch (homodyne parity).

    Workers are spawned once, perform a one-time initialization (e.g.
    JAX warm-up / JIT priming) under control of ``worker_init_fn``,
    signal readiness, and then process tasks from a per-worker queue
    until a ``None`` sentinel is received.  The pool exposes a
    round-robin :meth:`submit`, a blocking :meth:`get_result`, and a
    deterministic :meth:`shutdown` that drains task queues, joins
    processes, and terminates any that refuse to exit.

    The class supports the context-manager protocol; ``__exit__`` calls
    :meth:`shutdown`.

    Args:
        n_workers: Number of persistent worker processes.
        worker_fn: Picklable module-level callable invoked per task.
            Signature ``worker_fn(task: dict, **init_kwargs) -> dict |
            None``.  Returning ``None`` skips putting a result on the
            shared queue, which is useful when ``worker_fn`` already
            manages its own result emission.
        worker_init_kwargs: One-time kwargs forwarded to both
            ``worker_init_fn`` (if provided) and every ``worker_fn``
            call.  Must be picklable.
        worker_init_fn: Optional one-time initializer with signature
            ``init_fn(worker_id: int, **init_kwargs) -> None``.  Use it
            for expensive setup like JAX backend selection or JIT
            pre-warming.
        startup_timeout: Maximum seconds to wait for all workers to
            signal readiness before continuing with whatever subset is
            ready.
    """

    def __init__(
        self,
        n_workers: int,
        worker_fn: Callable[..., dict[str, Any] | None],
        worker_init_kwargs: dict[str, Any] | None = None,
        worker_init_fn: Callable[..., None] | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        self._n_workers = max(1, n_workers)
        self._worker_fn = worker_fn
        self._init_kwargs = dict(worker_init_kwargs or {})
        self._init_fn = worker_init_fn
        self._startup_timeout = startup_timeout

        ctx = multiprocessing.get_context("spawn")
        self._task_queues: list[multiprocessing.Queue] = [
            ctx.Queue() for _ in range(self._n_workers)
        ]
        self._result_queue: multiprocessing.Queue = ctx.Queue(
            maxsize=self._n_workers * 4
        )
        self._processes: list[Any] = []
        self._next_worker = 0
        self._alive = False
        self._lock = threading.Lock()

        self._start_workers(ctx)

    def _start_workers(self, ctx: Any) -> None:
        ready_queue: multiprocessing.Queue = ctx.Queue()
        for i in range(self._n_workers):
            p = ctx.Process(
                target=_persistent_worker_event_loop,
                args=(
                    i,
                    self._task_queues[i],
                    self._result_queue,
                    self._worker_fn,
                    self._init_kwargs,
                    self._init_fn,
                    ready_queue,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        ready_count = 0
        for _ in range(self._n_workers):
            try:
                ready_queue.get(timeout=self._startup_timeout)
                ready_count += 1
            except queue.Empty:
                logger.warning(
                    "PersistentWorkerPool startup timed out after %.0fs (%d/%d ready)",
                    self._startup_timeout,
                    ready_count,
                    self._n_workers,
                )
                break
        try:
            ready_queue.close()
        except (OSError, ValueError):
            pass

        self._alive = True
        logger.info(
            "PersistentWorkerPool started: %d/%d workers ready",
            ready_count,
            self._n_workers,
        )

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def result_queue(self) -> multiprocessing.Queue:
        return self._result_queue

    def is_alive(self) -> bool:
        return self._alive and any(p.is_alive() for p in self._processes)

    def submit(self, task: dict[str, Any]) -> None:
        """Round-robin submit ``task`` to the next worker's queue."""
        with self._lock:
            worker_idx = self._next_worker % self._n_workers
            self._task_queues[worker_idx].put(task)
            self._next_worker += 1

    def get_result(self, timeout: float = 300.0) -> dict[str, Any]:
        """Block until a result is available; raises ``queue.Empty`` on timeout."""
        return cast("dict[str, Any]", self._result_queue.get(timeout=timeout))

    def results_pending(self) -> bool:
        """True when the shared result queue has at least one entry."""
        return not self._result_queue.empty()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Send ``None`` sentinels to all workers and join them."""
        if not self._alive:
            return
        for tq in self._task_queues:
            try:
                tq.put(None)
            except (OSError, ValueError):
                pass
        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning(
                    "PersistentWorkerPool: worker %d did not exit cleanly; terminating",
                    p.pid,
                )
                p.terminate()
                p.join(timeout=15)
                if p.is_alive():
                    p.kill()
        for tq in self._task_queues:
            try:
                tq.close()
            except (OSError, ValueError):
                pass
        try:
            self._result_queue.close()
        except (OSError, ValueError):
            pass
        self._alive = False
        logger.info("PersistentWorkerPool shut down")

    def __enter__(self) -> PersistentWorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _persistent_worker_event_loop(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_fn: Callable[..., dict[str, Any] | None],
    init_kwargs: dict[str, Any],
    init_fn: Callable[..., None] | None,
    ready_queue: multiprocessing.Queue | None,
) -> None:
    """Persistent worker event loop — homodyne-parity.

    Calls ``init_fn`` once (if provided), signals ``ready_queue`` on
    successful initialisation, then loops processing tasks until a
    ``None`` sentinel arrives.  Exceptions are caught and surfaced as
    failure-result payloads on the shared result queue.  On
    ``MemoryError``, the worker drains its remaining task queue with
    failure payloads so the parent's blocking ``get_result()`` won't
    hang waiting for results that will never come.
    """
    try:
        if init_fn is not None:
            init_fn(worker_id, **init_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "PersistentWorkerPool worker %d init_fn raised: %s", worker_id, exc
        )
        if ready_queue is not None:
            try:
                ready_queue.put(worker_id)  # still signal so parent doesn't hang
            except (OSError, ValueError):
                pass
        return

    if ready_queue is not None:
        try:
            ready_queue.put(worker_id)
        except (OSError, ValueError):
            pass

    while True:
        try:
            task = task_queue.get()
        except (OSError, EOFError):
            break
        if task is None:
            break

        task_id = task.get("task_id", "unknown")
        try:
            result = worker_fn(task, **init_kwargs)
            if result is not None:
                result["worker_id"] = os.getpid()
                result_queue.put(result)
        except MemoryError:
            result_queue.put(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": "MemoryError: worker ran out of memory",
                    "worker_id": os.getpid(),
                }
            )
            # Drain any remaining queued tasks so parent get_result doesn't hang.
            while True:
                try:
                    remaining = task_queue.get(block=False)
                    if remaining is None:
                        break
                    result_queue.put(
                        {
                            "task_id": remaining.get("task_id", "unknown"),
                            "success": False,
                            "error": "Worker terminated due to MemoryError",
                            "worker_id": os.getpid(),
                        }
                    )
                except (queue.Empty, queue.Full, EOFError):
                    break
            break
        except Exception as exc:  # noqa: BLE001 — surface any error as a result
            result_queue.put(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "worker_id": os.getpid(),
                }
            )
