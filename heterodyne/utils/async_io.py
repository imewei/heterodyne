"""Asynchronous I/O utilities for prefetching data and writing results.

Provides two primitives:

- ``PrefetchLoader``: Wraps any iterable and prefetches the next item in a
  background thread so that data loading overlaps with computation.
- ``AsyncWriter``: Dispatches ``np.savez`` / ``json.dump`` calls to a
  background thread, returning ``Future`` objects so callers can optionally
  wait for completion.

Both classes are pure Python/NumPy — no JAX dependency.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from heterodyne.utils.logging import get_logger

__all__ = ["PrefetchLoader", "AsyncWriter"]

_log = get_logger(__name__)


# Sentinel used to signal end-of-iteration inside PrefetchLoader.
_DONE = object()
# Sentinel used to propagate exceptions from the producer thread.
_ERROR = object()

# Timeout (seconds) for each queue.put() attempt in the producer loop.
# Allows the producer to check the stop-event and exit cleanly when the
# consumer breaks out early instead of blocking indefinitely on a full queue.
_PUT_TIMEOUT = 0.1


class PrefetchLoader[T]:
    """Prefetches the next item from an iterable in a background thread.

    The background thread stays exactly one item ahead of the consumer.
    This hides I/O latency (disk reads, network, preprocessing) behind
    the computation that processes the current item.

    Parameters
    ----------
    iterable:
        Any finite iterable whose ``__next__`` may block (e.g. a dataset
        loader, a file-chunk reader).
    max_prefetch:
        Depth of the internal queue.  ``1`` means one item is prefetched
        while the current item is being processed.  Larger values trade
        memory for higher throughput on bursty sources.

    Examples
    --------
    >>> for batch in PrefetchLoader(my_dataset, max_prefetch=2):
    ...     model(batch)

    Or as a context manager to guarantee deterministic cleanup on early break::

    >>> with PrefetchLoader(my_dataset, max_prefetch=2) as loader:
    ...     for batch in loader:
    ...         if early_stop_condition:
    ...             break
    """

    def __init__(self, iterable: Iterable[T], max_prefetch: int = 1) -> None:
        if max_prefetch < 1:
            raise ValueError(f"max_prefetch must be >= 1, got {max_prefetch}")
        self._iterable = iterable
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_prefetch)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch"
        )
        self._future: Future[None] | None = None
        self._started = False
        # Stop event lets the producer exit cleanly when the consumer breaks
        # early; without this, a full queue would block the producer forever.
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        self._start_producer()
        return self

    def __next__(self) -> T:
        if not self._started:
            self._start_producer()

        item = self._queue.get()

        if item is _DONE:
            self._shutdown()
            raise StopIteration

        if item is _ERROR:
            exc: BaseException = self._queue.get()
            self._shutdown()
            raise exc

        return item  # type: ignore[no-any-return]  # queue typed Any; caller gets T via PrefetchLoader[T]

    # ------------------------------------------------------------------
    # Context manager (preferred for loops that may break early)
    # ------------------------------------------------------------------

    def __enter__(self) -> PrefetchLoader[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Signal the producer to stop and release resources.

        Safe to call multiple times.  Called automatically by ``__exit__``;
        also called internally after normal or error exhaustion of the source.
        """
        self._stop_event.set()
        # Drain the queue so the producer can unblock from a put() and notice
        # the stop event.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_producer(self) -> None:
        if self._started:
            return
        self._started = True
        self._future = self._executor.submit(self._produce)

    def _produce(self) -> None:
        try:
            for item in self._iterable:
                # Use a timed put so the producer can react to stop_event
                # when the consumer breaks out early and the queue is full.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(item, timeout=_PUT_TIMEOUT)
                        break
                    except queue.Full:
                        continue
                else:
                    return  # consumer signalled stop; exit without _DONE
            if not self._stop_event.is_set():
                self._queue.put(_DONE)
        except Exception as exc:  # noqa: BLE001
            _log.debug("PrefetchLoader producer raised %s: %s", type(exc).__name__, exc)
            if not self._stop_event.is_set():
                self._queue.put(_ERROR)
                self._queue.put(exc)

    def _shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            # Best-effort cleanup; deterministic cleanup requires close() or
            # the context manager.
            self._stop_event.set()
            self._shutdown()
        except Exception:  # noqa: BLE001
            pass


class AsyncWriter:
    """Asynchronous file writer for NumPy arrays and JSON data.

    All writes are dispatched to a single background thread, keeping the
    main computation loop unblocked.  Callers receive a ``Future`` they
    can optionally ``.result()`` on to confirm completion or surface errors.

    Parameters
    ----------
    max_pending:
        Maximum number of write tasks queued before the submitter blocks.
        Prevents unbounded memory growth when writes are slower than
        submissions.

    Examples
    --------
    >>> with AsyncWriter(max_pending=4) as writer:
    ...     writer.write_npz("results/batch_0.npz", g2=arr)
    ...     writer.write_json("results/meta.json", {"q": 0.01})
    """

    def __init__(self, max_pending: int = 4) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}")
        self._semaphore = threading.Semaphore(max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="async_writer"
        )
        self._closed = False
        # Guards _closed + submit to prevent semaphore leak when shutdown()
        # races with write_npz()/write_json(): acquire → shutdown → submit fails
        # → semaphore never released.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def write_npz(self, path: str | Path, **arrays: np.ndarray) -> Future[None]:
        """Write *arrays* to *path* as a compressed ``.npz`` file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created
            automatically.
        **arrays:
            Keyword-argument arrays forwarded to ``np.savez_compressed``.

        Returns
        -------
        Future
            Resolves to ``None`` on success; raises on write failure.
        """
        dest = Path(path)
        self._semaphore.acquire()
        try:
            with self._lock:
                self._check_open()
                future = self._executor.submit(self._do_write_npz, dest, arrays)
        except Exception:
            self._semaphore.release()
            raise
        future.add_done_callback(lambda _: self._semaphore.release())
        return future

    def write_json(self, path: str | Path, data: dict[str, Any]) -> Future[None]:
        """Serialise *data* to *path* as a UTF-8 JSON file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created
            automatically.
        data:
            A JSON-serialisable dictionary.

        Returns
        -------
        Future
            Resolves to ``None`` on success; raises on serialisation or
            write failure.
        """
        dest = Path(path)
        self._semaphore.acquire()
        try:
            with self._lock:
                self._check_open()
                future = self._executor.submit(self._do_write_json, dest, data)
        except Exception:
            self._semaphore.release()
            raise
        future.add_done_callback(lambda _: self._semaphore.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background executor.

        Parameters
        ----------
        wait:
            If ``True`` (default), block until all pending writes complete.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        _log.debug("AsyncWriter shut down (wait=%s)", wait)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AsyncWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Private write implementations (run in background thread)
    # ------------------------------------------------------------------

    def _do_write_npz(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy stubs incorrectly type **kwargs as bool
        _log.debug("AsyncWriter wrote npz: %s", path)

    def _do_write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        _log.debug("AsyncWriter wrote json: %s", path)

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "AsyncWriter has been shut down and cannot accept new writes."
            )
