"""Tests for heterodyne.utils.async_io — PrefetchLoader and AsyncWriter.

Covers:
- PrefetchLoader: iteration, prefetch depth, error propagation, empty iterable
- AsyncWriter: write_npz, write_json, context manager, shutdown, error handling
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from heterodyne.utils.async_io import AsyncWriter, PrefetchLoader

# ---------------------------------------------------------------------------
# PrefetchLoader
# ---------------------------------------------------------------------------


class TestPrefetchLoader:
    """Tests for the prefetch iterator wrapper."""

    def test_iterates_all_elements(self) -> None:
        data = [1, 2, 3, 4, 5]
        result = list(PrefetchLoader(data, max_prefetch=1))
        assert result == data

    def test_empty_iterable(self) -> None:
        result = list(PrefetchLoader([], max_prefetch=1))
        assert result == []

    def test_single_element(self) -> None:
        result = list(PrefetchLoader([42], max_prefetch=1))
        assert result == [42]

    def test_max_prefetch_depth(self) -> None:
        data = list(range(100))
        result = list(PrefetchLoader(data, max_prefetch=5))
        assert result == data

    def test_invalid_max_prefetch(self) -> None:
        with pytest.raises(ValueError, match="max_prefetch must be >= 1"):
            PrefetchLoader([1, 2], max_prefetch=0)

    def test_negative_max_prefetch(self) -> None:
        with pytest.raises(ValueError, match="max_prefetch must be >= 1"):
            PrefetchLoader([1, 2], max_prefetch=-1)

    def test_preserves_order(self) -> None:
        data = [10, 20, 30, 40, 50]
        result = list(PrefetchLoader(data, max_prefetch=2))
        assert result == data

    def test_works_with_generator(self) -> None:
        def gen():
            for i in range(5):
                yield i * 10

        result = list(PrefetchLoader(gen(), max_prefetch=1))
        assert result == [0, 10, 20, 30, 40]

    def test_propagates_producer_exception(self) -> None:
        def failing_iter():
            yield 1
            yield 2
            raise RuntimeError("producer error")

        loader = PrefetchLoader(failing_iter(), max_prefetch=1)
        it = iter(loader)
        assert next(it) == 1
        assert next(it) == 2
        with pytest.raises(RuntimeError, match="producer error"):
            next(it)

    def test_iter_returns_self(self) -> None:
        loader = PrefetchLoader([1, 2])
        it = iter(loader)
        assert it is loader
        # Consume all to cleanly shut down the producer thread
        list(it)

    def test_manual_next_calls(self) -> None:
        loader = PrefetchLoader([10, 20, 30])
        it = iter(loader)
        assert next(it) == 10
        assert next(it) == 20
        assert next(it) == 30
        with pytest.raises(StopIteration):
            next(it)

    def test_numpy_arrays(self) -> None:
        arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = list(PrefetchLoader(arrays, max_prefetch=1))
        assert len(result) == 2
        npt.assert_allclose(result[0], [1.0, 2.0])
        npt.assert_allclose(result[1], [3.0, 4.0])

    def test_large_batch(self) -> None:
        data = list(range(1000))
        result = list(PrefetchLoader(data, max_prefetch=10))
        assert result == data


# ---------------------------------------------------------------------------
# AsyncWriter
# ---------------------------------------------------------------------------


class TestAsyncWriter:
    """Tests for asynchronous file writer."""

    def test_write_npz(self, tmp_path: Path) -> None:
        dest = tmp_path / "test.npz"
        arr = np.array([1.0, 2.0, 3.0])

        with AsyncWriter(max_pending=2) as writer:
            future = writer.write_npz(dest, data=arr)
            future.result(timeout=5.0)

        loaded = np.load(dest)
        npt.assert_allclose(loaded["data"], arr)

    def test_write_json(self, tmp_path: Path) -> None:
        dest = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        with AsyncWriter(max_pending=2) as writer:
            future = writer.write_json(dest, data)
            future.result(timeout=5.0)

        with dest.open() as f:
            loaded = json.load(f)
        assert loaded == data

    def test_write_npz_creates_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "nested" / "dir" / "test.npz"
        arr = np.array([1.0])

        with AsyncWriter() as writer:
            future = writer.write_npz(dest, data=arr)
            future.result(timeout=5.0)

        assert dest.exists()

    def test_write_json_creates_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "deep" / "nested" / "test.json"

        with AsyncWriter() as writer:
            future = writer.write_json(dest, {"a": 1})
            future.result(timeout=5.0)

        assert dest.exists()

    def test_multiple_writes(self, tmp_path: Path) -> None:
        with AsyncWriter(max_pending=4) as writer:
            futures = []
            for i in range(5):
                dest = tmp_path / f"batch_{i}.npz"
                arr = np.ones(10) * i
                futures.append(writer.write_npz(dest, data=arr))

            for f in futures:
                f.result(timeout=5.0)

        for i in range(5):
            loaded = np.load(tmp_path / f"batch_{i}.npz")
            npt.assert_allclose(loaded["data"], np.ones(10) * i)

    def test_context_manager(self, tmp_path: Path) -> None:
        writer = AsyncWriter()
        with writer as w:
            assert w is writer
            dest = tmp_path / "ctx.npz"
            w.write_npz(dest, x=np.array([1.0])).result(timeout=5.0)
        # After exit, writer should be closed
        assert writer._closed

    def test_shutdown_idempotent(self) -> None:
        writer = AsyncWriter()
        writer.shutdown()
        writer.shutdown()  # Should not raise

    def test_write_after_shutdown_raises(self, tmp_path: Path) -> None:
        writer = AsyncWriter()
        writer.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            writer.write_npz(tmp_path / "fail.npz", data=np.array([1.0]))

    def test_write_json_after_shutdown_raises(self, tmp_path: Path) -> None:
        writer = AsyncWriter()
        writer.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            writer.write_json(tmp_path / "fail.json", {"a": 1})

    def test_invalid_max_pending(self) -> None:
        with pytest.raises(ValueError, match="max_pending must be >= 1"):
            AsyncWriter(max_pending=0)

    def test_negative_max_pending(self) -> None:
        with pytest.raises(ValueError, match="max_pending must be >= 1"):
            AsyncWriter(max_pending=-1)

    def test_write_npz_multiple_arrays(self, tmp_path: Path) -> None:
        dest = tmp_path / "multi.npz"
        a = np.array([1.0, 2.0])
        b = np.array([[3.0, 4.0], [5.0, 6.0]])

        with AsyncWriter() as writer:
            writer.write_npz(dest, arr_a=a, arr_b=b).result(timeout=5.0)

        loaded = np.load(dest)
        npt.assert_allclose(loaded["arr_a"], a)
        npt.assert_allclose(loaded["arr_b"], b)

    def test_write_json_utf8(self, tmp_path: Path) -> None:
        dest = tmp_path / "unicode.json"
        data = {"label": "wavelength_\u00c5", "value": 1.55}

        with AsyncWriter() as writer:
            writer.write_json(dest, data).result(timeout=5.0)

        with dest.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["label"] == "wavelength_\u00c5"

    def test_future_returns_none(self, tmp_path: Path) -> None:
        dest = tmp_path / "result.npz"
        with AsyncWriter() as writer:
            future = writer.write_npz(dest, data=np.array([1.0]))
            result = future.result(timeout=5.0)
        assert result is None

    def test_shutdown_wait_false(self, tmp_path: Path) -> None:
        writer = AsyncWriter()
        # Submit a write then immediately shut down without waiting
        dest = tmp_path / "fast.npz"
        future = writer.write_npz(dest, data=np.array([1.0]))
        future.result(timeout=5.0)  # Ensure write completes before shutdown
        writer.shutdown(wait=False)
        assert writer._closed

    def test_default_max_pending(self) -> None:
        with AsyncWriter() as writer:
            # Default is 4
            assert writer._semaphore._value == 4
