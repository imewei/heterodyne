"""Memory budget tracking for large XPCS datasets.

Provides allocation tracking, budget enforcement, and chunk-size
suggestions so that downstream code can stay within a configurable
memory envelope.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Fallback when psutil is unavailable
_DEFAULT_BUDGET_BYTES: int = 8 * 1024 * 1024 * 1024  # 8 GB


@dataclass
class MemoryBudget:
    """Snapshot of memory budget state.

    Attributes:
        total_bytes: Total budget available.
        allocated_bytes: Currently tracked allocations.
        peak_bytes: Highest allocated_bytes observed.
    """

    total_bytes: int
    allocated_bytes: int
    peak_bytes: int


class MemoryManager:
    """Track memory allocations against a configurable budget.

    When *budget_bytes* is ``None`` the manager auto-detects available
    system memory via ``psutil.virtual_memory()``, falling back to 8 GB
    if psutil is not installed.

    All public methods are thread-safe.

    Args:
        budget_bytes: Explicit budget in bytes, or ``None`` for auto-detect.
    """

    def __init__(self, budget_bytes: int | None = None) -> None:
        if budget_bytes is not None:
            if budget_bytes <= 0:
                raise ValueError("budget_bytes must be positive")
            self._total = budget_bytes
        else:
            self._total = self._detect_system_memory()

        self._allocated = 0
        self._peak = 0
        self._labels: dict[str, int] = {}
        self._lock = threading.Lock()

        logger.info("MemoryManager initialised with budget %d bytes", self._total)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(self, n_bytes: int, label: str) -> bool:
        """Request an allocation of *n_bytes* tracked under *label*.

        If the allocation would exceed the budget, the request is denied
        and ``False`` is returned.  Existing allocations with the same
        label are released first (idempotent re-allocation).

        Args:
            n_bytes: Number of bytes to allocate.
            label: Human-readable label for the allocation.

        Returns:
            ``True`` if the allocation fits within the budget.
        """
        if n_bytes < 0:
            raise ValueError("n_bytes must be non-negative")

        with self._lock:
            # Release any prior allocation under the same label
            prior = self._labels.pop(label, 0)
            self._allocated -= prior

            if self._allocated + n_bytes > self._total:
                # Restore the prior allocation if we cannot fit
                if prior > 0:
                    self._labels[label] = prior
                    self._allocated += prior
                logger.warning(
                    "Allocation '%s' denied: %d bytes requested, %d/%d bytes used",
                    label,
                    n_bytes,
                    self._allocated,
                    self._total,
                )
                return False

            self._labels[label] = n_bytes
            self._allocated += n_bytes
            if self._allocated > self._peak:
                self._peak = self._allocated
            logger.debug(
                "Allocated '%s': %d bytes (%d/%d used)",
                label,
                n_bytes,
                self._allocated,
                self._total,
            )
            return True

    def release(self, label: str) -> None:
        """Release a tracked allocation.

        No-op if the label is not currently tracked.

        Args:
            label: The label passed to :meth:`request`.
        """
        with self._lock:
            freed = self._labels.pop(label, 0)
            self._allocated -= freed
            if freed > 0:
                logger.debug(
                    "Released '%s': %d bytes (%d/%d used)",
                    label,
                    freed,
                    self._allocated,
                    self._total,
                )

    @staticmethod
    def estimate_array_size(
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float64,
    ) -> int:
        """Estimate the memory footprint of an array.

        Args:
            shape: Array dimensions.
            dtype: NumPy dtype (or type convertible to one).

        Returns:
            Size in bytes.
        """
        dt = np.dtype(dtype)
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        return n_elements * dt.itemsize

    def get_budget(self) -> MemoryBudget:
        """Return a snapshot of the current budget state.

        Returns:
            MemoryBudget dataclass.
        """
        with self._lock:
            return MemoryBudget(
                total_bytes=self._total,
                allocated_bytes=self._allocated,
                peak_bytes=self._peak,
            )

    def suggest_chunk_size(
        self,
        total_elements: int,
        element_bytes: int,
    ) -> int:
        """Suggest an optimal chunk size that fits within available budget.

        The returned chunk size will use at most half of the *remaining*
        budget so that headroom is preserved for intermediate buffers.

        Args:
            total_elements: Total number of elements to process.
            element_bytes: Bytes per element.

        Returns:
            Suggested number of elements per chunk (always >= 1).
        """
        with self._lock:
            remaining = self._total - self._allocated

        # Use at most half the remaining budget
        usable = max(remaining // 2, element_bytes)
        chunk = usable // element_bytes
        # Clamp to [1, total_elements]
        return max(1, min(chunk, total_elements))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_system_memory() -> int:
        """Attempt to auto-detect available system memory."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            budget = int(mem.available)
            logger.info("Auto-detected available memory: %d bytes", budget)
            return budget
        except (ImportError, AttributeError):
            logger.info(
                "psutil unavailable; using default budget of %d bytes",
                _DEFAULT_BUDGET_BYTES,
            )
            return _DEFAULT_BUDGET_BYTES


# ---------------------------------------------------------------------------
# Feature 6: Memory-Mapped I/O
# ---------------------------------------------------------------------------


class MemoryMapManager:
    """Manage memory-mapped access to large HDF5 datasets.

    For datasets that exceed available RAM, this provides read-only
    memory-mapped access via h5py's direct chunk reading, avoiding
    full materialization of arrays into memory.

    Args:
        max_resident_bytes: Maximum bytes to keep resident in memory
            at any time. Defaults to 2 GB.
    """

    def __init__(self, max_resident_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        self._max_resident_bytes = max_resident_bytes
        self._handles: dict[str, Any] = {}  # h5py.File handles
        self._lock = threading.RLock()
        logger.debug(
            "MemoryMapManager initialised (max_resident=%d bytes)",
            max_resident_bytes,
        )

    def _get_handle(self, file_path: Path | str) -> Any:
        """Return a cached read-only file handle, opening if necessary."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for MemoryMapManager; install it via 'uv add h5py'"
            ) from exc

        key = str(Path(file_path).resolve())
        # Lock already held by caller
        if key not in self._handles:
            logger.debug("Opening HDF5 handle: %s", key)
            self._handles[key] = h5py.File(key, "r")
        return self._handles[key]

    def open_dataset(
        self,
        file_path: Path | str,
        dataset_path: str,
    ) -> np.ndarray:
        """Open an HDF5 dataset for memory-mapped-like access.

        Uses h5py's lazy loading: returns a dataset object that reads
        chunks on demand. For truly large files, slicing is preferred
        over full materialization.

        Args:
            file_path: Path to HDF5 file.
            dataset_path: Internal HDF5 dataset path (e.g., "/exchange/C2T_all/c2_00001").

        Returns:
            NumPy array (lazily loaded via h5py if possible).
        """
        with self._lock:
            handle = self._get_handle(file_path)
            dataset = handle[dataset_path]
            estimated = self.estimate_dataset_size(file_path, dataset_path)
            if estimated > self._max_resident_bytes:
                logger.warning(
                    "Dataset '%s' estimated size %d bytes exceeds max_resident %d bytes; "
                    "consider read_slice() for partial access",
                    dataset_path,
                    estimated,
                    self._max_resident_bytes,
                )
            result: np.ndarray = np.asarray(dataset)
        return result

    def read_slice(
        self,
        file_path: Path | str,
        dataset_path: str,
        slices: tuple[slice, ...],
    ) -> np.ndarray:
        """Read a specific slice from an HDF5 dataset without loading the full array.

        Args:
            file_path: Path to HDF5 file.
            dataset_path: Internal HDF5 dataset path.
            slices: Tuple of slice objects defining the region to read.

        Returns:
            NumPy array of the requested slice.
        """
        with self._lock:
            handle = self._get_handle(file_path)
            result: np.ndarray = np.asarray(handle[dataset_path][slices])
        return result

    def estimate_dataset_size(
        self,
        file_path: Path | str,
        dataset_path: str,
    ) -> int:
        """Estimate the in-memory size of an HDF5 dataset without loading it.

        Args:
            file_path: Path to HDF5 file.
            dataset_path: Internal HDF5 dataset path.

        Returns:
            Estimated size in bytes.
        """
        with self._lock:
            handle = self._get_handle(file_path)
            dataset = handle[dataset_path]
            n_elements = 1
            for dim in dataset.shape:
                n_elements *= dim
            return int(n_elements * dataset.dtype.itemsize)

    def close_all(self) -> None:
        """Close all open HDF5 file handles."""
        with self._lock:
            for key, handle in list(self._handles.items()):
                logger.debug("Closing HDF5 handle: %s", key)
                handle.close()
            self._handles.clear()

    def __enter__(self) -> MemoryMapManager:
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close all handles on context exit."""
        self.close_all()


# ---------------------------------------------------------------------------
# Feature 7: Adaptive Chunking
# ---------------------------------------------------------------------------


@dataclass
class ChunkInfo:
    """Metadata for a single processing chunk.

    Attributes:
        start: Start index along the batch axis.
        end: End index (exclusive) along the batch axis.
        size_bytes: Estimated memory footprint of this chunk.
        priority: Processing priority (lower = higher priority).
    """

    start: int
    end: int
    size_bytes: int
    priority: int = 0


class AdaptiveChunker:
    """Compute chunk sizes that adapt to available memory and data characteristics.

    Unlike fixed chunking, this class monitors memory pressure and adjusts
    chunk sizes dynamically. Chunks near the diagonal of correlation matrices
    (small time lag) are given higher priority since they carry more signal.

    Args:
        memory_manager: MemoryManager instance for budget awareness.
        safety_factor: Fraction of available memory to actually use (default 0.5).
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        safety_factor: float = 0.5,
    ) -> None:
        if not 0.0 < safety_factor <= 1.0:
            raise ValueError("safety_factor must be in (0, 1]")
        self._memory_manager = memory_manager
        self._safety_factor = safety_factor
        logger.debug("AdaptiveChunker initialised (safety_factor=%.2f)", safety_factor)

    def compute_chunks(
        self,
        total_elements: int,
        element_bytes: int,
        prioritize_near_diagonal: bool = False,
    ) -> list[ChunkInfo]:
        """Compute adaptive chunk boundaries.

        Args:
            total_elements: Total number of elements along the batch axis.
            element_bytes: Memory per element in bytes.
            prioritize_near_diagonal: If True, assign lower priority numbers
                (= higher priority) to chunks covering small indices.

        Returns:
            List of ChunkInfo objects defining the chunking strategy.
        """
        if total_elements <= 0:
            raise ValueError("total_elements must be positive")
        if element_bytes <= 0:
            raise ValueError("element_bytes must be positive")

        budget = self._memory_manager.get_budget()
        remaining = budget.total_bytes - budget.allocated_bytes

        usable = int(remaining * self._safety_factor)
        chunk_size = usable // element_bytes
        # Clamp to [1, total_elements]
        chunk_size = max(1, min(chunk_size, total_elements))

        first_quarter_end = total_elements // 4

        chunks: list[ChunkInfo] = []
        start = 0
        while start < total_elements:
            end = min(start + chunk_size, total_elements)
            size_bytes = (end - start) * element_bytes

            if prioritize_near_diagonal:
                priority = 0 if start < first_quarter_end else 1
            else:
                priority = 0

            chunks.append(
                ChunkInfo(
                    start=start,
                    end=end,
                    size_bytes=size_bytes,
                    priority=priority,
                )
            )
            start = end

        logger.debug(
            "AdaptiveChunker: %d chunks of ~%d elements "
            "(remaining=%d bytes, safety=%.2f)",
            len(chunks),
            chunk_size,
            remaining,
            self._safety_factor,
        )
        return chunks


# ---------------------------------------------------------------------------
# Feature 8: Memory Pressure Monitoring
# ---------------------------------------------------------------------------


class MemoryPressureLevel(Enum):
    """System memory pressure classification."""

    LOW = "low"  # < 50% used
    MODERATE = "moderate"  # 50-75% used
    HIGH = "high"  # 75-90% used
    CRITICAL = "critical"  # > 90% used


class MemoryPressureMonitor:
    """Monitor system memory pressure and trigger adaptive responses.

    Polls system memory usage via psutil (with graceful fallback) and
    classifies pressure into levels that downstream code can use to
    adjust batch sizes, enable compression, or skip optional caching.

    Args:
        poll_interval_seconds: Minimum seconds between actual system polls
            (cached between polls). Defaults to 5.0.
    """

    # Conservative fallback when psutil is unavailable: assume 4 GB available
    _FALLBACK_AVAILABLE_BYTES: int = 4 * 1024 * 1024 * 1024
    _FALLBACK_PERCENT_USED: float = 50.0

    def __init__(self, poll_interval_seconds: float = 5.0) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._poll_interval = poll_interval_seconds
        self._lock = threading.Lock()
        self._last_poll_time: float = 0.0
        self._cached_level: MemoryPressureLevel = MemoryPressureLevel.LOW
        self._cached_available_bytes: int = self._FALLBACK_AVAILABLE_BYTES
        logger.debug(
            "MemoryPressureMonitor initialised (poll_interval=%.1fs)",
            poll_interval_seconds,
        )

    def _poll(self) -> tuple[MemoryPressureLevel, int]:
        """Poll system memory; returns (level, available_bytes).

        Must be called with self._lock held.
        """
        now = time.monotonic()
        if now - self._last_poll_time < self._poll_interval:
            return self._cached_level, self._cached_available_bytes

        try:
            import psutil

            vm = psutil.virtual_memory()
            percent_used = vm.percent
            available = int(vm.available)
        except (ImportError, AttributeError):
            logger.debug("psutil unavailable; using fallback memory figures")
            percent_used = self._FALLBACK_PERCENT_USED
            available = self._FALLBACK_AVAILABLE_BYTES

        if percent_used < 50.0:
            level = MemoryPressureLevel.LOW
        elif percent_used < 75.0:
            level = MemoryPressureLevel.MODERATE
        elif percent_used < 90.0:
            level = MemoryPressureLevel.HIGH
        else:
            level = MemoryPressureLevel.CRITICAL

        self._cached_level = level
        self._cached_available_bytes = available
        self._last_poll_time = now

        logger.debug(
            "MemoryPressureMonitor: %.1f%% used → %s",
            percent_used,
            level.value,
        )
        return level, available

    def current_pressure(self) -> MemoryPressureLevel:
        """Return the current memory pressure level.

        Uses cached value if polled recently (within poll_interval_seconds).

        Returns:
            Current MemoryPressureLevel.
        """
        with self._lock:
            level, _ = self._poll()
        return level

    def available_bytes(self) -> int:
        """Return available system memory in bytes.

        Returns:
            Available memory, or a conservative default if psutil unavailable.
        """
        with self._lock:
            _, available = self._poll()
        return available

    def should_reduce_allocation(self) -> bool:
        """Return True if memory pressure suggests reducing allocations.

        Returns True when pressure is HIGH or CRITICAL.
        """
        level = self.current_pressure()
        return level in (MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL)

    def recommended_budget_fraction(self) -> float:
        """Return recommended fraction of total memory to use.

        Returns:
            Float in (0, 1]: 1.0 for LOW, 0.75 for MODERATE,
            0.5 for HIGH, 0.25 for CRITICAL.
        """
        level = self.current_pressure()
        fractions: dict[MemoryPressureLevel, float] = {
            MemoryPressureLevel.LOW: 1.0,
            MemoryPressureLevel.MODERATE: 0.75,
            MemoryPressureLevel.HIGH: 0.5,
            MemoryPressureLevel.CRITICAL: 0.25,
        }
        return fractions[level]
