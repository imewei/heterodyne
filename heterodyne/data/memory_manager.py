"""Memory budget tracking for large XPCS datasets.

Provides allocation tracking, budget enforcement, and chunk-size
suggestions so that downstream code can stay within a configurable
memory envelope.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

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
                    "Allocation '%s' denied: %d bytes requested, "
                    "%d/%d bytes used",
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
