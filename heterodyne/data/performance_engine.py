"""Caching and lazy loading for XPCS datasets.

Provides an LRU-eviction cache backed by a thread-safe lock, designed for
host-side NumPy arrays that may be repeatedly accessed during iterative
analysis workflows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cached dataset entry.

    Attributes:
        data: The cached NumPy array.
        timestamp: Time of last access (seconds since epoch).
        access_count: Number of times this entry has been accessed.
        size_bytes: Size of the cached array in bytes.
    """

    data: np.ndarray
    timestamp: float
    access_count: int
    size_bytes: int


class PerformanceEngine:
    """LRU-eviction cache for NumPy datasets.

    Thread-safe via ``threading.Lock``.  When the total cache size exceeds
    ``max_cache_bytes``, the least-recently-used entries are evicted until
    the new dataset fits.

    Args:
        max_cache_bytes: Maximum cache budget in bytes (default 1 GB).
    """

    def __init__(self, max_cache_bytes: int = 1_000_000_000) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        self._max_bytes = max_cache_bytes
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cache_dataset(self, key: str, data: np.ndarray) -> None:
        """Store an array in the cache, evicting LRU entries if needed.

        Args:
            key: Unique identifier for the dataset.
            data: NumPy array to cache.
        """
        size = int(data.nbytes)
        if size > self._max_bytes:
            logger.warning(
                "Array for '%s' (%d bytes) exceeds max cache size (%d bytes); "
                "not caching",
                key,
                size,
                self._max_bytes,
            )
            return

        with self._lock:
            # If the key already exists, remove the old entry first
            if key in self._cache:
                del self._cache[key]

            self._evict_until_fits(size)
            self._cache[key] = CacheEntry(
                data=data,
                timestamp=time.monotonic(),
                access_count=1,
                size_bytes=size,
            )
            logger.debug("Cached '%s' (%d bytes)", key, size)

    def get_cached(self, key: str) -> np.ndarray | None:
        """Retrieve a cached array.

        Returns ``None`` on a cache miss.  On a hit the access count and
        timestamp are updated.

        Args:
            key: Dataset identifier.

        Returns:
            The cached array, or ``None`` if not present.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            entry.timestamp = time.monotonic()
            entry.access_count += 1
            self._hits += 1
            return entry.data

    def invalidate(self, key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            key: Dataset identifier to remove.
        """
        with self._lock:
            removed = self._cache.pop(key, None)
            if removed is not None:
                logger.debug("Invalidated '%s' (%d bytes)", key, removed.size_bytes)

    def clear_cache(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            logger.debug("Cleared cache (%d entries)", n)

    def get_stats(self) -> dict[str, int | float]:
        """Return cache statistics.

        Returns:
            Dict with keys: ``total_size``, ``n_entries``, ``max_bytes``,
            ``hits``, ``misses``, ``hit_rate``.
        """
        with self._lock:
            total_size = sum(e.size_bytes for e in self._cache.values())
            n_entries = len(self._cache)
            total_lookups = self._hits + self._misses
            hit_rate = self._hits / total_lookups if total_lookups > 0 else 0.0
            return {
                "total_size": total_size,
                "n_entries": n_entries,
                "max_bytes": self._max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_size(self) -> int:
        """Total bytes currently cached (caller must hold lock)."""
        return sum(e.size_bytes for e in self._cache.values())

    def _evict_until_fits(self, needed: int) -> None:
        """Evict LRU entries until *needed* bytes can fit (caller must hold lock)."""
        while self._current_size() + needed > self._max_bytes and self._cache:
            # Find the entry with the oldest timestamp (LRU)
            lru_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            evicted = self._cache.pop(lru_key)
            logger.debug(
                "Evicted '%s' (%d bytes) to make room",
                lru_key,
                evicted.size_bytes,
            )
