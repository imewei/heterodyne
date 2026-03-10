"""Caching and lazy loading for XPCS datasets.

Provides an LRU-eviction cache backed by a thread-safe lock, designed for
host-side NumPy arrays that may be repeatedly accessed during iterative
analysis workflows.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Tiered cache (L1 memory + L2 disk)
# ---------------------------------------------------------------------------


@dataclass
class TieredCacheConfig:
    """Configuration for multi-level cache.

    Attributes:
        memory_max_bytes: Maximum size for in-memory (L1) cache.
        disk_cache_dir: Directory for disk-based (L2) cache. If None, disk
            caching is disabled.
        disk_max_bytes: Maximum size for disk cache.
        compression: Whether to compress disk-cached arrays.
    """

    memory_max_bytes: int = 1_000_000_000  # 1 GB
    disk_cache_dir: Path | None = None
    disk_max_bytes: int = 10_000_000_000  # 10 GB
    compression: bool = True


class TieredCache:
    """Two-level cache: fast in-memory L1 + persistent disk L2.

    On a miss in L1, checks L2 (disk). On a miss in both, returns None.
    When storing, data goes to both L1 and L2. L1 uses LRU eviction
    (delegated to PerformanceEngine). L2 stores arrays as compressed NPZ
    files on disk.

    Args:
        config: Tiered cache configuration.
    """

    def __init__(self, config: TieredCacheConfig | None = None) -> None:
        self._config = config or TieredCacheConfig()
        self._l1 = PerformanceEngine(max_cache_bytes=self._config.memory_max_bytes)
        self._lock = threading.Lock()
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

        if self._config.disk_cache_dir is not None:
            self._config.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "TieredCache: disk cache enabled at %s (max %d bytes)",
                self._config.disk_cache_dir,
                self._config.disk_max_bytes,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> np.ndarray | None:
        """Look up a cached array, checking L1 then L2.

        Args:
            key: Cache key.

        Returns:
            Cached array or None on miss.
        """
        safe_key = self._sanitize_key(key)

        # L1 check (PerformanceEngine is already thread-safe)
        result = self._l1.get_cached(safe_key)
        if result is not None:
            with self._lock:
                self._l1_hits += 1
            logger.debug("TieredCache L1 hit for '%s'", key)
            return result

        # L2 check
        disk_path = self._disk_path(safe_key)
        if disk_path is not None and disk_path.exists():
            try:
                loaded = np.load(disk_path, allow_pickle=False)  # noqa: S301
                data: np.ndarray = loaded["data"]
                # Promote to L1
                self._l1.cache_dataset(safe_key, data)
                with self._lock:
                    self._l2_hits += 1
                logger.debug("TieredCache L2 hit for '%s'; promoted to L1", key)
                return data
            except Exception:
                logger.warning(
                    "TieredCache: failed to load disk cache for '%s'; treating as miss",
                    key,
                    exc_info=True,
                )

        with self._lock:
            self._misses += 1
        logger.debug("TieredCache miss for '%s'", key)
        return None

    def put(self, key: str, data: np.ndarray) -> None:
        """Store an array in both L1 and L2 caches.

        Args:
            key: Cache key.
            data: Array to cache.
        """
        safe_key = self._sanitize_key(key)

        # Write to L1
        self._l1.cache_dataset(safe_key, data)

        # Write to L2
        disk_path = self._disk_path(safe_key)
        if disk_path is not None:
            self._write_disk(disk_path, data)

    def invalidate(self, key: str) -> None:
        """Remove a key from both cache levels."""
        safe_key = self._sanitize_key(key)
        self._l1.invalidate(safe_key)
        disk_path = self._disk_path(safe_key)
        if disk_path is not None and disk_path.exists():
            try:
                disk_path.unlink()
                logger.debug("TieredCache: removed disk entry for '%s'", key)
            except OSError:
                logger.warning(
                    "TieredCache: could not remove disk entry for '%s'", key
                )

    def clear(self) -> None:
        """Clear both cache levels."""
        self._l1.clear_cache()
        disk_dir = self._config.disk_cache_dir
        if disk_dir is not None and disk_dir.exists():
            removed = 0
            for npz_file in disk_dir.glob("*.npz"):
                try:
                    npz_file.unlink()
                    removed += 1
                except OSError:
                    pass
            logger.debug("TieredCache: cleared %d disk entries", removed)
        with self._lock:
            self._l1_hits = 0
            self._l2_hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Return combined cache statistics."""
        l1_stats = self._l1.get_stats()
        with self._lock:
            l1_hits = self._l1_hits
            l2_hits = self._l2_hits
            misses = self._misses

        total_lookups = l1_hits + l2_hits + misses
        hit_rate = (l1_hits + l2_hits) / total_lookups if total_lookups > 0 else 0.0

        disk_size_bytes = 0
        disk_dir = self._config.disk_cache_dir
        if disk_dir is not None and disk_dir.exists():
            disk_size_bytes = sum(
                f.stat().st_size for f in disk_dir.glob("*.npz") if f.is_file()
            )

        return {
            "l1_hits": l1_hits,
            "l2_hits": l2_hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "l1_total_size": l1_stats["total_size"],
            "l1_n_entries": l1_stats["n_entries"],
            "l1_max_bytes": l1_stats["max_bytes"],
            "l2_disk_size_bytes": disk_size_bytes,
            "l2_disk_max_bytes": self._config.disk_max_bytes,
            "disk_cache_enabled": disk_dir is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Map arbitrary cache keys to collision-free filesystem-safe names.

        Uses a SHA-256 hash to avoid collisions between keys that differ
        only in characters like ``/`` vs ``\\``.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _disk_path(self, safe_key: str) -> Path | None:
        """Return the disk path for a sanitized key, or None if disk is disabled."""
        if self._config.disk_cache_dir is None:
            return None
        return self._config.disk_cache_dir / f"{safe_key}.npz"

    def _write_disk(self, path: Path, data: np.ndarray) -> None:
        """Write *data* to *path* atomically using compression if configured.

        Writes to a temporary file first, then renames to the target path
        to avoid leaving partial files on crash or interruption.
        """
        try:
            fd, tmp_path_str = tempfile.mkstemp(
                dir=path.parent, suffix=".npz.tmp"
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)
            try:
                if self._config.compression:
                    np.savez_compressed(tmp_path, data=data)
                else:
                    np.savez(tmp_path, data=data)
                tmp_path.rename(path)
                logger.debug(
                    "TieredCache: wrote disk entry '%s' (%d bytes)",
                    path.name,
                    path.stat().st_size,
                )
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        except Exception:
            logger.warning(
                "TieredCache: failed to write disk cache to '%s'",
                path,
                exc_info=True,
            )
