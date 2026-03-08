"""Save and restore optimization state via JSON checkpoints.

Provides a :class:`CheckpointManager` that persists :class:`CheckpointData`
snapshots to disk, enabling warm-restart and forensic analysis of
optimization runs.  Uses ``json`` for serialization so that checkpoint files
are human-readable and safe to load from untrusted sources.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def _get_version() -> str:
    """Return the heterodyne package version, or fallback."""
    try:
        from heterodyne import __version__

        return str(__version__)
    except ImportError:
        return "unknown"


@dataclass
class CheckpointData:
    """Snapshot of optimization state at a single point in time.

    Attributes:
        parameters: Fitted parameter values (length-14 for heterodyne model).
        cost: Scalar cost / objective value at this iteration.
        iteration: Iteration number (0-based).
        metadata: Arbitrary key-value metadata (solver name, config hash, ...).
        timestamp: ISO-8601 UTC timestamp of when the checkpoint was created.
        version: Package version that created this checkpoint.
        checksum: SHA-256 hex digest for integrity verification.
    """

    parameters: np.ndarray
    cost: float
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC).isoformat()
    )
    version: str = field(default_factory=_get_version)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Compute checksum if not already set."""
        if not self.checksum:
            self.checksum = self.compute_checksum(self.parameters, self.cost)

    # -- integrity helpers ----------------------------------------------------

    @staticmethod
    def compute_checksum(parameters: np.ndarray, cost: float) -> str:
        """Compute SHA-256 checksum from parameters and cost.

        Args:
            parameters: Parameter array.
            cost: Cost value.

        Returns:
            Hex digest string.
        """
        h = hashlib.sha256()
        h.update(np.asarray(parameters, dtype=np.float64).tobytes())
        h.update(struct.pack("<d", float(cost)))
        return h.hexdigest()

    def verify_integrity(self) -> bool:
        """Recompute checksum and compare to stored value.

        Returns:
            ``True`` if checksum matches, ``False`` otherwise.
        """
        expected = self.compute_checksum(self.parameters, self.cost)
        return self.checksum == expected

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary.

        NumPy arrays are stored as plain Python lists so that the
        checkpoint file is portable and human-readable.
        """
        d: dict[str, Any] = asdict(self)
        d["parameters"] = self.parameters.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointData:
        """Reconstruct from a dictionary (inverse of :meth:`to_dict`).

        Args:
            d: Dictionary as produced by :meth:`to_dict`.

        Returns:
            Restored :class:`CheckpointData` instance.
        """
        params = np.asarray(d["parameters"], dtype=np.float64)
        cost = float(d["cost"])
        stored_checksum = d.get("checksum", "")

        instance = cls(
            parameters=params,
            cost=cost,
            iteration=int(d["iteration"]),
            metadata=dict(d.get("metadata", {})),
            timestamp=str(d["timestamp"]),
            version=d.get("version", "unknown"),
            checksum=stored_checksum if stored_checksum else cls.compute_checksum(params, cost),
        )
        return instance


class CheckpointManager:
    """Manage a directory of JSON optimization checkpoints.

    Parameters:
        checkpoint_dir: Directory in which checkpoint files are stored.
            Created on first :meth:`save` if it does not exist.
        max_checkpoints: Maximum number of checkpoints to retain.
            Older files are removed automatically after each save.
    """

    def __init__(self, checkpoint_dir: Path | str, max_checkpoints: int = 5) -> None:
        self._dir = Path(checkpoint_dir)
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be >= 1")
        self._max = max_checkpoints

    # -- public API -----------------------------------------------------------

    def save(self, data: CheckpointData) -> Path:
        """Persist *data* as a JSON file and enforce the retention limit.

        Uses atomic write (write to temp file, then ``os.replace``) to
        prevent half-written files on crash.

        Args:
            data: Checkpoint payload to save.

        Returns:
            Path to the newly created checkpoint file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%S")
        filename = f"checkpoint_iter{data.iteration:06d}_{ts}.json"
        path = self._dir / filename
        tmp_path = path.with_suffix(".tmp")

        content = json.dumps(data.to_dict(), indent=2)
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)

        logger.debug("Saved checkpoint: %s", path)

        self.cleanup(keep=self._max)
        return path

    def load_latest(self) -> CheckpointData | None:
        """Load the most recent checkpoint, or ``None`` if none exist.

        Returns:
            Latest :class:`CheckpointData`, or ``None``.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.debug("No checkpoints found in %s", self._dir)
            return None
        return self.load(checkpoints[-1])

    def load(self, path: Path) -> CheckpointData:
        """Load a specific checkpoint file and verify integrity.

        Args:
            path: Path to the JSON checkpoint.

        Returns:
            Restored :class:`CheckpointData`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = CheckpointData.from_dict(raw)

        if not data.verify_integrity():
            logger.warning(
                "Checkpoint integrity check failed for %s — "
                "stored checksum does not match recomputed value",
                path,
            )

        logger.debug("Loaded checkpoint: %s (version=%s)", path, data.version)
        return data

    def find_latest_valid(self) -> CheckpointData | None:
        """Find the most recent checkpoint that passes integrity verification.

        Iterates from newest to oldest, skipping corrupted checkpoints.

        Returns:
            Latest valid :class:`CheckpointData`, or ``None`` if all are
            corrupted or no checkpoints exist.
        """
        for path in reversed(self.list_checkpoints()):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                data = CheckpointData.from_dict(raw)
                if data.verify_integrity():
                    logger.debug("Found valid checkpoint: %s", path)
                    return data
                logger.warning("Skipping corrupted checkpoint: %s", path)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Skipping unreadable checkpoint %s: %s", path, exc)
        return None

    def list_checkpoints(self) -> list[Path]:
        """Return all checkpoint paths, sorted oldest-first by filename.

        Returns:
            Sorted list of checkpoint file paths.
        """
        if not self._dir.is_dir():
            return []
        return sorted(self._dir.glob("checkpoint_*.json"))

    def cleanup(self, keep: int) -> None:
        """Remove the oldest checkpoints so that at most *keep* remain.

        Args:
            keep: Number of most-recent checkpoints to retain.
        """
        if keep < 0:
            raise ValueError("keep must be >= 0")

        checkpoints = self.list_checkpoints()
        to_remove = checkpoints[: max(0, len(checkpoints) - keep)]
        for p in to_remove:
            p.unlink()
            logger.debug("Removed old checkpoint: %s", p)
