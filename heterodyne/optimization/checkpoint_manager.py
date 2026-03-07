"""Save and restore optimization state via JSON checkpoints.

Provides a :class:`CheckpointManager` that persists :class:`CheckpointData`
snapshots to disk, enabling warm-restart and forensic analysis of
optimization runs.  Uses ``json`` for serialization so that checkpoint files
are human-readable and safe to load from untrusted sources.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointData:
    """Snapshot of optimization state at a single point in time.

    Attributes:
        parameters: Fitted parameter values (length-14 for heterodyne model).
        cost: Scalar cost / objective value at this iteration.
        iteration: Iteration number (0-based).
        metadata: Arbitrary key-value metadata (solver name, config hash, ...).
        timestamp: ISO-8601 UTC timestamp of when the checkpoint was created.
    """

    parameters: np.ndarray
    cost: float
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=datetime.UTC).isoformat()
    )

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
        return cls(
            parameters=np.asarray(d["parameters"], dtype=np.float64),
            cost=float(d["cost"]),
            iteration=int(d["iteration"]),
            metadata=dict(d.get("metadata", {})),
            timestamp=str(d["timestamp"]),
        )


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

        The filename encodes the iteration number and a UTC timestamp so
        that checkpoints sort lexicographically by creation time.

        Args:
            data: Checkpoint payload to save.

        Returns:
            Path to the newly created checkpoint file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%S")
        filename = f"checkpoint_iter{data.iteration:06d}_{ts}.json"
        path = self._dir / filename

        path.write_text(json.dumps(data.to_dict(), indent=2), encoding="utf-8")
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
        """Load a specific checkpoint file.

        Args:
            path: Path to the JSON checkpoint.

        Returns:
            Restored :class:`CheckpointData`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("Loaded checkpoint: %s", path)
        return CheckpointData.from_dict(raw)

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
