"""Data configuration for XPCS loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from heterodyne.data.types import AngleRange, QRange


@dataclass
class DataConfig:
    """Configuration for XPCS data loading and preprocessing.

    Attributes:
        file_path: Path to the data file.
        format: File format ('auto', 'hdf5', 'npz', 'mat').
            When 'auto', the format is detected from the file extension.
        q_range: Optional wavevector range for filtering.
        angle_range: Optional azimuthal angle range for filtering.
        time_range: Optional (t_min, t_max) time window for cropping.
        diagonal_width: Number of super-/sub-diagonals to exclude from
            off-diagonal statistics. Defaults to 1 (exclude main diagonal only).
        normalize: Whether to normalize correlation data (diagonal to 1).
        remove_outliers: Whether to apply outlier removal.
        outlier_sigma: Number of standard deviations for outlier threshold.
    """

    file_path: str = ""
    format: str = "auto"
    q_range: QRange | None = None
    angle_range: AngleRange | None = None
    time_range: tuple[float, float] | None = None
    diagonal_width: int = 1
    normalize: bool = True
    remove_outliers: bool = True
    outlier_sigma: float = 3.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        """Create a DataConfig from a dictionary.

        Nested ``q_range`` and ``angle_range`` entries may be given as
        dicts with the appropriate keys, lists/tuples of two floats, or
        pre-constructed ``QRange``/``AngleRange`` instances.

        Args:
            d: Configuration dictionary.

        Returns:
            DataConfig instance.
        """
        kwargs: dict[str, Any] = {}

        # Simple scalar fields
        for key in (
            "file_path",
            "format",
            "diagonal_width",
            "normalize",
            "remove_outliers",
            "outlier_sigma",
        ):
            if key in d:
                kwargs[key] = d[key]

        # q_range
        q_raw = d.get("q_range")
        if q_raw is not None:
            if isinstance(q_raw, QRange):
                kwargs["q_range"] = q_raw
            elif isinstance(q_raw, dict):
                kwargs["q_range"] = QRange(
                    q_min=q_raw["q_min"],
                    q_max=q_raw["q_max"],
                )
            elif isinstance(q_raw, (list, tuple)) and len(q_raw) == 2:
                kwargs["q_range"] = QRange(q_min=q_raw[0], q_max=q_raw[1])
            else:
                raise ValueError(
                    f"q_range must be a QRange, dict, or 2-element sequence, "
                    f"got {type(q_raw).__name__}"
                )

        # angle_range
        angle_raw = d.get("angle_range")
        if angle_raw is not None:
            if isinstance(angle_raw, AngleRange):
                kwargs["angle_range"] = angle_raw
            elif isinstance(angle_raw, dict):
                kwargs["angle_range"] = AngleRange(
                    phi_min=angle_raw["phi_min"],
                    phi_max=angle_raw["phi_max"],
                )
            elif isinstance(angle_raw, (list, tuple)) and len(angle_raw) == 2:
                kwargs["angle_range"] = AngleRange(
                    phi_min=angle_raw[0],
                    phi_max=angle_raw[1],
                )
            else:
                raise ValueError(
                    f"angle_range must be an AngleRange, dict, or 2-element "
                    f"sequence, got {type(angle_raw).__name__}"
                )

        # time_range
        time_raw = d.get("time_range")
        if time_raw is not None:
            if isinstance(time_raw, (list, tuple)) and len(time_raw) == 2:
                kwargs["time_range"] = (float(time_raw[0]), float(time_raw[1]))
            else:
                raise ValueError(
                    f"time_range must be a 2-element sequence, "
                    f"got {type(time_raw).__name__}"
                )

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to a plain dictionary.

        Returns:
            Dictionary representation with JSON-compatible values.
        """
        result: dict[str, Any] = {
            "file_path": self.file_path,
            "format": self.format,
            "diagonal_width": self.diagonal_width,
            "normalize": self.normalize,
            "remove_outliers": self.remove_outliers,
            "outlier_sigma": self.outlier_sigma,
        }

        if self.q_range is not None:
            result["q_range"] = {
                "q_min": self.q_range.q_min,
                "q_max": self.q_range.q_max,
            }
        else:
            result["q_range"] = None

        if self.angle_range is not None:
            result["angle_range"] = {
                "phi_min": self.angle_range.phi_min,
                "phi_max": self.angle_range.phi_max,
            }
        else:
            result["angle_range"] = None

        if self.time_range is not None:
            result["time_range"] = list(self.time_range)
        else:
            result["time_range"] = None

        return result
