"""Data configuration for XPCS loading and preprocessing."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heterodyne.data.types import AngleRange, QRange
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigValidationResult:
    """Result of validating an XPCS configuration dictionary.

    Attributes:
        is_valid: Whether the configuration passed all required checks.
        errors: List of validation error messages (fatal).
        warnings: List of non-fatal warnings.
        missing_optional: List of optional fields not present in config.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)


class XPCSConfigurationError(Exception):
    """Raised when an XPCS configuration is invalid or cannot be loaded."""

    pass


XPCS_CONFIG_SCHEMA: dict[str, Any] = {
    "required": {
        "data": {"file_path": str},
    },
    "optional": {
        "data": {
            "format": str,
            "q_range": (list, dict),
            "angle_range": (list, dict),
            "time_range": list,
            "diagonal_width": int,
            "normalize": bool,
            "remove_outliers": bool,
            "outlier_sigma": (int, float),
        },
        "analysis": {
            "method": str,
            "phi_angles": list,
            "q": (int, float),
            "dt": (int, float),
        },
        "output": {
            "directory": str,
            "save_plots": bool,
            "save_results": bool,
        },
        "mcmc": {
            "num_warmup": int,
            "num_samples": int,
            "num_chains": int,
            "target_accept_prob": (int, float),
        },
    },
}


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


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file does not exist.
        XPCSConfigurationError: If the file cannot be parsed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading YAML config from %s", path)
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise XPCSConfigurationError(
            f"Failed to parse YAML config {path}: {exc}"
        ) from exc

    # safe_load returns None for empty files
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise XPCSConfigurationError(
            f"YAML config must be a mapping, got {type(data).__name__}"
        )

    return data


def load_json_config(path: Path | str) -> dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        XPCSConfigurationError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading JSON config from %s", path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise XPCSConfigurationError(
            f"Failed to parse JSON config {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise XPCSConfigurationError(
            f"JSON config must be a mapping, got {type(data).__name__}"
        )

    return data


def validate_config_schema(
    config: dict[str, Any],
) -> ConfigValidationResult:
    """Validate a configuration dictionary against XPCS_CONFIG_SCHEMA.

    Checks that all required fields are present and that provided optional
    fields have acceptable types.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        ConfigValidationResult with validation outcome.
    """
    errors: list[str] = []
    warnings: list[str] = []
    missing_optional: list[str] = []

    # Check required fields
    for section, fields in XPCS_CONFIG_SCHEMA["required"].items():
        section_data = config.get(section)
        if section_data is None:
            errors.append(f"Missing required section: '{section}'")
            continue
        if not isinstance(section_data, dict):
            errors.append(
                f"Section '{section}' must be a mapping, "
                f"got {type(section_data).__name__}"
            )
            continue
        for field_name, expected_type in fields.items():
            if field_name not in section_data:
                errors.append(
                    f"Missing required field: '{section}.{field_name}'"
                )
            else:
                value = section_data[field_name]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{section}.{field_name}' must be "
                        f"{expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

    # Check optional fields
    for section, fields in XPCS_CONFIG_SCHEMA["optional"].items():
        section_data = config.get(section)
        if section_data is None:
            for field_name in fields:
                missing_optional.append(f"{section}.{field_name}")
            continue
        if not isinstance(section_data, dict):
            warnings.append(
                f"Optional section '{section}' should be a mapping, "
                f"got {type(section_data).__name__}"
            )
            continue
        for field_name, expected_type in fields.items():
            if field_name not in section_data:
                missing_optional.append(f"{section}.{field_name}")
            else:
                value = section_data[field_name]
                # expected_type can be a single type or tuple of types
                if isinstance(expected_type, tuple):
                    valid_types = expected_type
                else:
                    valid_types = (expected_type,)
                if not isinstance(value, valid_types):
                    type_names = ", ".join(t.__name__ for t in valid_types)
                    warnings.append(
                        f"Field '{section}.{field_name}' should be "
                        f"{type_names}, got {type(value).__name__}"
                    )

    is_valid = len(errors) == 0
    return ConfigValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        missing_optional=missing_optional,
    )


def apply_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default values for missing optional configuration fields.

    Args:
        config: Configuration dictionary (not modified in place).

    Returns:
        New dictionary with defaults applied for missing fields.
    """
    result = copy.deepcopy(config)

    # data section defaults
    data = result.setdefault("data", {})
    data.setdefault("format", "auto")
    data.setdefault("normalize", True)
    data.setdefault("remove_outliers", True)
    data.setdefault("outlier_sigma", 3.0)
    data.setdefault("diagonal_width", 1)

    # analysis section defaults
    analysis = result.setdefault("analysis", {})
    analysis.setdefault("method", "nlsq")

    # output section defaults
    output = result.setdefault("output", {})
    output.setdefault("save_plots", True)
    output.setdefault("save_results", True)

    # mcmc section defaults
    mcmc = result.setdefault("mcmc", {})
    mcmc.setdefault("num_warmup", 500)
    mcmc.setdefault("num_samples", 1000)
    mcmc.setdefault("num_chains", 4)
    mcmc.setdefault("target_accept_prob", 0.8)

    return result


def save_yaml_config(config: dict[str, Any], path: Path | str) -> None:
    """Save a configuration dictionary as a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Destination file path.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to save YAML config files. "
            "Install it with: pip install pyyaml"
        ) from None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Saving YAML config to %s", path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def create_example_yaml_config(path: Path | str) -> None:
    """Write a complete annotated example YAML configuration file.

    The generated file includes all sections (data, analysis, output, mcmc)
    with helpful inline comments.

    Args:
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    example = """\
# =============================================================================
# XPCS Heterodyne Analysis Configuration
# =============================================================================

# --- Data loading and preprocessing ---
data:
  # Path to the input data file (required)
  file_path: "/path/to/your/data.hdf5"

  # File format: "auto", "hdf5", "npz", or "mat"
  format: "auto"

  # Wavevector range for filtering (optional)
  # Can be a dict with q_min/q_max or a two-element list [q_min, q_max]
  # q_range:
  #   q_min: 0.001
  #   q_max: 0.1

  # Azimuthal angle range for filtering (optional)
  # Can be a dict with phi_min/phi_max or a two-element list [phi_min, phi_max]
  # angle_range:
  #   phi_min: 0.0
  #   phi_max: 360.0

  # Time window for cropping [t_min, t_max] (optional)
  # time_range: [0.0, 100.0]

  # Number of super-/sub-diagonals to exclude (default: 1)
  diagonal_width: 1

  # Normalize correlation data (default: true)
  normalize: true

  # Apply outlier removal (default: true)
  remove_outliers: true

  # Standard deviations for outlier threshold (default: 3.0)
  outlier_sigma: 3.0

# --- Analysis parameters ---
analysis:
  # Fitting method: "nlsq" or "mcmc" (default: "nlsq")
  method: "nlsq"

  # Azimuthal angles to analyze (degrees)
  # phi_angles: [0.0, 45.0, 90.0, 135.0]

  # Wavevector magnitude in Angstroms^-1
  # q: 0.01

  # Time step in seconds
  # dt: 1.0

# --- Output settings ---
output:
  # Output directory for results and plots
  directory: "./results"

  # Save diagnostic plots (default: true)
  save_plots: true

  # Save fit results to file (default: true)
  save_results: true

# --- MCMC (NumPyro) settings ---
mcmc:
  # Number of warmup/burn-in samples (default: 500)
  num_warmup: 500

  # Number of posterior samples (default: 1000)
  num_samples: 1000

  # Number of independent chains (default: 4)
  num_chains: 4

  # Target acceptance probability for NUTS (default: 0.8)
  target_accept_prob: 0.8
"""

    logger.debug("Writing example YAML config to %s", path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(example)


def migrate_json_to_yaml_config(
    json_path: Path | str,
    yaml_path: Path | str,
) -> None:
    """Migrate a JSON configuration file to YAML format.

    Loads the JSON config, applies defaults for any missing fields, and
    saves the result as a YAML file.

    Args:
        json_path: Path to the source JSON config file.
        yaml_path: Path for the destination YAML config file.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the JSON file does not exist.
        XPCSConfigurationError: If the JSON file cannot be parsed.
    """
    logger.info("Migrating JSON config %s -> YAML %s", json_path, yaml_path)
    config = load_json_config(json_path)
    config = apply_config_defaults(config)
    save_yaml_config(config, yaml_path)
