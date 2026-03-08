"""Configuration manager for heterodyne analysis."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, cast

import yaml

from heterodyne.utils.path_validation import validate_file_exists

logger = logging.getLogger(__name__)

_ALLOWED_OPTIMIZATION_METHODS = {"nlsq", "cmc"}


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


class ConfigManager:
    """Manager for heterodyne analysis configuration.

    Handles loading, validation, and access to configuration settings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration dictionary.

        Args:
            config: Configuration dictionary
        """
        self._config = config
        self._normalize_schema()
        self._validate()

    def _normalize_schema(self) -> None:
        """Normalize deprecated configuration keys to canonical names."""
        from heterodyne.config.types import PARAMETER_NAME_MAPPING

        # Normalize parameter names in all parameter groups
        params = self._config.get("parameters", {})
        for group_name, group_config in params.items():
            if not isinstance(group_config, dict):
                continue
            normalized: dict[str, Any] = {}
            for key, value in group_config.items():
                canonical = PARAMETER_NAME_MAPPING.get(key, key)
                if canonical != key:
                    logger.debug("Normalized parameter key '%s' -> '%s'", key, canonical)
                normalized[canonical] = value
            params[group_name] = normalized

        # Normalize CMC config keys
        cmc = self._config.get("optimization", {}).get("cmc", {})
        if isinstance(cmc, dict):
            normalized_cmc: dict[str, Any] = {}
            for key, value in cmc.items():
                canonical = PARAMETER_NAME_MAPPING.get(key, key)
                if canonical != key:
                    logger.debug("Normalized CMC key '%s' -> '%s'", key, canonical)
                normalized_cmc[canonical] = value
            if "optimization" in self._config and "cmc" in self._config["optimization"]:
                self._config["optimization"]["cmc"] = normalized_cmc

    def _validate(self) -> None:
        """Validate configuration structure."""
        required_sections = ["experimental_data", "temporal", "scattering", "parameters"]
        missing = [s for s in required_sections if s not in self._config]
        if missing:
            raise ConfigurationError(f"Missing required sections: {missing}")

        # Warn if optimization section is absent; validate method if present
        if "optimization" not in self._config:
            logger.warning(
                "Configuration has no 'optimization' section; "
                "defaults will be used (method='nlsq')"
            )
        else:
            method = self._config["optimization"].get("method")
            if method is not None and method not in _ALLOWED_OPTIMIZATION_METHODS:
                raise ConfigurationError(
                    f"Invalid optimization method '{method}'. "
                    f"Allowed values: {sorted(_ALLOWED_OPTIMIZATION_METHODS)}"
                )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ConfigManager:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ConfigManager instance
        """
        path = validate_file_exists(path, "Configuration file")
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ConfigManager:
        """Create from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ConfigManager instance
        """
        return cls(config)

    @classmethod
    def from_json(cls, path: Path | str) -> ConfigManager:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ConfigManager instance
        """
        import json

        path = validate_file_exists(path, "Configuration file")
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)

    @property
    def raw_config(self) -> dict[str, Any]:
        """Get raw configuration dictionary (deep copy to prevent mutation)."""
        return copy.deepcopy(self._config)

    # === Experimental Data ===

    @property
    def data_file_path(self) -> Path:
        """Path to experimental data file."""
        return Path(self._config["experimental_data"]["file_path"])

    @property
    def data_folder_path(self) -> Path | None:
        """Optional folder path for data."""
        path = self._config["experimental_data"].get("data_folder_path")
        return Path(path) if path else None

    @property
    def file_format(self) -> str:
        """Data file format."""
        return cast(str, self._config["experimental_data"].get("file_format", "hdf5"))

    # === Temporal Settings ===

    @property
    def dt(self) -> float:
        """Time step."""
        return float(self._config["temporal"]["dt"])

    @property
    def time_length(self) -> int:
        """Number of time points."""
        return int(self._config["temporal"]["time_length"])

    @property
    def t_start(self) -> int:
        """Starting time index."""
        return int(self._config["temporal"].get("t_start", 0))

    # === Scattering Settings ===

    @property
    def wavevector_q(self) -> float:
        """Scattering wavevector magnitude."""
        return float(self._config["scattering"]["wavevector_q"])

    @property
    def phi_angles(self) -> list[float] | None:
        """List of phi angles for analysis."""
        angles = self._config["scattering"].get("phi_angles")
        return [float(a) for a in angles] if angles else None

    # === Parameter Settings ===

    @property
    def parameters_config(self) -> dict[str, Any]:
        """Get parameters configuration section."""
        return cast(dict[str, Any], self._config.get("parameters", {}))

    def get_parameter_value(self, group: str, name: str) -> float:
        """Get a specific parameter value.

        Args:
            group: Parameter group ('reference', 'sample', etc.)
            name: Parameter name within group

        Returns:
            Parameter value
        """
        group_config = self._config["parameters"].get(group, {})
        param_config = group_config.get(name, {})
        if isinstance(param_config, dict):
            if "value" not in param_config:
                raise ConfigurationError(
                    f"Parameter '{name}' in group '{group}' is missing "
                    f"the required 'value' key"
                )
            return float(param_config["value"])
        return float(param_config)

    def get_parameter_vary(self, group: str, name: str) -> bool:
        """Check if parameter varies in optimization.

        Args:
            group: Parameter group
            name: Parameter name

        Returns:
            Whether parameter varies
        """
        group_config = self._config["parameters"].get(group, {})
        param_config = group_config.get(name, {})
        if isinstance(param_config, dict):
            return bool(param_config.get("vary", True))
        return True

    # === Optimization Settings ===

    @property
    def optimization_method(self) -> str:
        """Optimization method ('nlsq' or 'cmc')."""
        return cast(str, self._config.get("optimization", {}).get("method", "nlsq"))

    @property
    def nlsq_config(self) -> dict[str, Any]:
        """NLSQ optimization settings (returns a copy to prevent mutation)."""
        return copy.deepcopy(
            cast(dict[str, Any], self._config.get("optimization", {}).get("nlsq", {}))
        )

    @property
    def cmc_config(self) -> dict[str, Any]:
        """CMC analysis settings (returns a copy to prevent mutation)."""
        return copy.deepcopy(
            cast(dict[str, Any], self._config.get("optimization", {}).get("cmc", {}))
        )

    def _merge_cmc_config(self) -> dict[str, Any]:
        """Merge CMC config with sensible defaults.

        Config values override defaults.

        Returns:
            Merged CMC configuration dictionary
        """
        defaults: dict[str, Any] = {
            "num_warmup": 500,
            "num_samples": 1000,
            "num_chains": 4,
            "target_accept_prob": 0.8,
            "max_tree_depth": 10,
        }
        cmc = self._config.get("optimization", {}).get("cmc", {})
        if isinstance(cmc, dict):
            defaults.update(cmc)
        return defaults

    def _validate_cmc_config(self, cmc_config: dict[str, Any]) -> list[str]:
        """Validate CMC config values.

        Args:
            cmc_config: CMC configuration dictionary to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        num_warmup = cmc_config.get("num_warmup")
        if num_warmup is not None and (not isinstance(num_warmup, int) or num_warmup <= 0):
            errors.append(f"num_warmup must be > 0, got {num_warmup}")

        num_samples = cmc_config.get("num_samples")
        if num_samples is not None and (not isinstance(num_samples, int) or num_samples <= 0):
            errors.append(f"num_samples must be > 0, got {num_samples}")

        num_chains = cmc_config.get("num_chains")
        if num_chains is not None and (not isinstance(num_chains, int) or num_chains <= 0):
            errors.append(f"num_chains must be > 0, got {num_chains}")

        target_accept_prob = cmc_config.get("target_accept_prob")
        if target_accept_prob is not None and (
            not isinstance(target_accept_prob, (int, float))
            or target_accept_prob <= 0
            or target_accept_prob >= 1
        ):
            errors.append(
                f"target_accept_prob must be in (0, 1), got {target_accept_prob}"
            )

        max_tree_depth = cmc_config.get("max_tree_depth")
        if max_tree_depth is not None and (
            not isinstance(max_tree_depth, int)
            or max_tree_depth < 1
            or max_tree_depth > 20
        ):
            errors.append(
                f"max_tree_depth must be in [1, 20], got {max_tree_depth}"
            )

        return errors

    def get_config(self) -> dict[str, Any]:
        """Return raw config dict without deep copy (for internal use).

        Returns:
            Configuration dictionary (not copied)
        """
        return self._config

    def get_cmc_config(self) -> dict[str, Any]:
        """Return merged CMC config with defaults applied.

        Returns:
            CMC configuration with defaults merged in
        """
        return self._merge_cmc_config()

    # === Output Settings ===

    @property
    def output_dir(self) -> Path:
        """Output directory path."""
        output = self._config.get("output", {})
        return Path(output.get("output_dir", "./output"))

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


def load_xpcs_config(path: Path | str) -> ConfigManager:
    """Load XPCS analysis configuration from file.

    Convenience function for loading configuration.

    Args:
        path: Path to YAML configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager.from_yaml(path)
