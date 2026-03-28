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
        self._config = copy.deepcopy(config)
        self._normalize_schema()
        self._validate()

    def _normalize_schema(self) -> None:
        """Normalize deprecated configuration keys to canonical names."""
        from heterodyne.config.types import PARAMETER_NAME_MAPPING

        # Normalize parameter names in all parameter groups
        params: dict[str, Any] = self._config.get("parameters", {})
        for group_name in list(params.keys()):
            group_config = params[group_name]
            if not isinstance(group_config, dict):
                continue
            normalized: dict[str, Any] = {}
            for key, value in group_config.items():
                canonical: str = PARAMETER_NAME_MAPPING.get(str(key), str(key))
                if canonical != key:
                    logger.debug(
                        "Normalized parameter key '%s' -> '%s'", key, canonical
                    )
                normalized[canonical] = value
            params[group_name] = normalized

        # Normalize legacy temporal/scattering sections into analyzer_parameters
        self._normalize_analyzer_parameters()

        # Normalize CMC config keys
        cmc = self._config.get("optimization", {}).get("cmc", {})
        if isinstance(cmc, dict):
            normalized_cmc: dict[str, Any] = {}
            for key, value in cmc.items():
                cmc_canonical: str = PARAMETER_NAME_MAPPING.get(str(key), str(key))
                if cmc_canonical != key:
                    logger.debug("Normalized CMC key '%s' -> '%s'", key, cmc_canonical)
                normalized_cmc[cmc_canonical] = value
            opt = self._config.get("optimization")
            if isinstance(opt, dict) and "cmc" in opt:
                opt["cmc"] = normalized_cmc

    def _normalize_analyzer_parameters(self) -> None:
        """Merge legacy ``temporal``/``scattering`` sections into ``analyzer_parameters``.

        Supports three config styles:
        1. **New** — ``analyzer_parameters`` with dt, start_frame, end_frame,
           scattering, and geometry sub-keys.
        2. **Legacy** — separate ``temporal`` and ``scattering`` top-level sections.
        3. **Mixed** — ``analyzer_parameters`` exists but legacy sections also
           present; legacy values are used as fallbacks only.

        After normalization, ``temporal`` and ``scattering`` top-level keys are
        synthesized from ``analyzer_parameters`` so downstream code that reads
        the raw config dict keeps working during migration.
        """
        ap = self._config.get("analyzer_parameters", {})
        temporal = self._config.get("temporal", {})
        scattering = self._config.get("scattering", {})

        if not ap and not temporal and not scattering:
            # Nothing to normalize — will fail validation later
            return

        if not ap and (temporal or scattering):
            logger.info(
                "Migrating legacy 'temporal'/'scattering' sections "
                "into 'analyzer_parameters'"
            )

        # --- Build canonical analyzer_parameters --------------------------
        merged: dict[str, Any] = {}

        # dt: top-level in analyzer_parameters (parity with homodyne)
        merged["dt"] = ap.get("dt", temporal.get("dt", 1.0))

        # Frame range: prefer start_frame/end_frame (1-indexed, inclusive)
        if "start_frame" in ap:
            merged["start_frame"] = int(ap["start_frame"])
        elif "t_start" in temporal:
            # Legacy: t_start is 0-indexed → start_frame is 1-indexed
            merged["start_frame"] = int(temporal["t_start"]) + 1
        else:
            merged["start_frame"] = 1

        if "end_frame" in ap:
            merged["end_frame"] = int(ap["end_frame"])
        elif "time_length" in temporal:
            # Legacy: end_frame = t_start + time_length (inclusive)
            t_start = int(temporal.get("t_start", 0))
            merged["end_frame"] = t_start + int(temporal["time_length"])
        else:
            merged["end_frame"] = 1000

        # Scattering sub-section
        ap_scat = ap.get("scattering", {})
        merged_scat: dict[str, Any] = {}
        merged_scat["wavevector_q"] = ap_scat.get(
            "wavevector_q", scattering.get("wavevector_q", 0.01)
        )
        # phi_angles (optional)
        phi = ap_scat.get("phi_angles", scattering.get("phi_angles"))
        if phi is not None:
            merged_scat["phi_angles"] = phi
        merged["scattering"] = merged_scat

        # Geometry sub-section (new — parity with homodyne)
        ap_geom = ap.get("geometry", {})
        if ap_geom:
            merged["geometry"] = dict(ap_geom)

        self._config["analyzer_parameters"] = merged

        # --- Synthesize legacy keys for downstream raw-config readers -----
        start_frame = merged["start_frame"]
        end_frame = merged["end_frame"]
        t_start = start_frame - 1  # 1-indexed → 0-indexed
        time_length = end_frame - t_start  # inclusive range length

        self._config["temporal"] = {
            "dt": merged["dt"],
            "time_length": time_length,
            "t_start": t_start,
        }
        self._config["scattering"] = {
            "wavevector_q": merged["scattering"]["wavevector_q"],
        }
        if "phi_angles" in merged["scattering"]:
            self._config["scattering"]["phi_angles"] = merged["scattering"][
                "phi_angles"
            ]

    def _validate(self) -> None:
        """Validate configuration structure."""
        required_sections = [
            "experimental_data",
            "analyzer_parameters",
            "parameters",
        ]
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

        # Validate config_version if present
        self._validate_config_version()

    def _validate_config_version(self) -> None:
        """Warn if config_version doesn't match package version."""
        metadata = self._config.get("metadata", {})
        config_version = metadata.get("config_version")
        if config_version is None:
            return
        try:
            from heterodyne._version import __version__

            # Compare major.minor only (patch mismatches are fine)
            cv_parts = str(config_version).split(".")[:2]
            pkg_parts = __version__.split(".")[:2]
            if cv_parts != pkg_parts:
                logger.warning(
                    "Config version %s does not match package version %s. "
                    "Some settings may have changed.",
                    config_version,
                    __version__,
                )
        except ImportError:
            pass  # Version not available (editable install without SCM)

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

    # === Analyzer Parameters ===

    @property
    def _ap(self) -> dict[str, Any]:
        """Canonical analyzer_parameters section."""
        return cast(dict[str, Any], self._config["analyzer_parameters"])

    @property
    def dt(self) -> float:
        """Time step [seconds]."""
        return float(self._ap["dt"])

    @property
    def start_frame(self) -> int:
        """Starting frame (1-indexed)."""
        return int(self._ap["start_frame"])

    @property
    def end_frame(self) -> int:
        """Ending frame (1-indexed, inclusive)."""
        return int(self._ap["end_frame"])

    @property
    def time_length(self) -> int:
        """Number of time points (derived from frame range)."""
        return self.end_frame - (self.start_frame - 1)

    @property
    def t_start(self) -> int:
        """Starting time index, 0-indexed (derived from start_frame)."""
        return self.start_frame - 1

    @property
    def wavevector_q(self) -> float:
        """Scattering wavevector magnitude [Å⁻¹]."""
        return float(self._ap["scattering"]["wavevector_q"])

    @property
    def phi_angles(self) -> list[float] | None:
        """List of phi angles for analysis."""
        angles = self._ap["scattering"].get("phi_angles")
        return [float(a) for a in angles] if angles else None

    @property
    def stator_rotor_gap(self) -> float | None:
        """Stator-rotor gap [Å] (optional geometry metadata)."""
        geom = self._ap.get("geometry", {})
        gap = geom.get("stator_rotor_gap")
        return float(gap) if gap is not None else None

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
        if num_warmup is not None and (
            not isinstance(num_warmup, int) or num_warmup <= 0
        ):
            errors.append(f"num_warmup must be > 0, got {num_warmup}")

        num_samples = cmc_config.get("num_samples")
        if num_samples is not None and (
            not isinstance(num_samples, int) or num_samples <= 0
        ):
            errors.append(f"num_samples must be > 0, got {num_samples}")

        num_chains = cmc_config.get("num_chains")
        if num_chains is not None and (
            not isinstance(num_chains, int) or num_chains <= 0
        ):
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
            errors.append(f"max_tree_depth must be in [1, 20], got {max_tree_depth}")

        return errors

    def update_optimization_config(self, section: str, key: str, value: Any) -> None:
        """Update a single optimization config key in-place.

        Args:
            section: Optimization sub-section ("nlsq" or "cmc").
            key: Configuration key to update.
            value: New value for the key.
        """
        self._config.setdefault("optimization", {}).setdefault(section, {})[key] = value

    def get_config(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration dictionary.

        Returns:
            Deep copy of the full configuration dictionary
        """
        return copy.deepcopy(self._config)

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
