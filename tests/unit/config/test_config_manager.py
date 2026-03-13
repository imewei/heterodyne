"""Tests for configuration manager module.

Tests ConfigManager class for loading and accessing configuration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from heterodyne.config.manager import (
    ConfigManager,
    ConfigurationError,
    load_xpcs_config,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def minimal_config() -> dict:
    """Minimal valid configuration."""
    return {
        "experimental_data": {
            "file_path": "/path/to/data.h5",
        },
        "temporal": {
            "dt": 0.001,
            "time_length": 100,
        },
        "scattering": {
            "wavevector_q": 0.01,
        },
        "parameters": {},
    }


@pytest.fixture
def full_config() -> dict:
    """Full configuration with all sections."""
    return {
        "experimental_data": {
            "file_path": "/path/to/data.h5",
            "data_folder_path": "/path/to/folder",
            "file_format": "hdf5",
        },
        "temporal": {
            "dt": 0.001,
            "time_length": 100,
            "t_start": 5,
        },
        "scattering": {
            "wavevector_q": 0.01,
            "phi_angles": [0.0, 90.0, 180.0],
        },
        "parameters": {
            "reference": {
                "D0": {"value": 1.0, "vary": True},
                "alpha": {"value": 1.0, "vary": False},
            },
            "sample": {
                "D0": 2.0,  # Scalar value
            },
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {"max_iterations": 100},
            "cmc": {"num_samples": 1000},
        },
        "output": {
            "output_dir": "./results",
        },
    }


@pytest.fixture
def config_yaml_file(full_config: dict) -> Path:
    """Create temporary YAML config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(full_config, f)
        path = Path(f.name)

    yield path
    path.unlink()


# ============================================================================
# Initialization Tests
# ============================================================================


class TestConfigManagerInit:
    """Tests for ConfigManager initialization."""

    @pytest.mark.unit
    def test_init_with_valid_config(self, minimal_config: dict) -> None:
        """Initialize with valid minimal config."""
        manager = ConfigManager(minimal_config)
        assert manager.raw_config == minimal_config

    @pytest.mark.unit
    def test_init_missing_experimental_data_raises(self) -> None:
        """Missing experimental_data section raises error."""
        config = {
            "temporal": {"dt": 0.001, "time_length": 100},
            "scattering": {"wavevector_q": 0.01},
            "parameters": {},
        }
        with pytest.raises(ConfigurationError) as excinfo:
            ConfigManager(config)
        assert "experimental_data" in str(excinfo.value)

    @pytest.mark.unit
    def test_init_missing_temporal_raises(self) -> None:
        """Missing temporal section raises error."""
        config = {
            "experimental_data": {"file_path": "/path"},
            "scattering": {"wavevector_q": 0.01},
            "parameters": {},
        }
        with pytest.raises(ConfigurationError) as excinfo:
            ConfigManager(config)
        assert "temporal" in str(excinfo.value)

    @pytest.mark.unit
    def test_init_missing_scattering_raises(self) -> None:
        """Missing scattering section raises error."""
        config = {
            "experimental_data": {"file_path": "/path"},
            "temporal": {"dt": 0.001, "time_length": 100},
            "parameters": {},
        }
        with pytest.raises(ConfigurationError) as excinfo:
            ConfigManager(config)
        assert "scattering" in str(excinfo.value)

    @pytest.mark.unit
    def test_init_missing_multiple_sections(self) -> None:
        """Missing multiple sections lists all in error."""
        config = {"experimental_data": {"file_path": "/path"}}
        with pytest.raises(ConfigurationError) as excinfo:
            ConfigManager(config)
        error_msg = str(excinfo.value)
        assert "temporal" in error_msg
        assert "scattering" in error_msg
        assert "parameters" in error_msg


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestConfigManagerFactoryMethods:
    """Tests for factory methods."""

    @pytest.mark.unit
    def test_from_dict(self, minimal_config: dict) -> None:
        """from_dict creates ConfigManager."""
        manager = ConfigManager.from_dict(minimal_config)
        assert isinstance(manager, ConfigManager)
        assert manager.raw_config == minimal_config

    @pytest.mark.unit
    def test_from_yaml(self, config_yaml_file: Path) -> None:
        """from_yaml loads from file."""
        manager = ConfigManager.from_yaml(config_yaml_file)
        assert isinstance(manager, ConfigManager)
        assert "experimental_data" in manager.raw_config

    @pytest.mark.unit
    def test_from_yaml_string_path(self, config_yaml_file: Path) -> None:
        """from_yaml accepts string path."""
        manager = ConfigManager.from_yaml(str(config_yaml_file))
        assert isinstance(manager, ConfigManager)

    @pytest.mark.unit
    def test_from_yaml_nonexistent_file_raises(self) -> None:
        """from_yaml with nonexistent file raises error."""
        from heterodyne.utils.path_validation import PathValidationError

        with pytest.raises(PathValidationError):
            ConfigManager.from_yaml("/nonexistent/config.yaml")


# ============================================================================
# Property Tests - Experimental Data
# ============================================================================


class TestConfigManagerExperimentalData:
    """Tests for experimental data properties."""

    @pytest.mark.unit
    def test_data_file_path(self, full_config: dict) -> None:
        """data_file_path returns correct path."""
        manager = ConfigManager(full_config)
        assert manager.data_file_path == Path("/path/to/data.h5")

    @pytest.mark.unit
    def test_data_folder_path_when_set(self, full_config: dict) -> None:
        """data_folder_path returns path when set."""
        manager = ConfigManager(full_config)
        assert manager.data_folder_path == Path("/path/to/folder")

    @pytest.mark.unit
    def test_data_folder_path_when_not_set(self, minimal_config: dict) -> None:
        """data_folder_path returns None when not set."""
        manager = ConfigManager(minimal_config)
        assert manager.data_folder_path is None

    @pytest.mark.unit
    def test_file_format_when_set(self, full_config: dict) -> None:
        """file_format returns value when set."""
        manager = ConfigManager(full_config)
        assert manager.file_format == "hdf5"

    @pytest.mark.unit
    def test_file_format_default(self, minimal_config: dict) -> None:
        """file_format returns default 'hdf5'."""
        manager = ConfigManager(minimal_config)
        assert manager.file_format == "hdf5"


# ============================================================================
# Property Tests - Temporal Settings
# ============================================================================


class TestConfigManagerTemporal:
    """Tests for temporal settings properties."""

    @pytest.mark.unit
    def test_dt(self, full_config: dict) -> None:
        """dt returns correct value."""
        manager = ConfigManager(full_config)
        assert manager.dt == 0.001

    @pytest.mark.unit
    def test_time_length(self, full_config: dict) -> None:
        """time_length returns correct value."""
        manager = ConfigManager(full_config)
        assert manager.time_length == 100

    @pytest.mark.unit
    def test_t_start_when_set(self, full_config: dict) -> None:
        """t_start returns value when set."""
        manager = ConfigManager(full_config)
        assert manager.t_start == 5

    @pytest.mark.unit
    def test_t_start_default(self, minimal_config: dict) -> None:
        """t_start returns default 0."""
        manager = ConfigManager(minimal_config)
        assert manager.t_start == 0


# ============================================================================
# Property Tests - Scattering Settings
# ============================================================================


class TestConfigManagerScattering:
    """Tests for scattering settings properties."""

    @pytest.mark.unit
    def test_wavevector_q(self, full_config: dict) -> None:
        """wavevector_q returns correct value."""
        manager = ConfigManager(full_config)
        assert manager.wavevector_q == 0.01

    @pytest.mark.unit
    def test_phi_angles_when_set(self, full_config: dict) -> None:
        """phi_angles returns list when set."""
        manager = ConfigManager(full_config)
        assert manager.phi_angles == [0.0, 90.0, 180.0]

    @pytest.mark.unit
    def test_phi_angles_when_not_set(self, minimal_config: dict) -> None:
        """phi_angles returns None when not set."""
        manager = ConfigManager(minimal_config)
        assert manager.phi_angles is None


# ============================================================================
# Property Tests - Parameters
# ============================================================================


class TestConfigManagerParameters:
    """Tests for parameter access methods."""

    @pytest.mark.unit
    def test_parameters_config(self, full_config: dict) -> None:
        """parameters_config returns parameters section."""
        manager = ConfigManager(full_config)
        assert "reference" in manager.parameters_config
        assert "sample" in manager.parameters_config

    @pytest.mark.unit
    def test_get_parameter_value_dict_format(self, full_config: dict) -> None:
        """get_parameter_value works with dict format."""
        manager = ConfigManager(full_config)
        assert manager.get_parameter_value("reference", "D0") == 1.0
        assert manager.get_parameter_value("reference", "alpha") == 1.0

    @pytest.mark.unit
    def test_get_parameter_value_scalar_format(self, full_config: dict) -> None:
        """get_parameter_value works with scalar format."""
        manager = ConfigManager(full_config)
        assert manager.get_parameter_value("sample", "D0") == 2.0

    @pytest.mark.unit
    def test_get_parameter_vary_true(self, full_config: dict) -> None:
        """get_parameter_vary returns True when vary=True."""
        manager = ConfigManager(full_config)
        assert manager.get_parameter_vary("reference", "D0") is True

    @pytest.mark.unit
    def test_get_parameter_vary_false(self, full_config: dict) -> None:
        """get_parameter_vary returns False when vary=False."""
        manager = ConfigManager(full_config)
        assert manager.get_parameter_vary("reference", "alpha") is False

    @pytest.mark.unit
    def test_get_parameter_vary_default(self, full_config: dict) -> None:
        """get_parameter_vary defaults to True for scalar format."""
        manager = ConfigManager(full_config)
        assert manager.get_parameter_vary("sample", "D0") is True


# ============================================================================
# Property Tests - Optimization
# ============================================================================


class TestConfigManagerOptimization:
    """Tests for optimization settings properties."""

    @pytest.mark.unit
    def test_optimization_method_when_set(self, full_config: dict) -> None:
        """optimization_method returns value when set."""
        manager = ConfigManager(full_config)
        assert manager.optimization_method == "nlsq"

    @pytest.mark.unit
    def test_optimization_method_default(self, minimal_config: dict) -> None:
        """optimization_method defaults to 'nlsq'."""
        manager = ConfigManager(minimal_config)
        assert manager.optimization_method == "nlsq"

    @pytest.mark.unit
    def test_nlsq_config(self, full_config: dict) -> None:
        """nlsq_config returns NLSQ settings."""
        manager = ConfigManager(full_config)
        assert manager.nlsq_config == {"max_iterations": 100}

    @pytest.mark.unit
    def test_cmc_config(self, full_config: dict) -> None:
        """cmc_config returns CMC settings."""
        manager = ConfigManager(full_config)
        assert manager.cmc_config == {"num_samples": 1000}

    @pytest.mark.unit
    def test_nlsq_config_empty_default(self, minimal_config: dict) -> None:
        """nlsq_config returns empty dict when not set."""
        manager = ConfigManager(minimal_config)
        assert manager.nlsq_config == {}

    @pytest.mark.unit
    def test_update_optimization_config_persists(self, full_config: dict) -> None:
        """update_optimization_config writes through to stored config."""
        manager = ConfigManager(full_config)
        manager.update_optimization_config("cmc", "num_samples", 5000)
        manager.update_optimization_config("nlsq", "multistart", True)

        # Verify via properties (which return deep copies)
        assert manager.cmc_config["num_samples"] == 5000
        assert manager.nlsq_config["multistart"] is True
        # Original key should still be present
        assert manager.nlsq_config["max_iterations"] == 100

    @pytest.mark.unit
    def test_update_optimization_config_creates_sections(
        self, minimal_config: dict
    ) -> None:
        """update_optimization_config creates missing optimization sections."""
        manager = ConfigManager(minimal_config)
        manager.update_optimization_config("cmc", "num_chains", 4)

        assert manager.cmc_config["num_chains"] == 4
        # get_config should reflect the change
        raw = manager.get_config()
        assert raw["optimization"]["cmc"]["num_chains"] == 4


# ============================================================================
# Property Tests - Output
# ============================================================================


class TestConfigManagerOutput:
    """Tests for output settings properties."""

    @pytest.mark.unit
    def test_output_dir_when_set(self, full_config: dict) -> None:
        """output_dir returns value when set."""
        manager = ConfigManager(full_config)
        assert manager.output_dir == Path("./results")

    @pytest.mark.unit
    def test_output_dir_default(self, minimal_config: dict) -> None:
        """output_dir defaults to './output'."""
        manager = ConfigManager(minimal_config)
        assert manager.output_dir == Path("./output")


# ============================================================================
# to_yaml Tests
# ============================================================================


class TestConfigManagerToYaml:
    """Tests for to_yaml method."""

    @pytest.mark.unit
    def test_to_yaml_saves_file(self, full_config: dict) -> None:
        """to_yaml creates YAML file."""
        manager = ConfigManager(full_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.yaml"
            manager.to_yaml(output_path)

            assert output_path.exists()

    @pytest.mark.unit
    def test_to_yaml_content_loadable(self, full_config: dict) -> None:
        """to_yaml creates loadable YAML."""
        manager = ConfigManager(full_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.yaml"
            manager.to_yaml(output_path)

            # Reload and verify
            loaded = ConfigManager.from_yaml(output_path)
            assert loaded.dt == manager.dt
            assert loaded.wavevector_q == manager.wavevector_q

    @pytest.mark.unit
    def test_to_yaml_creates_parent_dirs(self, full_config: dict) -> None:
        """to_yaml creates parent directories."""
        manager = ConfigManager(full_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "dir" / "output.yaml"
            manager.to_yaml(output_path)

            assert output_path.exists()


# ============================================================================
# load_xpcs_config Tests
# ============================================================================


class TestLoadXpcsConfig:
    """Tests for load_xpcs_config function."""

    @pytest.mark.unit
    def test_load_xpcs_config(self, config_yaml_file: Path) -> None:
        """load_xpcs_config returns ConfigManager."""
        manager = load_xpcs_config(config_yaml_file)
        assert isinstance(manager, ConfigManager)

    @pytest.mark.unit
    def test_load_xpcs_config_string_path(self, config_yaml_file: Path) -> None:
        """load_xpcs_config accepts string path."""
        manager = load_xpcs_config(str(config_yaml_file))
        assert isinstance(manager, ConfigManager)
