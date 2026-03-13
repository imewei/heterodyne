"""Unit tests for heterodyne.cli.args_parser module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from heterodyne.cli.args_parser import create_parser, validate_args


@pytest.mark.unit
class TestRequiredArguments:
    """Tests that required arguments are enforced."""

    def test_missing_config_raises_system_exit(self) -> None:
        """Omitting --config must cause SystemExit (argparse error)."""
        parser = create_parser()
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args([])

    def test_config_with_short_flag(self) -> None:
        """The -c shorthand must set the config path."""
        parser = create_parser()
        args = parser.parse_args(["-c", "analysis.yaml"])
        assert args.config == Path("analysis.yaml")

    def test_config_with_long_flag(self) -> None:
        """The --config flag must set the config path as a Path object."""
        parser = create_parser()
        args = parser.parse_args(["--config", "/data/run.yaml"])
        assert args.config == Path("/data/run.yaml")
        assert isinstance(args.config, Path)


@pytest.mark.unit
class TestOptionalArguments:
    """Tests for default values and explicit overrides of optional arguments."""

    @pytest.fixture()
    def parser(self) -> argparse.ArgumentParser:
        return create_parser()

    @pytest.fixture()
    def defaults(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Parse with only the required --config flag."""
        return parser.parse_args(["--config", "test.yaml"])

    # --- Default values ---

    def test_default_method_is_nlsq(self, defaults: argparse.Namespace) -> None:
        assert defaults.method == "nlsq"

    def test_default_output_is_none(self, defaults: argparse.Namespace) -> None:
        assert defaults.output is None

    def test_default_output_format_is_both(self, defaults: argparse.Namespace) -> None:
        assert defaults.output_format == "both"

    def test_default_phi_is_none(self, defaults: argparse.Namespace) -> None:
        assert defaults.phi is None

    def test_default_multistart_is_false(self, defaults: argparse.Namespace) -> None:
        assert defaults.multistart is False

    def test_default_multistart_n_is_10(self, defaults: argparse.Namespace) -> None:
        assert defaults.multistart_n == 10

    def test_default_num_samples_is_none(self, defaults: argparse.Namespace) -> None:
        assert defaults.num_samples is None

    def test_default_num_chains_is_none(self, defaults: argparse.Namespace) -> None:
        assert defaults.num_chains is None

    def test_default_verbose_is_zero(self, defaults: argparse.Namespace) -> None:
        assert defaults.verbose == 0

    def test_default_quiet_is_false(self, defaults: argparse.Namespace) -> None:
        assert defaults.quiet is False

    def test_default_threads_is_none(self, defaults: argparse.Namespace) -> None:
        assert defaults.threads is None

    def test_default_no_jit_is_false(self, defaults: argparse.Namespace) -> None:
        assert defaults.no_jit is False

    def test_default_plot_is_true(self, defaults: argparse.Namespace) -> None:
        assert defaults.plot is True

    # --- Explicit overrides ---

    def test_method_cmc(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--method", "cmc"])
        assert args.method == "cmc"

    def test_method_both(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-m", "both"])
        assert args.method == "both"

    def test_output_override(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-o", "/results"])
        assert args.output == Path("/results")

    def test_output_format_json(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--output-format", "json"])
        assert args.output_format == "json"

    def test_phi_single_value(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--phi", "0.5"])
        assert args.phi == [0.5]

    def test_phi_multiple_values(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--phi", "0.1", "0.5", "1.0"])
        assert args.phi == [0.1, 0.5, 1.0]

    def test_multistart_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--multistart"])
        assert args.multistart is True

    def test_multistart_n_override(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--multistart-n", "25"])
        assert args.multistart_n == 25

    def test_num_samples_override(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--num-samples", "2000"])
        assert args.num_samples == 2000

    def test_num_chains_override(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--num-chains", "4"])
        assert args.num_chains == 4

    def test_threads_override(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--threads", "8"])
        assert args.threads == 8

    def test_no_jit_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--no-jit"])
        assert args.no_jit is True

    def test_verbose_single(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-v"])
        assert args.verbose == 1

    def test_verbose_double(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-vv"])
        assert args.verbose == 2

    def test_verbose_triple(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-vvv"])
        assert args.verbose == 3

    def test_quiet_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--quiet"])
        assert args.quiet is True

    def test_quiet_short_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "-q"])
        assert args.quiet is True

    def test_no_plot_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--config", "t.yaml", "--no-plot"])
        assert args.plot is False


@pytest.mark.unit
class TestArgumentValidation:
    """Tests for invalid arguments and mutual exclusivity."""

    @pytest.fixture()
    def parser(self) -> argparse.ArgumentParser:
        return create_parser()

    def test_invalid_method_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """An unrecognised --method value must be rejected by argparse."""
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--method", "gradient"])

    def test_invalid_output_format_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--output-format", "csv"])

    def test_plot_and_no_plot_mutually_exclusive(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Passing both --plot and --no-plot must fail."""
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--plot", "--no-plot"])

    def test_non_integer_num_chains_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--num-chains", "3.5"])

    def test_non_integer_threads_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--threads", "abc"])

    def test_non_numeric_phi_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--phi", "abc"])

    def test_non_integer_multistart_n_raises_system_exit(
        self, parser: argparse.ArgumentParser
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            parser.parse_args(["--config", "t.yaml", "--multistart-n", "1.5"])


@pytest.mark.unit
class TestValidateArgs:
    """Tests for the validate_args post-parse validation function."""

    @pytest.fixture()
    def parser(self) -> argparse.ArgumentParser:
        return create_parser()

    def test_missing_config_file_raises_file_not_found(
        self, parser: argparse.ArgumentParser
    ) -> None:
        args = parser.parse_args(["--config", "/nonexistent/config.yaml"])
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            validate_args(args)

    def test_existing_config_file_passes(
        self, parser: argparse.ArgumentParser, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "valid.yaml"
        cfg.write_text("method: nlsq\n")
        args = parser.parse_args(["--config", str(cfg)])
        warnings = validate_args(args)
        assert isinstance(warnings, list)

    def test_verbose_and_quiet_produces_warning(
        self, parser: argparse.ArgumentParser, tmp_path: Path
    ) -> None:
        """When both --verbose and --quiet are given, a warning is emitted
        and verbose is reset to 0."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("")
        args = parser.parse_args(["--config", str(cfg), "-v", "--quiet"])
        warnings = validate_args(args)
        assert len(warnings) == 1
        assert "quiet" in warnings[0].lower()
        assert args.verbose == 0

    def test_quiet_only_no_warning(
        self, parser: argparse.ArgumentParser, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("")
        args = parser.parse_args(["--config", str(cfg), "--quiet"])
        warnings = validate_args(args)
        assert warnings == []

    def test_verbose_only_no_warning(
        self, parser: argparse.ArgumentParser, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("")
        args = parser.parse_args(["--config", str(cfg), "-vv"])
        warnings = validate_args(args)
        assert warnings == []
        assert args.verbose == 2


@pytest.mark.unit
class TestConfigFromArgs:
    """Tests that CLI overrides are correctly applied to ConfigManager."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        """Build a minimal Namespace with sensible defaults."""
        defaults: dict[str, object] = {
            "config": Path("dummy.yaml"),
            "method": "nlsq",
            "output": None,
            "output_format": "both",
            "phi": None,
            "multistart": False,
            "multistart_n": 10,
            "num_samples": None,
            "num_chains": None,
            "verbose": 0,
            "quiet": False,
            "threads": None,
            "no_jit": False,
            "plot": True,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("heterodyne.cli.config_handling.ConfigManager")
    def test_num_samples_override_applied(self, mock_cm_cls: object) -> None:
        """--num-samples value should be written via update_optimization_config."""
        from unittest.mock import MagicMock

        from heterodyne.cli.config_handling import apply_cli_overrides

        mgr = MagicMock()
        args = self._make_args(num_samples=5000)

        apply_cli_overrides(mgr, args)

        mgr.update_optimization_config.assert_any_call("cmc", "num_samples", 5000)

    @patch("heterodyne.cli.config_handling.ConfigManager")
    def test_num_chains_override_applied(self, mock_cm_cls: object) -> None:
        from unittest.mock import MagicMock

        from heterodyne.cli.config_handling import apply_cli_overrides

        mgr = MagicMock()
        args = self._make_args(num_chains=8)

        apply_cli_overrides(mgr, args)

        mgr.update_optimization_config.assert_any_call("cmc", "num_chains", 8)

    @patch("heterodyne.cli.config_handling.ConfigManager")
    def test_multistart_override_applied(self, mock_cm_cls: object) -> None:
        from unittest.mock import MagicMock

        from heterodyne.cli.config_handling import apply_cli_overrides

        mgr = MagicMock()
        args = self._make_args(multistart=True, multistart_n=50)

        apply_cli_overrides(mgr, args)

        mgr.update_optimization_config.assert_any_call("nlsq", "multistart", True)
        mgr.update_optimization_config.assert_any_call("nlsq", "multistart_n", 50)

    @patch("heterodyne.cli.config_handling.ConfigManager")
    def test_no_override_when_none(self, mock_cm_cls: object) -> None:
        """When CLI args are at defaults (None/False), no config keys are set."""
        from unittest.mock import MagicMock

        from heterodyne.cli.config_handling import apply_cli_overrides

        mgr = MagicMock()
        args = self._make_args()

        apply_cli_overrides(mgr, args)

        mgr.update_optimization_config.assert_not_called()
