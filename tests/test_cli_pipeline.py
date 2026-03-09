"""Tests for CLI components (lightweight, no subprocess invocation).

Verifies that CLI modules import cleanly and that the argument parser
produces correct defaults and rejects invalid inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestCliImports:
    """Verify that CLI modules can be imported without error."""

    def test_import_args_parser(self) -> None:
        from heterodyne.cli import args_parser  # noqa: F401

    def test_import_main(self) -> None:
        from heterodyne.cli import main  # noqa: F401

    def test_import_xla_config(self) -> None:
        from heterodyne.cli import xla_config  # noqa: F401

    def test_create_parser_returns_argument_parser(self) -> None:
        from heterodyne.cli.args_parser import create_parser

        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)


# ---------------------------------------------------------------------------
# Argument parser defaults
# ---------------------------------------------------------------------------


class TestCreateParser:
    @pytest.fixture(autouse=True)
    def _parser(self) -> None:
        from heterodyne.cli.args_parser import create_parser

        self.parser = create_parser()

    def test_method_default_is_nlsq(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.method == "nlsq"

    def test_output_format_default_is_both(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.output_format == "both"

    def test_verbose_default_is_zero(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.verbose == 0

    def test_quiet_default_is_false(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.quiet is False

    def test_plot_default_is_true(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.plot is True

    def test_no_plot_disables_plot(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg), "--no-plot"])
        assert args.plot is False

    def test_multistart_default_is_false(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.multistart is False

    def test_multistart_n_default(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.multistart_n == 10

    def test_method_choices(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        for method in ("nlsq", "cmc", "both"):
            args = self.parser.parse_args(["--config", str(cfg), "--method", method])
            assert args.method == method

    def test_invalid_method_raises_system_exit(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--config", str(cfg), "--method", "invalid"])

    def test_verbose_count_accumulates(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg), "-vv"])
        assert args.verbose == 2

    def test_phi_accepts_multiple_floats(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(
            ["--config", str(cfg), "--phi", "0.0", "45.0", "90.0"]
        )
        assert args.phi == pytest.approx([0.0, 45.0, 90.0])

    def test_phi_default_is_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        args = self.parser.parse_args(["--config", str(cfg)])
        assert args.phi is None

    def test_output_path_parsed_as_path(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        out = tmp_path / "results"
        args = self.parser.parse_args(
            ["--config", str(cfg), "--output", str(out)]
        )
        assert isinstance(args.output, Path)


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------


class TestValidateArgs:
    def test_missing_config_raises_file_not_found(self) -> None:
        from heterodyne.cli.args_parser import create_parser, validate_args

        parser = create_parser()
        args = parser.parse_args(
            ["--config", "/nonexistent/path/cfg.yaml"]
        )
        with pytest.raises(FileNotFoundError):
            validate_args(args)

    def test_valid_config_returns_empty_warnings(self, tmp_path: Path) -> None:
        from heterodyne.cli.args_parser import create_parser, validate_args

        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        parser = create_parser()
        args = parser.parse_args(["--config", str(cfg)])
        warnings = validate_args(args)
        assert warnings == []

    def test_verbose_and_quiet_generates_warning(self, tmp_path: Path) -> None:
        from heterodyne.cli.args_parser import create_parser, validate_args

        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        parser = create_parser()
        args = parser.parse_args(["--config", str(cfg), "-v", "--quiet"])
        warnings = validate_args(args)
        assert len(warnings) == 1
        assert "quiet" in warnings[0].lower() or "verbose" in warnings[0].lower()

    def test_verbose_suppressed_when_quiet_wins(self, tmp_path: Path) -> None:
        from heterodyne.cli.args_parser import create_parser, validate_args

        cfg = tmp_path / "cfg.yaml"
        cfg.touch()
        parser = create_parser()
        args = parser.parse_args(["--config", str(cfg), "-v", "--quiet"])
        validate_args(args)
        assert args.verbose == 0


# ---------------------------------------------------------------------------
# XLA config module
# ---------------------------------------------------------------------------


class TestXlaConfig:
    def test_configure_xla_is_callable(self) -> None:
        from heterodyne.cli.xla_config import configure_xla

        assert callable(configure_xla)

    def test_configure_xla_no_error_on_default_args(self) -> None:
        from heterodyne.cli.xla_config import configure_xla

        # Should not raise; JAX may already be initialised
        configure_xla()
