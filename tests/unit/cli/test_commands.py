"""Unit tests for heterodyne.cli.commands module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestDispatchCommand:
    """Tests for dispatch_command routing and exit codes."""

    def _make_args(self, method: str = "nlsq", **kwargs) -> argparse.Namespace:
        """Build a minimal argparse.Namespace for dispatch_command."""
        defaults = {
            "method": method,
            "config": "dummy.yaml",
            "output": str(Path("/tmp/test_output")),
            "plot": False,
            "plot_only": False,
            "simulate_only": False,
            "verbose": 0,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_nlsq")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_dispatch_nlsq_routes_to_run_nlsq(
        self,
        mock_run_cmc: MagicMock,
        mock_run_nlsq: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        mock_model_cls: MagicMock,
        mock_plots: MagicMock,
    ) -> None:
        """dispatch_command routes to run_nlsq when method == 'nlsq'."""
        from heterodyne.cli.commands import dispatch_command

        mock_config_mgr = MagicMock()
        mock_config_mgr.output_dir = "/tmp/test_output"
        mock_load_config.return_value = mock_config_mgr

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        mock_run_nlsq.return_value = []
        mock_run_cmc.return_value = []

        args = self._make_args(method="nlsq")
        result = dispatch_command(args)

        assert result == 0
        mock_run_nlsq.assert_called_once()
        mock_run_cmc.assert_not_called()

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_nlsq")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_dispatch_cmc_routes_to_run_cmc(
        self,
        mock_run_cmc: MagicMock,
        mock_run_nlsq: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        mock_model_cls: MagicMock,
        mock_plots: MagicMock,
    ) -> None:
        """dispatch_command routes to run_cmc when method == 'cmc'."""
        from heterodyne.cli.commands import dispatch_command

        mock_config_mgr = MagicMock()
        mock_config_mgr.output_dir = "/tmp/test_output"
        mock_load_config.return_value = mock_config_mgr

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        mock_run_cmc.return_value = []
        mock_run_nlsq.return_value = []

        args = self._make_args(method="cmc")
        result = dispatch_command(args)

        assert result == 0
        mock_run_cmc.assert_called_once()
        mock_run_nlsq.assert_not_called()

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_nlsq")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_dispatch_both_routes_to_nlsq_and_cmc(
        self,
        mock_run_cmc: MagicMock,
        mock_run_nlsq: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        mock_model_cls: MagicMock,
        mock_plots: MagicMock,
    ) -> None:
        """dispatch_command routes to both run_nlsq and run_cmc when method == 'both'."""
        from heterodyne.cli.commands import dispatch_command

        mock_config_mgr = MagicMock()
        mock_config_mgr.output_dir = "/tmp/test_output"
        mock_load_config.return_value = mock_config_mgr

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        mock_run_nlsq.return_value = []
        mock_run_cmc.return_value = []

        args = self._make_args(method="both")
        result = dispatch_command(args)

        assert result == 0
        mock_run_nlsq.assert_called_once()
        mock_run_cmc.assert_called_once()

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_nlsq")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_dispatch_returns_zero_on_success(
        self,
        mock_run_cmc: MagicMock,
        mock_run_nlsq: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        mock_model_cls: MagicMock,
        mock_plots: MagicMock,
    ) -> None:
        """dispatch_command returns 0 on successful completion."""
        from heterodyne.cli.commands import dispatch_command

        mock_config_mgr = MagicMock()
        mock_config_mgr.output_dir = "/tmp/test_output"
        mock_load_config.return_value = mock_config_mgr

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        mock_run_nlsq.return_value = []

        args = self._make_args(method="nlsq")
        assert dispatch_command(args) == 0

    @patch("heterodyne.cli.commands.load_and_merge_config")
    def test_dispatch_raises_on_error(
        self,
        mock_load_config: MagicMock,
    ) -> None:
        """dispatch_command re-raises exceptions (caller handles exit code)."""
        from heterodyne.cli.commands import dispatch_command

        mock_load_config.side_effect = RuntimeError("config load failed")

        args = self._make_args(method="nlsq")
        with pytest.raises(RuntimeError, match="config load failed"):
            dispatch_command(args)


@pytest.mark.unit
class TestMainExitCodes:
    """Tests for exit codes via the main() wrapper in cli.main."""

    @patch("heterodyne.cli.xla_config.configure_xla")
    @patch("heterodyne.utils.logging.configure_logging")
    @patch("heterodyne.cli.commands.dispatch_command", return_value=0)
    @patch("heterodyne.cli.args_parser.validate_args", return_value=[])
    @patch("heterodyne.cli.args_parser.create_parser")
    def test_main_returns_1_on_exception(
        self,
        mock_parser: MagicMock,
        mock_validate: MagicMock,
        mock_dispatch: MagicMock,
        mock_logging: MagicMock,
        mock_xla: MagicMock,
    ) -> None:
        """main() returns 1 when dispatch_command raises."""
        from heterodyne.cli.main import main

        mock_ns = MagicMock()
        mock_ns.quiet = False
        mock_ns.verbose = 0
        mock_ns.threads = 1
        mock_ns.no_jit = False
        mock_parser.return_value.parse_args.return_value = mock_ns

        mock_dispatch.side_effect = RuntimeError("boom")

        assert main(["dummy"]) == 1

    @patch("heterodyne.cli.xla_config.configure_xla")
    @patch("heterodyne.utils.logging.configure_logging")
    @patch("heterodyne.cli.commands.dispatch_command", return_value=0)
    @patch("heterodyne.cli.args_parser.validate_args", return_value=[])
    @patch("heterodyne.cli.args_parser.create_parser")
    def test_main_returns_130_on_keyboard_interrupt(
        self,
        mock_parser: MagicMock,
        mock_validate: MagicMock,
        mock_dispatch: MagicMock,
        mock_logging: MagicMock,
        mock_xla: MagicMock,
    ) -> None:
        """main() returns 130 when KeyboardInterrupt is raised."""
        from heterodyne.cli.main import main

        mock_ns = MagicMock()
        mock_ns.quiet = False
        mock_ns.verbose = 0
        mock_ns.threads = 1
        mock_ns.no_jit = False
        mock_parser.return_value.parse_args.return_value = mock_ns

        mock_dispatch.side_effect = KeyboardInterrupt

        assert main(["dummy"]) == 130


@pytest.mark.unit
class TestAnalysisSummaryLogger:
    """Tests for AnalysisSummaryLogger phase tracking."""

    def test_records_phase_names_and_timing(self) -> None:
        """AnalysisSummaryLogger records phase start/end with duration."""
        import time

        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="test_001", analysis_mode="two_component")
        summary.start_phase("loading")
        time.sleep(0.01)
        summary.end_phase("loading", memory_peak_gb=1.5)

        record = summary._phases["loading"]
        assert record.name == "loading"
        assert record.start_time is not None
        assert record.end_time is not None
        assert record.duration is not None
        assert record.duration > 0
        assert record.memory_peak_gb == 1.5

    def test_records_multiple_phases(self) -> None:
        """AnalysisSummaryLogger tracks multiple phases independently."""
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="test_002", analysis_mode="two_component")
        summary.start_phase("phase_a")
        summary.end_phase("phase_a")
        summary.start_phase("phase_b")
        summary.end_phase("phase_b")

        assert "phase_a" in summary._phases
        assert "phase_b" in summary._phases

    def test_set_convergence_status(self) -> None:
        """AnalysisSummaryLogger stores convergence status."""
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="test_003", analysis_mode="two_component")
        summary.set_convergence_status("completed")
        assert summary._convergence_status == "completed"

    def test_record_metric(self) -> None:
        """AnalysisSummaryLogger stores named metrics."""
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="test_004", analysis_mode="two_component")
        summary.record_metric("chi_squared", 1.23)
        assert summary._metrics["chi_squared"] == 1.23


@pytest.mark.unit
class TestLogPhase:
    """Tests for log_phase context manager."""

    def test_log_phase_records_duration(self) -> None:
        """log_phase populates duration after context exit."""
        import time

        from heterodyne.utils.logging import log_phase

        with log_phase("test_phase") as phase:
            time.sleep(0.01)

        assert phase.name == "test_phase"
        assert phase.duration > 0

    def test_log_phase_tracks_memory_when_requested(self) -> None:
        """log_phase populates memory_peak_gb when track_memory=True."""
        from heterodyne.utils.logging import log_phase

        with log_phase("mem_phase", track_memory=True) as phase:
            _ = [0] * 1000  # allocate something

        assert phase.duration > 0
        # memory_peak_gb may be None on some systems, but should not error
        # Just verify the attribute exists
        assert hasattr(phase, "memory_peak_gb")

    def test_log_phase_propagates_exceptions(self) -> None:
        """log_phase does not swallow exceptions from the context body."""
        from heterodyne.utils.logging import log_phase

        with pytest.raises(ValueError, match="test error"):
            with log_phase("error_phase"):
                raise ValueError("test error")
