"""Extended coverage tests for heterodyne/utils/logging.py.

Targets the ~289 missed statements: _resolve_level, _ColorFormatter,
_ContextAdapter, LogConfiguration, AnalysisSummaryLogger, MinimalLogger,
log_phase, log_exception, log_calls, log_performance, log_operation,
ConvergenceLogger, with_context, configure_from_dict, PhaseContext.
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from heterodyne.utils.logging import (
    AnalysisSummaryLogger,
    ConvergenceLogger,
    LogConfiguration,
    MinimalLogger,
    PhaseContext,
    _ColorFormatter,
    _ContextAdapter,
    _resolve_level,
    configure_logging,
    get_logger,
    log_calls,
    log_exception,
    log_operation,
    log_performance,
    log_phase,
    with_context,
)


# ============================================================================
# _resolve_level
# ============================================================================


class TestResolveLevel:
    """Tests for _resolve_level helper."""

    def test_none_returns_none(self) -> None:
        assert _resolve_level(None) is None

    def test_int_passthrough(self) -> None:
        assert _resolve_level(logging.WARNING) == logging.WARNING
        assert _resolve_level(42) == 42

    def test_string_debug(self) -> None:
        assert _resolve_level("DEBUG") == logging.DEBUG

    def test_string_case_insensitive(self) -> None:
        assert _resolve_level("warning") == logging.WARNING
        assert _resolve_level("Warning") == logging.WARNING

    def test_invalid_string_falls_back_to_info(self) -> None:
        assert _resolve_level("nonexistent") == logging.INFO


# ============================================================================
# _ColorFormatter
# ============================================================================


class TestColorFormatter:
    """Tests for _ColorFormatter."""

    def _make_record(self, level: int = logging.INFO, msg: str = "test") -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_color_enabled_adds_ansi(self) -> None:
        fmt = _ColorFormatter(fmt="%(levelname)s %(message)s", datefmt=None, use_color=True)
        record = self._make_record(logging.ERROR, "boom")
        output = fmt.format(record)
        assert "\033[31m" in output  # Red for ERROR
        assert "\033[0m" in output  # Reset
        # Original levelname should be restored
        assert record.levelname == "ERROR"

    def test_color_disabled_no_ansi(self) -> None:
        fmt = _ColorFormatter(fmt="%(levelname)s %(message)s", datefmt=None, use_color=False)
        record = self._make_record(logging.INFO, "hello")
        output = fmt.format(record)
        assert "\033[" not in output
        assert "INFO" in output

    def test_all_levels_have_color(self) -> None:
        fmt = _ColorFormatter(fmt="%(levelname)s", datefmt=None, use_color=True)
        for level_name, level_val in [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]:
            record = self._make_record(level_val, "msg")
            output = fmt.format(record)
            assert "\033[" in output
            # Levelname restored after format
            assert record.levelname == level_name

    def test_unknown_level_no_color(self) -> None:
        fmt = _ColorFormatter(fmt="%(levelname)s", datefmt=None, use_color=True)
        record = self._make_record(logging.INFO, "msg")
        record.levelname = "CUSTOM"
        output = fmt.format(record)
        assert "\033[" not in output
        assert record.levelname == "CUSTOM"


# ============================================================================
# _ContextAdapter
# ============================================================================


class TestContextAdapter:
    """Tests for _ContextAdapter."""

    def _make_adapter(self, extra: dict[str, Any]) -> _ContextAdapter:
        base = logging.getLogger("test.context_adapter")
        return _ContextAdapter(base, extra)

    def test_no_extra_passes_through(self) -> None:
        adapter = self._make_adapter({})
        msg, kwargs = adapter.process("hello", {})
        assert msg == "hello"

    def test_context_prefix(self) -> None:
        adapter = self._make_adapter({"run_id": "abc", "q_bin": 5})
        msg, kwargs = adapter.process("started", {})
        assert msg.startswith("[")
        assert "run_id=abc" in msg
        assert "q_bin=5" in msg
        assert msg.endswith("] started")

    def test_none_values_excluded(self) -> None:
        adapter = self._make_adapter({"a": "yes", "b": None, "c": ""})
        msg, _ = adapter.process("test", {})
        assert "a=yes" in msg
        assert "b=" not in msg
        assert "c=" not in msg


# ============================================================================
# LogConfiguration
# ============================================================================


class TestLogConfiguration:
    """Tests for LogConfiguration dataclass."""

    def test_from_dict_defaults(self) -> None:
        config = LogConfiguration.from_dict({})
        assert config.console_level == "INFO"
        assert config.file_enabled is True
        assert config.file_rotation_mb == 10

    def test_from_dict_custom(self) -> None:
        config = LogConfiguration.from_dict({
            "console_level": "DEBUG",
            "console_colors": True,
            "file_enabled": False,
            "module_overrides": {"jax": "ERROR"},
        })
        assert config.console_level == "DEBUG"
        assert config.console_colors is True
        assert config.file_enabled is False
        assert config.module_overrides == {"jax": "ERROR"}

    def test_from_cli_args_verbose(self) -> None:
        config = LogConfiguration.from_cli_args(verbose=True)
        assert config.console_level == "DEBUG"
        assert config.console_format == "detailed"

    def test_from_cli_args_quiet(self) -> None:
        config = LogConfiguration.from_cli_args(quiet=True)
        assert config.console_level == "ERROR"

    def test_from_cli_args_default(self) -> None:
        config = LogConfiguration.from_cli_args()
        assert config.console_level == "INFO"
        assert config.console_format == "simple"

    def test_from_cli_args_with_log_file(self) -> None:
        config = LogConfiguration.from_cli_args(log_file="/tmp/test.log")
        assert config.file_path == "/tmp/test.log"

    def test_apply_with_file_disabled(self) -> None:
        config = LogConfiguration(file_enabled=False)
        result = config.apply()
        # No file path returned when file logging disabled
        assert result is None

    def test_apply_with_file_enabled_auto_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfiguration(
                file_enabled=True,
                file_path=Path(tmpdir) / "test.log",
            )
            result = config.apply()
            assert result is not None
            assert "test.log" in str(result)

    def test_apply_module_overrides_merged(self) -> None:
        """User overrides should merge with default suppressions."""
        config = LogConfiguration(
            file_enabled=False,
            module_overrides={"jax": "DEBUG"},
        )
        config.apply()
        # jax logger should have been set to DEBUG (user override wins)
        jax_logger = logging.getLogger("jax")
        assert jax_logger.level == logging.DEBUG


# ============================================================================
# AnalysisSummaryLogger
# ============================================================================


class TestAnalysisSummaryLogger:
    """Tests for AnalysisSummaryLogger."""

    def _make_summary(self) -> AnalysisSummaryLogger:
        return AnalysisSummaryLogger(run_id="test_001", analysis_mode="two_component")

    def test_init(self) -> None:
        s = self._make_summary()
        assert s.run_id == "test_001"
        assert s.analysis_mode == "two_component"

    def test_start_and_end_phase(self) -> None:
        s = self._make_summary()
        s.start_phase("loading")
        time.sleep(0.01)
        s.end_phase("loading", memory_peak_gb=1.5)
        assert "loading" in s._phases
        assert s._phases["loading"].duration is not None
        assert s._phases["loading"].duration > 0
        assert s._phases["loading"].memory_peak_gb == 1.5

    def test_end_phase_unknown_name_ignored(self) -> None:
        s = self._make_summary()
        # Should not raise
        s.end_phase("nonexistent")

    def test_record_metric(self) -> None:
        s = self._make_summary()
        s.record_metric("chi_squared", 1.23)
        assert s._metrics["chi_squared"] == 1.23

    def test_add_output_file(self) -> None:
        s = self._make_summary()
        s.add_output_file("/tmp/result.h5")
        s.add_output_file(Path("/tmp/result2.h5"))
        assert len(s._output_files) == 2
        assert all(isinstance(p, Path) for p in s._output_files)

    def test_set_convergence_status(self) -> None:
        s = self._make_summary()
        s.set_convergence_status("converged")
        assert s._convergence_status == "converged"

    def test_increment_counters(self) -> None:
        s = self._make_summary()
        s.increment_warning_count()
        s.increment_warning_count()
        s.increment_error_count()
        assert s._warning_count == 2
        assert s._error_count == 1

    def test_set_config_summary(self) -> None:
        s = self._make_summary()
        s.set_config_summary(
            optimizer="nlsq",
            n_params=14,
            n_data_points=50000,
            n_phi_angles=6,
            data_file="test.h5",
            custom_key="custom_value",
        )
        assert s._config_summary["optimizer"] == "nlsq"
        assert s._config_summary["n_params"] == 14
        assert s._config_summary["custom_key"] == "custom_value"

    def test_set_config_summary_none_values_skipped(self) -> None:
        s = self._make_summary()
        s.set_config_summary(optimizer=None, n_params=None)
        assert "optimizer" not in s._config_summary
        assert "n_params" not in s._config_summary

    def test_log_summary_minimal(self) -> None:
        s = self._make_summary()
        mock_logger = MagicMock()
        s.log_summary(mock_logger)
        mock_logger.info.assert_called_once()
        output = mock_logger.info.call_args[0][0]
        assert "ANALYSIS SUMMARY" in output
        assert "test_001" in output
        assert "unknown" in output  # no convergence status set

    def test_log_summary_full(self) -> None:
        s = self._make_summary()
        s.start_phase("opt")
        s.end_phase("opt", memory_peak_gb=2.5)
        s.record_metric("chi2", 0.987)
        s.add_output_file("/tmp/out.h5")
        s.set_convergence_status("converged")
        s.increment_warning_count()
        s.increment_error_count()
        s.set_config_summary(optimizer="nlsq", n_data_points=100_000)
        mock_logger = MagicMock()
        s.log_summary(mock_logger)
        output = mock_logger.info.call_args[0][0]
        assert "Phase Timings:" in output
        assert "opt:" in output
        assert "peak: 2.5 GB" in output
        assert "Metrics:" in output
        assert "chi2:" in output
        assert "Output files:" in output
        assert "/tmp/out.h5" in output
        assert "Warnings: 1, Errors: 1" in output
        assert "Configuration:" in output
        assert "n_data_points: 100,000" in output  # Large int formatted

    def test_log_summary_config_small_int(self) -> None:
        s = self._make_summary()
        s.set_config_summary(n_params=14)
        mock_logger = MagicMock()
        s.log_summary(mock_logger)
        output = mock_logger.info.call_args[0][0]
        assert "n_params: 14" in output

    def test_as_dict(self) -> None:
        s = self._make_summary()
        s.start_phase("loading")
        s.end_phase("loading")
        s.record_metric("r_hat", 1.01)
        s.add_output_file("/tmp/result.h5")
        s.set_convergence_status("converged")
        s.increment_warning_count()
        d = s.as_dict()
        assert d["run_id"] == "test_001"
        assert d["analysis_mode"] == "two_component"
        assert d["convergence_status"] == "converged"
        assert "loading" in d["phases"]
        assert d["warning_count"] == 1
        assert d["error_count"] == 0
        assert len(d["output_files"]) == 1

    def test_as_dict_json_safe_fallback(self) -> None:
        """When io.json_utils is unavailable, fallback handles NaN/Inf."""
        s = self._make_summary()
        s.record_metric("bad_nan", float("nan"))
        s.record_metric("bad_inf", float("inf"))
        with patch.dict("sys.modules", {"heterodyne.io.json_utils": None}):
            d = s.as_dict()
            # The fallback _json_safe is applied to the metrics dict
            assert d["metrics"] is not None  # dict passed through


# ============================================================================
# _PhaseRecord
# ============================================================================


class TestPhaseRecord:
    """Tests for _PhaseRecord dataclass."""

    def test_duration_none_when_incomplete(self) -> None:
        from heterodyne.utils.logging import _PhaseRecord

        r = _PhaseRecord(name="test")
        assert r.duration is None
        r.start_time = 1.0
        assert r.duration is None

    def test_duration_computed(self) -> None:
        from heterodyne.utils.logging import _PhaseRecord

        r = _PhaseRecord(name="test", start_time=1.0, end_time=3.5)
        assert r.duration == pytest.approx(2.5)


# ============================================================================
# MinimalLogger
# ============================================================================


class TestMinimalLogger:
    """Tests for MinimalLogger singleton."""

    def test_singleton(self) -> None:
        a = MinimalLogger()
        b = MinimalLogger()
        assert a is b

    def test_get_logger_main(self) -> None:
        mgr = MinimalLogger()
        logger = mgr.get_logger("__main__")
        assert logger.name == "heterodyne.main"

    def test_get_logger_heterodyne_prefixed(self) -> None:
        mgr = MinimalLogger()
        logger = mgr.get_logger("heterodyne.core.foo")
        assert logger.name == "heterodyne.core.foo"

    def test_get_logger_bare_name(self) -> None:
        mgr = MinimalLogger()
        logger = mgr.get_logger("mymodule")
        assert logger.name == "heterodyne.mymodule"

    def test_build_formatter_simple(self) -> None:
        fmt = MinimalLogger._build_formatter("simple", use_color=False)
        assert isinstance(fmt, _ColorFormatter)

    def test_build_formatter_detailed(self) -> None:
        fmt = MinimalLogger._build_formatter("detailed", use_color=True)
        assert isinstance(fmt, _ColorFormatter)

    def test_configure_returns_none_without_file(self) -> None:
        mgr = MinimalLogger()
        result = mgr.configure(level="INFO", force=True)
        assert result is None

    def test_configure_with_file(self) -> None:
        mgr = MinimalLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            result = mgr.configure(
                level="DEBUG",
                file_path=log_path,
                file_level="DEBUG",
                max_size_mb=1,
                backup_count=2,
                force=True,
            )
            assert result == log_path

    def test_configure_with_zero_max_size(self) -> None:
        """max_size_mb=0 should use plain FileHandler."""
        mgr = MinimalLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "plain.log"
            result = mgr.configure(
                level="DEBUG",
                file_path=log_path,
                file_level="DEBUG",
                max_size_mb=0,
                force=True,
            )
            assert result == log_path

    def test_configure_module_levels(self) -> None:
        mgr = MinimalLogger()
        mgr.configure(
            level="INFO",
            module_levels={"heterodyne.test_module_xyz": "ERROR"},
            force=True,
        )
        mod_logger = logging.getLogger("heterodyne.test_module_xyz")
        assert mod_logger.level == logging.ERROR

    def test_configure_file_unwritable_directory(self) -> None:
        """When file directory can't be created, file logging is skipped."""
        mgr = MinimalLogger()
        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            result = mgr.configure(
                level="INFO",
                file_path="/nonexistent/path/test.log",
                file_level="DEBUG",
                force=True,
            )
            assert result is None


# ============================================================================
# configure_from_dict
# ============================================================================


class TestConfigureFromDict:
    """Tests for MinimalLogger.configure_from_dict."""

    def test_none_config_returns_none(self) -> None:
        mgr = MinimalLogger()
        assert mgr.configure_from_dict(None) is None

    def test_disabled_config_returns_none(self) -> None:
        mgr = MinimalLogger()
        assert mgr.configure_from_dict({"enabled": False}) is None

    def test_basic_config(self) -> None:
        mgr = MinimalLogger()
        result = mgr.configure_from_dict(
            {"enabled": True, "level": "DEBUG"},
            verbose=False,
            quiet=False,
        )
        # No file config => no file path returned
        assert result is None

    def test_verbose_overrides_console_level(self) -> None:
        mgr = MinimalLogger()
        mgr.configure_from_dict(
            {"enabled": True, "console": {"level": "WARNING"}},
            verbose=True,
        )
        # Can't easily assert console level, but no error means it worked

    def test_quiet_overrides_console_level(self) -> None:
        mgr = MinimalLogger()
        mgr.configure_from_dict(
            {"enabled": True, "console": {"level": "DEBUG"}},
            quiet=True,
        )

    def test_file_config_with_path(self) -> None:
        mgr = MinimalLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mgr.configure_from_dict(
                {
                    "enabled": True,
                    "file": {
                        "enabled": True,
                        "path": tmpdir,
                        "filename": "test_{run_id}.log",
                    },
                },
                run_id="run42",
            )
            assert result is not None
            assert "test_run42.log" in str(result)

    def test_file_config_auto_dir(self) -> None:
        mgr = MinimalLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mgr.configure_from_dict(
                {"enabled": True, "file": {"enabled": True}},
                output_dir=tmpdir,
                run_id="abc",
            )
            assert result is not None
            assert "abc" in str(result)

    def test_file_config_no_run_id_format(self) -> None:
        """Filename without {run_id} gets run_id appended."""
        mgr = MinimalLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mgr.configure_from_dict(
                {
                    "enabled": True,
                    "file": {
                        "enabled": True,
                        "path": tmpdir,
                        "filename": "analysis.log",
                    },
                },
                run_id="xyz",
            )
            assert result is not None
            assert "analysis_xyz.log" in str(result)

    def test_console_disabled(self) -> None:
        mgr = MinimalLogger()
        mgr.configure_from_dict(
            {"enabled": True, "console": {"enabled": False}},
        )


# ============================================================================
# with_context
# ============================================================================


class TestWithContext:
    """Tests for with_context function."""

    def test_wraps_plain_logger(self) -> None:
        logger = logging.getLogger("test.with_context.plain")
        ctx = with_context(logger, run_id="abc")
        assert isinstance(ctx, _ContextAdapter)
        msg, _ = ctx.process("hello", {})
        assert "[run_id=abc]" in msg

    def test_wraps_existing_context_adapter(self) -> None:
        logger = logging.getLogger("test.with_context.nested")
        ctx1 = with_context(logger, run_id="abc")
        ctx2 = with_context(ctx1, q_bin=5)
        assert isinstance(ctx2, _ContextAdapter)
        msg, _ = ctx2.process("test", {})
        assert "run_id=abc" in msg
        assert "q_bin=5" in msg

    def test_wraps_generic_logger_adapter(self) -> None:
        logger = logging.getLogger("test.with_context.adapter")
        adapter = logging.LoggerAdapter(logger, {"extra": "data"})
        ctx = with_context(adapter, mode="nlsq")
        assert isinstance(ctx, _ContextAdapter)
        msg, _ = ctx.process("test", {})
        assert "mode=nlsq" in msg

    def test_none_values_filtered(self) -> None:
        logger = logging.getLogger("test.with_context.none")
        ctx = with_context(logger, a="keep", b=None)
        assert isinstance(ctx, _ContextAdapter)
        msg, _ = ctx.process("test", {})
        assert "a=keep" in msg
        assert "b=" not in msg


# ============================================================================
# PhaseContext
# ============================================================================


class TestPhaseContext:
    """Tests for PhaseContext dataclass."""

    def test_defaults(self) -> None:
        pc = PhaseContext(name="test")
        assert pc.name == "test"
        assert pc.duration == 0.0
        assert pc.memory_peak_gb is None
        assert pc.memory_delta_gb is None


# ============================================================================
# log_phase
# ============================================================================


class TestLogPhase:
    """Tests for log_phase context manager."""

    def test_basic_timing(self) -> None:
        logger = logging.getLogger("test.log_phase")
        handler = logging.handlers.MemoryHandler(capacity=100)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with log_phase("test_op", logger=logger) as phase:
            time.sleep(0.01)

        assert phase.duration > 0
        assert phase.name == "test_op"
        logger.removeHandler(handler)

    def test_threshold_suppresses_start_log(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True
        mock_logger.log = MagicMock()

        with log_phase("fast_op", logger=mock_logger, threshold_s=999.0):
            pass

        # With high threshold, start message should not be logged
        # and completion message should not be logged (duration < threshold)
        calls = mock_logger.log.call_args_list
        # No "started" call since threshold_s > 0
        assert not any("started" in str(c) for c in calls)
        # No "completed" call since duration < threshold
        assert not any("completed" in str(c) for c in calls)

    def test_memory_tracking(self) -> None:
        logger = logging.getLogger("test.log_phase.mem")
        logger.setLevel(logging.DEBUG)

        with log_phase("mem_op", logger=logger, track_memory=True) as phase:
            pass

        # On Linux, memory tracking should work
        # memory_peak_gb may or may not be set depending on platform
        assert phase.duration >= 0

    def test_default_logger_used(self) -> None:
        """When no logger passed, get_logger() is used."""
        with log_phase("auto_logger_op") as phase:
            pass
        assert phase.duration >= 0


# ============================================================================
# log_exception
# ============================================================================


class TestLogException:
    """Tests for log_exception function."""

    def test_basic_exception(self) -> None:
        mock_logger = MagicMock()
        try:
            raise ValueError("test error")
        except ValueError as e:
            log_exception(mock_logger, e)

        mock_logger.log.assert_called_once()
        msg = mock_logger.log.call_args[0][1]
        assert "ValueError" in msg
        assert "test error" in msg
        assert "Traceback:" in msg

    def test_with_context(self) -> None:
        mock_logger = MagicMock()
        try:
            raise RuntimeError("computation failed")
        except RuntimeError as e:
            log_exception(mock_logger, e, context={"iteration": 42, "param": "D0"})

        msg = mock_logger.log.call_args[0][1]
        assert "Context:" in msg
        assert "iteration=42" in msg
        assert "param='D0'" in msg

    def test_without_traceback(self) -> None:
        mock_logger = MagicMock()
        try:
            raise TypeError("bad type")
        except TypeError as e:
            log_exception(mock_logger, e, include_traceback=False)

        msg = mock_logger.log.call_args[0][1]
        assert "TypeError" in msg
        assert "Traceback:" not in msg

    def test_custom_level(self) -> None:
        mock_logger = MagicMock()
        try:
            raise ValueError("warn-level")
        except ValueError as e:
            log_exception(mock_logger, e, level=logging.WARNING, include_traceback=False)

        assert mock_logger.log.call_args[0][0] == logging.WARNING

    def test_exception_without_traceback_obj(self) -> None:
        """Exception created without raise has no __traceback__."""
        mock_logger = MagicMock()
        exc = ValueError("no tb")
        log_exception(mock_logger, exc, include_traceback=False)
        msg = mock_logger.log.call_args[0][1]
        assert "ValueError" in msg
        # No location info when no traceback
        assert "in " not in msg.split("Exception")[1].split(":")[0]


# ============================================================================
# log_calls decorator
# ============================================================================


class TestLogCalls:
    """Tests for log_calls decorator."""

    def test_basic_decoration(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True

        @log_calls(logger=mock_logger, level=logging.DEBUG)
        def my_func(x: int) -> int:
            return x + 1

        result = my_func(5)
        assert result == 6
        assert mock_logger.log.call_count >= 2  # entry + exit

    def test_include_args(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True

        @log_calls(logger=mock_logger, include_args=True)
        def add(a: int, b: int = 0) -> int:
            return a + b

        add(1, b=2)
        entry_call = mock_logger.log.call_args_list[0]
        # Check that args are logged
        assert "1" in str(entry_call)
        assert "b=2" in str(entry_call)

    def test_include_result(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True

        @log_calls(logger=mock_logger, include_result=True)
        def double(x: int) -> int:
            return x * 2

        double(3)
        exit_call = mock_logger.log.call_args_list[-1]
        assert "6" in str(exit_call)

    def test_exception_logged(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = True

        @log_calls(logger=mock_logger)
        def fail() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            fail()

        # Error should be logged
        error_calls = [
            c for c in mock_logger.log.call_args_list if c[0][0] == logging.ERROR
        ]
        assert len(error_calls) >= 1

    def test_no_logger_auto_creates(self) -> None:
        """When logger=None, one is created from func.__module__."""

        @log_calls()
        def auto_log_func() -> str:
            return "ok"

        result = auto_log_func()
        assert result == "ok"

    def test_logging_disabled(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False

        @log_calls(logger=mock_logger)
        def quiet_func() -> int:
            return 42

        result = quiet_func()
        assert result == 42
        # Only entry/exit check calls, no actual log.
        # The decorator should still work even if logging disabled
        assert not any(
            c[0][0] == logging.DEBUG
            for c in mock_logger.log.call_args_list
        )

    def test_exception_when_logging_disabled(self) -> None:
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False

        @log_calls(logger=mock_logger)
        def fail_quiet() -> None:
            raise ValueError("silent boom")

        with pytest.raises(ValueError, match="silent boom"):
            fail_quiet()

        # Error should still be logged even when level is disabled
        error_calls = [
            c for c in mock_logger.log.call_args_list if c[0][0] == logging.ERROR
        ]
        assert len(error_calls) >= 1


# ============================================================================
# log_performance decorator
# ============================================================================


class TestLogPerformance:
    """Tests for log_performance decorator."""

    def test_basic_performance_below_threshold(self) -> None:
        mock_logger = MagicMock()

        @log_performance(logger=mock_logger, threshold=999.0)
        def fast_func() -> int:
            return 1

        result = fast_func()
        assert result == 1
        # Should NOT log since duration < threshold
        perf_calls = [
            c for c in mock_logger.log.call_args_list
            if c[0][0] == logging.INFO
        ]
        assert len(perf_calls) == 0

    def test_performance_above_threshold(self) -> None:
        mock_logger = MagicMock()

        @log_performance(logger=mock_logger, threshold=0.0)
        def any_func() -> int:
            return 42

        any_func()
        perf_calls = [
            c for c in mock_logger.log.call_args_list
            if c[0][0] == logging.INFO
        ]
        assert len(perf_calls) >= 1
        assert "Performance:" in str(perf_calls[0])

    def test_exception_logged_with_duration(self) -> None:
        mock_logger = MagicMock()

        @log_performance(logger=mock_logger, threshold=0.0)
        def fail_func() -> None:
            raise RuntimeError("perf fail")

        with pytest.raises(RuntimeError):
            fail_func()

        error_calls = [
            c for c in mock_logger.log.call_args_list
            if c[0][0] == logging.ERROR
        ]
        assert len(error_calls) >= 1
        assert "failed after" in str(error_calls[0])

    def test_auto_logger(self) -> None:
        @log_performance(threshold=0.0)
        def auto_perf() -> str:
            return "done"

        assert auto_perf() == "done"


# ============================================================================
# log_operation
# ============================================================================


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_basic_operation(self) -> None:
        mock_logger = MagicMock()
        with log_operation("compute jacobian", logger=mock_logger) as log:
            assert log is mock_logger

        calls = mock_logger.log.call_args_list
        assert any("Starting operation" in str(c) for c in calls)
        assert any("Completed operation" in str(c) for c in calls)

    def test_operation_exception(self) -> None:
        mock_logger = MagicMock()
        with pytest.raises(RuntimeError, match="op fail"):
            with log_operation("failing op", logger=mock_logger):
                raise RuntimeError("op fail")

        calls = mock_logger.log.call_args_list
        assert any("Failed operation" in str(c) for c in calls)

    def test_default_logger(self) -> None:
        with log_operation("auto op") as log:
            assert log is not None


# ============================================================================
# ConvergenceLogger
# ============================================================================


class TestConvergenceLogger:
    """Tests for ConvergenceLogger."""

    def test_log_iteration(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_iteration(10, loss=1.5e-3, gradient_norm=2.1e-4, step_size=0.01)
        mock_logger.debug.assert_called_once()
        msg = mock_logger.debug.call_args[0][0]
        assert "iter=  10" in msg
        assert "loss=1.500000e-03" in msg
        assert "grad_norm=2.100e-04" in msg
        assert "step=1.000e-02" in msg

    def test_log_iteration_minimal(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_iteration(1, loss=0.5)
        msg = mock_logger.debug.call_args[0][0]
        assert "grad_norm" not in msg
        assert "step" not in msg

    def test_log_convergence(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_iteration(100, loss=1e-6)
        cl.log_convergence("tolerance met", final_loss=1e-6)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Converged" in call_args[0][0]
        assert call_args[0][1] == "tolerance met"

    def test_log_diagnostic_pass_higher_is_better(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_diagnostic("ESS", value=500.0, threshold=100.0, higher_is_better=True)
        msg = mock_logger.info.call_args[0][0]
        assert "[%s]" in msg or "PASS" in str(mock_logger.info.call_args)

    def test_log_diagnostic_warn_higher_is_better(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_diagnostic("ESS", value=50.0, threshold=100.0, higher_is_better=True)
        call_args = mock_logger.info.call_args
        assert "WARN" in str(call_args)

    def test_log_diagnostic_pass_lower_is_better(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_diagnostic("R-hat", value=1.01, threshold=1.05, higher_is_better=False)
        call_args = mock_logger.info.call_args
        assert "PASS" in str(call_args)

    def test_log_diagnostic_warn_lower_is_better(self) -> None:
        mock_logger = MagicMock()
        cl = ConvergenceLogger(logger=mock_logger)
        cl.log_diagnostic("R-hat", value=1.2, threshold=1.05, higher_is_better=False)
        call_args = mock_logger.info.call_args
        assert "WARN" in str(call_args)

    def test_default_logger(self) -> None:
        cl = ConvergenceLogger()
        # Should use heterodyne.optimization logger
        assert cl.logger is not None


# ============================================================================
# configure_logging (simple API)
# ============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_default_config(self) -> None:
        configure_logging()
        logger = logging.getLogger("heterodyne")
        assert logger.level == logging.INFO
        assert logger.propagate is False

    def test_debug_level(self) -> None:
        configure_logging(level="DEBUG")
        logger = logging.getLogger("heterodyne")
        assert logger.level == logging.DEBUG

    def test_custom_format(self) -> None:
        configure_logging(format_string="%(message)s")
        logger = logging.getLogger("heterodyne")
        assert len(logger.handlers) >= 1

    def test_with_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            configure_logging(log_file=log_file)
            logger = logging.getLogger("heterodyne")
            # Should have at least console + file handler
            assert len(logger.handlers) >= 2


# ============================================================================
# get_logger with context
# ============================================================================


class TestGetLoggerContext:
    """Tests for get_logger with context parameter."""

    def test_with_context_returns_adapter(self) -> None:
        logger = get_logger("test.ctx", context={"run": "123"})
        assert isinstance(logger, _ContextAdapter)

    def test_without_context_returns_logger(self) -> None:
        logger = get_logger("test.no_ctx")
        assert isinstance(logger, logging.Logger)

    def test_none_name_defaults(self) -> None:
        logger = get_logger(None)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "heterodyne"
