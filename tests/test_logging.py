"""Tests for utils/logging.py module.

Covers configure_logging and ConvergenceLogger for improved coverage.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from heterodyne.utils.logging import (
    ConvergenceLogger,
    configure_logging,
    get_logger,
)

# ============================================================================
# Test get_logger
# ============================================================================


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_caches_loggers(self) -> None:
        """Same name returns same logger instance."""
        name = "test.cached.logger"
        logger1 = get_logger(name)
        logger2 = get_logger(name)
        assert logger1 is logger2

    def test_get_logger_default_name(self) -> None:
        """Default name is 'heterodyne'."""
        logger = get_logger()
        assert logger.name == "heterodyne"


# ============================================================================
# Test configure_logging
# ============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default_level(self) -> None:
        """configure_logging sets INFO level by default."""
        configure_logging()
        logger = logging.getLogger("heterodyne")
        assert logger.level == logging.INFO

    def test_configure_logging_debug_level(self) -> None:
        """configure_logging sets DEBUG level when specified."""
        configure_logging(level="DEBUG")
        logger = logging.getLogger("heterodyne")
        assert logger.level == logging.DEBUG

    def test_configure_logging_with_file(self) -> None:
        """configure_logging creates file handler when log_file specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            configure_logging(level="INFO", log_file=log_path)

            logger = logging.getLogger("heterodyne")

            # Should have at least 2 handlers (console + file)
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) >= 1

            # Write a test message
            logger.info("test message")

            # Verify file was created
            assert log_path.exists()

    def test_configure_logging_creates_parent_dirs(self) -> None:
        """configure_logging creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "subdir" / "test.log"
            configure_logging(level="INFO", log_file=log_path)

            logger = logging.getLogger("heterodyne")
            logger.info("test")

            assert log_path.parent.exists()

    def test_configure_logging_custom_format(self) -> None:
        """configure_logging uses custom format when specified."""
        custom_format = "%(levelname)s - %(message)s"
        configure_logging(format_string=custom_format)

        logger = logging.getLogger("heterodyne")
        # Check that handlers have the custom format
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format

    def test_configure_logging_no_propagation(self) -> None:
        """configure_logging disables propagation to root logger."""
        configure_logging()
        logger = logging.getLogger("heterodyne")
        assert logger.propagate is False


# ============================================================================
# Test ConvergenceLogger
# ============================================================================


class TestConvergenceLogger:
    """Tests for ConvergenceLogger class."""

    @pytest.fixture
    def conv_logger(self) -> ConvergenceLogger:
        """Create a ConvergenceLogger for testing."""
        return ConvergenceLogger()

    def test_init_default_logger(self) -> None:
        """ConvergenceLogger uses optimization logger by default."""
        conv = ConvergenceLogger()
        assert conv.logger.name == "heterodyne.optimization"

    def test_init_custom_logger(self) -> None:
        """ConvergenceLogger accepts custom logger."""
        custom = logging.getLogger("custom.logger")
        conv = ConvergenceLogger(logger=custom)
        assert conv.logger is custom

    def test_log_iteration_basic(self, conv_logger: ConvergenceLogger) -> None:
        """log_iteration logs basic iteration info."""
        conv_logger.log_iteration(iteration=10, loss=1.5e-4)
        assert conv_logger._iteration == 10

    def test_log_iteration_with_gradient(self, conv_logger: ConvergenceLogger) -> None:
        """log_iteration logs with gradient norm."""
        conv_logger.log_iteration(iteration=5, loss=1e-3, gradient_norm=0.01)
        assert conv_logger._iteration == 5

    def test_log_iteration_with_step_size(self, conv_logger: ConvergenceLogger) -> None:
        """log_iteration logs with step size."""
        conv_logger.log_iteration(iteration=3, loss=2e-3, step_size=0.1)
        assert conv_logger._iteration == 3

    def test_log_iteration_full(self, conv_logger: ConvergenceLogger) -> None:
        """log_iteration logs with all optional parameters."""
        conv_logger.log_iteration(
            iteration=20, loss=5e-5, gradient_norm=1e-4, step_size=0.05
        )
        assert conv_logger._iteration == 20

    def test_log_convergence(self, conv_logger: ConvergenceLogger) -> None:
        """log_convergence logs convergence result."""
        conv_logger._iteration = 100
        conv_logger.log_convergence(reason="tolerance reached", final_loss=1e-6)
        # No assertion needed, just verify no errors

    def test_log_diagnostic_pass(self, conv_logger: ConvergenceLogger) -> None:
        """log_diagnostic logs passing metric."""
        conv_logger.log_diagnostic(metric_name="R-hat", value=1.05, threshold=1.1)
        # No assertion needed, just verify no errors

    def test_log_diagnostic_warn(self, conv_logger: ConvergenceLogger) -> None:
        """log_diagnostic logs warning for failing metric."""
        conv_logger.log_diagnostic(metric_name="R-hat", value=1.5, threshold=1.1)
        # No assertion needed, just verify no errors
