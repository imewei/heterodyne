"""Structured logging configuration for heterodyne analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


def get_logger(name: str = "heterodyne") -> logging.Logger:
    """Get or create a logger with the given name.

    Delegates directly to ``logging.getLogger`` which already caches
    logger instances by name.

    Args:
        name: Logger name, typically module __name__

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for the heterodyne package.

    Args:
        level: Logging level
        log_file: Optional path to log file
        format_string: Custom format string, or None for default
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    new_handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    new_handlers.append(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        new_handlers.append(file_handler)

    # Configure root heterodyne logger — close existing handlers first
    root_logger = logging.getLogger("heterodyne")
    root_logger.setLevel(getattr(logging, level))

    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    for handler in new_handlers:
        root_logger.addHandler(handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


class ConvergenceLogger:
    """Structured logger for optimization convergence diagnostics."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or get_logger("heterodyne.optimization")
        self._iteration = 0

    def log_iteration(
        self,
        iteration: int,
        loss: float,
        gradient_norm: float | None = None,
        step_size: float | None = None,
    ) -> None:
        """Log optimization iteration metrics."""
        self._iteration = iteration
        msg = f"iter={iteration:4d} | loss={loss:.6e}"
        if gradient_norm is not None:
            msg += f" | grad_norm={gradient_norm:.3e}"
        if step_size is not None:
            msg += f" | step={step_size:.3e}"
        self.logger.debug(msg)

    def log_convergence(self, reason: str, final_loss: float) -> None:
        """Log convergence result."""
        self.logger.info(
            f"Converged: {reason} at iteration {self._iteration} | "
            f"final_loss={final_loss:.6e}"
        )

    def log_diagnostic(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        higher_is_better: bool = True,
    ) -> None:
        """Log diagnostic metric with pass/fail status.

        Args:
            metric_name: Name of the diagnostic metric
            value: Observed value
            threshold: Threshold for pass/fail
            higher_is_better: If True, values above threshold pass (e.g. ESS).
                If False, values below threshold pass (e.g. R-hat).
        """
        if higher_is_better:
            status = "PASS" if value >= threshold else "WARN"
        else:
            status = "PASS" if value <= threshold else "WARN"
        self.logger.info(f"[{status}] {metric_name}: {value:.4f} (threshold: {threshold:.4f})")
