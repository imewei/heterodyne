"""Runtime utilities for heterodyne package.

This module provides:
- System validation utilities
- Shell completion scripts
- XLA activation scripts

Example:
    >>> from heterodyne.runtime import run_validation
    >>> results = run_validation(verbose=True)
    >>> all_passed = all(r.success for r in results)
"""

from heterodyne.runtime.utils import (
    SystemValidator,
    ValidationResult,
    Severity,
    run_validation,
)
from heterodyne.runtime.shell import (
    get_completion_script,
    get_xla_config_script,
)

__all__ = [
    # Validation
    "SystemValidator",
    "ValidationResult",
    "Severity",
    "run_validation",
    # Shell
    "get_completion_script",
    "get_xla_config_script",
]
