"""System validation utilities for heterodyne installation.

This module provides comprehensive validation of the heterodyne installation,
including environment detection, dependency verification, and JAX configuration
testing.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Severity(Enum):
    """Severity level for validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ValidationResult:
    """Result of a validation test.

    Attributes:
        name: Name of the validation test.
        success: Whether the test passed.
        message: Description of the result.
        severity: Severity level.
        remediation: Suggested fix if the test failed.
        details: Additional details about the result.
    """

    name: str
    success: bool
    message: str
    severity: Severity = Severity.INFO
    remediation: str = ""
    details: dict[str, object] | None = None


class SystemValidator:
    """Comprehensive system validator for heterodyne installation.

    This class runs a series of validation tests to ensure the heterodyne
    package is correctly installed and configured.

    Example:
        >>> validator = SystemValidator(verbose=True)
        >>> results = validator.run_all()
        >>> for r in results:
        ...     print(f"{r.name}: {'PASS' if r.success else 'FAIL'}")
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the validator.

        Args:
            verbose: If True, print detailed output during validation.
        """
        self.verbose = verbose
        self._tests: list[Callable[[], ValidationResult]] = [
            self.test_environment_detection,
            self.test_heterodyne_installation,
            self.test_python_version,
            self.test_jax_installation,
            self.test_jax_cpu_backend,
            self.test_numpy_installation,
            self.test_numpyro_installation,
            self.test_shell_completion,
        ]

    def run_all(self) -> list[ValidationResult]:
        """Run all validation tests.

        Returns:
            List of ValidationResult objects.
        """
        results = []
        for test in self._tests:
            try:
                result = test()
            except Exception as e:
                result = ValidationResult(
                    name=test.__name__.replace("test_", "").replace("_", " ").title(),
                    success=False,
                    message=f"Test raised exception: {e}",
                    severity=Severity.ERROR,
                )
            results.append(result)
            if self.verbose:
                status = "PASS" if result.success else "FAIL"
                print(f"[{status}] {result.name}: {result.message}")
        return results

    def test_environment_detection(self) -> ValidationResult:
        """Test environment detection capabilities."""
        details = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "in_venv": sys.prefix != sys.base_prefix,
        }

        # Check if running in a virtual environment
        in_venv = sys.prefix != sys.base_prefix
        conda_env = os.environ.get("CONDA_PREFIX")

        if in_venv or conda_env:
            return ValidationResult(
                name="Environment Detection",
                success=True,
                message=f"Running in virtual environment: {sys.prefix}",
                severity=Severity.SUCCESS,
                details=details,
            )
        else:
            return ValidationResult(
                name="Environment Detection",
                success=False,
                message="Not running in a virtual environment",
                severity=Severity.WARNING,
                remediation="Consider using 'uv venv' or 'python -m venv' to create an isolated environment",
                details=details,
            )

    def test_heterodyne_installation(self) -> ValidationResult:
        """Test that heterodyne package is properly installed."""
        try:
            import heterodyne

            version = getattr(heterodyne, "__version__", "unknown")

            # Check if CLI commands are available
            heterodyne_cmd = shutil.which("heterodyne")
            config_cmd = shutil.which("heterodyne-config")

            details = {
                "version": version,
                "heterodyne_cli": heterodyne_cmd is not None,
                "config_cli": config_cmd is not None,
            }

            if heterodyne_cmd and config_cmd:
                return ValidationResult(
                    name="Heterodyne Installation",
                    success=True,
                    message=f"heterodyne {version} installed with CLI commands",
                    severity=Severity.SUCCESS,
                    details=details,
                )
            else:
                missing = []
                if not heterodyne_cmd:
                    missing.append("heterodyne")
                if not config_cmd:
                    missing.append("heterodyne-config")
                return ValidationResult(
                    name="Heterodyne Installation",
                    success=False,
                    message=f"Missing CLI commands: {', '.join(missing)}",
                    severity=Severity.WARNING,
                    remediation="Run 'uv pip install -e .' to install in development mode",
                    details=details,
                )

        except ImportError as e:
            return ValidationResult(
                name="Heterodyne Installation",
                success=False,
                message=f"Failed to import heterodyne: {e}",
                severity=Severity.ERROR,
                remediation="Run 'uv sync' to install the package",
            )

    def test_python_version(self) -> ValidationResult:
        """Test Python version compatibility."""
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        details = {
            "version": version_str,
            "implementation": platform.python_implementation(),
        }

        if version_info >= (3, 12):
            return ValidationResult(
                name="Python Version",
                success=True,
                message=f"Python {version_str} meets requirements (>=3.12)",
                severity=Severity.SUCCESS,
                details=details,
            )
        else:
            return ValidationResult(
                name="Python Version",
                success=False,
                message=f"Python {version_str} is below minimum (3.12)",
                severity=Severity.ERROR,
                remediation="Install Python 3.12 or later",
                details=details,
            )

    def test_jax_installation(self) -> ValidationResult:
        """Test JAX installation and version."""
        try:
            import jax

            version = jax.__version__

            # Parse version
            major, minor, *_ = version.split(".")
            major, minor = int(major), int(minor.split("+")[0].split("rc")[0])

            details = {
                "version": version,
                "devices": len(jax.devices()),
            }

            if (major, minor) >= (0, 8):
                return ValidationResult(
                    name="JAX Installation",
                    success=True,
                    message=f"JAX {version} installed (required: >=0.8.2)",
                    severity=Severity.SUCCESS,
                    details=details,
                )
            else:
                return ValidationResult(
                    name="JAX Installation",
                    success=False,
                    message=f"JAX {version} is below minimum (0.8.2)",
                    severity=Severity.ERROR,
                    remediation="Run 'uv pip install jax>=0.8.2'",
                    details=details,
                )

        except ImportError as e:
            return ValidationResult(
                name="JAX Installation",
                success=False,
                message=f"JAX not installed: {e}",
                severity=Severity.ERROR,
                remediation="Run 'uv sync' to install dependencies",
            )

    def test_jax_cpu_backend(self) -> ValidationResult:
        """Test JAX CPU backend configuration."""
        try:
            import jax

            devices = jax.devices()
            device_count = len(devices)
            platform = devices[0].platform if devices else "unknown"

            details = {
                "device_count": device_count,
                "platform": platform,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            }

            if platform == "cpu":
                msg = f"JAX CPU backend active with {device_count} device(s)"
                return ValidationResult(
                    name="JAX CPU Backend",
                    success=True,
                    message=msg,
                    severity=Severity.SUCCESS,
                    details=details,
                )
            else:
                return ValidationResult(
                    name="JAX CPU Backend",
                    success=False,
                    message=f"JAX using {platform} backend instead of CPU",
                    severity=Severity.WARNING,
                    remediation="Set JAX_PLATFORMS=cpu before importing JAX",
                    details=details,
                )

        except Exception as e:
            return ValidationResult(
                name="JAX CPU Backend",
                success=False,
                message=f"Failed to check JAX backend: {e}",
                severity=Severity.ERROR,
            )

    def test_numpy_installation(self) -> ValidationResult:
        """Test NumPy installation and BLAS backend."""
        try:
            import numpy as np

            version = np.__version__

            # Check BLAS backend
            try:
                config = np.show_config(mode="dicts")
                if isinstance(config, dict) and "Build Dependencies" in config:
                    blas_info = config["Build Dependencies"].get("blas", {})
                    blas_name = blas_info.get("name", "unknown")
                else:
                    blas_name = "unknown"
            except Exception:
                blas_name = "unknown"

            details = {
                "version": version,
                "blas_backend": blas_name,
            }

            # Check version (numpy 2.x required)
            major = int(version.split(".")[0])
            if major >= 2:
                return ValidationResult(
                    name="NumPy Installation",
                    success=True,
                    message=f"NumPy {version} with BLAS: {blas_name}",
                    severity=Severity.SUCCESS,
                    details=details,
                )
            else:
                return ValidationResult(
                    name="NumPy Installation",
                    success=False,
                    message=f"NumPy {version} is below minimum (2.x)",
                    severity=Severity.ERROR,
                    remediation="Run 'uv pip install numpy>=2.3'",
                    details=details,
                )

        except ImportError as e:
            return ValidationResult(
                name="NumPy Installation",
                success=False,
                message=f"NumPy not installed: {e}",
                severity=Severity.ERROR,
                remediation="Run 'uv sync' to install dependencies",
            )

    def test_numpyro_installation(self) -> ValidationResult:
        """Test NumPyro installation for MCMC."""
        try:
            import numpyro

            version = numpyro.__version__

            details = {
                "version": version,
            }

            # Check version (0.19+ required)
            major, minor = version.split(".")[:2]
            major, minor = int(major), int(minor)
            if (major, minor) >= (0, 19):
                return ValidationResult(
                    name="NumPyro Installation",
                    success=True,
                    message=f"NumPyro {version} installed for MCMC",
                    severity=Severity.SUCCESS,
                    details=details,
                )
            else:
                return ValidationResult(
                    name="NumPyro Installation",
                    success=False,
                    message=f"NumPyro {version} is below minimum (0.19)",
                    severity=Severity.WARNING,
                    remediation="Run 'uv pip install numpyro>=0.19.0'",
                    details=details,
                )

        except ImportError as e:
            return ValidationResult(
                name="NumPyro Installation",
                success=False,
                message=f"NumPyro not installed: {e}",
                severity=Severity.WARNING,
                remediation="NumPyro is optional. Install with 'uv pip install numpyro>=0.19.0'",
            )

    def test_shell_completion(self) -> ValidationResult:
        """Test shell completion installation."""
        # Check for completion files
        venv_path = sys.prefix
        completion_paths = [
            os.path.join(venv_path, "etc", "bash_completion.d", "heterodyne"),
            os.path.join(venv_path, "etc", "zsh", "heterodyne-completion.zsh"),
            os.path.expanduser("~/.local/share/bash-completion/completions/heterodyne"),
        ]

        installed = [p for p in completion_paths if os.path.exists(p)]

        details = {
            "installed_completions": installed,
            "venv_path": venv_path,
        }

        if installed:
            return ValidationResult(
                name="Shell Completion",
                success=True,
                message=f"Shell completion installed at {len(installed)} location(s)",
                severity=Severity.SUCCESS,
                details=details,
            )
        else:
            return ValidationResult(
                name="Shell Completion",
                success=False,
                message="Shell completion not installed",
                severity=Severity.INFO,
                remediation="Run 'heterodyne-post-install' to install shell completion",
                details=details,
            )


def run_validation(verbose: bool = True) -> list[ValidationResult]:
    """Run all validation tests and return results.

    Args:
        verbose: If True, print output during validation.

    Returns:
        List of ValidationResult objects.
    """
    validator = SystemValidator(verbose=verbose)
    return validator.run_all()


def main() -> int:
    """CLI entry point for heterodyne-validate command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate heterodyne installation and configuration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    results = run_validation(verbose=args.verbose and not args.json)

    if args.json:
        import json
        output = [
            {
                "name": r.name,
                "success": r.success,
                "message": r.message,
                "severity": r.severity.value,
                "remediation": r.remediation,
                "details": r.details,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        # Summary
        passed = sum(1 for r in results if r.success)
        total = len(results)
        print(f"\n{'='*50}")
        print(f"Validation complete: {passed}/{total} tests passed")

        # Show remediations for failures
        failures = [r for r in results if not r.success and r.remediation]
        if failures:
            print("\nSuggested fixes:")
            for r in failures:
                print(f"  - {r.name}: {r.remediation}")

    # Return non-zero if any errors
    has_errors = any(r.severity == Severity.ERROR and not r.success for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
