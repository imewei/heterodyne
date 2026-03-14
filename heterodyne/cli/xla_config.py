"""XLA configuration for JAX on CPU."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def configure_xla(
    num_threads: int | None = None,
    disable_jit: bool = False,
    enable_x64: bool = True,
) -> dict[str, str]:
    """Configure XLA/JAX environment variables for CPU execution.

    MUST be called before importing JAX.

    Args:
        num_threads: Number of CPU threads (None for auto)
        disable_jit: Disable JIT compilation (for debugging)
        enable_x64: Enable 64-bit float precision

    Returns:
        Dict of environment variables that were set
    """
    env_vars = {}

    # Force CPU backend
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    env_vars["JAX_PLATFORM_NAME"] = "cpu"

    # Thread configuration
    if num_threads is not None:
        existing = os.environ.get("XLA_FLAGS", "")
        new_flags = (
            "--xla_cpu_multi_thread_eigen=true"
            f" --intra_op_parallelism_threads={num_threads}"
        )
        if new_flags not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {new_flags}".strip()
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        env_vars["XLA_FLAGS"] = os.environ["XLA_FLAGS"]
        env_vars["OMP_NUM_THREADS"] = str(num_threads)
        env_vars["MKL_NUM_THREADS"] = str(num_threads)

    # Disable JIT for debugging
    if disable_jit:
        os.environ["JAX_DISABLE_JIT"] = "1"
        env_vars["JAX_DISABLE_JIT"] = "1"

    # Enable 64-bit precision
    if enable_x64:
        os.environ["JAX_ENABLE_X64"] = "1"
        env_vars["JAX_ENABLE_X64"] = "1"

    return env_vars


def get_cpu_info() -> dict[str, int | str]:
    """Get CPU information for configuration.

    Returns:
        Dict with cpu_count, physical_cores, etc.
    """
    import psutil

    info = {
        "cpu_count": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(),
    }

    # Available memory
    mem = psutil.virtual_memory()
    info["available_memory_gb"] = round(mem.available / (1024**3), 1)
    info["total_memory_gb"] = round(mem.total / (1024**3), 1)

    return info


def auto_configure() -> dict[str, str]:
    """Automatically configure XLA based on system resources.

    Returns:
        Dict of environment variables set
    """
    cpu_info = get_cpu_info()

    # Use physical cores (not hyperthreaded)
    num_threads = cpu_info.get("physical_cores", 4)

    return configure_xla(num_threads=num_threads, enable_x64=True)


def main() -> None:
    """CLI entry point for XLA configuration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Configure XLA for heterodyne analysis"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto)",
    )
    parser.add_argument(
        "--no-x64",
        action="store_true",
        help="Disable 64-bit precision",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable JIT for debugging",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print CPU info and exit",
    )

    args = parser.parse_args()

    if args.info:
        info = get_cpu_info()
        print("CPU Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return

    env_vars = configure_xla(
        num_threads=args.threads,
        disable_jit=args.debug,
        enable_x64=not args.no_x64,
    )

    print("XLA Configuration:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")


if __name__ == "__main__":
    main()
