"""Main entry point for heterodyne CLI."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


def _bootstrap_xla_env(argv: list[str] | None) -> None:
    """Inject ``XLA_FLAGS`` from ``--threads``/``--no-jit`` *before* any
    ``heterodyne`` import triggers ``import jax``.

    ``heterodyne/__init__.py`` imports JAX eagerly to set ``jax_enable_x64``.
    JAX reads ``XLA_FLAGS`` only once during backend initialization, so
    configuring after importing ``heterodyne.cli.args_parser`` was too late
    for ``--threads`` to take effect. We pre-parse the affected flags from
    ``argv`` using only the stdlib and seed the env before any package import.
    """
    raw = list(sys.argv[1:] if argv is None else argv)

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"

    threads: int | None = None
    no_jit = False
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "--threads" and i + 1 < len(raw):
            try:
                threads = int(raw[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if token.startswith("--threads="):
            try:
                threads = int(token.split("=", 1)[1])
            except ValueError:
                pass
            i += 1
            continue
        if token == "--no-jit":
            no_jit = True
        i += 1

    if threads is not None:
        existing = os.environ.get("XLA_FLAGS", "")
        tflags = (
            "--xla_cpu_multi_thread_eigen=true"
            f" --intra_op_parallelism_threads={threads}"
        )
        if tflags not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {tflags}".strip()
        # An explicit ``--threads`` must win over any inherited
        # ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` (e.g. set high by a batch
        # scheduler or login profile). Using ``setdefault`` here would leave
        # BLAS/OpenMP oversubscribed relative to the XLA intra-op limit, so we
        # assign unconditionally — matching the old ``configure_xla()``.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

    if no_jit:
        os.environ["JAX_DISABLE_JIT"] = "1"


def main(argv: list[str] | None = None) -> int:
    """Main entry point for heterodyne CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    # MUST run before the first ``heterodyne`` import below — that import
    # cascades into ``heterodyne/__init__.py`` which eagerly imports JAX.
    _bootstrap_xla_env(argv)

    import logging as _logging

    # Suppress JAX backend logs (homodyne parity: hide GPU fallback warnings)
    _logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
    _logging.getLogger("jax._src.compiler").setLevel(_logging.ERROR)

    from heterodyne.cli.args_parser import create_parser, validate_args

    parser = create_parser()
    args = parser.parse_args(argv)

    # Validate arguments
    try:
        warnings = validate_args(args)
        for warn in warnings:
            print(f"Warning: {warn}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Now import JAX-dependent modules
    from heterodyne.cli.commands import dispatch_command

    # Set up logging
    from heterodyne.utils.logging import configure_logging

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose >= 1:
        log_level = "INFO"
    else:
        log_level = "WARNING"

    configure_logging(level=log_level)

    # Run analysis
    start_time = time.perf_counter()

    from heterodyne.utils.logging import get_logger, log_exception

    logger = get_logger(__name__)

    try:
        exit_code = dispatch_command(args)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        log_exception(logger, e, context={"command": "main"})
        return 1

    elapsed = time.perf_counter() - start_time
    if not args.quiet:
        logger.info("Analysis completed in %.1f seconds", elapsed)

    return exit_code


def main_hexp() -> int:
    """Entry point for ``hexp`` — plot experimental data."""
    return main(["--plot-experimental-data"] + sys.argv[1:])


def main_hsim() -> int:
    """Entry point for ``hsim`` — plot simulated data."""
    return main(["--plot-simulated-data"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
