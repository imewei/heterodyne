"""Main entry point for heterodyne CLI."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Literal

# Configure XLA BEFORE importing JAX
from heterodyne.cli.xla_config import configure_xla

if TYPE_CHECKING:
    pass


def main(argv: list[str] | None = None) -> int:
    """Main entry point for heterodyne CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
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

    # Configure XLA based on arguments
    configure_xla(
        num_threads=args.threads,
        disable_jit=args.no_jit,
        enable_x64=True,
    )

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
