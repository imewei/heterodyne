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

    try:
        exit_code = dispatch_command(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - start_time
    if not args.quiet:
        print(f"\nCompleted in {elapsed:.1f} seconds")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
