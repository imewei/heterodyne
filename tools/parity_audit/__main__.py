"""CLI entry point: `python -m tools.parity_audit`.

Subcommands are wired in Task 12 once extractors and the differ exist.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.parity_audit")
    parser.add_argument(
        "subcommand",
        choices=["extract", "diff", "run-all"],
        help="extract: run all extractors against one package. "
        "diff: compare two extract dirs. "
        "run-all: convenience wrapper for extract+diff on both packages.",
    )
    args, _ = parser.parse_known_args(argv)
    print(f"Subcommand {args.subcommand!r} not yet wired (Task 12).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
