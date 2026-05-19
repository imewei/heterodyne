"""Synthetic input for CLI extractor tests."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config_path", default="default.yaml", help="Path to YAML"
    )
    parser.add_argument(
        "--method",
        choices=["nlsq", "cmc"],
        default="nlsq",
        help="Which optimizer to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.add_parser("fit", aliases=["f"])
    subparsers.add_parser("plot")
    return parser
