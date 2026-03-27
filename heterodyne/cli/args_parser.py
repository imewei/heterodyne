"""Argument parser for heterodyne CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for heterodyne CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="heterodyne",
        description="Heterodyne XPCS analysis: CPU-optimized JAX-based fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run NLSQ fitting
  heterodyne --config analysis.yaml --method nlsq

  # Run CMC (Bayesian) analysis
  heterodyne --config analysis.yaml --method cmc

  # Specify output directory
  heterodyne --config analysis.yaml --output ./results

  # Run with verbose output
  heterodyne --config analysis.yaml --verbose
""",
    )

    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    # Method selection
    parser.add_argument(
        "--method",
        "-m",
        choices=["nlsq", "cmc", "both"],
        default="nlsq",
        help="Optimization method: nlsq, cmc, or both (default: nlsq)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--output-format",
        choices=["json", "npz", "both"],
        default="both",
        help="Output format (default: both)",
    )

    # Analysis options
    parser.add_argument(
        "--phi",
        type=float,
        nargs="+",
        default=None,
        help="Phi angles to analyze (overrides config)",
    )

    parser.add_argument(
        "--multistart",
        action="store_true",
        help="Use multi-start optimization for NLSQ",
    )

    parser.add_argument(
        "--multistart-n",
        type=int,
        default=10,
        help="Number of starting points for multi-start (default: 10)",
    )

    # CMC options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of CMC samples (overrides config)",
    )

    parser.add_argument(
        "--num-chains",
        type=int,
        default=None,
        help="Number of CMC chains (overrides config)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    # Performance options
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads",
    )

    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation (for debugging)",
    )

    # Plotting (mutually exclusive)
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Generate plots (default)",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip plot generation",
    )

    # Post-optimization plot saving
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save fit comparison and fitted simulation plots to output directory",
    )

    parser.add_argument(
        "--plotting-backend",
        type=str,
        choices=["auto", "matplotlib", "datashader"],
        default="auto",
        help=(
            "Plotting backend: auto (Datashader if available), "
            "matplotlib, datashader (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--parallel-plots",
        action="store_true",
        help="Generate plots in parallel using multiprocessing (requires Datashader)",
    )

    parser.add_argument(
        "--phi-angles",
        type=str,
        default=None,
        help=(
            "Comma-separated phi angles in degrees for simulated data "
            "(e.g., '0,45,90,135')"
        ),
    )

    # Standalone plot modes (skip optimization)
    parser.add_argument(
        "--plot-experimental-data",
        action="store_true",
        help="Plot experimental data for quality checking (skip optimization)",
    )
    parser.add_argument(
        "--plot-simulated-data",
        action="store_true",
        help="Plot simulated C2 heatmaps from config parameters (skip optimization)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.3,
        help="Contrast for simulated data (default: %(default)s, requires --plot-simulated-data)",
    )
    parser.add_argument(
        "--offset-sim",
        type=float,
        default=1.0,
        help="Offset for simulated data (default: %(default)s, requires --plot-simulated-data)",
    )

    # Version
    from heterodyne._version import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    """Validate parsed arguments.

    Args:
        args: Parsed arguments

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    # Check config file exists
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    # Check conflicting options — enforce --quiet taking precedence
    if args.verbose > 0 and args.quiet:
        warnings.append("Both --verbose and --quiet specified; using --quiet")
        args.verbose = 0

    # Validate --phi-angles format (comma-separated floats)
    phi_angles_str = getattr(args, "phi_angles", None)
    if phi_angles_str is not None:
        try:
            [float(x.strip()) for x in phi_angles_str.split(",")]
        except ValueError:
            warnings.append(
                f"--phi-angles must be comma-separated numbers "
                f"(e.g., '0,45,90,135'), got: '{phi_angles_str}'"
            )

    return warnings
