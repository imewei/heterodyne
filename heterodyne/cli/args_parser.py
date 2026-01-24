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
    
    # Plotting
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots",
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}",
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
    
    # Check conflicting options
    if args.verbose > 0 and args.quiet:
        warnings.append("Both --verbose and --quiet specified; using --quiet")
    
    if args.plot and args.no_plot:
        warnings.append("Both --plot and --no-plot specified; using --no-plot")
    
    return warnings
