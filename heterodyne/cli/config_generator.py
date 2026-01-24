"""Configuration file generator for heterodyne analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def get_template_path() -> Path:
    """Get path to master template file.
    
    Returns:
        Path to template YAML
    """
    import heterodyne
    
    pkg_dir = Path(heterodyne.__file__).parent
    template_path = pkg_dir / "config" / "templates" / "heterodyne_master_template.yaml"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    return template_path


def generate_config(
    output_path: Path | str,
    data_path: str | None = None,
    q: float | None = None,
    dt: float | None = None,
    time_length: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Generate configuration file from template.
    
    Args:
        output_path: Output path for configuration
        data_path: Path to experimental data file
        q: Wavevector value
        dt: Time step
        time_length: Number of time points
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to generated config
    """
    output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {output_path}. Use --overwrite to replace.")
    
    template_path = get_template_path()
    
    # Read template
    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    
    # Substitute values if provided
    if data_path is not None:
        content = content.replace('file_path: ""', f'file_path: "{data_path}"')
    
    if q is not None:
        content = content.replace("wavevector_q: 0.01", f"wavevector_q: {q}")
    
    if dt is not None:
        content = content.replace("dt: 1.0", f"dt: {dt}")
    
    if time_length is not None:
        content = content.replace("time_length: 1000", f"time_length: {time_length}")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"Generated configuration: {output_path}")
    
    return output_path


def main() -> None:
    """CLI entry point for config generator."""
    parser = argparse.ArgumentParser(
        prog="heterodyne-config",
        description="Generate heterodyne configuration file from template",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("heterodyne_config.yaml"),
        help="Output path for configuration file (default: heterodyne_config.yaml)",
    )
    
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Path to experimental data file",
    )
    
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Wavevector magnitude",
    )
    
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step",
    )
    
    parser.add_argument(
        "--time-length",
        type=int,
        default=None,
        help="Number of time points",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file",
    )
    
    parser.add_argument(
        "--show-template",
        action="store_true",
        help="Print template path and exit",
    )
    
    args = parser.parse_args()
    
    if args.show_template:
        print(f"Template: {get_template_path()}")
        return
    
    try:
        output = generate_config(
            output_path=args.output,
            data_path=args.data,
            q=args.q,
            dt=args.dt,
            time_length=args.time_length,
            overwrite=args.overwrite,
        )
        print(f"Created: {output}")
    except FileExistsError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
