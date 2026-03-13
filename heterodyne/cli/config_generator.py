"""Configuration file generator for heterodyne analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


_VALID_MODES = ("full", "minimal", "nlsq_only", "cmc_only")


def generate_config(
    output_path: Path | str,
    data_path: str | None = None,
    q: float | None = None,
    dt: float | None = None,
    time_length: int | None = None,
    overwrite: bool = False,
    mode: str = "full",
) -> Path:
    """Generate configuration file from template.

    Args:
        output_path: Output path for configuration
        data_path: Path to experimental data file
        q: Wavevector value
        dt: Time step
        time_length: Number of time points
        overwrite: Whether to overwrite existing file
        mode: Template mode — "full" (all sections), "minimal" (data+temporal+
            scattering only), "nlsq_only" (NLSQ without CMC), or "cmc_only"
            (CMC without NLSQ).

    Returns:
        Path to generated config
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {', '.join(_VALID_MODES)}"
        )

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File exists: {output_path}. Use --overwrite to replace."
        )

    template_path = get_template_path()

    # Read template
    with open(template_path, encoding="utf-8") as f:
        content = f.read()

    # Substitute values if provided
    import yaml

    substitutions: list[tuple[str, str]] = []

    if data_path is not None:
        safe_data_path = yaml.dump(data_path, default_flow_style=True).strip()
        substitutions.append(('file_path: ""', f"file_path: {safe_data_path}"))

    if q is not None:
        safe_q = yaml.dump(q, default_flow_style=True).strip()
        substitutions.append(("wavevector_q: 0.01", f"wavevector_q: {safe_q}"))

    if dt is not None:
        safe_dt = yaml.dump(dt, default_flow_style=True).strip()
        substitutions.append(("dt: 1.0", f"dt: {safe_dt}"))

    if time_length is not None:
        safe_tl = yaml.dump(time_length, default_flow_style=True).strip()
        substitutions.append(("time_length: 1000", f"time_length: {safe_tl}"))

    for placeholder, replacement in substitutions:
        if placeholder not in content:
            logger.warning("Placeholder '%s' not found in template", placeholder)
        content = content.replace(placeholder, replacement)

    # Apply mode-based filtering
    if mode != "full":
        config_dict: dict[str, Any] = yaml.safe_load(content) or {}
        config_dict = _apply_mode_filter(config_dict, mode)
        content = yaml.safe_dump(config_dict, default_flow_style=False)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Generated configuration: %s (mode=%s)", output_path, mode)

    return output_path


def _apply_mode_filter(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Filter config dict sections based on the requested mode.

    Args:
        config: Full configuration dictionary.
        mode: One of "minimal", "nlsq_only", "cmc_only".

    Returns:
        Filtered configuration dictionary.
    """
    if mode == "minimal":
        # Keep only data, temporal, and scattering sections
        keep_sections = {"data", "temporal", "scattering"}
        return {k: v for k, v in config.items() if k in keep_sections}

    if mode == "nlsq_only":
        # Set method to nlsq and remove CMC section
        opt = config.get("optimization", {})
        if isinstance(opt, dict):
            opt["method"] = "nlsq"
            opt.pop("cmc", None)
            config["optimization"] = opt
        return config

    if mode == "cmc_only":
        # Set method to cmc
        opt = config.get("optimization", {})
        if isinstance(opt, dict):
            opt["method"] = "cmc"
            config["optimization"] = opt
        return config

    return config


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

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive config builder",
    )

    parser.add_argument(
        "--validate",
        "-V",
        action="store_true",
        help="Validate an existing config file (uses --output as path)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=_VALID_MODES,
        help="Template mode: full, minimal, nlsq_only, or cmc_only (default: full)",
    )

    args = parser.parse_args()

    if args.show_template:
        print(f"Template: {get_template_path()}")
        return

    # Validate existing config
    if args.validate:
        is_valid = validate_config(args.output)
        raise SystemExit(0 if is_valid else 1)

    # Interactive builder
    if args.interactive:
        from heterodyne.data.config import save_yaml_config

        config = interactive_builder()
        output_path = Path(args.output)
        if output_path.exists() and not args.overwrite:
            print(f"Error: File exists: {output_path}. Use --overwrite to replace.")
            raise SystemExit(1)
        save_yaml_config(config, output_path)
        print(f"Created: {output_path}")
        return

    # Template-based generation
    try:
        output = generate_config(
            output_path=args.output,
            data_path=args.data,
            q=args.q,
            dt=args.dt,
            time_length=args.time_length,
            overwrite=args.overwrite,
            mode=args.mode,
        )
        print(f"Created: {output}")
    except FileExistsError as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e


def _prompt(
    label: str,
    default: str,
    *,
    required: bool = False,
    cast: type | None = None,
) -> Any:
    """Prompt the user for input with a default value.

    Re-prompts on invalid input when *cast* is specified.

    Args:
        label: Display label for the prompt.
        default: Default value shown in brackets.
        required: If True, empty input is not accepted.
        cast: If given, attempt to cast the input to this type.

    Returns:
        The user-supplied (or default) value, optionally cast.
    """
    while True:
        suffix = f" [{default}]" if default and not required else ""
        raw = input(f"{label}{suffix}: ").strip()

        if not raw:
            if required:
                print("  This field is required.")
                continue
            raw = default

        if cast is not None:
            try:
                return cast(raw)
            except (ValueError, TypeError):
                print(f"  Invalid value. Expected {cast.__name__}.")
                continue

        return raw


def interactive_builder() -> dict[str, Any]:
    """Build a configuration dict interactively via sequential prompts.

    Returns:
        Complete configuration dictionary matching the expected schema.
    """
    print("=== Heterodyne Config Builder ===\n")

    data_path = _prompt("Data file path", "", required=True)
    q = _prompt("Wavevector q", "0.01", cast=float)
    dt = _prompt("Time step dt", "1.0", cast=float)
    time_length = _prompt("Number of time points", "1000", cast=int)

    phi_raw = _prompt("Phi angles (comma-separated)", "0.0")
    try:
        phi_angles = [float(p.strip()) for p in phi_raw.split(",")]
    except ValueError:
        print("  Invalid phi angles, using default [0.0].")
        phi_angles = [0.0]

    method = _prompt("Optimization method (nlsq/cmc/both)", "nlsq")
    while method not in ("nlsq", "cmc", "both"):
        print("  Must be one of: nlsq, cmc, both")
        method = _prompt("Optimization method (nlsq/cmc/both)", "nlsq")

    output_dir = _prompt("Output directory", "./output")

    config: dict[str, Any] = {
        "data": {
            "file_path": data_path,
        },
        "temporal": {
            "dt": dt,
            "time_length": time_length,
        },
        "scattering": {
            "wavevector_q": q,
            "phi_angles": phi_angles,
        },
        "optimization": {
            "method": method,
        },
        "output": {
            "directory": output_dir,
        },
    }

    logger.info("Interactive config built successfully")
    print("\nConfiguration built successfully.")
    return config


def validate_config(path: Path | str) -> bool:
    """Validate an existing YAML configuration file.

    Loads the file, runs schema validation via ``validate_config_schema()``,
    and attempts to load it into ``ConfigManager`` to catch structural issues.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        True if the configuration is valid, False otherwise.
    """
    from heterodyne.data.config import load_yaml_config, validate_config_schema

    path = Path(path)
    print(f"Validating: {path}\n")

    # Load YAML
    try:
        config = load_yaml_config(path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        return False
    except Exception as exc:
        print(f"ERROR: Failed to load YAML: {exc}")
        return False

    # Schema validation
    result = validate_config_schema(config)

    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for warn in result.warnings:
            print(f"  - {warn}")

    if result.missing_optional:
        print(f"Missing optional fields ({len(result.missing_optional)}):")
        for field in result.missing_optional:
            print(f"  - {field}")

    # Structural validation via ConfigManager
    if result.is_valid:
        try:
            from heterodyne.config.manager import ConfigManager

            ConfigManager(config)
        except Exception as exc:
            logger.error("Structural validation failed: %s", exc)
            print(f"\nStructural validation failed: {exc}")
            return False

    # Summary
    if result.is_valid:
        print("\nResult: VALID")
    else:
        print("\nResult: INVALID")

    return result.is_valid


if __name__ == "__main__":
    main()
