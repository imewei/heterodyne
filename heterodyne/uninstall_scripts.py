"""Cleanup utilities for heterodyne package.

This module provides utilities to remove:
- Shell completion files
- XLA configuration files
- Activation script modifications

CLI Entry Point: heterodyne-cleanup
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


class CleanupTarget(NamedTuple):
    """A file or directory to clean up."""

    path: Path
    description: str
    exists: bool


def get_venv_path() -> Path | None:
    """Get the virtual environment path if in one.

    Returns:
        Path to venv or None if not in a virtual environment.
    """
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)

    return None


def find_cleanup_targets() -> list[CleanupTarget]:
    """Find all heterodyne-related files that can be cleaned up.

    Returns:
        List of CleanupTarget objects.
    """
    targets: list[CleanupTarget] = []

    # XLA mode configuration — check all possible locations
    home = Path.home()

    # New location: per-env or XDG
    from heterodyne.post_install import get_xla_mode_path

    xla_mode_file = get_xla_mode_path()
    targets.append(
        CleanupTarget(
            path=xla_mode_file,
            description="XLA mode configuration",
            exists=xla_mode_file.exists(),
        )
    )

    # Legacy location
    legacy_xla = home / ".heterodyne_xla_mode"
    if legacy_xla.exists():
        targets.append(
            CleanupTarget(
                path=legacy_xla,
                description="XLA mode configuration (legacy)",
                exists=True,
            )
        )

    # Virtual environment files
    venv_path = get_venv_path()
    if venv_path:
        # Bash completion
        bash_completion = venv_path / "etc" / "bash_completion.d" / "heterodyne"
        targets.append(
            CleanupTarget(
                path=bash_completion,
                description="Bash completion script",
                exists=bash_completion.exists(),
            )
        )

        # Zsh completion
        zsh_completion = venv_path / "etc" / "zsh" / "heterodyne-completion.zsh"
        targets.append(
            CleanupTarget(
                path=zsh_completion,
                description="Zsh completion script",
                exists=zsh_completion.exists(),
            )
        )

        # Fish completion
        fish_completion = (
            venv_path / "share" / "fish" / "vendor_completions.d" / "heterodyne.fish"
        )
        targets.append(
            CleanupTarget(
                path=fish_completion,
                description="Fish completion script",
                exists=fish_completion.exists(),
            )
        )

        # Check for empty parent directories
        for completion in [bash_completion, zsh_completion, fish_completion]:
            if completion.parent.exists() and not any(completion.parent.iterdir()):
                targets.append(
                    CleanupTarget(
                        path=completion.parent,
                        description=f"Empty directory: {completion.parent.name}",
                        exists=True,
                    )
                )

    # User-level completion directories
    local_bash = (
        home / ".local" / "share" / "bash-completion" / "completions" / "heterodyne"
    )
    targets.append(
        CleanupTarget(
            path=local_bash,
            description="User bash completion",
            exists=local_bash.exists(),
        )
    )

    return targets


def cleanup_completion_files(
    dry_run: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Remove shell completion files.

    Args:
        dry_run: If True, don't actually delete files.
        verbose: Print detailed output.

    Returns:
        List of paths that were (or would be) removed.
    """
    removed: list[Path] = []
    targets = find_cleanup_targets()

    for target in targets:
        if not target.exists:
            continue

        if "completion" not in target.description.lower():
            continue

        if verbose:
            action = "Would remove" if dry_run else "Removing"
            print(f"{action}: {target.path} ({target.description})")

        if not dry_run:
            try:
                if target.path.is_dir():
                    target.path.rmdir()
                else:
                    target.path.unlink()
                removed.append(target.path)
            except OSError as e:
                if verbose:
                    print(f"  Failed: {e}")
        else:
            removed.append(target.path)

    return removed


def cleanup_xla_config(
    dry_run: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Remove XLA configuration files.

    Args:
        dry_run: If True, don't actually delete files.
        verbose: Print detailed output.

    Returns:
        List of paths that were (or would be) removed.
    """
    removed: list[Path] = []

    # XLA mode file — remove from all locations
    from heterodyne.post_install import get_xla_mode_path

    for xla_mode_file in [get_xla_mode_path(), Path.home() / ".heterodyne_xla_mode"]:
        if xla_mode_file.exists():
            if verbose:
                action = "Would remove" if dry_run else "Removing"
                print(f"{action}: {xla_mode_file} (XLA mode configuration)")

            if not dry_run:
                try:
                    xla_mode_file.unlink()
                    removed.append(xla_mode_file)
                    # Clean up empty parent dir
                    parent = xla_mode_file.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except OSError as e:
                    if verbose:
                        print(f"  Failed: {e}")
            else:
                removed.append(xla_mode_file)

    return removed


def cleanup_xla_activation_scripts(
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """Remove XLA configuration from venv activation scripts.

    Args:
        dry_run: If True, don't actually modify files.
        verbose: Print detailed output.

    Returns:
        True if any modifications were made.
    """
    modified = False
    venv_path = get_venv_path()

    if not venv_path:
        return False

    # Bash/Zsh activate script
    activate_script = venv_path / "bin" / "activate"
    if activate_script.exists():
        content = activate_script.read_text()

        # Pattern to match the heterodyne XLA configuration block
        pattern = r"\n# heterodyne XLA configuration\nif \[.*?fi\n"

        if "heterodyne XLA configuration" in content:
            if verbose:
                action = "Would modify" if dry_run else "Modifying"
                print(f"{action}: {activate_script} (removing XLA config)")

            if not dry_run:
                new_content = re.sub(pattern, "", content, flags=re.DOTALL)
                try:
                    activate_script.write_text(new_content)
                    modified = True
                except OSError as e:
                    if verbose:
                        print(f"  Failed: {e}")
            else:
                modified = True

    # Fish activate script
    fish_activate = venv_path / "bin" / "activate.fish"
    if fish_activate.exists():
        content = fish_activate.read_text()

        pattern = r"\n# heterodyne XLA configuration\nif test.*?end\n"

        if "heterodyne XLA configuration" in content:
            if verbose:
                action = "Would modify" if dry_run else "Modifying"
                print(f"{action}: {fish_activate} (removing XLA config)")

            if not dry_run:
                new_content = re.sub(pattern, "", content, flags=re.DOTALL)
                try:
                    fish_activate.write_text(new_content)
                    modified = True
                except OSError as e:
                    if verbose:
                        print(f"  Failed: {e}")
            else:
                modified = True

    return modified


def show_dry_run(verbose: bool = True) -> None:
    """Show what would be removed without actually removing anything.

    Args:
        verbose: Print detailed output.
    """
    print("Dry run - showing what would be removed:")
    print("-" * 50)

    targets = find_cleanup_targets()
    existing = [t for t in targets if t.exists]

    if not existing:
        print("No heterodyne files found to clean up.")
        return

    for target in existing:
        print(f"  {target.path}")
        if verbose:
            print(f"    ({target.description})")

    # Check activation scripts
    venv_path = get_venv_path()
    if venv_path:
        activate = venv_path / "bin" / "activate"
        if activate.exists() and "heterodyne XLA" in activate.read_text():
            print(f"  {activate} (would modify)")
            if verbose:
                print("    (remove XLA configuration block)")

        fish_activate = venv_path / "bin" / "activate.fish"
        if fish_activate.exists() and "heterodyne XLA" in fish_activate.read_text():
            print(f"  {fish_activate} (would modify)")
            if verbose:
                print("    (remove XLA configuration block)")


def interactive_cleanup() -> None:
    """Run interactive cleanup process."""
    print("=" * 60)
    print("Heterodyne Cleanup")
    print("=" * 60)
    print()

    targets = find_cleanup_targets()
    existing = [t for t in targets if t.exists]

    if not existing:
        print("No heterodyne files found to clean up.")
        return

    print("The following files were found:")
    for target in existing:
        print(f"  - {target.path}")
        print(f"    ({target.description})")
    print()

    # Check activation scripts
    venv_path = get_venv_path()
    has_activation_mods = False
    if venv_path:
        activate = venv_path / "bin" / "activate"
        if activate.exists() and "heterodyne XLA" in activate.read_text():
            print(f"  - {activate} contains XLA configuration")
            has_activation_mods = True

    print()
    response = input("Remove all heterodyne files? [y/N]: ").strip().lower()

    if response != "y":
        print("Cleanup cancelled.")
        return

    # Perform cleanup
    print("\nCleaning up...")

    removed = cleanup_completion_files(verbose=True)
    removed.extend(cleanup_xla_config(verbose=True))

    if has_activation_mods:
        cleanup_xla_activation_scripts(verbose=True)

    print()
    print(f"Removed {len(removed)} file(s).")
    print("Cleanup complete!")


def main() -> int:
    """CLI entry point for heterodyne-cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up heterodyne installation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  heterodyne-cleanup                  # Interactive cleanup
  heterodyne-cleanup --dry-run        # Show what would be removed
  heterodyne-cleanup --force          # Remove without confirmation
""",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be removed without removing",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Remove without confirmation",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive cleanup (default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.dry_run:
        show_dry_run(args.verbose)
        return 0

    if args.force:
        # Non-interactive removal
        removed = cleanup_completion_files(verbose=args.verbose)
        removed.extend(cleanup_xla_config(verbose=args.verbose))
        cleanup_xla_activation_scripts(verbose=args.verbose)

        if args.verbose:
            print(f"\nRemoved {len(removed)} file(s).")
        return 0

    # Default: interactive mode
    interactive_cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
