"""Post-installation setup for heterodyne package.

This module provides interactive setup for:
- Shell completion installation (bash/zsh/fish)
- XLA_FLAGS configuration
- Virtual environment integration

CLI Entry Point: heterodyne-post-install
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Literal


def detect_shell_type() -> Literal["bash", "zsh", "fish", "unknown"]:
    """Detect the current shell type.

    Returns:
        Shell type string or "unknown" if detection fails.
    """
    # Check SHELL environment variable
    shell_path = os.environ.get("SHELL", "")
    shell_name = os.path.basename(shell_path)

    if "zsh" in shell_name:
        return "zsh"
    elif "bash" in shell_name:
        return "bash"
    elif "fish" in shell_name:
        return "fish"

    # Fallback: check parent process name
    try:
        import psutil

        parent = psutil.Process().parent()
        if parent:
            pname = parent.name().lower()
            if "zsh" in pname:
                return "zsh"
            elif "bash" in pname:
                return "bash"
            elif "fish" in pname:
                return "fish"
    except (ImportError, Exception):
        pass

    return "unknown"


def is_virtual_environment() -> bool:
    """Check if running in a virtual environment.

    Returns:
        True if in a venv, conda env, or similar.
    """
    # Standard venv check
    if sys.prefix != sys.base_prefix:
        return True

    # Conda environment check
    if os.environ.get("CONDA_PREFIX"):
        return True

    # Check for VIRTUAL_ENV marker
    if os.environ.get("VIRTUAL_ENV"):
        return True

    return False


def is_conda_environment() -> bool:
    """Check if running in a conda/mamba environment.

    Returns:
        True if in a conda environment.
    """
    return bool(os.environ.get("CONDA_PREFIX"))


def get_venv_path() -> Path:
    """Get the virtual environment path.

    Returns:
        Path to the virtual environment directory.
    """
    # Prefer VIRTUAL_ENV if set
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv)

    # Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)

    # Fallback to sys.prefix
    return Path(sys.prefix)


def get_completion_source_path() -> Path:
    """Get the path to the completion script in the package.

    Returns:
        Path to completion.sh in the installed package.
    """
    try:
        from heterodyne.runtime.shell import COMPLETION_SCRIPT

        return COMPLETION_SCRIPT
    except ImportError:
        # Fallback: find relative to this file
        return Path(__file__).parent / "runtime" / "shell" / "completion.sh"


def get_xla_config_source_path(shell: str) -> Path:
    """Get the path to the XLA config script.

    Args:
        shell: Shell type ("bash", "zsh", or "fish")

    Returns:
        Path to the XLA config script.
    """
    try:
        from heterodyne.runtime.shell import XLA_CONFIG_BASH, XLA_CONFIG_FISH

        if shell == "fish":
            return XLA_CONFIG_FISH
        return XLA_CONFIG_BASH
    except ImportError:
        # Fallback
        base = Path(__file__).parent / "runtime" / "shell" / "activation"
        if shell == "fish":
            return base / "xla_config.fish"
        return base / "xla_config.bash"


def install_bash_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install bash completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install to venv/etc/bash_completion.d/
    dest_dir = venv_path / "etc" / "bash_completion.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "heterodyne"

    try:
        shutil.copy2(source, dest)
        if verbose:
            print(f"Installed bash completion to: {dest}")
        return True
    except (OSError, shutil.Error) as e:
        if verbose:
            print(f"Failed to install bash completion: {e}")
        return False


def install_zsh_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install zsh completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install to venv/etc/zsh/
    dest_dir = venv_path / "etc" / "zsh"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "heterodyne-completion.zsh"

    try:
        # Create zsh wrapper that sources bash completion
        content = f"""# Zsh completion for heterodyne (generated)
# Source the bash completion in zsh-compatible mode

autoload -Uz bashcompinit
bashcompinit

source "{source}"
"""
        dest.write_text(content)
        if verbose:
            print(f"Installed zsh completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install zsh completion: {e}")
        return False


def install_fish_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install fish completion (basic support).

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    # Fish completions go to a specific location
    dest_dir = venv_path / "share" / "fish" / "vendor_completions.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "heterodyne.fish"

    try:
        # Basic fish completion
        content = """# Fish completion for heterodyne (generated)

complete -c heterodyne -s c -l config -d 'Configuration file' -F
complete -c heterodyne -s d -l data-file -d 'Input data file' -F
complete -c heterodyne -s m -l method -d 'Optimization method' -a 'nlsq cmc'
complete -c heterodyne -s v -l verbose -d 'Verbose output'
complete -c heterodyne -s q -l quiet -d 'Quiet output'
complete -c heterodyne -s h -l help -d 'Show help'
complete -c heterodyne -l version -d 'Show version'

complete -c heterodyne-config -s o -l output -d 'Output file' -F
complete -c heterodyne-config -l template -d 'Template type' -a 'default minimal cmc'
complete -c heterodyne-config -s h -l help -d 'Show help'

complete -c heterodyne-post-install -l interactive -d 'Interactive setup'
complete -c heterodyne-post-install -s h -l help -d 'Show help'

complete -c heterodyne-cleanup -l dry-run -d 'Show what would be removed'
complete -c heterodyne-cleanup -l force -d 'Force cleanup without confirmation'
complete -c heterodyne-cleanup -s h -l help -d 'Show help'
"""
        dest.write_text(content)
        if verbose:
            print(f"Installed fish completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install fish completion: {e}")
        return False


def install_shell_completion(
    shell: str | None = None,
    verbose: bool = False,
) -> bool:
    """Install shell completion for the detected or specified shell.

    Args:
        shell: Shell type or None for auto-detection.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping completion install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell == "unknown":
        if verbose:
            print("Could not detect shell type, trying bash completion")
        detected_shell = "bash"

    if verbose:
        print(f"Installing {detected_shell} completion to {venv_path}")

    if detected_shell == "zsh":
        return install_zsh_completion(venv_path, verbose)
    elif detected_shell == "fish":
        return install_fish_completion(venv_path, verbose)
    else:
        return install_bash_completion(venv_path, verbose)


def install_xla_activation(
    shell: str | None = None,
    mode: str = "auto",
    verbose: bool = False,
) -> bool:
    """Install XLA configuration to venv activation script.

    Args:
        shell: Shell type or None for auto-detection.
        mode: XLA mode (auto, nlsq, cmc, cmc-hpc).
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping XLA activation install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell in ("bash", "zsh", "unknown"):
        return _install_xla_bash_activation(venv_path, mode, verbose)
    elif detected_shell == "fish":
        return _install_xla_fish_activation(venv_path, mode, verbose)
    else:
        return False


def _install_xla_bash_activation(
    venv_path: Path,
    mode: str,
    verbose: bool,
) -> bool:
    """Install XLA config to bash/zsh activate script."""
    activate_script = venv_path / "bin" / "activate"
    if not activate_script.exists():
        if verbose:
            print(f"Activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text()
    marker = "# heterodyne XLA configuration"

    if marker in content:
        if verbose:
            print("XLA activation already installed in activate script")
        return True

    # Get source script path
    xla_script = get_xla_config_source_path("bash")

    # Append XLA configuration sourcing
    addition = f"""
{marker}
if [ -f "{xla_script}" ]; then
    source "{xla_script}" {mode}
fi
"""

    try:
        with open(activate_script, "a") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify activate script: {e}")
        return False


def _install_xla_fish_activation(
    venv_path: Path,
    mode: str,
    verbose: bool,
) -> bool:
    """Install XLA config to fish activate script."""
    activate_script = venv_path / "bin" / "activate.fish"
    if not activate_script.exists():
        if verbose:
            print(f"Fish activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text()
    marker = "# heterodyne XLA configuration"

    if marker in content:
        if verbose:
            print("XLA activation already installed in fish activate script")
        return True

    # Get source script path
    xla_script = get_xla_config_source_path("fish")

    # Append XLA configuration sourcing
    addition = f"""
{marker}
if test -f "{xla_script}"
    source "{xla_script}" {mode}
end
"""

    try:
        with open(activate_script, "a") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify fish activate script: {e}")
        return False


def configure_xla_mode(mode: str = "auto", verbose: bool = False) -> bool:
    """Configure the XLA mode in the user's home directory.

    Args:
        mode: XLA mode (auto, nlsq, cmc, cmc-hpc, or a number).
        verbose: Print verbose output.

    Returns:
        True if configuration succeeded.
    """
    config_file = Path.home() / ".heterodyne_xla_mode"

    try:
        config_file.write_text(mode)
        if verbose:
            print(f"Set XLA mode to '{mode}' in {config_file}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to write XLA mode config: {e}")
        return False


def interactive_setup() -> None:
    """Run interactive post-installation setup."""
    print("=" * 60)
    print("Heterodyne Post-Installation Setup")
    print("=" * 60)
    print()

    # Detect environment
    shell = detect_shell_type()
    in_venv = is_virtual_environment()
    is_conda = is_conda_environment()

    print(f"Detected shell: {shell}")
    print(f"Virtual environment: {in_venv}")
    if is_conda:
        print(f"Conda environment: {os.environ.get('CONDA_PREFIX', '')}")
    elif in_venv:
        print(f"Venv path: {get_venv_path()}")
    print()

    if not in_venv:
        print("WARNING: Not running in a virtual environment.")
        print("Shell completion and XLA activation require a virtual environment.")
        print()
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Shell completion
    print("\n--- Shell Completion ---")
    response = input(f"Install {shell} shell completion? [Y/n]: ").strip().lower()
    if response != "n":
        success = install_shell_completion(shell, verbose=True)
        if success:
            print("Shell completion installed successfully!")
            if shell == "zsh":
                print(
                    "Add to ~/.zshrc: source $VIRTUAL_ENV/etc/zsh/heterodyne-completion.zsh"
                )
            elif shell == "bash":
                print(
                    "Add to ~/.bashrc: source $VIRTUAL_ENV/etc/bash_completion.d/heterodyne"
                )
        else:
            print("Shell completion installation failed.")
    print()

    # XLA Configuration
    print("\n--- XLA Configuration ---")
    print("XLA modes control how many CPU devices JAX uses:")
    print("  auto    - Auto-detect based on CPU cores (recommended)")
    print("  nlsq    - Single device for NLSQ fitting")
    print("  cmc     - 4 devices for CMC sampling")
    print("  cmc-hpc - 8 devices for HPC CMC")
    print()

    mode = input("Select XLA mode [auto]: ").strip().lower() or "auto"
    if mode not in ("auto", "nlsq", "cmc", "cmc-hpc"):
        # Check if it's a number
        try:
            int(mode)
        except ValueError:
            print(f"Invalid mode: {mode}, using 'auto'")
            mode = "auto"

    success = configure_xla_mode(mode, verbose=True)
    if success:
        print(f"XLA mode set to '{mode}'")

    # Install XLA activation
    response = (
        input("\nAdd XLA config to venv activate script? [Y/n]: ").strip().lower()
    )
    if response != "n":
        success = install_xla_activation(shell, mode, verbose=True)
        if success:
            print("XLA activation installed!")
            print("Deactivate and reactivate your venv to apply.")
        else:
            print("XLA activation installation failed.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("To verify installation, run: heterodyne-validate")
    print("=" * 60)


def main() -> int:
    """CLI entry point for heterodyne-post-install."""
    parser = argparse.ArgumentParser(
        description="Post-installation setup for heterodyne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  heterodyne-post-install                  # Interactive setup
  heterodyne-post-install --shell zsh      # Install zsh completion
  heterodyne-post-install --no-xla         # Skip XLA configuration
""",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive setup (default if no options)",
    )
    parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish"],
        help="Shell type for completion installation",
    )
    parser.add_argument(
        "--no-completion",
        action="store_true",
        help="Skip shell completion installation",
    )
    parser.add_argument(
        "--no-xla",
        action="store_true",
        help="Skip XLA configuration",
    )
    parser.add_argument(
        "--xla-mode",
        choices=["auto", "nlsq", "cmc", "cmc-hpc"],
        default="auto",
        help="XLA configuration mode (default: auto)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run interactive setup if no specific options given
    if args.interactive or (
        not args.no_completion and not args.no_xla and not args.shell
    ):
        interactive_setup()
        return 0

    # Non-interactive mode
    success = True

    if not args.no_completion:
        result = install_shell_completion(args.shell, args.verbose)
        if not result:
            print("Shell completion installation failed")
            success = False

    if not args.no_xla:
        result = configure_xla_mode(args.xla_mode, args.verbose)
        if not result:
            print("XLA mode configuration failed")
            success = False

        result = install_xla_activation(args.shell, args.xla_mode, args.verbose)
        if not result:
            print("XLA activation installation failed")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
