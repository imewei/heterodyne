"""Shell completion and activation scripts for heterodyne.

This subpackage contains shell scripts for:
- Bash/Zsh completion for heterodyne CLI commands
- XLA_FLAGS auto-configuration scripts

The scripts are installed by the post_install module and can be
sourced in shell startup files or virtual environment activation scripts.
"""

from pathlib import Path

# Path to shell scripts
SHELL_DIR = Path(__file__).parent
COMPLETION_SCRIPT = SHELL_DIR / "completion.sh"
ACTIVATION_DIR = SHELL_DIR / "activation"
XLA_CONFIG_BASH = ACTIVATION_DIR / "xla_config.bash"
XLA_CONFIG_FISH = ACTIVATION_DIR / "xla_config.fish"


def get_completion_script() -> str:
    """Get the path to the bash completion script.

    Returns:
        Absolute path to completion.sh

    Raises:
        FileNotFoundError: If the completion script is not present in the installation.
    """
    path = COMPLETION_SCRIPT.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Completion script not found at {path}. "
            "Re-install the package to restore shell script assets."
        )
    return str(path)


def get_xla_config_script(shell: str = "bash") -> str:
    """Get the path to the XLA configuration script.

    Args:
        shell: Shell type ("bash", "zsh", or "fish")

    Returns:
        Absolute path to the appropriate XLA config script.

    Raises:
        FileNotFoundError: If the script is not present in the installation.
        ValueError: If an unsupported shell is specified.
    """
    if shell in ("bash", "zsh"):
        path = XLA_CONFIG_BASH.resolve()
    elif shell == "fish":
        path = XLA_CONFIG_FISH.resolve()
    else:
        raise ValueError(f"Unsupported shell: {shell}")
    if not path.exists():
        raise FileNotFoundError(
            f"XLA config script not found at {path}. "
            "Re-install the package to restore shell script assets."
        )
    return str(path)


__all__ = [
    "SHELL_DIR",
    "COMPLETION_SCRIPT",
    "ACTIVATION_DIR",
    "XLA_CONFIG_BASH",
    "XLA_CONFIG_FISH",
    "get_completion_script",
    "get_xla_config_script",
]
