"""Command-line interface for heterodyne analysis."""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy imports to avoid eagerly loading heavy dependencies (JAX, etc.)."""
    _imports = {
        "main": ("heterodyne.cli.main", "main"),
        "config_main": ("heterodyne.cli.config_generator", "main"),
        "configure_xla": ("heterodyne.cli.xla_config", "configure_xla"),
        "create_parser": ("heterodyne.cli.args_parser", "create_parser"),
        "dispatch_command": ("heterodyne.cli.commands", "dispatch_command"),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "main",
    "config_main",
    "configure_xla",
    "create_parser",
    "dispatch_command",
]
