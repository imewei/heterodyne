"""Path validation and filesystem utilities."""

from __future__ import annotations

import os
from pathlib import Path


class PathValidationError(Exception):
    """Raised when path validation fails."""


def resolve_path(path: str | Path) -> Path:
    """Resolve path to absolute, expanding user and symlinks.
    
    Args:
        path: Path string or Path object
        
    Returns:
        Resolved absolute Path
    """
    return Path(path).expanduser().resolve()


def validate_file_exists(path: str | Path, description: str = "File") -> Path:
    """Validate that a file exists and is readable.
    
    Args:
        path: Path to validate
        description: Description for error messages
        
    Returns:
        Resolved Path object
        
    Raises:
        PathValidationError: If file doesn't exist or isn't readable
    """
    resolved = resolve_path(path)

    if not resolved.exists():
        raise PathValidationError(f"{description} not found: {resolved}")

    if not resolved.is_file():
        raise PathValidationError(f"{description} is not a file: {resolved}")

    if not os.access(resolved, os.R_OK):
        raise PathValidationError(f"{description} is not readable: {resolved}")

    return resolved


def validate_output_path(path: str | Path, create_parents: bool = True) -> Path:
    """Validate and prepare output path.
    
    Args:
        path: Output path to validate
        create_parents: Whether to create parent directories
        
    Returns:
        Resolved Path object
        
    Raises:
        PathValidationError: If path is invalid
    """
    resolved = resolve_path(path)

    if resolved.exists() and resolved.is_dir():
        raise PathValidationError(f"Output path is a directory: {resolved}")

    if create_parents:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    elif not resolved.parent.exists():
        raise PathValidationError(f"Parent directory does not exist: {resolved.parent}")

    return resolved


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Resolved Path object
    """
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
