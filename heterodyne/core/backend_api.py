"""Backend selection and abstraction for heterodyne computations.

The heterodyne package is JAX-first, but this module provides a clean
abstraction for the rare case when JAX is unavailable (e.g., documentation
building, quick inspection, or lightweight environments).

Typical usage::

    from heterodyne.core.backend_api import get_current_backend, get_array_module

    backend = get_current_backend()
    xp = get_array_module(backend)
    x = xp.linspace(0, 1, 100)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, unique
from types import ModuleType
from typing import Any

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level cache for JAX availability probe
_jax_available: bool | None = None


@unique
class Backend(Enum):
    """Supported computational backends."""

    JAX = "jax"
    NUMPY = "numpy"


@dataclass
class BackendConfig:
    """Configuration for the computational backend.

    Attributes:
        backend: Which backend to use.
        device: Target device string (e.g., 'cpu', 'gpu').
        precision: Floating-point precision ('float32' or 'float64').
    """

    backend: Backend = Backend.JAX
    device: str = "cpu"
    precision: str = "float64"


def _check_jax_available() -> bool:
    """Probe whether JAX can be imported and is functional.

    The result is cached at module level to avoid repeated import attempts.

    Returns:
        True if JAX is importable and functional.
    """
    global _jax_available  # noqa: PLW0603
    if _jax_available is not None:
        return _jax_available

    try:
        import jax.numpy as jnp  # noqa: F811

        # Smoke test: ensure basic operations work
        _ = jnp.array([1.0, 2.0])
        _jax_available = True
    except Exception:
        _jax_available = False

    logger.debug("JAX availability check: %s", _jax_available)
    return _jax_available


def get_current_backend() -> Backend:
    """Detect the best available backend.

    Returns Backend.JAX if JAX is importable and functional, otherwise
    returns Backend.NUMPY.

    Returns:
        The detected backend.
    """
    if _check_jax_available():
        return Backend.JAX
    return Backend.NUMPY


def set_backend(backend: Backend) -> None:
    """Configure the environment for the specified backend.

    For JAX: sets float64 precision and CPU platform flags.
    For NUMPY: no environment changes needed.

    This should be called early, before JAX traces any functions, because
    JAX configuration flags are read at import time.

    Args:
        backend: The backend to configure.
    """
    if backend is Backend.JAX:
        # Enable float64 (JAX defaults to float32)
        os.environ.setdefault("JAX_ENABLE_X64", "True")

        # Default to CPU if no platform is set
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

        logger.info(
            "Configured JAX backend: JAX_ENABLE_X64=%s, JAX_PLATFORMS=%s",
            os.environ.get("JAX_ENABLE_X64"),
            os.environ.get("JAX_PLATFORMS"),
        )
    elif backend is Backend.NUMPY:
        logger.info("Configured NumPy-only backend (no JAX).")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_array_module(backend: Backend | None = None) -> ModuleType:
    """Return the appropriate array module for the given backend.

    Args:
        backend: Backend to use. If None, auto-detects via
            :func:`get_current_backend`.

    Returns:
        ``jax.numpy`` for JAX backend, ``numpy`` for NumPy backend.

    Raises:
        RuntimeError: If JAX backend is requested but JAX is not available.
    """
    if backend is None:
        backend = get_current_backend()

    if backend is Backend.JAX:
        if not _check_jax_available():
            raise RuntimeError(
                "JAX backend requested but JAX is not available. "
                "Install JAX or use Backend.NUMPY."
            )
        import jax.numpy as jnp

        return jnp

    return np


def ensure_array(
    x: Any,
    backend: Backend | None = None,
) -> Any:
    """Convert input to an array of the appropriate backend type.

    - For JAX: converts to ``jax.numpy.ndarray`` via ``jnp.asarray``.
    - For NumPy: converts to ``numpy.ndarray`` via ``np.asarray``.

    Args:
        x: Input data (array-like, scalar, list, etc.).
        backend: Backend to target. If None, auto-detects.

    Returns:
        Array in the target backend's format.
    """
    if backend is None:
        backend = get_current_backend()

    xp = get_array_module(backend)
    return xp.asarray(x)
