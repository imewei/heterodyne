"""JSON serialization utilities for JAX arrays and numpy types."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np


def _sanitize_float(value: float) -> float | None:
    """Check a float for NaN/Inf and raise ValueError.

    Args:
        value: Float value to check

    Returns:
        The original value if finite

    Raises:
        ValueError: If value is NaN or Inf
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Cannot serialize non-finite float to JSON: {value!r}")
    return value


def json_safe(obj: Any) -> Any:
    """Convert object to JSON-serializable form.

    Handles:
    - JAX arrays -> lists
    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - complex numbers -> {"real": ..., "imag": ...} dicts
    - Path objects -> strings
    - datetime -> ISO format strings
    - Nested dicts/lists recursively

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable equivalent
    """
    # Handle JAX arrays via proper isinstance check
    try:
        import jax

        if isinstance(obj, jax.Array):
            return json_safe(np.asarray(obj))
    except ImportError:
        pass

    # Handle complex numpy arrays — preserve shape for round-trip
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.complexfloating):
        return {
            "__complex_array__": True,
            "shape": list(obj.shape),
            "data": [{"real": float(z.real), "imag": float(z.imag)} for z in obj.flat],
        }

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy complex scalars
    if isinstance(obj, np.complexfloating):
        return {"real": obj.real.item(), "imag": obj.imag.item()}

    # Handle Python complex
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}

    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Handle datetime
    from datetime import datetime

    if isinstance(obj, datetime):
        return obj.isoformat()

    # Handle nested structures
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]

    # Return primitives as-is (with NaN/Inf check for floats)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        _sanitize_float(obj)
        return obj

    # Fallback: try string conversion
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def json_serializer(obj: Any) -> str:
    """Serialize object to JSON string with pretty formatting.

    Args:
        obj: Object to serialize

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(json_safe(obj), indent=2, allow_nan=False)


def load_json(path: Path | str) -> dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def save_json(data: Any, path: Path | str) -> None:
    """Save data to JSON file with pretty formatting.

    Uses atomic write (write-to-temp + rename) to prevent partial writes.

    Args:
        data: Data to save
        path: Output path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_serializer(data))
        os.replace(tmp_path, str(output_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
