"""JSON serialization utilities for JAX arrays and numpy types."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def json_safe(obj: Any) -> Any:
    """Convert object to JSON-serializable form.
    
    Handles:
    - JAX arrays -> lists
    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - Path objects -> strings
    - datetime -> ISO format strings
    - Nested dicts/lists recursively
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable equivalent
    """
    # Handle JAX arrays (check for jax.Array)
    if hasattr(obj, "__jax_array__") or type(obj).__module__.startswith("jax"):
        return json_safe(np.asarray(obj))
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle nested structures
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    
    # Return primitives as-is
    if obj is None or isinstance(obj, (bool, int, float, str)):
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
    return json.dumps(json_safe(obj), indent=2, cls=DateTimeEncoder)


def load_json(path: Path | str) -> dict[str, Any]:
    """Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str) -> None:
    """Save data to JSON file with pretty formatting.
    
    Args:
        data: Data to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_serializer(data))
