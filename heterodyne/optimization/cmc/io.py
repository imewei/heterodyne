"""Shard I/O for CMC (Consensus Monte Carlo) results.

Provides functions to persist and reload posterior samples as ``.npz``
archives and ArviZ ``InferenceData`` objects as NetCDF files.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Shard I/O (NumPy .npz)
# ------------------------------------------------------------------


def save_shard_results(
    results: dict[str, np.ndarray],
    output_dir: str | Path,
    shard_id: int,
) -> Path:
    """Save posterior samples for a single shard as a ``.npz`` archive.

    The file is written to ``<output_dir>/shard_<shard_id>.npz``.

    Args:
        results: Mapping of parameter name to sample array.
        output_dir: Directory in which to save the archive.
        shard_id: Integer shard identifier.

    Returns:
        Path to the saved file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{shard_id}.npz"

    np.savez(path, **results)
    logger.info("Saved shard %d (%d params) to %s", shard_id, len(results), path)
    return path


def load_shard_results(
    output_dir: str | Path,
    shard_id: int,
) -> dict[str, np.ndarray]:
    """Load posterior samples for a single shard.

    Args:
        output_dir: Directory containing shard archives.
        shard_id: Integer shard identifier.

    Returns:
        Mapping of parameter name to sample array.

    Raises:
        FileNotFoundError: If the shard file does not exist.
    """
    path = Path(output_dir) / f"shard_{shard_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Shard file not found: {path}")

    with np.load(path) as data:
        result = {key: data[key] for key in data.files}

    logger.info("Loaded shard %d (%d params) from %s", shard_id, len(result), path)
    return result


def list_shards(output_dir: str | Path) -> list[int]:
    """Discover saved shard IDs in *output_dir*.

    Scans for files matching ``shard_<N>.npz`` and returns a sorted list
    of the integer shard IDs.

    Args:
        output_dir: Directory to scan.

    Returns:
        Sorted list of shard IDs found.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return []

    pattern = re.compile(r"^shard_(\d+)\.npz$")
    ids: list[int] = []
    for p in out.iterdir():
        m = pattern.match(p.name)
        if m is not None:
            ids.append(int(m.group(1)))

    ids.sort()
    logger.debug("Found %d shards in %s", len(ids), out)
    return ids


# ------------------------------------------------------------------
# ArviZ InferenceData I/O (NetCDF)
# ------------------------------------------------------------------


def save_inference_data(idata: object, path: str | Path) -> Path:
    """Save an ArviZ InferenceData object as a NetCDF file.

    Args:
        idata: ArviZ InferenceData instance.
        path: Destination file path (should end in ``.nc``).

    Returns:
        Path to the saved file.

    Raises:
        ImportError: If ``arviz`` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required for InferenceData I/O.  "
            "Install it with: uv add arviz"
        ) from exc

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, str(save_path))
    logger.info("Saved InferenceData to %s", save_path)
    return save_path


def load_inference_data(path: str | Path) -> object:
    """Load an ArviZ InferenceData object from a NetCDF file.

    Args:
        path: Path to a NetCDF file previously written by
            :func:`save_inference_data`.

    Returns:
        ArviZ InferenceData object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If ``arviz`` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required for InferenceData I/O.  "
            "Install it with: uv add arviz"
        ) from exc

    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"InferenceData file not found: {load_path}")

    idata = az.from_netcdf(str(load_path))
    logger.info("Loaded InferenceData from %s", load_path)
    return idata
