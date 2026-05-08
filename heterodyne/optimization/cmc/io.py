"""Shard I/O for CMC (Consensus Monte Carlo) results.

Provides functions to persist and reload posterior samples as ``.npz``
archives and ArviZ ``InferenceData`` objects as NetCDF files.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult

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
            "arviz is required for InferenceData I/O.  Install it with: uv add arviz"
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
            "arviz is required for InferenceData I/O.  Install it with: uv add arviz"
        ) from exc

    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"InferenceData file not found: {load_path}")

    idata = az.from_netcdf(str(load_path))
    logger.info("Loaded InferenceData from %s", load_path)
    return idata


# ---------------------------------------------------------------------------
# Full result serialization pipeline (homodyne parity)
# ---------------------------------------------------------------------------

SAMPLES_SCHEMA_VERSION = (1, 0)


def _r_hat_dict(result: CMCResult) -> dict[str, float]:
    if result.r_hat is None:
        return {n: float("nan") for n in result.parameter_names}
    return {n: float(result.r_hat[i]) for i, n in enumerate(result.parameter_names)}


def _ess_bulk_dict(result: CMCResult) -> dict[str, float]:
    if result.ess_bulk is None:
        return {n: float("nan") for n in result.parameter_names}
    return {n: float(result.ess_bulk[i]) for i, n in enumerate(result.parameter_names)}


def _ess_tail_dict(result: CMCResult) -> dict[str, float]:
    if result.ess_tail is None:
        return {n: float("nan") for n in result.parameter_names}
    return {n: float(result.ess_tail[i]) for i, n in enumerate(result.parameter_names)}


def save_samples_npz(result: CMCResult, output_path: Path) -> None:
    """Save posterior samples as a compressed NPZ archive (ArviZ-compatible).

    Shape stored: ``posterior_samples`` is ``(n_chains, n_samples, n_params)``.
    """
    from heterodyne.config.parameter_registry import ParameterRegistry

    samples_3d = result.get_samples_array()
    names = result.parameter_names

    r_hat_arr = np.array(
        [
            float(result.r_hat[i]) if result.r_hat is not None else np.nan
            for i in range(len(names))
        ]
    )
    ess_bulk_arr = np.array(
        [
            float(result.ess_bulk[i]) if result.ess_bulk is not None else np.nan
            for i in range(len(names))
        ]
    )
    ess_tail_arr = np.array(
        [
            float(result.ess_tail[i]) if result.ess_tail is not None else np.nan
            for i in range(len(names))
        ]
    )

    scaling = ParameterRegistry().get_scaling_names()
    first_s = scaling[0] if scaling else "contrast"
    n_phi = sum(1 for n in names if n.startswith(f"{first_s}_")) or (
        1 if first_s in names else 0
    )

    np.savez_compressed(
        output_path,
        schema_version=np.array(SAMPLES_SCHEMA_VERSION),
        posterior_samples=samples_3d,
        param_names=np.array(names),
        r_hat=r_hat_arr,
        ess_bulk=ess_bulk_arr,
        ess_tail=ess_tail_arr,
        divergences=np.array([int(result.metadata.get("num_divergences", 0))]),
        analysis_mode=np.array([str(result.metadata.get("analysis_mode", "unknown"))]),
        n_phi=np.array([n_phi]),
        n_chains=np.array([result.num_chains]),
        n_samples=np.array([result.num_samples]),
    )
    logger.info("Saved samples.npz: %s %s", output_path, samples_3d.shape)


def load_samples_npz(input_path: Path) -> dict[str, Any]:
    """Load samples NPZ and return a plain Python dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the suffix is not ``.npz`` or the path is not a regular file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Samples file not found: {input_path}")
    if input_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {input_path.suffix}")
    if not input_path.is_file():
        raise ValueError(f"Path is not a regular file: {input_path}")

    # NPZ archives store raw numpy binary arrays; no object serialization is used.
    data = np.load(input_path)
    return {
        "schema_version": tuple(data["schema_version"]),
        "posterior_samples": data["posterior_samples"],
        "param_names": data["param_names"].tolist(),
        "r_hat": data["r_hat"],
        "ess_bulk": data["ess_bulk"],
        "ess_tail": data["ess_tail"],
        "divergences": int(data["divergences"][0]),
        "analysis_mode": str(data["analysis_mode"][0]),
        "n_phi": int(data["n_phi"][0]),
        "n_chains": int(data["n_chains"][0]),
        "n_samples": int(data["n_samples"][0]),
    }


def samples_to_arviz(samples_data: dict[str, Any]) -> Any:
    """Convert loaded samples dict to ArviZ InferenceData.

    Parameters
    ----------
    samples_data : dict[str, Any]
        Data returned by :func:`load_samples_npz`.
    """
    import arviz as az  # type: ignore[import-untyped]

    samples = samples_data["posterior_samples"]  # (n_chains, n_samples, n_params)
    param_names = samples_data["param_names"]
    posterior = {name: samples[:, :, i] for i, name in enumerate(param_names)}
    return az.from_dict({"posterior": posterior})


def save_fitted_data_npz(
    result: CMCResult,
    c2_exp: np.ndarray,
    c2_fitted: np.ndarray,
    c2_fitted_std: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_angles: np.ndarray,
    q: float,
    output_path: Path,
) -> None:
    """Save fitted C2 data in NLSQ-compatible NPZ format."""
    residuals = c2_exp - c2_fitted
    c2_fitted_5pct = c2_fitted - 1.645 * c2_fitted_std
    c2_fitted_95pct = c2_fitted + 1.645 * c2_fitted_std

    np.savez_compressed(
        output_path,
        c2_exp=c2_exp,
        c2_fitted=c2_fitted,
        residuals=residuals,
        q=np.array([q]),
        phi_angles=phi_angles,
        t1=t1,
        t2=t2,
        c2_fitted_std=c2_fitted_std,
        c2_fitted_5pct=c2_fitted_5pct,
        c2_fitted_95pct=c2_fitted_95pct,
    )
    logger.info("Saved fitted_data.npz: %s shape=%s", output_path, c2_exp.shape)


def save_parameters_json(result: CMCResult, output_path: Path) -> None:
    """Save posterior statistics per parameter to JSON.

    NaN -> null, Inf -> "Infinity"/"-Infinity".
    """
    stats = result.get_posterior_stats()

    def _san(v: float) -> float | str | None:
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v

    stats_json = {
        name: {k: _san(float(val)) for k, val in ps.items()}
        for name, ps in stats.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
    logger.info("Saved parameters.json: %s (%d params)", output_path, len(stats_json))


def save_diagnostics_json(
    result: CMCResult,
    output_path: Path,
    warnings: list[str] | None = None,
) -> None:
    """Save convergence diagnostics to JSON."""
    from heterodyne.optimization.cmc.diagnostics import create_diagnostics_dict

    convergence_status = "converged" if result.convergence_passed else "not_converged"
    num_shards = int(result.metadata.get("n_shards", 1))
    divergences = int(result.metadata.get("num_divergences", 0))
    exec_time = result.wall_time_seconds or 0.0

    diag = create_diagnostics_dict(
        r_hat=_r_hat_dict(result),
        ess_bulk=_ess_bulk_dict(result),
        ess_tail=_ess_tail_dict(result),
        divergences=divergences,
        convergence_status=convergence_status,
        warnings=warnings or [],
        n_chains=result.num_chains,
        n_warmup=result.num_warmup,
        n_samples=result.num_samples,
        warmup_time=0.0,
        sampling_time=exec_time,
        num_shards=num_shards,
    )

    def _conv(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return _conv(obj.tolist())
        if isinstance(obj, (np.integer, np.floating)):
            v = float(obj)
            if math.isnan(v):
                return None
            if math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
            return v
        if isinstance(obj, dict):
            return {k: _conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_conv(i) for i in obj]
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_conv(diag), f, indent=2)
    logger.info("Saved diagnostics.json: %s", output_path)


def save_all_results(
    result: CMCResult,
    output_dir: Path,
    c2_exp: np.ndarray | None = None,
    c2_fitted: np.ndarray | None = None,
    c2_fitted_std: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    phi_angles: np.ndarray | None = None,
    q: float | None = None,
) -> dict[str, Path]:
    """Save all CMC result files and return a dict of ``{key: path}``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    save_samples_npz(result, output_dir / "samples.npz")
    saved["samples"] = output_dir / "samples.npz"

    save_parameters_json(result, output_dir / "parameters.json")
    saved["parameters"] = output_dir / "parameters.json"

    save_diagnostics_json(result, output_dir / "diagnostics.json")
    saved["diagnostics"] = output_dir / "diagnostics.json"

    if all(
        x is not None for x in [c2_exp, c2_fitted, c2_fitted_std, t1, t2, phi_angles, q]
    ):
        save_fitted_data_npz(
            result=result,
            c2_exp=c2_exp,  # type: ignore[arg-type]
            c2_fitted=c2_fitted,  # type: ignore[arg-type]
            c2_fitted_std=c2_fitted_std,  # type: ignore[arg-type]
            t1=t1,  # type: ignore[arg-type]
            t2=t2,  # type: ignore[arg-type]
            phi_angles=phi_angles,  # type: ignore[arg-type]
            q=q,  # type: ignore[arg-type]
            output_path=output_dir / "fitted_data.npz",
        )
        saved["fitted_data"] = output_dir / "fitted_data.npz"

    return saved
