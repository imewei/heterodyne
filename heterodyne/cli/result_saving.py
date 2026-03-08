"""Result persistence utilities for heterodyne CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def save_nlsq_results(
    results: list[NLSQResult],
    output_dir: Path,
    phi_angles: list[float],
) -> list[Path]:
    """Save NLSQ results to disk.

    Args:
        results: NLSQ results to save.
        output_dir: Output directory.
        phi_angles: Corresponding phi angles.

    Returns:
        List of paths to saved files.
    """
    from heterodyne.io.nlsq_writers import save_nlsq_json_files, save_nlsq_npz_file

    saved_paths: list[Path] = []

    for result, phi in zip(results, phi_angles, strict=True):
        prefix = f"nlsq_phi{int(phi)}" if len(phi_angles) > 1 else "nlsq"

        json_paths = save_nlsq_json_files(result, output_dir, prefix=prefix)
        saved_paths.extend(json_paths.values())

        npz_path = output_dir / f"{prefix}_data.npz"
        save_nlsq_npz_file(result, npz_path)
        saved_paths.append(npz_path)

    logger.info("Saved %d NLSQ result files to %s", len(saved_paths), output_dir)
    return saved_paths


def save_cmc_results(
    results: list[CMCResult],
    output_dir: Path,
    phi_angles: list[float],
) -> list[Path]:
    """Save CMC results to disk.

    Args:
        results: CMC results to save.
        output_dir: Output directory.
        phi_angles: Corresponding phi angles.

    Returns:
        List of paths to saved files.
    """
    from heterodyne.io.mcmc_writers import save_mcmc_results

    saved_paths: list[Path] = []

    for result, phi in zip(results, phi_angles, strict=True):
        prefix = f"cmc_phi{int(phi)}" if len(phi_angles) > 1 else "cmc"
        result_paths = save_mcmc_results(result, output_dir, prefix=prefix)
        saved_paths.extend(result_paths.values())

    logger.info("Saved %d CMC result files to %s", len(saved_paths), output_dir)
    return saved_paths


def save_summary_manifest(
    nlsq_paths: list[Path],
    cmc_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Write a JSON manifest summarizing all saved result files.

    Args:
        nlsq_paths: Paths to NLSQ result files.
        cmc_paths: Paths to CMC result files.
        output_dir: Output directory for the manifest.

    Returns:
        Path to the manifest file.
    """
    manifest = {
        "nlsq_files": [str(p) for p in nlsq_paths],
        "cmc_files": [str(p) for p in cmc_paths],
        "total_files": len(nlsq_paths) + len(cmc_paths),
    }

    manifest_path = output_dir / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved results manifest to %s", manifest_path)
    return manifest_path
