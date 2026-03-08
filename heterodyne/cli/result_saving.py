"""Result persistence utilities for heterodyne CLI."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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


# ---------------------------------------------------------------------------
# Structured result extraction helpers
# ---------------------------------------------------------------------------


def _create_mcmc_parameters_dict(result: CMCResult) -> dict[str, Any]:
    """Extract posterior parameter summaries from a CMC result.

    For each parameter, computes mean, std, and 95% credible interval.
    When posterior samples are available the credible interval is computed
    from the 2.5th and 97.5th percentiles; otherwise a Gaussian
    approximation using ``posterior_mean +/- 1.96 * posterior_std`` is used.

    Args:
        result: Completed CMC analysis result.

    Returns:
        Dict mapping parameter name to ``{mean, std, ci_lower, ci_upper}``.
    """
    params: dict[str, Any] = {}

    for i, name in enumerate(result.parameter_names):
        mean = float(result.posterior_mean[i])
        std = float(result.posterior_std[i])

        # Try sample-based credible intervals first
        samples = result.get_samples(name) if result.samples is not None else None
        if samples is not None and len(samples) > 0:
            ci_lower = float(np.percentile(samples, 2.5))
            ci_upper = float(np.percentile(samples, 97.5))
        elif name in result.credible_intervals:
            ci = result.credible_intervals[name]
            ci_lower = ci.get("lower_95", mean - 1.96 * std)
            ci_upper = ci.get("upper_95", mean + 1.96 * std)
        else:
            # Gaussian fallback
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std

        params[name] = {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return params


def _create_mcmc_analysis_dict(
    result: CMCResult,
    c2_data: Any | None = None,
) -> dict[str, Any]:
    """Extract analysis quality metrics from a CMC result.

    Gathers chi-squared, log-likelihood, sample counts, and chain counts
    from ``result.metadata`` (when available).

    Args:
        result: Completed CMC analysis result.
        c2_data: Optional observed correlation data (reserved for future
            goodness-of-fit computation).

    Returns:
        Dict with available analysis quality metrics.
    """
    analysis: dict[str, Any] = {}

    # Metrics from metadata
    if "chi_squared" in result.metadata:
        analysis["chi_squared"] = float(result.metadata["chi_squared"])
    if "log_likelihood" in result.metadata:
        analysis["log_likelihood"] = float(result.metadata["log_likelihood"])

    # Sampling configuration
    analysis["n_samples"] = result.metadata.get("num_samples", result.num_samples)
    analysis["n_chains"] = result.metadata.get("num_chains", result.num_chains)
    analysis["n_warmup"] = result.num_warmup

    if result.wall_time_seconds is not None:
        analysis["wall_time_seconds"] = result.wall_time_seconds

    return analysis


def _create_mcmc_diagnostics_dict(result: CMCResult) -> dict[str, Any]:
    """Extract convergence diagnostics from a CMC result.

    Collects R-hat, bulk ESS, BFMI, divergence count, and an overall
    convergence flag (max R-hat < 1.1 and min bulk ESS > 400).

    Args:
        result: Completed CMC analysis result.

    Returns:
        Dict with diagnostic arrays keyed by parameter name plus
        aggregate convergence indicators.
    """
    diagnostics: dict[str, Any] = {}

    # Per-parameter R-hat
    r_hat_dict: dict[str, float] = {}
    if result.r_hat is not None:
        for i, name in enumerate(result.parameter_names):
            r_hat_dict[name] = float(result.r_hat[i])
    diagnostics["r_hat"] = r_hat_dict

    # Per-parameter ESS (bulk)
    ess_dict: dict[str, float] = {}
    if result.ess_bulk is not None:
        for i, name in enumerate(result.parameter_names):
            ess_dict[name] = float(result.ess_bulk[i])
    diagnostics["ess_bulk"] = ess_dict

    # Per-chain BFMI
    diagnostics["bfmi"] = list(result.bfmi) if result.bfmi is not None else []

    # Divergence count
    diagnostics["n_divergences"] = int(result.metadata.get("n_divergences", 0))

    # Convergence check: max R-hat < 1.1 and min ESS > 400
    if result.r_hat is not None and result.ess_bulk is not None:
        max_r_hat = float(np.max(result.r_hat))
        min_ess = float(np.min(result.ess_bulk))
        diagnostics["convergence_passed"] = bool(
            max_r_hat < 1.1 and min_ess > 400
        )
        diagnostics["max_r_hat"] = max_r_hat
        diagnostics["min_ess_bulk"] = min_ess
    else:
        diagnostics["convergence_passed"] = result.convergence_passed

    return diagnostics


def _extract_nlsq_metadata(result: NLSQResult) -> dict[str, Any]:
    """Extract key metadata from an NLSQ result.

    Gathers fit quality metrics, parameter values, and uncertainties
    into a flat dictionary suitable for JSON serialization.

    Args:
        result: Completed NLSQ result.

    Returns:
        Dict with fit statistics, parameter values, and uncertainties.
    """
    meta: dict[str, Any] = {
        "success": result.success,
        "n_iterations": result.n_iterations,
    }

    if result.reduced_chi_squared is not None:
        meta["reduced_chi_squared"] = float(result.reduced_chi_squared)
    if result.final_cost is not None:
        meta["cost"] = float(result.final_cost)

    # Parameter values
    meta["parameter_values"] = {
        name: float(result.parameters[i])
        for i, name in enumerate(result.parameter_names)
    }

    # Parameter uncertainties from covariance diagonal
    if result.uncertainties is not None:
        meta["parameter_uncertainties"] = {
            name: float(result.uncertainties[i])
            for i, name in enumerate(result.parameter_names)
        }
    else:
        meta["parameter_uncertainties"] = None

    # Jacobian norm if stored
    if "jacobian_norm" in result.metadata:
        meta["jacobian_norm"] = float(result.metadata["jacobian_norm"])

    if result.wall_time_seconds is not None:
        meta["wall_time_seconds"] = result.wall_time_seconds

    return meta


def _prepare_parameter_data(result: CMCResult | NLSQResult) -> dict[str, Any]:
    """Unified parameter extraction for either result type.

    Returns a dict mapping each parameter name to its best-estimate
    value and uncertainty regardless of whether the source is an NLSQ
    point estimate or a CMC posterior.

    Args:
        result: Either a ``CMCResult`` or ``NLSQResult``.

    Returns:
        Dict mapping parameter name to ``{value, uncertainty}``.
    """
    from heterodyne.optimization.cmc.results import CMCResult as _CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult as _NLSQResult

    params: dict[str, Any] = {}

    if isinstance(result, _CMCResult):
        for i, name in enumerate(result.parameter_names):
            params[name] = {
                "value": float(result.posterior_mean[i]),
                "uncertainty": float(result.posterior_std[i]),
            }
    elif isinstance(result, _NLSQResult):
        for i, name in enumerate(result.parameter_names):
            unc: float | None = None
            if result.uncertainties is not None:
                unc = float(result.uncertainties[i])
            params[name] = {
                "value": float(result.parameters[i]),
                "uncertainty": unc,
            }

    return params


def _compute_theoretical_g1_from_mcmc(
    result: CMCResult,
    model: Any,
) -> Any:
    """Compute theoretical g1 correlation from posterior mean parameters.

    This is heterodyne-specific: the function computes g1 (the field
    correlation) rather than the intensity correlation c2 used in
    homodyne analysis.

    Args:
        result: Completed CMC result with posterior means.
        model: Model object that exposes a ``compute_g1`` method.

    Returns:
        ndarray of theoretical g1 values, or ``None`` if computation
        is not possible.
    """
    if model is None:
        logger.debug("No model provided; skipping theoretical g1 computation.")
        return None

    param_dict = result.params_dict()

    if hasattr(model, "compute_g1"):
        try:
            g1 = model.compute_g1(param_dict)
            logger.debug("Computed theoretical g1 from posterior means.")
            return g1
        except Exception:
            logger.warning(
                "Failed to compute theoretical g1 from posterior means.",
                exc_info=True,
            )
            return None

    logger.warning(
        "Model does not expose compute_g1(); cannot compute theoretical g1."
    )
    return None


def save_results(
    method: str,
    nlsq_results: list[NLSQResult] | None,
    cmc_results: list[CMCResult] | None,
    output_dir: Path,
    phi_angles: list[float] | None = None,
    model: Any | None = None,
) -> dict[str, list[Path]]:
    """Unified dispatcher that saves all results and a summary JSON.

    Creates an ``output_dir / "results"`` subdirectory and delegates to
    :func:`save_nlsq_results` and :func:`save_cmc_results` for the
    per-angle files.  Additionally writes a unified summary JSON that
    aggregates method, timestamp, phi angles, per-angle parameter
    summaries, and per-angle diagnostics (for CMC).

    Args:
        method: Analysis method identifier (e.g. ``"nlsq"``, ``"cmc"``,
            ``"nlsq+cmc"``).
        nlsq_results: NLSQ results per angle, or ``None``.
        cmc_results: CMC results per angle, or ``None``.
        output_dir: Base output directory.
        phi_angles: Phi angles corresponding to the results lists.
        model: Optional model for theoretical g1 computation.

    Returns:
        Dict with keys ``"nlsq"``, ``"cmc"``, ``"summary"`` mapping to
        lists of saved file paths.
    """
    from heterodyne.io.json_utils import json_safe

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, list[Path]] = {"nlsq": [], "cmc": [], "summary": []}
    angles = phi_angles or []

    # --- Save NLSQ results ---
    if nlsq_results and angles:
        saved["nlsq"] = save_nlsq_results(nlsq_results, results_dir, angles)

    # --- Save CMC results ---
    if cmc_results and angles:
        saved["cmc"] = save_cmc_results(cmc_results, results_dir, angles)

    # --- Build unified summary ---
    summary: dict[str, Any] = {
        "method": method,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "phi_angles": angles,
    }

    # Per-angle NLSQ summaries
    if nlsq_results:
        nlsq_summaries: list[dict[str, Any]] = []
        for i, res in enumerate(nlsq_results):
            angle = angles[i] if i < len(angles) else None
            entry: dict[str, Any] = {"phi": angle}
            entry["parameters"] = _prepare_parameter_data(res)
            entry["metadata"] = _extract_nlsq_metadata(res)
            nlsq_summaries.append(entry)
        summary["nlsq"] = nlsq_summaries

    # Per-angle CMC summaries
    if cmc_results:
        cmc_summaries: list[dict[str, Any]] = []
        for i, res in enumerate(cmc_results):
            angle = angles[i] if i < len(angles) else None
            entry = {"phi": angle}
            entry["parameters"] = _create_mcmc_parameters_dict(res)
            entry["analysis"] = _create_mcmc_analysis_dict(res)
            entry["diagnostics"] = _create_mcmc_diagnostics_dict(res)

            # Theoretical g1 (best-effort)
            g1 = _compute_theoretical_g1_from_mcmc(res, model)
            if g1 is not None:
                entry["theoretical_g1_shape"] = list(np.shape(g1))

            cmc_summaries.append(entry)
        summary["cmc"] = cmc_summaries

    # Write summary JSON
    summary_path = results_dir / "unified_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    saved["summary"] = [summary_path]

    logger.info(
        "Saved unified results summary to %s (%d NLSQ, %d CMC files)",
        summary_path,
        len(saved["nlsq"]),
        len(saved["cmc"]),
    )

    return saved
