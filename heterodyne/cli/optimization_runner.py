"""Optimization execution for heterodyne CLI.

Manages NLSQ and CMC fitting runs, including warm-start resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.io.mcmc_writers import format_mcmc_summary, save_mcmc_results
from heterodyne.io.nlsq_writers import (
    format_nlsq_summary,
    save_nlsq_json_files,
    save_nlsq_npz_file,
)
from heterodyne.optimization.cmc import CMCConfig, fit_cmc_jax
from heterodyne.optimization.nlsq import NLSQConfig, fit_nlsq_jax
from heterodyne.utils.logging import AnalysisSummaryLogger, get_logger, log_phase

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def run_nlsq(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    summary: AnalysisSummaryLogger | None = None,
) -> list[NLSQResult]:
    """Run NLSQ analysis for all phi angles.

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data (2D or 3D).
        phi_angles: Phi angles to analyze.
        config_manager: Configuration manager.
        args: CLI arguments.
        output_dir: Output directory for results.
        summary: Optional summary logger for phase tracking.

    Returns:
        List of NLSQResult objects, one per phi angle.
    """
    logger.info("Starting NLSQ analysis")

    if getattr(args, "multistart", False):
        config_manager.update_optimization_config("nlsq", "multistart", True)
        config_manager.update_optimization_config(
            "nlsq", "multistart_n", getattr(args, "multistart_n", 10)
        )

    nlsq_config = NLSQConfig.from_dict(config_manager.nlsq_config)
    nlsq_config.verbose = getattr(args, "verbose", 1)

    results: list[NLSQResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info("Fitting phi=%s° (%d/%d)", phi, i + 1, len(phi_angles))

        c2_phi = c2_data[i] if c2_data.ndim == 3 else c2_data

        with log_phase(f"nlsq_phi_{i}", logger=logger, track_memory=True) as phase:
            result = fit_nlsq_jax(
                model=model,
                c2_data=c2_phi,
                phi_angle=phi,
                config=nlsq_config,
            )

        result.metadata["phi_angle"] = phi
        results.append(result)

        logger.info(
            "NLSQ phi=%s° completed in %.2fs",
            phi,
            phase.duration,
        )

        if summary and result.reduced_chi_squared is not None:
            summary.record_metric(
                f"nlsq_chi2_phi{int(phi)}", result.reduced_chi_squared
            )

        print(f"\n{'=' * 50}")
        print(f"NLSQ Results for phi={phi}°")
        print(format_nlsq_summary(result))

        prefix = f"nlsq_phi{int(phi)}" if len(phi_angles) > 1 else "nlsq"
        save_nlsq_json_files(result, output_dir, prefix=prefix)
        save_nlsq_npz_file(result, output_dir / f"{prefix}_data.npz")

    logger.info("NLSQ analysis complete")
    return results


def run_cmc(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    nlsq_results: list[NLSQResult] | None = None,
    summary: AnalysisSummaryLogger | None = None,
) -> list[CMCResult]:
    """Run CMC Bayesian analysis for all phi angles.

    Args:
        model: Configured HeterodyneModel.
        c2_data: Correlation data.
        phi_angles: Phi angles to analyze.
        config_manager: Configuration manager.
        args: CLI arguments.
        output_dir: Output directory.
        nlsq_results: Optional NLSQ results for warm-starting.
        summary: Optional summary logger for phase tracking.

    Returns:
        List of CMCResult objects, one per phi angle.
    """
    logger.info("Starting CMC analysis")

    if getattr(args, "num_samples", None) is not None:
        config_manager.update_optimization_config(
            "cmc", "num_samples", args.num_samples
        )
    if getattr(args, "num_chains", None) is not None:
        config_manager.update_optimization_config("cmc", "num_chains", args.num_chains)

    cmc_config = CMCConfig.from_dict(config_manager.cmc_config)

    results: list[CMCResult] = []

    for i, phi in enumerate(phi_angles):
        logger.info("CMC for phi=%s° (%d/%d)", phi, i + 1, len(phi_angles))

        c2_phi = c2_data[i] if c2_data.ndim == 3 else c2_data

        nlsq_result_i = (
            nlsq_results[i] if nlsq_results and i < len(nlsq_results) else None
        )

        if nlsq_result_i is not None:
            if _validate_warmstart_quality(nlsq_result_i):
                _log_warmstart_physical_params(nlsq_result_i)
            else:
                logger.warning(
                    "Warm-start quality below threshold for phi=%s°; using anyway", phi
                )

        with log_phase(f"cmc_phi_{i}", logger=logger, track_memory=True) as phase:
            result = fit_cmc_jax(
                model=model,
                c2_data=c2_phi,
                phi_angle=phi,
                config=cmc_config,
                nlsq_result=nlsq_result_i,
            )

        result.metadata["phi_angle"] = phi
        results.append(result)

        logger.info(
            "CMC phi=%s° completed in %.2fs",
            phi,
            phase.duration,
        )

        if summary is not None:
            summary.record_metric(
                f"cmc_n_samples_phi{int(phi)}", float(cmc_config.num_samples)
            )

        print(f"\n{'=' * 50}")
        print(f"CMC Results for phi={phi}°")
        print(format_mcmc_summary(result))

        prefix = f"cmc_phi{int(phi)}" if len(phi_angles) > 1 else "cmc"
        save_mcmc_results(result, output_dir, prefix=prefix)

    logger.info("CMC analysis complete")
    return results


def resolve_nlsq_warmstart(
    args: argparse.Namespace,
    output_dir: Path,
) -> NLSQResult | None:
    """Attempt to load previously saved NLSQ results for warm-starting CMC.

    Args:
        args: CLI arguments (may have .warmstart_path).
        output_dir: Default directory to search for NLSQ results.

    Returns:
        NLSQResult if found, None otherwise.
    """
    warmstart_path = getattr(args, "warmstart_path", None)
    if warmstart_path is None:
        # Try default location
        default_path = output_dir / "nlsq_data.npz"
        if default_path.exists():
            warmstart_path = default_path
        else:
            return None

    try:
        from heterodyne.io.nlsq_writers import load_nlsq_npz_file

        result = load_nlsq_npz_file(Path(warmstart_path))
        logger.info("Loaded NLSQ warm-start from %s", warmstart_path)
        return result
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not load NLSQ warm-start from %s: %s", warmstart_path, exc
        )
        return None


def _get_warmstart_reduced_chi2(result: NLSQResult) -> float | None:
    """Extract reduced chi-squared from NLSQ result.

    Tries ``result.reduced_chi_squared`` first, then falls back to
    ``result.metadata["reduced_chi_squared"]``.

    Args:
        result: NLSQ result to inspect.

    Returns:
        Reduced chi-squared value, or ``None`` if unavailable.
    """
    chi2 = getattr(result, "reduced_chi_squared", None)
    if chi2 is not None:
        return float(chi2)
    return result.metadata.get("reduced_chi_squared") if result.metadata else None


def _validate_warmstart_quality(
    result: NLSQResult,
    chi2_threshold: float = 10.0,
) -> bool:
    """Check whether an NLSQ result is suitable for warm-starting CMC.

    Validates convergence success, reduced chi-squared, and (when the
    parameter registry is available) whether fitted values lie within
    their declared bounds.

    Args:
        result: NLSQ result to validate.
        chi2_threshold: Maximum acceptable reduced chi-squared.

    Returns:
        ``True`` if quality is acceptable, ``False`` otherwise.
    """
    ok = True

    # --- convergence flag ---
    if hasattr(result, "success") and not result.success:
        logger.warning(
            "Warm-start NLSQ did not converge: %s", getattr(result, "message", "")
        )
        ok = False

    # --- reduced chi-squared ---
    chi2 = _get_warmstart_reduced_chi2(result)
    if chi2 is not None:
        if chi2 >= chi2_threshold:
            logger.warning(
                "Warm-start reduced chi² = %.3f exceeds threshold %.1f",
                chi2,
                chi2_threshold,
            )
            ok = False
        else:
            logger.debug(
                "Warm-start reduced chi² = %.3f (threshold %.1f)", chi2, chi2_threshold
            )

    # --- parameter bounds check via registry ---
    try:
        from heterodyne.config.parameter_registry import ParameterRegistry

        registry = ParameterRegistry()
        params = result.params_dict
        for name, value in params.items():
            try:
                info = registry[name]
            except KeyError:
                continue
            if not (info.min_bound <= value <= info.max_bound):
                logger.warning(
                    "Warm-start param %s = %.4e outside bounds [%.4e, %.4e]",
                    name,
                    value,
                    info.min_bound,
                    info.max_bound,
                )
                ok = False
    except (ImportError, AttributeError, KeyError):
        # Registry unavailable — skip bounds check
        pass

    return ok


_WARMSTART_LOG_PARAMS = ("D0_ref", "D0_sample", "v0", "alpha_ref", "alpha_sample")


def _log_warmstart_physical_params(result: NLSQResult) -> None:
    """Log key physical parameter values from an NLSQ warm-start result.

    Logs at INFO level using scientific notation for easy inspection.
    Missing parameters are silently skipped.

    Args:
        result: NLSQ result whose parameters are logged.
    """
    params = result.params_dict
    parts: list[str] = []
    for name in _WARMSTART_LOG_PARAMS:
        if name in params:
            parts.append(f"{name}={params[name]:.2e}")
    if parts:
        logger.info("Warm-start params: %s", ", ".join(parts))
