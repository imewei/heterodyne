"""Optimization execution for heterodyne CLI.

Manages NLSQ and CMC fitting runs, including warm-start resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.io.mcmc_writers import format_mcmc_summary, save_mcmc_results
from heterodyne.io.nlsq_writers import (
    format_nlsq_summary,
    save_nlsq_json_files,
    save_nlsq_npz_file,
)
from heterodyne.optimization.cmc import CMCConfig
from heterodyne.optimization.cmc.core import (
    CMC_ALPHA_SINGULARITY as _ALPHA_SINGULARITY,
)
from heterodyne.optimization.cmc.core import (
    CMC_F0_DEGEN_THRESHOLD as _F0_DEGEN_THRESHOLD,
)
from heterodyne.optimization.cmc.core import fit_cmc_multi_phi
from heterodyne.optimization.nlsq import NLSQConfig, fit_nlsq_multi_phi
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import AnalysisSummaryLogger, get_logger, log_phase

if TYPE_CHECKING:
    from heterodyne.config.manager import ConfigManager
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)


def _closest_phi_index(data_phi_angles: np.ndarray, target: float) -> int:
    """Return the index of the data phi angle closest to *target* (degrees).

    Uses circular distance so that 179° and -179° are treated as 2° apart
    instead of 358° apart. A linear ``argmin(abs(d - t))`` would pick the
    wrong slice near the ±180° boundary.
    """
    normalized_data = (
        (np.asarray(data_phi_angles, dtype=float) + 180.0) % 360.0
    ) - 180.0
    normalized_target = ((float(target) + 180.0) % 360.0) - 180.0
    delta = ((normalized_data - normalized_target + 180.0) % 360.0) - 180.0
    return int(np.argmin(np.abs(delta)))


def _select_c2_for_phi_angles(
    c2_data: np.ndarray,
    phi_angles: list[float],
    data_phi_angles: np.ndarray | None = None,
) -> np.ndarray:
    """Return a C2 stack aligned with selected phi angles."""
    if c2_data.ndim != 3:
        return c2_data

    slices: list[np.ndarray] = []
    for i, phi in enumerate(phi_angles):
        if data_phi_angles is not None and len(data_phi_angles) == c2_data.shape[0]:
            idx = _closest_phi_index(data_phi_angles, phi)
            logger.info(
                "Selected data slice %d (phi=%.2f°) for fitting phi=%.2f°",
                idx,
                float(data_phi_angles[idx]),
                phi,
            )
            slices.append(c2_data[idx])
        else:
            slices.append(c2_data[i])

    return np.stack(slices, axis=0)


def _combine_nlsq_results(results: list[NLSQResult]) -> NLSQResult:
    """Build a single aggregate result for disk output."""
    if not results:
        raise ValueError("Cannot combine empty NLSQ result list")

    first = results[0]

    def _stack_optional(attr: str) -> np.ndarray | None:
        values = [getattr(result, attr) for result in results]
        if any(value is None for value in values):
            return None
        return np.stack([np.asarray(value) for value in values], axis=0)

    residual_values = [result.residuals for result in results]
    residuals = (
        np.concatenate([np.asarray(value).ravel() for value in residual_values])
        if all(value is not None for value in residual_values)
        else None
    )
    costs = [
        float(result.final_cost) for result in results if result.final_cost is not None
    ]
    final_cost = (
        float(0.5 * np.sum(residuals**2))
        if residuals is not None
        else (float(np.sum(costs)) if costs else None)
    )
    chi2_values = [
        float(result.reduced_chi_squared)
        for result in results
        if result.reduced_chi_squared is not None
    ]

    metadata = {
        "aggregate": True,
        "n_angles": len(results),
        "phi_angles": [result.metadata.get("phi_angle") for result in results],
        "per_angle": [
            {
                "phi_angle": result.metadata.get("phi_angle"),
                "success": result.success,
                "message": result.message,
                "final_cost": result.final_cost,
                "reduced_chi_squared": result.reduced_chi_squared,
            }
            for result in results
        ],
    }

    return NLSQResult(
        parameters=np.asarray(first.parameters),
        parameter_names=list(first.parameter_names),
        success=all(result.success for result in results),
        message="multi-angle NLSQ complete",
        uncertainties=first.uncertainties,
        covariance=first.covariance,
        final_cost=final_cost,
        reduced_chi_squared=float(np.mean(chi2_values)) if chi2_values else None,
        n_iterations=max((result.n_iterations for result in results), default=0),
        n_function_evals=sum(result.n_function_evals for result in results),
        convergence_reason=first.convergence_reason,
        residuals=residuals,
        jacobian=None,
        fitted_correlation=_stack_optional("fitted_correlation"),
        wall_time_seconds=first.metadata.get("wall_time_total")
        or first.wall_time_seconds,
        metadata=metadata,
    )


def run_nlsq(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float],
    config_manager: ConfigManager,
    args: argparse.Namespace,
    output_dir: Path,
    summary: AnalysisSummaryLogger | None = None,
    data_phi_angles: np.ndarray | None = None,
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

    nlsq_config = NLSQConfig.from_dict(config_manager.nlsq_config)
    nlsq_config.verbose = getattr(args, "verbose", 1)

    c2_fit = _select_c2_for_phi_angles(c2_data, phi_angles, data_phi_angles)

    with log_phase("nlsq_multi_phi", logger=logger, track_memory=True) as phase:
        results = fit_nlsq_multi_phi(
            model=model,
            c2_data=c2_fit,
            phi_angles=phi_angles,
            config=nlsq_config,
        )

    logger.info(
        "NLSQ multi-angle optimization completed in %.2fs for %d phi angles",
        phase.duration,
        len(phi_angles),
    )

    for i, (phi, result) in enumerate(zip(phi_angles, results, strict=True)):
        result.metadata["phi_angle"] = phi
        _warn_nlsq_bound_saturation(result)

        if summary and result.reduced_chi_squared is not None:
            summary.record_metric(
                f"nlsq_chi2_phi{int(phi)}", result.reduced_chi_squared
            )

        # Post-fit RecoveryPlan diagnosis for failed angles (homodyne parity).
        if not result.success:
            try:
                from heterodyne.optimization.nlsq.recovery import diagnose_failure

                plan = diagnose_failure(result, nlsq_config)
                logger.warning(
                    "NLSQ phi=%s° failed → recovery plan: %s (%s)",
                    phi,
                    plan.action.value,
                    plan.message,
                )
                result.metadata["recovery_plan"] = {
                    "action": plan.action.value,
                    "message": plan.message,
                }
            except (ValueError, AttributeError, RuntimeError) as exc:
                logger.warning(
                    "diagnose_failure crashed on phi=%s° (%s); recovery plan unavailable",
                    phi,
                    exc,
                )

        summary_lines = format_nlsq_summary(result)
        logger.info(
            "NLSQ Results for phi=%s° (%d/%d)\n%s\n%s",
            phi,
            i + 1,
            len(results),
            "=" * 50,
            summary_lines,
        )

    aggregate = _combine_nlsq_results(results)

    # Per-phi batch statistics (homodyne parity)
    if len(results) >= 2:
        try:
            from heterodyne.optimization.batch_statistics import (
                compute_batch_statistics,
                format_batch_report,
            )

            batch = compute_batch_statistics(results)
            logger.info("NLSQ batch statistics:\n%s", format_batch_report(batch))
            if summary is not None:
                summary.record_metric(
                    "nlsq_batch_success_rate", batch.overall_success_rate
                )
                summary.record_metric("nlsq_batch_mean_chi2", batch.mean_chi2)
        except (ValueError, AttributeError) as exc:
            logger.warning("Batch statistics unavailable (%s); continuing", exc)

    output_format = getattr(args, "output_format", "both")
    if output_format in ("json", "both"):
        saved_json = save_nlsq_json_files(aggregate, output_dir, prefix="nlsq")
        for label, path in saved_json.items():
            logger.info("Saved NLSQ %s: %s", label, path)
    if output_format in ("npz", "both"):
        npz_path = output_dir / "nlsq_data.npz"
        save_nlsq_npz_file(aggregate, npz_path, c2_exp=c2_fit)
        logger.info("Saved NLSQ data: %s", npz_path)

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
    data_phi_angles: np.ndarray | None = None,
) -> CMCResult:
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
        Joint multi-phi :class:`CMCResult` (homodyne parity). Reflects ONE
        NUTS inference across all phi angles with shared 14 physics params
        and per-angle scaling in ``mean_contrast`` / ``std_contrast`` /
        ``mean_offset`` / ``std_offset`` arrays of length ``n_phi``.
    """
    logger.info("Starting CMC analysis (joint multi-phi, homodyne parity)")

    if getattr(args, "num_samples", None) is not None:
        logger.info("Overriding CMC num_samples from CLI: %s", args.num_samples)
        config_manager.update_optimization_config(
            "cmc", "num_samples", args.num_samples
        )
    if getattr(args, "num_chains", None) is not None:
        logger.info("Overriding CMC num_chains from CLI: %s", args.num_chains)
        config_manager.update_optimization_config("cmc", "num_chains", args.num_chains)

    cmc_config = CMCConfig.from_dict(config_manager.cmc_config)
    if cmc_config.backend_name == "jit":
        cmc_config.backend_name = "pjit"

    # ---- Stack per-angle c2 slices into (n_phi, N, N) for joint inference ----
    c2_stack_list: list[np.ndarray] = []
    nlsq_stack: list[NLSQResult] = []
    has_nlsq = bool(nlsq_results)
    fixed_overrides: dict[str, float] | None = None
    if has_nlsq:
        _varying_set = set(model.varying_names)
        fixed_overrides = {
            name: float(val)
            for name, val in model.get_params_dict().items()
            if name not in _varying_set
        } or None

    for i, phi in enumerate(phi_angles):
        if c2_data.ndim == 3:
            if data_phi_angles is not None and len(data_phi_angles) == c2_data.shape[0]:
                idx = _closest_phi_index(data_phi_angles, phi)
                c2_phi = c2_data[idx]
            else:
                c2_phi = c2_data[i]
        else:
            c2_phi = c2_data
        c2_stack_list.append(np.asarray(c2_phi))

        if has_nlsq and nlsq_results is not None and i < len(nlsq_results):
            nr = nlsq_results[i]
            if _validate_warmstart_quality(nr):
                _log_warmstart_physical_params(nr)
            else:
                logger.warning(
                    "Warm-start quality below threshold for phi=%s°; using anyway", phi
                )
            nr = _clamp_warmstart_to_interior(nr, fixed_param_overrides=fixed_overrides)
            _warn_degenerate_sample_regime(nr)
            nlsq_stack.append(nr)

    c2_stacked = np.stack(c2_stack_list, axis=0)
    nlsq_for_engine: list[NLSQResult] | None = nlsq_stack if has_nlsq else None

    n_points = int(c2_stacked.size)
    logger.info(
        "[CMC joint] n_phi=%d, total_points=%d, num_warmup=%d, num_samples=%d, "
        "num_chains=%d",
        len(phi_angles),
        n_points,
        cmc_config.num_warmup,
        cmc_config.num_samples,
        cmc_config.num_chains,
    )

    with log_phase("cmc_joint_multi_phi", logger=logger, track_memory=True) as phase:
        result = fit_cmc_multi_phi(
            model=model,
            c2_data=c2_stacked,
            phi_angles=list(phi_angles),
            config=cmc_config,
            nlsq_results=nlsq_for_engine,
        )

    result.metadata["phi_angles"] = [float(p) for p in phi_angles]
    result.metadata["joint_multi_phi_runtime_s"] = phase.duration

    logger.info(
        "CMC joint multi-phi completed in %.2fs (n_phi=%d)",
        phase.duration,
        len(phi_angles),
    )

    if summary is not None:
        summary.record_metric("cmc_n_samples", float(cmc_config.num_samples))
        summary.record_metric("cmc_n_phi", float(len(phi_angles)))

    logger.info(
        "\n%s\nCMC Results (joint multi-phi)\n%s",
        "=" * 50,
        format_mcmc_summary(result),
    )

    output_format = getattr(args, "output_format", "both")
    prefix = "cmc"
    saved_paths = save_mcmc_results(result, output_dir, prefix=prefix)
    if output_format == "json":
        samples_path = saved_paths.get("samples")
        if samples_path is not None:
            samples_path.unlink(missing_ok=True)
    elif output_format == "npz":
        for key in ("summary", "diagnostics"):
            json_path = saved_paths.get(key)
            if json_path is not None:
                json_path.unlink(missing_ok=True)
    logger.info("Saved CMC results → %s (prefix=%s)", output_dir, prefix)

    if _is_degenerate_cmc_result(result):
        logger.error(
            "[CMC] Joint multi-phi result is degenerate (no usable samples). "
            "The warm-start parameters may be degenerate or the model is "
            "unidentifiable for this dataset. Fixes: freeze degenerate params "
            "in YAML (optimization.cmc.fixed_params), tighten NLSQ bounds, or "
            "use allow_degenerate_warmstart: true to override."
        )

    logger.info("CMC analysis complete")
    return result


def _is_degenerate_cmc_result(result: CMCResult) -> bool:
    """Return True when a CMC result is the all-shards-failed sentinel.

    ``_combine_shard_posteriors`` returns a CMCResult with no samples and
    ``convergence_passed=False`` when zero shards survived the
    R-hat/ESS/no-samples gates. Detecting this lets the per-angle loop bail
    out instead of running the next angle with the same broken warm-start.
    """
    if getattr(result, "convergence_passed", True):
        return False
    samples = getattr(result, "samples", None)
    if samples is None:
        return True
    # samples present but empty is also degenerate
    try:
        return all(np.asarray(s).size == 0 for s in samples.values())
    except Exception:
        return False


def resolve_nlsq_warmstart(
    args: argparse.Namespace,
    output_dir: Path,
) -> NLSQResult | None:
    """Attempt to load previously saved NLSQ results for warm-starting CMC.

    Args:
        args: CLI arguments (``--nlsq-result PATH`` stored as
            ``args.nlsq_result``; legacy ``args.warmstart_path`` accepted).
        output_dir: Default directory to search for NLSQ results.

    Returns:
        NLSQResult if found, None otherwise.
    """
    # ``--nlsq-result`` is the documented user-facing flag (args_parser.py).
    # ``warmstart_path`` is a legacy attribute name kept for programmatic
    # callers; honour both so the CLI flag is not silently ignored.
    warmstart_path = getattr(args, "nlsq_result", None) or getattr(
        args, "warmstart_path", None
    )
    if warmstart_path is None:
        # Try default location
        default_path = output_dir / "nlsq_data.npz"
        logger.debug(
            "No warmstart path specified; checking default location %s", default_path
        )
        if default_path.exists():
            warmstart_path = default_path
        else:
            logger.debug(
                "No NLSQ warm-start available; CMC will use config initial values"
            )
            return None

    try:
        from heterodyne.io.nlsq_writers import load_nlsq_npz_file

        warmstart_file = Path(warmstart_path)
        if warmstart_file.is_dir():
            warmstart_file = warmstart_file / "nlsq_data.npz"
        result = load_nlsq_npz_file(warmstart_file)
        logger.info("Loaded NLSQ warm-start from %s", warmstart_file)
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
    elif hasattr(result, "success"):
        logger.debug("Warm-start convergence: OK")

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

    if ok:
        chi2_str = f"{chi2:.3f}" if chi2 is not None else "N/A"
        logger.info(
            "NLSQ warm-start accepted (reduced chi²=%s). Using as CMC initial values.",
            chi2_str,
        )

    return ok


def _warn_nlsq_bound_saturation(result: NLSQResult) -> None:
    """Log a WARNING for each parameter whose uncertainty is zero or near-zero.

    Zero uncertainty in the NLSQ covariance has two causes:
    1. Parameter hit a bound → Jacobian column is zeroed by the box constraint.
    2. Fraction model clipped to [0,1] everywhere → Jacobian w.r.t. f1/f2 = 0.

    Both indicate a degenerate solution that will produce a pathological CMC
    posterior (NUTS step-size collapses, chains freeze at initialization).
    """
    if result.uncertainties is None or result.parameter_names is None:
        return
    try:
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        registry: Any = DEFAULT_REGISTRY
    except ImportError:
        registry = None

    params = result.params_dict
    saturated: list[str] = []
    for name, unc in zip(result.parameter_names, result.uncertainties, strict=True):
        if float(unc) < 1e-30:
            val = params.get(name, float("nan"))
            hint = ""
            if registry is not None:
                try:
                    info = registry[name]
                    if abs(val - info.min_bound) < 1e-10 * max(abs(info.min_bound), 1):
                        hint = " [AT LOWER BOUND]"
                    elif abs(val - info.max_bound) < 1e-10 * max(
                        abs(info.max_bound), 1
                    ):
                        hint = " [AT UPPER BOUND]"
                    else:
                        hint = " [DEGENERATE JACOBIAN — check clipping]"
                except KeyError:
                    pass
            logger.warning(
                "NLSQ bound saturation: %s = %.4g ± 0%s — "
                "posterior will be unreliable; CMC chains may freeze",
                name,
                val,
                hint,
            )
            saturated.append(name)

    if saturated:
        logger.warning(
            "%d parameter(s) saturated at bounds or degenerate: %s. "
            "Consider tightening bounds, adjusting initial values, or fixing "
            "these parameters before running CMC.",
            len(saturated),
            saturated,
        )


_BOUNDARY_INTERIOR_MARGIN = 5e-2  # fraction of bound range to keep away from walls


def _clamp_warmstart_to_interior(
    result: NLSQResult,
    fixed_param_overrides: dict[str, float] | None = None,
) -> NLSQResult:
    """CLI wrapper for ``heterodyne.optimization.cmc.priors.clamp_to_interior``.

    P2-a / Phase 4 PR 4 Task 4.4: the implementation was absorbed from
    :mod:`heterodyne.optimization.cmc.warmstart` into
    :mod:`heterodyne.optimization.cmc.priors` (spec §7 Rule 1).
    This wrapper exists only to preserve the private CLI-internal symbol
    used by the rest of this module.
    """
    from heterodyne.optimization.cmc.priors import clamp_to_interior

    return clamp_to_interior(result, fixed_param_overrides)


_WARMSTART_LOG_PARAMS = ("D0_ref", "D0_sample", "v0", "alpha_ref", "alpha_sample")

# Import thresholds from the authoritative source so the two warning systems
# stay in sync when a threshold is tuned.


def _warn_degenerate_sample_regime(result: NLSQResult) -> None:
    """Warn when the NLSQ solution is in a degenerate sample-transport regime.

    Two conditions — individually or combined — cause 100% CMC shard
    bad_convergence and BFMI=0.000 (the het_bb97531f failure mode):

    1. f0 < ``_F0_DEGEN_THRESHOLD``: sample fraction near zero makes
       alpha_sample, D0_sample, D_offset_sample unidentifiable.  Per-shard
       posteriors are dominated by the tempered prior; NUTS must thermalize
       from the warm-start across the full prior range with near-zero
       likelihood gradient for the sample-transport group.

    2. alpha_sample < ``_ALPHA_SINGULARITY``: J_sample(t) ∝ t^α has a
       non-integrable singularity at t→0.  NUTS step-size adaptation
       collapses for the sample-transport group.

    Calling this before CMC dispatch surfaces the problem immediately,
    giving the user time to act (freeze parameters, increase warmup) before
    investing hours of compute.
    """
    params = result.params_dict
    f0 = params.get("f0")
    alpha_s = params.get("alpha_sample")
    f0_degen = f0 is not None and float(f0) < _F0_DEGEN_THRESHOLD
    alpha_sing = alpha_s is not None and float(alpha_s) < _ALPHA_SINGULARITY
    if not f0_degen and not alpha_sing:
        return

    parts: list[str] = []
    if f0_degen:
        parts.append(
            f"f0={float(f0):.4f} < {_F0_DEGEN_THRESHOLD} — sample fraction "  # type: ignore[arg-type]
            "near-zero; alpha_sample / D0_sample / D_offset_sample are "
            "unidentifiable from data"
        )
    if alpha_sing:
        parts.append(
            f"alpha_sample={float(alpha_s):.3f} < {_ALPHA_SINGULARITY} — "  # type: ignore[arg-type]
            "J_sample ∝ t^α singularity at short lags; NUTS step-size collapses"
        )
    logger.warning(
        "Degenerate NLSQ warm-start (het_bb97531f failure mode) — CMC likely "
        "to fail for ALL shards:\n  %s\n"
        "Fixes:\n"
        "  • Freeze degenerate params in YAML → optimization.cmc.fixed_params: "
        "{alpha_sample: %.3f, D0_sample: %.3g}\n"
        "  • Or increase num_warmup to ≥2000\n"
        "  • If f0 < 0.05, consider fixing f0=0 (reference-only model)",
        ";\n  ".join(parts),
        float(alpha_s) if alpha_s is not None else float("nan"),
        float(params.get("D0_sample", float("nan"))),
    )


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
