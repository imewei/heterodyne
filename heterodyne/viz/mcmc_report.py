"""MCMC analysis report generation for heterodyne analysis.

Generates comprehensive Markdown reports summarizing NLSQ and CMC
results including parameter tables, convergence diagnostics, and
fit quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        include_diagnostics: Include convergence diagnostic tables.
        include_timing: Include timing information.
        include_correlation: Include parameter correlation analysis.
        ci_level: Credible interval level ("95" or "89").
        float_precision: Decimal places for floating-point values.
    """

    include_diagnostics: bool = True
    include_timing: bool = True
    include_correlation: bool = True
    ci_level: str = "95"
    float_precision: int = 4


def generate_report(
    nlsq_results: list[NLSQResult] | None = None,
    cmc_results: list[CMCResult] | None = None,
    output_dir: Path | str | None = None,
    config: ReportConfig | None = None,
) -> Path | str:
    """Generate a comprehensive Markdown analysis report.

    Creates a structured summary of the fitting results including:
    - Parameter estimates with uncertainties
    - Convergence diagnostics (R-hat, ESS, BFMI)
    - Fit quality metrics (chi-squared, cost)
    - Timing and configuration summary

    Args:
        nlsq_results: List of NLSQ results (one per phi angle).
        cmc_results: List of CMC results (one per phi angle).
        output_dir: Directory to write the report file. If None,
            returns the report as a string instead of writing.
        config: Report configuration. Uses defaults if None.

    Returns:
        Path to the written report file, or the report string if
        output_dir is None.
    """
    if config is None:
        config = ReportConfig()

    sections: list[str] = []

    # Header
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    sections.append(f"# Heterodyne Analysis Report\n\nGenerated: {timestamp}\n")

    # NLSQ Results
    if nlsq_results:
        sections.append("## NLSQ Optimization Results\n")
        for result in nlsq_results:
            phi = result.metadata.get("phi_angle", "N/A")
            sections.append(f"### Phi = {phi}\n")
            sections.append(_format_nlsq_table(result, config))

            if config.include_timing and result.wall_time_seconds is not None:
                sections.append(f"\nWall time: {result.wall_time_seconds:.2f} s\n")

    # CMC Results
    if cmc_results:
        sections.append("## CMC Bayesian Results\n")
        for cmc_result in cmc_results:
            phi = cmc_result.metadata.get("phi_angle", "N/A")
            sections.append(f"### Phi = {phi}\n")
            sections.append(_format_cmc_table(cmc_result, config))

            if config.include_diagnostics:
                sections.append(_format_diagnostics(cmc_result, config))

            if config.include_timing and cmc_result.wall_time_seconds is not None:
                sections.append(f"\nWall time: {cmc_result.wall_time_seconds:.1f} s\n")

    report_text = "\n".join(sections)

    if output_dir is None:
        return report_text

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Generated analysis report: %s", report_path)

    return report_path


def _format_nlsq_table(result: NLSQResult, config: ReportConfig) -> str:
    """Format NLSQ parameter table in Markdown."""
    prec = config.float_precision
    lines = [
        "| Parameter | Value | Uncertainty |",
        "|-----------|-------|-------------|",
    ]

    for i, name in enumerate(result.parameter_names):
        val = result.parameters[i]
        unc = result.uncertainties[i] if result.uncertainties is not None else None
        unc_str = f"{unc:.{prec}e}" if unc is not None else "N/A"
        lines.append(f"| {name} | {val:.{prec}e} | {unc_str} |")

    lines.append("")

    if result.reduced_chi_squared is not None:
        lines.append(f"Reduced chi-squared: {result.reduced_chi_squared:.{prec}f}")
    if result.final_cost is not None:
        lines.append(f"Final cost: {result.final_cost:.{prec}e}")
    lines.append(f"Success: {result.success}")
    lines.append(f"Function evaluations: {result.n_function_evals}")

    return "\n".join(lines) + "\n"


def _format_cmc_table(result: CMCResult, config: ReportConfig) -> str:
    """Format CMC posterior summary table in Markdown."""
    prec = config.float_precision
    ci_key_lo = f"lower_{config.ci_level}"
    ci_key_hi = f"upper_{config.ci_level}"

    lines = [
        f"| Parameter | Mean | Std | CI {config.ci_level}% |",
        "|-----------|------|-----|---------|",
    ]

    for i, name in enumerate(result.parameter_names):
        mean = float(result.posterior_mean[i])
        std = float(result.posterior_std[i])

        ci_str = "N/A"
        if name in result.credible_intervals:
            ci = result.credible_intervals[name]
            lo = ci.get(ci_key_lo)
            hi = ci.get(ci_key_hi)
            if lo is not None and hi is not None:
                ci_str = f"[{lo:.{prec}e}, {hi:.{prec}e}]"

        lines.append(f"| {name} | {mean:.{prec}e} | {std:.{prec}e} | {ci_str} |")

    lines.append("")
    lines.append(f"Convergence: {'PASSED' if result.convergence_passed else 'FAILED'}")
    lines.append(f"Chains: {result.num_chains} | Samples: {result.num_samples} | Warmup: {result.num_warmup}")

    return "\n".join(lines) + "\n"


def _format_diagnostics(result: CMCResult, config: ReportConfig) -> str:
    """Format convergence diagnostics table in Markdown."""
    lines = [
        "\n#### Convergence Diagnostics\n",
        "| Parameter | R-hat | ESS bulk | ESS tail |",
        "|-----------|-------|----------|----------|",
    ]

    for i, name in enumerate(result.parameter_names):
        r_hat = f"{result.r_hat[i]:.3f}" if result.r_hat is not None else "N/A"
        ess_b = f"{result.ess_bulk[i]:.0f}" if result.ess_bulk is not None else "N/A"
        ess_t = f"{result.ess_tail[i]:.0f}" if result.ess_tail is not None else "N/A"
        lines.append(f"| {name} | {r_hat} | {ess_b} | {ess_t} |")

    if result.bfmi is not None:
        lines.append(f"\nMin BFMI: {min(result.bfmi):.3f}")

    return "\n".join(lines) + "\n"
