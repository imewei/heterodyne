"""Convergence diagnostics for CMC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import arviz as az
import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)


@dataclass
class ConvergenceReport:
    """Report of convergence diagnostic checks."""

    passed: bool
    r_hat_passed: bool
    ess_passed: bool
    bfmi_passed: bool
    messages: list[str]


def validate_convergence(
    result: CMCResult,
    r_hat_threshold: float = 1.1,
    min_ess: int = 100,
    min_bfmi: float = 0.3,
) -> ConvergenceReport:
    """Validate MCMC convergence from CMC result.

    Checks:
    1. R-hat (Gelman-Rubin statistic) < threshold for all parameters
    2. Effective sample size (ESS) > minimum for all parameters
    3. Bayesian Fraction of Missing Information (BFMI) > minimum

    Args:
        result: CMC result with diagnostics
        r_hat_threshold: Maximum acceptable R-hat
        min_ess: Minimum acceptable ESS
        min_bfmi: Minimum acceptable BFMI

    Returns:
        ConvergenceReport
    """
    messages = []

    # Check R-hat
    r_hat_passed = True
    if result.r_hat is not None:
        max_rhat = np.max(result.r_hat)
        if max_rhat > r_hat_threshold:
            r_hat_passed = False
            bad_params = np.where(result.r_hat > r_hat_threshold)[0]
            for idx in bad_params:
                messages.append(
                    f"R-hat for {result.parameter_names[idx]}: "
                    f"{result.r_hat[idx]:.3f} > {r_hat_threshold}"
                )
        else:
            messages.append(f"R-hat: max={max_rhat:.3f} (PASS)")
    else:
        messages.append("R-hat: not computed")
        r_hat_passed = False

    # Check ESS
    ess_passed = True
    if result.ess_bulk is not None:
        min_ess_actual = np.min(result.ess_bulk)
        if min_ess_actual < min_ess:
            ess_passed = False
            low_ess = np.where(result.ess_bulk < min_ess)[0]
            for idx in low_ess:
                messages.append(
                    f"Low ESS for {result.parameter_names[idx]}: "
                    f"{result.ess_bulk[idx]:.0f} < {min_ess}"
                )
        else:
            messages.append(f"ESS: min={min_ess_actual:.0f} (PASS)")
    else:
        messages.append("ESS: not computed")
        ess_passed = False

    # Check BFMI
    bfmi_passed = True
    if result.bfmi is not None:
        min_bfmi_actual = min(result.bfmi)
        if min_bfmi_actual < min_bfmi:
            bfmi_passed = False
            messages.append(f"Low BFMI: {min_bfmi_actual:.3f} < {min_bfmi}")
        else:
            messages.append(f"BFMI: min={min_bfmi_actual:.3f} (PASS)")
    else:
        messages.append("BFMI: not computed")

    # Check posterior contraction (if prior_std available in metadata)
    metadata = getattr(result, "metadata", None)
    prior_std = metadata.get("prior_std") if metadata else None
    if prior_std and result.posterior_std is not None:
        pcr = compute_posterior_contraction(result, prior_std)
        for name, ratio in pcr.items():
            if ratio < 0:
                messages.append(
                    f"PCR for {name}: {ratio:.2f} (NEGATIVE — possible misspecification)"
                )
            elif ratio < 0.1:
                messages.append(
                    f"PCR for {name}: {ratio:.2f} (poorly identified)"
                )
            else:
                messages.append(f"PCR for {name}: {ratio:.2f}")

    passed = r_hat_passed and ess_passed and bfmi_passed

    return ConvergenceReport(
        passed=passed,
        r_hat_passed=r_hat_passed,
        ess_passed=ess_passed,
        bfmi_passed=bfmi_passed,
        messages=messages,
    )


def compute_posterior_contraction(
    result: CMCResult,
    prior_std: dict[str, float],
) -> dict[str, float]:
    """Compute Posterior Contraction Ratio for each parameter.

    PCR = 1 - posterior_std / prior_std

    Interpretation:
      ~1.0 = strongly constrained by data
      ~0.0 = poorly identified (prior dominates)
      <0   = possible model misspecification (posterior wider than prior)

    Args:
        result: CMC result with posterior_std.
        prior_std: Dict of prior standard deviations by parameter name.

    Returns:
        Dict mapping parameter name to PCR value.
    """
    pcr: dict[str, float] = {}
    for i, name in enumerate(result.parameter_names):
        if name in prior_std and prior_std[name] > 0:
            post_std = float(result.posterior_std[i])
            pcr[name] = 1.0 - post_std / prior_std[name]
    return pcr


def compute_r_hat(samples: np.ndarray) -> float:
    """Compute rank-normalized R-hat from samples.

    Delegates to ``arviz.rhat()`` which implements the recommended
    rank-normalized split-R-hat from Vehtari et al. (2021).

    Args:
        samples: Array of shape (n_chains, n_samples).
            Requires at least 2 chains; single-chain input returns NaN.

    Returns:
        R-hat value (1.0 indicates convergence; >1.01 suggests issues)
    """
    return float(az.rhat(samples))


def compute_ess(samples: np.ndarray) -> float:
    """Compute effective sample size.

    Delegates to ``arviz.ess()`` which uses FFT-based autocorrelation
    with Geyer's initial monotone sequence estimator.

    Args:
        samples: 1D array of samples, or 2D array of shape (n_chains, n_draws)

    Returns:
        Effective sample size (always >= 1.0)
    """
    # ArviZ 1.0 expects 2D (chains, draws); reshape 1D as single chain
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    return max(1.0, float(az.ess(samples)))


def compute_bfmi(energy: np.ndarray) -> float:
    """Compute Bayesian Fraction of Missing Information.

    Delegates to ``arviz.bfmi()``. Values < 0.3 indicate potential
    problems with HMC sampling.

    Args:
        energy: Array of HMC energies (1D or 2D)

    Returns:
        BFMI value (1.0 for constant energy)
    """
    if np.var(energy) == 0:
        return 1.0
    # ArviZ 1.0 expects 2D (chains, draws); reshape 1D as single chain
    if energy.ndim == 1:
        energy = energy[np.newaxis, :]
    result = az.bfmi(energy)
    return float(result[0])


# ---------------------------------------------------------------------------
# Divergence analysis
# ---------------------------------------------------------------------------


@dataclass
class DivergenceReport:
    """Divergence rate analysis."""

    divergence_rate: float  # fraction of divergent transitions
    n_divergent: int
    n_total: int
    severity: str  # "good" (<5%), "warning" (5-10%), "high" (10-20%), "critical" (>20%)
    messages: list[str]


def _classify_severity(rate: float) -> str:
    """Map a divergence rate to a severity label."""
    if rate < 0.05:
        return "good"
    if rate < 0.10:
        return "warning"
    if rate < 0.20:
        return "high"
    return "critical"


def analyze_divergences(
    samples: dict[str, np.ndarray] | CMCResult,
) -> DivergenceReport:
    """Analyse divergent transitions from MCMC samples.

    Accepts either a raw samples dict that may contain a ``"diverging"``
    field (shape ``(n_chains, n_draws)``) or a :class:`CMCResult` whose
    ``extra_fields`` attribute holds that field.

    Divergence rate is computed globally (all chains combined) and
    per-chain for contextual messages.

    Args:
        samples: Samples dict with optional ``"diverging"`` boolean array,
            or a CMCResult.

    Returns:
        DivergenceReport with severity classification and human-readable
        messages.
    """
    diverging: np.ndarray | None = None

    # Resolve input to a boolean diverging array
    if isinstance(samples, dict):
        raw = samples.get("diverging")
        if raw is not None:
            diverging = np.asarray(raw, dtype=bool)
    else:
        # Treat as CMCResult
        extra = getattr(samples, "extra_fields", None) or {}
        raw = extra.get("diverging")
        if raw is not None:
            diverging = np.asarray(raw, dtype=bool)

    messages: list[str] = []

    if diverging is None:
        messages.append("No diverging field found; cannot assess divergences.")
        return DivergenceReport(
            divergence_rate=0.0,
            n_divergent=0,
            n_total=0,
            severity="good",
            messages=messages,
        )

    # Ensure 2D: (n_chains, n_draws)
    if diverging.ndim == 1:
        diverging = diverging[np.newaxis, :]

    n_chains, n_draws = diverging.shape
    n_total = n_chains * n_draws
    n_divergent = int(np.sum(diverging))
    rate = n_divergent / n_total if n_total > 0 else 0.0
    severity = _classify_severity(rate)

    messages.append(
        f"Divergences: {n_divergent}/{n_total} "
        f"({rate * 100:.1f}%) — severity: {severity}"
    )

    # Per-chain breakdown
    for chain_idx in range(n_chains):
        chain_div = int(np.sum(diverging[chain_idx]))
        chain_rate = chain_div / n_draws if n_draws > 0 else 0.0
        chain_sev = _classify_severity(chain_rate)
        messages.append(
            f"  Chain {chain_idx}: {chain_div}/{n_draws} "
            f"({chain_rate * 100:.1f}%) — {chain_sev}"
        )

    if severity == "good":
        messages.append("Divergence rate acceptable (<5%).")
    elif severity == "warning":
        messages.append(
            "Divergence rate elevated (5-10%). "
            "Consider increasing target_accept_prob or reparameterizing."
        )
    elif severity == "high":
        messages.append(
            "High divergence rate (10-20%). "
            "Non-centered parameterization and/or tighter priors recommended."
        )
    else:
        messages.append(
            "Critical divergence rate (>20%). "
            "Posterior geometry is poorly suited to HMC. "
            "Reparameterize or reduce model complexity."
        )

    logger.info(
        "Divergence analysis: %d/%d (%.1f%%) — %s",
        n_divergent, n_total, rate * 100, severity,
    )

    return DivergenceReport(
        divergence_rate=rate,
        n_divergent=n_divergent,
        n_total=n_total,
        severity=severity,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Sharded convergence validation
# ---------------------------------------------------------------------------


def validate_convergence_sharded(
    results: list[CMCResult],
    r_hat_threshold: float = 1.1,
    min_ess: int = 100,
    min_bfmi: float = 0.3,
) -> ConvergenceReport:
    """Validate convergence across CMC shards.

    Runs :func:`validate_convergence` on each shard and returns a
    combined :class:`ConvergenceReport` that reflects the worst-case
    R-hat and the minimum ESS observed across all shards.  A single
    failing shard causes the combined report to fail.

    Args:
        results: One :class:`CMCResult` per shard.
        r_hat_threshold: Forwarded to :func:`validate_convergence`.
        min_ess: Forwarded to :func:`validate_convergence`.
        min_bfmi: Forwarded to :func:`validate_convergence`.

    Returns:
        Combined ConvergenceReport with worst-case statistics.
    """
    if not results:
        return ConvergenceReport(
            passed=False,
            r_hat_passed=False,
            ess_passed=False,
            bfmi_passed=False,
            messages=["No shard results provided."],
        )

    all_messages: list[str] = []
    r_hat_passed_all = True
    ess_passed_all = True
    bfmi_passed_all = True

    worst_r_hat: float | None = None
    min_ess_seen: float | None = None
    min_bfmi_seen: float | None = None

    for shard_idx, result in enumerate(results):
        report = validate_convergence(
            result,
            r_hat_threshold=r_hat_threshold,
            min_ess=min_ess,
            min_bfmi=min_bfmi,
        )
        prefix = f"[Shard {shard_idx}] "
        all_messages.extend(prefix + msg for msg in report.messages)

        if not report.r_hat_passed:
            r_hat_passed_all = False
        if not report.ess_passed:
            ess_passed_all = False
        if not report.bfmi_passed:
            bfmi_passed_all = False

        # Track worst-case scalars
        if result.r_hat is not None:
            shard_max_rhat = float(np.max(result.r_hat))
            if worst_r_hat is None or shard_max_rhat > worst_r_hat:
                worst_r_hat = shard_max_rhat

        if result.ess_bulk is not None:
            shard_min_ess = float(np.min(result.ess_bulk))
            if min_ess_seen is None or shard_min_ess < min_ess_seen:
                min_ess_seen = shard_min_ess

        if result.bfmi is not None:
            shard_min_bfmi = float(min(result.bfmi))
            if min_bfmi_seen is None or shard_min_bfmi < min_bfmi_seen:
                min_bfmi_seen = shard_min_bfmi

    # Summary line
    summary_parts: list[str] = [f"Shards: {len(results)}"]
    if worst_r_hat is not None:
        summary_parts.append(f"worst R-hat={worst_r_hat:.3f}")
    if min_ess_seen is not None:
        summary_parts.append(f"min ESS={min_ess_seen:.0f}")
    if min_bfmi_seen is not None:
        summary_parts.append(f"min BFMI={min_bfmi_seen:.3f}")
    all_messages.insert(0, "Sharded convergence summary: " + ", ".join(summary_parts))

    passed = r_hat_passed_all and ess_passed_all and bfmi_passed_all

    logger.info(
        "Sharded convergence: %d shards, passed=%s, "
        "worst_r_hat=%s, min_ess=%s, min_bfmi=%s",
        len(results), passed, worst_r_hat, min_ess_seen, min_bfmi_seen,
    )

    return ConvergenceReport(
        passed=passed,
        r_hat_passed=r_hat_passed_all,
        ess_passed=ess_passed_all,
        bfmi_passed=bfmi_passed_all,
        messages=all_messages,
    )


# ---------------------------------------------------------------------------
# Trace diagnostics
# ---------------------------------------------------------------------------


def compute_trace_diagnostics(
    samples: np.ndarray,
    lags: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    """Compute trace-level diagnostics for a single parameter's samples.

    Args:
        samples: Array of shape ``(n_chains, n_draws)`` or ``(n_draws,)``.
            When 1-D, treated as a single chain.
        lags: Autocorrelation lags to evaluate.  Defaults to (1, 5, 10).

    Returns:
        Dictionary with the following keys:

        ``autocorr``
            Dict mapping each lag to its mean autocorrelation across chains.
        ``stationary``
            ``True`` if the absolute autocorrelation at lag 1 is below 0.5
            for all chains (heuristic stationarity flag).
        ``mixing_quality``
            One of ``"good"``, ``"moderate"``, or ``"poor"`` based on
            lag-1 autocorrelation magnitude.
        ``n_chains``
            Number of chains.
        ``n_draws``
            Number of draws per chain.
        ``mean``
            Grand mean across all draws.
        ``std``
            Grand standard deviation across all draws.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    n_chains, n_draws = arr.shape

    # Autocorrelation per chain at requested lags
    autocorr_by_lag: dict[int, float] = {}
    lag1_values: list[float] = []

    for lag in lags:
        chain_acfs: list[float] = []
        for c in range(n_chains):
            chain = arr[c]
            if n_draws <= lag:
                chain_acfs.append(float("nan"))
                continue
            # Unbiased autocorrelation at given lag
            centered = chain - chain.mean()
            var = float(np.dot(centered, centered))
            if var == 0.0:
                chain_acfs.append(float("nan"))
                continue
            acf = float(np.dot(centered[:-lag], centered[lag:])) / var
            chain_acfs.append(acf)
            if lag == 1:
                lag1_values.append(acf)
        valid = [v for v in chain_acfs if not np.isnan(v)]
        autocorr_by_lag[lag] = float(np.mean(valid)) if valid else float("nan")

    # Stationarity: lag-1 autocorrelation < 0.5 for all chains
    stationary = all(abs(v) < 0.5 for v in lag1_values) if lag1_values else False

    # Mixing quality from mean lag-1 autocorrelation
    mean_lag1 = autocorr_by_lag.get(1, float("nan"))
    if np.isnan(mean_lag1):
        mixing_quality = "unknown"
    elif abs(mean_lag1) < 0.1:
        mixing_quality = "good"
    elif abs(mean_lag1) < 0.5:
        mixing_quality = "moderate"
    else:
        mixing_quality = "poor"

    flat = arr.ravel()
    return {
        "autocorr": autocorr_by_lag,
        "stationary": stationary,
        "mixing_quality": mixing_quality,
        "n_chains": n_chains,
        "n_draws": n_draws,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


# ---------------------------------------------------------------------------
# Parameter pair correlations
# ---------------------------------------------------------------------------


def compute_pair_correlations(
    samples: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute pairwise Pearson correlations between parameters.

    Flattens all chains and draws for each parameter before computing
    correlations, so the result is chain-agnostic.

    Useful for detecting parameter degeneracy: correlations with
    ``|r| > 0.9`` indicate near-redundant parameters.

    Args:
        samples: Mapping from parameter name to array of shape
            ``(n_chains, n_draws)`` or ``(n_draws,)``.

    Returns:
        Nested dict ``corr[param_a][param_b] = r`` where ``r`` is the
        Pearson correlation coefficient in ``[-1, 1]``.  The matrix is
        symmetric with ones on the diagonal.  Returns an empty dict if
        fewer than two parameters are provided.
    """
    names = list(samples.keys())
    if len(names) < 2:
        logger.debug(
            "compute_pair_correlations: fewer than 2 parameters; returning empty dict."
        )
        return {}

    # Flatten each parameter to 1-D
    flat: dict[str, np.ndarray] = {}
    for name, arr in samples.items():
        a = np.asarray(arr, dtype=float)
        flat[name] = a.ravel()

    # All flat arrays must have the same length; truncate to the minimum
    min_len = min(v.size for v in flat.values())
    flat = {k: v[:min_len] for k, v in flat.items()}

    # Build correlation matrix via numpy
    matrix = np.vstack([flat[n] for n in names])  # (n_params, n_samples)
    corr_matrix = np.corrcoef(matrix)  # (n_params, n_params)

    result: dict[str, dict[str, float]] = {}
    for i, name_i in enumerate(names):
        result[name_i] = {}
        for j, name_j in enumerate(names):
            r = float(corr_matrix[i, j])
            result[name_i][name_j] = r
            if i < j and abs(r) > 0.9:
                logger.warning(
                    "High parameter correlation: %s <-> %s (r=%.3f). "
                    "Possible degeneracy.",
                    name_i, name_j, r,
                )

    logger.info(
        "Computed %dx%d parameter correlation matrix.",
        len(names), len(names),
    )
    return result
