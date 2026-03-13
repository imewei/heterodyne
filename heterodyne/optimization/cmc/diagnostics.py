"""Convergence diagnostics for CMC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
                messages.append(f"PCR for {name}: {ratio:.2f} (poorly identified)")
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
        n_divergent,
        n_total,
        rate * 100,
        severity,
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
        len(results),
        passed,
        worst_r_hat,
        min_ess_seen,
        min_bfmi_seen,
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
                    name_i,
                    name_j,
                    r,
                )

    logger.info(
        "Computed %dx%d parameter correlation matrix.",
        len(names),
        len(names),
    )
    return result


# ---------------------------------------------------------------------------
# Bimodality detection
# ---------------------------------------------------------------------------


@dataclass
class BimodalResult:
    """Result of a bimodality test for a single parameter's samples.

    Attributes:
        param_name: Name of the parameter tested.
        is_bimodal: True if the 2-component GMM is favoured by BIC.
        bic_unimodal: BIC of the 1-component (unimodal) Gaussian mixture.
        bic_bimodal: BIC of the 2-component Gaussian mixture.
        delta_bic: ``bic_unimodal - bic_bimodal``.  Positive values
            favour the bimodal model.
        means: Tuple of the two component means (None when not bimodal).
        weights: Tuple of the two component mixing weights (None when
            not bimodal).
    """

    param_name: str
    is_bimodal: bool
    bic_unimodal: float
    bic_bimodal: float
    delta_bic: float
    means: tuple[float, float] | None
    weights: tuple[float, float] | None


def detect_bimodal(
    samples: np.ndarray,
    param_name: str,
    bic_threshold: float = 10.0,
) -> BimodalResult:
    """Fit 1- and 2-component Gaussian mixtures and compare BIC.

    Uses scikit-learn's ``GaussianMixture`` to estimate Bayesian
    Information Criterion for unimodal vs bimodal models.  A positive
    ``delta_bic`` larger than ``bic_threshold`` is treated as evidence
    for bimodality.

    Args:
        samples: 1-D array of posterior draws for the parameter.
        param_name: Name used for logging and result labelling.
        bic_threshold: Minimum ``delta_bic`` (BIC_unimodal − BIC_bimodal)
            required to declare bimodality.  Default 10.0 corresponds to
            strong evidence on the Raftery (1995) BIC scale.

    Returns:
        :class:`BimodalResult` with fitted statistics.

    Raises:
        ImportError: If scikit-learn is not installed, with a hint on
            how to add the optional dependency.
    """
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for bimodality detection. "
            "Install it with: uv add scikit-learn"
        ) from exc

    arr = np.asarray(samples, dtype=float).ravel()
    x = arr.reshape(-1, 1)

    # Fit 1-component model
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    bic1 = float(gm1.bic(x))

    # Fit 2-component model
    gm2 = GaussianMixture(n_components=2, n_init=3, random_state=0).fit(x)
    bic2 = float(gm2.bic(x))

    delta = bic1 - bic2
    is_bimodal = delta > bic_threshold

    means: tuple[float, float] | None = None
    weights: tuple[float, float] | None = None

    if is_bimodal:
        m = gm2.means_.ravel()
        w = gm2.weights_.ravel()
        means = (float(m[0]), float(m[1]))
        weights = (float(w[0]), float(w[1]))
        logger.warning(
            "Bimodal posterior detected for %s: delta_BIC=%.2f, "
            "means=(%.4e, %.4e), weights=(%.3f, %.3f)",
            param_name,
            delta,
            means[0],
            means[1],
            weights[0],
            weights[1],
        )
    else:
        logger.debug(
            "Unimodal posterior for %s: delta_BIC=%.2f (threshold=%.1f)",
            param_name,
            delta,
            bic_threshold,
        )

    return BimodalResult(
        param_name=param_name,
        is_bimodal=is_bimodal,
        bic_unimodal=bic1,
        bic_bimodal=bic2,
        delta_bic=delta,
        means=means,
        weights=weights,
    )


def check_shard_bimodality(
    shard_samples: dict[int, dict[str, np.ndarray]],
    bic_threshold: float = 10.0,
) -> dict[str, list[BimodalResult]]:
    """Detect bimodality for each parameter across all CMC shards.

    Runs :func:`detect_bimodal` for every (parameter, shard) combination
    and aggregates results per parameter.

    Args:
        shard_samples: Mapping of shard index to a dict of
            ``{param_name: samples_array}``.  Samples may be 1-D
            ``(n_draws,)`` or 2-D ``(n_chains, n_draws)``; they are
            flattened before testing.
        bic_threshold: Forwarded to :func:`detect_bimodal`.

    Returns:
        Mapping from parameter name to a list of
        :class:`BimodalResult`, one entry per shard (in shard-index
        order).
    """
    results: dict[str, list[BimodalResult]] = {}

    # Collect all parameter names in stable insertion order across shards
    seen: dict[str, None] = {}
    for shard_dict in shard_samples.values():
        for name in shard_dict:
            seen.setdefault(name, None)
    all_param_names = list(seen)

    for name in all_param_names:
        param_results: list[BimodalResult] = []
        for shard_idx in sorted(shard_samples.keys()):
            shard_dict = shard_samples[shard_idx]
            if name not in shard_dict:
                continue
            arr = np.asarray(shard_dict[name], dtype=float).ravel()
            result = detect_bimodal(arr, param_name=name, bic_threshold=bic_threshold)
            param_results.append(result)
        results[name] = param_results

    n_bimodal = sum(
        1 for param_list in results.values() for r in param_list if r.is_bimodal
    )
    logger.info(
        "check_shard_bimodality: %d params x %d shards; %d bimodal detections.",
        len(all_param_names),
        len(shard_samples),
        n_bimodal,
    )
    return results


# ---------------------------------------------------------------------------
# NLSQ comparison and precision analysis
# ---------------------------------------------------------------------------


def _compute_hdi_95(sorted_arr: np.ndarray) -> tuple[float, float]:
    """Return (low, high) for the shortest interval covering 95% of sorted samples."""
    sorted_arr = np.sort(sorted_arr)  # defensive: ensure sorted
    n = len(sorted_arr)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(sorted_arr[0]), float(sorted_arr[0])
    width = int(np.floor(0.95 * n))
    if width < n:
        intervals = sorted_arr[width:] - sorted_arr[: n - width]
        best = int(np.argmin(intervals))
        return float(sorted_arr[best]), float(sorted_arr[best + width])
    return float(sorted_arr[0]), float(sorted_arr[-1])


def compute_nlsq_comparison_metrics(
    posterior_samples: dict[str, np.ndarray],
    nlsq_values: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compare posterior statistics against NLSQ point estimates.

    For each parameter present in both ``posterior_samples`` and
    ``nlsq_values``, computes:

    * ``posterior_mean`` — mean of the flattened posterior draws.
    * ``posterior_std``  — standard deviation of the flattened draws.
    * ``nlsq_value``     — the NLSQ point estimate.
    * ``z_score``        — ``|nlsq_value - posterior_mean| / posterior_std``.
      NaN when ``posterior_std == 0``.
    * ``within_hdi``     — 1.0 if the NLSQ value falls inside the 95 % HDI,
      0.0 otherwise.

    Args:
        posterior_samples: Mapping of parameter name to sample array
            of shape ``(n_chains, n_draws)`` or ``(n_draws,)``.
        nlsq_values: NLSQ MAP estimates keyed by parameter name.

    Returns:
        Nested dict ``result[param_name][metric_name] = value``.
        Only parameters present in *both* inputs are included.
    """
    output: dict[str, dict[str, float]] = {}

    for name, nlsq_val in nlsq_values.items():
        if name not in posterior_samples:
            continue

        arr = np.asarray(posterior_samples[name], dtype=float).ravel()
        if arr.size == 0:
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr))

        z_score = float("nan")
        if std > 0:
            z_score = abs(nlsq_val - mean) / std

        # 95 % HDI
        hdi_low, hdi_high = _compute_hdi_95(np.sort(arr))
        within_hdi = 1.0 if hdi_low <= nlsq_val <= hdi_high else 0.0

        output[name] = {
            "posterior_mean": mean,
            "posterior_std": std,
            "nlsq_value": float(nlsq_val),
            "z_score": z_score,
            "within_hdi": within_hdi,
        }

        logger.debug(
            "NLSQ comparison for %s: mean=%.4e, std=%.4e, "
            "nlsq=%.4e, z=%.2f, within_hdi=%s",
            name,
            mean,
            std,
            nlsq_val,
            z_score,
            bool(within_hdi),
        )

    logger.info(
        "compute_nlsq_comparison_metrics: %d / %d parameters compared.",
        len(output),
        len(nlsq_values),
    )
    return output


def compute_precision_analysis(
    posterior_samples: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute precision metrics for each parameter's posterior.

    For each parameter, calculates:

    * ``mean``       — posterior mean.
    * ``std``        — posterior standard deviation.
    * ``cv``         — coefficient of variation = ``std / |mean|``.
      ``inf`` when ``mean == 0``.
    * ``hdi_width``  — width of the shortest interval containing 95 % of
      the posterior draws (highest-density interval).

    Args:
        posterior_samples: Mapping of parameter name to sample array
            of shape ``(n_chains, n_draws)`` or ``(n_draws,)``.

    Returns:
        Nested dict ``result[param_name][metric_name] = value``.
    """
    output: dict[str, dict[str, float]] = {}

    for name, samples in posterior_samples.items():
        arr = np.asarray(samples, dtype=float).ravel()
        if arr.size == 0:
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / abs(mean) if mean != 0.0 else float("inf")

        # 95 % HDI (shortest covering interval)
        hdi_low, hdi_high = _compute_hdi_95(np.sort(arr))
        hdi_width = hdi_high - hdi_low

        output[name] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "hdi_width": hdi_width,
        }

        logger.debug(
            "Precision for %s: mean=%.4e, std=%.4e, cv=%.4f, hdi_width=%.4e",
            name,
            mean,
            std,
            cv,
            hdi_width,
        )

    logger.info(
        "compute_precision_analysis: computed metrics for %d parameters.",
        len(output),
    )
    return output


# ---------------------------------------------------------------------------
# Cross-shard bimodal consensus
# ---------------------------------------------------------------------------


@dataclass
class ModeCluster:
    """A single mode from bimodal consensus combination.

    Attributes:
        mean: Per-parameter consensus mean for this mode.
        std: Per-parameter consensus std.
        weight: Fraction of shards supporting this mode (0-1).
        n_shards: Number of shards in this cluster.
    """

    mean: dict[str, float]
    std: dict[str, float]
    weight: float
    n_shards: int


@dataclass
class BimodalConsensusResult:
    """Result of mode-aware consensus combination.

    Attributes:
        modes: Mode clusters (typically 2) with per-mode consensus.
        modal_params: Parameter names that triggered bimodal detection.
        co_occurrence: Cross-parameter co-occurrence info.
    """

    modes: list[ModeCluster]
    modal_params: list[str]
    co_occurrence: dict[str, Any]


def summarize_cross_shard_bimodality(
    bimodal_detections: dict[str, list[BimodalResult]],
    n_shards: int,
    consensus_means: dict[str, float] | None = None,
    significance_threshold: float = 0.05,
) -> dict[str, Any]:
    """Aggregate per-shard bimodal detections into a cross-shard summary.

    Groups detections by parameter, computes mode statistics (mean of
    lower modes, mean of upper modes), and checks whether the consensus
    posterior mean falls between the modes (density trough).

    Args:
        bimodal_detections: Mapping from parameter name to a list of
            :class:`BimodalResult` (one per shard), as returned by
            :func:`check_shard_bimodality`.
        n_shards: Total number of shards.
        consensus_means: Consensus posterior means for each parameter.
            Used to check if consensus falls in the density trough
            between modes.
        significance_threshold: Minimum separation significance
            (``separation / pooled_std``) for a bimodal split to be
            reported.

    Returns:
        Dictionary with keys:

        - ``"per_param"``: ``{param_name -> {fraction_bimodal,
          lower_mode_mean, upper_mode_mean, separation, significance,
          consensus_in_trough}}``
        - ``"n_detections"``: Total bimodal detections across all params.
        - ``"n_shards"``: Number of shards.
    """
    per_param: dict[str, dict[str, Any]] = {}
    total_detections = 0

    for param_name, results in bimodal_detections.items():
        bimodal_results = [r for r in results if r.is_bimodal]
        n_bimodal = len(bimodal_results)
        total_detections += n_bimodal
        fraction_bimodal = n_bimodal / n_shards if n_shards > 0 else 0.0

        if n_bimodal == 0:
            per_param[param_name] = {
                "fraction_bimodal": 0.0,
                "lower_mode_mean": None,
                "upper_mode_mean": None,
                "separation": 0.0,
                "significance": 0.0,
                "consensus_in_trough": False,
            }
            continue

        # Collect the two means from each bimodal shard (sorted so
        # lower < upper within each detection).
        lower_modes: list[float] = []
        upper_modes: list[float] = []
        for r in bimodal_results:
            assert r.means is not None  # noqa: S101
            m0, m1 = sorted(r.means)
            lower_modes.append(m0)
            upper_modes.append(m1)

        lower_mean = float(np.mean(lower_modes))
        upper_mean = float(np.mean(upper_modes))
        separation = upper_mean - lower_mean

        # Pooled std across the two mode populations
        lower_std = float(np.std(lower_modes)) if len(lower_modes) > 1 else 1e-12
        upper_std = float(np.std(upper_modes)) if len(upper_modes) > 1 else 1e-12
        pooled_std = float(np.sqrt(0.5 * (lower_std**2 + upper_std**2)))
        pooled_std = max(pooled_std, 1e-12)  # avoid division by zero

        significance = separation / pooled_std

        # Check if consensus falls in the trough between modes
        consensus_in_trough = False
        if consensus_means is not None and param_name in consensus_means:
            c = consensus_means[param_name]
            # Trough defined as the middle 60% of the gap between modes
            margin = 0.2 * separation
            consensus_in_trough = (lower_mean + margin) < c < (upper_mean - margin)

        entry: dict[str, Any] = {
            "fraction_bimodal": fraction_bimodal,
            "lower_mode_mean": lower_mean,
            "upper_mode_mean": upper_mean,
            "separation": separation,
            "significance": significance,
            "consensus_in_trough": consensus_in_trough,
        }

        if significance < significance_threshold:
            logger.debug(
                "Bimodal split for %s has low significance (%.3f < %.3f); "
                "may not be meaningful.",
                param_name,
                significance,
                significance_threshold,
            )

        per_param[param_name] = entry

    logger.info(
        "summarize_cross_shard_bimodality: %d total detections across %d "
        "params, %d shards.",
        total_detections,
        len(per_param),
        n_shards,
    )

    return {
        "per_param": per_param,
        "n_detections": total_detections,
        "n_shards": n_shards,
    }


def cluster_shard_modes(
    bimodal_detections: dict[str, list[BimodalResult]],
    shard_samples: dict[int, dict[str, np.ndarray]],
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[int], list[int]]:
    """Jointly cluster shards into two mode populations.

    Uses the parameters that show bimodal behaviour to build a
    per-shard feature vector, then runs a simple 2-means clustering
    (no sklearn dependency) to partition shards.

    Args:
        bimodal_detections: Mapping from parameter name to a list of
            :class:`BimodalResult` as returned by
            :func:`check_shard_bimodality`.
        shard_samples: Per-shard samples mapping
            ``{shard_idx: {param_name: samples_array}}``.
        param_bounds: Optional per-parameter ``(lo, hi)`` bounds for
            normalization.  If *None*, the global range across shards
            is used.

    Returns:
        ``(cluster_0_indices, cluster_1_indices)`` where cluster 0 is
        the "lower" cluster (centroid with lower mean across features).
    """
    # Identify modal parameters (any param where ≥1 shard is bimodal)
    modal_params: list[str] = []
    for param_name, results in bimodal_detections.items():
        if any(r.is_bimodal for r in results):
            modal_params.append(param_name)

    shard_indices = sorted(shard_samples.keys())

    if not modal_params or len(shard_indices) < 2:
        logger.warning(
            "cluster_shard_modes: no modal params or < 2 shards; "
            "returning all shards in cluster 0."
        )
        return (shard_indices, [])

    # Build feature matrix: one row per shard, one col per modal param
    n_shards = len(shard_indices)
    n_features = len(modal_params)
    features = np.zeros((n_shards, n_features), dtype=float)

    for i, shard_idx in enumerate(shard_indices):
        shard_dict = shard_samples[shard_idx]
        for j, param_name in enumerate(modal_params):
            if param_name in shard_dict:
                arr = np.asarray(shard_dict[param_name], dtype=float).ravel()
                features[i, j] = float(np.mean(arr))
            else:
                features[i, j] = 0.0

    # Normalize each feature column
    for j, param_name in enumerate(modal_params):
        col = features[:, j]
        if param_bounds is not None and param_name in param_bounds:
            lo, hi = param_bounds[param_name]
            span = hi - lo
        else:
            lo = float(np.min(col))
            hi = float(np.max(col))
            span = hi - lo
        if span > 0:
            features[:, j] = (col - lo) / span

    # Simple 2-means clustering (iterative assignment)
    # Initialize centroids as min-mean and max-mean rows
    row_means = np.mean(features, axis=1)
    idx_lo = int(np.argmin(row_means))
    idx_hi = int(np.argmax(row_means))

    if idx_lo == idx_hi:
        # All shards identical; put all in cluster 0
        return (shard_indices, [])

    centroid_0 = features[idx_lo].copy()
    centroid_1 = features[idx_hi].copy()

    max_iters = 50
    labels = np.zeros(n_shards, dtype=int)

    for _iteration in range(max_iters):
        # Assign each shard to nearest centroid
        new_labels = np.zeros(n_shards, dtype=int)
        for i in range(n_shards):
            d0 = float(np.sum((features[i] - centroid_0) ** 2))
            d1 = float(np.sum((features[i] - centroid_1) ** 2))
            new_labels[i] = 0 if d0 <= d1 else 1

        if np.array_equal(new_labels, labels) and _iteration > 0:
            labels = new_labels
            break
        labels = new_labels

        # Recompute centroids
        mask_0 = labels == 0
        mask_1 = labels == 1
        if np.any(mask_0):
            centroid_0 = np.mean(features[mask_0], axis=0)
        if np.any(mask_1):
            centroid_1 = np.mean(features[mask_1], axis=0)

    # Ensure cluster 0 is the "lower" one (lower centroid mean)
    if float(np.mean(centroid_0)) > float(np.mean(centroid_1)):
        labels = 1 - labels

    cluster_0 = [shard_indices[i] for i in range(n_shards) if labels[i] == 0]
    cluster_1 = [shard_indices[i] for i in range(n_shards) if labels[i] == 1]

    logger.info(
        "cluster_shard_modes: %d shards in cluster 0, %d in cluster 1 "
        "(modal params: %s).",
        len(cluster_0),
        len(cluster_1),
        modal_params,
    )

    return (cluster_0, cluster_1)
