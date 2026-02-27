"""Convergence diagnostics for CMC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """Compute R-hat (Gelman-Rubin statistic) from samples.

    .. deprecated::
        Use ``arviz.rhat()`` instead.

    Args:
        samples: Array of shape (n_chains, n_samples)

    Returns:
        R-hat value
    """
    import warnings
    warnings.warn("Use arviz.rhat() instead", DeprecationWarning, stacklevel=2)
    n_chains, n_samples = samples.shape
    
    # Chain means
    chain_means = np.mean(samples, axis=1)
    
    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    chain_vars = np.var(samples, axis=1, ddof=1)
    W = np.mean(chain_vars)
    
    # Pooled variance estimate
    var_hat = (1 - 1/n_samples) * W + B / n_samples

    # R-hat (handle degenerate case where W=0)
    if W == 0:
        # No within-chain variance: if B=0 too, chains are identical (R-hat=1)
        # otherwise chains haven't mixed (R-hat=inf)
        return 1.0 if B == 0 else float("inf")

    r_hat = np.sqrt(var_hat / W)

    return float(r_hat)


def compute_ess(samples: np.ndarray) -> float:
    """Compute effective sample size using autocorrelation.

    .. deprecated::
        Use ``arviz.ess()`` instead.

    Args:
        samples: 1D array of samples

    Returns:
        Effective sample size
    """
    import warnings
    warnings.warn("Use arviz.ess() instead", DeprecationWarning, stacklevel=2)
    n = len(samples)
    
    # Mean-center
    x = samples - np.mean(samples)
    
    # Autocorrelation via FFT
    acf = np.correlate(x, x, mode='full')[n-1:]

    # Handle degenerate case where samples are constant (no variance)
    if acf[0] == 0:
        return 1.0  # Constant samples = effectively 1 independent sample

    acf = acf / acf[0]
    
    # Find first negative autocorrelation
    negative_idx = np.where(acf < 0)[0]
    if len(negative_idx) > 0:
        cutoff = negative_idx[0]
    else:
        cutoff = len(acf)
    
    # Sum of autocorrelations
    tau = 1 + 2 * np.sum(acf[1:cutoff])
    
    # ESS
    ess = n / tau
    
    return max(1.0, float(ess))


def compute_bfmi(energy: np.ndarray) -> float:
    """Compute Bayesian Fraction of Missing Information.

    .. deprecated::
        Use ``arviz.bfmi()`` instead.

    BFMI measures how well the sampler explores the posterior.
    Values < 0.3 indicate potential problems.

    Args:
        energy: Array of HMC energies

    Returns:
        BFMI value
    """
    import warnings
    warnings.warn("Use arviz.bfmi() instead", DeprecationWarning, stacklevel=2)
    energy_diff = np.diff(energy)
    var_diff = np.var(energy_diff)
    var_energy = np.var(energy)
    
    if var_energy == 0:
        return 1.0
    
    return float(var_diff / var_energy)
