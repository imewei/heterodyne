"""NLSQ-informed prior construction for heterodyne CMC analysis.

Builds NumPyro distribution dictionaries from NLSQ warm-start results
or from the parameter registry defaults, including log-space priors
for parameters flagged with ``log_space=True``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpyro.distributions as dist

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY, ParameterRegistry
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_space import ParameterSpace
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


def build_nlsq_informed_priors(
    nlsq_result: NLSQResult,
    param_space: ParameterSpace,
    width_factor: float = 2.0,
) -> dict[str, dist.Distribution]:
    """Build priors centered on NLSQ point estimates.

    For each varying parameter, constructs a truncated Normal prior
    centered on the NLSQ best-fit value with width equal to the NLSQ
    uncertainty multiplied by ``width_factor``.  When NLSQ uncertainty
    is unavailable, falls back to registry ``prior_std`` or a fraction
    of the parameter range.

    Args:
        nlsq_result: Converged NLSQ result with parameter values and
            (optionally) uncertainties.
        param_space: Parameter space defining which parameters vary
            and their physical bounds.
        width_factor: Multiplier on NLSQ uncertainty to set prior width.
            Larger values give more diffuse priors.  Default 2.0 gives
            a prior that spans roughly 4 sigma around the NLSQ estimate.

    Returns:
        Dictionary mapping parameter names to NumPyro distributions.
        Only includes parameters in ``param_space.varying_names``.
    """
    priors: dict[str, dist.Distribution] = {}

    for name in param_space.varying_names:
        low, high = param_space.bounds[name]

        # Center: NLSQ value if available, else registry default
        if name in nlsq_result.parameter_names:
            center = float(nlsq_result.get_param(name))
        else:
            center = param_space.values[name]

        # Scale: NLSQ uncertainty * width_factor, or fallback
        nlsq_unc = nlsq_result.get_uncertainty(name)
        if nlsq_unc is not None and nlsq_unc > 0:
            scale = nlsq_unc * width_factor
        else:
            # Fallback: registry prior_std or 1/6 of bounds range
            info = DEFAULT_REGISTRY[name]
            if info.prior_std is not None and info.prior_std > 0:
                scale = info.prior_std
            else:
                scale = (high - low) / 6.0

        # Ensure minimum scale to avoid degenerate priors
        scale = max(scale, 1e-10)

        # Truncated normal: Normal constrained to [low, high]
        priors[name] = dist.TruncatedNormal(
            loc=center, scale=scale, low=low, high=high
        )

        logger.debug(
            "NLSQ-informed prior for %s: "
            "TruncatedNormal(loc=%.4e, scale=%.4e, low=%.4e, high=%.4e)",
            name, center, scale, low, high,
        )

    logger.info(
        "Built %d NLSQ-informed priors (width_factor=%s)",
        len(priors), width_factor,
    )
    return priors


def build_default_priors(
    param_space: ParameterSpace,
    registry: ParameterRegistry | None = None,
) -> dict[str, dist.Distribution]:
    """Build default priors from the parameter registry.

    Uses ``prior_mean`` and ``prior_std`` from each parameter's
    :class:`~heterodyne.config.parameter_registry.ParameterInfo`.
    When those fields are ``None``, falls back to a Uniform prior
    spanning the parameter bounds.

    Args:
        param_space: Parameter space defining which parameters vary
            and their physical bounds.
        registry: Parameter registry to read metadata from.
            Defaults to :data:`DEFAULT_REGISTRY`.

    Returns:
        Dictionary mapping parameter names to NumPyro distributions.
        Only includes parameters in ``param_space.varying_names``.
    """
    if registry is None:
        registry = DEFAULT_REGISTRY

    priors: dict[str, dist.Distribution] = {}

    for name in param_space.varying_names:
        info = registry[name]
        low, high = param_space.bounds[name]

        if info.prior_mean is not None and info.prior_std is not None and info.prior_std > 0:
            # Truncated normal centered on registry prior
            priors[name] = dist.TruncatedNormal(
                loc=info.prior_mean,
                scale=info.prior_std,
                low=low,
                high=high,
            )
            logger.debug(
                "Default prior for %s: TruncatedNormal(loc=%.4e, scale=%.4e)",
                name, info.prior_mean, info.prior_std,
            )
        else:
            # Uniform fallback
            priors[name] = dist.Uniform(low=low, high=high)
            logger.debug(
                "Default prior for %s: Uniform(%.4e, %.4e)",
                name, low, high,
            )

    logger.info("Built %d default priors from registry", len(priors))
    return priors


def build_log_space_priors(
    param_names: list[str],
    registry: ParameterRegistry | None = None,
) -> dict[str, dist.Distribution]:
    """Build log-normal priors for parameters marked ``log_space=True``.

    For parameters where the registry's ``log_space`` flag is set, this
    constructs a LogNormal distribution whose median matches the
    registry ``prior_mean`` (or the parameter default) and whose
    spread corresponds to the registry ``prior_std``.

    Parameters not flagged as ``log_space`` are silently skipped.

    Args:
        param_names: List of parameter names to consider.
        registry: Parameter registry. Defaults to :data:`DEFAULT_REGISTRY`.

    Returns:
        Dictionary mapping parameter names to LogNormal distributions.
        Only includes parameters where ``log_space=True``.
    """
    if registry is None:
        registry = DEFAULT_REGISTRY

    priors: dict[str, dist.Distribution] = {}

    for name in param_names:
        info = registry[name]
        if not info.log_space:
            continue

        # Determine the location (mode center) in original space
        if info.prior_mean is not None and info.prior_mean > 0:
            center = info.prior_mean
        elif info.default > 0:
            center = info.default
        else:
            # Cannot construct LogNormal for non-positive center
            logger.warning(
                "Skipping log-space prior for %s: "
                "prior_mean=%s, default=%s (both non-positive)",
                name, info.prior_mean, info.default,
            )
            continue

        # LogNormal parameterization: median = exp(mu), so mu = log(center)
        mu = math.log(center)

        # Sigma in log-space: if prior_std is available, use coefficient
        # of variation to set log-space spread.
        # CV = prior_std / prior_mean, and for LogNormal:
        # sigma = sqrt(log(1 + CV^2))
        if info.prior_std is not None and info.prior_std > 0 and center > 0:
            cv = info.prior_std / center
            sigma = math.sqrt(math.log1p(cv**2))
        else:
            # Default: moderate uncertainty (CV ~ 1.0 -> sigma ~ 0.83)
            sigma = 1.0

        # Floor sigma to avoid degenerate distributions
        sigma = max(sigma, 0.01)

        priors[name] = dist.LogNormal(loc=mu, scale=sigma)

        logger.debug(
            "Log-space prior for %s: LogNormal(loc=%.4f, scale=%.4f) [median=%.4e]",
            name, mu, sigma, center,
        )

    logger.info(
        "Built %d log-space priors from %d candidates",
        len(priors), len(param_names),
    )
    return priors


# ---------------------------------------------------------------------------
# Consensus MC prior tempering
# ---------------------------------------------------------------------------


def temper_priors(
    priors: dict[str, dist.Distribution],
    num_shards: int,
) -> dict[str, dist.Distribution]:
    """Scale prior widths for Consensus MC shard sub-posteriors.

    Each shard sees 1/K of the data, so the prior should be tempered by
    ``sqrt(K)`` to maintain proper posterior geometry when K
    sub-posteriors are combined via the consensus step.

    Supported distribution types and their tempering rules:

    * ``TruncatedNormal`` — scale multiplied by ``sqrt(K)``.
    * ``LogNormal`` — scale multiplied by ``sqrt(K)``.
    * ``Uniform`` — left unchanged (uninformative; no tempering needed).
    * All others — kept unchanged with a warning logged.

    Args:
        priors: Dict of NumPyro distributions, one per varying parameter.
        num_shards: Number of CMC shards (K).  Must be >= 1.

    Returns:
        New dict with tempered distributions.  Existing dict is not
        mutated.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    if num_shards == 1:
        # No tempering needed for a single shard
        logger.debug("temper_priors: num_shards=1, returning priors unchanged.")
        return dict(priors)

    factor = math.sqrt(num_shards)
    tempered: dict[str, dist.Distribution] = {}

    for name, prior in priors.items():
        if isinstance(prior, dist.TruncatedNormal):
            # TruncatedNormal stores loc/scale as tensors; extract as floats
            loc = float(prior.loc)
            scale = float(prior.scale) * factor
            low = float(prior.low)
            high = float(prior.high)
            tempered[name] = dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
            logger.debug(
                "temper_priors: %s TruncatedNormal scale %.4e -> %.4e (x%.2f)",
                name, float(prior.scale), scale, factor,
            )

        elif isinstance(prior, dist.LogNormal):
            loc = float(prior.loc)
            scale = float(prior.scale) * factor
            tempered[name] = dist.LogNormal(loc=loc, scale=scale)
            logger.debug(
                "temper_priors: %s LogNormal scale %.4e -> %.4e (x%.2f)",
                name, float(prior.scale), scale, factor,
            )

        elif isinstance(prior, dist.Uniform):
            # Uninformative; keep unchanged
            tempered[name] = prior
            logger.debug("temper_priors: %s Uniform — left unchanged.", name)

        else:
            # Unsupported type; keep unchanged and warn
            logger.warning(
                "temper_priors: %s has unsupported type %s — left unchanged. "
                "Consider using TruncatedNormal or LogNormal for proper tempering.",
                name, type(prior).__name__,
            )
            tempered[name] = prior

    logger.info(
        "Tempered %d priors for %d shards (scale factor=%.4f).",
        len(tempered), num_shards, factor,
    )
    return tempered


# ---------------------------------------------------------------------------
# Prior validation
# ---------------------------------------------------------------------------


def validate_priors(
    priors: dict[str, dist.Distribution],
    param_space: ParameterSpace,
) -> list[str]:
    """Validate prior distributions against parameter space.

    Checks:

    1. All varying parameters have a corresponding prior.
    2. Prior support overlaps with the parameter bounds (non-empty
       intersection).
    3. No degenerate (effectively zero-width) priors.

    A prior is considered degenerate when its extractable scale is
    below ``1e-12``.  Uniform priors are never degenerate.

    Args:
        priors: Dict of NumPyro distributions.
        param_space: Defines varying parameter names and their bounds.

    Returns:
        List of warning/error strings.  Empty list means all checks
        passed.
    """
    issues: list[str] = []

    for name in param_space.varying_names:
        # Check 1: prior exists
        if name not in priors:
            issues.append(f"Missing prior for varying parameter '{name}'.")
            continue

        prior = priors[name]
        low_bound, high_bound = param_space.bounds[name]

        # Check 2: support overlap with bounds
        # For distributions with explicit support attributes
        if isinstance(prior, dist.TruncatedNormal):
            prior_low = float(prior.low)
            prior_high = float(prior.high)
            if prior_high <= low_bound or prior_low >= high_bound:
                issues.append(
                    f"Prior for '{name}' support [{prior_low:.4e}, {prior_high:.4e}] "
                    f"does not overlap with bounds [{low_bound:.4e}, {high_bound:.4e}]."
                )

        elif isinstance(prior, dist.Uniform):
            prior_low = float(prior.low)
            prior_high = float(prior.high)
            if prior_high <= low_bound or prior_low >= high_bound:
                issues.append(
                    f"Prior for '{name}' Uniform[{prior_low:.4e}, {prior_high:.4e}] "
                    f"does not overlap with bounds [{low_bound:.4e}, {high_bound:.4e}]."
                )

        elif isinstance(prior, dist.LogNormal):
            # LogNormal has support (0, inf); check lower bound is >= 0
            if high_bound <= 0:
                issues.append(
                    f"Prior for '{name}' is LogNormal (support > 0) but "
                    f"upper bound is {high_bound:.4e} <= 0."
                )

        # Check 3: degenerate prior (near-zero scale)
        _scale: float | None = None
        if isinstance(prior, dist.TruncatedNormal):
            _scale = float(prior.scale)
        elif isinstance(prior, dist.LogNormal):
            _scale = float(prior.scale)
        # Uniform is never degenerate by construction

        if _scale is not None and _scale < 1e-12:
            issues.append(
                f"Prior for '{name}' is degenerate: scale={_scale:.2e} < 1e-12."
            )

    # Report any priors defined for non-varying parameters (informational)
    varying_set = set(param_space.varying_names)
    for name in priors:
        if name not in varying_set:
            issues.append(
                f"Prior defined for '{name}' but it is not a varying parameter. "
                "This prior will be ignored by the sampler."
            )

    if issues:
        logger.warning(
            "validate_priors: %d issue(s) found:\n  %s",
            len(issues), "\n  ".join(issues),
        )
    else:
        logger.info("validate_priors: all %d priors passed validation.", len(priors))

    return issues


# ---------------------------------------------------------------------------
# Prior summary
# ---------------------------------------------------------------------------


def summarize_priors(priors: dict[str, dist.Distribution]) -> str:
    """Format a human-readable summary of prior distributions.

    For each prior, reports the distribution type and, where applicable,
    the mean, standard deviation, and support interval.

    Args:
        priors: Dict of NumPyro distributions.

    Returns:
        Multi-line string with one row per parameter.
    """
    if not priors:
        return "No priors defined."

    lines: list[str] = ["Prior summary:"]
    name_width = max(len(n) for n in priors) + 2

    for name, prior in priors.items():
        label = f"  {name:<{name_width}}"

        if isinstance(prior, dist.TruncatedNormal):
            loc = float(prior.loc)
            scale = float(prior.scale)
            low = float(prior.low)
            high = float(prior.high)
            lines.append(
                f"{label}TruncatedNormal  "
                f"loc={loc:.4e}  scale={scale:.4e}  "
                f"support=[{low:.4e}, {high:.4e}]"
            )

        elif isinstance(prior, dist.LogNormal):
            loc = float(prior.loc)
            scale = float(prior.scale)
            # Median of LogNormal = exp(loc); mean = exp(loc + scale^2/2)
            median = math.exp(loc)
            mean = math.exp(loc + 0.5 * scale ** 2)
            std = math.sqrt((math.exp(scale ** 2) - 1.0) * math.exp(2.0 * loc + scale ** 2))
            lines.append(
                f"{label}LogNormal        "
                f"loc={loc:.4f}  scale={scale:.4f}  "
                f"median={median:.4e}  mean={mean:.4e}  std={std:.4e}  "
                f"support=(0, +inf)"
            )

        elif isinstance(prior, dist.Uniform):
            low = float(prior.low)
            high = float(prior.high)
            mean = (low + high) / 2.0
            std = (high - low) / math.sqrt(12.0)
            lines.append(
                f"{label}Uniform          "
                f"support=[{low:.4e}, {high:.4e}]  "
                f"mean={mean:.4e}  std={std:.4e}"
            )

        else:
            lines.append(f"{label}{type(prior).__name__}")

    return "\n".join(lines)
