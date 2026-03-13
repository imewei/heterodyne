"""NLSQ-informed prior construction for heterodyne CMC analysis.

Builds NumPyro distribution dictionaries from NLSQ warm-start results
or from the parameter registry defaults, including log-space priors
for parameters flagged with ``log_space=True``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpyro.distributions as dist
from numpyro.distributions.truncated import TwoSidedTruncatedDistribution

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
        priors[name] = dist.TruncatedNormal(loc=center, scale=scale, low=low, high=high)

        logger.debug(
            "NLSQ-informed prior for %s: "
            "TruncatedNormal(loc=%.4e, scale=%.4e, low=%.4e, high=%.4e)",
            name,
            center,
            scale,
            low,
            high,
        )

    logger.info(
        "Built %d NLSQ-informed priors (width_factor=%s)",
        len(priors),
        width_factor,
    )
    return priors


def build_default_priors(
    param_space: ParameterSpace,
    registry: ParameterRegistry | None = None,
) -> dict[str, dist.Distribution]:
    """Build default priors from the parameter registry.

    Uses ``prior_mean`` and ``prior_std`` from each parameter's
    :class:`~heterodyne.config.parameter_registry.ParameterInfo`.
    For bounded fraction/contrast parameters (f0, f3, contrast),
    auto-selects BetaScaled priors when prior_mean and prior_std are
    available and bounds are finite. Otherwise falls back to
    TruncatedNormal or Uniform.

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

    # Parameters that benefit from BetaScaled priors (bounded [0, 1] or similar)
    _BETA_SCALED_CANDIDATES = {"f0", "f3", "contrast"}

    priors: dict[str, dist.Distribution] = {}

    for name in param_space.varying_names:
        info = registry[name]
        low, high = param_space.bounds[name]

        # Try BetaScaled for candidate parameters with finite bounds
        if (
            name in _BETA_SCALED_CANDIDATES
            and info.prior_mean is not None
            and info.prior_std is not None
            and info.prior_std > 0
            and math.isfinite(low)
            and math.isfinite(high)
            and low < high
        ):
            try:
                from heterodyne.config.parameter_space import (
                    _compute_beta_concentrations,
                )

                conc1, conc2 = _compute_beta_concentrations(
                    info.prior_mean,
                    info.prior_std,
                    low,
                    high,
                )
                base = dist.Beta(conc1, conc2)
                priors[name] = dist.TransformedDistribution(
                    base,
                    dist.transforms.AffineTransform(loc=low, scale=high - low),
                )
                logger.debug(
                    "BetaScaled prior for %s: Beta(%.4f, %.4f) on [%.4e, %.4e]",
                    name,
                    conc1,
                    conc2,
                    low,
                    high,
                )
                continue
            except ValueError:
                logger.debug(
                    "BetaScaled not feasible for %s (std too large); "
                    "falling back to TruncatedNormal",
                    name,
                )

        if (
            info.prior_mean is not None
            and info.prior_std is not None
            and info.prior_std > 0
        ):
            # Truncated normal centered on registry prior
            priors[name] = dist.TruncatedNormal(
                loc=info.prior_mean,
                scale=info.prior_std,
                low=low,
                high=high,
            )
            logger.debug(
                "Default prior for %s: TruncatedNormal(loc=%.4e, scale=%.4e)",
                name,
                info.prior_mean,
                info.prior_std,
            )
        else:
            # Uniform fallback
            priors[name] = dist.Uniform(low=low, high=high)
            logger.debug(
                "Default prior for %s: Uniform(%.4e, %.4e)",
                name,
                low,
                high,
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
                name,
                info.prior_mean,
                info.default,
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
            name,
            mu,
            sigma,
            center,
        )

    logger.info(
        "Built %d log-space priors from %d candidates",
        len(priors),
        len(param_names),
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
        if isinstance(prior, TwoSidedTruncatedDistribution):
            # TwoSidedTruncatedDistribution wraps a Normal base_dist
            loc = float(prior.base_dist.loc)
            old_scale = float(prior.base_dist.scale)
            scale = old_scale * factor
            low = float(prior.low)
            high = float(prior.high)
            tempered[name] = dist.TruncatedNormal(
                loc=loc, scale=scale, low=low, high=high
            )
            logger.debug(
                "temper_priors: %s TruncatedNormal scale %.4e -> %.4e (x%.2f)",
                name,
                old_scale,
                scale,
                factor,
            )

        elif isinstance(prior, dist.LogNormal):
            loc = float(prior.loc)
            scale = float(prior.scale) * factor
            tempered[name] = dist.LogNormal(loc=loc, scale=scale)
            logger.debug(
                "temper_priors: %s LogNormal scale %.4e -> %.4e (x%.2f)",
                name,
                float(prior.scale),
                scale,
                factor,
            )

        elif isinstance(prior, dist.Uniform):
            # Uninformative; keep unchanged
            tempered[name] = prior
            logger.debug("temper_priors: %s Uniform — left unchanged.", name)

        elif isinstance(prior, dist.TransformedDistribution):
            # BetaScaled: cannot easily temper — leave unchanged with warning
            logger.warning(
                "temper_priors: %s has TransformedDistribution (e.g. BetaScaled) — "
                "left unchanged. Consider using TruncatedNormal for tempered CMC.",
                name,
            )
            tempered[name] = prior

        else:
            # Unsupported type; keep unchanged and warn
            logger.warning(
                "temper_priors: %s has unsupported type %s — left unchanged. "
                "Consider using TruncatedNormal or LogNormal for proper tempering.",
                name,
                type(prior).__name__,
            )
            tempered[name] = prior

    logger.info(
        "Tempered %d priors for %d shards (scale factor=%.4f).",
        len(tempered),
        num_shards,
        factor,
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
        if isinstance(prior, TwoSidedTruncatedDistribution):
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

        elif isinstance(prior, dist.TransformedDistribution):
            # TransformedDistribution (e.g. BetaScaled) — trust the bounds
            # are correct since they're set during construction
            pass

        # Check 3: degenerate prior (near-zero scale)
        _scale: float | None = None
        if isinstance(prior, TwoSidedTruncatedDistribution):
            _scale = float(prior.base_dist.scale)
        elif isinstance(prior, dist.LogNormal):
            _scale = float(prior.scale)
        elif isinstance(prior, dist.TransformedDistribution):
            # Check base distribution for degeneracy
            base = prior.base_dist
            if isinstance(base, dist.Beta):
                # Beta is never degenerate if both concentrations > 0
                pass
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
            len(issues),
            "\n  ".join(issues),
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

        if isinstance(prior, TwoSidedTruncatedDistribution):
            loc = float(prior.base_dist.loc)
            scale = float(prior.base_dist.scale)
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
            mean = math.exp(loc + 0.5 * scale**2)
            std = math.sqrt((math.exp(scale**2) - 1.0) * math.exp(2.0 * loc + scale**2))
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

        elif isinstance(prior, dist.TransformedDistribution):
            base = prior.base_dist
            if isinstance(base, dist.Beta):
                conc1 = float(base.concentration1)
                conc2 = float(base.concentration0)
                # Extract affine transform parameters
                transforms = prior.transforms
                if transforms:
                    t = transforms[0]  # AffineTransform
                    loc = float(t.loc)
                    scale = float(t.scale)
                    mean = loc + scale * conc1 / (conc1 + conc2)
                    lines.append(
                        f"{label}BetaScaled       "
                        f"alpha={conc1:.4f}  beta={conc2:.4f}  "
                        f"support=[{loc:.4e}, {loc + scale:.4e}]  "
                        f"mean={mean:.4e}"
                    )
                else:
                    lines.append(
                        f"{label}TransformedBeta  alpha={conc1:.4f}  beta={conc2:.4f}"
                    )
            else:
                lines.append(f"{label}Transformed({type(base).__name__})")

        else:
            lines.append(f"{label}{type(prior).__name__}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter name helpers
# ---------------------------------------------------------------------------


def get_param_names_in_order(
    vary_flags: dict[str, bool] | None = None,
) -> list[str]:
    """Return the ordered list of parameter names that are set to vary.

    Iteration order matches the registry's insertion order, which
    follows the canonical group ordering defined in
    ``parameter_names.py`` (reference → sample → velocity → fraction →
    angle → scaling).

    Args:
        vary_flags: Optional override dict mapping parameter name to a
            bool indicating whether that parameter varies.  Parameters
            absent from ``vary_flags`` fall back to the registry's
            ``vary_default`` attribute.  Pass ``None`` to use registry
            defaults for all parameters.

    Returns:
        List of parameter names for which the effective ``vary`` flag is
        ``True``, in registry order.
    """
    vary_flags = vary_flags or {}
    names: list[str] = []
    for name in DEFAULT_REGISTRY:
        info = DEFAULT_REGISTRY[name]
        effective_vary = vary_flags.get(name, info.vary_default)
        if effective_vary:
            names.append(name)

    logger.debug(
        "get_param_names_in_order: %d varying parameters selected.", len(names)
    )
    return names


# ---------------------------------------------------------------------------
# Initial value construction and validation
# ---------------------------------------------------------------------------


def validate_initial_value_bounds(
    init_values: dict[str, float],
    param_specs: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Check that each initial value lies within the parameter's bounds.

    Args:
        init_values: Mapping of parameter name to proposed initial value.
        param_specs: Optional dict of ``{name: {min_bound, max_bound}}``
            overrides.  When not provided, bounds are read from
            :data:`DEFAULT_REGISTRY`.

    Returns:
        Mapping from parameter name to a list of warning strings.  An
        empty dict indicates all values are within bounds.
    """
    issues: dict[str, list[str]] = {}

    for name, value in init_values.items():
        # Determine bounds
        if param_specs and name in param_specs:
            spec = param_specs[name]
            low = float(spec.get("min_bound", -math.inf))
            high = float(spec.get("max_bound", math.inf))
        elif name in DEFAULT_REGISTRY:
            info = DEFAULT_REGISTRY[name]
            low = info.min_bound
            high = info.max_bound
        else:
            # Unknown parameter — skip bounds check but warn
            logger.warning(
                "validate_initial_value_bounds: unknown parameter '%s', "
                "not in registry or param_specs — skipping bounds check.",
                name,
            )
            continue

        param_issues: list[str] = []
        if value < low:
            param_issues.append(f"Value {value:.4e} is below min_bound {low:.4e}.")
        if value > high:
            param_issues.append(f"Value {value:.4e} is above max_bound {high:.4e}.")

        if param_issues:
            issues[name] = param_issues
            logger.warning(
                "validate_initial_value_bounds: %s — %s",
                name,
                "; ".join(param_issues),
            )

    if not issues:
        logger.debug(
            "validate_initial_value_bounds: all %d values within bounds.",
            len(init_values),
        )

    return issues


def build_init_values_dict(
    nlsq_values: dict[str, float] | None = None,
    vary_flags: dict[str, bool] | None = None,
    fallback: str = "prior_mean",
) -> dict[str, float]:
    """Build an initial-values dict for NUTS warm-starting.

    For each varying parameter the value is resolved in order:

    1. NLSQ estimate from ``nlsq_values`` (if available).
    2. Registry ``prior_mean`` when ``fallback="prior_mean"`` and
       ``prior_mean`` is not ``None``.
    3. Registry ``default`` value.

    All resolved values are validated against bounds and clamped when
    necessary, with a logged warning per clamped parameter.

    Args:
        nlsq_values: Optional NLSQ MAP estimates keyed by parameter
            name.
        vary_flags: Optional dict controlling which parameters vary (see
            :func:`get_param_names_in_order`).
        fallback: Strategy for parameters absent from ``nlsq_values``.
            ``"prior_mean"`` uses the registry prior mean (default);
            ``"default"`` uses the registry default value.

    Returns:
        Dict mapping each varying parameter name to its initial value,
        ready to pass to :meth:`~heterodyne.optimization.cmc.sampler.NUTSSampler.run_with_init_values`.
    """
    if fallback not in {"prior_mean", "default"}:
        raise ValueError(
            f"fallback must be 'prior_mean' or 'default', got {fallback!r}"
        )

    nlsq_values = nlsq_values or {}
    param_names = get_param_names_in_order(vary_flags)

    init_values: dict[str, float] = {}

    for name in param_names:
        info = DEFAULT_REGISTRY[name]

        # 1. NLSQ estimate
        if name in nlsq_values:
            value = float(nlsq_values[name])
        # 2/3. Fallback
        elif fallback == "prior_mean" and info.prior_mean is not None:
            value = float(info.prior_mean)
        else:
            value = float(info.default)

        # Clamp to bounds and warn if adjusted
        clamped = float(max(info.min_bound, min(info.max_bound, value)))
        if clamped != value:
            logger.warning(
                "build_init_values_dict: %s initial value %.4e clamped to [%.4e, %.4e] -> %.4e",
                name,
                value,
                info.min_bound,
                info.max_bound,
                clamped,
            )
        init_values[name] = clamped

    logger.info(
        "build_init_values_dict: built %d initial values (nlsq=%d, fallback=%r).",
        len(init_values),
        sum(1 for n in param_names if n in nlsq_values),
        fallback,
    )
    return init_values


# ---------------------------------------------------------------------------
# NLSQ value extraction for CMC warm-starting
# ---------------------------------------------------------------------------


def extract_nlsq_values_for_cmc(
    nlsq_result: NLSQResult,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Extract parameter values and uncertainties from an NLSQ result.

    Converts the array-based :class:`NLSQResult` into plain ``float``
    dictionaries suitable for CMC warm-starting.  NaN and inf values are
    filtered out so that downstream prior construction never receives
    non-finite inputs.

    Args:
        nlsq_result: Converged NLSQ result with ``.parameters``,
            ``.parameter_names``, and optionally ``.uncertainties``.

    Returns:
        A tuple ``(values, uncertainties)`` where:

        * ``values`` maps parameter name to its fitted float value
          (non-finite entries excluded).
        * ``uncertainties`` maps parameter name to its float
          uncertainty, or ``None`` when the NLSQ result carries no
          uncertainty information.  Non-finite entries are excluded.
    """
    values: dict[str, float] = {}
    n_skipped = 0

    for name, val in zip(
        nlsq_result.parameter_names,
        nlsq_result.parameters,
        strict=True,
    ):
        fval = float(val)
        if not math.isfinite(fval):
            logger.debug(
                "extract_nlsq_values_for_cmc: skipping %s value (non-finite: %s)",
                name,
                fval,
            )
            n_skipped += 1
            continue
        values[name] = fval

    # Uncertainties
    uncertainties: dict[str, float] | None = None
    if nlsq_result.uncertainties is not None:
        uncertainties = {}
        for name, unc in zip(
            nlsq_result.parameter_names,
            nlsq_result.uncertainties,
            strict=True,
        ):
            func = float(unc)
            if not math.isfinite(func):
                logger.debug(
                    "extract_nlsq_values_for_cmc: skipping %s uncertainty "
                    "(non-finite: %s)",
                    name,
                    func,
                )
                continue
            uncertainties[name] = func

    logger.info(
        "extract_nlsq_values_for_cmc: extracted %d values, %s uncertainties "
        "(%d non-finite skipped)",
        len(values),
        len(uncertainties) if uncertainties is not None else "no",
        n_skipped,
    )
    return values, uncertainties


# ---------------------------------------------------------------------------
# Per-angle scaling estimation
# ---------------------------------------------------------------------------


def estimate_per_angle_scaling(
    data_dict: dict[str, Any],
    angle_keys: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Estimate contrast and offset scaling per scattering angle.

    Uses simple heuristics on the raw g2 correlation data to provide
    starting-point estimates for the ``contrast`` and ``offset``
    scaling parameters:

    * ``contrast`` estimate ≈ ``max(g2) - min(g2)`` over the full
      lag range.
    * ``offset`` estimate ≈ mean of the last 10 % of g2 values
      (long-lag baseline), clamped to ``[0, 1]``.

    These are heuristics suitable for warm-starting, not MAP estimates.
    The NLSQ/MCMC optimisation will refine them.

    Args:
        data_dict: Dict mapping angle keys to g2 data.  Each value may
            be:

            * a 1-D array-like ``(n_lags,)`` of g2 values, or
            * a dict with a ``"g2"`` key holding such an array.

        angle_keys: Subset of keys in ``data_dict`` to process.
            Defaults to all keys when ``None``.

    Returns:
        Mapping of ``angle_key -> (contrast_estimate, offset_estimate)``.
        Keys for which data could not be parsed are silently omitted.
    """
    import numpy as np

    if angle_keys is None:
        angle_keys = list(data_dict.keys())

    result: dict[str, tuple[float, float]] = {}

    for key in angle_keys:
        raw = data_dict.get(key)
        if raw is None:
            logger.warning(
                "estimate_per_angle_scaling: key '%s' not found in data_dict.",
                key,
            )
            continue

        # Accept either a plain array or a dict with a "g2" sub-key
        if isinstance(raw, dict):
            g2_raw = raw.get("g2")
            if g2_raw is None:
                logger.debug(
                    "estimate_per_angle_scaling: key '%s' dict has no 'g2' entry.",
                    key,
                )
                continue
        else:
            g2_raw = raw

        try:
            g2 = np.asarray(g2_raw, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            logger.debug(
                "estimate_per_angle_scaling: cannot convert key '%s' to array: %s",
                key,
                exc,
            )
            continue

        if g2.size == 0:
            continue

        g2_max = float(np.nanmax(g2))
        g2_min = float(np.nanmin(g2))

        contrast_est = g2_max - g2_min

        # Baseline: mean of the last 10 % of points (long-tau asymptote)
        n_tail = max(1, int(np.ceil(0.1 * g2.size)))
        offset_est = float(np.nanmean(g2[-n_tail:]))
        # Physical constraint: offset in [0, 1]
        offset_est = float(np.clip(offset_est, 0.0, 1.0))

        result[key] = (contrast_est, offset_est)

        logger.debug(
            "estimate_per_angle_scaling: key='%s', contrast=%.4e, offset=%.4e",
            key,
            contrast_est,
            offset_est,
        )

    logger.info(
        "estimate_per_angle_scaling: estimated scaling for %d / %d angles.",
        len(result),
        len(angle_keys),
    )
    return result
