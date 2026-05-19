"""NLSQ-informed prior construction for heterodyne CMC analysis.

Builds NumPyro distribution dictionaries from NLSQ warm-start results
or from the parameter registry defaults, including log-space priors
for parameters flagged with ``log_space=True``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np
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


def _verify_dual_prior_sync(registry: ParameterRegistry | None) -> None:
    """Run the PriorBuilder construction-time sync gate without circular import.

    Built as a side-effect of :func:`build_default_priors` so every prior-
    construction call validates Rule 9.  Imports :class:`PriorBuilder`
    locally because ``prior_builder`` itself imports this module's
    legacy implementations as fallbacks.
    """
    from heterodyne.optimization.cmc.prior_builder import PriorBuilder

    # PriorBuilder.__init__ runs the dual-source comparison; we drop the
    # instance immediately because we already have the implementation
    # below — only the side-effect matters.
    PriorBuilder(registry=registry, use_log_space_priors=True)


def build_default_priors(
    param_space: ParameterSpace,
    registry: ParameterRegistry | None = None,
    use_log_space_priors: bool = True,
) -> dict[str, dist.Distribution]:
    """Build default priors from the parameter registry.

    Uses ``prior_mean`` and ``prior_std`` from each parameter's
    :class:`~heterodyne.config.parameter_registry.ParameterInfo`.
    All bounded parameters use TruncatedNormal so that ``temper_priors``
    can scale them by ``sqrt(K)`` for Consensus Monte Carlo sharding.

    Codex S1: when ``use_log_space_priors=True`` (default), parameters
    flagged ``log_space=True`` in the registry (D0_ref, D0_sample, v0)
    are overridden with LogNormal priors via :func:`build_log_space_priors`.
    LogNormal mass-matrix conditioning is much better for prefactors that
    span several orders of magnitude.  Pass ``use_log_space_priors=False``
    to fall back to TruncatedNormal uniformly.

    The reparameterized path (``CMCConfig.use_reparam=True``) is unaffected:
    it samples log_X_at_tref directly and never calls this function for
    those parameters.

    Args:
        param_space: Parameter space defining which parameters vary
            and their physical bounds.
        registry: Parameter registry to read metadata from.
            Defaults to :data:`DEFAULT_REGISTRY`.
        use_log_space_priors: Apply log-space priors to ``log_space=True``
            parameters from the registry.  Default ``True``.

    Returns:
        Dictionary mapping parameter names to NumPyro distributions.
        Only includes parameters in ``param_space.varying_names``.
    """
    if registry is None:
        registry = DEFAULT_REGISTRY
        # Gemini G2: enforce dual-prior sync (CLAUDE.md Rule 9) at
        # construction time only on the default path.  Callers who pass
        # an explicit ``registry=`` (e.g. unit tests using stub registries
        # to isolate one code path) are trusted to manage sync themselves.
        _verify_dual_prior_sync(None)

    priors: dict[str, dist.Distribution] = {}

    for name in param_space.varying_names:
        info = registry[name]
        low, high = param_space.bounds[name]

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

    # Codex S1: overlay LogNormal priors for parameters flagged log_space=True.
    # build_log_space_priors silently skips parameters that aren't flagged,
    # so this only touches D0_ref / D0_sample / v0 (or whatever the registry
    # declares); other parameters keep their TruncatedNormal/Uniform priors.
    if use_log_space_priors:
        log_priors = build_log_space_priors(
            list(param_space.varying_names), registry=registry
        )
        for name, prior in log_priors.items():
            priors[name] = prior

    logger.info(
        "Built %d default priors from registry (use_log_space_priors=%s)",
        len(priors),
        use_log_space_priors,
    )
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


def estimate_contrast_offset_from_data(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    contrast_bounds: tuple[float, float] = (0.0, 1.0),
    offset_bounds: tuple[float, float] = (0.5, 1.5),
    lag_floor_quantile: float = 0.80,
    lag_ceiling_quantile: float = 0.20,
    value_quantile_low: float = 0.10,
    value_quantile_high: float = 0.90,
) -> tuple[float, float]:
    """Estimate contrast and offset from C2 data via physics-informed quantile analysis.

    Uses the correlation decay: C2 = contrast × g1² + offset.
    At large lags g1² → 0 so C2 → offset; at small lags g1² ≈ 1
    so C2 ≈ contrast + offset.

    Returns
    -------
    tuple[float, float]
        ``(contrast_est, offset_est)`` each clipped to their bounds.
    """
    delta_t = np.abs(np.asarray(t1) - np.asarray(t2)).ravel()
    c2 = np.asarray(c2_data).ravel()

    if len(c2) < 100:
        contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0
        logger.debug(
            "estimate_contrast_offset_from_data: insufficient data (%d pts), "
            "returning midpoints contrast=%.3f offset=%.3f",
            len(c2),
            contrast_mid,
            offset_mid,
        )
        return contrast_mid, offset_mid

    lag_hi = np.percentile(delta_t, lag_floor_quantile * 100)
    lag_lo = np.percentile(delta_t, lag_ceiling_quantile * 100)

    # OFFSET: large-lag region where g1² ≈ 0
    mask_hi = delta_t >= lag_hi
    if np.sum(mask_hi) >= 10:
        offset_est = np.percentile(c2[mask_hi], value_quantile_low * 100)
    else:
        offset_est = np.percentile(c2, value_quantile_low * 100)
    offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

    # CONTRAST: small-lag region where g1² ≈ 1
    mask_lo = delta_t <= lag_lo
    if np.sum(mask_lo) >= 10:
        ceiling = np.percentile(c2[mask_lo], value_quantile_high * 100)
    else:
        ceiling = np.percentile(c2, value_quantile_high * 100)
    contrast_est = float(
        np.clip(ceiling - offset_est, contrast_bounds[0], contrast_bounds[1])
    )

    logger.debug(
        "estimate_contrast_offset_from_data: offset=%.4f contrast=%.4f",
        offset_est,
        contrast_est,
    )
    return contrast_est, offset_est


def validate_init_values_order(
    init_values: dict[str, float],
    expected_names: list[str],
) -> None:
    """Validate that init-values key order matches the expected parameter order.

    Homodyne CMC parity helper.  Python 3.7+ dicts preserve insertion order,
    so positional consumers of ``init_values`` (e.g. when zipping with NLSQ
    arrays) depend on a stable iteration order.  This check makes ordering
    bugs fail loudly with a descriptive error instead of producing silently
    wrong parameter bindings.

    Args:
        init_values: Initial-values mapping, typically the output of
            :func:`build_init_values_dict`.
        expected_names: Required parameter order — usually
            :func:`get_param_names_in_order` for the active mode.

    Raises:
        ValueError: When the key count or per-position order disagrees with
            ``expected_names``.  The error names the first mismatching index
            and quotes the full lists for fast diagnosis.
    """
    actual_names = list(init_values.keys())
    if actual_names == expected_names:
        return

    if len(actual_names) != len(expected_names):
        raise ValueError(
            "Parameter count mismatch in init_values:\n"
            f"  expected {len(expected_names)} params: {expected_names}\n"
            f"  actual   {len(actual_names)} params: {actual_names}"
        )

    for i, (actual, expected) in enumerate(
        zip(actual_names, expected_names, strict=False)
    ):
        if actual != expected:
            raise ValueError(
                "Parameter order mismatch at position "
                f"{i}:\n  expected: {expected!r}\n  actual:   {actual!r}\n"
                f"  full expected: {expected_names}\n"
                f"  full actual:   {actual_names}"
            )


# ---------------------------------------------------------------------------
# Absorbed from optimization/cmc/warmstart.py (Phase 4 PR 4 Task 4.4; spec §7 Rule 1)
# Preserves the two public call signatures + geometric-margin log-space math
# + fixed_param_overrides arg logic from REPORT.md §Manual narrative diffs §6.
# ---------------------------------------------------------------------------


#: Default boundary-clamp margin: 5% of the bound range (linear) or 5% of
#: the log-range (geometric, for ``log_space=True`` parameters).  NUTS
#: leapfrog step-size adaptation collapses if the chain initialises at a
#: TruncatedNormal boundary; this margin keeps the warm-start away from
#: the reflecting wall.
BOUNDARY_INTERIOR_MARGIN: float = 5e-2


def clamp_params_to_interior(
    params: np.ndarray,
    parameter_names: list[str],
    *,
    margin: float = BOUNDARY_INTERIOR_MARGIN,
) -> tuple[np.ndarray, list[str]]:
    """Shift parameter values inward from hard bounds (raw-array path).

    Lower-level companion to :func:`clamp_to_interior` (the NLSQResult
    entry point).  Named separately so the two public APIs don't share
    a symbol — callers without an ``NLSQResult`` (e.g. direct Python
    users passing raw arrays to :func:`fit_cmc_jax`) use this entry.

    Linear-scale parameters are clamped to ``[lo + margin * range,
    hi - margin * range]`` where ``range = hi - lo``.  Log-space
    parameters (D0_ref, D0_sample, v0 — registry ``log_space=True``,
    positive ``min_bound``) use a *geometric* margin so a 5% linear
    fraction of a multi-decade range does not produce a 500× clamp
    target.

    Args:
        params: Array of parameter values, shape ``(n,)``.
        parameter_names: Names corresponding to each entry in
            ``params``.  Names not in the registry are passed through.
        margin: Fraction of bound range to keep clear of each wall.
            Default :data:`BOUNDARY_INTERIOR_MARGIN`.

    Returns:
        ``(new_params, clamped_names)`` — a fresh array with values
        shifted inward and the list of names that were actually moved.
    """
    try:
        from heterodyne.config.parameter_registry import ParameterRegistry

        registry = ParameterRegistry()
    except ImportError:
        return np.asarray(params).copy(), []

    out = np.asarray(params, dtype=float).copy()
    clamped: list[str] = []

    for i, name in enumerate(parameter_names):
        try:
            info = registry[name]
        except KeyError:
            continue
        if info.log_space and info.min_bound > 0:
            log_range = np.log(info.max_bound / info.min_bound)
            factor = np.exp(margin * log_range)
            lo = info.min_bound * factor
            hi = info.max_bound / factor
        else:
            span = info.max_bound - info.min_bound
            lo = info.min_bound + margin * span
            hi = info.max_bound - margin * span
        if lo >= hi:
            # Degenerate bounds — leave the value alone rather than
            # collapse it to the midpoint.
            continue
        old = float(out[i])
        new = float(np.clip(old, lo, hi))
        if new != old:
            out[i] = new
            clamped.append(name)

    return out, clamped


def clamp_to_interior(
    result: NLSQResult,
    fixed_param_overrides: dict[str, float] | None = None,
    *,
    margin: float = BOUNDARY_INTERIOR_MARGIN,
) -> NLSQResult:
    """Return a copy of *result* with parameters shifted inward from hard bounds.

    NUTS step-size collapses when the chain initialises at a
    TruncatedNormal boundary.  Linear-scale parameters are clamped to
    ``[min_bound + margin, max_bound - margin]`` where margin is 5% of
    the bound range.  Log-space parameters (D0, v0) use a geometric
    margin so a linear fraction of a multi-decade range does not
    produce an absurd clamp target.  Both ensure the leapfrog step-size
    adaptation starts well away from the reflecting wall.

    ``fixed_param_overrides`` maps parameter names to values from the
    current model config's ``fixed_parameters``.  When provided, any
    NLSQ result value for a fixed parameter is replaced with the config
    value before bounds clamping.  This prevents a stale
    ``nlsq_data.npz`` (fitted in a prior run where the parameter was
    free) from propagating a superseded value into CMC initialisation,
    which can place the warm-start outside the reparameterised prior
    support and cause log-prior = −∞ → BFMI = 0 across all shards.

    Args:
        result: The NLSQ fit result to clamp.
        fixed_param_overrides: Optional map from parameter name to
            config-fixed value applied before clamping.
        margin: Fraction of bound range to keep clear of each wall.
            Default :data:`BOUNDARY_INTERIOR_MARGIN`.

    Returns:
        A new :class:`NLSQResult` with parameters clamped if any were
        out of the safe interior; the original ``result`` itself when
        no clamping was needed.
    """
    try:
        from heterodyne.config.parameter_registry import ParameterRegistry

        registry = ParameterRegistry()
    except ImportError:
        return result

    params = result.parameters.copy()
    clamped: list[str] = []

    for i, name in enumerate(result.parameter_names):
        # Apply fixed-parameter overrides before bounds clamping.
        if fixed_param_overrides and name in fixed_param_overrides:
            old = float(params[i])
            new = float(fixed_param_overrides[name])
            if abs(new - old) > 1e-12:
                logger.info(
                    "CMC init override: %s %.4g → %.4g "
                    "(fixed in current config; stale NLSQ value replaced)",
                    name,
                    old,
                    new,
                )
                params[i] = new
                clamped.append(name)
            continue
        try:
            info = registry[name]
        except KeyError:
            continue
        if info.log_space and info.min_bound > 0:
            log_range = np.log(info.max_bound / info.min_bound)
            factor = np.exp(margin * log_range)
            lo = info.min_bound * factor
            hi = info.max_bound / factor
        else:
            span = info.max_bound - info.min_bound
            lo = info.min_bound + margin * span
            hi = info.max_bound - margin * span
        if lo >= hi:
            continue
        old = float(params[i])
        new = float(np.clip(old, lo, hi))
        if new != old:
            _at_lb = abs(old - info.min_bound) < 1e-10 * max(abs(info.min_bound), 1)
            _at_ub = abs(old - info.max_bound) < 1e-10 * max(abs(info.max_bound), 1)
            if _at_lb or _at_ub:
                _side = "lower" if _at_lb else "upper"
                logger.warning(
                    "CMC init clamp: %s %.4g → %.4g (bounds [%.4g, %.4g]); "
                    "NLSQ hit the %s bound exactly — true mode may be outside "
                    "current bounds; expect high NUTS divergence rate. "
                    "Consider widening the %s bound in your config.",
                    name,
                    old,
                    new,
                    info.min_bound,
                    info.max_bound,
                    _side,
                    _side,
                )
            else:
                logger.warning(
                    "CMC init clamp: %s %.4g → %.4g (bounds [%.4g, %.4g]); "
                    "NUTS boundary-adjacent start prevented",
                    name,
                    old,
                    new,
                    info.min_bound,
                    info.max_bound,
                )
            clamped.append(name)
            params[i] = new

    if not clamped:
        return result

    return dataclasses.replace(result, parameters=params)
