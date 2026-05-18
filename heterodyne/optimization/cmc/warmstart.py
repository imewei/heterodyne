"""Public warm-start helpers for the CMC pipeline (P2-a follow-up).

Originally these helpers lived as private functions in
``heterodyne.cli.optimization_runner``; that meant Python API users
who invoked :func:`heterodyne.optimization.cmc.fit_cmc_jax` directly
(without going through the CLI) silently lost the boundary-clamp
protection that ``optimization_runner`` applies to NLSQ warm-starts.

This module promotes them to a stable public surface:

- :func:`clamp_to_interior` — wraps an
  :class:`~heterodyne.optimization.nlsq.results.NLSQResult` and returns a
  copy with parameters shifted away from hard bounds.
- :func:`clamp_params_to_interior` — the lower-level array path; takes
  ``(params, parameter_names)`` for callers without an NLSQResult.

The CLI now imports these public names; the legacy private alias is
preserved at the CLI to keep existing call sites working.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)


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
