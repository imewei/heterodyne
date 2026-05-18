"""Centralised prior construction for heterodyne CMC (codex/gemini G2).

This module factors the prior-construction logic that used to live in
:mod:`heterodyne.optimization.cmc.priors` into a single class,
:class:`PriorBuilder`, with two important properties:

1. **Construction-time sync gate** — Rule 9 of ``CLAUDE.md`` says the
   registry's ``prior_mean``/``prior_std`` and
   ``parameter_space._DEFAULT_PRIOR_SPECS`` (``(loc, scale)``) must
   agree.  The legacy implementation enforced this via a pytest sentinel
   (``test_dual_prior_specs_in_sync``).  ``PriorBuilder`` upgrades that
   to a runtime check: the constructor walks both data sources and
   raises ``RuntimeError`` with a descriptive message on any drift.  A
   ``PriorBuilder`` instance therefore cannot exist in a desynced state.

2. **Single dispatch** — ``PriorBuilder.build(space)`` returns the same
   ``dict[name -> dist.Distribution]`` the legacy
   ``build_default_priors(space, use_log_space_priors=...)`` returned.
   The module-level wrappers in :mod:`heterodyne.optimization.cmc.priors`
   are reduced to ~3-line shims that instantiate this class and
   forward.  Existing call sites do not need to change.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpyro.distributions as dist

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.optimization.cmc.priors import (
    build_default_priors as _legacy_build_default_priors,
)
from heterodyne.optimization.cmc.priors import (
    build_log_space_priors as _legacy_build_log_space_priors,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_registry import ParameterRegistry
    from heterodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


_DEFAULT_SYNC_REL_TOL: float = 1e-6
_DEFAULT_SYNC_ABS_TOL: float = 1e-9


class PriorBuilder:
    """Construct NumPyro priors from a parameter registry.

    Args:
        registry: Parameter registry (defaults to
            :data:`heterodyne.config.parameter_registry.DEFAULT_REGISTRY`).
        use_log_space_priors: When True (default), parameters flagged
            ``log_space=True`` in the registry are returned as LogNormal
            distributions instead of TruncatedNormal — mass-matrix
            conditioning for multi-decade prefactors (D0_ref, D0_sample,
            v0).  See codex S1.

    Raises:
        RuntimeError: When the registry and
            ``parameter_space._DEFAULT_PRIOR_SPECS`` disagree on
            ``(prior_mean, prior_std)`` vs ``(loc, scale)``.  This is
            Rule 9 from ``CLAUDE.md`` — the dual-prior system must
            stay in sync.  Tolerance is ``rel_tol=1e-6, abs_tol=1e-9``.
    """

    def __init__(
        self,
        registry: ParameterRegistry | None = None,
        use_log_space_priors: bool = True,
    ) -> None:
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._use_log = bool(use_log_space_priors)
        self._verify_registry_spec_sync()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, param_space: ParameterSpace) -> dict[str, dist.Distribution]:
        """Build priors for the varying parameters in *param_space*.

        Args:
            param_space: ParameterSpace defining which parameters vary
                and their physical bounds.

        Returns:
            Dictionary mapping each varying parameter name to its
            NumPyro distribution.  D0_ref/D0_sample/v0 are LogNormal
            when ``use_log_space_priors=True``, TruncatedNormal-family
            otherwise; remaining parameters are always TruncatedNormal.
        """
        return _legacy_build_default_priors(
            param_space,
            registry=self._registry,
            use_log_space_priors=self._use_log,
        )

    # ------------------------------------------------------------------
    # Construction-time invariant (CLAUDE.md Rule 9)
    # ------------------------------------------------------------------

    def _verify_registry_spec_sync(self) -> None:
        """Walk registry vs ``_DEFAULT_PRIOR_SPECS``; raise on drift.

        The two sources are:

        * ``parameter_registry.py`` — per-``ParameterInfo``
          ``prior_mean``/``prior_std`` (the source of truth for CMC
          prior construction via :func:`build_default_priors`).

        * ``parameter_space.py`` — ``_DEFAULT_PRIOR_SPECS`` dict of
          ``(loc, scale)`` per name (the source used by
          :func:`ParameterSpace._default_prior` to seed
          TruncatedNormal/Beta priors during ``ParameterSpace.__init__``).

        Both must agree, or the same parameter would receive different
        priors depending on which code path constructed it.  We use
        ``math.isclose(rel_tol=1e-6, abs_tol=1e-9)`` — tight enough to
        catch real desync (someone forgot to update one file),
        generous enough to survive ``0.1 + 0.2`` rounding artefacts.
        """
        # Lazy import to avoid an import cycle at module load:
        # parameter_space.py imports parameter_registry; this module
        # imports both at runtime so we don't add to the import graph.
        from heterodyne.config.parameter_space import _DEFAULT_PRIOR_SPECS

        mismatches: list[str] = []

        for name, (spec_loc, spec_scale) in _DEFAULT_PRIOR_SPECS.items():
            try:
                info = self._registry[name]
            except KeyError:
                mismatches.append(
                    f"{name!r}: present in _DEFAULT_PRIOR_SPECS "
                    f"({spec_loc}, {spec_scale}) but absent from the registry"
                )
                continue
            if info.prior_mean is None or info.prior_std is None:
                mismatches.append(
                    f"{name!r}: registry has prior_mean={info.prior_mean!r}, "
                    f"prior_std={info.prior_std!r}; spec wants "
                    f"({spec_loc}, {spec_scale})"
                )
                continue
            mean_ok = math.isclose(
                info.prior_mean,
                spec_loc,
                rel_tol=_DEFAULT_SYNC_REL_TOL,
                abs_tol=_DEFAULT_SYNC_ABS_TOL,
            )
            std_ok = math.isclose(
                info.prior_std,
                spec_scale,
                rel_tol=_DEFAULT_SYNC_REL_TOL,
                abs_tol=_DEFAULT_SYNC_ABS_TOL,
            )
            if not (mean_ok and std_ok):
                mismatches.append(
                    f"{name!r}: registry=({info.prior_mean}, {info.prior_std}), "
                    f"spec=({spec_loc}, {spec_scale})"
                )

        spec_names = set(_DEFAULT_PRIOR_SPECS.keys())
        for name in self._registry:
            if name in spec_names:
                continue
            info = self._registry[name]
            if info.prior_mean is not None or info.prior_std is not None:
                mismatches.append(
                    f"{name!r}: registry has prior "
                    f"(prior_mean={info.prior_mean}, prior_std={info.prior_std}) "
                    "but is missing from _DEFAULT_PRIOR_SPECS"
                )

        if mismatches:
            raise RuntimeError(
                "Dual-prior sync violation (CLAUDE.md Rule 9). "
                "parameter_registry.py and parameter_space._DEFAULT_PRIOR_SPECS "
                "must agree on (prior_mean, prior_std) vs (loc, scale) per "
                "parameter, within rel_tol=1e-6 / abs_tol=1e-9. Mismatches:\n  - "
                + "\n  - ".join(mismatches)
            )

        logger.debug(
            "PriorBuilder: registry/spec sync verified (%d parameters)",
            len(_DEFAULT_PRIOR_SPECS),
        )


# ---------------------------------------------------------------------------
# Module-level wrappers exposed for direct API parity
# ---------------------------------------------------------------------------


def build_default_priors_via_builder(
    param_space: ParameterSpace,
    registry: ParameterRegistry | None = None,
    use_log_space_priors: bool = True,
) -> dict[str, dist.Distribution]:
    """Functional shim around :class:`PriorBuilder` for callers that
    want a one-shot factory.  Constructing the builder runs the sync
    gate, so each call validates the dual-prior invariant.
    """
    return PriorBuilder(registry, use_log_space_priors).build(param_space)


def build_log_space_priors_via_builder(
    param_names: list[str],
    registry: ParameterRegistry | None = None,
) -> dict[str, dist.Distribution]:
    """Functional shim for the legacy log-space-only helper.

    The construction-time gate runs here too — calling this entry point
    is enough to fail loud on a desynced registry/spec pair.
    """
    PriorBuilder(registry, use_log_space_priors=True)
    return _legacy_build_log_space_priors(param_names, registry=registry)
