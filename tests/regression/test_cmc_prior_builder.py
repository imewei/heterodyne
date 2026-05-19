"""Regression tests for :class:`PriorBuilder` (gemini G2).

PriorBuilder centralises prior construction and refuses to operate
when ``parameter_registry`` and ``parameter_space._DEFAULT_PRIOR_SPECS``
disagree (CLAUDE.md Rule 9 — dual-prior sync).
"""

from __future__ import annotations

import numpyro.distributions as dist
import pytest

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.config.parameter_space import ParameterSpace
from heterodyne.optimization.cmc.priors import (
    PriorBuilder,
    build_default_priors,
    build_default_priors_via_builder,
)


class TestPriorBuilderSyncGate:
    def test_constructs_with_synced_registry_and_spec(self) -> None:
        # Current registry/spec pair must be in sync — PriorBuilder() is the
        # contract test for this.
        builder = PriorBuilder(DEFAULT_REGISTRY, use_log_space_priors=True)
        assert builder is not None

    def test_raises_on_registry_spec_desync_via_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Monkeypatch one _DEFAULT_PRIOR_SPECS entry so it disagrees with
        # the registry; PriorBuilder must refuse to construct.
        import heterodyne.config.parameter_space as ps_mod

        broken = dict(ps_mod._DEFAULT_PRIOR_SPECS)
        broken["D0_ref"] = (999.0, 999.0)  # registry has (1e4, 1e4)
        monkeypatch.setattr(ps_mod, "_DEFAULT_PRIOR_SPECS", broken)

        with pytest.raises(RuntimeError) as exc:
            PriorBuilder(DEFAULT_REGISTRY)
        msg = str(exc.value)
        assert "Dual-prior sync violation" in msg
        assert "Rule 9" in msg
        assert "D0_ref" in msg

    def test_default_registry_argument_falls_through(self) -> None:
        # registry=None uses DEFAULT_REGISTRY (and still runs the sync gate).
        builder = PriorBuilder(registry=None, use_log_space_priors=False)
        assert builder is not None


class TestPriorBuilderOutput:
    def test_log_space_branch_returns_lognormal_for_flagged_params(self) -> None:
        space = ParameterSpace()
        priors = PriorBuilder(use_log_space_priors=True).build(space)
        for name in ("D0_ref", "D0_sample", "v0"):
            assert isinstance(priors[name], dist.LogNormal), (
                f"{name}: expected LogNormal, got {type(priors[name]).__name__}"
            )

    def test_no_log_space_branch_returns_truncated_normal_family(self) -> None:
        space = ParameterSpace()
        priors = PriorBuilder(use_log_space_priors=False).build(space)
        for name in ("D0_ref", "D0_sample", "v0"):
            cls = type(priors[name]).__name__
            assert "Truncated" in cls or cls == "Normal", (
                f"{name}: expected TruncatedNormal-family, got {cls}"
            )
            assert not isinstance(priors[name], dist.LogNormal)


class TestPriorBuilderWrappers:
    def test_wrapper_matches_direct_builder_output(self) -> None:
        space = ParameterSpace()
        via_builder = PriorBuilder(use_log_space_priors=True).build(space)
        via_func = build_default_priors_via_builder(space, use_log_space_priors=True)
        assert set(via_func.keys()) == set(via_builder.keys())
        for name in via_builder:
            assert type(via_func[name]).__name__ == type(via_builder[name]).__name__

    def test_legacy_build_default_priors_still_works(self) -> None:
        # The module-level build_default_priors() routes its own sync check
        # via _verify_dual_prior_sync; output equivalence is what matters.
        space = ParameterSpace()
        legacy = build_default_priors(space, use_log_space_priors=True)
        via_builder = PriorBuilder(use_log_space_priors=True).build(space)
        assert set(legacy.keys()) == set(via_builder.keys())
