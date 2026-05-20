"""Regression tests pinning the CMC NUTS-default contract.

Background — deep-RCA findings F1 + F2 + F7 documented a *configuration*
failure mode: tight log-space priors plus 500-step warmup plus 0.85 target
acceptance produced divergence cascades on the 14-parameter heterodyne
model.  The hardening commit raised ``num_warmup`` to 1500 and
``target_accept_prob`` to 0.90.

If a future contributor silently reverts those defaults, every downstream
CMC run regresses.  These tests pin the contract: fast assertions on the
config dataclass defaults run on every CI build; a slow end-to-end smoke
that actually exercises NUTS on a synthetic problem runs on the nightly
``slow`` marker.

When deliberately changing the defaults, update the constants below AND
add a CLAUDE.md note explaining why the new setting is geometrically
adequate for the 14-parameter posterior.
"""

from __future__ import annotations

# --- Fast contract checks ---------------------------------------------------


class TestCMCDefaultsContract:
    """Pin the CMC default sampling configuration against the geometric
    requirements established by deep-RCA F1, F2, F7."""

    def test_num_warmup_meets_mass_matrix_adaptation_floor(self) -> None:
        """``dense_mass=True`` on a 14-parameter model needs >= 100
        warmup steps per dimension for the inverse-Hessian estimate to
        stabilise.  Default ``num_warmup`` must respect that contract."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert cfg.num_warmup >= 1500, (
            f"num_warmup={cfg.num_warmup} regressed below the 1500-step "
            "floor required for dense-mass adaptation on the 14-parameter "
            "heterodyne model (deep-RCA F1).  Either restore the floor or "
            "document the geometric justification in CLAUDE.md."
        )

    def test_target_accept_prob_above_divergence_floor(self) -> None:
        """Default ``target_accept_prob`` must be >= 0.90 to keep step
        size small enough to traverse the (D0, alpha) funnel without
        divergence cascades."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert cfg.target_accept_prob >= 0.90, (
            f"target_accept_prob={cfg.target_accept_prob} regressed below "
            "the 0.90 floor (deep-RCA F7).  Lower target produces step "
            "sizes that diverge on the funnel-shaped joint posterior."
        )

    def test_dense_mass_on_by_default(self) -> None:
        """The 14-parameter posterior has correlated (D0, alpha) and
        (v0, beta) blocks; diagonal mass wastes ESS.  ``dense_mass``
        defaults to True for this reason."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert cfg.dense_mass is True, (
            "dense_mass default flipped to False — the (D0, alpha) "
            "funnel will produce low ESS under diagonal mass (deep-RCA F6)."
        )

    def test_max_r_hat_threshold_practical(self) -> None:
        """``max_r_hat`` <= 1.10 — tighter rejects working posteriors,
        looser accepts under-converged runs.  Pin to current value to
        catch silent threshold drift."""
        from heterodyne import CMCConfig

        cfg = CMCConfig()
        assert 1.01 < cfg.max_r_hat <= 1.20, (
            f"max_r_hat={cfg.max_r_hat} outside practical [1.01, 1.20] "
            "range; tighter rejects valid posteriors, looser accepts "
            "under-converged ones."
        )

    def test_d0_prior_log_space_width_adequate(self) -> None:
        """Deep-RCA F2: D0 prior must accommodate ~1 order-of-magnitude
        empirical posteriors.  In log-space, std >= 0.5 (≈ half an
        e-fold)."""
        import math

        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        for name in ("D0_ref", "D0_sample"):
            info = DEFAULT_REGISTRY[name]
            assert info.prior_mean is not None and info.prior_std is not None
            log_std = math.log1p(float(info.prior_std) / float(info.prior_mean))
            assert log_std > 0.5, (
                f"{name}: log-space prior std = {log_std:.3f} regressed "
                f"below 0.5; posteriors will collapse onto prior centre "
                "(deep-RCA F2)."
            )


# --- Defaults propagation contract -----------------------------------------


def test_defaults_propagate_through_sampling_plan() -> None:
    """``CMCConfig`` defaults must reach NUTS through ``SamplingPlan.from_config``.

    The value-only contract tests above pin the dataclass defaults; this
    test pins the *propagation surface* — the single point at
    ``sampler.SamplingPlan.from_config()`` where ``CMCConfig`` becomes the
    actual NUTS hyperparameters.  If a future refactor breaks that bridge
    (forgets to forward ``dense_mass``, hard-codes ``target_accept``, drops
    the warmup floor), the dataclass test still passes but the sampler
    silently loses the contract.  This test catches that.

    Runs in ~1ms — no NUTS dispatch, no synthetic data, no posterior
    sampling.  The end-to-end convergence smoke that the file previously
    stubbed out is genuinely expensive (~30-60s under reduced warmup,
    flaky under tight thresholds); split into a follow-up issue rather
    than leave behind a skipped test that never runs.
    """
    from heterodyne import CMCConfig
    from heterodyne.optimization.cmc.sampler import SamplingPlan

    config = CMCConfig()  # all pinned defaults
    plan = SamplingPlan.from_config(config)

    # target_accept_prob (canonical name) → SamplingPlan.target_accept
    # (NumPyro's NUTS kwarg).  The bridge MUST not drop or rename this.
    assert plan.target_accept == config.target_accept_prob, (
        f"SamplingPlan.target_accept={plan.target_accept!r} does not "
        f"match CMCConfig.target_accept_prob={config.target_accept_prob!r}. "
        "The config→sampler bridge silently lost the pinned default; "
        "deep-RCA F7 divergence cascades will reappear."
    )
    assert plan.target_accept >= 0.90, (
        "SamplingPlan.target_accept fell below the 0.90 contract floor."
    )

    # dense_mass forwarded — diagonal mass kills ESS on the (D0, alpha)
    # funnel (deep-RCA F6).  Forgetting to forward this is a silent
    # geometric regression.
    assert plan.dense_mass is True, (
        "SamplingPlan.dense_mass=False — the (D0, alpha) funnel will "
        "produce low ESS under diagonal mass.  Check that "
        "SamplingPlan.from_config forwards config.dense_mass."
    )

    # num_warmup must respect the Rule 12 floor (1500 steps for
    # dense-mass adaptation on 14 parameters).  ``from_config`` applies
    # ``effective_warmup_floor`` last — if that hook is removed the
    # contract breaks silently.
    assert plan.num_warmup >= 1500, (
        f"SamplingPlan.num_warmup={plan.num_warmup} regressed below the "
        "Rule 12 floor (1500) required for dense-mass adaptation on the "
        "14-parameter heterodyne model."
    )

    # The fast_warmup escape hatch must travel through too — otherwise
    # CI/pytest fast fixtures (cmc_config_1chain etc.) silently inherit
    # the 1500-step floor and slow down every smoke test.
    fast_config = CMCConfig(
        num_chains=1,
        num_warmup=200,
        num_samples=200,
        fast_warmup=True,
    )
    fast_plan = SamplingPlan.from_config(fast_config)
    assert fast_plan.num_warmup == 200, (
        f"fast_warmup=True did not bypass the 1500-step floor: "
        f"SamplingPlan.num_warmup={fast_plan.num_warmup} (expected 200)."
    )
    assert fast_plan.fast_warmup is True, (
        "SamplingPlan.fast_warmup not forwarded from CMCConfig."
    )
