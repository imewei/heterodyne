"""Regression tests for Rule 12 NUTS warmup floor (Codex finding C3).

When ``dense_mass=True`` the dense mass-matrix adaptation needs at least
:data:`heterodyne.optimization.cmc.config.DENSE_MASS_WARMUP_FLOOR` warmup
steps for the 14-parameter heterodyne model.  Five independent code paths
must respect this floor:

1. ``CMCConfig.validate()`` — flags ``num_warmup < floor`` configs.
2. ``CMCConfig.get_adaptive_sample_counts()`` — adaptive scaling clamp.
3. ``SamplingPlan.from_config()`` — adaptive sqrt scaling.
4. ``SamplingPlan.for_shard()`` — per-shard scale-down.
5. ``AdaptiveSamplingPlan.get_plan()`` — wrapper sqrt scaling.

A ``fast_warmup=True`` escape hatch bypasses the floor for CI fast-mode.
"""

from __future__ import annotations

from heterodyne.optimization.cmc.config import (
    DENSE_MASS_WARMUP_FLOOR,
    CMCConfig,
    effective_warmup_floor,
)
from heterodyne.optimization.cmc.sampler import (
    AdaptiveSamplingPlan,
    SamplingPlan,
)

# ---------------------------------------------------------------------------
# effective_warmup_floor helper
# ---------------------------------------------------------------------------


class TestEffectiveWarmupFloor:
    """The shared helper must behave consistently regardless of caller."""

    def test_clamps_below_floor_when_dense_mass(self) -> None:
        assert effective_warmup_floor(100, dense_mass=True) == DENSE_MASS_WARMUP_FLOOR

    def test_passes_through_when_above_floor(self) -> None:
        assert effective_warmup_floor(2000, dense_mass=True) == 2000

    def test_passes_through_when_dense_mass_false(self) -> None:
        assert effective_warmup_floor(100, dense_mass=False) == 100

    def test_fast_warmup_bypasses_floor(self) -> None:
        assert effective_warmup_floor(50, dense_mass=True, fast_warmup=True) == 50

    def test_floor_value_is_1500(self) -> None:
        """Rule 12: heterodyne CMC defaults document this floor at 1500."""
        assert DENSE_MASS_WARMUP_FLOOR == 1500


# ---------------------------------------------------------------------------
# Site 1: CMCConfig.validate()
# ---------------------------------------------------------------------------


class TestCMCConfigValidate:
    def test_validate_rejects_low_warmup_with_dense_mass(self) -> None:
        cfg = CMCConfig(
            enable="never",  # avoid downstream side-effects
            num_warmup=500,
            num_samples=200,
            dense_mass=True,
            fast_warmup=False,
        )
        # ``validate()`` returns a list of error strings (does not raise).
        errors = cfg.validate()
        rule12 = [
            e for e in errors if "Rule 12" in e or "1500" in e or "dense_mass" in e
        ]
        assert rule12, f"Expected Rule 12 violation; got errors={errors!r}"

    def test_validate_accepts_low_warmup_when_dense_mass_false(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=500,
            num_samples=200,
            dense_mass=False,
            fast_warmup=False,
        )
        errors = cfg.validate()
        # No warmup-floor errors expected when dense_mass=False.
        assert not [e for e in errors if "Rule 12" in e or "1500" in e]

    def test_validate_accepts_low_warmup_with_fast_warmup(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=200,
            num_samples=200,
            dense_mass=True,
            fast_warmup=True,
        )
        errors = cfg.validate()
        # fast_warmup=True must bypass the Rule 12 gate.
        assert not [e for e in errors if "Rule 12" in e or "1500" in e]

    def test_validate_accepts_floor_warmup_with_dense_mass(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=200,
            dense_mass=True,
            fast_warmup=False,
        )
        errors = cfg.validate()
        assert not [e for e in errors if "Rule 12" in e or "1500" in e]


# ---------------------------------------------------------------------------
# Site 2: CMCConfig.get_adaptive_sample_counts
# ---------------------------------------------------------------------------


class TestAdaptiveSampleCounts:
    def test_scaler_respects_floor_for_small_shards(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            dense_mass=True,
            adaptive_sampling=True,
            fast_warmup=False,
        )
        warmup, _samples = cfg.get_adaptive_sample_counts(shard_size=100)
        assert warmup >= DENSE_MASS_WARMUP_FLOOR

    def test_scaler_bypass_when_fast_warmup(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            dense_mass=True,
            adaptive_sampling=True,
            fast_warmup=True,
        )
        warmup, _samples = cfg.get_adaptive_sample_counts(shard_size=100)
        # Without the floor, small shard collapses warmup below 1500.
        assert warmup < DENSE_MASS_WARMUP_FLOOR


# ---------------------------------------------------------------------------
# Site 3: SamplingPlan.from_config
# ---------------------------------------------------------------------------


class TestSamplingPlanFromConfig:
    def test_from_config_with_small_n_data_respects_floor(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            dense_mass=True,
            adaptive_sampling=True,
            fast_warmup=False,
        )
        plan = SamplingPlan.from_config(cfg, n_data=100)
        assert plan.num_warmup >= DENSE_MASS_WARMUP_FLOOR
        assert plan.dense_mass is True

    def test_from_config_propagates_fast_warmup(self) -> None:
        cfg = CMCConfig(
            enable="never",
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            dense_mass=True,
            adaptive_sampling=True,
            fast_warmup=True,
        )
        plan = SamplingPlan.from_config(cfg, n_data=100)
        assert plan.fast_warmup is True
        # With fast_warmup, sqrt-scaled warmup is allowed below floor.
        assert plan.num_warmup < DENSE_MASS_WARMUP_FLOOR


# ---------------------------------------------------------------------------
# Site 4: SamplingPlan.for_shard
# ---------------------------------------------------------------------------


class TestSamplingPlanForShard:
    def test_for_shard_respects_floor(self) -> None:
        base = SamplingPlan(
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            num_chains=4,
            dense_mass=True,
            fast_warmup=False,
        )
        shard_plan = base.for_shard(shard_size=10, full_size=10000)
        assert shard_plan.num_warmup >= DENSE_MASS_WARMUP_FLOOR
        assert shard_plan.fast_warmup is False

    def test_for_shard_bypass_when_fast_warmup(self) -> None:
        base = SamplingPlan(
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            num_chains=4,
            dense_mass=True,
            fast_warmup=True,
        )
        shard_plan = base.for_shard(shard_size=10, full_size=10000)
        assert shard_plan.num_warmup < DENSE_MASS_WARMUP_FLOOR
        assert shard_plan.fast_warmup is True

    def test_for_shard_when_dense_mass_false(self) -> None:
        base = SamplingPlan(
            num_warmup=500,
            num_samples=1500,
            num_chains=4,
            dense_mass=False,
            fast_warmup=False,
        )
        shard_plan = base.for_shard(shard_size=10, full_size=10000)
        # No floor when dense_mass=False.
        assert shard_plan.num_warmup < DENSE_MASS_WARMUP_FLOOR


# ---------------------------------------------------------------------------
# Site 5: AdaptiveSamplingPlan.get_plan
# ---------------------------------------------------------------------------


class TestAdaptiveSamplingPlanGetPlan:
    def test_get_plan_respects_floor(self) -> None:
        base = SamplingPlan(
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            num_chains=4,
            dense_mass=True,
            fast_warmup=False,
        )
        adaptive = AdaptiveSamplingPlan(base_plan=base, shard_size=100, n_params=14)
        plan = adaptive.get_plan()
        assert plan.num_warmup >= DENSE_MASS_WARMUP_FLOOR

    def test_get_plan_bypass_with_fast_warmup(self) -> None:
        base = SamplingPlan(
            num_warmup=DENSE_MASS_WARMUP_FLOOR,
            num_samples=1500,
            num_chains=4,
            dense_mass=True,
            fast_warmup=True,
        )
        adaptive = AdaptiveSamplingPlan(base_plan=base, shard_size=100, n_params=14)
        plan = adaptive.get_plan()
        # Adaptive sqrt scaling on tiny shard collapses warmup; fast_warmup honoured.
        assert plan.num_warmup < DENSE_MASS_WARMUP_FLOOR
