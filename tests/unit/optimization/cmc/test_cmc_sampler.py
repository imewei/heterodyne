"""Tests for CMC sampler module (SamplingPlan, NUTSSampler, perturbation)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from heterodyne.optimization.cmc.sampler import (
    NUTSSampler,
    SamplingPlan,
    _perturb_init_params,
)

# ===========================================================================
# SamplingPlan
# ===========================================================================


class TestSamplingPlan:
    @pytest.mark.unit
    def test_default_values(self) -> None:
        plan = SamplingPlan()
        assert plan.num_warmup == 500
        assert plan.num_samples == 1000
        assert plan.num_chains == 4
        assert plan.target_accept == 0.8
        assert plan.max_tree_depth == 10
        assert plan.adapt_step_size is True
        assert plan.dense_mass is False
        assert plan.seed is None

    @pytest.mark.unit
    def test_frozen_immutable(self) -> None:
        plan = SamplingPlan()
        with pytest.raises(AttributeError):
            plan.num_warmup = 100  # type: ignore[misc]

    @pytest.mark.unit
    def test_custom_values(self) -> None:
        plan = SamplingPlan(
            num_warmup=200,
            num_samples=500,
            num_chains=2,
            target_accept=0.9,
            max_tree_depth=8,
            dense_mass=True,
            seed=42,
        )
        assert plan.num_warmup == 200
        assert plan.num_samples == 500
        assert plan.num_chains == 2
        assert plan.target_accept == 0.9
        assert plan.max_tree_depth == 8
        assert plan.dense_mass is True
        assert plan.seed == 42

    @pytest.mark.unit
    def test_effective_seed_returns_explicit(self) -> None:
        plan = SamplingPlan(seed=42)
        assert plan.effective_seed == 42

    @pytest.mark.unit
    def test_effective_seed_generates_when_none(self) -> None:
        plan = SamplingPlan(seed=None)
        seed = plan.effective_seed
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31

    @pytest.mark.unit
    def test_effective_seed_nondeterministic(self) -> None:
        """Two calls without explicit seed should usually differ."""
        plan = SamplingPlan(seed=None)
        seeds = {plan.effective_seed for _ in range(10)}
        # Extremely unlikely to get all identical from 10 random draws
        assert len(seeds) > 1

    # --- Validation ---

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("num_warmup", 0),
            ("num_warmup", -1),
            ("num_samples", 0),
            ("num_samples", -5),
            ("num_chains", 0),
            ("num_chains", -1),
            ("max_tree_depth", 0),
            ("max_tree_depth", -1),
        ],
    )
    def test_positive_int_validation(self, field: str, bad_value: int) -> None:
        with pytest.raises(ValueError, match=field):
            SamplingPlan(**{field: bad_value})

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_accept", [0.0, 0.09, 1.0, -0.5])
    def test_target_accept_validation(self, bad_accept: float) -> None:
        with pytest.raises(ValueError, match="target_accept"):
            SamplingPlan(target_accept=bad_accept)

    @pytest.mark.unit
    def test_target_accept_boundary_valid(self) -> None:
        """Edge values 0.1 and 0.99 should be accepted."""
        plan_lo = SamplingPlan(target_accept=0.1)
        assert plan_lo.target_accept == 0.1
        plan_hi = SamplingPlan(target_accept=0.99)
        assert plan_hi.target_accept == 0.99


# ===========================================================================
# _perturb_init_params
# ===========================================================================


class TestPerturbInitParams:
    @pytest.mark.unit
    def test_output_shape(self) -> None:
        init = {"D0_ref": jnp.array(1000.0), "alpha": jnp.array(0.5)}
        result = _perturb_init_params(init, num_chains=4, seed=0)
        assert set(result.keys()) == {"D0_ref", "alpha"}
        assert result["D0_ref"].shape == (4,)
        assert result["alpha"].shape == (4,)

    @pytest.mark.unit
    def test_perturbation_is_small(self) -> None:
        """Perturbation should be small relative to base value (scale=0.01)."""
        base_val = 1000.0
        init = {"x": jnp.array(base_val)}
        result = _perturb_init_params(init, num_chains=8, seed=42)
        deviations = jnp.abs(result["x"] - base_val)
        # With relative scale=0.01, max deviation ~ 0.01 * |base| * |N(0,1)|
        # ≈ 10 * |N(0,1)|, so max over 8 chains should be < 5% of base
        assert float(jnp.max(deviations)) < 0.05 * base_val

    @pytest.mark.unit
    def test_chains_differ(self) -> None:
        """Each chain should get a different perturbation."""
        init = {"x": jnp.array(500.0)}
        result = _perturb_init_params(init, num_chains=4, seed=0)
        values = [float(result["x"][i]) for i in range(4)]
        assert len(set(values)) == 4  # all unique

    @pytest.mark.unit
    def test_reproducible_with_same_seed(self) -> None:
        init = {"x": jnp.array(100.0)}
        r1 = _perturb_init_params(init, num_chains=2, seed=99)
        r2 = _perturb_init_params(init, num_chains=2, seed=99)
        assert jnp.allclose(r1["x"], r2["x"])

    @pytest.mark.unit
    def test_different_seed_gives_different_result(self) -> None:
        init = {"x": jnp.array(100.0)}
        r1 = _perturb_init_params(init, num_chains=2, seed=0)
        r2 = _perturb_init_params(init, num_chains=2, seed=1)
        assert not jnp.allclose(r1["x"], r2["x"])

    @pytest.mark.unit
    def test_broadcast_scalar(self) -> None:
        """Scalar init values should be broadcast to (num_chains,)."""
        init = {"a": jnp.float32(3.14)}
        result = _perturb_init_params(init, num_chains=3, seed=0)
        assert result["a"].shape == (3,)


# ===========================================================================
# NUTSSampler
# ===========================================================================


class TestNUTSSampler:
    @pytest.mark.unit
    def test_from_plan_creates_instance(self) -> None:
        """from_plan should return a NUTSSampler without errors."""
        import numpyro
        import numpyro.distributions as dist

        def simple_model():
            numpyro.sample("x", dist.Normal(0, 1))

        plan = SamplingPlan(num_warmup=10, num_samples=10, num_chains=1, seed=42)
        sampler = NUTSSampler.from_plan(plan, simple_model)
        assert isinstance(sampler, NUTSSampler)
        assert sampler._plan is plan

    @pytest.mark.unit
    def test_diagnostics_before_run_raises(self) -> None:
        """get_diagnostics before run() should raise RuntimeError."""
        import numpyro
        import numpyro.distributions as dist

        def simple_model():
            numpyro.sample("x", dist.Normal(0, 1))

        plan = SamplingPlan(num_warmup=10, num_samples=10, num_chains=1, seed=42)
        sampler = NUTSSampler.from_plan(plan, simple_model)
        with pytest.raises(RuntimeError, match="before calling run"):
            sampler.get_diagnostics()
