"""Tests for heterodyne/optimization/cmc/reparameterization.py.

Covers:
- compute_t_ref computation and edge cases
- Forward/inverse round-trip transforms
- Delta-method uncertainty propagation
- ReparamConfig behavior
- transform_to_physics_space vectorized back-transform
- Known forward-transform values
- Edge cases: alpha=0, very large/small D0
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.optimization.cmc.reparameterization import (
    POWER_LAW_PAIRS,
    ReparamConfig,
    compute_t_ref,
    reparam_to_physics_jax,
    transform_nlsq_to_reparam_space,
    transform_to_physics_space,
    transform_to_sampling_space,
)

# ============================================================================
# compute_t_ref
# ============================================================================


@pytest.mark.unit
class TestComputeTRef:
    """Tests for compute_t_ref()."""

    def test_basic_computation(self) -> None:
        """t_ref = sqrt(dt * t_max)."""
        t_ref = compute_t_ref(dt=1.0, t_max=100.0)
        assert t_ref == pytest.approx(10.0)

    def test_geometric_mean(self) -> None:
        """Verify geometric mean property."""
        dt, t_max = 0.01, 1000.0
        t_ref = compute_t_ref(dt, t_max)
        assert t_ref == pytest.approx(math.sqrt(dt * t_max))

    def test_invalid_dt_raises(self) -> None:
        """Negative dt without fallback raises."""
        with pytest.raises(ValueError, match="positive"):
            compute_t_ref(dt=-1.0, t_max=100.0)

    def test_invalid_dt_with_fallback(self) -> None:
        """Negative dt with fallback returns fallback."""
        t_ref = compute_t_ref(dt=-1.0, t_max=100.0, fallback_value=5.0)
        assert t_ref == 5.0

    def test_zero_t_max_raises(self) -> None:
        """Zero t_max without fallback raises."""
        with pytest.raises(ValueError, match="positive"):
            compute_t_ref(dt=1.0, t_max=0.0)

    def test_zero_t_max_with_fallback(self) -> None:
        """Zero t_max with fallback returns fallback."""
        t_ref = compute_t_ref(dt=1.0, t_max=0.0, fallback_value=1.0)
        assert t_ref == 1.0

    def test_small_values(self) -> None:
        """Very small dt/t_max still produce valid results."""
        t_ref = compute_t_ref(dt=1e-6, t_max=1e-3)
        assert t_ref > 0
        assert math.isfinite(t_ref)

    def test_zero_dt_with_fallback(self) -> None:
        """Zero dt with fallback returns fallback value."""
        t_ref = compute_t_ref(dt=0.0, t_max=100.0, fallback_value=7.0)
        assert t_ref == 7.0

    def test_zero_dt_without_fallback_raises(self) -> None:
        """Zero dt without fallback raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            compute_t_ref(dt=0.0, t_max=100.0)


# ============================================================================
# ReparamConfig
# ============================================================================


@pytest.mark.unit
class TestReparamConfig:
    """Tests for ReparamConfig dataclass."""

    def test_default_enables_all(self) -> None:
        """Default config enables all 3 pairs."""
        config = ReparamConfig(t_ref=10.0)
        assert len(config.enabled_pairs) == 3

    def test_enabled_pairs_d_ref_only(self) -> None:
        """Only D0_ref/alpha_ref pair when others disabled."""
        config = ReparamConfig(
            t_ref=10.0,
            enable_d_ref=True,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        pairs = config.enabled_pairs
        assert len(pairs) == 1
        assert pairs[0] == ("D0_ref", "alpha_ref")

    def test_enabled_pairs_d_sample_only(self) -> None:
        """Only D0_sample/alpha_sample pair when others disabled."""
        config = ReparamConfig(
            t_ref=10.0,
            enable_d_ref=False,
            enable_d_sample=True,
            enable_v_ref=False,
        )
        pairs = config.enabled_pairs
        assert len(pairs) == 1
        assert pairs[0] == ("D0_sample", "alpha_sample")

    def test_enabled_pairs_v_ref_only(self) -> None:
        """Only v0/beta pair when others disabled."""
        config = ReparamConfig(
            t_ref=10.0,
            enable_d_ref=False,
            enable_d_sample=False,
            enable_v_ref=True,
        )
        pairs = config.enabled_pairs
        assert len(pairs) == 1
        assert pairs[0] == ("v0", "beta")

    def test_disable_one_pair(self) -> None:
        """Can selectively disable pairs."""
        config = ReparamConfig(t_ref=10.0, enable_v_ref=False)
        assert len(config.enabled_pairs) == 2
        prefactors = [p for p, _ in config.enabled_pairs]
        assert "v0" not in prefactors

    def test_all_disabled(self) -> None:
        """All flags False yields empty enabled_pairs."""
        config = ReparamConfig(
            t_ref=10.0,
            enable_d_ref=False,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        assert config.enabled_pairs == []

    def test_is_reparameterized(self) -> None:
        """is_reparameterized correctly identifies pair members."""
        config = ReparamConfig(t_ref=10.0)
        assert config.is_reparameterized("D0_ref")
        assert config.is_reparameterized("alpha_ref")
        assert config.is_reparameterized("D0_sample")
        assert config.is_reparameterized("alpha_sample")
        assert config.is_reparameterized("v0")
        assert config.is_reparameterized("beta")
        assert not config.is_reparameterized("f0")
        assert not config.is_reparameterized("phi0")

    def test_is_reparameterized_disabled(self) -> None:
        """Disabled pairs are not reparameterized."""
        config = ReparamConfig(t_ref=10.0, enable_d_ref=False)
        assert not config.is_reparameterized("D0_ref")
        assert not config.is_reparameterized("alpha_ref")

    def test_get_reparam_name(self) -> None:
        """Reparam names follow expected pattern."""
        config = ReparamConfig(t_ref=10.0)
        assert config.get_reparam_name("D0_ref") == "log_D0_ref_at_tref"
        assert config.get_reparam_name("D0_sample") == "log_D0_sample_at_tref"
        assert config.get_reparam_name("v0") == "log_v0_at_tref"

    def test_frozen(self) -> None:
        """Config is immutable."""
        config = ReparamConfig(t_ref=10.0)
        with pytest.raises(AttributeError):
            config.t_ref = 20.0  # type: ignore[misc]


# ============================================================================
# Forward transform: known values
# ============================================================================


@pytest.mark.unit
class TestForwardTransform:
    """Test forward transform with known analytical values."""

    def test_known_d0_1000_alpha_0p5_tref_0p01(self) -> None:
        """D0=1000, alpha=0.5, t_ref=0.01 → log_D0_at_tref = log(1000) + 0.5*log(0.01)."""
        t_ref = 0.01
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        values = {"D0_ref": 1000.0, "alpha_ref": 0.5}
        unc = {"D0_ref": 10.0, "alpha_ref": 0.01}

        reparam_vals, _ = transform_nlsq_to_reparam_space(values, unc, t_ref, config)

        expected_log = math.log(1000.0) + 0.5 * math.log(0.01)
        np.testing.assert_allclose(
            reparam_vals["log_D0_ref_at_tref"],
            expected_log,
            rtol=1e-12,
        )

    def test_alpha_zero_pure_diffusion(self) -> None:
        """alpha=0 (pure diffusion): log_D0_at_tref = log(D0), independent of t_ref."""
        t_ref = 42.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        D0 = 500.0
        values = {"D0_ref": D0, "alpha_ref": 0.0}
        unc = {"D0_ref": 5.0, "alpha_ref": 0.01}

        reparam_vals, _ = transform_nlsq_to_reparam_space(values, unc, t_ref, config)

        # With alpha=0: log(D0 * t_ref^0) = log(D0)
        np.testing.assert_allclose(
            reparam_vals["log_D0_ref_at_tref"],
            math.log(D0),
            rtol=1e-12,
        )

    def test_very_large_d0(self) -> None:
        """Very large D0 produces finite reparameterized value."""
        t_ref = 1.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        D0 = 1e10
        values = {"D0_ref": D0, "alpha_ref": 0.5}
        unc = {"D0_ref": 1e8, "alpha_ref": 0.01}

        reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
            values,
            unc,
            t_ref,
            config,
        )

        assert math.isfinite(reparam_vals["log_D0_ref_at_tref"])
        assert reparam_unc["log_D0_ref_at_tref"] > 0

    def test_very_small_d0(self) -> None:
        """Very small D0 produces finite reparameterized value."""
        t_ref = 1.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        D0 = 1e-10
        values = {"D0_ref": D0, "alpha_ref": 0.5}
        unc = {"D0_ref": 1e-12, "alpha_ref": 0.01}

        reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
            values,
            unc,
            t_ref,
            config,
        )

        assert math.isfinite(reparam_vals["log_D0_ref_at_tref"])
        assert reparam_unc["log_D0_ref_at_tref"] > 0

    def test_non_reparameterized_passthrough(self) -> None:
        """Non-reparameterized params pass through unchanged."""
        t_ref = 10.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        values = {"D0_ref": 1.0, "alpha_ref": 0.5, "f0": 0.8, "phi0": 3.14}
        unc = {"D0_ref": 0.1, "alpha_ref": 0.05, "f0": 0.01, "phi0": 0.1}

        reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
            values,
            unc,
            t_ref,
            config,
        )

        # f0 and phi0 should pass through identically
        assert reparam_vals["f0"] == 0.8
        assert reparam_vals["phi0"] == 3.14
        assert reparam_unc["f0"] == 0.01
        assert reparam_unc["phi0"] == 0.1


# ============================================================================
# Round-trip transforms
# ============================================================================


@pytest.mark.unit
class TestRoundTripTransforms:
    """Test forward -> inverse round-trip consistency."""

    def test_single_pair_round_trip(self) -> None:
        """Forward then inverse recovers original values."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 0.5, "alpha_ref": 0.8, "f0": 1.0}
        nlsq_uncertainties = {"D0_ref": 0.05, "alpha_ref": 0.1, "f0": 0.01}

        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )

        # Forward: physics -> reparam
        reparam_vals, _ = transform_nlsq_to_reparam_space(
            nlsq_values,
            nlsq_uncertainties,
            t_ref,
            config,
        )

        # Inverse: reparam -> physics (via numpy arrays, as posterior samples would be)
        samples = {k: np.array([v]) for k, v in reparam_vals.items()}
        physics = transform_to_physics_space(samples, config)

        np.testing.assert_allclose(physics["D0_ref"][0], 0.5, rtol=1e-10)
        np.testing.assert_allclose(physics["alpha_ref"][0], 0.8, rtol=1e-10)
        np.testing.assert_allclose(physics["f0"][0], 1.0, rtol=1e-10)

    def test_all_three_pairs_round_trip(self) -> None:
        """Round-trip works for all 3 power-law pairs simultaneously."""
        t_ref = 5.0
        nlsq_values = {
            "D0_ref": 1.0,
            "alpha_ref": 0.5,
            "D0_sample": 0.3,
            "alpha_sample": 1.2,
            "v0": 0.01,
            "beta": 0.7,
            "D_offset_ref": 0.0,
            "f0": 1.0,
        }
        nlsq_uncertainties = {
            "D0_ref": 0.1,
            "alpha_ref": 0.05,
            "D0_sample": 0.03,
            "alpha_sample": 0.1,
            "v0": 0.001,
            "beta": 0.07,
            "D_offset_ref": 0.0,
            "f0": 0.01,
        }

        config = ReparamConfig(t_ref=t_ref)
        reparam_vals, _ = transform_nlsq_to_reparam_space(
            nlsq_values,
            nlsq_uncertainties,
            t_ref,
            config,
        )

        samples = {k: np.array([v]) for k, v in reparam_vals.items()}
        physics = transform_to_physics_space(samples, config)

        np.testing.assert_allclose(physics["D0_ref"][0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(physics["D0_sample"][0], 0.3, rtol=1e-10)
        np.testing.assert_allclose(physics["v0"][0], 0.01, rtol=1e-10)
        np.testing.assert_allclose(physics["alpha_ref"][0], 0.5, rtol=1e-10)
        np.testing.assert_allclose(physics["alpha_sample"][0], 1.2, rtol=1e-10)
        np.testing.assert_allclose(physics["beta"][0], 0.7, rtol=1e-10)

    def test_sampling_space_round_trip(self) -> None:
        """transform_to_sampling_space -> transform_to_physics_space round-trips."""
        t_ref = 10.0
        config = ReparamConfig(t_ref=t_ref)
        params = {
            "D0_ref": 2.0,
            "alpha_ref": 0.6,
            "v0": 0.05,
            "beta": 0.3,
            "D0_sample": 0.1,
            "alpha_sample": 1.0,
        }

        sampling = transform_to_sampling_space(params, config)
        samples = {k: np.array([v]) for k, v in sampling.items()}
        physics = transform_to_physics_space(samples, config)

        for key in ["D0_ref", "D0_sample", "v0"]:
            np.testing.assert_allclose(physics[key][0], params[key], rtol=1e-10)

    def test_round_trip_alpha_zero(self) -> None:
        """Round-trip with alpha=0 (pure diffusion) recovers D0 exactly."""
        t_ref = 100.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        D0 = 42.0
        values = {"D0_ref": D0, "alpha_ref": 0.0}
        unc = {"D0_ref": 1.0, "alpha_ref": 0.01}

        reparam_vals, _ = transform_nlsq_to_reparam_space(values, unc, t_ref, config)
        samples = {k: np.array([v]) for k, v in reparam_vals.items()}
        physics = transform_to_physics_space(samples, config)

        np.testing.assert_allclose(physics["D0_ref"][0], D0, rtol=1e-10)
        np.testing.assert_allclose(physics["alpha_ref"][0], 0.0, atol=1e-15)


# ============================================================================
# Delta-method uncertainty propagation
# ============================================================================


@pytest.mark.unit
class TestDeltaMethodUQ:
    """Test delta-method uncertainty propagation."""

    def test_uncertainty_positive(self) -> None:
        """Reparam uncertainties are always positive."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 0.5}
        nlsq_uncertainties = {"D0_ref": 0.1, "alpha_ref": 0.05}

        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        _, reparam_unc = transform_nlsq_to_reparam_space(
            nlsq_values,
            nlsq_uncertainties,
            t_ref,
            config,
        )

        for name, unc in reparam_unc.items():
            assert unc > 0, f"Uncertainty for {name} should be positive, got {unc}"

    def test_zero_uncertainty_gets_floor(self) -> None:
        """Zero NLSQ uncertainty gets floored to 1e-6."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 0.5}
        nlsq_uncertainties = {"D0_ref": 0.0, "alpha_ref": 0.0}

        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        _, reparam_unc = transform_nlsq_to_reparam_space(
            nlsq_values,
            nlsq_uncertainties,
            t_ref,
            config,
        )

        assert reparam_unc["log_D0_ref_at_tref"] >= 1e-6
        assert reparam_unc["alpha_ref"] >= 1e-6

    def test_delta_method_formula(self) -> None:
        """Verify the delta-method formula explicitly."""
        t_ref = 10.0
        D0, alpha = 1.0, 0.5
        sigma_D0, sigma_alpha = 0.1, 0.05

        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        _, unc = transform_nlsq_to_reparam_space(
            {"D0_ref": D0, "alpha_ref": alpha},
            {"D0_ref": sigma_D0, "alpha_ref": sigma_alpha},
            t_ref,
            config,
        )

        # Manual: sqrt((sigma_D0/D0)^2 + (log(t_ref)*sigma_alpha)^2)
        expected = math.sqrt(
            (sigma_D0 / D0) ** 2 + (math.log(t_ref) * sigma_alpha) ** 2
        )
        np.testing.assert_allclose(
            unc["log_D0_ref_at_tref"],
            expected,
            rtol=1e-10,
        )

    def test_uncertainty_finite_for_all_pairs(self) -> None:
        """Delta-method uncertainties are finite for all 3 pairs."""
        t_ref = 0.01
        config = ReparamConfig(t_ref=t_ref)
        values = {
            "D0_ref": 1000.0,
            "alpha_ref": 0.5,
            "D0_sample": 500.0,
            "alpha_sample": 0.3,
            "v0": 1e3,
            "beta": 0.7,
        }
        unc_in = {
            "D0_ref": 100.0,
            "alpha_ref": 0.05,
            "D0_sample": 50.0,
            "alpha_sample": 0.03,
            "v0": 100.0,
            "beta": 0.07,
        }

        _, unc_out = transform_nlsq_to_reparam_space(values, unc_in, t_ref, config)

        for name in ["log_D0_ref_at_tref", "log_D0_sample_at_tref", "log_v0_at_tref"]:
            assert math.isfinite(unc_out[name]), f"{name} uncertainty not finite"
            assert unc_out[name] > 0, f"{name} uncertainty not positive"


# ============================================================================
# Edge cases
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case handling."""

    def test_missing_pair_member_skips(self) -> None:
        """If one member of a pair is missing, that pair is skipped."""
        t_ref = 10.0
        config = ReparamConfig(t_ref=t_ref)

        # Only D0_ref, no alpha_ref
        values = {"D0_ref": 1.0, "f0": 1.0}
        unc = {"D0_ref": 0.1, "f0": 0.01}

        reparam_vals, _ = transform_nlsq_to_reparam_space(values, unc, t_ref, config)

        # D0_ref should pass through since alpha_ref is missing
        assert "D0_ref" in reparam_vals
        assert "log_D0_ref_at_tref" not in reparam_vals

    def test_negative_prefactor_handled(self) -> None:
        """Negative prefactor uses log(|a0|) fallback."""
        t_ref = 10.0
        config = ReparamConfig(
            t_ref=t_ref,
            enable_d_sample=False,
            enable_v_ref=False,
        )
        values = {"D0_ref": -0.5, "alpha_ref": 0.8}
        unc = {"D0_ref": 0.1, "alpha_ref": 0.05}

        reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
            values,
            unc,
            t_ref,
            config,
        )

        # Should still produce a finite result
        assert math.isfinite(reparam_vals["log_D0_ref_at_tref"])
        assert reparam_unc["log_D0_ref_at_tref"] > 0

    def test_vectorized_physics_transform(self) -> None:
        """transform_to_physics_space works on arrays of samples."""
        config = ReparamConfig(t_ref=10.0)
        n_samples = 500
        rng = np.random.default_rng(42)

        samples = {
            "log_D0_ref_at_tref": rng.normal(0.0, 0.1, n_samples),
            "alpha_ref": rng.normal(0.5, 0.05, n_samples),
            "log_D0_sample_at_tref": rng.normal(-1.0, 0.2, n_samples),
            "alpha_sample": rng.normal(1.0, 0.1, n_samples),
            "log_v0_at_tref": rng.normal(-3.0, 0.5, n_samples),
            "beta": rng.normal(0.7, 0.05, n_samples),
            "f0": np.ones(n_samples),
        }

        physics = transform_to_physics_space(samples, config)

        assert "D0_ref" in physics
        assert "D0_sample" in physics
        assert "v0" in physics
        assert "f0" in physics
        assert len(physics["D0_ref"]) == n_samples
        assert np.all(physics["D0_ref"] > 0)  # exp() is always positive

    @pytest.mark.requires_jax
    def test_reparam_to_physics_jax(self) -> None:
        """JAX back-transform produces correct values."""
        t_ref = 10.0
        D0_true = 2.0
        alpha_true = 0.5
        log_at_tref = jnp.log(jnp.float64(D0_true)) + alpha_true * jnp.log(
            jnp.float64(t_ref)
        )

        D0_recovered = reparam_to_physics_jax(
            log_at_tref,
            jnp.float64(alpha_true),
            t_ref,
        )
        np.testing.assert_allclose(float(D0_recovered), D0_true, rtol=1e-10)

    def test_physics_transform_missing_reparam_sample(self) -> None:
        """Missing reparam sample (log_X_at_tref) in samples is skipped gracefully."""
        config = ReparamConfig(t_ref=10.0)
        samples = {
            "alpha_ref": np.array([0.5]),
            "f0": np.array([1.0]),
        }
        physics = transform_to_physics_space(samples, config)
        assert "f0" in physics
        assert "alpha_ref" in physics
        assert "D0_ref" not in physics

    def test_delta_method_large_tref_finite(self) -> None:
        """Delta-method stays finite for large t_ref (common in slow dynamics)."""
        t_ref = 1e6
        config = ReparamConfig(t_ref=t_ref, enable_d_sample=False, enable_v_ref=False)
        _, unc = transform_nlsq_to_reparam_space(
            {"D0_ref": 1.0, "alpha_ref": 0.5},
            {"D0_ref": 0.1, "alpha_ref": 0.05},
            t_ref,
            config,
        )
        assert math.isfinite(unc["log_D0_ref_at_tref"])
        assert unc["log_D0_ref_at_tref"] > 0
        # log(1e6) ~ 13.8, so uncertainty should be amplified but still reasonable
        assert unc["log_D0_ref_at_tref"] < 10.0

    def test_power_law_pairs_constant(self) -> None:
        """POWER_LAW_PAIRS matches the 3 heterodyne pairs."""
        assert len(POWER_LAW_PAIRS) == 3
        prefactors = [p for p, _ in POWER_LAW_PAIRS]
        assert prefactors == ["D0_ref", "D0_sample", "v0"]
