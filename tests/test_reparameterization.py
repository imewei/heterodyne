"""Tests for heterodyne/optimization/cmc/reparameterization.py.

Covers:
- compute_t_ref computation and edge cases
- Forward/inverse round-trip transforms
- Delta-method uncertainty propagation
- ReparamConfig behavior
- transform_to_physics_space vectorized back-transform
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


class TestComputeTRef:
    """Tests for compute_t_ref()."""

    @pytest.mark.unit
    def test_basic_computation(self) -> None:
        """t_ref = sqrt(dt * t_max)."""
        t_ref = compute_t_ref(dt=1.0, t_max=100.0)
        assert t_ref == pytest.approx(10.0)

    @pytest.mark.unit
    def test_geometric_mean(self) -> None:
        """Verify geometric mean property."""
        dt, t_max = 0.01, 1000.0
        t_ref = compute_t_ref(dt, t_max)
        assert t_ref == pytest.approx(math.sqrt(dt * t_max))

    @pytest.mark.unit
    def test_invalid_dt_raises(self) -> None:
        """Negative dt without fallback raises."""
        with pytest.raises(ValueError, match="positive"):
            compute_t_ref(dt=-1.0, t_max=100.0)

    @pytest.mark.unit
    def test_invalid_dt_with_fallback(self) -> None:
        """Negative dt with fallback returns fallback."""
        t_ref = compute_t_ref(dt=-1.0, t_max=100.0, fallback_value=5.0)
        assert t_ref == 5.0

    @pytest.mark.unit
    def test_zero_t_max_raises(self) -> None:
        """Zero t_max without fallback raises."""
        with pytest.raises(ValueError, match="positive"):
            compute_t_ref(dt=1.0, t_max=0.0)

    @pytest.mark.unit
    def test_zero_t_max_with_fallback(self) -> None:
        """Zero t_max with fallback returns fallback."""
        t_ref = compute_t_ref(dt=1.0, t_max=0.0, fallback_value=1.0)
        assert t_ref == 1.0

    @pytest.mark.unit
    def test_small_values(self) -> None:
        """Very small dt/t_max still produce valid results."""
        t_ref = compute_t_ref(dt=1e-6, t_max=1e-3)
        assert t_ref > 0
        assert math.isfinite(t_ref)


# ============================================================================
# ReparamConfig
# ============================================================================


class TestReparamConfig:
    """Tests for ReparamConfig dataclass."""

    @pytest.mark.unit
    def test_default_enables_all(self) -> None:
        """Default config enables all 3 pairs."""
        config = ReparamConfig(t_ref=10.0)
        assert len(config.enabled_pairs) == 3

    @pytest.mark.unit
    def test_disable_one_pair(self) -> None:
        """Can selectively disable pairs."""
        config = ReparamConfig(t_ref=10.0, enable_v_ref=False)
        assert len(config.enabled_pairs) == 2
        prefactors = [p for p, _ in config.enabled_pairs]
        assert "v0" not in prefactors

    @pytest.mark.unit
    def test_is_reparameterized(self) -> None:
        """is_reparameterized correctly identifies pair members."""
        config = ReparamConfig(t_ref=10.0)
        assert config.is_reparameterized("D0_ref")
        assert config.is_reparameterized("alpha_ref")
        assert not config.is_reparameterized("f0")
        assert not config.is_reparameterized("phi0")

    @pytest.mark.unit
    def test_is_reparameterized_disabled(self) -> None:
        """Disabled pairs are not reparameterized."""
        config = ReparamConfig(t_ref=10.0, enable_d_ref=False)
        assert not config.is_reparameterized("D0_ref")
        assert not config.is_reparameterized("alpha_ref")

    @pytest.mark.unit
    def test_get_reparam_name(self) -> None:
        """Reparam names follow expected pattern."""
        config = ReparamConfig(t_ref=10.0)
        assert config.get_reparam_name("D0_ref") == "log_D0_ref_at_tref"
        assert config.get_reparam_name("v0") == "log_v0_at_tref"

    @pytest.mark.unit
    def test_frozen(self) -> None:
        """Config is immutable."""
        config = ReparamConfig(t_ref=10.0)
        with pytest.raises(AttributeError):
            config.t_ref = 20.0  # type: ignore[misc]


# ============================================================================
# Round-trip transforms
# ============================================================================


class TestRoundTripTransforms:
    """Test forward → inverse round-trip consistency."""

    @pytest.mark.unit
    def test_single_pair_round_trip(self) -> None:
        """Forward then inverse recovers original values."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 0.5, "alpha_ref": 0.8, "f0": 1.0}
        nlsq_uncertainties = {"D0_ref": 0.05, "alpha_ref": 0.1, "f0": 0.01}

        config = ReparamConfig(
            t_ref=t_ref, enable_d_sample=False, enable_v_ref=False,
        )

        # Forward: physics → reparam
        reparam_vals, _ = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref, config,
        )

        # Inverse: reparam → physics (via numpy arrays, as posterior samples would be)
        samples = {k: np.array([v]) for k, v in reparam_vals.items()}
        physics = transform_to_physics_space(samples, config)

        assert physics["D0_ref"][0] == pytest.approx(0.5, rel=1e-10)
        assert physics["alpha_ref"][0] == pytest.approx(0.8, rel=1e-10)
        assert physics["f0"][0] == pytest.approx(1.0, rel=1e-10)

    @pytest.mark.unit
    def test_all_three_pairs_round_trip(self) -> None:
        """Round-trip works for all 3 power-law pairs simultaneously."""
        t_ref = 5.0
        nlsq_values = {
            "D0_ref": 1.0, "alpha_ref": 0.5,
            "D0_sample": 0.3, "alpha_sample": 1.2,
            "v0": 0.01, "beta": 0.7,
            "D_offset_ref": 0.0, "f0": 1.0,
        }
        nlsq_uncertainties = {
            "D0_ref": 0.1, "alpha_ref": 0.05,
            "D0_sample": 0.03, "alpha_sample": 0.1,
            "v0": 0.001, "beta": 0.07,
            "D_offset_ref": 0.0, "f0": 0.01,
        }

        config = ReparamConfig(t_ref=t_ref)
        reparam_vals, _ = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref, config,
        )

        samples = {k: np.array([v]) for k, v in reparam_vals.items()}
        physics = transform_to_physics_space(samples, config)

        assert physics["D0_ref"][0] == pytest.approx(1.0, rel=1e-10)
        assert physics["D0_sample"][0] == pytest.approx(0.3, rel=1e-10)
        assert physics["v0"][0] == pytest.approx(0.01, rel=1e-10)
        # Verify exponents are also recovered correctly
        assert physics["alpha_ref"][0] == pytest.approx(0.5, rel=1e-10)
        assert physics["alpha_sample"][0] == pytest.approx(1.2, rel=1e-10)
        assert physics["beta"][0] == pytest.approx(0.7, rel=1e-10)

    @pytest.mark.unit
    def test_sampling_space_round_trip(self) -> None:
        """transform_to_sampling_space → transform_to_physics_space round-trips."""
        t_ref = 10.0
        config = ReparamConfig(t_ref=t_ref)
        params = {"D0_ref": 2.0, "alpha_ref": 0.6, "v0": 0.05, "beta": 0.3,
                  "D0_sample": 0.1, "alpha_sample": 1.0}

        sampling = transform_to_sampling_space(params, config)
        samples = {k: np.array([v]) for k, v in sampling.items()}
        physics = transform_to_physics_space(samples, config)

        for key in ["D0_ref", "D0_sample", "v0"]:
            assert physics[key][0] == pytest.approx(params[key], rel=1e-10)


# ============================================================================
# Delta-method uncertainty propagation
# ============================================================================


class TestDeltaMethodUQ:
    """Test delta-method uncertainty propagation."""

    @pytest.mark.unit
    def test_uncertainty_positive(self) -> None:
        """Reparam uncertainties are always positive."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 0.5}
        nlsq_uncertainties = {"D0_ref": 0.1, "alpha_ref": 0.05}

        config = ReparamConfig(
            t_ref=t_ref, enable_d_sample=False, enable_v_ref=False,
        )
        _, reparam_unc = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref, config,
        )

        for name, unc in reparam_unc.items():
            assert unc > 0, f"Uncertainty for {name} should be positive, got {unc}"

    @pytest.mark.unit
    def test_zero_uncertainty_gets_floor(self) -> None:
        """Zero NLSQ uncertainty gets floored to 1e-6."""
        t_ref = 10.0
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 0.5}
        nlsq_uncertainties = {"D0_ref": 0.0, "alpha_ref": 0.0}

        config = ReparamConfig(
            t_ref=t_ref, enable_d_sample=False, enable_v_ref=False,
        )
        _, reparam_unc = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref, config,
        )

        assert reparam_unc["log_D0_ref_at_tref"] >= 1e-6
        assert reparam_unc["alpha_ref"] >= 1e-6

    @pytest.mark.unit
    def test_delta_method_formula(self) -> None:
        """Verify the delta-method formula explicitly."""
        t_ref = 10.0
        D0, alpha = 1.0, 0.5
        sigma_D0, sigma_alpha = 0.1, 0.05

        config = ReparamConfig(
            t_ref=t_ref, enable_d_sample=False, enable_v_ref=False,
        )
        _, unc = transform_nlsq_to_reparam_space(
            {"D0_ref": D0, "alpha_ref": alpha},
            {"D0_ref": sigma_D0, "alpha_ref": sigma_alpha},
            t_ref, config,
        )

        # Manual: sqrt((sigma_D0/D0)^2 + (log(t_ref)*sigma_alpha)^2)
        expected = math.sqrt(
            (sigma_D0 / D0) ** 2 + (math.log(t_ref) * sigma_alpha) ** 2
        )
        assert unc["log_D0_ref_at_tref"] == pytest.approx(expected, rel=1e-10)


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case handling."""

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_negative_prefactor_handled(self) -> None:
        """Negative prefactor uses log(|a0|) fallback."""
        t_ref = 10.0
        config = ReparamConfig(
            t_ref=t_ref, enable_d_sample=False, enable_v_ref=False,
        )
        values = {"D0_ref": -0.5, "alpha_ref": 0.8}
        unc = {"D0_ref": 0.1, "alpha_ref": 0.05}

        reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
            values, unc, t_ref, config,
        )

        # Should still produce a finite result
        assert math.isfinite(reparam_vals["log_D0_ref_at_tref"])
        assert reparam_unc["log_D0_ref_at_tref"] > 0

    @pytest.mark.unit
    def test_vectorized_physics_transform(self) -> None:
        """transform_to_physics_space works on arrays of samples."""
        config = ReparamConfig(t_ref=10.0)
        n_samples = 500

        samples = {
            "log_D0_ref_at_tref": np.random.normal(0.0, 0.1, n_samples),
            "alpha_ref": np.random.normal(0.5, 0.05, n_samples),
            "log_D0_sample_at_tref": np.random.normal(-1.0, 0.2, n_samples),
            "alpha_sample": np.random.normal(1.0, 0.1, n_samples),
            "log_v0_at_tref": np.random.normal(-3.0, 0.5, n_samples),
            "beta": np.random.normal(0.7, 0.05, n_samples),
            "f0": np.ones(n_samples),
        }

        physics = transform_to_physics_space(samples, config)

        assert "D0_ref" in physics
        assert "D0_sample" in physics
        assert "v0" in physics
        assert "f0" in physics
        assert len(physics["D0_ref"]) == n_samples
        assert np.all(physics["D0_ref"] > 0)  # exp() is always positive

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_reparam_to_physics_jax(self) -> None:
        """JAX back-transform produces correct values."""
        t_ref = 10.0
        # D0 * t_ref^alpha = exp(log_at_tref)
        D0_true = 2.0
        alpha_true = 0.5
        log_at_tref = jnp.log(jnp.float64(D0_true)) + alpha_true * jnp.log(jnp.float64(t_ref))

        D0_recovered = reparam_to_physics_jax(log_at_tref, jnp.float64(alpha_true), t_ref)
        assert float(D0_recovered) == pytest.approx(D0_true, rel=1e-10)

    @pytest.mark.unit
    def test_physics_transform_missing_reparam_sample(self) -> None:
        """Missing reparam sample (log_X_at_tref) in samples is skipped gracefully."""
        config = ReparamConfig(t_ref=10.0)
        # Include exponent but NOT the log_D0_ref_at_tref
        samples = {
            "alpha_ref": np.array([0.5]),
            "f0": np.array([1.0]),
        }
        physics = transform_to_physics_space(samples, config)
        # Should pass through alpha_ref and f0, not crash
        assert "f0" in physics
        assert "alpha_ref" in physics
        assert "D0_ref" not in physics  # Cannot reconstruct without log_D0_ref_at_tref

    @pytest.mark.unit
    def test_delta_method_large_tref_finite(self) -> None:
        """Delta-method stays finite for large t_ref (common in slow dynamics)."""
        t_ref = 1e6
        config = ReparamConfig(t_ref=t_ref, enable_d_sample=False, enable_v_ref=False)
        _, unc = transform_nlsq_to_reparam_space(
            {"D0_ref": 1.0, "alpha_ref": 0.5},
            {"D0_ref": 0.1, "alpha_ref": 0.05},
            t_ref, config,
        )
        assert math.isfinite(unc["log_D0_ref_at_tref"])
        assert unc["log_D0_ref_at_tref"] > 0
        # log(1e6) ≈ 13.8, so uncertainty should be amplified but still reasonable
        assert unc["log_D0_ref_at_tref"] < 10.0

    @pytest.mark.unit
    def test_power_law_pairs_constant(self) -> None:
        """POWER_LAW_PAIRS matches the 3 heterodyne pairs."""
        assert len(POWER_LAW_PAIRS) == 3
        prefactors = [p for p, _ in POWER_LAW_PAIRS]
        assert prefactors == ["D0_ref", "D0_sample", "v0"]
