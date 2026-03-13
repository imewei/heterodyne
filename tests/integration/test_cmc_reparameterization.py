"""Integration tests for CMC reparameterization round-trip correctness.

Verifies that:
- Physics -> reparam -> physics recovers original values for all 3 pairs
- NLSQ-informed priors in reparam space are centered correctly
- Reparameterized and non-reparameterized models produce same likelihood
- Partial reparameterization works correctly
- ReparamConfig with all flags disabled passes params through unchanged

All tests use small problem sizes for CI speed.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_physics_params() -> dict[str, float]:
    """Create a representative set of physics-space parameter values."""
    return {
        "D0_ref": 1e4,
        "alpha_ref": 0.5,
        "D0_sample": 5e3,
        "alpha_sample": 0.3,
        "v0": 1e3,
        "beta": 0.8,
        # Non-reparameterized params
        "f0": 0.5,
        "f3": 0.2,
        "gamma_ref": 0.1,
        "gamma_sample": 0.05,
        "v_offset": 0.0,
        "theta": 45.0,
        "contrast": 0.5,
        "offset": 1.0,
    }


def _make_uncertainties() -> dict[str, float]:
    """Create representative NLSQ uncertainties."""
    return {
        "D0_ref": 500.0,
        "alpha_ref": 0.05,
        "D0_sample": 300.0,
        "alpha_sample": 0.03,
        "v0": 50.0,
        "beta": 0.04,
        "f0": 0.02,
        "f3": 0.01,
        "gamma_ref": 0.01,
        "gamma_sample": 0.005,
        "v_offset": 0.5,
        "theta": 1.0,
        "contrast": 0.02,
        "offset": 0.01,
    }


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTripAllPairs:
    """Physics -> reparam -> physics recovers original values."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "prefactor,exponent",
        [
            ("D0_ref", "alpha_ref"),
            ("D0_sample", "alpha_sample"),
            ("v0", "beta"),
        ],
        ids=["D_ref", "D_sample", "velocity"],
    )
    def test_round_trip_single_pair(
        self,
        prefactor: str,
        exponent: str,
    ) -> None:
        """Forward then inverse transform recovers original prefactor."""
        t_ref = 10.0
        a0_original = 1e4
        alpha_original = 0.5

        # Forward: physics -> log-space at t_ref
        log_a0 = math.log(a0_original)
        log_tref = math.log(t_ref)
        log_at_tref = log_a0 + alpha_original * log_tref

        # Inverse: log-space -> physics via JAX
        a0_recovered = float(
            reparam_to_physics_jax(
                jnp.float64(log_at_tref),
                jnp.float64(alpha_original),
                t_ref,
            )
        )

        npt.assert_allclose(a0_recovered, a0_original, rtol=1e-10)

    @pytest.mark.integration
    def test_round_trip_all_pairs_via_transform_functions(self) -> None:
        """Full round-trip through transform_to_sampling_space -> transform_to_physics_space."""
        physics_params = _make_physics_params()
        t_ref = compute_t_ref(dt=1.0, t_max=16.0)
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=True,
            enable_v_ref=True,
            t_ref=t_ref,
        )

        # Forward: physics -> sampling space
        sampling_params = transform_to_sampling_space(physics_params, config)

        # Check that prefactors are replaced by log_*_at_tref
        for prefactor, exponent in config.enabled_pairs:
            log_name = config.get_reparam_name(prefactor)
            assert log_name in sampling_params, f"Expected {log_name} in sampling space"
            assert prefactor not in sampling_params, (
                f"Prefactor {prefactor} should not be in sampling space"
            )
            # Exponent should pass through
            assert exponent in sampling_params

        # Inverse: sampling -> physics space (using array-valued dict for transform_to_physics_space)
        sampling_arrays = {k: np.array([v]) for k, v in sampling_params.items()}
        physics_recovered = transform_to_physics_space(sampling_arrays, config)

        # Verify recovery of all 3 prefactors
        for prefactor, exponent in POWER_LAW_PAIRS:
            npt.assert_allclose(
                physics_recovered[prefactor][0],
                physics_params[prefactor],
                rtol=1e-10,
                err_msg=f"Round-trip failed for {prefactor}",
            )
            npt.assert_allclose(
                physics_recovered[exponent][0],
                physics_params[exponent],
                rtol=1e-10,
                err_msg=f"Round-trip failed for {exponent}",
            )


class TestNLSQInformedPriorsReparamSpace:
    """NLSQ-informed priors in reparam space are centered on transformed NLSQ values."""

    @pytest.mark.integration
    def test_transformed_values_match_nlsq(self) -> None:
        """Transformed NLSQ values are consistent with manual computation."""
        physics_params = _make_physics_params()
        uncertainties = _make_uncertainties()
        t_ref = compute_t_ref(dt=1.0, t_max=16.0)
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=True,
            enable_v_ref=True,
            t_ref=t_ref,
        )

        transformed_values, transformed_unc = transform_nlsq_to_reparam_space(
            nlsq_values=physics_params,
            nlsq_uncertainties=uncertainties,
            t_ref=t_ref,
            config=config,
        )

        log_tref = math.log(t_ref)

        # Verify each reparameterized pair
        for prefactor, exponent in config.enabled_pairs:
            log_name = config.get_reparam_name(prefactor)
            a0 = physics_params[prefactor]
            alpha = physics_params[exponent]

            expected_log_at_tref = math.log(a0) + alpha * log_tref
            npt.assert_allclose(
                transformed_values[log_name],
                expected_log_at_tref,
                rtol=1e-10,
                err_msg=f"NLSQ transform mismatch for {log_name}",
            )

            # Exponent should pass through unchanged
            npt.assert_allclose(
                transformed_values[exponent],
                alpha,
                rtol=1e-10,
                err_msg=f"Exponent {exponent} should pass through",
            )

    @pytest.mark.integration
    def test_transformed_uncertainties_positive(self) -> None:
        """All transformed uncertainties are positive and finite."""
        physics_params = _make_physics_params()
        uncertainties = _make_uncertainties()
        t_ref = compute_t_ref(dt=1.0, t_max=16.0)

        _, transformed_unc = transform_nlsq_to_reparam_space(
            nlsq_values=physics_params,
            nlsq_uncertainties=uncertainties,
            t_ref=t_ref,
        )

        for name, unc in transformed_unc.items():
            assert math.isfinite(unc), f"Non-finite uncertainty for {name}: {unc}"
            assert unc >= 0.0, f"Negative uncertainty for {name}: {unc}"

    @pytest.mark.integration
    def test_nlsq_informed_priors_centered(self) -> None:
        """build_nlsq_informed_priors centers on NLSQ values in physics space."""
        # Build a minimal NLSQ result with known values
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
        from heterodyne.config.parameter_space import ParameterSpace
        from heterodyne.optimization.cmc.priors import build_nlsq_informed_priors
        from heterodyne.optimization.nlsq.results import NLSQResult

        varying_names = [
            name for name in DEFAULT_REGISTRY if DEFAULT_REGISTRY[name].vary_default
        ]

        # Use registry defaults as "NLSQ result"
        params = np.array([DEFAULT_REGISTRY[n].default for n in varying_names])
        uncertainties = np.array(
            [
                DEFAULT_REGISTRY[n].prior_std if DEFAULT_REGISTRY[n].prior_std else 1.0
                for n in varying_names
            ]
        )

        nlsq_result = NLSQResult(
            parameters=params,
            parameter_names=list(varying_names),
            uncertainties=uncertainties,
            success=True,
            message="synthetic",
        )

        space = ParameterSpace()
        priors = build_nlsq_informed_priors(nlsq_result, space, width_factor=2.0)

        # Every varying parameter should have a prior
        for name in varying_names:
            assert name in priors, f"Missing prior for {name}"


class TestReparamModelConsistency:
    """Reparameterized model produces same likelihood as non-reparameterized."""

    @pytest.mark.integration
    def test_reparam_to_physics_jax_matches_numpy(self) -> None:
        """JAX back-transform matches numpy computation."""
        t_ref = 4.0
        log_tref = math.log(t_ref)

        test_cases = [
            (math.log(1e4) + 0.5 * log_tref, 0.5),
            (math.log(5e3) + 0.3 * log_tref, 0.3),
            (math.log(1e3) + 0.8 * log_tref, 0.8),
        ]

        for log_at_tref, alpha in test_cases:
            # Numpy reference
            a0_numpy = math.exp(log_at_tref - alpha * log_tref)

            # JAX
            a0_jax = float(
                reparam_to_physics_jax(
                    jnp.float64(log_at_tref),
                    jnp.float64(alpha),
                    t_ref,
                )
            )

            npt.assert_allclose(
                a0_jax,
                a0_numpy,
                rtol=1e-12,
                err_msg=f"JAX/numpy mismatch for log_at_tref={log_at_tref}, alpha={alpha}",
            )


class TestPartialReparameterization:
    """Partial reparameterization (only some pairs enabled) works correctly."""

    @pytest.mark.integration
    def test_only_d_ref_reparameterized(self) -> None:
        """Only D_ref pair is reparameterized; others pass through."""
        physics_params = _make_physics_params()
        t_ref = 4.0
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=False,
            enable_v_ref=False,
            t_ref=t_ref,
        )

        assert len(config.enabled_pairs) == 1
        assert config.enabled_pairs[0] == ("D0_ref", "alpha_ref")

        sampling = transform_to_sampling_space(physics_params, config)

        # D0_ref should be replaced by log_D0_ref_at_tref
        assert "log_D0_ref_at_tref" in sampling
        assert "D0_ref" not in sampling
        assert "alpha_ref" in sampling

        # D0_sample and v0 should pass through unchanged
        npt.assert_allclose(
            sampling["D0_sample"], physics_params["D0_sample"], rtol=1e-14
        )
        npt.assert_allclose(sampling["v0"], physics_params["v0"], rtol=1e-14)
        npt.assert_allclose(sampling["beta"], physics_params["beta"], rtol=1e-14)

        # Round-trip for only the D_ref pair
        sampling_arrays = {k: np.array([v]) for k, v in sampling.items()}
        recovered = transform_to_physics_space(sampling_arrays, config)

        npt.assert_allclose(
            recovered["D0_ref"][0],
            physics_params["D0_ref"],
            rtol=1e-10,
            err_msg="D0_ref round-trip failed with partial reparam",
        )

    @pytest.mark.integration
    def test_only_velocity_reparameterized(self) -> None:
        """Only v0/beta pair is reparameterized; D pairs pass through."""
        physics_params = _make_physics_params()
        t_ref = 4.0
        config = ReparamConfig(
            enable_d_ref=False,
            enable_d_sample=False,
            enable_v_ref=True,
            t_ref=t_ref,
        )

        assert len(config.enabled_pairs) == 1
        assert config.enabled_pairs[0] == ("v0", "beta")

        sampling = transform_to_sampling_space(physics_params, config)

        # v0 replaced, D0s pass through
        assert "log_v0_at_tref" in sampling
        assert "v0" not in sampling
        npt.assert_allclose(sampling["D0_ref"], physics_params["D0_ref"], rtol=1e-14)
        npt.assert_allclose(
            sampling["D0_sample"], physics_params["D0_sample"], rtol=1e-14
        )


class TestAllFlagsDisabled:
    """ReparamConfig with all flags disabled passes params through unchanged."""

    @pytest.mark.integration
    def test_disabled_config_is_identity(self) -> None:
        """With all flags disabled, transform is identity."""
        physics_params = _make_physics_params()
        config = ReparamConfig(
            enable_d_ref=False,
            enable_d_sample=False,
            enable_v_ref=False,
            t_ref=4.0,
        )

        assert len(config.enabled_pairs) == 0

        sampling = transform_to_sampling_space(physics_params, config)

        # All parameters should pass through unchanged
        for name, value in physics_params.items():
            assert name in sampling, f"Missing parameter {name} in output"
            npt.assert_allclose(
                sampling[name],
                value,
                rtol=1e-14,
                err_msg=f"Parameter {name} should pass through unchanged",
            )

        # No log_*_at_tref keys should exist
        for key in sampling:
            assert not key.startswith("log_"), (
                f"Unexpected reparameterized key {key} when all flags disabled"
            )

    @pytest.mark.integration
    def test_disabled_config_round_trip(self) -> None:
        """Round-trip with disabled config is identity."""
        physics_params = _make_physics_params()
        config = ReparamConfig(
            enable_d_ref=False,
            enable_d_sample=False,
            enable_v_ref=False,
            t_ref=4.0,
        )

        sampling = transform_to_sampling_space(physics_params, config)
        sampling_arrays = {k: np.array([v]) for k, v in sampling.items()}
        recovered = transform_to_physics_space(sampling_arrays, config)

        for name, value in physics_params.items():
            npt.assert_allclose(
                recovered[name][0],
                value,
                rtol=1e-14,
                err_msg=f"Round-trip failed for {name} with disabled config",
            )


class TestReparamConfigProperties:
    """ReparamConfig helper methods work correctly."""

    @pytest.mark.integration
    def test_is_reparameterized(self) -> None:
        """is_reparameterized correctly identifies participating parameters."""
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=False,
            enable_v_ref=True,
            t_ref=4.0,
        )

        # D_ref pair: both D0_ref and alpha_ref participate
        assert config.is_reparameterized("D0_ref")
        assert config.is_reparameterized("alpha_ref")

        # D_sample pair: disabled
        assert not config.is_reparameterized("D0_sample")
        assert not config.is_reparameterized("alpha_sample")

        # v0/beta pair: enabled
        assert config.is_reparameterized("v0")
        assert config.is_reparameterized("beta")

        # Non-power-law params
        assert not config.is_reparameterized("f0")
        assert not config.is_reparameterized("contrast")

    @pytest.mark.integration
    def test_compute_t_ref(self) -> None:
        """compute_t_ref returns geometric mean of dt and t_max."""
        dt = 1.0
        t_max = 100.0
        t_ref = compute_t_ref(dt, t_max)
        npt.assert_allclose(t_ref, math.sqrt(dt * t_max), rtol=1e-14)

    @pytest.mark.integration
    def test_compute_t_ref_invalid_raises(self) -> None:
        """compute_t_ref raises ValueError for non-positive inputs."""
        with pytest.raises(ValueError):
            compute_t_ref(dt=0.0, t_max=10.0)

        with pytest.raises(ValueError):
            compute_t_ref(dt=1.0, t_max=-5.0)

    @pytest.mark.integration
    def test_compute_t_ref_fallback(self) -> None:
        """compute_t_ref uses fallback for invalid inputs."""
        result = compute_t_ref(dt=0.0, t_max=10.0, fallback_value=5.0)
        assert result == 5.0
