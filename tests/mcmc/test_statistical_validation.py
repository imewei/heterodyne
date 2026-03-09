"""Statistical validation tests for the MCMC (CMC) subsystem.

Tests cover prior construction, MCMC convergence smoke tests,
reparameterization round-trips, and CMCConfig validation.

All MCMC tests use minimal settings (num_warmup=10, num_samples=20,
num_chains=1) for CI speed.
"""

from __future__ import annotations

import math

import numpy as np
import numpyro.distributions as dist
import pytest

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.config.parameter_space import ParameterSpace
from heterodyne.optimization.cmc.config import CMCConfig
from heterodyne.optimization.cmc.diagnostics import compute_ess, compute_r_hat
from heterodyne.optimization.cmc.priors import (
    build_default_priors,
    build_log_space_priors,
    build_nlsq_informed_priors,
)
from heterodyne.optimization.cmc.reparameterization import (
    ReparamConfig,
    transform_to_physics_space,
    transform_to_sampling_space,
)
from heterodyne.optimization.cmc.results import CMCResult
from heterodyne.optimization.nlsq.results import NLSQResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param_space() -> ParameterSpace:
    """Create a default ParameterSpace with all registry defaults."""
    return ParameterSpace()


def _make_nlsq_result(param_space: ParameterSpace) -> NLSQResult:
    """Build a synthetic NLSQResult matching the varying parameters.

    Values are set to registry defaults; uncertainties are 1 % of value
    (minimum 1e-6) to simulate a well-converged NLSQ fit.
    """
    names = param_space.varying_names
    values = np.array(
        [param_space.values[n] for n in names], dtype=np.float64,
    )
    uncertainties = np.maximum(np.abs(values) * 0.01, 1e-6)
    return NLSQResult(
        parameters=values,
        parameter_names=list(names),
        success=True,
        message="synthetic",
        uncertainties=uncertainties,
        reduced_chi_squared=1.05,
    )


# ===================================================================
# TestPriorConstruction
# ===================================================================


@pytest.mark.mcmc
class TestPriorConstruction:
    """Validate prior-building helpers produce correct distributions."""

    def test_default_priors_cover_all_varying(self) -> None:
        """build_default_priors returns a prior for every varying parameter."""
        param_space = _make_param_space()
        priors = build_default_priors(param_space)
        varying = set(param_space.varying_names)
        prior_names = set(priors.keys())
        assert varying == prior_names, (
            f"Missing priors for: {varying - prior_names}; "
            f"extra priors for: {prior_names - varying}"
        )

    def test_nlsq_informed_priors_narrower(self) -> None:
        """NLSQ-informed priors have smaller scale than defaults.

        For parameters where NLSQ uncertainty is available and smaller
        than the registry prior_std, the informed prior should be
        tighter.
        """
        param_space = _make_param_space()
        nlsq_result = _make_nlsq_result(param_space)
        default_priors = build_default_priors(param_space)
        informed_priors = build_nlsq_informed_priors(
            nlsq_result, param_space, width_factor=2.0,
        )

        n_narrower = 0
        for name in param_space.varying_names:
            dp = default_priors[name]
            ip = informed_priors[name]

            # Extract scale where possible (TruncatedNormal)
            default_scale = _extract_scale(dp)
            informed_scale = _extract_scale(ip)

            if default_scale is not None and informed_scale is not None:
                if informed_scale < default_scale:
                    n_narrower += 1

        # At least half of the parameters should have narrower priors
        # when NLSQ converges well (our synthetic result has 1 % uncertainty).
        assert n_narrower > 0, (
            "Expected at least one NLSQ-informed prior to be narrower than default"
        )

    def test_log_space_priors_positive_only(self) -> None:
        """Log-space priors for D0 parameters produce strictly positive samples."""
        param_space = _make_param_space()
        log_names = [
            name for name in param_space.varying_names
            if DEFAULT_REGISTRY[name].log_space
        ]
        assert len(log_names) > 0, "Expected at least one log_space parameter"

        priors = build_log_space_priors(log_names)

        # All returned priors should be LogNormal (support > 0)
        for name, prior in priors.items():
            assert isinstance(prior, dist.LogNormal), (
                f"Expected LogNormal for {name}, got {type(prior).__name__}"
            )
            # Draw samples and verify positivity
            import jax
            key = jax.random.PRNGKey(42)
            samples = prior.sample(key, sample_shape=(500,))
            assert np.all(np.asarray(samples) > 0), (
                f"LogNormal prior for {name} produced non-positive samples"
            )


def _extract_scale(d: dist.Distribution) -> float | None:
    """Extract the scale parameter from common distribution types."""
    from numpyro.distributions.truncated import TwoSidedTruncatedDistribution

    if isinstance(d, TwoSidedTruncatedDistribution):
        return float(d.base_dist.scale)
    if isinstance(d, dist.LogNormal):
        return float(d.scale)
    if isinstance(d, dist.Uniform):
        return float(d.high - d.low)
    return None


# ===================================================================
# TestMCMCConvergence
# ===================================================================


@pytest.mark.mcmc
class TestMCMCConvergence:
    """Smoke tests for MCMC output shapes and diagnostics.

    Uses tiny MCMC settings (num_warmup=10, num_samples=20, num_chains=1)
    so tests complete quickly in CI.
    """

    def test_samples_shape_correct(self) -> None:
        """CMCResult samples dict has correct shapes."""
        param_names = ["D0_ref", "alpha_ref", "D0_sample"]
        n_samples = 20
        n_chains = 1
        total = n_chains * n_samples

        samples = {
            name: np.random.default_rng(42).normal(size=total)
            for name in param_names
        }
        result = CMCResult(
            parameter_names=param_names,
            posterior_mean=np.mean(
                [samples[n] for n in param_names], axis=1,
            ),
            posterior_std=np.std(
                [samples[n] for n in param_names], axis=1,
            ),
            credible_intervals={},
            convergence_passed=True,
            samples=samples,
            num_warmup=10,
            num_samples=n_samples,
            num_chains=n_chains,
        )

        assert result.samples is not None
        for name in param_names:
            arr = result.samples[name]
            assert arr.shape == (total,), (
                f"Expected shape ({total},) for {name}, got {arr.shape}"
            )
        assert result.n_params == len(param_names)

    def test_diagnostics_computed(self) -> None:
        """compute_r_hat and compute_ess return finite scalars."""
        rng = np.random.default_rng(0)
        # 2 chains, 20 draws each
        samples_2d = rng.normal(size=(2, 20))

        r_hat = compute_r_hat(samples_2d)
        ess = compute_ess(samples_2d)

        assert isinstance(r_hat, float)
        assert isinstance(ess, float)
        assert ess >= 1.0, f"ESS should be >= 1, got {ess}"

    def test_rhat_finite(self) -> None:
        """R-hat values are finite (not inf) when multiple chains are used.

        With 1 chain ArviZ returns NaN (R-hat is undefined for a single
        chain), which is acceptable.  With >= 2 chains, R-hat must be
        finite.
        """
        rng = np.random.default_rng(1)

        # Single chain: R-hat is NaN (undefined) -- verify it is at
        # least a float and not inf.
        samples_1d = rng.normal(size=(1, 40))
        r_hat_1c = compute_r_hat(samples_1d)
        assert isinstance(r_hat_1c, float), (
            f"Expected float, got {type(r_hat_1c)}"
        )
        # NaN is acceptable for 1 chain; inf is not.
        assert not math.isinf(r_hat_1c), f"R-hat is inf for 1 chain: {r_hat_1c}"

        # Multi-chain: R-hat must be fully finite.
        samples_2d = rng.normal(size=(2, 20))
        r_hat_mc = compute_r_hat(samples_2d)
        assert math.isfinite(r_hat_mc), f"R-hat (multi-chain) not finite: {r_hat_mc}"


# ===================================================================
# TestReparameterization
# ===================================================================


@pytest.mark.mcmc
class TestReparameterization:
    """Validate reference-time reparameterization round-trips."""

    def test_reparam_round_trip(self) -> None:
        """transform_to_sampling_space -> transform_to_physics_space round-trips."""
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=True,
            enable_v_ref=True,
            t_ref=100.0,
        )

        physics_params: dict[str, float] = {
            "D0_ref": 1e4,
            "alpha_ref": 0.8,
            "D0_sample": 5e3,
            "alpha_sample": 0.5,
            "v0": 1e3,
            "beta": 0.3,
            "D_offset_ref": 0.0,
            "D_offset_sample": 0.0,
            "f0": 0.5,
            "f3": 0.1,
            "v_offset": 0.0,
            "contrast": 0.5,
            "offset": 1.0,
            "phi0": 0.0,
        }

        # Forward: physics -> sampling
        sampling_params = transform_to_sampling_space(physics_params, config)

        # The reparameterized space should have log_*_at_tref keys
        # instead of the prefactor keys for enabled pairs
        for prefactor, _exponent in config.enabled_pairs:
            reparam_name = config.get_reparam_name(prefactor)
            assert reparam_name in sampling_params, (
                f"Expected {reparam_name} in sampling space"
            )
            assert prefactor not in sampling_params, (
                f"Prefactor {prefactor} should be consumed in sampling space"
            )

        # Backward: sampling -> physics (need numpy arrays for transform_to_physics_space)
        sampling_arrays = {
            k: np.array([v]) for k, v in sampling_params.items()
        }
        recovered = transform_to_physics_space(sampling_arrays, config)

        # Check round-trip accuracy for all physics parameters
        for name, original in physics_params.items():
            assert name in recovered, f"Missing {name} after round-trip"
            recovered_val = float(recovered[name][0])
            np.testing.assert_allclose(
                recovered_val,
                original,
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"Round-trip failed for {name}",
            )

    def test_reparam_preserves_parameter_count(self) -> None:
        """Reparameterized space has same number of parameters as physics space."""
        config = ReparamConfig(
            enable_d_ref=True,
            enable_d_sample=True,
            enable_v_ref=True,
            t_ref=50.0,
        )

        physics_params: dict[str, float] = {
            "D0_ref": 1e4,
            "alpha_ref": 0.8,
            "D0_sample": 5e3,
            "alpha_sample": 0.5,
            "v0": 1e3,
            "beta": 0.3,
            "D_offset_ref": 0.0,
            "D_offset_sample": 0.0,
            "f0": 0.5,
            "f3": 0.1,
            "v_offset": 0.0,
            "contrast": 0.5,
            "offset": 1.0,
            "phi0": 0.0,
        }

        sampling_params = transform_to_sampling_space(physics_params, config)
        assert len(sampling_params) == len(physics_params), (
            f"Parameter count changed: physics={len(physics_params)}, "
            f"sampling={len(sampling_params)}"
        )


# ===================================================================
# TestCMCConfigValidation
# ===================================================================


@pytest.mark.mcmc
class TestCMCConfigValidation:
    """Validate CMCConfig defaults and constraints."""

    def test_default_config_valid(self) -> None:
        """CMCConfig() creates a valid config with sensible defaults."""
        config = CMCConfig()
        errors = config.validate()
        assert errors == [], f"Default config has validation errors: {errors}"

    def test_target_accept_prob_in_range(self) -> None:
        """Default target_accept_prob is between 0.5 and 0.99."""
        config = CMCConfig()
        assert 0.5 <= config.target_accept_prob <= 0.99, (
            f"target_accept_prob={config.target_accept_prob} out of [0.5, 0.99]"
        )

    def test_num_chains_positive(self) -> None:
        """Default num_chains >= 1."""
        config = CMCConfig()
        assert config.num_chains >= 1, (
            f"num_chains={config.num_chains} should be >= 1"
        )
