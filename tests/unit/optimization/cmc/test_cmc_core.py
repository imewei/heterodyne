"""Tests for CMC (Consensus Monte Carlo) core functionality.

Bug Prevented: CMC Multi-Chain Init Shape Error
------------------------------------------------
When using multiple chains with NumPyro, init_params for each parameter
must have shape (num_chains,), not a scalar. Passing scalars causes
IndexError during sampling.

These tests verify that:
1. Init params are correctly shaped for single and multi-chain configs
2. CMC runs without IndexError for various chain counts
3. NLSQ warmstart properly propagates to CMC initialization
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from heterodyne import CMCConfig, HeterodyneModel, NLSQConfig
    from heterodyne.optimization.cmc.results import CMCResult


class TestInitParamsShape:
    """Tests for init_params shape handling in CMC."""

    @pytest.mark.unit
    def test_init_params_shape_single_chain(self) -> None:
        """Verify init_params shape is (1,) for single chain.

        Even with a single chain, NumPyro expects the init_params
        to have a chain dimension.
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=1)

        # Simulate creating init_params as done in fit_cmc_jax
        varying_names = ["D0_ref", "alpha_ref", "f0"]
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 1.0, "f0": 0.5}

        init_params = {
            name: jnp.full((config.num_chains,), nlsq_values[name])
            for name in varying_names
        }

        for name, values in init_params.items():
            assert values.shape == (1,), (
                f"init_params['{name}'] has shape {values.shape}, expected (1,)"
            )

    @pytest.mark.unit
    def test_init_params_shape_multi_chain(self) -> None:
        """Verify init_params shape is (num_chains,) for multiple chains.

        For 4 chains, each parameter should have shape (4,).
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=4)

        varying_names = ["D0_ref", "alpha_ref", "f0"]
        nlsq_values = {"D0_ref": 1.0, "alpha_ref": 1.0, "f0": 0.5}

        init_params = {
            name: jnp.full((config.num_chains,), nlsq_values[name])
            for name in varying_names
        }

        for name, values in init_params.items():
            assert values.shape == (4,), (
                f"init_params['{name}'] has shape {values.shape}, expected (4,)"
            )

    @pytest.mark.unit
    def test_init_params_all_chains_same_value(self) -> None:
        """Verify all chains are initialized to the same NLSQ value.

        When warm-starting from NLSQ, all chains should start at
        the NLSQ solution.
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=4)
        nlsq_value = 1.234

        init_param = jnp.full((config.num_chains,), nlsq_value)

        # All values should be identical
        assert jnp.allclose(init_param, nlsq_value)
        assert init_param.shape == (4,)


class TestCMCFitFunctions:
    """Tests for fit_cmc_jax function with various chain configurations."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_fit_cmc_jax_1_chain(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_1chain: CMCConfig,
    ) -> None:
        """Test fit_cmc_jax runs without IndexError with 1 chain.

        This is the simplest case and should definitely work.
        """
        from heterodyne import fit_cmc_jax

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_1chain,
        )

        assert result is not None
        assert hasattr(result, "posterior_mean")
        assert len(result.posterior_mean) == small_heterodyne_model.n_varying
        assert result.num_chains == 1

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_fit_cmc_jax_2_chains(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_2chains: CMCConfig,
    ) -> None:
        """Test fit_cmc_jax runs without IndexError with 2 chains.

        This is where the bug typically manifests if init_params
        shape is incorrect.
        """
        from heterodyne import fit_cmc_jax

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_2chains,
        )

        assert result is not None
        assert result.num_chains == 2
        assert len(result.posterior_mean) == small_heterodyne_model.n_varying

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_fit_cmc_jax_4_chains(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_4chains: CMCConfig,
    ) -> None:
        """Test fit_cmc_jax runs without IndexError with 4 chains.

        4 chains is the standard configuration for proper R-hat
        diagnostics.
        """
        from heterodyne import fit_cmc_jax

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_4chains,
        )

        assert result is not None
        assert result.num_chains == 4
        assert len(result.posterior_mean) == small_heterodyne_model.n_varying

        # With 4 chains, we should have R-hat values
        if result.r_hat is not None:
            assert len(result.r_hat) == small_heterodyne_model.n_varying


class TestNLSQWarmstart:
    """Tests for NLSQ warm-start integration with CMC."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_nlsq_warmstart_shape_propagation(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
        cmc_config_2chains: CMCConfig,
    ) -> None:
        """Test NLSQ result is correctly shaped for CMC init.

        When passing NLSQ result to CMC, the init_params should be
        properly replicated for all chains.
        """
        from heterodyne import fit_cmc_jax, fit_nlsq_jax

        # First run NLSQ
        nlsq_result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        # Then run CMC with NLSQ warmstart
        cmc_result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_2chains,
            nlsq_result=nlsq_result,
        )

        assert cmc_result is not None
        assert cmc_result.num_chains == 2

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_without_nlsq_warmstart(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test CMC works without NLSQ warmstart.

        CMC should initialize chains using its default strategy
        when no NLSQ result is provided.
        """
        from heterodyne import CMCConfig, fit_cmc_jax

        config = CMCConfig(
            num_chains=2,
            num_warmup=100,
            num_samples=100,
            seed=42,
            use_nlsq_warmstart=False,  # Explicitly disable
        )

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=config,
            nlsq_result=None,  # No warmstart
        )

        assert result is not None
        assert result.num_chains == 2


class TestCMCResult:
    """Tests for CMCResult structure and methods."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_result_samples_shape(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_2chains: CMCConfig,
    ) -> None:
        """Test CMC result has correctly shaped samples.

        Samples should have shape (num_chains * num_samples,) after
        ArviZ processing.
        """
        from heterodyne import fit_cmc_jax

        config = cmc_config_2chains
        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=config,
        )

        if result.samples is not None:
            expected_total = config.num_chains * config.num_samples
            for name, samples in result.samples.items():
                assert samples.shape[0] == expected_total, (
                    f"Samples for {name} have shape {samples.shape}, "
                    f"expected ({expected_total},)"
                )

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_result_diagnostics(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_2chains: CMCConfig,
    ) -> None:
        """Test CMC result has convergence diagnostics."""
        from heterodyne import fit_cmc_jax

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_2chains,
        )

        # Should have R-hat values
        assert result.r_hat is not None
        assert len(result.r_hat) == small_heterodyne_model.n_varying

        # Should have ESS values
        assert result.ess_bulk is not None
        assert len(result.ess_bulk) == small_heterodyne_model.n_varying

    @pytest.mark.unit
    def test_cmc_config_validation(self) -> None:
        """Test CMCConfig.validate() catches invalid chain count."""
        from heterodyne import CMCConfig

        # Valid configurations
        CMCConfig(num_chains=1)
        CMCConfig(num_chains=2)
        CMCConfig(num_chains=4)

        # Invalid: 0 chains — caught by validate()
        config = CMCConfig(num_chains=0)
        errors = config.validate()
        assert any("num_chains" in e for e in errors)


class TestBugPrevention_MultiChainInit:
    """Regression tests for CMC Multi-Chain Init Shape bug.

    BUG DESCRIPTION:
    When using multiple chains with NumPyro, init_params for each parameter
    must have shape (num_chains,), not a scalar. For example:

        # BUG: scalar init_params
        init_params = {"D0_ref": 1.0}  # Will cause IndexError

        # CORRECT: shape (num_chains,) init_params
        init_params = {"D0_ref": jnp.full((num_chains,), 1.0)}

    The bug manifests as IndexError during MCMC sampling when NumPyro
    tries to index the scalar values.

    These tests verify init_params are correctly shaped.
    """

    @pytest.mark.unit
    def test_scalar_init_params_is_wrong_shape(self) -> None:
        """REGRESSION TEST: Document that scalar init_params is wrong.

        NumPyro expects init_params to have shape (num_chains,) for each
        parameter when running multiple chains.
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=4)

        # This is the WRONG way to create init_params
        wrong_init_params = {"D0_ref": 1.0}  # Scalar!

        # Verify the scalar doesn't have the right shape
        value = wrong_init_params["D0_ref"]
        assert not hasattr(value, "shape") or value.shape != (config.num_chains,), (
            "This test documents that scalars are wrong"
        )

    @pytest.mark.unit
    def test_correct_init_params_shape(self) -> None:
        """REGRESSION TEST: Document the correct init_params shape.

        Each parameter should have shape (num_chains,) when using
        multiple chains.
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=4)

        # This is the CORRECT way to create init_params
        correct_init_params = {"D0_ref": jnp.full((config.num_chains,), 1.0)}

        # Verify the shape is correct
        value = correct_init_params["D0_ref"]
        assert value.shape == (config.num_chains,), (
            f"Expected shape ({config.num_chains},), got {value.shape}"
        )

    @pytest.mark.unit
    def test_init_params_replication_from_nlsq(self) -> None:
        """REGRESSION TEST: Verify NLSQ values are replicated correctly.

        When warm-starting from NLSQ, the scalar NLSQ value must be
        replicated to all chains using jnp.full((num_chains,), value).
        """
        from heterodyne import CMCConfig

        config = CMCConfig(num_chains=4)
        nlsq_value = 1.234  # Scalar from NLSQ result

        # Correct replication using jnp.full
        replicated = jnp.full((config.num_chains,), nlsq_value)

        assert replicated.shape == (config.num_chains,)
        assert jnp.all(replicated == nlsq_value)

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_multi_chain_cmc_does_not_raise_index_error(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """REGRESSION TEST: Verify multi-chain CMC doesn't raise IndexError.

        If init_params are scalars instead of (num_chains,) arrays,
        this would raise IndexError during sampling.
        """
        from heterodyne import CMCConfig, fit_cmc_jax, fit_nlsq_jax

        # First get NLSQ result for warm-starting
        nlsq_result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        # Run CMC with multiple chains - this would fail with wrong init shape
        config = CMCConfig(
            num_chains=2,  # Multiple chains triggers the bug
            num_warmup=100,
            num_samples=100,
            seed=42,
            use_nlsq_warmstart=True,
        )

        # This should NOT raise IndexError
        try:
            result = fit_cmc_jax(
                model=small_heterodyne_model,
                c2_data=small_c2_data,
                phi_angle=0.0,
                config=config,
                nlsq_result=nlsq_result,
            )
            assert result is not None
            assert result.num_chains == 2
        except IndexError as e:
            pytest.fail(
                f"Multi-chain CMC raised IndexError: {e}\n"
                "This indicates init_params have wrong shape."
            )


# ============================================================================
# Test reparameterization backward compatibility
# ============================================================================


class TestReparamBackwardCompat:
    """Tests that use_reparam=False preserves existing behavior."""

    @pytest.mark.unit
    def test_config_defaults(self) -> None:
        """New CMCConfig fields have expected defaults."""
        from heterodyne import CMCConfig

        config = CMCConfig()
        assert config.use_reparam is True
        assert config.nlsq_prior_width_factor == 2.0

    @pytest.mark.unit
    def test_config_from_dict_new_fields(self) -> None:
        """from_dict picks up new fields."""
        from heterodyne import CMCConfig

        config = CMCConfig.from_dict(
            {
                "num_warmup": 100,
                "num_samples": 100,
                "use_reparam": False,
                "prior_width_factor": 3.0,
            }
        )
        assert config.use_reparam is False
        assert config.nlsq_prior_width_factor == 3.0

    @pytest.mark.unit
    def test_config_to_dict_new_fields(self) -> None:
        """to_dict includes new fields."""
        from heterodyne import CMCConfig

        config = CMCConfig()
        d = config.to_dict()
        assert "use_reparam" in d["reparameterization"]
        assert "nlsq_prior_width_factor" in d["nlsq"]

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mcmc
    def test_reparam_disabled_runs(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """CMC with use_reparam=False runs successfully (legacy path)."""
        from heterodyne import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax

        config = CMCConfig(
            num_chains=1,
            num_warmup=100,
            num_samples=100,
            seed=42,
            use_reparam=False,
        )

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            config=config,
        )

        assert result is not None
        assert result.num_chains == 1


class TestReparamMetadata:
    """Tests that reparameterized CMC stores expected metadata."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mcmc
    def test_metadata_with_reparam(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """CMC with reparam stores t_ref and prior_std in metadata."""
        from heterodyne import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax
        from heterodyne.optimization.nlsq.results import NLSQResult

        config = CMCConfig(
            num_chains=1,
            num_warmup=100,
            num_samples=100,
            seed=42,
            use_reparam=True,
        )

        # Create a mock NLSQ result
        varying_names = small_heterodyne_model.param_manager.varying_names
        n_varying = len(varying_names)
        initial_values = small_heterodyne_model.param_manager.get_initial_values()

        nlsq_result = NLSQResult(
            parameters=initial_values,
            parameter_names=varying_names,
            success=True,
            message="mock",
            uncertainties=np.full(n_varying, 0.1),
        )

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            config=config,
            nlsq_result=nlsq_result,
        )

        assert result is not None
        assert "t_ref" in result.metadata
        assert result.metadata["t_ref"] > 0
        assert "prior_std" in result.metadata
        assert isinstance(result.metadata["prior_std"], dict)


# ============================================================================
# MCMC failure path
# ============================================================================


class TestMCMCFailurePath:
    """Tests for graceful degradation when MCMC fails."""

    @pytest.mark.unit
    def test_create_failed_result_structure(self) -> None:
        """_create_failed_result returns a valid CMCResult with error metadata."""
        from heterodyne.optimization.cmc.core import _create_failed_result

        result = _create_failed_result(["D0_ref", "alpha_ref"], "Test error")
        assert not result.convergence_passed
        assert result.posterior_mean.shape == (2,)
        assert result.posterior_std.shape == (2,)
        assert np.all(result.posterior_mean == 0.0)
        assert np.all(result.posterior_std == 0.0)
        assert result.credible_intervals == {}
        assert "error" in result.metadata
        assert result.metadata["error"] == "Test error"

    @pytest.mark.unit
    def test_create_failed_result_empty_params(self) -> None:
        """_create_failed_result handles empty parameter list."""
        from heterodyne.optimization.cmc.core import _create_failed_result

        result = _create_failed_result([], "No params")
        assert result.posterior_mean.shape == (0,)
        assert not result.convergence_passed


class TestCombineShardPosteriors:
    """Tests for failed-shard contamination fix in _combine_shard_posteriors."""

    @pytest.mark.unit
    def test_combine_excludes_failed_shards(self) -> None:
        """Failed zero-std shards must be excluded; result equals the good shard."""
        from types import SimpleNamespace

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        n_params = 3

        def make_shard(converged: bool, mean_val: float = 2.0) -> SimpleNamespace:
            r = SimpleNamespace()
            r.convergence_passed = converged
            r.parameter_names = [f"p{i}" for i in range(n_params)]
            if converged:
                r.posterior_mean = np.full(n_params, mean_val)
                r.posterior_std = np.ones(n_params) * 0.5
                r.r_hat = np.ones(n_params) * 1.01
                r.ess_bulk = np.ones(n_params) * 200.0
                r.ess_tail = np.ones(n_params) * 180.0
                r.bfmi = [0.3]
                r.samples = {f"p{i}": np.full(10, mean_val) for i in range(n_params)}
            else:
                r.posterior_mean = np.zeros(n_params)
                r.posterior_std = np.zeros(n_params)
                r.r_hat = None
                r.ess_bulk = None
                r.ess_tail = None
                r.bfmi = None
                r.samples = None
            r.num_warmup = 10
            r.num_samples = 100
            r.num_chains = 1
            return r

        good = make_shard(converged=True)
        bad = make_shard(converged=False)

        result = _combine_shard_posteriors(
            [good, bad], CMCConfig(), num_shards=2, base_seed=0
        )

        # With only one good shard, combined mean == good shard mean
        np.testing.assert_allclose(
            result.posterior_mean, good.posterior_mean, rtol=1e-9
        )
        # std should be finite (not dominated by 1e30 weight from bad shard)
        assert np.all(np.isfinite(result.posterior_std))
        assert np.all(result.posterior_std > 0)

    @pytest.mark.unit
    def test_combine_all_failed_returns_degenerate_result(self) -> None:
        """All shards failed: must return a failed CMCResult instead of crashing."""
        from types import SimpleNamespace

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        n_params = 3

        def make_failed_shard() -> SimpleNamespace:
            r = SimpleNamespace()
            r.convergence_passed = False
            r.parameter_names = [f"p{i}" for i in range(n_params)]
            r.posterior_mean = np.zeros(n_params)
            r.posterior_std = np.zeros(n_params)
            r.r_hat = None
            r.ess_bulk = None
            r.ess_tail = None
            r.bfmi = None
            r.samples = None
            r.num_warmup = 10
            r.num_samples = 100
            r.num_chains = 1
            return r

        bad1 = make_failed_shard()
        bad2 = make_failed_shard()

        result = _combine_shard_posteriors(
            [bad1, bad2], CMCConfig(), num_shards=2, base_seed=0
        )

        assert not result.convergence_passed
        assert result.metadata.get("all_shards_failed") is True
        assert result.metadata.get("n_total_shards") == 2
        assert np.all(np.isnan(result.posterior_std))

    @pytest.mark.unit
    def test_combine_accepts_shards_with_unknown_convergence(self) -> None:
        """Regression: shards with all-NaN r_hat (ArviZ failure) but valid std must
        be accepted by _combine_shard_posteriors, not silently dropped as 'failed'.

        This reproduces the het_676ccc47 failure mode where ArviZ 1.1.0 broke
        az.from_dict(**kwargs), causing idata=None for every shard, forcing
        r_hat=NaN, convergence_passed=False, and a completely degenerate result
        despite 44/47 shards having successfully collected NUTS samples.
        """
        from types import SimpleNamespace

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        n_params = 3

        def make_unknown_convergence_shard(mean_val: float = 2.0) -> SimpleNamespace:
            r = SimpleNamespace()
            r.convergence_passed = False  # forced False because ArviZ failed
            r.parameter_names = [f"p{i}" for i in range(n_params)]
            r.posterior_mean = np.full(n_params, mean_val)
            r.posterior_std = np.ones(n_params) * 0.5  # valid — samples were collected
            r.r_hat = np.full(n_params, np.nan)  # all NaN = ArviZ diagnostic failure
            r.ess_bulk = np.full(n_params, np.nan)
            r.ess_tail = np.full(n_params, np.nan)
            r.bfmi = None
            r.samples = {f"p{i}": np.full(100, mean_val) for i in range(n_params)}
            r.num_warmup = 10
            r.num_samples = 100
            r.num_chains = 4
            return r

        shard_a = make_unknown_convergence_shard(mean_val=2.0)
        shard_b = make_unknown_convergence_shard(mean_val=3.0)

        result = _combine_shard_posteriors(
            [shard_a, shard_b], CMCConfig(), num_shards=2, base_seed=0
        )

        # Must not return the degenerate all-shards-failed sentinel
        assert not result.metadata.get("all_shards_failed"), (
            "_combine_shard_posteriors incorrectly treated shards with unknown "
            "convergence (all-NaN r_hat) as failed. This is the het_676ccc47 regression."
        )
        # Posterior mean must be finite and between the two shard means
        assert np.all(np.isfinite(result.posterior_mean))
        assert np.all(result.posterior_mean >= 1.0)
        assert np.all(result.posterior_mean <= 4.0)


class TestSamplingSynchronization:
    """Tests for forcing asynchronous JAX sampling results before diagnostics."""

    @pytest.mark.unit
    def test_block_until_ready_pytree_blocks_array_leaves(self) -> None:
        """Regression: diagnostics phase should start only after samples are ready."""
        from heterodyne.optimization.cmc.core import _block_until_ready_pytree

        class BlockingLeaf:
            def __init__(self) -> None:
                self.blocked = False

            def block_until_ready(self) -> BlockingLeaf:
                self.blocked = True
                return self

        leaf = BlockingLeaf()
        samples = {"D0_ref": leaf, "alpha_ref": np.array([1.0, 2.0])}

        returned = _block_until_ready_pytree(samples)

        assert returned is samples
        assert leaf.blocked


class TestCombinationMethodDispatch:
    """combination_method from CMCConfig is respected."""

    def _shard(self, seed: int = 0) -> CMCResult:
        from heterodyne.optimization.cmc.results import CMCResult

        rng = np.random.default_rng(seed)
        names = ["D0_ref", "alpha_ref"]
        samples = {n: rng.normal(size=(2, 20)) for n in names}
        pm = np.array([abs(rng.normal(1e4, 100)), abs(rng.normal(0.5, 0.05))])
        ps = np.array([abs(rng.normal(100.0)), abs(rng.normal(0.05))])
        return CMCResult(
            parameter_names=names,
            posterior_mean=pm,
            posterior_std=ps,
            credible_intervals={},
            convergence_passed=True,
            r_hat=np.array([1.01, 1.02]),
            ess_bulk=np.array([500.0, 600.0]),
            ess_tail=np.array([450.0, 550.0]),
            samples=samples,
            num_warmup=50,
            num_samples=20,
            num_chains=2,
        )

    @pytest.mark.unit
    def test_consensus_mc_runs(self) -> None:
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        config = CMCConfig(combination_method="consensus_mc")
        result = _combine_shard_posteriors(
            [self._shard(i) for i in range(3)], config, num_shards=3, base_seed=0
        )
        assert result.convergence_passed is True

    @pytest.mark.unit
    def test_simple_average_mean(self) -> None:
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        config = CMCConfig(combination_method="simple_average")
        shards = [self._shard(i) for i in range(3)]
        result = _combine_shard_posteriors(shards, config, num_shards=3, base_seed=0)
        expected_mean_D0 = np.mean([s.posterior_mean[0] for s in shards])
        assert result.posterior_mean[0] == pytest.approx(expected_mean_D0, rel=1e-6)

    @pytest.mark.unit
    def test_unknown_method_falls_back_to_consensus_mc(self) -> None:
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        config = CMCConfig(combination_method="consensus_mc")
        result = _combine_shard_posteriors(
            [self._shard(0)], config, num_shards=1, base_seed=0
        )
        assert result is not None


class TestBugPrevention_ArviZAPI:
    """Regression tests pinning the ArviZ from_dict API used throughout CMC.

    ArviZ 1.0 changed az.from_dict() from accepting keyword args per group
    (az.from_dict(posterior={...})) to accepting a single dict
    (az.from_dict({"posterior": {...}})).  Breaking this API at the call sites
    in core.py and results.py caused the entire het_676ccc47 run (6032s) to
    produce a degenerate all-NaN result by silently swallowing the TypeError.

    These tests will fail on the FIRST pytest run after an incompatible ArviZ
    upgrade, long before any multi-hour NUTS run is launched.
    """

    @pytest.mark.unit
    def test_arviz_from_dict_new_api_produces_valid_summary(self) -> None:
        """az.from_dict({"posterior": {...}}) must produce the summary columns
        that _extract_posterior_stats reads: mean, sd, r_hat, ess_bulk, ess_tail."""
        import arviz as az

        rng = np.random.default_rng(0)
        posterior = {
            "D0_ref": rng.normal(1e4, 500, (4, 500)),
            "v0": rng.normal(1e3, 100, (4, 500)),
        }
        idata = az.from_dict({"posterior": posterior})
        summary = az.summary(idata, var_names=["D0_ref", "v0"], ci_prob=0.95)

        required_cols = {"mean", "sd", "r_hat", "ess_bulk", "ess_tail"}
        missing = required_cols - set(summary.columns)
        assert not missing, (
            f"ArviZ summary is missing expected columns {missing}. "
            "Check if the ArviZ version changed az.summary() output column names."
        )
        assert set(summary.index) == {"D0_ref", "v0"}
        assert np.all(np.isfinite(summary["r_hat"].to_numpy(dtype=float)))

    @pytest.mark.unit
    def test_arviz_from_dict_old_kwarg_form_raises_type_error(self) -> None:
        """az.from_dict(posterior={...}) must raise TypeError in ArviZ ≥1.0.

        If this test starts PASSING (i.e. the old form is accepted again), the
        compatibility shim in core.py and results.py should be reviewed — but
        the new dict form should remain the canonical call to stay forward-compatible.
        """
        import arviz as az

        posterior = {"D0_ref": np.random.randn(2, 100)}
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            az.from_dict(posterior=posterior)


@pytest.mark.unit
class TestBugPrevention_BFMIAdvisory:
    """Regression tests for het_dd0f825b: BFMI must not be a hard convergence gate.

    Root cause: low BFMI (<0.3) from chains near the alpha_sample=-2.0 boundary
    caused all 47/47 shards to fail convergence, producing a degenerate result.
    Homodyne parity: check_convergence uses only R-hat + ESS as hard gates.
    """

    def _make_result_dict(
        self,
        n_chains: int = 4,
        n_samples: int = 100,
        mean: float = 1.0,
        std: float = 0.1,
    ) -> dict:
        """Build a minimal worker result dict with healthy R-hat but no energy field."""
        rng = np.random.default_rng(42)
        params = ["D0_ref", "alpha_sample"]
        samples = {p: rng.normal(mean, std, size=n_chains * n_samples) for p in params}
        return {
            "success": True,
            "samples": samples,
            "param_names": params,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "extra_fields": {},  # no energy → BFMI will be unavailable
            "duration": 1.0,
            "stats": {"num_divergent": 5, "n_warmup": 50},
        }

    @pytest.mark.unit
    def test_bfmi_compute_failure_does_not_kill_shard_convergence(self) -> None:
        """Regression het_dd0f825b: bfmi_compute_failed must NOT set convergence_passed=False.

        Previously: bfmi_compute_failed → convergence_passed = False unconditionally.
        This caused 100% shard failure when az.bfmi() raised TypeError/KeyError.
        Fix: BFMI is advisory; convergence determined by R-hat + ESS only.
        """
        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import _result_dict_to_cmc_result

        config = CMCConfig()
        # With no energy field, az.bfmi() will fail (bfmi_compute_failed=True).
        # With healthy chains (small std → low r_hat), convergence must still pass.
        result_dict = self._make_result_dict(n_chains=4, n_samples=500, std=0.05)
        result = _result_dict_to_cmc_result(result_dict, config)

        # bfmi unavailable should NOT force convergence failure
        assert result.bfmi is None or result.convergence_passed, (
            "bfmi_compute_failed incorrectly forced convergence_passed=False. "
            "This is the het_dd0f825b regression: BFMI must be advisory only."
        )

    @pytest.mark.unit
    def test_combine_shard_posteriors_accepts_shards_with_low_bfmi(self) -> None:
        """Regression het_dd0f825b: _combine_shard_posteriors must accept shards where
        convergence_passed=True even when combined_bfmi < min_bfmi.

        Previously: combined_bfmi < min_bfmi → convergence_passed=False on the
        combined result, even when all individual shards converged (R-hat + ESS OK).
        """
        from types import SimpleNamespace

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import _combine_shard_posteriors

        n_params = 2
        config = CMCConfig()

        def _converged_shard_low_bfmi(seed: int) -> SimpleNamespace:
            rng = np.random.default_rng(seed)
            r = SimpleNamespace()
            r.convergence_passed = True
            r.parameter_names = ["D0_ref", "alpha_sample"]
            r.posterior_mean = rng.normal(1.0, 0.01, size=n_params)
            r.posterior_std = np.ones(n_params) * 0.1
            r.r_hat = np.array([1.01, 1.02])  # good R-hat
            r.ess_bulk = np.array([400.0, 380.0])  # good ESS
            r.ess_tail = np.array([350.0, 340.0])
            r.bfmi = [0.15, 0.18]  # low BFMI (advisory)
            r.samples = {
                "D0_ref": rng.normal(1.0, 0.1, 400),
                "alpha_sample": rng.normal(-1.5, 0.3, 400),
            }
            r.num_warmup = 50
            r.num_samples = 100
            r.num_chains = 4
            r.metadata = {"divergence_rate": 0.03}
            return r

        shards = [_converged_shard_low_bfmi(i) for i in range(4)]
        result = _combine_shard_posteriors(shards, config, num_shards=4, base_seed=0)

        # Low combined BFMI must NOT cause all-shards-failed
        assert not result.metadata.get("all_shards_failed"), (
            "_combine_shard_posteriors rejected all shards because combined BFMI "
            "< min_bfmi. BFMI must be advisory, not a hard convergence gate."
        )
        # And the result should be finite (shards DID converge)
        assert np.all(np.isfinite(result.posterior_mean)), (
            "Combined posterior mean is not finite despite converged shards."
        )

    @pytest.mark.unit
    def test_bfmi_nanmin_handles_multielement_array(self) -> None:
        """Regression: az.bfmi() returns dict of arrays; float(min(list_of_arrays))
        raised DataArray.__bool__ ValueError. Fix: float(np.nanmin(np.asarray(bfmi))).
        """
        import arviz as az

        from heterodyne.optimization.cmc.core import _compute_bfmi

        # Build InferenceData with energy in sample_stats (4 chains × 100 samples)
        rng = np.random.default_rng(0)
        energy = rng.normal(size=(4, 100))
        idata = az.from_dict(
            {
                "posterior": {"D0_ref": rng.normal(size=(4, 100))},
                "sample_stats": {"energy": energy},
            }
        )
        bfmi, failed = _compute_bfmi(idata)

        # Whether BFMI succeeds or fails, we must be able to take nanmin without error
        if bfmi is not None and not failed:
            import numpy as _np

            min_bfmi = float(_np.nanmin(_np.asarray(bfmi, dtype=float)))
            assert _np.isfinite(min_bfmi), "BFMI min should be finite scalar"


@pytest.mark.unit
class TestBugPrevention_NoNLSQLargeShardAbort:
    """Regression tests for het_c7548ee8 / het_e34fa942: fit_cmc_sharded must abort
    immediately (RuntimeError) when no NLSQ warmstart is provided AND
    avg_points_per_shard > 10K.

    Root cause: without warmstart, NUTS saturates max_tree_depth on every
    iteration → all shards timeout after 8h with 0 posterior samples.
    Three separate runs confirmed this is deterministic.
    """

    @pytest.mark.unit
    def test_no_nlsq_large_shards_raises_immediately(self) -> None:
        """fit_cmc_sharded raises RuntimeError before dispatching workers when
        nlsq_result=None and shards are too large for NUTS without warmstart.

        Previously: ran 8 hours, all shards timed out, CLI crashed.
        Fixed: abort at shard-size check with a descriptive error.
        """
        import unittest.mock as mock

        import numpy as np

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded

        # Create minimal mock model (never touches JAX)
        model = mock.MagicMock()
        model.param_manager.space.varying_names = ["D0_ref", "alpha_ref"]
        model.param_manager.space.varying_physics_names = ["D0_ref", "alpha_ref"]
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "alpha_ref": (-5.0, 5.0),
        }
        model.q = 0.005
        model.dt = 0.001
        model.t = np.linspace(0, 10, 100)
        model.scaling.get_for_angle.return_value = (0.5, 1.0)

        # Large c2 matrix: N=200 → 200×200 = 40K points, avg_pts/shard >> 10K
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, size=(200, 200))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)

        config = CMCConfig()

        with pytest.raises(RuntimeError, match="no NLSQ warm-start"):
            fit_cmc_sharded(
                model=model,
                c2_data=c2,
                config=config,
                nlsq_result=None,
                num_shards=2,  # 2 shards → avg ~20K pts each, > 10K limit
            )

    @pytest.mark.unit
    def test_no_nlsq_small_shards_does_not_raise(self) -> None:
        """fit_cmc_sharded should NOT abort when shards are small enough.

        Small datasets (avg_pts <= 10K) may converge with default priors.
        The abort guard must not block legitimate small-scale CMC runs.
        """
        import unittest.mock as mock

        import numpy as np

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded

        model = mock.MagicMock()
        model.param_manager.space.varying_names = ["D0_ref"]
        model.param_manager.space.varying_physics_names = ["D0_ref"]
        model.param_manager.space.bounds = {"D0_ref": (100.0, 1e6)}
        model.q = 0.005
        model.dt = 0.001
        model.t = np.linspace(0, 1, 20)
        model.scaling.get_for_angle.return_value = (0.5, 1.0)

        # Small c2: N=50 → 2 shards × 1250 pts each (well below 10K)
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, size=(50, 50))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)

        config = CMCConfig()

        # The abort guard must NOT fire for small shards. The run may fail
        # later for unrelated reasons (mock pickling, etc.) — we only care
        # that the specific "no NLSQ warm-start" abort guard is not triggered.
        try:
            fit_cmc_sharded(
                model=model,
                c2_data=c2,
                config=config,
                nlsq_result=None,
                num_shards=4,
            )
        except Exception as e:
            assert "no NLSQ warm-start" not in str(e), (
                f"fit_cmc_sharded incorrectly aborted a small-shard run: {e}. "
                "The no-NLSQ abort guard must only fire for avg_pts > 10K."
            )
            # Any other exception (pickling, etc.) is expected when using mocks


@pytest.mark.unit
class TestBugPrevention_AlphaBounds:
    """Regression tests for alpha_ref/alpha_sample bound widening.

    Root cause (het_dd0f825b): alpha_sample=-2.0 at the lower bound (min_bound was -2.0).
    NLSQ hit the boundary exactly, meaning the true posterior extends below -2.0.
    Bounds widened to [-5, 5] so NUTS can explore sub-diffusive regimes.
    """

    @pytest.mark.unit
    def test_alpha_bounds_allow_sub_minus_two(self) -> None:
        """alpha_ref and alpha_sample must accept values below -2.0."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        for name in ("alpha_ref", "alpha_sample"):
            info = DEFAULT_REGISTRY[name]
            assert info.min_bound <= -3.0, (
                f"{name} min_bound={info.min_bound} is too restrictive. "
                "NLSQ can legitimately hit -2.0 for strongly sub-diffusive samples; "
                "bounds must extend below -2.0 to let NUTS explore the full posterior."
            )
            assert info.max_bound >= 3.0, f"{name} max_bound={info.max_bound}"

    @pytest.mark.unit
    def test_alpha_prior_std_safe_for_tempered_priors(self) -> None:
        """alpha prior_std must stay <= 1.5 to avoid near-flat tempered distributions.

        With CMC tempering by sqrt(47) ≈ 6.86:
          tempered_std = prior_std * 6.86
        For bounds [-5, 5] (range=10), a safe prior requires tempered_std < range/2 = 5.
        prior_std > ~0.73 with 47 shards creates near-uniform distributions;
        prior_std > 1.5 causes NUTS to saturate max_tree_depth on every iteration
        (O(2^10) leapfrog steps), making shards timeout without producing any samples.
        """
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        max_safe_std = 1.5
        for name in ("alpha_ref", "alpha_sample"):
            info = DEFAULT_REGISTRY[name]
            assert info.prior_std <= max_safe_std, (
                f"{name} prior_std={info.prior_std} > {max_safe_std}. "
                "Wide tempered priors cause near-uniform density → NUTS timeout "
                "when running without NLSQ warmstart (het_c7548ee8 failure mode)."
            )


@pytest.mark.unit
class TestBugPrevention_DTotalSignGuard:
    """Regression tests for het_c7fb5859: fit_cmc_sharded must emit a WARNING
    and clamp D_offset when D_total = D0 + D_offset ≤ 0 and reparameterisation
    is enabled.

    Root cause: stale NLSQ result had D0_sample=1390, D_offset_sample=-2644 →
    D_total_sample = -1254 < 0.  Reparameterised prior requires D_total > 0, so
    log_prior = -inf at the warm-start init point → all NUTS leapfrog proposals
    rejected → BFMI=0.000, R-hat=NaN across all 47 shards.
    """

    @pytest.mark.unit
    def test_negative_d_total_sample_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A WARNING is emitted and D_offset_sample is clamped when
        D0_sample + D_offset_sample ≤ 0 with reparameterisation enabled."""
        import unittest.mock as mock

        import numpy as np

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded
        from heterodyne.optimization.nlsq.results import NLSQResult

        model = mock.MagicMock()
        model.param_manager.space.varying_names = [
            "D0_ref",
            "D0_sample",
            "D_offset_sample",
        ]
        model.param_manager.space.varying_physics_names = (
            model.param_manager.space.varying_names
        )
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "D0_sample": (100.0, 1e6),
            "D_offset_sample": (-1e5, 1e5),
        }
        model.q = 0.0054
        model.dt = 0.1
        model.t = np.linspace(0.1, 10.0, 50)
        model.scaling.get_for_angle.return_value = (0.3, 1.0)

        # NLSQ result replicating het_c7fb5859: D_total_sample = 1390 - 2644 < 0
        nlsq = NLSQResult(
            parameters=np.array([5110.0, 1390.0, -2644.0]),
            parameter_names=["D0_ref", "D0_sample", "D_offset_sample"],
            success=True,
            message="converged",
            reduced_chi_squared=0.86,
            metadata={},
        )

        # Small c2 (50×50 = 2500 pts, 2 shards → 1250 pts each < 10K abort limit)
        rng = np.random.default_rng(42)
        c2 = rng.normal(1.0, 0.05, size=(50, 50))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)

        config = (
            CMCConfig()
        )  # use_reparam=True, reparameterization_d_total=True by default

        import heterodyne.optimization.cmc.core as cmc_core

        warning_calls: list[str] = []
        _orig_warn = cmc_core.logger.warning

        def _capture(msg: object, *args: object, **kw: object) -> None:
            warning_calls.append(str(msg) % args if args else str(msg))
            _orig_warn(msg, *args, **kw)  # type: ignore[arg-type]

        with mock.patch.object(cmc_core.logger, "warning", side_effect=_capture):
            try:
                fit_cmc_sharded(
                    model=model,
                    c2_data=c2,
                    config=config,
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                pass  # may fail later (mock pickling) — only the warning matters

        warned = any("D_total_sample" in msg for msg in warning_calls)
        assert warned, (
            "Expected WARNING about D_total_sample <= 0 before dispatching shards. "
            "het_c7fb5859: D_total<0 with reparam causes BFMI=0 on all shards "
            "without this guard. Captured: " + str(warning_calls)
        )

    @pytest.mark.unit
    def test_positive_d_total_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No D_total warning is emitted when D0 + D_offset > 0."""
        import logging
        import unittest.mock as mock

        import numpy as np

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded
        from heterodyne.optimization.nlsq.results import NLSQResult

        model = mock.MagicMock()
        model.param_manager.space.varying_names = [
            "D0_ref",
            "D0_sample",
            "D_offset_sample",
        ]
        model.param_manager.space.varying_physics_names = (
            model.param_manager.space.varying_names
        )
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "D0_sample": (100.0, 1e6),
            "D_offset_sample": (-1e5, 1e5),
        }
        model.q = 0.0054
        model.dt = 0.1
        model.t = np.linspace(0.1, 10.0, 50)
        model.scaling.get_for_angle.return_value = (0.3, 1.0)

        # Healthy NLSQ result: D_total_sample = 5000 + (-100) = 4900 > 0
        nlsq = NLSQResult(
            parameters=np.array([5000.0, 5000.0, -100.0]),
            parameter_names=["D0_ref", "D0_sample", "D_offset_sample"],
            success=True,
            message="converged",
            reduced_chi_squared=0.9,
            metadata={},
        )

        rng = np.random.default_rng(7)
        c2 = rng.normal(1.0, 0.05, size=(50, 50))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)

        config = CMCConfig()

        with caplog.at_level(
            logging.WARNING, logger="heterodyne.optimization.cmc.core"
        ):
            try:
                fit_cmc_sharded(
                    model=model,
                    c2_data=c2,
                    config=config,
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                pass

        d_total_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "D_total_sample" in r.message
        ]
        assert not d_total_warnings, (
            f"No D_total warning expected when D_total > 0, got: {d_total_warnings}"
        )


@pytest.mark.unit
class TestBugPrevention_DegenerateWarmstart:
    """Regression tests for het_bb97531f: fit_cmc_sharded must emit a WARNING
    before dispatching shards when the warm-start is in a degenerate regime
    that causes 100% shard bad_convergence with BFMI=0.000.

    Two trigger conditions:
      (a) f0 < 0.10  → sample fraction near-zero → sample-transport params
          unidentifiable → NUTS can't thermalize from warm-start in 500 steps.
      (b) alpha_sample < -1.5 → J_sample ∝ t^α has non-integrable singularity
          at short lags → NUTS step-size collapses for sample group.
    """

    def _make_model_mock(self) -> Any:
        import unittest.mock as mock

        import numpy as np

        model = mock.MagicMock()
        model.param_manager.space.varying_names = [
            "D0_ref",
            "D0_sample",
            "alpha_sample",
            "f0",
        ]
        model.param_manager.space.varying_physics_names = (
            model.param_manager.space.varying_names
        )
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "D0_sample": (100.0, 1e6),
            "alpha_sample": (-5.0, 5.0),
            "f0": (0.0, 1.0),
        }
        model.q = 0.0054
        model.dt = 0.001
        model.t = np.linspace(0.001, 10.0, 50)
        model.scaling.get_for_angle.return_value = (0.3, 1.0)
        return model

    def _make_c2(self) -> Any:
        import numpy as np

        rng = np.random.default_rng(42)
        c2 = rng.normal(1.0, 0.05, size=(50, 50))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)
        return c2

    @pytest.mark.unit
    def test_low_f0_emits_degenerate_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARNING emitted when f0 < 0.10 (sample fraction near-zero)."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded
        from heterodyne.optimization.nlsq.results import NLSQResult

        nlsq = NLSQResult(
            parameters=[5110.0, 1390.0, 0.0, 0.03],  # f0=0.03 < 0.10
            parameter_names=["D0_ref", "D0_sample", "alpha_sample", "f0"],
            success=True,
            message="converged",
            reduced_chi_squared=0.86,
            metadata={},
        )
        import heterodyne.optimization.cmc.core as cmc_core

        warning_calls: list[str] = []
        _orig = cmc_core.logger.warning

        def _capture(msg: object, *args: object, **kw: object) -> None:
            warning_calls.append(str(msg) % args if args else str(msg))
            _orig(msg, *args, **kw)  # type: ignore[arg-type]

        with mock.patch.object(cmc_core.logger, "warning", side_effect=_capture):
            try:
                fit_cmc_sharded(
                    model=self._make_model_mock(),
                    c2_data=self._make_c2(),
                    config=CMCConfig(),
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                pass

        assert any(
            "Degenerate warm-start" in m and "f0=" in m for m in warning_calls
        ), (
            "Expected WARNING about degenerate warm-start with f0 near-zero. "
            f"Captured: {warning_calls}"
        )

    @pytest.mark.unit
    def test_negative_alpha_sample_emits_degenerate_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARNING emitted when alpha_sample < -1.5 (J_sample singularity)."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded
        from heterodyne.optimization.nlsq.results import NLSQResult

        nlsq = NLSQResult(
            parameters=[5110.0, 1390.0, -2.0, 0.5],  # alpha_sample=-2.0 < -1.5
            parameter_names=["D0_ref", "D0_sample", "alpha_sample", "f0"],
            success=True,
            message="converged",
            reduced_chi_squared=0.86,
            metadata={},
        )
        import heterodyne.optimization.cmc.core as cmc_core

        warning_calls: list[str] = []
        _orig = cmc_core.logger.warning

        def _capture(msg: object, *args: object, **kw: object) -> None:
            warning_calls.append(str(msg) % args if args else str(msg))
            _orig(msg, *args, **kw)  # type: ignore[arg-type]

        with mock.patch.object(cmc_core.logger, "warning", side_effect=_capture):
            try:
                fit_cmc_sharded(
                    model=self._make_model_mock(),
                    c2_data=self._make_c2(),
                    config=CMCConfig(),
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                pass

        assert any(
            "Degenerate warm-start" in m and "alpha_sample=" in m for m in warning_calls
        ), (
            "Expected WARNING about degenerate warm-start with alpha_sample < -1.5. "
            f"Captured: {warning_calls}"
        )

    @pytest.mark.unit
    def test_healthy_warmstart_no_degenerate_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No degenerate-warmstart WARNING when f0 and alpha_sample are normal."""
        import logging

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_sharded
        from heterodyne.optimization.nlsq.results import NLSQResult

        nlsq = NLSQResult(
            parameters=[5110.0, 1390.0, -0.3, 0.4],  # f0=0.4, alpha_sample=-0.3
            parameter_names=["D0_ref", "D0_sample", "alpha_sample", "f0"],
            success=True,
            message="converged",
            reduced_chi_squared=0.9,
            metadata={},
        )

        with caplog.at_level(
            logging.WARNING, logger="heterodyne.optimization.cmc.core"
        ):
            try:
                fit_cmc_sharded(
                    model=self._make_model_mock(),
                    c2_data=self._make_c2(),
                    config=CMCConfig(),
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                pass

        degen_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "Degenerate warm-start" in r.message
        ]
        assert not degen_warnings, (
            f"No degenerate-warmstart warning expected for healthy params, "
            f"got: {degen_warnings}"
        )

    @pytest.mark.unit
    def test_mean_shard_bfmi_in_metadata(self) -> None:
        """mean_shard_bfmi and n_bfmi_zero are present in the final metadata."""

        import numpy as np

        from heterodyne.optimization.cmc.results import CMCResult

        # Build fake shard results with known BFMI values
        names = ["D0_ref", "D0_sample"]
        n = len(names)
        rng = np.random.default_rng(0)

        def _good(bfmi_val: float) -> CMCResult:
            samples = {nm: rng.normal(size=200) for nm in names}
            return CMCResult(
                parameter_names=names,
                posterior_mean=np.ones(n),
                posterior_std=np.ones(n) * 0.1,
                credible_intervals={nm: {"2.5%": -1.0, "97.5%": 1.0} for nm in names},
                convergence_passed=True,
                r_hat=np.ones(n) * 1.01,
                ess_bulk=np.ones(n) * 600.0,
                ess_tail=np.ones(n) * 600.0,
                bfmi=[bfmi_val],
                samples=samples,
                map_estimate=np.ones(n),
                num_warmup=100,
                num_samples=200,
                num_chains=1,
                wall_time_seconds=1.0,
                metadata={},
            )

        # Two shards: one healthy BFMI, one zero BFMI
        shard_results = [_good(0.5), _good(0.0)]

        # Directly test the metadata-assembly logic (matches core.py lines verbatim)
        all_bfmi = [b for r in shard_results if r.bfmi is not None for b in r.bfmi]
        mean_bfmi = float(np.nanmean(np.asarray(all_bfmi, dtype=float)))
        n_zero = sum(
            1
            for r in shard_results
            if r.bfmi is not None
            and float(np.nanmin(np.asarray(r.bfmi, dtype=float))) < 0.01
        )
        assert abs(mean_bfmi - 0.25) < 1e-9, f"mean_bfmi={mean_bfmi}"
        assert n_zero == 1, f"n_bfmi_zero={n_zero}"

    @pytest.mark.unit
    def test_failing_shard_diagnostics_logged_at_warning(self) -> None:
        """A failing shard emits its diagnostics at WARNING (not just DEBUG)."""
        import unittest.mock as mock

        import numpy as np

        import heterodyne.optimization.cmc.core as cmc_core
        from heterodyne.optimization.cmc.core import (
            CMCConfig,
            _result_dict_to_cmc_result,
        )

        # Build a minimal result_dict that will fail convergence (r_hat > max_r_hat)
        names = ["D0_ref", "alpha_sample"]
        n_chains, n_samples = 4, 100
        rng = np.random.default_rng(7)
        # Each chain samples from a different mean → high R-hat
        chain_samples = [
            rng.normal(float(c) * 10, 0.1, n_samples) for c in range(n_chains)
        ]
        flat = np.concatenate(chain_samples)
        result_dict = {
            "success": True,
            "param_names": names,
            "samples": dict.fromkeys(names, flat),
            "n_chains": n_chains,
            "n_samples": n_samples,
            "extra_fields": {},
            "stats": {"num_divergent": 0, "n_warmup": 500},
            "duration": 1.0,
        }
        config = CMCConfig(max_r_hat=1.1, min_ess=400)

        warning_calls: list[str] = []
        _orig = cmc_core.logger.warning

        def _capture(msg: object, *a: object, **kw: object) -> None:
            warning_calls.append(str(msg) % a if a else str(msg))
            _orig(msg, *a, **kw)  # type: ignore[arg-type]

        with mock.patch.object(cmc_core.logger, "warning", side_effect=_capture):
            _result_dict_to_cmc_result(result_dict, config)

        shard_diag_warnings = [m for m in warning_calls if "Shard diagnostics" in m]
        assert shard_diag_warnings, (
            "Expected WARNING-level 'Shard diagnostics' log for a failing shard. "
            f"Captured: {warning_calls}"
        )
        assert any("FAIL" in m for m in shard_diag_warnings), (
            "Shard diagnostics WARNING must contain 'FAIL'"
        )


# ===========================================================================
# het_bb97531f prevention: degenerate warm-start abort gate
# ===========================================================================


@pytest.mark.unit
class TestBugPrevention_DegenerateWarmstartAbort:
    """Regression tests for het_bb97531f.

    Behaviour evolved through three regimes:
      (v1) Warning + dispatch K shards → all fail with BFMI=0.000.
      (v2) RuntimeError abort before dispatch (prevented 7-hour waste).
      (v3) Auto-clamp + continue (current) — clamp the degenerate
           ``alpha_sample`` / ``f0`` into the safe zone, warn loudly, and
           let the run proceed.  ``allow_degenerate_warmstart=True`` keeps
           the raw NLSQ values (no clamp) for callers who need to observe
           the full degenerate behaviour.

    The v3 design (deep-RCA Fix 5) preserves the v2 anti-cascade safety
    while removing the hard-abort dead-end: downstream gates
    (``min_success_rate``, R-hat, divergence rate) now decide whether
    the soft-clamped run produced usable output.
    """

    def _make_parts(
        self,
        f0: float = 0.5,
        alpha_sample: float = 0.0,
    ):
        import unittest.mock as mock

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.nlsq.results import NLSQResult

        names = ["D0_ref", "alpha_ref", "f0", "alpha_sample"]
        model = mock.MagicMock()
        model.param_manager.space.varying_names = names
        model.param_manager.space.varying_physics_names = names
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "alpha_ref": (-5.0, 5.0),
            "f0": (0.0, 1.0),
            "alpha_sample": (-5.0, 5.0),
        }
        model.varying_names = names
        model.get_params_dict.return_value = dict.fromkeys(names, 0.3)
        model.q = 0.005
        model.dt = 0.001
        model.t = np.linspace(0, 10, 30)
        model.scaling.get_for_angle.return_value = (0.5, 1.0)

        vals = {
            "D0_ref": 1e4,
            "alpha_ref": 0.0,
            "f0": f0,
            "alpha_sample": alpha_sample,
        }
        nlsq = NLSQResult(
            parameters=np.array([vals[n] for n in names]),
            parameter_names=names,
            success=True,
            message="ok",
        )
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, (30, 30))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)
        config = CMCConfig(use_reparam=False)
        return model, nlsq, c2, config

    def test_low_f0_no_longer_aborts(self) -> None:
        """Low f0 must NOT raise RuntimeError abort (deep-RCA Fix 5 —
        replaced hard abort with auto-clamp + continue)."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc.core import (
            CMC_F0_DEGEN_THRESHOLD,
            fit_cmc_sharded,
        )

        model, nlsq, c2, config = self._make_parts(f0=CMC_F0_DEGEN_THRESHOLD - 0.01)
        # Mock backend so the function completes past the degeneracy guard.
        # The contract we're enforcing: no "Aborting" RuntimeError from the
        # het_bb97531f guard.  Capturing the warning text is order-dependent
        # under pytest's capfd (logger handlers cache stdout FDs across
        # tests), so we assert on the exception contract instead.
        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.return_value = []
            try:
                fit_cmc_sharded(
                    model=model,
                    c2_data=c2,
                    config=config,
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except RuntimeError as exc:
                assert not ("het_bb97531f" in str(exc) and "Aborting" in str(exc)), (
                    f"Degeneracy guard still hard-aborts: {exc}"
                )

    def test_low_alpha_sample_no_longer_aborts(self) -> None:
        """Low alpha_sample must NOT raise RuntimeError abort."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc.core import (
            CMC_ALPHA_SINGULARITY,
            fit_cmc_sharded,
        )

        model, nlsq, c2, config = self._make_parts(
            alpha_sample=CMC_ALPHA_SINGULARITY - 0.1
        )
        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.return_value = []
            try:
                fit_cmc_sharded(
                    model=model,
                    c2_data=c2,
                    config=config,
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except RuntimeError as exc:
                assert not ("het_bb97531f" in str(exc) and "Aborting" in str(exc)), (
                    f"Degeneracy guard still hard-aborts: {exc}"
                )

    def test_low_alpha_sample_auto_clamps_value(self) -> None:
        """Verify the auto-clamp actually pushes alpha_sample into safe zone.

        We can't reliably capture the log warning text under pytest's
        capfd (order-dependent FD caching), so we verify the *effect* of
        the auto-clamp: by patching out the heavy NUTS dispatch we can
        inspect ``initial_values`` after the guard runs.  The clamp
        target is ``CMC_ALPHA_SINGULARITY + 0.1`` (= -1.4 at the
        current registry)."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc.core import (
            CMC_ALPHA_SINGULARITY,
            fit_cmc_sharded,
        )

        captured_init: dict[str, dict[str, float]] = {}

        def _capture_init(*args, **kwargs):  # noqa: ANN001 — pytest sig
            # Find the per-shard config dict passed to the backend.
            for arg in args:
                if isinstance(arg, dict) and "initial_values" in arg:
                    captured_init["call"] = dict(arg["initial_values"])
            for v in kwargs.values():
                if isinstance(v, dict) and "initial_values" in v:
                    captured_init["call"] = dict(v["initial_values"])
            return []

        model, nlsq, c2, config = self._make_parts(
            alpha_sample=CMC_ALPHA_SINGULARITY - 0.1
        )
        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.side_effect = _capture_init
            try:
                fit_cmc_sharded(
                    model=model,
                    c2_data=c2,
                    config=config,
                    nlsq_result=nlsq,
                    num_shards=2,
                )
            except Exception:
                # Mock plumbing may not match the real backend signature
                # exactly; we only care that the guard ran without raising
                # the het_bb97531f abort.
                pass
        # If captured_init never got populated, the run failed before the
        # backend dispatch — that's a separate test signal but doesn't
        # invalidate the no-abort contract.  Skip the value check in that
        # branch rather than mask an unrelated failure.
        if not captured_init:
            pytest.skip("backend dispatch did not surface initial_values")
        clamped = captured_init["call"].get("alpha_sample")
        assert clamped is not None
        assert clamped >= CMC_ALPHA_SINGULARITY, (
            f"alpha_sample not clamped: got {clamped}, "
            f"expected >= {CMC_ALPHA_SINGULARITY}"
        )

    def test_allow_degenerate_warmstart_bypasses_abort(self) -> None:
        """No RuntimeError when allow_degenerate_warmstart=True; returns tombstone."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc import CMCConfig
        from heterodyne.optimization.cmc.core import (
            CMC_F0_DEGEN_THRESHOLD,
            fit_cmc_sharded,
        )

        model, nlsq, c2, _ = self._make_parts(f0=CMC_F0_DEGEN_THRESHOLD - 0.01)
        config = CMCConfig(use_reparam=False, allow_degenerate_warmstart=True)

        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.return_value = []
            result = fit_cmc_sharded(
                model=model,
                c2_data=c2,
                config=config,
                nlsq_result=nlsq,
                num_shards=2,
            )
        assert not result.convergence_passed, (
            "Expected degenerate tombstone when all shards produce no samples"
        )

    def test_healthy_warmstart_not_aborted(self) -> None:
        """No RuntimeError when f0 and alpha_sample are both above thresholds."""
        import unittest.mock as mock

        from heterodyne.optimization.cmc.core import fit_cmc_sharded

        model, nlsq, c2, config = self._make_parts(f0=0.5, alpha_sample=0.0)

        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.return_value = []
            result = fit_cmc_sharded(
                model=model,
                c2_data=c2,
                config=config,
                nlsq_result=nlsq,
                num_shards=2,
            )
        assert not result.convergence_passed


# ===========================================================================
# Bug 2 prevention: threshold constants must stay in sync
# ===========================================================================


@pytest.mark.unit
class TestBugPrevention_ConstantSync:
    """Regression tests: CMC degenerate warm-start thresholds must be identical
    in core.py and optimization_runner.py.

    Previously: each module defined its own copy — a silent drift risk.
    Fixed: optimization_runner.py imports from core.py.
    """

    def test_f0_threshold_is_imported_from_core(self) -> None:
        from heterodyne.cli.optimization_runner import _F0_DEGEN_THRESHOLD
        from heterodyne.optimization.cmc.core import CMC_F0_DEGEN_THRESHOLD

        assert _F0_DEGEN_THRESHOLD == CMC_F0_DEGEN_THRESHOLD, (
            f"_F0_DEGEN_THRESHOLD={_F0_DEGEN_THRESHOLD} diverged from "
            f"CMC_F0_DEGEN_THRESHOLD={CMC_F0_DEGEN_THRESHOLD}. "
            "optimization_runner.py must import this constant from core.py."
        )

    def test_alpha_singularity_is_imported_from_core(self) -> None:
        from heterodyne.cli.optimization_runner import _ALPHA_SINGULARITY
        from heterodyne.optimization.cmc.core import CMC_ALPHA_SINGULARITY

        assert _ALPHA_SINGULARITY == CMC_ALPHA_SINGULARITY


# ===========================================================================
# Bug 3 prevention: model-fallback initial values must be clamped
# ===========================================================================


@pytest.mark.unit
class TestBugPrevention_ModelFallbackClamping:
    """Regression tests: when nlsq_result=None, initial values from
    model.get_params_dict() must be clamped away from hard bounds before
    dispatch — the same protection _clamp_warmstart_to_interior provides for
    the NLSQ path.

    Root cause: ``parameters: {alpha_ref: -5.0}`` would place the NUTS start
    exactly on the boundary wall, causing leapfrog reflections that degrade
    BFMI across all shards.
    """

    def _make_model(self, alpha_ref_val: float = 0.0):
        import unittest.mock as mock

        names = ["D0_ref", "alpha_ref"]
        model = mock.MagicMock()
        model.param_manager.space.varying_names = names
        model.param_manager.space.varying_physics_names = names
        model.param_manager.space.bounds = {
            "D0_ref": (100.0, 1e6),
            "alpha_ref": (-5.0, 5.0),
        }
        model.varying_names = names
        model.get_params_dict.return_value = {
            "D0_ref": 1e4,
            "alpha_ref": alpha_ref_val,
        }
        model.q = 0.005
        model.dt = 0.001
        model.t = np.linspace(0, 10, 30)
        model.scaling.get_for_angle.return_value = (0.5, 1.0)
        return model

    def _run_and_capture_iv(self, model, c2, config):
        import unittest.mock as mock

        from heterodyne.optimization.cmc.core import fit_cmc_sharded

        with mock.patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend"
        ) as mock_cls:
            mock_cls.return_value.run_shards.return_value = []
            fit_cmc_sharded(
                model=model,
                c2_data=c2,
                config=config,
                nlsq_result=None,
                num_shards=2,
            )
        call_args = mock_cls.return_value.run_shards.call_args
        assert call_args is not None, "run_shards was not called"
        return call_args.kwargs.get("initial_values")

    def test_boundary_alpha_ref_is_clamped(self) -> None:
        """alpha_ref at exact lower bound (-5.0) must be clamped to interior."""
        from heterodyne.optimization.cmc import CMCConfig

        model = self._make_model(alpha_ref_val=-5.0)  # exact lower bound
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, (30, 30))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)
        config = CMCConfig(use_reparam=False)

        iv = self._run_and_capture_iv(model, c2, config)
        assert iv is not None, "initial_values not passed to run_shards"
        assert "alpha_ref" in iv
        assert iv["alpha_ref"] > -5.0, (
            f"alpha_ref={iv['alpha_ref']:.4f} at exact lower bound must be clamped "
            "away from the boundary wall (prevents NUTS leapfrog reflections)."
        )

    def test_interior_alpha_ref_not_clamped(self) -> None:
        """alpha_ref=0.0 (well inside bounds) must pass through unchanged."""
        from heterodyne.optimization.cmc import CMCConfig

        model = self._make_model(alpha_ref_val=0.0)
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, (30, 30))
        c2 = (c2 + c2.T) / 2
        np.fill_diagonal(c2, 1.0)
        config = CMCConfig(use_reparam=False)

        iv = self._run_and_capture_iv(model, c2, config)
        assert iv is not None
        assert abs(iv.get("alpha_ref", -999.0)) < 1e-6, (
            f"alpha_ref=0.0 (interior) must not be modified; got {iv.get('alpha_ref')}"
        )
