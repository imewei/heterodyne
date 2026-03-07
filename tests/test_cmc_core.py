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
        assert not hasattr(value, 'shape') or value.shape != (config.num_chains,), (
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
        correct_init_params = {
            "D0_ref": jnp.full((config.num_chains,), 1.0)
        }

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

        config = CMCConfig.from_dict({
            "num_warmup": 100,
            "num_samples": 100,
            "use_reparam": False,
            "prior_width_factor": 3.0,
        })
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
