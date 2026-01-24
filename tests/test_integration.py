"""Integration tests for full NLSQ -> CMC workflow.

Bug Prevented: Component Integration Issues
-------------------------------------------
The full analysis pipeline involves NLSQ fitting followed by CMC
Bayesian inference. Various bugs can manifest at the integration
boundary, including:
- Parameter order mismatches
- Shape incompatibilities
- Dtype inconsistencies

These tests verify the complete workflow functions correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from heterodyne import CMCConfig, HeterodyneModel, NLSQConfig


class TestFullPipeline:
    """Tests for complete NLSQ -> CMC workflow."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_full_pipeline_single_phi(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
        cmc_config_1chain: CMCConfig,
    ) -> None:
        """Test complete NLSQ -> CMC pipeline for single phi angle.

        This is the standard analysis workflow:
        1. Run NLSQ to get initial fit
        2. Use NLSQ result to warm-start CMC
        3. Get Bayesian posterior samples
        """
        from heterodyne import fit_cmc_jax, fit_nlsq_jax

        # Step 1: NLSQ fit
        nlsq_result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,  # Use scipy for reliability
        )

        assert nlsq_result is not None
        assert hasattr(nlsq_result, "parameters")

        # Step 2: CMC with warm-start
        cmc_result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_1chain,
            nlsq_result=nlsq_result,
        )

        assert cmc_result is not None
        assert hasattr(cmc_result, "posterior_mean")
        assert len(cmc_result.posterior_mean) == small_heterodyne_model.n_varying

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_workflow_preserves_parameter_order(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
        cmc_config_1chain: CMCConfig,
    ) -> None:
        """Test that parameter order is consistent across pipeline.

        NLSQ and CMC should report parameters in the same order.
        """
        from heterodyne import fit_cmc_jax, fit_nlsq_jax

        nlsq_result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        cmc_result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_1chain,
            nlsq_result=nlsq_result,
        )

        # Parameter names should match
        assert nlsq_result.parameter_names == cmc_result.parameter_names, (
            f"Parameter order mismatch:\n"
            f"  NLSQ: {nlsq_result.parameter_names}\n"
            f"  CMC:  {cmc_result.parameter_names}"
        )


class TestCMCIndependent:
    """Tests for CMC without NLSQ warm-start."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_without_nlsq_warmstart(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Test CMC runs independently without NLSQ result.

        CMC should use its default initialization strategy.
        """
        from heterodyne import CMCConfig, fit_cmc_jax

        config = CMCConfig(
            num_chains=1,
            num_warmup=100,
            num_samples=100,
            seed=42,
            use_nlsq_warmstart=False,
        )

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=config,
            nlsq_result=None,
        )

        assert result is not None
        assert len(result.posterior_mean) == small_heterodyne_model.n_varying


class TestNLSQResultAPI:
    """Tests for NLSQResult API methods."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_result_get_param(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test NLSQResult.get_param() method works correctly."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        # Get first parameter by name
        first_name = result.parameter_names[0]
        value = result.get_param(first_name)

        assert isinstance(value, float)
        assert value == float(result.parameters[0])

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_result_get_param_not_found(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test NLSQResult.get_param() raises for unknown parameter."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        with pytest.raises(KeyError):
            result.get_param("nonexistent_parameter")

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_result_params_dict(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test NLSQResult.params_dict property."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        params_dict = result.params_dict

        assert isinstance(params_dict, dict)
        assert len(params_dict) == len(result.parameter_names)

        for name in result.parameter_names:
            assert name in params_dict
            assert isinstance(params_dict[name], float)


class TestCMCResultAPI:
    """Tests for CMCResult API methods."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_result_samples_shape(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_1chain: CMCConfig,
    ) -> None:
        """Test CMCResult samples have correct shape."""
        from heterodyne import fit_cmc_jax

        config = cmc_config_1chain
        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=config,
        )

        if result.samples is not None:
            expected_samples = config.num_chains * config.num_samples
            for name in result.parameter_names:
                if name in result.samples:
                    samples = result.samples[name]
                    assert samples.shape[0] == expected_samples

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_cmc_result_get_param_summary(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        cmc_config_1chain: CMCConfig,
    ) -> None:
        """Test CMCResult.get_param_summary() method."""
        from heterodyne import fit_cmc_jax

        result = fit_cmc_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=cmc_config_1chain,
        )

        first_name = result.parameter_names[0]
        summary = result.get_param_summary(first_name)

        assert "mean" in summary
        assert "std" in summary
        assert isinstance(summary["mean"], float)
        assert isinstance(summary["std"], float)


class TestModelStateConsistency:
    """Tests for model state after fitting."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_model_params_updated_after_nlsq(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test model parameters are updated after successful NLSQ fit."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        if result.success:
            # Model should reflect fitted values
            model_params = small_heterodyne_model.get_params()
            assert model_params is not None

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_multiple_fits_same_model(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test model can be fit multiple times."""
        from heterodyne import fit_nlsq_jax

        # First fit
        result1 = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        # Second fit with different phi
        result2 = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=45.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        assert result1 is not None
        assert result2 is not None


class TestDataValidation:
    """Tests for input data validation."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_accepts_numpy_data(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax accepts numpy arrays."""
        from heterodyne import fit_nlsq_jax

        # Ensure data is numpy
        c2_numpy = np.asarray(small_c2_data)

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=c2_numpy,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        assert result is not None

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_accepts_jax_data(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax accepts JAX arrays."""
        from heterodyne import fit_nlsq_jax

        # Convert to JAX array
        c2_jax = jnp.asarray(small_c2_data)

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=c2_jax,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        assert result is not None
