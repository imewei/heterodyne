"""Integration tests for the full heterodyne analysis pipeline.

End-to-end: data loading -> NLSQ warm-start -> CMC/NUTS -> diagnostics.

All tests use synthetic data with minimal problem sizes (n_times=16,
num_warmup=10, num_samples=20, num_chains=1) for CI speed.

Bug Prevented: Pipeline Integration Failures
---------------------------------------------
The full analysis pipeline (NLSQ -> CMC) can fail silently at
integration boundaries due to parameter order mismatches, shape
incompatibilities, or dtype inconsistencies. These tests exercise
the complete workflow with known-good synthetic data.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_model(n_times: int = 16):
    """Create a minimal HeterodyneModel for fast pipeline tests."""
    from heterodyne.core.heterodyne_model import HeterodyneModel

    config = {
        "temporal": {"dt": 1.0, "time_length": n_times},
        "scattering": {"wavevector_q": 0.01},
        "parameters": {},
    }
    return HeterodyneModel.from_config(config)


def _make_synthetic_c2(
    model, phi_angle: float = 0.0, noise_scale: float = 0.01, seed: int = 42
):
    """Generate synthetic correlation matrix with small additive noise."""
    import jax

    c2_clean = model.compute_correlation(phi_angle=phi_angle)
    c2_clean = jnp.asarray(c2_clean)
    key = jax.random.PRNGKey(seed)
    noise = (
        jax.random.normal(key, shape=c2_clean.shape)
        * noise_scale
        * jnp.max(jnp.abs(c2_clean))
    )
    return np.asarray(c2_clean + noise)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNLSQFitOnSyntheticData:
    """NLSQ fitting on synthetic correlation data."""

    @pytest.mark.integration
    def test_nlsq_returns_success(self) -> None:
        """NLSQ fit on synthetic data returns a result (may or may not converge)."""
        from heterodyne.optimization.nlsq.config import NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model)

        config = NLSQConfig(max_iterations=20, tolerance=1e-4, method="trf", verbose=0)

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=config,
            use_nlsq_library=False,
        )

        assert result is not None
        assert hasattr(result, "parameters")
        assert hasattr(result, "success")
        assert len(result.parameters) == model.n_varying
        assert result.parameter_names == model.varying_names

    @pytest.mark.integration
    def test_nlsq_result_has_finite_parameters(self) -> None:
        """NLSQ result parameters and any uncertainties are finite."""
        from heterodyne.optimization.nlsq.config import NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model)

        config = NLSQConfig(max_iterations=30, tolerance=1e-4, method="trf", verbose=0)

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=config,
            use_nlsq_library=False,
        )

        assert np.all(np.isfinite(result.parameters)), (
            f"Non-finite parameters: {result.parameters}"
        )

        if result.uncertainties is not None:
            assert np.all(np.isfinite(result.uncertainties)), (
                f"Non-finite uncertainties: {result.uncertainties}"
            )


class TestNLSQToCMCWarmStart:
    """NLSQ result fed to CMC as warm-start."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    def test_nlsq_to_cmc_warmstart(self) -> None:
        """NLSQ -> CMC warm-start pipeline produces a CMCResult."""
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax
        from heterodyne.optimization.nlsq.config import NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model)

        nlsq_config = NLSQConfig(
            max_iterations=20,
            tolerance=1e-4,
            method="trf",
            verbose=0,
        )
        nlsq_result = fit_nlsq_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=nlsq_config,
            use_nlsq_library=False,
        )

        cmc_config = CMCConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=42,
            use_nlsq_warmstart=True,
            chain_method="sequential",
            enable_checkpoints=False,
            adaptive_sampling=False,
        )

        cmc_result = fit_cmc_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=cmc_config,
            nlsq_result=nlsq_result,
        )

        assert cmc_result is not None
        assert hasattr(cmc_result, "posterior_mean")
        assert len(cmc_result.posterior_mean) == model.n_varying

    @pytest.mark.integration
    @pytest.mark.mcmc
    def test_cmc_returns_correct_parameter_names(self) -> None:
        """CMC posterior contains the same parameter names as the model."""
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model)

        cmc_config = CMCConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=42,
            use_nlsq_warmstart=False,
            chain_method="sequential",
            enable_checkpoints=False,
            adaptive_sampling=False,
        )

        cmc_result = fit_cmc_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=cmc_config,
        )

        # Parameter names from CMC should match the model's varying names
        assert set(cmc_result.parameter_names) == set(model.varying_names)

    @pytest.mark.integration
    @pytest.mark.mcmc
    def test_cmc_diagnostics_computed(self) -> None:
        """CMC result has R-hat and ESS diagnostics (possibly NaN for 1 chain)."""
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model)

        cmc_config = CMCConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=42,
            use_nlsq_warmstart=False,
            chain_method="sequential",
            enable_checkpoints=False,
            adaptive_sampling=False,
        )

        cmc_result = fit_cmc_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=cmc_config,
        )

        # Diagnostics should be present (arrays or None)
        assert hasattr(cmc_result, "r_hat")
        assert hasattr(cmc_result, "ess_bulk")

        # Posterior mean/std should be finite
        assert np.all(np.isfinite(cmc_result.posterior_mean))
        assert np.all(np.isfinite(cmc_result.posterior_std))


class TestFullPipelineEndToEnd:
    """Full pipeline: model creation -> NLSQ -> CMC -> result extraction."""

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_full_pipeline_model_to_results(self) -> None:
        """Complete pipeline: create model, NLSQ fit, CMC sampling, extract results."""
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax
        from heterodyne.optimization.nlsq.config import NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_jax

        # Step 1: Create model
        model = _make_small_model(n_times=16)
        assert model.n_times == 16
        assert model.n_varying > 0

        # Step 2: Generate synthetic data
        c2_data = _make_synthetic_c2(model, phi_angle=0.0, seed=123)
        assert c2_data.shape == (16, 16)

        # Step 3: NLSQ fit
        nlsq_config = NLSQConfig(
            max_iterations=30,
            tolerance=1e-4,
            method="trf",
            verbose=0,
        )
        nlsq_result = fit_nlsq_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=nlsq_config,
            use_nlsq_library=False,
        )
        assert nlsq_result is not None
        assert np.all(np.isfinite(nlsq_result.parameters))

        # Step 4: CMC with warm-start from NLSQ
        cmc_config = CMCConfig(
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=42,
            use_nlsq_warmstart=True,
            chain_method="sequential",
            enable_checkpoints=False,
            adaptive_sampling=False,
        )
        cmc_result = fit_cmc_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=cmc_config,
            nlsq_result=nlsq_result,
        )

        # Step 5: Extract and validate results
        assert cmc_result is not None
        assert len(cmc_result.posterior_mean) == model.n_varying
        assert len(cmc_result.posterior_std) == model.n_varying

        # Posterior means should be finite
        assert np.all(np.isfinite(cmc_result.posterior_mean))
        assert np.all(np.isfinite(cmc_result.posterior_std))

        # Parameter names should be consistent between NLSQ and CMC
        assert set(nlsq_result.parameter_names) == set(cmc_result.parameter_names)

        # get_param_summary should work for all parameters
        for name in cmc_result.parameter_names:
            summary = cmc_result.get_param_summary(name)
            assert "mean" in summary
            assert "std" in summary
            assert isinstance(summary["mean"], float)

    @pytest.mark.integration
    @pytest.mark.mcmc
    @pytest.mark.slow
    def test_pipeline_preserves_parameter_order(self) -> None:
        """Parameter ordering is consistent across NLSQ and CMC results."""
        from heterodyne.optimization.cmc.config import CMCConfig
        from heterodyne.optimization.cmc.core import fit_cmc_jax
        from heterodyne.optimization.nlsq.config import NLSQConfig
        from heterodyne.optimization.nlsq.core import fit_nlsq_jax

        model = _make_small_model(n_times=16)
        c2_data = _make_synthetic_c2(model, seed=99)

        nlsq_result = fit_nlsq_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=NLSQConfig(
                max_iterations=20, tolerance=1e-4, method="trf", verbose=0
            ),
            use_nlsq_library=False,
        )

        cmc_result = fit_cmc_jax(
            model=model,
            c2_data=c2_data,
            phi_angle=0.0,
            config=CMCConfig(
                num_warmup=10,
                num_samples=20,
                num_chains=1,
                seed=42,
                use_nlsq_warmstart=True,
                chain_method="sequential",
                enable_checkpoints=False,
                adaptive_sampling=False,
            ),
            nlsq_result=nlsq_result,
        )

        assert nlsq_result.parameter_names == cmc_result.parameter_names, (
            f"Parameter order mismatch:\n"
            f"  NLSQ: {nlsq_result.parameter_names}\n"
            f"  CMC:  {cmc_result.parameter_names}"
        )
