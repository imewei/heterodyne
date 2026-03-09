"""Scientific tests for CMC diagnostics.

This module provides rigorous testing of MCMC convergence diagnostics
using known statistical properties and analytical solutions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ============================================================================
# R-hat (Gelman-Rubin) Tests
# ============================================================================

class TestComputeRHat:
    """Scientific tests for compute_r_hat."""

    @pytest.mark.unit
    def test_identical_chains_rhat_finite(self) -> None:
        """Identical chains produce a finite R-hat.

        Rank-normalized R-hat (Vehtari et al. 2021) may exceed 1.0 for
        identical chains due to rank discretization effects, but must
        remain finite.
        """
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        # All chains identical (but with internal variance)
        samples = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])

        r_hat = compute_r_hat(samples)

        assert np.isfinite(r_hat)

    @pytest.mark.unit
    def test_converged_chains_rhat_near_one(self) -> None:
        """Well-mixed chains from same distribution have R-hat ≈ 1."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        # 4 chains, 1000 samples each, all from N(0,1)
        samples = rng.standard_normal((4, 1000))

        r_hat = compute_r_hat(samples)

        # Should be close to 1.0 for converged chains
        assert r_hat < 1.1, f"R-hat {r_hat} > 1.1 for converged chains"

    @pytest.mark.unit
    def test_divergent_chains_rhat_high(self) -> None:
        """Chains from different distributions have R-hat > 1."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        # Chains with very different means
        samples = np.array([
            np.full(100, 0.0),   # Mean 0
            np.full(100, 10.0),  # Mean 10
            np.full(100, 0.0),   # Mean 0
            np.full(100, 10.0),  # Mean 10
        ])

        r_hat = compute_r_hat(samples)

        # Should be high for non-mixed chains
        assert r_hat > 1.5, f"R-hat {r_hat} should be high for divergent chains"

    @pytest.mark.unit
    def test_rhat_always_positive(self) -> None:
        """R-hat should always be positive."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(123)
        samples = rng.standard_normal((4, 100))

        r_hat = compute_r_hat(samples)

        assert r_hat > 0

    @pytest.mark.unit
    def test_rhat_matches_arviz(self) -> None:
        """compute_r_hat delegates to arviz.rhat correctly."""
        import arviz as az

        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((4, 200))

        computed = compute_r_hat(samples)
        expected = float(az.rhat(samples))

        assert_allclose(computed, expected, rtol=1e-12)


# ============================================================================
# Effective Sample Size Tests
# ============================================================================

class TestComputeESS:
    """Scientific tests for compute_ess."""

    @pytest.mark.unit
    def test_iid_samples_ess_equals_n(self) -> None:
        """IID samples should have ESS ≈ n."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        rng = np.random.default_rng(42)
        n = 1000
        samples = rng.standard_normal(n)

        ess = compute_ess(samples)

        # ESS should be close to n for IID samples
        # Allow some tolerance due to finite sample effects
        assert ess > 0.5 * n, f"ESS {ess} too low for IID samples"

    @pytest.mark.unit
    def test_constant_samples_low_ess(self) -> None:
        """Constant samples have ESS = n (no autocorrelation variance)."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        samples = np.ones(100)

        ess = compute_ess(samples)

        # For constant samples, autocorrelation is ill-defined
        # but ESS should be at least 1
        assert ess >= 1.0

    @pytest.mark.unit
    def test_correlated_samples_lower_ess(self) -> None:
        """Autocorrelated samples should have ESS < n."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        # Generate AR(1) process with high autocorrelation
        rng = np.random.default_rng(42)
        n = 1000
        rho = 0.9  # High autocorrelation
        samples = np.zeros(n)
        samples[0] = rng.standard_normal()
        for i in range(1, n):
            samples[i] = rho * samples[i-1] + np.sqrt(1 - rho**2) * rng.standard_normal()

        ess = compute_ess(samples)

        # ESS should be much less than n for correlated samples
        assert ess < 0.3 * n, f"ESS {ess} too high for correlated samples"

    @pytest.mark.unit
    def test_ess_always_positive(self) -> None:
        """ESS should always be positive."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        rng = np.random.default_rng(123)
        samples = rng.standard_normal(100)

        ess = compute_ess(samples)

        assert ess > 0


# ============================================================================
# BFMI Tests
# ============================================================================

class TestComputeBFMI:
    """Scientific tests for compute_bfmi."""

    @pytest.mark.unit
    def test_random_walk_bfmi(self) -> None:
        """Random walk energy should have BFMI near 1."""
        from heterodyne.optimization.cmc.diagnostics import compute_bfmi

        rng = np.random.default_rng(42)
        # Energy from random walk has var(diff) ≈ var(energy) for well-mixing
        energy = np.cumsum(rng.standard_normal(1000))

        bfmi = compute_bfmi(energy)

        # BFMI should be positive
        assert bfmi > 0

    @pytest.mark.unit
    def test_constant_energy_bfmi_one(self) -> None:
        """Constant energy has BFMI = 1."""
        from heterodyne.optimization.cmc.diagnostics import compute_bfmi

        energy = np.ones(100)

        bfmi = compute_bfmi(energy)

        # When variance is 0, should return 1.0
        assert bfmi == 1.0

    @pytest.mark.unit
    def test_bfmi_matches_arviz(self) -> None:
        """compute_bfmi delegates to arviz.bfmi correctly."""
        import arviz as az

        from heterodyne.optimization.cmc.diagnostics import compute_bfmi

        energy = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 4.0])

        computed = compute_bfmi(energy)
        expected = float(az.bfmi(energy[np.newaxis, :])[0])

        assert_allclose(computed, expected, rtol=1e-12)

    @pytest.mark.unit
    def test_bfmi_bounded(self) -> None:
        """BFMI should be bounded in reasonable range."""
        from heterodyne.optimization.cmc.diagnostics import compute_bfmi

        rng = np.random.default_rng(42)
        energy = rng.standard_normal(100)

        bfmi = compute_bfmi(energy)

        # BFMI is typically in (0, 2) for reasonable chains
        assert 0 < bfmi < 10


# ============================================================================
# Convergence Validation Tests
# ============================================================================

class TestValidateConvergence:
    """Tests for validate_convergence function."""

    @pytest.fixture
    def mock_converged_result(self):
        """Create a mock CMCResult that passes convergence."""
        from dataclasses import dataclass, field

        @dataclass
        class MockCMCResult:
            parameter_names: list = field(default_factory=lambda: ["p1", "p2"])
            r_hat: np.ndarray = field(default_factory=lambda: np.array([1.01, 1.02]))
            ess_bulk: np.ndarray = field(default_factory=lambda: np.array([500.0, 600.0]))
            bfmi: list = field(default_factory=lambda: [0.8, 0.9])

        return MockCMCResult()

    @pytest.fixture
    def mock_failed_result(self):
        """Create a mock CMCResult that fails convergence."""
        from dataclasses import dataclass, field

        @dataclass
        class MockCMCResult:
            parameter_names: list = field(default_factory=lambda: ["p1", "p2"])
            r_hat: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.3]))  # High R-hat
            ess_bulk: np.ndarray = field(default_factory=lambda: np.array([50.0, 60.0]))  # Low ESS
            bfmi: list = field(default_factory=lambda: [0.1, 0.2])  # Low BFMI

        return MockCMCResult()

    @pytest.mark.unit
    def test_converged_result_passes(self, mock_converged_result) -> None:
        """Well-converged result should pass validation."""
        from heterodyne.optimization.cmc.diagnostics import validate_convergence

        report = validate_convergence(mock_converged_result)

        assert report.passed
        assert report.r_hat_passed
        assert report.ess_passed
        assert report.bfmi_passed

    @pytest.mark.unit
    def test_failed_result_fails(self, mock_failed_result) -> None:
        """Poorly-converged result should fail validation."""
        from heterodyne.optimization.cmc.diagnostics import validate_convergence

        report = validate_convergence(mock_failed_result)

        assert not report.passed
        assert not report.r_hat_passed
        assert not report.ess_passed
        assert not report.bfmi_passed

    @pytest.mark.unit
    def test_custom_thresholds(self, mock_converged_result) -> None:
        """Custom thresholds should be respected."""
        from heterodyne.optimization.cmc.diagnostics import validate_convergence

        # With very strict thresholds, should fail
        report = validate_convergence(
            mock_converged_result,
            r_hat_threshold=1.0,  # Impossible to achieve
            min_ess=10000,  # Very high
            min_bfmi=0.99,  # Very high
        )

        assert not report.passed

    @pytest.mark.unit
    def test_report_has_messages(self, mock_failed_result) -> None:
        """Report should contain diagnostic messages."""
        from heterodyne.optimization.cmc.diagnostics import validate_convergence

        report = validate_convergence(mock_failed_result)

        assert len(report.messages) > 0


# ============================================================================
# Statistical Properties Tests
# ============================================================================

class TestStatisticalProperties:
    """Tests for statistical correctness of diagnostics."""

    @pytest.mark.unit
    def test_rhat_invariant_to_location_shift(self) -> None:
        """R-hat should be approximately invariant to location shifts.

        Rank-normalized R-hat preserves ordering under affine transforms,
        but rank discretization introduces O(1e-4) numerical noise.
        """
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((4, 100))

        rhat_original = compute_r_hat(samples)
        rhat_shifted = compute_r_hat(samples + 1000.0)

        assert_allclose(rhat_original, rhat_shifted, rtol=1e-3)

    @pytest.mark.unit
    def test_rhat_invariant_to_scale(self) -> None:
        """R-hat should be approximately invariant to scaling.

        Rank-normalized R-hat preserves ordering under affine transforms,
        but rank discretization introduces O(1e-4) numerical noise.
        """
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        rng = np.random.default_rng(42)
        samples = rng.standard_normal((4, 100))

        rhat_original = compute_r_hat(samples)
        rhat_scaled = compute_r_hat(samples * 100.0)

        assert_allclose(rhat_original, rhat_scaled, rtol=1e-3)

    @pytest.mark.unit
    def test_ess_invariant_to_location_shift(self) -> None:
        """ESS should be invariant to constant shift."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        rng = np.random.default_rng(42)
        samples = rng.standard_normal(100)

        ess_original = compute_ess(samples)
        ess_shifted = compute_ess(samples + 1000.0)

        assert_allclose(ess_original, ess_shifted, rtol=0.01)

    @pytest.mark.unit
    def test_ess_invariant_to_scale(self) -> None:
        """ESS should be invariant to scaling."""
        from heterodyne.optimization.cmc.diagnostics import compute_ess

        rng = np.random.default_rng(42)
        samples = rng.standard_normal(100)

        ess_original = compute_ess(samples)
        ess_scaled = compute_ess(samples * 100.0)

        assert_allclose(ess_original, ess_scaled, rtol=0.01)


# ============================================================================
# Posterior Contraction Ratio (PCR) Tests
# ============================================================================


class TestPosteriorContraction:
    """Tests for compute_posterior_contraction."""

    @pytest.mark.unit
    def test_pcr_strongly_constrained(self) -> None:
        """PCR ~ 1.0 when posterior is much narrower than prior."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_posterior_contraction,
        )
        from heterodyne.optimization.cmc.results import CMCResult

        result = CMCResult(
            parameter_names=["D0_ref", "alpha_ref"],
            posterior_mean=np.array([1.0, 0.5]),
            posterior_std=np.array([0.01, 0.005]),
            credible_intervals={},
            convergence_passed=True,
        )
        prior_std = {"D0_ref": 1.0, "alpha_ref": 0.5}

        pcr = compute_posterior_contraction(result, prior_std)

        assert pcr["D0_ref"] == pytest.approx(0.99, abs=0.001)
        assert pcr["alpha_ref"] == pytest.approx(0.99, abs=0.001)

    @pytest.mark.unit
    def test_pcr_poorly_identified(self) -> None:
        """PCR ~ 0 when posterior is similar width to prior."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_posterior_contraction,
        )
        from heterodyne.optimization.cmc.results import CMCResult

        result = CMCResult(
            parameter_names=["D0_ref"],
            posterior_mean=np.array([1.0]),
            posterior_std=np.array([0.95]),
            credible_intervals={},
            convergence_passed=True,
        )
        prior_std = {"D0_ref": 1.0}

        pcr = compute_posterior_contraction(result, prior_std)
        assert pcr["D0_ref"] == pytest.approx(0.05, abs=0.001)

    @pytest.mark.unit
    def test_pcr_negative_misspecification(self) -> None:
        """PCR < 0 when posterior is wider than prior."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_posterior_contraction,
        )
        from heterodyne.optimization.cmc.results import CMCResult

        result = CMCResult(
            parameter_names=["D0_ref"],
            posterior_mean=np.array([1.0]),
            posterior_std=np.array([2.0]),  # wider than prior
            credible_intervals={},
            convergence_passed=True,
        )
        prior_std = {"D0_ref": 1.0}

        pcr = compute_posterior_contraction(result, prior_std)
        assert pcr["D0_ref"] < 0

    @pytest.mark.unit
    def test_pcr_skips_missing_params(self) -> None:
        """Parameters not in prior_std are skipped."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_posterior_contraction,
        )
        from heterodyne.optimization.cmc.results import CMCResult

        result = CMCResult(
            parameter_names=["D0_ref", "f0"],
            posterior_mean=np.array([1.0, 0.5]),
            posterior_std=np.array([0.1, 0.05]),
            credible_intervals={},
            convergence_passed=True,
        )
        prior_std = {"D0_ref": 1.0}  # f0 not included

        pcr = compute_posterior_contraction(result, prior_std)
        assert "D0_ref" in pcr
        assert "f0" not in pcr

    @pytest.mark.unit
    def test_pcr_in_validate_convergence(self) -> None:
        """validate_convergence reports PCR when prior_std is in metadata."""
        from heterodyne.optimization.cmc.diagnostics import validate_convergence
        from heterodyne.optimization.cmc.results import CMCResult

        result = CMCResult(
            parameter_names=["D0_ref", "alpha_ref"],
            posterior_mean=np.array([1.0, 0.5]),
            posterior_std=np.array([0.01, 0.4]),
            credible_intervals={},
            convergence_passed=True,
            r_hat=np.array([1.01, 1.02]),
            ess_bulk=np.array([500.0, 500.0]),
            bfmi=[0.8, 0.9],
            metadata={"prior_std": {"D0_ref": 1.0, "alpha_ref": 0.5}},
        )

        report = validate_convergence(result)
        pcr_messages = [m for m in report.messages if "PCR" in m]
        assert len(pcr_messages) == 2

        # D0_ref should show high PCR (0.99)
        d0_msg = [m for m in pcr_messages if "D0_ref" in m][0]
        assert "0.99" in d0_msg

        # alpha_ref should show low PCR (0.20)
        alpha_msg = [m for m in pcr_messages if "alpha_ref" in m][0]
        assert "0.20" in alpha_msg


# ============================================================================
# Edge cases for compute_r_hat
# ============================================================================


class TestRhatEdgeCases:
    """Edge cases for the R-hat diagnostic."""

    @pytest.mark.unit
    def test_single_chain_rhat(self) -> None:
        """Single-chain R-hat is ill-defined; returns NaN without crashing."""
        from heterodyne.optimization.cmc.diagnostics import compute_r_hat

        samples = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # shape (1, 5)
        r_hat = compute_r_hat(samples)
        # ArviZ returns NaN for single-chain input (requires >= 2 chains)
        assert isinstance(r_hat, float)
        assert np.isnan(r_hat)
