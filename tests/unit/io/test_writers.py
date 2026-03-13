"""Unit tests for heterodyne.io.nlsq_writers and heterodyne.io.mcmc_writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heterodyne.io.json_utils import load_json
from heterodyne.io.mcmc_writers import (
    _save_posterior_samples,
    format_mcmc_summary,
    save_mcmc_diagnostics,
    save_mcmc_results,
)
from heterodyne.io.nlsq_writers import (
    format_nlsq_summary,
    load_nlsq_npz_file,
    save_nlsq_json_files,
    save_nlsq_npz_file,
)
from heterodyne.optimization.cmc.results import CMCResult
from heterodyne.optimization.nlsq.results import NLSQResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nlsq_result_minimal() -> NLSQResult:
    """Minimal successful NLSQResult with no optional fields."""
    return NLSQResult(
        parameters=np.array([1.0, 2.0]),
        parameter_names=["alpha", "beta"],
        success=True,
        message="Converged",
    )


@pytest.fixture()
def nlsq_result_full() -> NLSQResult:
    """Fully populated NLSQResult."""
    params = np.array([1.0, 2.0, 3.0])
    return NLSQResult(
        parameters=params,
        parameter_names=["D0", "alpha", "v0"],
        success=True,
        message="Converged (xtol)",
        uncertainties=np.array([0.1, 0.05, 0.5]),
        covariance=np.eye(3) * 0.01,
        final_cost=0.0025,
        reduced_chi_squared=1.02,
        n_iterations=15,
        n_function_evals=42,
        convergence_reason="xtol",
        residuals=np.array([0.01, -0.02, 0.005]),
        jacobian=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        fitted_correlation=np.array([0.9, 0.8, 0.7]),
        wall_time_seconds=3.14,
    )


@pytest.fixture()
def nlsq_result_failed() -> NLSQResult:
    """Failed NLSQResult."""
    return NLSQResult(
        parameters=np.array([0.0, 0.0]),
        parameter_names=["a", "b"],
        success=False,
        message="Max iterations exceeded",
        n_iterations=100,
        n_function_evals=500,
    )


@pytest.fixture()
def cmc_result_minimal() -> CMCResult:
    """Minimal CMCResult."""
    return CMCResult(
        parameter_names=["D0", "alpha"],
        posterior_mean=np.array([100.0, 0.5]),
        posterior_std=np.array([10.0, 0.05]),
        credible_intervals={
            "D0": {"2.5%": 80.0, "97.5%": 120.0},
            "alpha": {"2.5%": 0.4, "97.5%": 0.6},
        },
        convergence_passed=True,
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
    )


@pytest.fixture()
def cmc_result_full() -> CMCResult:
    """Fully populated CMCResult with diagnostics and samples."""
    rng = np.random.default_rng(42)
    return CMCResult(
        parameter_names=["D0", "alpha", "v0"],
        posterior_mean=np.array([100.0, 0.5, 1000.0]),
        posterior_std=np.array([10.0, 0.05, 50.0]),
        credible_intervals={
            "D0": {"2.5%": 80.0, "97.5%": 120.0},
            "alpha": {"2.5%": 0.4, "97.5%": 0.6},
            "v0": {"2.5%": 900.0, "97.5%": 1100.0},
        },
        convergence_passed=True,
        r_hat=np.array([1.01, 1.005, 1.02]),
        ess_bulk=np.array([800.0, 900.0, 750.0]),
        ess_tail=np.array([600.0, 700.0, 550.0]),
        bfmi=[0.9, 0.85, 0.88, 0.92],
        samples={
            "D0": rng.normal(100, 10, 4000),
            "alpha": rng.normal(0.5, 0.05, 4000),
            "v0": rng.normal(1000, 50, 4000),
        },
        map_estimate=np.array([101.0, 0.51, 1010.0]),
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        wall_time_seconds=120.5,
    )


@pytest.fixture()
def cmc_result_failed() -> CMCResult:
    """Failed CMCResult with poor diagnostics."""
    return CMCResult(
        parameter_names=["D0"],
        posterior_mean=np.array([100.0]),
        posterior_std=np.array([50.0]),
        credible_intervals={"D0": {"2.5%": 0.0, "97.5%": 200.0}},
        convergence_passed=False,
        r_hat=np.array([1.5]),
        ess_bulk=np.array([50.0]),
        bfmi=[0.1],
        num_warmup=100,
        num_samples=200,
        num_chains=2,
    )


# ===========================================================================
# NLSQ Writers
# ===========================================================================


class TestSaveNlsqJsonFiles:
    """Tests for save_nlsq_json_files."""

    def test_creates_parameter_and_metadata_files(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_full, tmp_path)
        assert "parameters" in paths
        assert "metadata" in paths
        assert paths["parameters"].exists()
        assert paths["metadata"].exists()

    def test_parameter_file_contents(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_full, tmp_path)
        data = load_json(paths["parameters"])
        assert data["parameter_names"] == ["D0", "alpha", "v0"]
        assert data["parameters"] == [1.0, 2.0, 3.0]
        assert data["uncertainties"] == [0.1, 0.05, 0.5]
        assert "timestamp" in data

    def test_metadata_file_contents(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_full, tmp_path)
        data = load_json(paths["metadata"])
        assert data["success"] is True
        assert data["n_iterations"] == 15
        assert data["n_function_evals"] == 42
        assert data["final_cost"] == pytest.approx(0.0025)
        assert data["reduced_chi_squared"] == pytest.approx(1.02)
        assert data["wall_time_seconds"] == pytest.approx(3.14)

    def test_custom_prefix(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_minimal, tmp_path, prefix="run01")
        assert paths["parameters"].name == "run01_parameters.json"
        assert paths["metadata"].name == "run01_metadata.json"

    def test_creates_output_dir(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        out = tmp_path / "sub" / "dir"
        paths = save_nlsq_json_files(nlsq_result_minimal, out)
        assert out.is_dir()
        assert paths["parameters"].exists()

    def test_no_uncertainties(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_minimal, tmp_path)
        data = load_json(paths["parameters"])
        assert data["uncertainties"] is None

    def test_none_final_cost(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        paths = save_nlsq_json_files(nlsq_result_minimal, tmp_path)
        data = load_json(paths["metadata"])
        assert data["final_cost"] is None
        assert data["reduced_chi_squared"] is None


# ---------------------------------------------------------------------------
# NLSQ NPZ round-trip
# ---------------------------------------------------------------------------


class TestNlsqNpzRoundTrip:
    """Tests for save_nlsq_npz_file and load_nlsq_npz_file."""

    def test_round_trip_full(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        path = tmp_path / "result.npz"
        saved = save_nlsq_npz_file(nlsq_result_full, path, include_jacobian=True)
        assert saved.suffix == ".npz"

        loaded = load_nlsq_npz_file(saved)
        np.testing.assert_array_almost_equal(
            loaded.parameters, nlsq_result_full.parameters
        )
        assert loaded.parameter_names == nlsq_result_full.parameter_names
        assert loaded.success is True
        assert loaded.final_cost == pytest.approx(0.0025)
        np.testing.assert_array_almost_equal(
            loaded.uncertainties, nlsq_result_full.uncertainties
        )
        np.testing.assert_array_almost_equal(
            loaded.covariance, nlsq_result_full.covariance
        )
        np.testing.assert_array_almost_equal(
            loaded.residuals, nlsq_result_full.residuals
        )
        np.testing.assert_array_almost_equal(loaded.jacobian, nlsq_result_full.jacobian)
        np.testing.assert_array_almost_equal(
            loaded.fitted_correlation, nlsq_result_full.fitted_correlation
        )

    def test_round_trip_minimal(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        path = tmp_path / "min.npz"
        save_nlsq_npz_file(nlsq_result_minimal, path)
        loaded = load_nlsq_npz_file(path)
        np.testing.assert_array_almost_equal(
            loaded.parameters, nlsq_result_minimal.parameters
        )
        assert loaded.parameter_names == ["alpha", "beta"]
        assert loaded.success is True
        assert loaded.final_cost is None  # was None -> stored as nan -> loaded as None
        assert loaded.uncertainties is None
        assert loaded.covariance is None

    def test_adds_npz_suffix(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        path = tmp_path / "no_ext"
        saved = save_nlsq_npz_file(nlsq_result_minimal, path)
        assert saved.suffix == ".npz"
        assert saved.exists()

    def test_exclude_residuals(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        path = tmp_path / "no_resid.npz"
        save_nlsq_npz_file(nlsq_result_full, path, include_residuals=False)
        loaded = load_nlsq_npz_file(path)
        assert loaded.residuals is None

    def test_exclude_jacobian_by_default(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        path = tmp_path / "default.npz"
        save_nlsq_npz_file(nlsq_result_full, path)
        loaded = load_nlsq_npz_file(path)
        assert loaded.jacobian is None

    def test_failed_result(
        self, tmp_path: Path, nlsq_result_failed: NLSQResult
    ) -> None:
        path = tmp_path / "failed.npz"
        save_nlsq_npz_file(nlsq_result_failed, path)
        loaded = load_nlsq_npz_file(path)
        assert loaded.success is False
        assert loaded.final_cost is None  # was None

    def test_creates_parent_dir(
        self, tmp_path: Path, nlsq_result_minimal: NLSQResult
    ) -> None:
        path = tmp_path / "deep" / "dir" / "result.npz"
        save_nlsq_npz_file(nlsq_result_minimal, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_nlsq_npz_file(tmp_path / "missing.npz")


# ---------------------------------------------------------------------------
# format_nlsq_summary
# ---------------------------------------------------------------------------


class TestFormatNlsqSummary:
    """Tests for format_nlsq_summary."""

    def test_success_summary(self, nlsq_result_full: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_full)
        assert "SUCCESS" in text
        assert "D0" in text
        assert "alpha" in text
        assert "v0" in text
        assert "Converged" in text

    def test_failed_summary(self, nlsq_result_failed: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_failed)
        assert "FAILED" in text
        assert "Max iterations" in text

    def test_with_uncertainties(self, nlsq_result_full: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_full)
        # The +/- symbol should appear for parameters with uncertainties
        assert "\u00b1" in text or "±" in text

    def test_without_uncertainties(self, nlsq_result_minimal: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_minimal)
        # No +/- symbol when uncertainties are None
        assert "±" not in text

    def test_statistics_section(self, nlsq_result_full: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_full)
        assert "Iterations" in text
        assert "15" in text
        assert "42" in text
        assert "Wall time" in text

    def test_no_cost_no_chi2(self, nlsq_result_minimal: NLSQResult) -> None:
        text = format_nlsq_summary(nlsq_result_minimal)
        assert "Final cost" not in text
        assert "Reduced" not in text


# ===========================================================================
# MCMC Writers
# ===========================================================================


class TestSaveMcmcResults:
    """Tests for save_mcmc_results."""

    def test_creates_all_files(
        self, tmp_path: Path, cmc_result_full: CMCResult
    ) -> None:
        paths = save_mcmc_results(cmc_result_full, tmp_path)
        assert "summary" in paths
        assert "diagnostics" in paths
        assert "samples" in paths
        assert paths["summary"].exists()
        assert paths["diagnostics"].exists()
        assert paths["samples"].exists()

    def test_summary_contents(self, tmp_path: Path, cmc_result_full: CMCResult) -> None:
        paths = save_mcmc_results(cmc_result_full, tmp_path)
        data = load_json(paths["summary"])
        assert data["parameter_names"] == ["D0", "alpha", "v0"]
        assert data["num_samples"] == 1000
        assert data["num_chains"] == 4
        assert data["map_estimate"] is not None
        assert "timestamp" in data

    def test_custom_prefix(self, tmp_path: Path, cmc_result_minimal: CMCResult) -> None:
        paths = save_mcmc_results(cmc_result_minimal, tmp_path, prefix="cmc_run")
        assert paths["summary"].name == "cmc_run_summary.json"
        assert paths["diagnostics"].name == "cmc_run_diagnostics.json"
        assert paths["samples"].name == "cmc_run_samples.npz"

    def test_creates_output_dir(
        self, tmp_path: Path, cmc_result_minimal: CMCResult
    ) -> None:
        out = tmp_path / "nested" / "output"
        paths = save_mcmc_results(cmc_result_minimal, out)
        assert out.is_dir()
        assert paths["summary"].exists()

    def test_no_map_estimate(
        self, tmp_path: Path, cmc_result_minimal: CMCResult
    ) -> None:
        paths = save_mcmc_results(cmc_result_minimal, tmp_path)
        data = load_json(paths["summary"])
        assert data["map_estimate"] is None


# ---------------------------------------------------------------------------
# save_mcmc_diagnostics
# ---------------------------------------------------------------------------


class TestSaveMcmcDiagnostics:
    """Tests for save_mcmc_diagnostics."""

    def test_full_diagnostics(self, tmp_path: Path, cmc_result_full: CMCResult) -> None:
        path = tmp_path / "diag.json"
        save_mcmc_diagnostics(cmc_result_full, path)
        data = load_json(path)

        assert data["convergence_passed"] is True
        assert data["all_r_hat_passed"] is True
        assert data["r_hat_threshold"] == 1.1
        assert "max_r_hat" in data
        assert "min_ess_bulk" in data
        assert "bfmi_passed" in data
        assert data["bfmi_passed"] is True

    def test_per_parameter_diagnostics(
        self, tmp_path: Path, cmc_result_full: CMCResult
    ) -> None:
        path = tmp_path / "diag.json"
        save_mcmc_diagnostics(cmc_result_full, path)
        data = load_json(path)
        pd = data["parameter_diagnostics"]
        assert "D0" in pd
        assert "alpha" in pd
        assert "v0" in pd
        assert pd["D0"]["r_hat_passed"] is True
        assert "ess_bulk" in pd["D0"]
        assert "ess_tail" in pd["D0"]

    def test_failed_diagnostics(
        self, tmp_path: Path, cmc_result_failed: CMCResult
    ) -> None:
        path = tmp_path / "diag_fail.json"
        save_mcmc_diagnostics(cmc_result_failed, path)
        data = load_json(path)

        assert data["convergence_passed"] is False
        assert data["all_r_hat_passed"] is False
        pd_d0 = data["parameter_diagnostics"]["D0"]
        assert pd_d0["r_hat"] == pytest.approx(1.5)
        assert pd_d0["r_hat_passed"] is False

    def test_custom_thresholds(
        self, tmp_path: Path, cmc_result_full: CMCResult
    ) -> None:
        path = tmp_path / "diag_custom.json"
        save_mcmc_diagnostics(
            cmc_result_full, path, r_hat_threshold=1.005, min_bfmi=0.95
        )
        data = load_json(path)
        # With strict threshold of 1.005, some params should fail
        assert data["r_hat_threshold"] == 1.005
        # bfmi of [0.9, 0.85, 0.88, 0.92] are all < 0.95
        assert data["bfmi_passed"] is False

    def test_no_diagnostics(
        self, tmp_path: Path, cmc_result_minimal: CMCResult
    ) -> None:
        path = tmp_path / "diag_none.json"
        save_mcmc_diagnostics(cmc_result_minimal, path)
        data = load_json(path)
        assert "max_r_hat" not in data
        assert "min_ess_bulk" not in data
        assert "bfmi" not in data

    def test_sampling_info(self, tmp_path: Path, cmc_result_full: CMCResult) -> None:
        path = tmp_path / "diag.json"
        save_mcmc_diagnostics(cmc_result_full, path)
        data = load_json(path)
        info = data["sampling_info"]
        assert info["num_warmup"] == 500
        assert info["num_samples"] == 1000
        assert info["num_chains"] == 4
        assert info["wall_time_seconds"] == pytest.approx(120.5)

    def test_creates_parent_dir(
        self, tmp_path: Path, cmc_result_minimal: CMCResult
    ) -> None:
        path = tmp_path / "sub" / "diag.json"
        save_mcmc_diagnostics(cmc_result_minimal, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# _save_posterior_samples
# ---------------------------------------------------------------------------


class TestSavePosteriorSamples:
    """Tests for _save_posterior_samples."""

    def test_saves_samples(self, tmp_path: Path, cmc_result_full: CMCResult) -> None:
        path = tmp_path / "samples.npz"
        _save_posterior_samples(cmc_result_full, path)
        assert path.exists()

        data = np.load(path)
        names = list(data["parameter_names"])
        assert names == ["D0", "alpha", "v0"]
        assert "samples_D0" in data
        assert "samples_alpha" in data
        assert "samples_v0" in data
        assert "r_hat" in data
        assert "ess_bulk" in data
        assert "ess_tail" in data

    def test_no_samples(self, tmp_path: Path, cmc_result_minimal: CMCResult) -> None:
        path = tmp_path / "no_samples.npz"
        _save_posterior_samples(cmc_result_minimal, path)
        data = np.load(path)
        # Should still have parameter_names
        assert "parameter_names" in data
        # But no sample arrays
        assert "samples_D0" not in data

    def test_no_diagnostics_arrays(self, tmp_path: Path) -> None:
        result = CMCResult(
            parameter_names=["x"],
            posterior_mean=np.array([1.0]),
            posterior_std=np.array([0.1]),
            credible_intervals={"x": {"2.5%": 0.8, "97.5%": 1.2}},
            convergence_passed=True,
        )
        path = tmp_path / "bare.npz"
        _save_posterior_samples(result, path)
        data = np.load(path)
        assert "r_hat" not in data
        assert "ess_bulk" not in data
        assert "ess_tail" not in data


# ---------------------------------------------------------------------------
# format_mcmc_summary
# ---------------------------------------------------------------------------


class TestFormatMcmcSummary:
    """Tests for format_mcmc_summary."""

    def test_passed_summary(self, cmc_result_full: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_full)
        assert "PASSED" in text
        assert "D0" in text
        assert "alpha" in text
        assert "v0" in text
        assert "Chains: 4" in text
        assert "Samples: 1000" in text

    def test_failed_summary(self, cmc_result_failed: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_failed)
        assert "FAILED" in text

    def test_diagnostics_section(self, cmc_result_full: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_full)
        assert "Max R-hat" in text
        assert "Min ESS" in text
        assert "Min BFMI" in text
        assert "Wall time" in text

    def test_no_diagnostics(self, cmc_result_minimal: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_minimal)
        assert "Max R-hat" not in text
        assert "Min ESS" not in text

    def test_rhat_pass_flag(self, cmc_result_full: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_full)
        assert "(PASS)" in text

    def test_rhat_warn_flag(self, cmc_result_failed: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_failed)
        assert "(WARN)" in text

    def test_bfmi_warn_flag(self, cmc_result_failed: CMCResult) -> None:
        text = format_mcmc_summary(cmc_result_failed)
        # bfmi=0.1 < 0.3 should show WARN
        assert "(WARN)" in text

    def test_missing_credible_interval_keys(self) -> None:
        """When credible_intervals dict lacks expected keys, NaN used."""
        result = CMCResult(
            parameter_names=["x"],
            posterior_mean=np.array([1.0]),
            posterior_std=np.array([0.1]),
            credible_intervals={"x": {}},  # empty CI dict
            convergence_passed=True,
            num_chains=1,
            num_samples=100,
            num_warmup=50,
        )
        text = format_mcmc_summary(result)
        assert "x" in text  # Should not crash

    def test_missing_parameter_in_ci(self) -> None:
        """When parameter not in credible_intervals dict at all."""
        result = CMCResult(
            parameter_names=["x"],
            posterior_mean=np.array([1.0]),
            posterior_std=np.array([0.1]),
            credible_intervals={},  # missing entirely
            convergence_passed=True,
            num_chains=1,
            num_samples=100,
            num_warmup=50,
        )
        text = format_mcmc_summary(result)
        assert "x" in text
