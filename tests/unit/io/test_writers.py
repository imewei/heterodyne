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
        nlsq_result_full.metadata = {
            "optimizer": "joint_auto_averaged",
            "contrast": 0.31,
            "offset": 1.04,
        }
        paths = save_nlsq_json_files(nlsq_result_full, tmp_path)
        data = load_json(paths["metadata"])
        assert data["success"] is True
        assert data["n_iterations"] == 15
        assert data["n_function_evals"] == 42
        assert data["final_cost"] == pytest.approx(0.0025)
        assert data["reduced_chi_squared"] == pytest.approx(1.02)
        assert data["wall_time_seconds"] == pytest.approx(3.14)
        assert data["metadata"]["optimizer"] == "joint_auto_averaged"
        assert data["metadata"]["contrast"] == pytest.approx(0.31)
        assert data["metadata"]["offset"] == pytest.approx(1.04)

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


class TestNlsqNpzWithC2Exp:
    """Tests for save_nlsq_npz_file with c2_exp → residuals_normalized."""

    def _result(self, residuals: np.ndarray) -> NLSQResult:
        return NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["p"],
            success=True,
            message="ok",
            residuals=residuals,
        )

    def test_shape_match_computes_normalized(self, tmp_path: Path) -> None:
        """When residuals.shape == c2_exp.shape, normalized residuals are saved."""
        resid = np.ones(5)
        c2 = np.full(5, 2.0)
        path = tmp_path / "r.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2)
        data = np.load(path)
        assert "residuals_normalized" in data
        np.testing.assert_allclose(data["residuals_normalized"], resid / (0.05 * c2))

    def test_flat_matching_size_computes_normalized(self, tmp_path: Path) -> None:
        """residuals.ndim==1 and sizes match after ravel → normalized is saved."""
        N = 4
        resid = np.ones(N * N)
        c2 = np.full((N, N), 2.0)
        path = tmp_path / "flat.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2)
        data = np.load(path)
        assert "residuals_normalized" in data
        np.testing.assert_allclose(
            data["residuals_normalized"], resid / (0.05 * c2.ravel())
        )

    def test_offdiag_residuals_computes_normalized(self, tmp_path: Path) -> None:
        """Off-diagonal residuals (n_phi * N*(N-1),) normalized via off-diagonal mask."""
        n_phi, n_time = 3, 5
        mask = ~np.eye(n_time, dtype=bool)
        rng = np.random.default_rng(0)
        resid = rng.normal(size=n_phi * int(mask.sum()))
        c2_exp = np.abs(rng.normal(1.0, 0.1, size=(n_phi, n_time, n_time))) + 0.1
        path = tmp_path / "offdiag.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2_exp)
        data = np.load(path)
        assert "residuals_normalized" in data
        # Verify element ordering matches jax_backend convention
        denom = np.where(c2_exp != 0, c2_exp, 1.0)
        denom_flat = np.concatenate([denom[i][mask] for i in range(n_phi)])
        np.testing.assert_allclose(
            data["residuals_normalized"], resid / (0.05 * denom_flat)
        )

    def test_offdiag_zero_c2_replaced_by_one(self, tmp_path: Path) -> None:
        """Zero c2_exp off-diagonal entries are replaced by 1.0, no division by zero."""
        n_phi, n_time = 1, 4
        mask = ~np.eye(n_time, dtype=bool)
        resid = np.ones(n_phi * int(mask.sum()))
        c2_exp = np.ones((n_phi, n_time, n_time))
        c2_exp[0, 0, 1] = 0.0  # off-diagonal zero
        path = tmp_path / "zero.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2_exp)
        data = np.load(path)
        assert "residuals_normalized" in data
        assert np.isfinite(data["residuals_normalized"]).all()

    def test_incompatible_shapes_skips_normalized(self, tmp_path: Path) -> None:
        """Sizes that don't match any supported convention: key absent, no crash."""
        # 7 residuals vs (2,5,5): 7 ≠ 50 (full), 7 ≠ 40 (offdiag), 7 ≠ 50 (ravel)
        path = tmp_path / "incompat.npz"
        save_nlsq_npz_file(self._result(np.ones(7)), path, c2_exp=np.ones((2, 5, 5)))
        data = np.load(path)
        assert "residuals_normalized" not in data
        assert "residuals" in data

    def test_no_c2_exp_skips_normalized(self, tmp_path: Path) -> None:
        """Without c2_exp, residuals_normalized is never saved."""
        path = tmp_path / "noc2.npz"
        save_nlsq_npz_file(self._result(np.ones(10)), path)
        data = np.load(path)
        assert "residuals_normalized" not in data
        assert "residuals" in data

    def test_single_angle_offdiag(self, tmp_path: Path) -> None:
        """Single phi angle off-diagonal case: n_phi=1, N=6 → 30 residuals."""
        n_phi, n_time = 1, 6
        mask = ~np.eye(n_time, dtype=bool)
        resid = np.full(n_phi * int(mask.sum()), 0.5)
        c2_exp = np.full((n_phi, n_time, n_time), 2.0)
        path = tmp_path / "single.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2_exp)
        data = np.load(path)
        assert "residuals_normalized" in data
        np.testing.assert_allclose(data["residuals_normalized"], 0.5 / (0.05 * 2.0))

    def test_2d_per_angle_offdiag_computes_normalized(self, tmp_path: Path) -> None:
        """Per-angle save with 2-D c2_exp (N,N) and off-diagonal residuals (N*(N-1),)."""
        n_time = 5
        mask = ~np.eye(n_time, dtype=bool)
        rng = np.random.default_rng(7)
        resid = rng.normal(size=int(mask.sum()))  # N*(N-1) = 20
        c2_exp = np.abs(rng.normal(1.0, 0.1, size=(n_time, n_time))) + 0.1
        path = tmp_path / "per_angle.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2_exp)
        data = np.load(path)
        assert "residuals_normalized" in data
        denom = np.where(c2_exp != 0, c2_exp, 1.0)
        np.testing.assert_allclose(
            data["residuals_normalized"], resid / (0.05 * denom[mask])
        )

    def test_2d_per_angle_zero_c2_replaced_by_one(self, tmp_path: Path) -> None:
        """Zero 2-D c2_exp entries replaced by 1.0 before normalizing."""
        n_time = 4
        mask = ~np.eye(n_time, dtype=bool)
        resid = np.ones(int(mask.sum()))
        c2_exp = np.ones((n_time, n_time))
        c2_exp[0, 1] = 0.0  # off-diagonal zero
        path = tmp_path / "zero2d.npz"
        save_nlsq_npz_file(self._result(resid), path, c2_exp=c2_exp)
        data = np.load(path)
        assert "residuals_normalized" in data
        assert np.isfinite(data["residuals_normalized"]).all()


class TestNlsqNpzRoundTrip:
    """Tests for save_nlsq_npz_file and load_nlsq_npz_file."""

    def test_round_trip_full(
        self, tmp_path: Path, nlsq_result_full: NLSQResult
    ) -> None:
        nlsq_result_full.metadata = {
            "optimizer": "joint_cmaes_warmstart_auto_skip",
            "contrast": 0.31,
            "offset": 1.04,
            "per_angle": [{"phi_angle": 0.0}, {"phi_angle": 90.0}],
        }
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
        assert loaded.reduced_chi_squared == pytest.approx(1.02)
        assert loaded.n_iterations == 15
        assert loaded.n_function_evals == 42
        assert loaded.convergence_reason == "xtol"
        assert loaded.wall_time_seconds == pytest.approx(3.14)
        assert loaded.metadata["optimizer"] == "joint_cmaes_warmstart_auto_skip"
        assert loaded.metadata["contrast"] == pytest.approx(0.31)
        assert loaded.metadata["offset"] == pytest.approx(1.04)
        assert loaded.metadata["per_angle"][1]["phi_angle"] == pytest.approx(90.0)
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


# ===========================================================================
# Regression: degenerate all-shards-failed result must not crash save_mcmc_results
# ===========================================================================


class TestSaveMCMCResultsAllShardsFailed:
    """Regression for het_676ccc47: save_mcmc_results must write a tombstone
    JSON rather than crashing with ValueError when all shards failed and the
    result contains NaN-filled posterior_std/r_hat arrays.

    Root cause: ArviZ 1.1.0 API change made az.from_dict(**kwargs) raise
    TypeError for every shard, forcing convergence_passed=False for all,
    which caused _combine_shard_posteriors to return a degenerate CMCResult
    with posterior_std=NaN.  json_safe() then raised ValueError on those NaN
    values, crashing the entire run and writing no output files at all.
    """

    @staticmethod
    def _degenerate_result(n_params: int = 3) -> CMCResult:
        names = [f"p{i}" for i in range(n_params)]
        return CMCResult(
            parameter_names=names,
            posterior_mean=np.zeros(n_params),
            posterior_std=np.full(n_params, np.nan),
            credible_intervals={},
            convergence_passed=False,
            r_hat=np.full(n_params, np.nan),
            ess_bulk=np.full(n_params, np.nan),
            ess_tail=np.full(n_params, np.nan),
            bfmi=None,
            samples=None,
            num_warmup=10,
            num_samples=100,
            num_chains=4,
            metadata={"all_shards_failed": True, "n_total_shards": 47},
        )

    @pytest.mark.unit
    def test_does_not_crash(self, tmp_path: Path) -> None:
        """save_mcmc_results on a degenerate all-shards-failed result must not raise."""
        save_mcmc_results(self._degenerate_result(), tmp_path, prefix="mcmc")

    @pytest.mark.unit
    def test_writes_tombstone_summary_json(self, tmp_path: Path) -> None:
        """A tombstone summary.json is written so downstream tools can detect failure."""
        import json

        save_mcmc_results(self._degenerate_result(), tmp_path, prefix="mcmc")
        tombstone_path = tmp_path / "mcmc_summary.json"
        assert tombstone_path.exists(), "tombstone summary.json was not written"
        data = json.loads(tombstone_path.read_text())
        assert data["status"] == "failed"
        assert data["reason"] == "all_shards_failed"
        assert data["metadata"]["all_shards_failed"] is True

    @pytest.mark.unit
    def test_tombstone_contains_no_nan(self, tmp_path: Path) -> None:
        """The tombstone JSON must be parseable with allow_nan=False (no NaN literals)."""
        import json

        save_mcmc_results(self._degenerate_result(), tmp_path, prefix="mcmc")
        raw = (tmp_path / "mcmc_summary.json").read_text()
        # Standard json.loads rejects NaN; this will raise if any slipped through.
        json.loads(raw)

    @pytest.mark.unit
    def test_no_samples_npz_written(self, tmp_path: Path) -> None:
        """No samples.npz should be created for a degenerate result (nothing to save)."""
        save_mcmc_results(self._degenerate_result(), tmp_path, prefix="mcmc")
        assert not (tmp_path / "mcmc_samples.npz").exists()

    @pytest.mark.unit
    def test_tombstone_with_nan_shard_diagnostics_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """Regression for het_2e41117d: fit_cmc_sharded augments the degenerate
        CMCResult metadata with shard_diagnostics that contain NaN r_hat/ESS
        arrays (from shards that completed NUTS but failed convergence checks).
        The tombstone path must not crash when serializing that metadata.
        """
        import json

        n_params = 3
        # Simulate the shard_diagnostics structure added by fit_cmc_sharded
        shard_diagnostics = [
            {
                "convergence_passed": False,
                "r_hat": [float("nan")] * n_params,
                "ess_bulk": [float("nan")] * n_params,
                "bfmi": None,
                "wall_time_seconds": 3541.2,
            }
            for _ in range(44)
        ] + [
            # 3 timed-out shards: no r_hat/ESS
            {
                "convergence_passed": False,
                "r_hat": None,
                "ess_bulk": None,
                "bfmi": None,
                "wall_time_seconds": 3604.0,
            }
            for _ in range(3)
        ]
        names = [f"p{i}" for i in range(n_params)]
        result = CMCResult(
            parameter_names=names,
            posterior_mean=np.zeros(n_params),
            posterior_std=np.full(n_params, np.nan),
            credible_intervals={},
            convergence_passed=False,
            r_hat=np.full(n_params, np.nan),
            ess_bulk=np.full(n_params, np.nan),
            ess_tail=np.full(n_params, np.nan),
            bfmi=None,
            samples=None,
            num_warmup=500,
            num_samples=1000,
            num_chains=4,
            metadata={
                "all_shards_failed": True,
                "n_total_shards": 47,
                "n_failed_shards": 47,
                "num_shards": 47,
                "shard_diagnostics": shard_diagnostics,
            },
        )
        # Must not raise ValueError: Cannot serialize non-finite float to JSON
        save_mcmc_results(result, tmp_path, prefix="cmc_phi0")
        raw = (tmp_path / "cmc_phi0_summary.json").read_text()
        data = json.loads(raw)  # also verifies strict allow_nan=False
        assert data["status"] == "failed"
        # shard_diagnostics should be summarised, not raw NaN arrays
        assert isinstance(data["metadata"]["shard_diagnostics"], str)
        assert "47 shards" in data["metadata"]["shard_diagnostics"]
