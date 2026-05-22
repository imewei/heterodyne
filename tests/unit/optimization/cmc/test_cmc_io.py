"""Tests for CMC io.py save/load pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from heterodyne.optimization.cmc.io import (
    load_samples_npz,
    samples_to_arviz,
    save_all_results,
    save_diagnostics_json,
    save_fitted_data_npz,
    save_parameters_json,
    save_samples_npz,
)
from heterodyne.optimization.cmc.results import CMCResult


def _make_cmc_result(n_chains=2, n_samples=50):
    names = ["D0_ref", "alpha_ref", "v0"]
    n_params = len(names)
    rng = np.random.default_rng(7)
    samples = {
        n: rng.normal(size=(n_chains, n_samples)).astype(np.float64) for n in names
    }
    pm = np.array([float(np.mean(samples[n])) for n in names])
    ps = np.array([float(np.std(samples[n])) for n in names])
    return CMCResult(
        parameter_names=names,
        posterior_mean=pm,
        posterior_std=ps,
        credible_intervals={
            n: {"2.5%": pm[i] - 2 * ps[i], "97.5%": pm[i] + 2 * ps[i]}
            for i, n in enumerate(names)
        },
        convergence_passed=True,
        r_hat=np.ones(n_params) * 1.01,
        ess_bulk=np.ones(n_params) * 500.0,
        ess_tail=np.ones(n_params) * 450.0,
        bfmi=[0.85, 0.90],
        samples=samples,
        num_warmup=200,
        num_samples=n_samples,
        num_chains=n_chains,
        wall_time_seconds=12.3,
        metadata={"n_shards": 2, "analysis_mode": "static", "num_divergences": 3},
    )


class TestSaveSamplesNpz:
    def test_creates_file(self, tmp_path):
        save_samples_npz(_make_cmc_result(), tmp_path / "s.npz")
        assert (tmp_path / "s.npz").exists()

    def test_roundtrip_shape(self, tmp_path):
        result = _make_cmc_result(n_chains=2, n_samples=50)
        save_samples_npz(result, tmp_path / "s.npz")
        loaded = load_samples_npz(tmp_path / "s.npz")
        assert loaded["posterior_samples"].shape == (2, 50, 3)

    def test_roundtrip_param_names(self, tmp_path):
        result = _make_cmc_result()
        save_samples_npz(result, tmp_path / "s.npz")
        loaded = load_samples_npz(tmp_path / "s.npz")
        assert loaded["param_names"] == result.parameter_names

    def test_roundtrip_n_chains_n_samples(self, tmp_path):
        result = _make_cmc_result(n_chains=3, n_samples=40)
        save_samples_npz(result, tmp_path / "s.npz")
        loaded = load_samples_npz(tmp_path / "s.npz")
        assert loaded["n_chains"] == 3
        assert loaded["n_samples"] == 40

    def test_schema_version_present(self, tmp_path):
        save_samples_npz(_make_cmc_result(), tmp_path / "s.npz")
        loaded = load_samples_npz(tmp_path / "s.npz")
        assert loaded["schema_version"] == (1, 0)


class TestLoadSamplesNpz:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_samples_npz(tmp_path / "missing.npz")

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="Expected .npz"):
            load_samples_npz(f)


class TestSamplesToArviz:
    def test_returns_inference_data(self, tmp_path):
        result = _make_cmc_result()
        save_samples_npz(result, tmp_path / "s.npz")
        idata = samples_to_arviz(load_samples_npz(tmp_path / "s.npz"))
        assert hasattr(idata, "posterior")

    def test_param_names_in_posterior(self, tmp_path):
        result = _make_cmc_result()
        save_samples_npz(result, tmp_path / "s.npz")
        idata = samples_to_arviz(load_samples_npz(tmp_path / "s.npz"))
        for name in result.parameter_names:
            assert name in idata.posterior


class TestSaveFittedDataNpz:
    def _arrays(self, n=5):
        t = np.linspace(0, 1, n)
        t1, t2 = np.meshgrid(t, t)
        c2 = np.ones(n * n)
        return c2, t1.ravel(), t2.ravel()

    def test_creates_file(self, tmp_path):
        c2, t1, t2 = self._arrays()
        save_fitted_data_npz(
            result=_make_cmc_result(),
            c2_exp=c2,
            c2_fitted=c2 * 0.99,
            c2_fitted_std=np.ones_like(c2) * 0.01,
            t1=t1,
            t2=t2,
            phi_angles=np.array([0.0, 30.0]),
            q=0.005,
            output_path=tmp_path / "fd.npz",
        )
        assert (tmp_path / "fd.npz").exists()

    def test_residuals_stored(self, tmp_path):
        c2, t1, t2 = self._arrays()
        save_fitted_data_npz(
            result=_make_cmc_result(),
            c2_exp=c2,
            c2_fitted=c2,
            c2_fitted_std=np.zeros_like(c2),
            t1=t1,
            t2=t2,
            phi_angles=np.array([0.0]),
            q=0.005,
            output_path=tmp_path / "fd.npz",
        )
        data = np.load(tmp_path / "fd.npz")
        assert "residuals" in data
        np.testing.assert_allclose(data["residuals"], 0.0, atol=1e-10)


class TestSaveParametersJson:
    def test_creates_file(self, tmp_path):
        save_parameters_json(_make_cmc_result(), tmp_path / "p.json")
        assert (tmp_path / "p.json").exists()

    def test_json_has_param_entries(self, tmp_path):
        result = _make_cmc_result()
        save_parameters_json(result, tmp_path / "p.json")
        data = json.loads((tmp_path / "p.json").read_text())
        for name in result.parameter_names:
            assert name in data
            assert "mean" in data[name]
            assert "std" in data[name]

    def test_nan_serialized_as_null(self, tmp_path):
        result = _make_cmc_result()
        assert result.r_hat is not None
        result.r_hat[0] = float("nan")
        save_parameters_json(result, tmp_path / "p.json")
        data = json.loads((tmp_path / "p.json").read_text())
        first = result.parameter_names[0]
        assert data[first]["r_hat"] is None


class TestSaveDiagnosticsJson:
    def test_creates_file(self, tmp_path):
        save_diagnostics_json(_make_cmc_result(), tmp_path / "d.json")
        assert (tmp_path / "d.json").exists()

    def test_required_keys(self, tmp_path):
        save_diagnostics_json(_make_cmc_result(), tmp_path / "d.json")
        data = json.loads((tmp_path / "d.json").read_text())
        for k in ("convergence_status", "total_divergences", "sampling_config"):
            assert k in data

    def test_warnings_stored(self, tmp_path):
        save_diagnostics_json(
            _make_cmc_result(), tmp_path / "d.json", warnings=["test warning"]
        )
        data = json.loads((tmp_path / "d.json").read_text())
        assert "test warning" in data["warnings"]


class TestSaveAllResults:
    def test_core_files_saved(self, tmp_path):
        saved = save_all_results(_make_cmc_result(), tmp_path)
        assert (tmp_path / "samples.npz").exists()
        assert (tmp_path / "parameters.json").exists()
        assert (tmp_path / "diagnostics.json").exists()
        assert "samples" in saved and "parameters" in saved and "diagnostics" in saved

    def test_fitted_data_saved_when_provided(self, tmp_path):
        n = 5
        t = np.linspace(0, 1, n)
        t1, t2 = np.meshgrid(t, t)
        c2 = np.ones(n * n)
        saved = save_all_results(
            _make_cmc_result(),
            tmp_path,
            c2_exp=c2,
            c2_fitted=c2,
            c2_fitted_std=np.zeros(n * n),
            t1=t1.ravel(),
            t2=t2.ravel(),
            phi_angles=np.array([0.0]),
            q=0.005,
        )
        assert (tmp_path / "fitted_data.npz").exists()
        assert "fitted_data" in saved

    def test_skips_fitted_data_when_absent(self, tmp_path):
        saved = save_all_results(_make_cmc_result(), tmp_path)
        assert "fitted_data" not in saved
