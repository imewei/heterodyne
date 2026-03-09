"""Unit tests for heterodyne.cli.optimization_runner module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.optimization.nlsq.results import NLSQResult


def _make_nlsq_result(
    success: bool = True,
    reduced_chi2: float = 1.5,
    params: dict[str, float] | None = None,
) -> NLSQResult:
    """Build a minimal NLSQResult for testing."""
    if params is None:
        params = {"D0_ref": 1e4, "D0_sample": 1e4}
    names = list(params.keys())
    values = np.array(list(params.values()))
    return NLSQResult(
        parameters=values,
        parameter_names=names,
        success=success,
        message="converged" if success else "failed",
        reduced_chi_squared=reduced_chi2,
        metadata={},
    )


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for runner functions."""
    defaults = {
        "verbose": 0,
        "multistart": False,
        "multistart_n": 10,
        "num_samples": None,
        "num_chains": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
class TestRunNLSQ:
    """Tests for run_nlsq function."""

    @patch("heterodyne.cli.optimization_runner.save_nlsq_npz_file")
    @patch("heterodyne.cli.optimization_runner.save_nlsq_json_files")
    @patch("heterodyne.cli.optimization_runner.format_nlsq_summary", return_value="summary")
    @patch("heterodyne.cli.optimization_runner.fit_nlsq_jax")
    def test_calls_fit_nlsq_for_each_angle(
        self,
        mock_fit: MagicMock,
        mock_fmt: MagicMock,
        mock_save_json: MagicMock,
        mock_save_npz: MagicMock,
    ) -> None:
        """run_nlsq calls fit_nlsq_jax once per phi angle."""
        from heterodyne.cli.optimization_runner import run_nlsq

        phi_angles = [0.0, 45.0, 90.0]
        mock_fit.return_value = _make_nlsq_result()
        mock_fit.return_value.metadata = {}

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.nlsq_config = {}

        c2_data = np.zeros((3, 10, 10))

        results = run_nlsq(
            model=mock_model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
        )

        assert mock_fit.call_count == len(phi_angles)
        assert len(results) == len(phi_angles)

    @patch("heterodyne.cli.optimization_runner.save_nlsq_npz_file")
    @patch("heterodyne.cli.optimization_runner.save_nlsq_json_files")
    @patch("heterodyne.cli.optimization_runner.format_nlsq_summary", return_value="summary")
    @patch("heterodyne.cli.optimization_runner.fit_nlsq_jax")
    def test_returns_list_of_nlsq_results(
        self,
        mock_fit: MagicMock,
        mock_fmt: MagicMock,
        mock_save_json: MagicMock,
        mock_save_npz: MagicMock,
    ) -> None:
        """run_nlsq returns a list of NLSQResult objects."""
        from heterodyne.cli.optimization_runner import run_nlsq

        result = _make_nlsq_result()
        result.metadata = {}
        mock_fit.return_value = result

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.nlsq_config = {}

        c2_data = np.zeros((1, 10, 10))
        results = run_nlsq(
            model=mock_model,
            c2_data=c2_data,
            phi_angles=[0.0],
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].success is True

    @patch("heterodyne.cli.optimization_runner.save_nlsq_npz_file")
    @patch("heterodyne.cli.optimization_runner.save_nlsq_json_files")
    @patch("heterodyne.cli.optimization_runner.format_nlsq_summary", return_value="summary")
    @patch("heterodyne.cli.optimization_runner.fit_nlsq_jax")
    def test_records_chi2_metric_in_summary(
        self,
        mock_fit: MagicMock,
        mock_fmt: MagicMock,
        mock_save_json: MagicMock,
        mock_save_npz: MagicMock,
    ) -> None:
        """run_nlsq records reduced_chi_squared in summary when provided."""
        from heterodyne.cli.optimization_runner import run_nlsq
        from heterodyne.utils.logging import AnalysisSummaryLogger

        result = _make_nlsq_result(reduced_chi2=2.5)
        result.metadata = {}
        mock_fit.return_value = result

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.nlsq_config = {}

        summary = AnalysisSummaryLogger(run_id="test", analysis_mode="two_component")

        run_nlsq(
            model=mock_model,
            c2_data=np.zeros((1, 10, 10)),
            phi_angles=[0.0],
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
            summary=summary,
        )

        assert "nlsq_chi2_phi0" in summary._metrics
        assert summary._metrics["nlsq_chi2_phi0"] == 2.5


@pytest.mark.unit
class TestRunCMC:
    """Tests for run_cmc function."""

    @patch("heterodyne.cli.optimization_runner.save_mcmc_results")
    @patch("heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="summary")
    @patch("heterodyne.cli.optimization_runner.fit_cmc_jax")
    def test_calls_fit_cmc_for_each_angle(
        self,
        mock_fit: MagicMock,
        mock_fmt: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """run_cmc calls fit_cmc_jax once per phi angle."""
        from heterodyne.optimization.cmc.results import CMCResult

        from heterodyne.cli.optimization_runner import run_cmc

        phi_angles = [0.0, 90.0]
        mock_result = CMCResult(
            parameter_names=["D0_ref"],
            posterior_mean=np.array([1e4]),
            posterior_std=np.array([100.0]),
            credible_intervals={},
            convergence_passed=True,
            metadata={},
        )
        mock_fit.return_value = mock_result

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.cmc_config = {}

        c2_data = np.zeros((2, 10, 10))

        results = run_cmc(
            model=mock_model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
        )

        assert mock_fit.call_count == len(phi_angles)
        assert len(results) == len(phi_angles)

    @patch("heterodyne.cli.optimization_runner.save_mcmc_results")
    @patch("heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="summary")
    @patch("heterodyne.cli.optimization_runner._validate_warmstart_quality", return_value=True)
    @patch("heterodyne.cli.optimization_runner._log_warmstart_physical_params")
    @patch("heterodyne.cli.optimization_runner.fit_cmc_jax")
    def test_validates_warmstart_quality(
        self,
        mock_fit: MagicMock,
        mock_log_params: MagicMock,
        mock_validate: MagicMock,
        mock_fmt: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """run_cmc validates warm-start quality when nlsq_results are provided."""
        from heterodyne.optimization.cmc.results import CMCResult

        from heterodyne.cli.optimization_runner import run_cmc

        mock_cmc_result = CMCResult(
            parameter_names=["D0_ref"],
            posterior_mean=np.array([1e4]),
            posterior_std=np.array([100.0]),
            credible_intervals={},
            convergence_passed=True,
            metadata={},
        )
        mock_fit.return_value = mock_cmc_result

        nlsq_result = _make_nlsq_result()

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.cmc_config = {}

        run_cmc(
            model=mock_model,
            c2_data=np.zeros((1, 10, 10)),
            phi_angles=[0.0],
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
            nlsq_results=[nlsq_result],
        )

        mock_validate.assert_called_once_with(nlsq_result)


@pytest.mark.unit
class TestValidateWarmstartQuality:
    """Tests for _validate_warmstart_quality."""

    def test_returns_true_for_good_result(self) -> None:
        """Good result (success=True, low chi2) passes validation."""
        from heterodyne.cli.optimization_runner import _validate_warmstart_quality

        result = _make_nlsq_result(success=True, reduced_chi2=1.5)
        assert _validate_warmstart_quality(result) is True

    def test_returns_false_for_failed_result(self) -> None:
        """Failed NLSQ (success=False) fails validation."""
        from heterodyne.cli.optimization_runner import _validate_warmstart_quality

        result = _make_nlsq_result(success=False, reduced_chi2=1.5)
        assert _validate_warmstart_quality(result) is False

    def test_returns_false_for_high_chi2(self) -> None:
        """High reduced chi-squared fails validation."""
        from heterodyne.cli.optimization_runner import _validate_warmstart_quality

        result = _make_nlsq_result(success=True, reduced_chi2=15.0)
        assert _validate_warmstart_quality(result) is False

    def test_respects_custom_threshold(self) -> None:
        """Custom chi2_threshold is honored."""
        from heterodyne.cli.optimization_runner import _validate_warmstart_quality

        result = _make_nlsq_result(success=True, reduced_chi2=8.0)
        # Default threshold 10.0 → passes
        assert _validate_warmstart_quality(result, chi2_threshold=10.0) is True
        # Stricter threshold → fails
        assert _validate_warmstart_quality(result, chi2_threshold=5.0) is False

    def test_returns_true_when_chi2_is_none(self) -> None:
        """Missing chi2 does not fail validation by itself."""
        from heterodyne.cli.optimization_runner import _validate_warmstart_quality

        result = _make_nlsq_result(success=True, reduced_chi2=1.0)
        result.reduced_chi_squared = None
        assert _validate_warmstart_quality(result) is True
