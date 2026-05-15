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

    def test_calls_multi_phi_fit_once_for_all_angles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_nlsq delegates all selected phi angles to one multi-angle fit."""
        import heterodyne.cli.optimization_runner as runner
        from heterodyne.cli.optimization_runner import run_nlsq

        phi_angles = [0.0, 45.0, 90.0]
        multi_result = [_make_nlsq_result() for _ in phi_angles]
        calls: list[dict[str, object]] = []

        def fake_multi_phi(**kwargs):
            calls.append(kwargs)
            return multi_result

        def fail_single_phi(**kwargs):
            raise AssertionError("single-angle fit should not be called")

        monkeypatch.setattr(runner, "fit_nlsq_multi_phi", fake_multi_phi, raising=False)
        monkeypatch.setattr(runner, "fit_nlsq_jax", fail_single_phi, raising=False)
        monkeypatch.setattr(runner, "format_nlsq_summary", lambda result: "summary")
        monkeypatch.setattr(runner, "save_nlsq_json_files", lambda *a, **k: {})
        monkeypatch.setattr(
            runner, "save_nlsq_npz_file", lambda *a, **k: Path("/tmp/out.npz")
        )

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

        assert len(calls) == 1
        assert calls[0]["model"] is mock_model
        np.testing.assert_allclose(calls[0]["c2_data"], c2_data[:, 1:, 1:])
        assert list(calls[0]["phi_angles"]) == phi_angles
        assert len(results) == len(phi_angles)
        mock_model.sync_time_axis.assert_called_once()
        np.testing.assert_allclose(
            mock_model.sync_time_axis.call_args.args[0],
            np.arange(c2_data.shape[-1] - 1, dtype=float),
        )

    def test_excludes_first_time_point_for_2d_nlsq_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_nlsq excludes the leading time point before fitting 2D data."""
        import heterodyne.cli.optimization_runner as runner
        from heterodyne.cli.optimization_runner import run_nlsq

        c2_data = np.arange(16, dtype=float).reshape(4, 4)
        result = _make_nlsq_result()
        calls: list[dict[str, object]] = []

        def fake_multi_phi(**kwargs):
            calls.append(kwargs)
            return [result]

        monkeypatch.setattr(runner, "fit_nlsq_multi_phi", fake_multi_phi)
        monkeypatch.setattr(runner, "format_nlsq_summary", lambda result: "summary")
        monkeypatch.setattr(runner, "save_nlsq_json_files", lambda *a, **k: {})
        monkeypatch.setattr(
            runner, "save_nlsq_npz_file", lambda *a, **k: Path("/tmp/out.npz")
        )

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.nlsq_config = {}

        run_nlsq(
            model=mock_model,
            c2_data=c2_data,
            phi_angles=[0.0],
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
        )

        np.testing.assert_allclose(calls[0]["c2_data"], c2_data[1:, 1:])
        mock_model.sync_time_axis.assert_called_once()

    def test_selects_phi_slices_after_angle_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_nlsq matches normalized target angles to raw detector angles."""
        import heterodyne.cli.optimization_runner as runner
        from heterodyne.cli.optimization_runner import run_nlsq

        c2_data = np.stack(
            [
                np.full((4, 4), -25.0),
                np.full((4, 4), 185.0),
                np.full((4, 4), 196.0),
            ],
            axis=0,
        )
        result = _make_nlsq_result()
        calls: list[dict[str, object]] = []

        def fake_multi_phi(**kwargs):
            calls.append(kwargs)
            return [result]

        monkeypatch.setattr(runner, "fit_nlsq_multi_phi", fake_multi_phi)
        monkeypatch.setattr(runner, "format_nlsq_summary", lambda result: "summary")
        monkeypatch.setattr(runner, "save_nlsq_json_files", lambda *a, **k: {})
        monkeypatch.setattr(
            runner, "save_nlsq_npz_file", lambda *a, **k: Path("/tmp/out.npz")
        )

        mock_model = MagicMock()
        mock_config_mgr = MagicMock()
        mock_config_mgr.nlsq_config = {}

        run_nlsq(
            model=mock_model,
            c2_data=c2_data,
            phi_angles=[-175.0],
            config_manager=mock_config_mgr,
            args=_make_args(),
            output_dir=Path("/tmp/test_out"),
            data_phi_angles=np.array([-25.0, 185.0, 196.0]),
        )

        np.testing.assert_allclose(calls[0]["c2_data"], c2_data[1:2, 1:, 1:])

    @patch("heterodyne.cli.optimization_runner.save_nlsq_npz_file")
    @patch("heterodyne.cli.optimization_runner.save_nlsq_json_files")
    @patch(
        "heterodyne.cli.optimization_runner.format_nlsq_summary", return_value="summary"
    )
    @patch("heterodyne.cli.optimization_runner.fit_nlsq_multi_phi")
    def test_returns_list_of_nlsq_results(
        self,
        mock_fit_multi: MagicMock,
        mock_fmt: MagicMock,
        mock_save_json: MagicMock,
        mock_save_npz: MagicMock,
    ) -> None:
        """run_nlsq returns a list of NLSQResult objects."""
        from heterodyne.cli.optimization_runner import run_nlsq

        result = _make_nlsq_result()
        result.metadata = {}
        mock_fit_multi.return_value = [result]

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
    @patch(
        "heterodyne.cli.optimization_runner.format_nlsq_summary", return_value="summary"
    )
    @patch("heterodyne.cli.optimization_runner.fit_nlsq_multi_phi")
    def test_records_chi2_metric_in_summary(
        self,
        mock_fit_multi: MagicMock,
        mock_fmt: MagicMock,
        mock_save_json: MagicMock,
        mock_save_npz: MagicMock,
    ) -> None:
        """run_nlsq records reduced_chi_squared in summary when provided."""
        from heterodyne.cli.optimization_runner import run_nlsq
        from heterodyne.utils.logging import AnalysisSummaryLogger

        result = _make_nlsq_result(reduced_chi2=2.5)
        result.metadata = {}
        mock_fit_multi.return_value = [result]

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

    def test_combined_result_computes_cost_from_stacked_residuals(self) -> None:
        """Aggregate saved cost should not multiply shared joint-fit cost."""
        from heterodyne.cli.optimization_runner import _combine_nlsq_results

        results = [
            NLSQResult(
                parameters=np.array([1.0]),
                parameter_names=["D0_ref"],
                success=True,
                message="ok",
                final_cost=100.0,
                residuals=np.array([1.0, 2.0]),
                metadata={"phi_angle": 0.0},
            ),
            NLSQResult(
                parameters=np.array([1.0]),
                parameter_names=["D0_ref"],
                success=True,
                message="ok",
                final_cost=100.0,
                residuals=np.array([3.0]),
                metadata={"phi_angle": 90.0},
            ),
        ]

        aggregate = _combine_nlsq_results(results)

        assert aggregate.final_cost == pytest.approx(0.5 * (1.0 + 4.0 + 9.0))


@pytest.mark.unit
class TestRunCMC:
    """Tests for run_cmc function."""

    @patch("heterodyne.cli.optimization_runner.save_mcmc_results")
    @patch(
        "heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="summary"
    )
    @patch("heterodyne.cli.optimization_runner.fit_cmc_jax")
    def test_calls_fit_cmc_for_each_angle(
        self,
        mock_fit: MagicMock,
        mock_fmt: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """run_cmc calls fit_cmc_jax once per phi angle."""
        from heterodyne.cli.optimization_runner import run_cmc
        from heterodyne.optimization.cmc.results import CMCResult

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
    @patch(
        "heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="summary"
    )
    @patch(
        "heterodyne.cli.optimization_runner._validate_warmstart_quality",
        return_value=True,
    )
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
        from heterodyne.cli.optimization_runner import run_cmc
        from heterodyne.optimization.cmc.results import CMCResult

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


@pytest.mark.unit
class TestClampWarmstartToInterior:
    """Tests for _clamp_warmstart_to_interior (5% boundary margin)."""

    def _clamp(self, **params: float):  # type: ignore[return]
        from heterodyne.cli.optimization_runner import _clamp_warmstart_to_interior

        result = _make_nlsq_result(params=params)
        return _clamp_warmstart_to_interior(result)

    def test_alpha_at_lower_bound_is_clamped(self) -> None:
        """alpha_sample=-5 (lower hard bound) must be shifted to -4.5 (5% of range 10)."""
        clamped = self._clamp(alpha_sample=-5.0)
        idx = list(clamped.parameter_names).index("alpha_sample")
        assert abs(float(clamped.parameters[idx]) - (-4.5)) < 1e-9

    def test_alpha_at_upper_bound_is_clamped(self) -> None:
        """alpha_ref=5 (upper hard bound) must be shifted to 4.5 (5% of range 10)."""
        clamped = self._clamp(alpha_ref=5.0)
        idx = list(clamped.parameter_names).index("alpha_ref")
        assert abs(float(clamped.parameters[idx]) - 4.5) < 1e-9

    def test_interior_alpha_is_unchanged(self) -> None:
        """alpha_sample in the interior must not be modified."""
        result = self._clamp(alpha_sample=0.5)
        idx = list(result.parameter_names).index("alpha_sample")
        assert float(result.parameters[idx]) == pytest.approx(0.5)

    def test_log_space_param_uses_geometric_margin(self) -> None:
        """D0_ref at min_bound=100 must be clamped via geometric, not linear, margin."""
        import numpy as np

        # Expected: 100 × exp(0.05 × log(1e6 / 100)) ≈ 158.49
        expected = 100.0 * np.exp(0.05 * np.log(1e6 / 100.0))
        clamped = self._clamp(D0_ref=100.0)
        idx = list(clamped.parameter_names).index("D0_ref")
        assert float(clamped.parameters[idx]) == pytest.approx(expected, rel=1e-6)

    def test_log_space_interior_param_is_unchanged(self) -> None:
        """D0_ref at its default 1e4 is well inside bounds and must not be clamped."""
        result = self._clamp(D0_ref=1e4)
        idx = list(result.parameter_names).index("D0_ref")
        assert float(result.parameters[idx]) == pytest.approx(1e4)
