"""Regression tests for the het_a10cf27e failure mode.

The het_a10cf27e run (39,270 s wall time) reported ``Status: converged`` in
the top-level CLI analysis summary even though 47/47 shards failed
convergence for every one of the 3 phi angles. Two distinct bugs combined:

1. ``commands.py`` evaluated overall convergence via
   ``getattr(r, "success", True)``. ``CMCResult`` has no ``success``
   attribute, so every CMC result silently counted as converged.
2. ``optimization_runner.run_cmc`` continued to the next phi angle after the
   first angle returned a fully-degenerate ``CMCResult`` (all shards failed
   with ``convergence_passed=False``). The remaining angles wasted ~6.5 h
   producing identical empty results.

These tests pin down the corrected behavior.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.cli.commands import _result_converged
from heterodyne.cli.optimization_runner import _is_degenerate_cmc_result
from heterodyne.optimization.cmc.results import CMCResult


def _make_degenerate_cmc_result() -> CMCResult:
    """CMCResult that ``_combine_shard_posteriors`` returns when 0 shards succeed."""
    n_params = 5
    return CMCResult(
        parameter_names=["D0_ref", "alpha_ref", "D0_sample", "alpha_sample", "v0"],
        posterior_mean=np.zeros(n_params),
        posterior_std=np.full(n_params, np.nan),
        credible_intervals={},
        convergence_passed=False,
        r_hat=np.full(n_params, np.nan),
        ess_bulk=np.full(n_params, np.nan),
        ess_tail=np.full(n_params, np.nan),
        bfmi=None,
        samples=None,
    )


def _make_converged_cmc_result() -> CMCResult:
    n_params = 5
    return CMCResult(
        parameter_names=["D0_ref", "alpha_ref", "D0_sample", "alpha_sample", "v0"],
        posterior_mean=np.zeros(n_params),
        posterior_std=np.ones(n_params),
        credible_intervals={},
        convergence_passed=True,
        r_hat=np.full(n_params, 1.01),
        ess_bulk=np.full(n_params, 500.0),
        samples={
            "D0_ref": np.ones((4, 1000)),
            "alpha_ref": np.zeros((4, 1000)),
            "D0_sample": np.ones((4, 1000)),
            "alpha_sample": np.full((4, 1000), -0.5),
            "v0": np.ones((4, 1000)),
        },
    )


@pytest.mark.regression
class TestResultConverged:
    """``_result_converged`` must read ``convergence_passed`` on CMCResult."""

    def test_degenerate_cmc_result_is_not_converged(self) -> None:
        """het_a10cf27e: CMC result with all-shards-failed must NOT report converged."""
        result = _make_degenerate_cmc_result()
        assert _result_converged(result) is False

    def test_converged_cmc_result_is_converged(self) -> None:
        result = _make_converged_cmc_result()
        assert _result_converged(result) is True

    def test_nlsq_result_uses_success_attribute(self) -> None:
        """NLSQResult exposes ``success``, not ``convergence_passed``."""
        nlsq_ok = MagicMock(spec=["success", "reduced_chi_squared"])
        nlsq_ok.success = True
        assert _result_converged(nlsq_ok) is True

        nlsq_bad = MagicMock(spec=["success", "reduced_chi_squared"])
        nlsq_bad.success = False
        assert _result_converged(nlsq_bad) is False

    def test_unknown_result_type_is_not_converged(self) -> None:
        """Conservative default: missing both attrs returns False (not True)."""

        class _Bare:
            pass

        assert _result_converged(_Bare()) is False


@pytest.mark.regression
class TestDegenerateCMCDetector:
    """``_is_degenerate_cmc_result`` recognizes the all-shards-failed sentinel."""

    def test_detects_no_samples_with_failed_convergence(self) -> None:
        assert _is_degenerate_cmc_result(_make_degenerate_cmc_result()) is True

    def test_does_not_flag_converged_result(self) -> None:
        assert _is_degenerate_cmc_result(_make_converged_cmc_result()) is False

    def test_flags_failed_convergence_with_empty_samples_dict(self) -> None:
        result = _make_degenerate_cmc_result()
        result.samples = {"D0_ref": np.array([]), "alpha_ref": np.array([])}
        assert _is_degenerate_cmc_result(result) is True


@pytest.mark.regression
class TestRunCMCShortCircuit:
    """``run_cmc`` must abort remaining angles after a fully-degenerate result.

    Pinned to prevent the het_a10cf27e regression where 11 h was wasted
    producing 3 identical garbage results when angle 0 already returned the
    all-shards-failed sentinel.
    """

    def _make_args(self) -> argparse.Namespace:
        return argparse.Namespace(num_samples=None, num_chains=None)

    @patch("heterodyne.cli.optimization_runner.save_mcmc_results")
    @patch("heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="")
    @patch("heterodyne.cli.optimization_runner.fit_cmc_sharded")
    @patch("heterodyne.cli.optimization_runner.fit_cmc_jax")
    @patch("heterodyne.cli.optimization_runner.CMCConfig")
    def test_three_angle_run_short_circuits_after_first_degenerate(
        self,
        mock_cmc_config_cls: MagicMock,
        mock_fit_cmc_jax: MagicMock,
        mock_fit_cmc_sharded: MagicMock,
        _mock_fmt: MagicMock,
        _mock_save: MagicMock,
        tmp_path,
    ) -> None:
        from heterodyne.cli.optimization_runner import run_cmc

        # Force the small-data path (fit_cmc_jax) so we don't depend on
        # ``should_enable_cmc``/``get_num_shards`` returning a sharded run.
        cmc_cfg = MagicMock()
        cmc_cfg.should_enable_cmc.return_value = False
        cmc_cfg.get_num_shards.return_value = 1
        cmc_cfg.num_samples = 1000
        mock_cmc_config_cls.from_dict.return_value = cmc_cfg

        mock_fit_cmc_jax.return_value = _make_degenerate_cmc_result()

        model = MagicMock()
        model.get_params_dict.return_value = {}
        model.varying_names = ["D0_ref"]

        cm = MagicMock()
        cm.cmc_config = {}

        results = run_cmc(
            model=model,
            c2_data=np.ones((3, 4, 4)),
            phi_angles=[-5.79, 4.88, 90.0],
            config_manager=cm,
            args=self._make_args(),
            output_dir=tmp_path,
            nlsq_results=None,
            summary=None,
            data_phi_angles=None,
        )

        # Only 1 angle should have been fit; remaining 2 must be skipped.
        assert mock_fit_cmc_jax.call_count == 1
        assert mock_fit_cmc_sharded.call_count == 0
        assert len(results) == 1
        assert results[0].convergence_passed is False

    @patch("heterodyne.cli.optimization_runner.save_mcmc_results")
    @patch("heterodyne.cli.optimization_runner.format_mcmc_summary", return_value="")
    @patch("heterodyne.cli.optimization_runner.fit_cmc_jax")
    @patch("heterodyne.cli.optimization_runner.CMCConfig")
    def test_converged_first_angle_still_runs_remaining_angles(
        self,
        mock_cmc_config_cls: MagicMock,
        mock_fit_cmc_jax: MagicMock,
        _mock_fmt: MagicMock,
        _mock_save: MagicMock,
        tmp_path,
    ) -> None:
        """Short-circuit must NOT trigger when the first angle converged."""
        from heterodyne.cli.optimization_runner import run_cmc

        cmc_cfg = MagicMock()
        cmc_cfg.should_enable_cmc.return_value = False
        cmc_cfg.get_num_shards.return_value = 1
        cmc_cfg.num_samples = 1000
        mock_cmc_config_cls.from_dict.return_value = cmc_cfg

        mock_fit_cmc_jax.return_value = _make_converged_cmc_result()

        model = MagicMock()
        model.get_params_dict.return_value = {}
        model.varying_names = ["D0_ref"]

        cm = MagicMock()
        cm.cmc_config = {}

        results = run_cmc(
            model=model,
            c2_data=np.ones((3, 4, 4)),
            phi_angles=[-5.79, 4.88, 90.0],
            config_manager=cm,
            args=self._make_args(),
            output_dir=tmp_path,
            nlsq_results=None,
            summary=None,
            data_phi_angles=None,
        )

        assert mock_fit_cmc_jax.call_count == 3
        assert len(results) == 3
        assert all(r.convergence_passed for r in results)


@pytest.mark.regression
class TestStickyFailureStatus:
    """``AnalysisSummaryLogger.set_convergence_status`` is sticky on failure.

    Defends against future bugs where a code path optimistically calls
    ``set_convergence_status("converged")`` after a failure status was
    already recorded. With this guard the het_a10cf27e CMC failure mode
    would have been caught at the logger boundary even without the
    dispatch-side fix.
    """

    def test_failed_is_not_overwritten_by_converged(self) -> None:
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t1", analysis_mode="two_component")
        summary.set_convergence_status("failed")
        summary.set_convergence_status("converged")  # must be ignored
        assert summary.convergence_status == "failed"
        assert summary.is_failure() is True

    def test_not_converged_is_not_overwritten_by_converged(self) -> None:
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t2", analysis_mode="two_component")
        summary.set_convergence_status("not_converged")
        summary.set_convergence_status("converged")
        assert summary.convergence_status == "not_converged"

    def test_max_iter_is_not_overwritten_by_completed(self) -> None:
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t3", analysis_mode="two_component")
        summary.set_convergence_status("max_iter")
        summary.set_convergence_status("completed")
        assert summary.convergence_status == "max_iter"

    def test_converged_can_be_replaced_by_failed(self) -> None:
        """Success → failure overwrite IS allowed (real failures during cleanup)."""
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t4", analysis_mode="two_component")
        summary.set_convergence_status("converged")
        summary.set_convergence_status("failed")
        assert summary.convergence_status == "failed"

    def test_failure_to_failure_transitions_allowed(self) -> None:
        """A worse failure can overwrite a milder failure (e.g. not_converged → failed)."""
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t5", analysis_mode="two_component")
        summary.set_convergence_status("not_converged")
        summary.set_convergence_status("failed")
        assert summary.convergence_status == "failed"

    def test_initial_status_is_set_unconditionally(self) -> None:
        from heterodyne.utils.logging import AnalysisSummaryLogger

        summary = AnalysisSummaryLogger(run_id="t6", analysis_mode="two_component")
        assert summary.convergence_status is None
        summary.set_convergence_status("converged")
        assert summary.convergence_status == "converged"


@pytest.mark.regression
class TestDispatchExitCode:
    """dispatch_command must return non-zero when convergence failed."""

    def _make_args(self) -> argparse.Namespace:
        from pathlib import Path

        return argparse.Namespace(
            method="cmc",
            config="dummy.yaml",
            output=str(Path("/tmp/test_output")),
            plot=False,
            plot_only=False,
            simulate_only=False,
            verbose=0,
            quiet=False,
        )

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_exit_code_two_when_cmc_not_converged(
        self,
        mock_run_cmc: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        _mock_model_cls: MagicMock,
        _mock_plots: MagicMock,
    ) -> None:
        from heterodyne.cli.commands import dispatch_command

        cm = MagicMock()
        cm.output_dir = "/tmp/test_output"
        mock_load_config.return_value = cm

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        # Return a degenerate CMC result (convergence_passed=False, no samples)
        mock_run_cmc.return_value = [_make_degenerate_cmc_result()]

        rc = dispatch_command(self._make_args())
        assert rc == 2, (
            f"dispatch_command must return 2 on not_converged; got {rc}. "
            "Without this exit code, CI cannot distinguish a successful run "
            "from the het_a10cf27e failure mode."
        )

    @patch("heterodyne.cli.commands.dispatch_plots")
    @patch("heterodyne.cli.commands.HeterodyneModel")
    @patch("heterodyne.cli.commands._load_data")
    @patch("heterodyne.cli.commands.load_and_merge_config")
    @patch("heterodyne.cli.commands.run_cmc")
    def test_exit_code_zero_when_cmc_converged(
        self,
        mock_run_cmc: MagicMock,
        mock_load_config: MagicMock,
        mock_load_data: MagicMock,
        _mock_model_cls: MagicMock,
        _mock_plots: MagicMock,
    ) -> None:
        from heterodyne.cli.commands import dispatch_command

        cm = MagicMock()
        cm.output_dir = "/tmp/test_output"
        mock_load_config.return_value = cm

        mock_data = MagicMock()
        mock_data.c2.shape = (10, 10)
        mock_load_data.return_value = (mock_data, [0.0])

        mock_run_cmc.return_value = [_make_converged_cmc_result()]

        rc = dispatch_command(self._make_args())
        assert rc == 0
