"""Unit tests for NLSQ internal plumbing modules.

Covers:
- parameter_utils (perturb, clip, sensitivity, format)
- parameter_index_mapper (ParameterIndexMapper)
- progress (ProgressTracker, ProgressRecord)
- gradient_monitor (GradientMonitor, GradientSnapshot)
- adaptive_regularization (AdaptiveRegularizer, RegularizationConfig)
- recovery (diagnose_error, safe_uncertainties_from_pcov, execute_with_recovery)
- fallback_chain (OptimizationStrategy, get_fallback_strategy, _is_memory_error)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# parameter_utils
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.parameter_utils import (
    clip_to_bounds,
    compute_parameter_sensitivity,
    format_parameter_table,
    perturb_parameters,
)


class TestPerturbParameters:
    """Tests for perturb_parameters()."""

    def test_basic_perturbation_stays_in_bounds(self) -> None:
        params = np.array([5.0, 50.0, 500.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([10.0, 100.0, 1000.0]))
        rng = np.random.default_rng(42)
        result = perturb_parameters(params, scale=0.1, bounds=bounds, rng=rng)
        assert result.shape == params.shape
        np.testing.assert_array_less(bounds[0] - 1e-15, result)
        np.testing.assert_array_less(result, bounds[1] + 1e-15)

    def test_zero_scale_returns_original(self) -> None:
        params = np.array([1.0, 2.0])
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        rng = np.random.default_rng(0)
        result = perturb_parameters(params, scale=0.0, bounds=bounds, rng=rng)
        np.testing.assert_allclose(result, params, atol=1e-28)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            perturb_parameters(
                np.array([1.0]),
                scale=0.1,
                bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            )

    def test_default_rng_works(self) -> None:
        params = np.array([5.0])
        bounds = (np.array([0.0]), np.array([10.0]))
        result = perturb_parameters(params, scale=0.05, bounds=bounds, rng=None)
        assert result.shape == (1,)

    def test_single_element(self) -> None:
        params = np.array([0.5])
        bounds = (np.array([0.0]), np.array([1.0]))
        rng = np.random.default_rng(99)
        result = perturb_parameters(params, scale=0.5, bounds=bounds, rng=rng)
        assert 0.0 <= result[0] <= 1.0


class TestClipToBounds:
    """Tests for clip_to_bounds()."""

    def test_within_bounds_unchanged(self) -> None:
        params = np.array([1.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        result = clip_to_bounds(params, lower, upper)
        np.testing.assert_array_equal(result, params)

    def test_clips_below(self) -> None:
        result = clip_to_bounds(np.array([-5.0]), np.array([0.0]), np.array([10.0]))
        assert result[0] == 0.0

    def test_clips_above(self) -> None:
        result = clip_to_bounds(np.array([20.0]), np.array([0.0]), np.array([10.0]))
        assert result[0] == 10.0

    def test_empty_array(self) -> None:
        result = clip_to_bounds(np.array([]), np.array([]), np.array([]))
        assert result.shape == (0,)


class TestComputeParameterSensitivity:
    """Tests for compute_parameter_sensitivity()."""

    def test_quadratic_sensitivity(self) -> None:
        # f(x) = x^2 => residual = [x], cost = x^2
        # Sensitivity at x=3 with h: |((3+h)^2 - 9)/h| ≈ 6
        def residual_fn(p: np.ndarray) -> np.ndarray:
            return p.copy()

        params = np.array([3.0])
        sens = compute_parameter_sensitivity(residual_fn, params)
        assert sens.shape == (1,)
        np.testing.assert_allclose(sens[0], 6.0, rtol=0.01)

    def test_custom_step_sizes(self) -> None:
        def residual_fn(p: np.ndarray) -> np.ndarray:
            return p.copy()

        params = np.array([1.0, 2.0])
        step = np.array([0.001, 0.001])
        sens = compute_parameter_sensitivity(residual_fn, params, step_sizes=step)
        assert sens.shape == (2,)

    def test_zero_sensitivity(self) -> None:
        # Constant residual => zero sensitivity
        def residual_fn(p: np.ndarray) -> np.ndarray:
            return np.array([1.0])

        params = np.array([5.0])
        sens = compute_parameter_sensitivity(residual_fn, params)
        np.testing.assert_allclose(sens[0], 0.0, atol=1e-6)


class TestFormatParameterTable:
    """Tests for format_parameter_table()."""

    def test_basic_table(self) -> None:
        names = ["alpha", "beta"]
        values = np.array([1.23e-3, 4.56e2])
        result = format_parameter_table(names, values)
        assert "alpha" in result
        assert "beta" in result
        assert "Parameter" in result

    def test_with_uncertainties(self) -> None:
        names = ["x"]
        values = np.array([1.0])
        unc = np.array([0.1])
        result = format_parameter_table(names, values, uncertainties=unc)
        assert "Uncertainty" in result

    def test_with_bounds(self) -> None:
        names = ["x"]
        values = np.array([5.0])
        bounds = (np.array([0.0]), np.array([10.0]))
        result = format_parameter_table(names, values, bounds=bounds)
        assert "Lower" in result
        assert "Upper" in result

    def test_with_all_columns(self) -> None:
        names = ["a", "b"]
        values = np.array([1.0, 2.0])
        unc = np.array([0.1, 0.2])
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        result = format_parameter_table(names, values, uncertainties=unc, bounds=bounds)
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 rows


# ---------------------------------------------------------------------------
# parameter_index_mapper
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper


class TestParameterIndexMapper:
    """Tests for ParameterIndexMapper."""

    @pytest.fixture()
    def mapper(self) -> ParameterIndexMapper:
        return ParameterIndexMapper(
            varying_names=["D0_ref", "alpha_ref", "v0"],
            varying_full_indices=[0, 1, 6],
            log_mask=[True, False, True],
        )

    def test_n_varying(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.n_varying == 3

    def test_n_full(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.n_full == 14

    def test_varying_names(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.varying_names == ["D0_ref", "alpha_ref", "v0"]

    def test_varying_full_indices(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.varying_full_indices == [0, 1, 6]

    def test_fixed_full_indices(self, mapper: ParameterIndexMapper) -> None:
        fixed = mapper.fixed_full_indices
        assert 0 not in fixed
        assert 1 not in fixed
        assert 6 not in fixed
        assert len(fixed) == 11

    def test_log_mask(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.log_mask == [True, False, True]

    def test_default_log_mask_all_false(self) -> None:
        m = ParameterIndexMapper(
            varying_names=["D0_ref"],
            varying_full_indices=[0],
        )
        assert m.log_mask == [False]

    def test_full_to_varying(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.full_to_varying(0) == 0
        assert mapper.full_to_varying(1) == 1
        assert mapper.full_to_varying(6) == 2

    def test_full_to_varying_nonvarying_raises(self, mapper: ParameterIndexMapper) -> None:
        with pytest.raises(KeyError, match="not varying"):
            mapper.full_to_varying(3)

    def test_varying_to_full(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.varying_to_full(0) == 0
        assert mapper.varying_to_full(2) == 6

    def test_varying_to_full_out_of_range(self, mapper: ParameterIndexMapper) -> None:
        with pytest.raises(IndexError, match="out of range"):
            mapper.varying_to_full(5)
        with pytest.raises(IndexError, match="out of range"):
            mapper.varying_to_full(-1)

    def test_get_name(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.get_name(0) == "D0_ref"
        assert mapper.get_name(2) == "v0"

    def test_get_name_out_of_range(self, mapper: ParameterIndexMapper) -> None:
        with pytest.raises(IndexError):
            mapper.get_name(10)

    def test_name_to_varying(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.name_to_varying("v0") == 2

    def test_name_to_varying_missing(self, mapper: ParameterIndexMapper) -> None:
        with pytest.raises(KeyError, match="not varying"):
            mapper.name_to_varying("f0")

    def test_is_log_transformed(self, mapper: ParameterIndexMapper) -> None:
        assert mapper.is_log_transformed(0) is True
        assert mapper.is_log_transformed(1) is False

    def test_is_log_transformed_out_of_range(self, mapper: ParameterIndexMapper) -> None:
        with pytest.raises(IndexError):
            mapper.is_log_transformed(10)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            ParameterIndexMapper(
                varying_names=["a", "b"],
                varying_full_indices=[0],
            )

    def test_build_from_manager_no_log(self) -> None:
        pm = SimpleNamespace(
            varying_names=["D0_ref", "v0"],
            varying_indices=[0, 6],
        )
        m = ParameterIndexMapper.build_from_manager(pm)  # type: ignore[arg-type]
        assert m.n_varying == 2
        assert m.log_mask == [False, False]

    def test_build_from_manager_with_log(self) -> None:
        pm = SimpleNamespace(
            varying_names=["D0_ref", "v0"],
            varying_indices=[0, 6],
        )
        m = ParameterIndexMapper.build_from_manager(pm, use_log=True)  # type: ignore[arg-type]
        assert m.n_varying == 2
        # Just verify it produced some log_mask (exact values depend on registry)
        assert len(m.log_mask) == 2


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.progress import ProgressRecord, ProgressTracker


class TestProgressRecord:
    """Tests for ProgressRecord dataclass."""

    def test_defaults(self) -> None:
        rec = ProgressRecord(iteration=0, cost=1.0)
        assert rec.cost_change is None
        assert rec.gradient_norm is None
        assert rec.step_norm is None
        assert rec.wall_time == 0.0


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_empty_tracker(self) -> None:
        tracker = ProgressTracker()
        assert tracker.n_records == 0
        assert not tracker.is_stalled()
        assert "no iterations" in tracker.summary()

    def test_record_single_iteration(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=10.0)
        assert tracker.n_records == 1
        history = tracker.get_history()
        assert history[0].cost == 10.0
        assert history[0].cost_change is None

    def test_cost_change_tracked(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=10.0)
        tracker.record(1, cost=8.0)
        history = tracker.get_history()
        assert history[1].cost_change == pytest.approx(-2.0)

    def test_gradient_norm_tracked(self) -> None:
        tracker = ProgressTracker()
        grad = np.array([3.0, 4.0])
        tracker.record(0, cost=1.0, gradient=grad)
        assert tracker.get_history()[0].gradient_norm == pytest.approx(5.0)

    def test_step_norm_tracked(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=10.0, params=np.array([0.0, 0.0]))
        tracker.record(1, cost=9.0, params=np.array([3.0, 4.0]))
        assert tracker.get_history()[1].step_norm == pytest.approx(5.0)

    def test_step_norm_none_on_first(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=10.0, params=np.array([1.0]))
        assert tracker.get_history()[0].step_norm is None

    def test_is_stalled_true(self) -> None:
        tracker = ProgressTracker()
        for i in range(15):
            tracker.record(i, cost=10.0)
        assert tracker.is_stalled(patience=10)

    def test_is_stalled_false_with_improvement(self) -> None:
        tracker = ProgressTracker()
        for i in range(15):
            tracker.record(i, cost=10.0 - i * 0.1)
        assert not tracker.is_stalled(patience=10, min_improvement=0.01)

    def test_is_stalled_not_enough_history(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=1.0)
        tracker.record(1, cost=1.0)
        assert not tracker.is_stalled(patience=10)

    def test_summary_with_records(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=100.0, gradient=np.array([1.0]), params=np.array([0.0]))
        tracker.record(1, cost=50.0, gradient=np.array([0.5]), params=np.array([1.0]))
        summary = tracker.summary()
        assert "Initial cost" in summary
        assert "Final cost" in summary
        assert "Cost reduction" in summary
        assert "50.00%" in summary

    def test_summary_zero_initial_cost(self) -> None:
        tracker = ProgressTracker()
        tracker.record(0, cost=0.0)
        summary = tracker.summary()
        assert "Cost reduction" not in summary


# ---------------------------------------------------------------------------
# gradient_monitor
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.gradient_monitor import GradientMonitor, GradientSnapshot


class TestGradientSnapshot:
    """Tests for GradientSnapshot dataclass."""

    def test_creation(self) -> None:
        snap = GradientSnapshot(
            iteration=0,
            gradient_norm=1.0,
            max_gradient=0.5,
            parameter_gradients=np.array([0.5, -0.3]),
        )
        assert snap.iteration == 0
        assert snap.gradient_norm == 1.0


class TestGradientMonitor:
    """Tests for GradientMonitor."""

    def test_empty_monitor(self) -> None:
        monitor = GradientMonitor()
        assert monitor.n_records == 0
        assert not monitor.check_vanishing()
        assert not monitor.check_exploding()

    def test_record_and_history(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([3.0, 4.0]))
        assert monitor.n_records == 1
        snap = monitor.history[0]
        assert snap.gradient_norm == pytest.approx(5.0)
        assert snap.max_gradient == pytest.approx(4.0)

    def test_empty_gradient_vector(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([]))
        assert monitor.history[0].max_gradient == 0.0

    def test_check_vanishing_true(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([1e-15]))
        assert monitor.check_vanishing(threshold=1e-12)

    def test_check_vanishing_false(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([1.0]))
        assert not monitor.check_vanishing()

    def test_check_exploding_true(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([1e12]))
        assert monitor.check_exploding(threshold=1e10)

    def test_check_exploding_nan(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([np.nan]))
        assert monitor.check_exploding()

    def test_check_exploding_inf(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([np.inf]))
        assert monitor.check_exploding()

    def test_get_summary_empty(self) -> None:
        monitor = GradientMonitor()
        summary = monitor.get_summary()
        assert summary["n_iterations"] == 0
        assert summary["final_norm"] is None

    def test_get_summary_with_records(self) -> None:
        monitor = GradientMonitor(parameter_names=["a", "b", "c"])
        monitor.record(0, np.array([1.0, 2.0, 3.0]))
        monitor.record(1, np.array([0.5, 1.0, 0.1]))
        summary = monitor.get_summary()
        assert summary["n_iterations"] == 2
        assert summary["final_norm"] == pytest.approx(float(np.linalg.norm([0.5, 1.0, 0.1])))
        assert summary["is_vanishing"] == False  # noqa: E712
        assert summary["is_exploding"] == False  # noqa: E712
        assert len(summary["worst_parameters"]) == 3

    def test_get_summary_without_names(self) -> None:
        monitor = GradientMonitor()
        monitor.record(0, np.array([1.0]))
        summary = monitor.get_summary()
        assert summary["worst_parameters"] == []


# ---------------------------------------------------------------------------
# adaptive_regularization
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizer,
    RegularizationConfig,
)


class TestRegularizationConfig:
    """Tests for RegularizationConfig."""

    def test_defaults(self) -> None:
        cfg = RegularizationConfig()
        assert cfg.lambda_init == 1e-6
        assert cfg.adaptation_rate == 2.0

    def test_invalid_lambda_init(self) -> None:
        with pytest.raises(ValueError, match="lambda_init must be positive"):
            RegularizationConfig(lambda_init=-1.0)

    def test_invalid_lambda_min(self) -> None:
        with pytest.raises(ValueError, match="lambda_min must be positive"):
            RegularizationConfig(lambda_min=0.0)

    def test_lambda_max_below_min(self) -> None:
        with pytest.raises(ValueError, match="lambda_max.*must exceed"):
            RegularizationConfig(lambda_min=1.0, lambda_max=0.5)

    def test_adaptation_rate_too_low(self) -> None:
        with pytest.raises(ValueError, match="adaptation_rate must be > 1.0"):
            RegularizationConfig(adaptation_rate=0.5)


class TestAdaptiveRegularizer:
    """Tests for AdaptiveRegularizer."""

    def test_default_config(self) -> None:
        reg = AdaptiveRegularizer()
        assert reg.current_lambda == 1e-6

    def test_custom_config(self) -> None:
        cfg = RegularizationConfig(lambda_init=1e-3)
        reg = AdaptiveRegularizer(config=cfg)
        assert reg.current_lambda == 1e-3

    def test_compute_regularized_step_simple(self) -> None:
        # 1-param: J = [[2]], r = [1] => JtJ=4, Jtr=2
        # (4 + lambda*1)*delta = -2 => delta = -2/(4+lambda)
        reg = AdaptiveRegularizer()
        J = np.array([[2.0]])
        r = np.array([1.0])
        step = reg.compute_regularized_step(J, r, lambda_=0.0)
        np.testing.assert_allclose(step, np.array([-0.5]), atol=1e-12)

    def test_compute_regularized_step_with_regularization(self) -> None:
        J = np.array([[2.0]])
        r = np.array([1.0])
        reg = AdaptiveRegularizer()
        step = reg.compute_regularized_step(J, r, lambda_=4.0)
        # (4+4)*delta = -2 => delta = -0.25
        np.testing.assert_allclose(step, np.array([-0.25]), atol=1e-12)

    def test_compute_regularized_step_multivariate(self) -> None:
        J = np.eye(3)
        r = np.array([1.0, 2.0, 3.0])
        reg = AdaptiveRegularizer()
        step = reg.compute_regularized_step(J, r, lambda_=0.0)
        np.testing.assert_allclose(step, -r, atol=1e-12)

    def test_adapt_lambda_cost_decrease(self) -> None:
        cfg = RegularizationConfig(lambda_init=1e-3, adaptation_rate=2.0)
        reg = AdaptiveRegularizer(config=cfg)
        new_lambda = reg.adapt_lambda(cost_new=5.0, cost_old=10.0)
        assert new_lambda == pytest.approx(5e-4)
        assert reg.current_lambda == pytest.approx(5e-4)

    def test_adapt_lambda_cost_increase(self) -> None:
        cfg = RegularizationConfig(lambda_init=1e-3, adaptation_rate=2.0)
        reg = AdaptiveRegularizer(config=cfg)
        new_lambda = reg.adapt_lambda(cost_new=15.0, cost_old=10.0)
        assert new_lambda == pytest.approx(2e-3)

    def test_adapt_lambda_respects_min(self) -> None:
        cfg = RegularizationConfig(lambda_init=1e-12, lambda_min=1e-12)
        reg = AdaptiveRegularizer(config=cfg)
        new_lambda = reg.adapt_lambda(cost_new=1.0, cost_old=10.0)
        assert new_lambda >= cfg.lambda_min

    def test_adapt_lambda_respects_max(self) -> None:
        cfg = RegularizationConfig(lambda_init=0.5, lambda_max=1.0)
        reg = AdaptiveRegularizer(config=cfg)
        new_lambda = reg.adapt_lambda(cost_new=20.0, cost_old=10.0)
        assert new_lambda <= cfg.lambda_max

    def test_regularize_covariance_positive_definite(self) -> None:
        reg = AdaptiveRegularizer()
        cov = np.eye(3) * 0.01
        result = reg.regularize_covariance(cov, lambda_=0.001)
        assert result.shape == (3, 3)
        # Should be positive definite
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)

    def test_regularize_covariance_non_square_raises(self) -> None:
        reg = AdaptiveRegularizer()
        with pytest.raises(ValueError, match="square"):
            reg.regularize_covariance(np.ones((2, 3)))

    def test_regularize_covariance_uses_internal_lambda(self) -> None:
        cfg = RegularizationConfig(lambda_init=0.1)
        reg = AdaptiveRegularizer(config=cfg)
        cov = np.eye(2)
        result = reg.regularize_covariance(cov)
        expected = np.eye(2) + 0.1 * np.eye(2)
        np.testing.assert_allclose(result, expected)

    def test_regularize_covariance_negative_definite_escalation(self) -> None:
        """Regularize a negative-definite matrix, triggering lambda escalation."""
        reg = AdaptiveRegularizer(
            config=RegularizationConfig(lambda_init=1e-12, lambda_min=1e-12, lambda_max=1.0)
        )
        # Matrix with negative eigenvalue
        cov = np.array([[-10.0, 0.0], [0.0, 1.0]])
        result = reg.regularize_covariance(cov, lambda_=1e-12)
        # Result should have been regularized enough to be usable
        assert result.shape == (2, 2)
        # The diagonal should be shifted
        assert result[0, 0] > cov[0, 0]

    def test_reset(self) -> None:
        cfg = RegularizationConfig(lambda_init=1e-3)
        reg = AdaptiveRegularizer(config=cfg)
        reg.adapt_lambda(cost_new=20.0, cost_old=10.0)
        assert reg.current_lambda != 1e-3
        reg.reset()
        assert reg.current_lambda == 1e-3


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------
from heterodyne.optimization.nlsq.recovery import (
    CATEGORY_BOUNDS,
    CATEGORY_CONVERGENCE,
    CATEGORY_ILL_CONDITIONED,
    CATEGORY_NAN,
    CATEGORY_OOM,
    CATEGORY_UNKNOWN,
    diagnose_error,
    execute_with_recovery,
    safe_uncertainties_from_pcov,
)


class TestDiagnoseError:
    """Tests for diagnose_error()."""

    def test_memory_error_type(self) -> None:
        diag = diagnose_error(MemoryError("out of memory"))
        assert diag.category == CATEGORY_OOM
        assert diag.recoverable is True

    def test_memory_keyword_in_message(self) -> None:
        diag = diagnose_error(RuntimeError("XLA OOM during allocation"))
        assert diag.category == CATEGORY_OOM

    def test_nan_error(self) -> None:
        diag = diagnose_error(ValueError("result contains NaN"))
        assert diag.category == CATEGORY_NAN
        assert diag.suggested_action == "perturb_parameters"

    def test_inf_error(self) -> None:
        diag = diagnose_error(ValueError("value is inf"))
        assert diag.category == CATEGORY_NAN

    def test_bounds_error(self) -> None:
        diag = diagnose_error(ValueError("parameter out of bounds"))
        assert diag.category == CATEGORY_BOUNDS

    def test_singular_error(self) -> None:
        diag = diagnose_error(np.linalg.LinAlgError("Singular matrix"))
        assert diag.category == CATEGORY_ILL_CONDITIONED

    def test_convergence_error(self) -> None:
        diag = diagnose_error(RuntimeError("Exceeded max iterations"))
        assert diag.category == CATEGORY_CONVERGENCE
        assert diag.suggested_action == "relax_tolerance"

    def test_max_nfev(self) -> None:
        diag = diagnose_error(RuntimeError("max nfev reached"))
        assert diag.category == CATEGORY_CONVERGENCE

    def test_unknown_error(self) -> None:
        diag = diagnose_error(TypeError("something weird"))
        assert diag.category == CATEGORY_UNKNOWN
        assert diag.recoverable is False


class TestSafeUncertaintiesFromPcov:
    """Tests for safe_uncertainties_from_pcov()."""

    def test_none_pcov(self) -> None:
        result = safe_uncertainties_from_pcov(None, n_params=3)
        assert result.shape == (3,)
        assert np.all(np.isinf(result))

    def test_valid_pcov(self) -> None:
        pcov = np.diag([4.0, 9.0, 16.0])
        result = safe_uncertainties_from_pcov(pcov, n_params=3)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_negative_diagonal(self) -> None:
        pcov = np.diag([-1.0, 4.0])
        result = safe_uncertainties_from_pcov(pcov, n_params=2)
        assert np.isinf(result[0])
        assert result[1] == pytest.approx(2.0)

    def test_wrong_shape(self) -> None:
        pcov = np.eye(2)
        result = safe_uncertainties_from_pcov(pcov, n_params=5)
        assert result.shape == (5,)
        assert np.all(np.isinf(result))

    def test_zero_diagonal(self) -> None:
        pcov = np.diag([0.0, 1.0])
        result = safe_uncertainties_from_pcov(pcov, n_params=2)
        assert np.isinf(result[0])
        assert result[1] == pytest.approx(1.0)


class TestExecuteWithRecovery:
    """Tests for execute_with_recovery()."""

    def _make_result(self, *, success: bool, cost: float = 1.0, message: str = "ok") -> SimpleNamespace:
        return SimpleNamespace(
            success=success,
            final_cost=cost,
            message=message,
            metadata={},
        )

    def test_succeeds_on_first_attempt(self) -> None:
        result = self._make_result(success=True, cost=0.5)
        fit_fn = MagicMock(return_value=result)
        config = SimpleNamespace(ftol=1e-8, xtol=1e-8, gtol=1e-8, method="trf")

        out = execute_with_recovery(
            fit_fn,
            initial_params=np.array([1.0, 2.0]),
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            config=config,  # type: ignore[arg-type]
        )
        assert out.success is True
        assert fit_fn.call_count == 1

    def test_succeeds_on_retry_after_failure(self) -> None:
        fail_result = self._make_result(success=False, message="no converge")
        ok_result = self._make_result(success=True, cost=1.0)
        fit_fn = MagicMock(side_effect=[fail_result, ok_result])
        config = SimpleNamespace(ftol=1e-8, xtol=1e-8, gtol=1e-8, method="trf")

        out = execute_with_recovery(
            fit_fn,
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
            config=config,  # type: ignore[arg-type]
            max_retries=3,
        )
        assert out.success is True
        assert fit_fn.call_count == 2

    def test_raises_after_all_retries_fail(self) -> None:
        fail_result = self._make_result(success=False, message="no converge")
        fit_fn = MagicMock(return_value=fail_result)
        config = SimpleNamespace(ftol=1e-8, xtol=1e-8, gtol=1e-8, method="trf")

        with pytest.raises(RuntimeError, match="All .* recovery attempts failed"):
            execute_with_recovery(
                fit_fn,
                initial_params=np.array([1.0]),
                bounds=(np.array([0.0]), np.array([10.0])),
                config=config,  # type: ignore[arg-type]
                max_retries=2,
            )

    def test_raises_on_unrecoverable_exception(self) -> None:
        def bad_fit(params, bounds, cfg):
            raise TypeError("completely broken")

        config = SimpleNamespace(ftol=1e-8, xtol=1e-8, gtol=1e-8, method="trf")

        with pytest.raises(RuntimeError, match="recovery attempts failed"):
            execute_with_recovery(
                bad_fit,
                initial_params=np.array([1.0]),
                bounds=(np.array([0.0]), np.array([10.0])),
                config=config,  # type: ignore[arg-type]
                max_retries=1,
            )

    def test_recovery_metadata_added(self) -> None:
        result = self._make_result(success=True)
        fit_fn = MagicMock(return_value=result)
        config = SimpleNamespace(ftol=1e-8, xtol=1e-8, gtol=1e-8, method="trf")

        out = execute_with_recovery(
            fit_fn,
            initial_params=np.array([1.0]),
            bounds=(np.array([0.0]), np.array([10.0])),
            config=config,  # type: ignore[arg-type]
        )
        assert "recovery" in out.metadata
        assert out.metadata["recovery"]["successful_action"] == "initial"


# ---------------------------------------------------------------------------
# fallback_chain — tests moved to test_fallback_chain.py (NLSQ redesign)
# ---------------------------------------------------------------------------
