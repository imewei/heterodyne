"""Tests for heterodyne.optimization.nlsq.hierarchical."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_GROUPS
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.hierarchical import (
    HierarchicalConfig,
    HierarchicalFitter,
    HierarchicalResult,
    _resolve_stage_params,
)
from heterodyne.optimization.nlsq.results import NLSQResult

# ---------------------------------------------------------------------------
# HierarchicalConfig
# ---------------------------------------------------------------------------


class TestHierarchicalConfig:
    """Tests for HierarchicalConfig dataclass."""

    def test_default_stages(self) -> None:
        config = HierarchicalConfig()
        assert len(config.stages) == 4
        names = [s["name"] for s in config.stages]
        assert names == ["transport", "velocity", "fraction", "all"]

    def test_stage_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required 'name'"):
            HierarchicalConfig(stages=[{"groups": ["reference"]}])

    def test_stage_missing_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'groups'"):
            HierarchicalConfig(stages=[{"name": "test"}])

    def test_invalid_group_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown group"):
            HierarchicalConfig(
                stages=[{"name": "bad", "groups": ["nonexistent_group"]}]
            )

    def test_custom_stages(self) -> None:
        stages = [
            {"name": "ref_only", "groups": ["reference"]},
            {
                "name": "all",
                "groups": ["reference", "sample", "velocity", "fraction", "angle"],
            },
        ]
        config = HierarchicalConfig(stages=stages)
        assert len(config.stages) == 2

    def test_per_stage_config(self) -> None:
        custom_config = NLSQConfig(max_iterations=500)
        config = HierarchicalConfig(per_stage_config={"transport": custom_config})
        assert config.per_stage_config["transport"].max_iterations == 500

    def test_skip_failed_stages_default(self) -> None:
        config = HierarchicalConfig()
        assert config.skip_failed_stages is False


# ---------------------------------------------------------------------------
# _resolve_stage_params
# ---------------------------------------------------------------------------


class TestResolveStageParams:
    """Tests for _resolve_stage_params."""

    def test_single_group(self) -> None:
        stage = {"name": "ref", "groups": ["reference"]}
        user_vary = dict.fromkeys(ALL_PARAM_NAMES, True)
        result = _resolve_stage_params(stage, user_vary)

        assert result == set(PARAM_GROUPS["reference"])

    def test_multiple_groups(self) -> None:
        stage = {"name": "transport", "groups": ["reference", "sample"]}
        user_vary = dict.fromkeys(ALL_PARAM_NAMES, True)
        result = _resolve_stage_params(stage, user_vary)

        expected = set(PARAM_GROUPS["reference"]) | set(PARAM_GROUPS["sample"])
        assert result == expected

    def test_respects_user_vary_flags(self) -> None:
        stage = {"name": "ref", "groups": ["reference"]}
        # Only D0_ref varies, alpha_ref and D_offset_ref are fixed
        user_vary = dict.fromkeys(ALL_PARAM_NAMES, False)
        user_vary["D0_ref"] = True
        result = _resolve_stage_params(stage, user_vary)

        assert result == {"D0_ref"}

    def test_no_varying_params(self) -> None:
        stage = {"name": "ref", "groups": ["reference"]}
        user_vary = dict.fromkeys(ALL_PARAM_NAMES, False)
        result = _resolve_stage_params(stage, user_vary)

        assert result == set()

    def test_param_not_in_user_vary(self) -> None:
        """Parameters missing from user_vary dict default to not varying."""
        stage = {"name": "ref", "groups": ["reference"]}
        user_vary: dict[str, bool] = {}  # empty
        result = _resolve_stage_params(stage, user_vary)

        assert result == set()


# ---------------------------------------------------------------------------
# HierarchicalResult
# ---------------------------------------------------------------------------


class TestHierarchicalResult:
    """Tests for HierarchicalResult dataclass."""

    def test_construction(self) -> None:
        result = HierarchicalResult(
            best_params={"D0_ref": 1e4, "alpha_ref": 0.5},
            best_cost=0.01,
            stage_results=[
                {
                    "stage": "transport",
                    "success": True,
                    "cost": 0.05,
                    "n_iterations": 10,
                },
                {"stage": "all", "success": True, "cost": 0.01, "n_iterations": 20},
            ],
            n_stages_completed=2,
            converged=True,
            total_iterations=30,
        )
        assert result.converged is True
        assert result.total_iterations == 30

    def test_convergence_trajectory(self) -> None:
        result = HierarchicalResult(
            best_params={},
            best_cost=0.01,
            stage_results=[
                {"stage": "s1", "success": True, "cost": 0.1, "n_iterations": 5},
                {"stage": "s2", "success": True, "cost": 0.05, "n_iterations": 5},
                {"stage": "s3", "success": True, "cost": 0.01, "n_iterations": 5},
            ],
            n_stages_completed=3,
            converged=True,
            total_iterations=15,
        )
        trajectory = result.convergence_trajectory
        assert trajectory == [0.1, 0.05, 0.01]

    def test_convergence_trajectory_with_none_cost(self) -> None:
        result = HierarchicalResult(
            best_params={},
            best_cost=0.01,
            stage_results=[
                {"stage": "s1", "success": True, "cost": 0.1, "n_iterations": 5},
                {"stage": "s2", "success": False, "cost": None, "n_iterations": 0},
            ],
            n_stages_completed=2,
            converged=False,
            total_iterations=5,
        )
        trajectory = result.convergence_trajectory
        assert trajectory[0] == 0.1
        assert np.isnan(trajectory[1])

    def test_frozen_dataclass(self) -> None:
        result = HierarchicalResult(
            best_params={},
            best_cost=0.01,
            stage_results=[],
            n_stages_completed=0,
            converged=False,
            total_iterations=0,
        )
        with pytest.raises(AttributeError):
            result.converged = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HierarchicalFitter
# ---------------------------------------------------------------------------


class TestHierarchicalFitter:
    """Tests for HierarchicalFitter.fit using mocks."""

    def _make_mock_model(self, vary_names: list[str] | None = None) -> MagicMock:
        """Create a mock HeterodyneModel with ParameterManager."""
        if vary_names is None:
            vary_names = list(ALL_PARAM_NAMES)

        model = MagicMock()
        pm = MagicMock()

        # Mock space.vary to return True for specified names
        vary_dict = {name: name in vary_names for name in ALL_PARAM_NAMES}
        pm.space.vary = vary_dict
        pm.n_varying = len(vary_names)
        pm.get_full_values.return_value = np.ones(len(ALL_PARAM_NAMES))
        pm.get_bounds.return_value = (
            np.zeros(len(vary_names)),
            np.ones(len(vary_names)) * 100,
        )
        pm.get_initial_values.return_value = np.ones(len(vary_names))
        pm.varying_names = vary_names
        pm.expand_varying_to_full.return_value = np.ones(len(ALL_PARAM_NAMES))

        model.param_manager = pm
        return model

    def _make_mock_adapter(self, success: bool = True) -> MagicMock:
        """Create a mock NLSQAdapterBase."""
        adapter = MagicMock()
        adapter.fit.return_value = NLSQResult(
            parameters=np.ones(3),
            parameter_names=["a", "b", "c"],
            success=success,
            message="converged" if success else "failed",
            final_cost=0.01 if success else 1e5,
            n_iterations=10,
        )
        return adapter

    def test_all_stages_succeed(self) -> None:
        """All stages converge, final result reflects last stage."""
        model = self._make_mock_model()
        adapter = self._make_mock_adapter(success=True)
        config = NLSQConfig()

        fitter = HierarchicalFitter(adapter)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        assert result.success is True
        assert "hierarchical_stages" in result.metadata
        # Adapter should be called once per stage with varying params
        assert adapter.fit.call_count >= 1

    def test_stage_failure_stops_by_default(self) -> None:
        """When a stage fails and skip_failed_stages=False, fitting stops."""
        model = self._make_mock_model()
        adapter = self._make_mock_adapter(success=False)
        config = NLSQConfig()

        hier_config = HierarchicalConfig(skip_failed_stages=False)
        fitter = HierarchicalFitter(adapter, hier_config)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        assert result.success is False
        # Should stop after first stage failure
        assert adapter.fit.call_count == 1

    def test_skip_failed_stages_continues(self) -> None:
        """When skip_failed_stages=True, fitting continues after failure."""
        model = self._make_mock_model()
        adapter = self._make_mock_adapter(success=False)
        config = NLSQConfig()

        hier_config = HierarchicalConfig(skip_failed_stages=True)
        fitter = HierarchicalFitter(adapter, hier_config)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        _result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        # Should attempt all stages
        assert adapter.fit.call_count == 4

    def test_restores_vary_flags(self) -> None:
        """Original vary flags should be restored after fit."""
        vary_names = ["D0_ref", "alpha_ref", "v0"]
        model = self._make_mock_model(vary_names=vary_names)
        adapter = self._make_mock_adapter(success=True)
        config = NLSQConfig()

        fitter = HierarchicalFitter(adapter)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        # Check that set_vary was called to restore original flags
        pm = model.param_manager
        restore_calls = pm.set_vary.call_args_list
        # The last N calls should restore original flags
        last_calls = restore_calls[-len(ALL_PARAM_NAMES) :]
        restored = {call.args[0]: call.args[1] for call in last_calls}
        for name in ALL_PARAM_NAMES:
            assert restored[name] == (name in vary_names)

    def test_per_stage_config_override(self) -> None:
        """Per-stage config should be used when available."""
        model = self._make_mock_model()
        adapter = self._make_mock_adapter(success=True)
        base_config = NLSQConfig(max_iterations=100)
        transport_config = NLSQConfig(max_iterations=500)

        hier_config = HierarchicalConfig(
            per_stage_config={"transport": transport_config}
        )
        fitter = HierarchicalFitter(adapter, hier_config)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=base_config,
            residual_fn=residual_fn,
        )

        # First call should use transport_config (passed as keyword arg)
        first_call = adapter.fit.call_args_list[0]
        first_call_config = first_call.kwargs.get("config")
        assert first_call_config is not None
        assert first_call_config.max_iterations == 500

    def test_no_varying_params_stage_skipped(self) -> None:
        """Stages with no varying parameters should be skipped."""
        # All params fixed
        model = self._make_mock_model(vary_names=[])
        adapter = self._make_mock_adapter(success=True)
        config = NLSQConfig()

        fitter = HierarchicalFitter(adapter)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        # No stages should have been executed
        assert adapter.fit.call_count == 0
        assert result.success is False
        assert "No stages" in result.message

    def test_metadata_includes_stage_summary(self) -> None:
        """Final result should include hierarchical metadata."""
        model = self._make_mock_model()
        adapter = self._make_mock_adapter(success=True)
        config = NLSQConfig()

        fitter = HierarchicalFitter(adapter)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=0.0,
            config=config,
            residual_fn=residual_fn,
        )

        assert "hierarchical_stages" in result.metadata
        assert "stage_results_summary" in result.metadata
        assert isinstance(result.metadata["stage_results_summary"], list)

    def test_uses_model_residual_when_no_residual_fn(self) -> None:
        """When residual_fn is None, should build one from the model."""
        model = self._make_mock_model()
        model.compute_residuals.return_value = np.zeros(10)
        adapter = self._make_mock_adapter(success=True)
        config = NLSQConfig()

        # Use single-stage config to simplify
        hier_config = HierarchicalConfig(
            stages=[{"name": "ref", "groups": ["reference"]}]
        )
        fitter = HierarchicalFitter(adapter, hier_config)

        _result = fitter.fit(
            model,
            np.ones((10, 10)),
            phi_angle=45.0,
            config=config,
            residual_fn=None,
        )

        assert adapter.fit.call_count == 1

    def test_restores_flags_on_exception(self) -> None:
        """Vary flags should be restored even if adapter.fit raises."""
        model = self._make_mock_model(vary_names=["D0_ref"])
        adapter = self._make_mock_adapter()
        adapter.fit.side_effect = RuntimeError("boom")
        config = NLSQConfig()

        fitter = HierarchicalFitter(adapter)
        residual_fn = lambda x: np.zeros(10)  # noqa: E731

        with pytest.raises(RuntimeError, match="boom"):
            fitter.fit(
                model,
                np.ones((10, 10)),
                phi_angle=0.0,
                config=config,
                residual_fn=residual_fn,
            )

        # Vary flags should still be restored
        pm = model.param_manager
        restore_calls = pm.set_vary.call_args_list
        # Last calls should be the restore
        last_calls = restore_calls[-len(ALL_PARAM_NAMES) :]
        assert len(last_calls) == len(ALL_PARAM_NAMES)
