"""Hierarchical multi-stage fitting strategy for heterodyne model.

Breaks the 14-parameter fit into progressive stages, each warm-starting
from the previous result:

  Stage 1: Transport params only (D0_ref, alpha_ref, D0_sample, alpha_sample)
  Stage 2: Add velocity params (v0, beta, v_offset)
  Stage 3: Add fraction params (f0, f1, f2, f3)
  Stage 4: Fit all varying params together
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES, PARAM_GROUPS
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

logger = get_logger(__name__)


# Default stage definitions using parameter group names
_DEFAULT_STAGES: list[dict[str, Any]] = [
    {
        "name": "transport",
        "groups": ["reference", "sample"],
        "description": "Fit transport parameters (D0, alpha) only",
    },
    {
        "name": "velocity",
        "groups": ["reference", "sample", "velocity"],
        "description": "Add velocity parameters (v0, beta, v_offset)",
    },
    {
        "name": "fraction",
        "groups": ["reference", "sample", "velocity", "fraction"],
        "description": "Add fraction parameters (f0, f1, f2, f3)",
    },
    {
        "name": "all",
        "groups": ["reference", "sample", "velocity", "fraction", "angle"],
        "description": "Fit all varying parameters together",
    },
]


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical multi-stage fitting.

    Attributes:
        stages: List of stage definitions. Each stage is a dict with keys:
            - name (str): Stage identifier
            - groups (list[str]): Parameter group names to vary in this stage
            - description (str, optional): Human-readable description
        per_stage_config: Optional per-stage NLSQConfig overrides keyed by stage name.
            If a stage name is not present, the base config is used.
        skip_failed_stages: If True, continue to next stage even if current stage fails.
    """

    stages: list[dict[str, Any]] = field(default_factory=lambda: list(_DEFAULT_STAGES))
    per_stage_config: dict[str, NLSQConfig] = field(default_factory=dict)
    skip_failed_stages: bool = False

    def __post_init__(self) -> None:
        """Validate stage definitions."""
        valid_groups = set(PARAM_GROUPS.keys())
        for i, stage in enumerate(self.stages):
            if "name" not in stage:
                raise ValueError(f"Stage {i} missing required 'name' key")
            if "groups" not in stage:
                raise ValueError(f"Stage {i} ('{stage['name']}') missing 'groups' key")
            for group in stage["groups"]:
                if group not in valid_groups:
                    raise ValueError(
                        f"Stage '{stage['name']}' references unknown group '{group}'. "
                        f"Valid groups: {sorted(valid_groups)}"
                    )


def _resolve_stage_params(
    stage: dict[str, Any],
    user_vary: dict[str, bool],
) -> set[str]:
    """Resolve which parameters should vary for a given stage.

    A parameter is varied in this stage only if:
    1. It belongs to one of the stage's groups, AND
    2. The user's original vary flags allow it to vary.

    Args:
        stage: Stage definition dict with 'groups' key.
        user_vary: Original user vary flags from ParameterManager.

    Returns:
        Set of parameter names to vary in this stage.
    """
    stage_params: set[str] = set()
    for group in stage["groups"]:
        for name in PARAM_GROUPS[group]:
            if user_vary.get(name, False):
                stage_params.add(name)
    return stage_params


class HierarchicalFitter:
    """Multi-stage hierarchical fitter for the heterodyne model.

    Progressively unfreezes parameter groups across stages, warm-starting
    each stage from the previous stage's optimum. This improves convergence
    in the full 14-parameter model by avoiding local minima.
    """

    def __init__(
        self,
        adapter: NLSQAdapterBase,
        hierarchical_config: HierarchicalConfig | None = None,
    ) -> None:
        """Initialize hierarchical fitter.

        Args:
            adapter: NLSQ adapter for running individual fits.
            hierarchical_config: Stage configuration, or None for defaults.
        """
        self._adapter = adapter
        self._config = hierarchical_config or HierarchicalConfig()

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run hierarchical multi-stage fitting.

        Args:
            model: HeterodyneModel instance with ParameterManager.
            c2_data: Observed correlation data.
            phi_angle: Scattering angle in degrees.
            config: Base NLSQConfig (may be overridden per-stage).
            residual_fn: Optional custom residual function. If None, uses
                model.compute_residuals via a default closure.
            jacobian_fn: Optional Jacobian function.

        Returns:
            NLSQResult from the final stage.
        """
        pm = model.param_manager
        stages = self._config.stages

        # Save user's original vary flags so we can restore them
        original_vary: dict[str, bool] = {
            name: pm.space.vary[name] for name in ALL_PARAM_NAMES
        }

        logger.info(
            "Starting hierarchical fit with %d stages (%d user-varying params)",
            len(stages),
            pm.n_varying,
        )

        stage_results: list[NLSQResult] = []
        current_values = pm.get_full_values().copy()

        try:
            for stage_idx, stage in enumerate(stages):
                stage_name = stage["name"]
                stage_desc = stage.get("description", stage_name)

                # Determine which params vary in this stage
                stage_vary_names = _resolve_stage_params(stage, original_vary)

                if not stage_vary_names:
                    logger.info(
                        "Stage %d/%d ('%s'): no varying params, skipping",
                        stage_idx + 1,
                        len(stages),
                        stage_name,
                    )
                    continue

                # Set vary flags for this stage
                for name in ALL_PARAM_NAMES:
                    pm.set_vary(name, name in stage_vary_names)

                # Warm-start from current best values
                pm.update_values(current_values)

                stage_config = self._config.per_stage_config.get(stage_name, config)

                logger.info(
                    "Stage %d/%d ('%s'): %s — varying %d params: %s",
                    stage_idx + 1,
                    len(stages),
                    stage_name,
                    stage_desc,
                    len(stage_vary_names),
                    sorted(stage_vary_names),
                )

                # Build bounds for varying params
                lower, upper = pm.get_bounds()
                bounds = (lower, upper)
                x0 = pm.get_initial_values()

                # Run optimization for this stage
                if residual_fn is not None:
                    result = self._adapter.fit(
                        residual_fn=residual_fn,
                        initial_params=x0,
                        bounds=bounds,
                        config=stage_config,
                        jacobian_fn=jacobian_fn,
                    )
                else:
                    # Build residual closure from model
                    def _make_residual(
                        _model: HeterodyneModel,
                        _c2: np.ndarray,
                        _phi: float,
                    ) -> Callable[[np.ndarray], np.ndarray]:
                        def _residual(params: np.ndarray) -> np.ndarray:
                            full = _model.param_manager.expand_varying_to_full(params)
                            return np.asarray(
                                _model.compute_residuals(
                                    _c2, phi_angle=_phi, params=full
                                )
                            )

                        return _residual

                    res_fn = _make_residual(model, c2_data, phi_angle)
                    result = self._adapter.fit(
                        residual_fn=res_fn,
                        initial_params=x0,
                        bounds=bounds,
                        config=stage_config,
                        jacobian_fn=jacobian_fn,
                    )

                stage_results.append(result)

                if result.success:
                    # Update current_values from fit result
                    fitted_full = pm.expand_varying_to_full(result.parameters)
                    current_values = fitted_full
                    logger.info(
                        "Stage '%s' converged: cost=%.4e, %d iterations",
                        stage_name,
                        result.final_cost
                        if result.final_cost is not None
                        else float("nan"),
                        result.n_iterations,
                    )
                else:
                    logger.warning(
                        "Stage '%s' failed: %s",
                        stage_name,
                        result.message,
                    )
                    if not self._config.skip_failed_stages:
                        logger.warning("Stopping hierarchical fit due to stage failure")
                        break

        finally:
            # Restore original vary flags
            for name, vary in original_vary.items():
                pm.set_vary(name, vary)

        # Return the last successful result, or the last result overall
        final_result = (
            stage_results[-1]
            if stage_results
            else NLSQResult(
                parameters=pm.get_initial_values(),
                parameter_names=pm.varying_names,
                success=False,
                message="No stages were executed",
            )
        )

        # Attach stage metadata
        final_result.metadata["hierarchical_stages"] = len(stages)
        final_result.metadata["stage_results_summary"] = [
            {
                "stage": stages[i]["name"],
                "success": r.success,
                "cost": r.final_cost,
                "n_iterations": r.n_iterations,
            }
            for i, r in enumerate(stage_results)
        ]

        logger.info(
            "Hierarchical fit complete: %d/%d stages successful",
            sum(1 for r in stage_results if r.success),
            len(stage_results),
        )

        return final_result


# ---------------------------------------------------------------------------
# HierarchicalResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalResult:
    """Structured result from a completed hierarchical multi-stage fit.

    Unlike the raw :class:`~heterodyne.optimization.nlsq.results.NLSQResult`
    returned by :meth:`HierarchicalFitter.fit`, this container retains the
    per-stage detail needed for convergence analysis and diagnostics.

    Attributes:
        best_params: Mapping of parameter name to fitted value at the end
            of the final completed stage.
        best_cost: Scalar cost (sum of squared residuals) of the best fit.
        stage_results: List of per-stage summary dicts.  Each dict contains
            at least the keys ``"stage"`` (str), ``"success"`` (bool),
            ``"cost"`` (float | None), and ``"n_iterations"`` (int).
        n_stages_completed: Number of stages that were actually executed
            (stages with no varying parameters are not counted).
        converged: ``True`` if the final stage reported
            ``success=True``.
        total_iterations: Sum of ``n_iterations`` across all executed stages.
    """

    best_params: dict[str, float]
    best_cost: float
    stage_results: list[dict[str, Any]]
    n_stages_completed: int
    converged: bool
    total_iterations: int

    @property
    def convergence_trajectory(self) -> list[float]:
        """Per-stage best cost, in stage execution order.

        Stages whose cost is ``None`` (e.g. failed stages when
        ``skip_failed_stages=True``) are represented as ``float('nan')``.

        Returns:
            List of scalar costs, one per entry in :attr:`stage_results`.
        """
        trajectory: list[float] = []
        for entry in self.stage_results:
            cost = entry.get("cost")
            trajectory.append(float(cost) if cost is not None else float("nan"))
        return trajectory
