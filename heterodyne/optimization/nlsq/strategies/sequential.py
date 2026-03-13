"""Sequential per-angle fitting strategy with warm-start propagation.

When the caller has correlation data at multiple detector phi angles, fitting
each angle independently and in sequence allows the optimised parameters from
angle *i* to seed the initial guess for angle *i+1*.  This warm-start
approach exploits continuity of physical parameters across adjacent angles
and generally requires far fewer iterations per angle than cold-start fits.

Architecture
------------
:class:`SequentialStrategy` is a meta-strategy: it dispatches each per-angle
fit to an *inner* strategy (``'residual'``, ``'jit'``, or ``'chunked'``).
The inner strategy handles all residual evaluation and Jacobian computation;
:class:`SequentialStrategy` only manages the warm-start state propagation and
multi-angle book-keeping.

Single-angle fitting
--------------------
``fit()`` delegates directly to the inner strategy.  It behaves identically
to calling the inner strategy directly and exists so that
:class:`SequentialStrategy` satisfies the :class:`FittingStrategy` protocol.

Multi-angle fitting
-------------------
``fit_multi_angle()`` accepts a ``(n_phi, N, N)`` data array and a sequence of
phi angles.  For each angle it:

1. Calls the inner strategy ``fit()``.
2. If the fit succeeded, updates ``model.param_manager`` initial values to the
   converged parameters (warm-start for the next angle).
3. Records per-angle metadata: phi value, index, warm-start origin, contrast
   and offset from the model's per-angle scaling (if configured).

Result combination
------------------
Each angle returns its own :class:`StrategyResult`.  The list is returned
directly; callers that need a single combined result can use
:func:`combine_angle_results`.

Fixed-parameter handling
------------------------
The TRF solver inside scipy requires strict ``lower < upper`` for every
parameter.  :func:`strip_fixed_parameters` removes equality-constrained
parameters before the call and :func:`restore_fixed_parameters` re-inserts
them afterwards.  The inner strategies handle this automatically via the
model's ``ParameterManager``.

Partial convergence
-------------------
Multi-angle fitting proceeds even when individual angles fail.  Failed angles
are logged at WARNING level.  The ``success_rate`` key in the aggregate
metadata reports the fraction of successfully converged angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.strategies.base import StrategyResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AngleSubset:
    """Data subset for a single phi angle.

    Attributes:
        phi_angle: The scalar phi angle value (degrees).
        angle_index: Position of this angle in the multi-angle stack.
        n_points: Number of data points (``N × N``).
        c2_data: Correlation matrix for this angle, shape ``(N, N)``.
        weights: Optional weight matrix, same shape as ``c2_data``.
    """

    phi_angle: float
    angle_index: int
    n_points: int
    c2_data: np.ndarray
    weights: np.ndarray | None


@dataclass
class MultiAngleResult:
    """Aggregate result from multi-angle sequential fitting.

    Attributes:
        per_angle_results: One :class:`StrategyResult` per angle.
        n_angles_total: Total number of angles attempted.
        n_angles_success: Number of angles that converged.
        n_angles_failed: Number of angles that failed.
        success_rate: ``n_angles_success / n_angles_total``.
        phi_angles: Array of phi angles, shape ``(n_angles_total,)``.
    """

    per_angle_results: list[StrategyResult]
    n_angles_total: int
    n_angles_success: int
    n_angles_failed: int
    success_rate: float
    phi_angles: np.ndarray


# ---------------------------------------------------------------------------
# Fixed-parameter helpers
# ---------------------------------------------------------------------------


def strip_fixed_parameters(
    initial_params: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove equality-constrained parameters from optimizer inputs.

    TRF requires strict ``lower < upper`` for every parameter.  Parameters
    where ``lower == upper`` (fixed) must be excluded before the call.

    Args:
        initial_params: Full parameter vector.
        lower_bounds: Lower bound array, same length.
        upper_bounds: Upper bound array, same length.

    Returns:
        ``(free_params, free_lower, free_upper, free_mask)`` where
        ``free_mask`` is a boolean array (``True`` for free parameters).

    Example::

        p = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 2.0, 0.0])
        hi = np.array([5.0, 2.0, 5.0])
        free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
        # free = [1.0, 3.0], mask = [True, False, True]
    """
    free_mask = lower_bounds < upper_bounds
    return (
        initial_params[free_mask],
        lower_bounds[free_mask],
        upper_bounds[free_mask],
        free_mask,
    )


def restore_fixed_parameters(
    free_result: np.ndarray,
    fixed_values: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    """Re-insert fixed parameter values into the optimised result.

    Inverse of :func:`strip_fixed_parameters`.

    Args:
        free_result: Optimised values for the free parameters.
        fixed_values: Full reference parameter vector; fixed positions are
            read from here.
        free_mask: Boolean mask returned by :func:`strip_fixed_parameters`.

    Returns:
        Full parameter vector with fixed values restored.
    """
    result = np.array(fixed_values, dtype=np.float64)
    result[free_mask] = free_result
    return result


# ---------------------------------------------------------------------------
# Result combination
# ---------------------------------------------------------------------------


def combine_angle_results(
    per_angle_results: list[StrategyResult],
    weighting: str = "inverse_variance",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine per-angle optimisation results into a single parameter estimate.

    Statistical combination methods:

    ``'inverse_variance'`` (default):
        Optimal weighting when per-angle errors are independent.
        ``w_i = 1 / mean(diag(Cov_i))``

    ``'n_points'``:
        Weight proportional to the number of data points per angle.

    ``'uniform'``:
        Equal weights for all angles.

    Args:
        per_angle_results: List of :class:`StrategyResult`, one per angle.
        weighting: Combination scheme — ``'inverse_variance'``, ``'n_points'``,
            or ``'uniform'``.

    Returns:
        ``(combined_params, combined_cov, total_cost)`` where
        ``combined_params`` is the weighted average parameter vector,
        ``combined_cov`` is the combined covariance matrix, and
        ``total_cost`` is the sum of per-angle costs.

    Raises:
        ValueError: If no angles converged or ``weighting`` is unknown.
    """
    successful = [sr for sr in per_angle_results if sr.result.success]

    if not successful:
        raise ValueError(
            "combine_angle_results: no angles converged — cannot combine results"
        )

    logger.info(
        "Combining %d/%d successful angle results (weighting='%s')",
        len(successful),
        len(per_angle_results),
        weighting,
    )

    params_list = np.array([sr.result.parameters for sr in successful])

    # Build covariance list (fall back to identity when not available)
    n_params = params_list.shape[1]
    cov_list = []
    for sr in successful:
        if sr.result.covariance is not None:
            cov_list.append(sr.result.covariance)
        else:
            cov_list.append(np.eye(n_params))
    cov_arr = np.array(cov_list)

    # Compute raw weights
    if weighting == "inverse_variance":
        weights_raw = np.array(
            [1.0 / (np.mean(np.diag(cov)) + 1e-10) for cov in cov_arr]
        )
    elif weighting == "n_points":
        weights_raw = np.array(
            [
                float(sr.result.residuals.size)
                if sr.result.residuals is not None
                else 1.0
                for sr in successful
            ],
            dtype=np.float64,
        )
    elif weighting == "uniform":
        weights_raw = np.ones(len(successful), dtype=np.float64)
    else:
        raise ValueError(
            f"combine_angle_results: unknown weighting={weighting!r}; "
            f"must be 'inverse_variance', 'n_points', or 'uniform'"
        )

    # Normalise
    weights = weights_raw / (weights_raw.sum() + 1e-10)

    # Weighted parameter average
    combined_params = np.sum(params_list * weights[:, np.newaxis], axis=0)

    # Combined covariance
    if weighting == "inverse_variance":
        inv_vars = np.array([1.0 / (np.diag(cov) + 1e-10) for cov in cov_arr])
        combined_var = 1.0 / inv_vars.sum(axis=0)
        combined_cov = np.diag(combined_var)
    else:
        combined_cov = np.sum(cov_arr * weights[:, np.newaxis, np.newaxis], axis=0)

    total_cost = sum(
        float(sr.result.final_cost) if sr.result.final_cost is not None else 0.0
        for sr in successful
    )

    logger.debug("Combined parameters (first 4): %s ...", combined_params[:4])
    logger.info("Total combined cost: %.4e", total_cost)

    return combined_params, combined_cov, total_cost


# ---------------------------------------------------------------------------
# SequentialStrategy
# ---------------------------------------------------------------------------


class SequentialStrategy:
    """Sequential per-angle fitting with warm-start propagation.

    Fits each scattering angle sequentially using the fitted parameters from
    the previous angle as the initial guess for the next.  This exploits
    parameter continuity across adjacent angles.

    The inner strategy handles residual evaluation; :class:`SequentialStrategy`
    manages warm-start propagation, per-angle metadata, and optional per-angle
    scaling extraction.

    Args:
        inner_strategy_name: Which inner strategy to use for each angle fit.
            One of ``'residual'``, ``'jit'`` (default), ``'chunked'``.
        min_success_rate: Minimum fraction of angles that must converge for
            :meth:`fit_multi_angle` to proceed without raising.  Default 0.0
            (never raise).
        clamp_to_bounds: Whether to clamp warm-started initial values to the
            parameter bounds before each angle fit.  Default ``True``.

    Example::

        strategy = SequentialStrategy(inner_strategy_name="jit")

        # Single angle
        sr = strategy.fit(model, c2_data, phi_angle=0.0, config=config)

        # Multiple angles
        results = strategy.fit_multi_angle(
            model, c2_stack, phi_angles=[0.0, 45.0, 90.0], config=config
        )
    """

    def __init__(
        self,
        inner_strategy_name: str = "jit",
        min_success_rate: float = 0.0,
        clamp_to_bounds: bool = True,
    ) -> None:
        self._inner_name = inner_strategy_name
        self._min_success_rate = min_success_rate
        self._clamp_to_bounds = clamp_to_bounds

    @property
    def name(self) -> str:
        return "sequential"

    # ------------------------------------------------------------------
    # FittingStrategy.fit() — single angle, delegates to inner
    # ------------------------------------------------------------------

    def fit(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angle: float,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> StrategyResult:
        """Fit a single angle by delegating to the inner strategy.

        This method satisfies the :class:`FittingStrategy` protocol.  For
        multi-angle fitting with warm-start, use :meth:`fit_multi_angle`.

        Args:
            model: Configured heterodyne model.
            c2_data: Correlation matrix, shape ``(N, N)``.
            phi_angle: Detector phi angle in degrees.
            config: NLSQ configuration.
            weights: Optional per-point weights.

        Returns:
            :class:`StrategyResult` from the inner strategy.
        """
        inner = self._get_inner_strategy()
        logger.debug(
            "SequentialStrategy.fit: delegating to %s (phi=%.2f°)",
            inner.name,
            phi_angle,
        )
        return inner.fit(model, c2_data, phi_angle, config, weights)

    # ------------------------------------------------------------------
    # Multi-angle fitting
    # ------------------------------------------------------------------

    def fit_multi_angle(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angles: list[float] | np.ndarray,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
        weighting: str = "inverse_variance",
    ) -> MultiAngleResult:
        """Fit multiple phi angles sequentially with warm-start propagation.

        For each angle:
        1. The inner strategy is called with the current model initial values.
        2. On convergence the model's ``ParameterManager`` initial values are
           updated to the converged parameters (warm-start for the next angle).
        3. The converged parameters are clamped to bounds (if
           ``clamp_to_bounds=True``) before being used as the next seed.

        Args:
            model: Configured heterodyne model.
            c2_data: Correlation data.  Must be shape ``(n_phi, N, N)`` for
                multiple angles or ``(N, N)`` for a single angle (will be
                promoted to ``(1, N, N)``).
            phi_angles: Sequence of phi angles in degrees, length ``n_phi``.
            config: NLSQ configuration.
            weights: Optional weights.  Shape must match ``c2_data``, i.e.
                ``(n_phi, N, N)`` for per-angle weights or ``(N, N)`` for
                shared weights.
            weighting: Combination scheme forwarded to
                :func:`combine_angle_results` for the aggregate summary.

        Returns:
            :class:`MultiAngleResult` with one :class:`StrategyResult` per
            angle plus aggregate statistics.

        Raises:
            ValueError: If ``len(c2_data) != len(phi_angles)``.
            RuntimeError: If ``success_rate < min_success_rate``.
        """
        phi_angles = np.asarray(phi_angles, dtype=np.float64)
        n_phi = len(phi_angles)

        # Promote (N, N) → (1, N, N)
        c2_data = np.asarray(c2_data)
        if c2_data.ndim == 2:
            c2_data = c2_data[np.newaxis, ...]

        if len(c2_data) != n_phi:
            raise ValueError(
                f"SequentialStrategy.fit_multi_angle: c2_data has {len(c2_data)} "
                f"slices but {n_phi} phi angles were supplied"
            )

        inner = self._get_inner_strategy()
        logger.info(
            "SequentialStrategy.fit_multi_angle: %d angles, inner='%s'",
            n_phi,
            inner.name,
        )

        # Build subsets
        subsets = self._build_subsets(c2_data, phi_angles, weights)

        # Iterate
        results: list[StrategyResult] = []
        n_success = 0

        for subset in subsets:
            i = subset.angle_index
            phi = subset.phi_angle
            warm_start_info = (
                f" (warm-start from φ={phi_angles[i - 1]:.2f}°)" if i > 0 else ""
            )
            logger.info(
                "SequentialStrategy: angle %d/%d: φ=%.2f°%s",
                i + 1,
                n_phi,
                phi,
                warm_start_info,
            )

            sr = self._fit_single_subset(
                inner=inner,
                model=model,
                subset=subset,
                config=config,
            )

            # Warm-start: propagate converged params to next angle
            if sr.result.success:
                n_success += 1
                self._propagate_warm_start(model, sr.result.parameters)
            else:
                logger.warning(
                    "SequentialStrategy: angle %d (φ=%.2f°) did not converge: %s",
                    i + 1,
                    phi,
                    sr.result.message,
                )

            # Annotate metadata
            sr.metadata["phi_angle"] = phi
            sr.metadata["angle_index"] = i
            sr.metadata["warm_started"] = i > 0

            # Attach per-angle scaling if available
            self._annotate_scaling(model, i, sr)

            results.append(sr)

        # Aggregate statistics
        n_failed = n_phi - n_success
        success_rate = n_success / n_phi if n_phi > 0 else 0.0

        logger.info(
            "SequentialStrategy.fit_multi_angle complete: %d/%d converged (%.0f%%)",
            n_success,
            n_phi,
            success_rate * 100,
        )

        if success_rate < self._min_success_rate:
            raise RuntimeError(
                f"SequentialStrategy: insufficient convergence — "
                f"{n_success}/{n_phi} angles converged ({success_rate:.1%}), "
                f"minimum required: {self._min_success_rate:.1%}"
            )

        return MultiAngleResult(
            per_angle_results=results,
            n_angles_total=n_phi,
            n_angles_success=n_success,
            n_angles_failed=n_failed,
            success_rate=success_rate,
            phi_angles=phi_angles,
        )

    # ------------------------------------------------------------------
    # Convenience: multi-angle → list[StrategyResult] (legacy API)
    # ------------------------------------------------------------------

    def fit_multi_angle_list(
        self,
        model: HeterodyneModel,
        c2_data: np.ndarray,
        phi_angles: list[float] | np.ndarray,
        config: NLSQConfig,
        weights: np.ndarray | None = None,
    ) -> list[StrategyResult]:
        """Multi-angle fit returning a flat list of :class:`StrategyResult`.

        Convenience wrapper around :meth:`fit_multi_angle` for callers that
        only need the per-angle results without the aggregate statistics.

        Args:
            model: Configured heterodyne model.
            c2_data: Correlation data, shape ``(n_phi, N, N)`` or ``(N, N)``.
            phi_angles: Phi angles in degrees.
            config: NLSQ configuration.
            weights: Optional weights.

        Returns:
            List of :class:`StrategyResult`, one per angle.
        """
        mar = self.fit_multi_angle(model, c2_data, phi_angles, config, weights)
        return mar.per_angle_results

    # ------------------------------------------------------------------
    # Inner strategy factory
    # ------------------------------------------------------------------

    def _get_inner_strategy(self) -> Any:
        """Instantiate and return the configured inner strategy.

        Returns:
            Strategy instance implementing the :class:`FittingStrategy`
            protocol.
        """
        from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy
        from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        registry: dict[str, type] = {
            "residual": ResidualStrategy,
            "jit": JITStrategy,
            "chunked": ChunkedStrategy,
        }
        cls = registry.get(self._inner_name)
        if cls is None:
            logger.warning(
                "SequentialStrategy: unknown inner_strategy_name=%r; "
                "falling back to JITStrategy",
                self._inner_name,
            )
            cls = JITStrategy
        return cls()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subsets(
        c2_data: np.ndarray,
        phi_angles: np.ndarray,
        weights: np.ndarray | None,
    ) -> list[AngleSubset]:
        """Build :class:`AngleSubset` objects for each phi angle.

        Args:
            c2_data: Shape ``(n_phi, N, N)``.
            phi_angles: Shape ``(n_phi,)``.
            weights: Shape ``(n_phi, N, N)`` or ``(N, N)`` or ``None``.

        Returns:
            List of :class:`AngleSubset`, one per angle.
        """
        subsets: list[AngleSubset] = []
        for i, phi in enumerate(phi_angles):
            c2_i = c2_data[i]
            if weights is None:
                w_i = None
            elif weights.ndim == 3:
                w_i = weights[i]
            else:
                w_i = weights  # shared weights

            subsets.append(
                AngleSubset(
                    phi_angle=float(phi),
                    angle_index=i,
                    n_points=int(c2_i.size),
                    c2_data=c2_i,
                    weights=w_i,
                )
            )
        return subsets

    @staticmethod
    def _fit_single_subset(
        inner: Any,
        model: HeterodyneModel,
        subset: AngleSubset,
        config: NLSQConfig,
    ) -> StrategyResult:
        """Call the inner strategy for a single angle subset.

        Wraps the call in a try/except so that a single angle failure does
        not abort the entire multi-angle run.

        Args:
            inner: Inner strategy instance.
            model: Heterodyne model.
            subset: Data for this angle.
            config: NLSQ configuration.

        Returns:
            :class:`StrategyResult`; on exception returns a failed result.
        """
        try:
            sr = inner.fit(
                model,
                subset.c2_data,
                subset.phi_angle,
                config,
                subset.weights,
            )
            if sr.result.jacobian is not None:
                jac_norm = float(np.linalg.norm(sr.result.jacobian))
                sr.metadata["jacobian_norm"] = jac_norm
                logger.debug(
                    "Jacobian norm for angle_index=%d (φ=%.2f°): %.6g",
                    subset.angle_index,
                    subset.phi_angle,
                    jac_norm,
                )
            return sr
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SequentialStrategy: inner strategy raised for "
                "angle_index=%d (φ=%.2f°): %s",
                subset.angle_index,
                subset.phi_angle,
                exc,
            )
            # Return a failed StrategyResult with minimal content
            from heterodyne.optimization.nlsq.results import NLSQResult

            pm = model.param_manager
            failed_result = NLSQResult(
                parameters=np.asarray(pm.get_initial_values(), dtype=np.float64),
                parameter_names=pm.varying_names,
                success=False,
                message=f"Exception: {exc}",
                convergence_reason=f"Exception: {exc}",
            )
            return StrategyResult(
                result=failed_result,
                strategy_name=inner.name,
                metadata={"exception": str(exc)},
            )

    def _propagate_warm_start(
        self,
        model: HeterodyneModel,
        converged_params: np.ndarray,
    ) -> None:
        """Update the model's initial values for the next angle fit.

        Args:
            model: Heterodyne model whose ``ParameterManager`` will be updated.
            converged_params: Converged varying-parameter vector from the
                previous angle fit.
        """
        pm = model.param_manager
        lower, upper = pm.get_bounds()
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        new_initial = np.asarray(converged_params, dtype=np.float64)
        if self._clamp_to_bounds:
            new_initial = np.clip(new_initial, lower, upper)

        pm.set_initial_values(new_initial)
        logger.debug(
            "SequentialStrategy: warm-start updated (first 4 values: %s ...)",
            new_initial[:4],
        )

    @staticmethod
    def _annotate_scaling(
        model: HeterodyneModel,
        angle_index: int,
        sr: StrategyResult,
    ) -> None:
        """Attach per-angle contrast and offset to the result metadata.

        Reads from ``model.scaling`` when ``n_angles > 1``.  Silently skips
        when the scaling attribute is absent or single-angle.

        Args:
            model: Heterodyne model.
            angle_index: Zero-based angle index.
            sr: Result whose ``metadata`` dict will be updated in-place.
        """
        scaling = getattr(model, "scaling", None)
        if scaling is None:
            return
        n_angles = getattr(scaling, "n_angles", 1)
        if n_angles <= 1:
            return
        try:
            safe_idx = min(angle_index, n_angles - 1)
            contrast, offset = scaling.get_for_angle(safe_idx)
            sr.metadata["contrast"] = float(contrast)
            sr.metadata["offset"] = float(offset)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "SequentialStrategy: could not read scaling for angle %d: %s",
                angle_index,
                exc,
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SequentialStrategy(inner='{self._inner_name}', "
            f"min_success_rate={self._min_success_rate})"
        )
