"""NLSQWrapper — stable fallback adapter for heterodyne XPCS optimization.

Role and When to Use
--------------------

**NLSQWrapper** (this module) is the **stable fallback adapter** for:

- Complex optimizations requiring the full 14-parameter heterodyne model
- Large datasets (> 100M points) requiring streaming/chunking strategies
- Advanced recovery mechanisms for convergence failures
- Production stability where reliability is critical

Use **NLSQAdapter** instead for:

- Standard optimizations with small to medium datasets (< 10M points)
- Multi-start optimization (model caching provides 3-5× speedup)
- Performance-critical workflows requiring JIT compilation

**Key Differences:**

* Model caching: NLSQWrapper=None, NLSQAdapter=Built-in
* JIT compilation: NLSQWrapper=Manual, NLSQAdapter=Auto
* Workflow auto-select: NLSQWrapper=Custom, NLSQAdapter=Via NLSQ
* Recovery system: NLSQWrapper=3-attempt, NLSQAdapter=NLSQ native
* Streaming support: NLSQWrapper=Full custom, NLSQAdapter=Via NLSQ

**Decision Guide:**

1. If you need robust streaming for 100M+ points: Use NLSQWrapper
2. If you need full anti-degeneracy control: Use NLSQWrapper
3. If you need maximum speed for multi-start optimization: Use NLSQAdapter
4. Default recommendation: NLSQAdapter with automatic fallback to NLSQWrapper

This module provides a high-level adapter between heterodyne's optimization API
and the NLSQ package's trust-region nonlinear least squares interface.

The NLSQWrapper class implements the Adapter pattern to translate:
- Heterodyne's multi-dimensional XPCS data → NLSQ's flattened array format
- Heterodyne's parameter bounds tuple → NLSQ's (lower, upper) format
- NLSQ's result → Heterodyne's NLSQResult dataclass

Key Features:
- Automatic dataset size detection and memory-based strategy selection
- Intelligent error recovery with 3-attempt retry strategy
- Actionable error diagnostics with multiple error categories
- CPU-optimized execution through JAX
- Progress logging and convergence diagnostics
- Serves as fallback when NLSQAdapter fails

Ported from homodyne's NLSQWrapper, adapted for heterodyne's 14-parameter
model. Shear-related parameters (gamma_dot_*, shear_transforms) have been
removed — they are a D3 divergence (Layer 5 not applicable to heterodyne's
velocity phase model).

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Heterodyne 14-parameter model: see heterodyne/core/physics.py
- Phase 4 PR 1 Task 1.5 — closes audit row optimization.nlsq.wrapper
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from heterodyne.optimization.nlsq.adapter import (
    NLSQWrapper as _NLSQWrapperLowLevel,
)
from heterodyne.optimization.nlsq.memory import select_nlsq_strategy
from heterodyne.optimization.nlsq.recovery import (
    execute_with_recovery,
    safe_uncertainties_from_pcov,
)
from heterodyne.optimization.nlsq.result_builder import build_failed_result
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heterodyne's 14 physics parameters + 2 scaling parameters (16 total).
# These replace homodyne's mode-dependent lists (D0/alpha/D_offset for
# static_isotropic; D0/alpha/D_offset/gamma_dot_*/beta/phi0 for laminar_flow).
# Heterodyne always operates in "full" mode with all 14 physics params.
# ---------------------------------------------------------------------------

#: Physical parameter names — the 14 heterodyne physics parameters.
HETERODYNE_PHYSICAL_PARAM_NAMES: list[str] = [
    "D0_ref",
    "alpha_ref",
    "D_offset_ref",
    "D0_sample",
    "alpha_sample",
    "D_offset_sample",
    "v0",
    "beta",
    "v_offset",
    "f0",
    "f1",
    "f2",
    "f3",
    "phi0",
]

#: Scaling parameter names (contrast and offset, one pair per angle).
HETERODYNE_SCALING_PARAM_NAMES: list[str] = ["contrast", "offset"]

#: All 16 default parameter names (physical + scaling).
HETERODYNE_ALL_PARAM_NAMES: list[str] = (
    HETERODYNE_PHYSICAL_PARAM_NAMES + HETERODYNE_SCALING_PARAM_NAMES
)


# ---------------------------------------------------------------------------
# Module-level helpers (ported from homodyne wrapper.py)
# ---------------------------------------------------------------------------


def _extract_n_points(data: Any) -> int:
    """Extract number of data points from various data formats.

    Handles XPCSData objects, numpy arrays, lists, and other iterables.

    Args:
        data: Data object with ``g2`` attribute or array-like.

    Returns:
        Number of data points (0 if cannot determine).
    """
    # Try g2 attribute (XPCSData)
    if hasattr(data, "g2"):
        g2 = data.g2
        if hasattr(g2, "size"):
            return int(g2.size)
        if hasattr(g2, "__len__"):
            return len(g2)
    # Try direct array-like
    if hasattr(data, "size"):
        return int(data.size)
    if hasattr(data, "__len__"):
        return len(data)
    return 0


def _extract_nlsq_settings(config: Any) -> dict[str, Any]:
    """Return NLSQ-specific settings from the config tree (if present)."""
    config_dict: dict[str, Any] | None = None

    if hasattr(config, "config") and isinstance(config.config, dict):
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return {}

    nlsq_settings = config_dict.get("optimization", {}).get("nlsq", {})
    return cast(dict[str, Any], nlsq_settings)


# ---------------------------------------------------------------------------
# NLSQWrapper — high-level stable fallback adapter
# ---------------------------------------------------------------------------


class NLSQWrapper:
    """High-level stable fallback adapter for heterodyne NLSQ optimization.

    Wraps the low-level ``_NLSQWrapperLowLevel`` (from ``adapter.py``) with a
    high-level ``fit(data, config)`` API that mirrors homodyne's NLSQWrapper.

    Strategy routing:
        Memory-based strategy selection (STANDARD → LARGE → STREAMING) is
        handled by the inner ``_NLSQWrapperLowLevel``.

    Recovery:
        3-attempt recovery loop: perturbation → tolerance relaxation →
        method switching.  Delegates to ``execute_with_recovery()`` from
        ``recovery.py``.

    Differences from homodyne's NLSQWrapper:
        - ``analysis_mode`` removed — heterodyne always uses all 14 physics
          parameters (no mode switching between static_isotropic/laminar_flow).
        - ``shear_transforms`` removed — D3 divergence, not applicable to
          heterodyne's velocity phase model.
        - Return type is ``NLSQResult`` (not homodyne's ``OptimizationResult``).

    Usage::

        wrapper = NLSQWrapper()
        result = wrapper.fit(data, config)
        # Or with explicit overrides:
        result = wrapper.fit(data, config, initial_params=p0, bounds=bounds)
    """

    def __init__(
        self,
        enable_large_dataset: bool = True,
        enable_recovery: bool = True,
        max_retries: int = 3,
        fast_mode: bool = False,
    ) -> None:
        """Initialise NLSQWrapper.

        Args:
            enable_large_dataset: Allow LARGE tier when memory warrants it.
            enable_recovery: Enable 3-attempt cross-tier fallback on failure.
            max_retries: Maximum per-tier retries before falling back
                (default 3, matching homodyne's 3-attempt recovery spec).
            fast_mode: Disable non-essential checks for lower overhead.
        """
        self.enable_large_dataset = enable_large_dataset
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        self.fast_mode = fast_mode

        # Inner low-level wrapper (handles tier routing: STANDARD/LARGE/STREAMING)
        self._inner: _NLSQWrapperLowLevel | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Adapter name."""
        return "nlsq.NLSQWrapper"

    def supports_bounds(self) -> bool:
        """Return True — bounds are always supported."""
        return True

    def supports_jacobian(self) -> bool:
        """Return True — analytic Jacobians are forwarded if provided."""
        return True

    def fit(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        per_angle_scaling: bool = True,
        diagnostics_enabled: bool = False,
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
    ) -> NLSQResult:
        """Execute NLSQ optimisation with memory-based strategy routing.

        This is the high-level entry point that accepts heterodyne XPCS data
        and a configuration manager (mirrors homodyne's NLSQWrapper.fit API).

        The method:
        1. Selects an NLSQ strategy based on estimated dataset size and RAM.
        2. Builds a residual function from ``data``.
        3. Runs 3-attempt recovery (perturbation → tolerance relaxation →
           method switching) before falling back to a lower tier.

        Args:
            data: XPCS experimental data (``XPCSData`` or array-like with
                a ``g2`` attribute).
            config: Configuration manager (``ConfigManager``) or plain dict.
                NLSQ settings are read from
                ``config.config["optimization"]["nlsq"]``.
            initial_params: Initial parameter guess.  Required when
                ``bounds`` is provided; auto-loaded from registry defaults
                if ``None``.
            bounds: Parameter bounds as ``(lower, upper)`` tuple.
            per_angle_scaling: Must be True.  Per-angle contrast/offset
                parameters are physically required (each detector angle has
                distinct optical properties).
            diagnostics_enabled: Enable extended diagnostics logging.
            per_angle_scaling_initial: Optional per-angle initial values for
                contrast and offset, keyed by parameter name.

        Returns:
            NLSQResult with fitted parameters and diagnostics.

        Raises:
            ValueError: If ``per_angle_scaling=False`` (unsupported) or if
                the residual probe fails.
        """
        start_time = time.perf_counter()

        if not per_angle_scaling:
            raise ValueError(
                "per_angle_scaling=False is not supported in heterodyne. "
                "Per-angle contrast/offset is physically required."
            )

        fit_logger = get_logger(__name__)

        # ---- Extract NLSQ settings from config ----
        nlsq_settings = _extract_nlsq_settings(config)
        loss_name = nlsq_settings.get("loss", "soft_l1")
        x_scale_override = nlsq_settings.get("x_scale")
        x_scale_value = x_scale_override if x_scale_override is not None else "jac"
        diagnostics_cfg = nlsq_settings.get("diagnostics", {})
        diagnostics_enabled = diagnostics_enabled or bool(
            diagnostics_cfg.get("enable", False)
        )

        # ---- Estimate data size for strategy selection ----
        n_est_points = _extract_n_points(data)
        n_params = (
            len(initial_params)
            if initial_params is not None
            else len(HETERODYNE_ALL_PARAM_NAMES)
        )

        strategy_decision = select_nlsq_strategy(n_est_points, n_params)
        fit_logger.info(
            "NLSQWrapper strategy: %s (%s)",
            strategy_decision.strategy.value,
            strategy_decision.reason,
        )

        # ---- Build NLSQConfig from config dict ----
        nlsq_config = self._build_nlsq_config(config, nlsq_settings)

        # ---- Resolve parameter names ----
        parameter_names = self._resolve_parameter_names(
            data=data,
            initial_params=initial_params,
        )

        # ---- Resolve initial params and bounds ----
        resolved_params, resolved_bounds = self._resolve_params_and_bounds(
            data=data,
            initial_params=initial_params,
            bounds=bounds,
            parameter_names=parameter_names,
        )

        # ---- Build residual function from data ----
        residual_fn = self._build_residual_fn(
            data=data,
            config=config,
            parameter_names=parameter_names,
        )

        if residual_fn is None:
            wall_time = time.perf_counter() - start_time
            return build_failed_result(
                parameter_names=parameter_names,
                message="Could not build residual function from data",
                initial_params=resolved_params,
                wall_time=wall_time,
            )

        fit_logger.info(
            "NLSQWrapper.fit: %d parameters, ~%d data points, "
            "loss=%s x_scale=%r diagnostics=%s",
            len(resolved_params),
            n_est_points,
            loss_name,
            x_scale_value,
            diagnostics_enabled,
        )

        # ---- Inner low-level wrapper (handles STANDARD/LARGE/STREAMING tiers) ----
        inner = _NLSQWrapperLowLevel(
            parameter_names=parameter_names,
            enable_large_dataset=self.enable_large_dataset,
            enable_recovery=self.enable_recovery,
            max_retries=self.max_retries,
        )

        # ---- 3-attempt recovery loop ----
        if self.enable_recovery:
            result = execute_with_recovery(
                fit_fn=lambda params, bds, cfg: inner.fit(
                    residual_fn=residual_fn,
                    initial_params=params,
                    bounds=bds,
                    config=cfg,
                ),
                initial_params=resolved_params,
                bounds=resolved_bounds,
                config=nlsq_config,
                max_retries=self.max_retries,
            )
        else:
            result = inner.fit(
                residual_fn=residual_fn,
                initial_params=resolved_params,
                bounds=resolved_bounds,
                config=nlsq_config,
            )

        wall_time = time.perf_counter() - start_time
        fit_logger.info(
            "NLSQWrapper.fit: %s in %.2fs (success=%s)",
            result.message,
            wall_time,
            result.success,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_nlsq_config(
        self, config: Any, nlsq_settings: dict[str, Any]
    ) -> NLSQConfig:
        """Build an NLSQConfig from a ConfigManager or dict.

        Reads solver settings from the NLSQ config sub-dict.  Falls back to
        NLSQConfig defaults when keys are absent.
        """
        from heterodyne.optimization.nlsq.config import NLSQConfig

        # If config is already an NLSQConfig, return it directly
        if isinstance(config, NLSQConfig):
            return config


        _loss_raw = str(nlsq_settings.get("loss", "soft_l1"))
        # Validate against allowed literals; fall back to "soft_l1" for unknown values.
        _allowed_losses = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
        if _loss_raw not in _allowed_losses:
            logger.warning(
                "NLSQWrapper: unknown loss %r; falling back to 'soft_l1'", _loss_raw
            )
            _loss_raw = "soft_l1"
        _loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = cast(
            "Literal['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']", _loss_raw
        )

        return NLSQConfig(
            ftol=float(nlsq_settings.get("ftol", 1e-8)),
            xtol=float(nlsq_settings.get("xtol", 1e-8)),
            gtol=float(nlsq_settings.get("gtol", 1e-8)),
            loss=_loss,
            x_scale=nlsq_settings.get("x_scale", "jac"),
            max_nfev=nlsq_settings.get("max_nfev"),
        )

    def _resolve_parameter_names(
        self,
        data: Any,
        initial_params: np.ndarray | None,
    ) -> list[str]:
        """Resolve the list of parameter names for this fit.

        Falls back to the full 16-name list (14 physics + 2 scaling).
        If ``data`` exposes a ``parameter_names`` attribute, that takes
        precedence.
        """
        if hasattr(data, "parameter_names") and data.parameter_names:
            names = list(data.parameter_names)
            logger.debug(
                "NLSQWrapper: using parameter_names from data (%d)", len(names)
            )
            return names

        if initial_params is not None:
            n = len(initial_params)
            if n <= len(HETERODYNE_ALL_PARAM_NAMES):
                return HETERODYNE_ALL_PARAM_NAMES[:n]
            # More params than the default list — generate generic names
            return [f"p{i}" for i in range(n)]

        return list(HETERODYNE_ALL_PARAM_NAMES)

    def _resolve_params_and_bounds(
        self,
        data: Any,
        initial_params: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        parameter_names: list[str],
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Resolve initial parameters and bounds.

        Priority order for initial params:
        1. Explicit ``initial_params`` argument
        2. ``data.initial_params`` attribute
        3. Zeros (last resort)

        Priority order for bounds:
        1. Explicit ``bounds`` argument
        2. ``data.bounds`` attribute
        3. Unconstrained (-inf, +inf)
        """
        n = len(parameter_names)

        # Resolve initial params
        if initial_params is not None:
            p0 = np.asarray(initial_params, dtype=np.float64)
        elif hasattr(data, "initial_params") and data.initial_params is not None:
            p0 = np.asarray(data.initial_params, dtype=np.float64)
        else:
            logger.warning(
                "NLSQWrapper: no initial_params provided; using zeros for %d params",
                n,
            )
            p0 = np.zeros(n, dtype=np.float64)

        # Resolve bounds
        if bounds is not None:
            lower = np.asarray(bounds[0], dtype=np.float64)
            upper = np.asarray(bounds[1], dtype=np.float64)
        elif hasattr(data, "bounds") and data.bounds is not None:
            lower = np.asarray(data.bounds[0], dtype=np.float64)
            upper = np.asarray(data.bounds[1], dtype=np.float64)
        else:
            lower = np.full(n, -np.inf)
            upper = np.full(n, np.inf)

        # Clip initial params to bounds
        finite_lower = np.where(np.isfinite(lower), lower, -1e10)
        finite_upper = np.where(np.isfinite(upper), upper, 1e10)
        p0 = np.clip(p0, finite_lower, finite_upper)

        return p0, (lower, upper)

    def _build_residual_fn(
        self,
        data: Any,
        config: Any,
        parameter_names: list[str],
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        """Build a residual function from XPCS data.

        If ``data`` exposes a ``residual_fn`` attribute (pre-built callable),
        it is used directly.  Otherwise, if ``data`` exposes a ``g2``
        attribute, a simple flat-residual function is constructed.

        Returns:
            Callable ``(params: ndarray) -> residuals: ndarray``, or None
            if the data format is not recognised.
        """
        # Pre-built residual function (preferred)
        if hasattr(data, "residual_fn") and callable(data.residual_fn):
            logger.debug("NLSQWrapper: using pre-built residual_fn from data")
            return data.residual_fn  # type: ignore[no-any-return]

        # Fall back: build from g2 array
        if hasattr(data, "g2"):
            g2 = np.asarray(data.g2, dtype=np.float64)
            g2_flat = g2.ravel()
            n_data = len(g2_flat)
            logger.debug(
                "NLSQWrapper: building residual_fn from g2 array (%d points)", n_data
            )

            # Try to obtain a model callable from data
            if hasattr(data, "model") and callable(data.model):
                model_fn = data.model

                def _residual_from_model(params: np.ndarray) -> np.ndarray:
                    prediction = np.asarray(model_fn(params), dtype=np.float64).ravel()
                    return prediction - g2_flat

                return _residual_from_model

            # No model available — return flat zeros residual (for API-only use)
            logger.warning(
                "NLSQWrapper: no model callable on data; returning zero residuals. "
                "Pass a pre-built residual_fn via data.residual_fn for real fits."
            )

            def _zero_residual(params: np.ndarray) -> np.ndarray:
                return np.zeros(n_data, dtype=np.float64)

            return _zero_residual

        # Unknown format
        logger.error(
            "NLSQWrapper: cannot build residual_fn — data has neither "
            "'residual_fn' nor 'g2' attribute"
        )
        return None

    @staticmethod
    def _safe_uncertainties_from_pcov(
        pcov: np.ndarray | None, n_params: int
    ) -> np.ndarray:
        """Extract uncertainties with diagonal regularisation for singular pcov."""
        return safe_uncertainties_from_pcov(pcov, n_params)


__all__ = [
    "HETERODYNE_ALL_PARAM_NAMES",
    "HETERODYNE_PHYSICAL_PARAM_NAMES",
    "HETERODYNE_SCALING_PARAM_NAMES",
    "NLSQWrapper",
]
