"""Core NLSQ fitting for heterodyne analysis.

Unified entry point for NLSQ optimization with:
- Global optimization selection (CMA-ES → multi-start → local)
- Adapter/wrapper fallback with automatic recovery
- Memory-aware strategy selection
- Per-angle and multi-angle fitting
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from heterodyne.core.jax_backend import (
    compute_c2_heterodyne,
    compute_multi_angle_residuals,
    compute_residuals,
)
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    # Static-analyser imports: give Pyright/mypy the names unconditionally
    # so call sites guarded by ``HAS_X`` flags don't trip "possibly
    # unbound" diagnostics. Runtime behaviour is controlled by the
    # try/except blocks below.
    from heterodyne.core.heterodyne_model import HeterodyneModel
    from heterodyne.optimization.nlsq.adapter import (
        LowLevelNLSQWrapper as NLSQWrapper,
    )
    from heterodyne.optimization.nlsq.adapter import NLSQAdapter
    from heterodyne.optimization.nlsq.cmaes_wrapper import fit_with_cmaes
    from heterodyne.optimization.nlsq.memory import (
        NLSQStrategy,
        select_nlsq_strategy,
    )
    from heterodyne.optimization.nlsq.multistart import (
        MultiStartConfig,
        MultiStartOptimizer,
    )

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — gated for graceful degradation
# ---------------------------------------------------------------------------

try:
    # NLSQWrapper here is the LOW-LEVEL wrapper (adapter.NLSQWrapper);
    # alias it explicitly to avoid shadowing wrapper.NLSQWrapper (the public
    # high-level stable-fallback re-exported from heterodyne.optimization.nlsq).
    from heterodyne.optimization.nlsq.adapter import (  # noqa: F811
        LowLevelNLSQWrapper as NLSQWrapper,
    )
    from heterodyne.optimization.nlsq.adapter import NLSQAdapter  # noqa: F811

    HAS_ADAPTERS = True
    HAS_WRAPPER = True
except ImportError:
    # Call sites are guarded by ``HAS_ADAPTERS`` / ``HAS_WRAPPER``; the
    # names above remain unbound but are visible to static analysers via
    # the ``TYPE_CHECKING`` block.
    HAS_ADAPTERS = False
    HAS_WRAPPER = False

try:
    from heterodyne.optimization.nlsq.multistart import (  # noqa: F811
        MultiStartConfig,
        MultiStartOptimizer,
    )

    HAS_MULTISTART = True
except ImportError:
    HAS_MULTISTART = False

try:
    from heterodyne.optimization.nlsq.cmaes_wrapper import (
        CMAES_AVAILABLE,
        fit_with_cmaes,  # noqa: F811
    )

    HAS_CMAES = CMAES_AVAILABLE
except ImportError:
    HAS_CMAES = False

try:
    from heterodyne.optimization.nlsq.memory import (  # noqa: F811
        NLSQStrategy,
        select_nlsq_strategy,
    )

    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

# Export availability flag for tests
NLSQ_AVAILABLE = HAS_ADAPTERS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_nlsq_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
    use_nlsq_library: bool = True,
    *,
    _skip_global_selection: bool = False,
) -> NLSQResult:
    """Fit heterodyne model to correlation data using NLSQ.

    This is the unified entry point for all NLSQ optimization.  When called
    it first checks for global optimization methods:

    1. If ``cmaes.enable: true`` → delegates to CMA-ES
    2. If ``multi_start.enable: true`` → delegates to multi-start
    3. Otherwise → runs local trust-region optimization

    The adapter is tried first; on failure the wrapper provides automatic
    retry with progressive recovery (HybridRecoveryConfig).

    Args:
        model: HeterodyneModel instance with parameters configured.
        c2_data: Experimental correlation data, shape (N, N).
        phi_angle: Detector phi angle (degrees).
        config: NLSQ configuration (default if None).
        weights: Optional weights (1/sigma²) for weighted least squares.
        use_nlsq_library: Whether to prefer nlsq library over scipy.
        _skip_global_selection: Internal flag — skip CMA-ES / multi-start check.

    Returns:
        NLSQResult with fitted parameters and diagnostics.
    """
    if config is None:
        config = NLSQConfig()

    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("phi=%s°, method=%s", phi_angle, config.method)

    # ------------------------------------------------------------------
    # Input validation (homodyne parity — non-strict observer).
    # Emits WARNINGs on shape/bounds/finiteness issues but does not
    # block the fit. The fit proceeds either way; strict mode is
    # opt-in via the validator's constructor.
    # ------------------------------------------------------------------
    _run_input_validation(model=model, c2_data=c2_data)

    # ------------------------------------------------------------------
    # Global optimization selection (CMA-ES → multi-start → local)
    # ------------------------------------------------------------------
    if not _skip_global_selection:
        global_result = _try_global_optimization(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library,
        )
        if global_result is not None:
            _run_result_validation(result=global_result, model=model)
            return global_result
        logger.debug("No global optimization enabled, using local optimization")

    # ------------------------------------------------------------------
    # Local optimization
    # ------------------------------------------------------------------
    result = _fit_local(model, c2_data, phi_angle, config, weights, use_nlsq_library)
    _run_result_validation(result=result, model=model)
    return result


def fit_nlsq_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float] | np.ndarray,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> list[NLSQResult]:
    """Fit model to correlation data at multiple phi angles.

    Two modes of operation controlled by ``config.per_angle_mode``:

    - **Joint fit** (``"fourier"``, ``"independent"``, or ``"auto"``
      with multiple angles) -- All angles are fit simultaneously in a
      single optimization.  In ``"fourier"`` mode, the optimizer vector is
      ``[physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]``,
      where the Fourier basis constrains smooth angular variation.
      In ``"independent"`` mode, each angle has its own contrast/offset
      (``2*n_phi`` scaling parameters), all optimized jointly.

    - **Sequential mode** (single angle or fallback) -- Angles are fit one
      at a time with warm-starting.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data, shape ``(n_phi, N, N)`` or ``(N, N)``.
        phi_angles: Array of phi angles (degrees).
        config: NLSQ configuration.
        weights: Optional weights, shape ``(n_phi, N, N)`` or ``(N, N)``.

    Returns:
        List of :class:`NLSQResult`, one per angle.
    """
    phi_angles = np.asarray(phi_angles)

    if c2_data.ndim == 2:
        c2_data = c2_data[np.newaxis, ...]

    if len(c2_data) != len(phi_angles):
        raise ValueError(
            f"Number of c2 matrices ({len(c2_data)}) doesn't match "
            f"number of phi angles ({len(phi_angles)})"
        )

    # ------------------------------------------------------------------
    # Anti-degeneracy diagnostics (homodyne parity).
    #
    # The AntiDegeneracyController is the ported homodyne 4-layer defense
    # class. Heterodyne's joint-fit path (``_fit_joint_multi_phi``) is the
    # actual orchestrator, but the controller is consulted here as an
    # active observer so its mode-resolution logic and group-variance
    # diagnostics are exercised on every multi-angle fit. This keeps the
    # ported class on the production hot path rather than shelf-ware.
    # ------------------------------------------------------------------
    _log_anti_degeneracy_diagnostics(
        config=config,
        phi_angles=phi_angles,
    )

    # ------------------------------------------------------------------
    # Determine whether to use homodyne-style joint multi-angle fitting.
    # ------------------------------------------------------------------
    use_joint = False
    fourier: Any = None
    if config is not None and len(phi_angles) > 1:
        if getattr(config, "enable_cmaes", False) and HAS_CMAES:
            logger.info("CMA-ES enabled, delegating to joint multi-angle CMA-ES")
            return _fit_joint_cmaes_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
            )

        constant_threshold = max(
            int(getattr(config, "constant_scaling_threshold", 3)), 1
        )
        if _use_fixed_constant_scaling_mode(config, len(phi_angles)):
            logger.info(
                "Fixed-constant scaling selected (homodyne 'constant' parity): "
                "per-angle β,o frozen from quantile, n_phi=%d "
                "(joint fit of 14 physics only)",
                len(phi_angles),
            )
            return _fit_joint_fixed_constant_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
            )
        if _use_averaged_constant_scaling_mode(config, len(phi_angles)):
            logger.info(
                "Auto averaged scaling selected: mode=%s, n_phi=%d, threshold=%d "
                "(joint fit of 14 physics + 2 averaged β,o = 16 params)",
                config.per_angle_mode,
                len(phi_angles),
                constant_threshold,
            )
            return _fit_joint_averaged_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
            )

        try:
            from heterodyne.optimization.nlsq.fourier_reparam import (
                FourierReparamConfig,
                FourierReparameterizer,
            )

            fourier_config = FourierReparamConfig(
                mode=config.per_angle_mode,
                fourier_order=config.fourier_order,
                auto_threshold=config.fourier_auto_threshold,
            )
            phi_rad = np.deg2rad(phi_angles.astype(np.float64))
            fourier = FourierReparameterizer(phi_rad, fourier_config)
            use_joint = True
        except ImportError:
            logger.warning(
                "fourier_reparam not available, falling back to sequential fits"
            )

    if use_joint:
        assert config is not None  # use_joint=True implies config was non-None
        assert fourier is not None
        return _fit_joint_multi_phi(
            model,
            c2_data,
            phi_angles,
            config,
            weights,
            fourier,
        )

    # ------------------------------------------------------------------
    # Sequential per-angle fitting (warm-start chain)
    # ------------------------------------------------------------------
    results = []
    for i, phi in enumerate(phi_angles):
        if i > 0:
            logger.info(
                "Fitting phi angle %d/%d: %s° (warm-start from angle %s°)",
                i + 1,
                len(phi_angles),
                phi,
                phi_angles[i - 1],
            )
        else:
            logger.info("Fitting phi angle %d/%d: %s°", i + 1, len(phi_angles), phi)

        c2_i = c2_data[i]
        weights_i = weights[i] if weights is not None and weights.ndim == 3 else weights

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_i,
            phi_angle=float(phi),
            config=config,
            weights=weights_i,
        )
        result.metadata["phi_angle"] = float(phi)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Public global-optimization entry points (parity with homodyne)
# ---------------------------------------------------------------------------


def fit_nlsq_cmaes(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
) -> NLSQResult:
    """CMA-ES global optimization for multi-scale parameter problems.

    Public entry point that delegates to the internal ``_fit_cmaes``
    implementation.  Raises ``ImportError`` if CMA-ES is not available and
    ``ValueError`` if CMA-ES is not enabled in *config*.

    Args:
        model: HeterodyneModel instance with parameters configured.
        c2_data: Experimental correlation data, shape (N, N).
        phi_angle: Detector phi angle (degrees).
        config: NLSQ configuration.  Defaults to ``NLSQConfig()``.
        weights: Optional weights (1/σ²) for weighted least squares.

    Returns:
        NLSQResult with fitted parameters and diagnostics.

    Raises:
        ImportError: If CMA-ES (``cma`` package) is not available.
        ValueError: If CMA-ES is not enabled in *config*.

    Examples:
        >>> config = NLSQConfig(enable_cmaes=True)
        >>> result = fit_nlsq_cmaes(model, c2_data, config=config)
        >>> print(f"Chi2: {result.reduced_chi_squared:.4f}")
    """
    if not HAS_CMAES:
        raise ImportError("CMA-ES requires the 'cma' package. Install with: uv add cma")
    if config is None:
        config = NLSQConfig()
    if not getattr(config, "enable_cmaes", False):
        raise ValueError(
            "CMA-ES optimization is not enabled. Set enable_cmaes=True in NLSQConfig."
        )
    return _fit_cmaes(model, c2_data, phi_angle, config, weights)


def fit_nlsq_multistart(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
    use_nlsq_library: bool = True,
) -> NLSQResult:
    """Multi-start NLSQ optimization with Latin Hypercube Sampling.

    Public entry point that delegates to the internal ``_fit_multistart``
    implementation.  Explores the parameter space to avoid local minima.
    FULL strategy is always used — no subsampling.

    Args:
        model: HeterodyneModel instance with parameters configured.
        c2_data: Experimental correlation data, shape (N, N).
        phi_angle: Detector phi angle (degrees).
        config: NLSQ configuration.  Defaults to ``NLSQConfig()``.
        weights: Optional weights (1/σ²) for weighted least squares.
        use_nlsq_library: Whether to prefer nlsq library over scipy.

    Returns:
        NLSQResult with best parameters across all starts.

    Raises:
        ImportError: If the multi-start module is not available.
        ValueError: If multi-start is not enabled in *config*.

    Examples:
        >>> config = NLSQConfig(multistart=True, multistart_n=20)
        >>> result = fit_nlsq_multistart(model, c2_data, config=config)
        >>> print(f"Best chi2: {result.reduced_chi_squared:.4f}")
    """
    if not HAS_MULTISTART:
        raise ImportError(
            "Multi-start optimization requires the multistart module. "
            "Ensure heterodyne.optimization.nlsq.multistart is importable."
        )
    if config is None:
        config = NLSQConfig()
    if not getattr(config, "multistart", False):
        raise ValueError(
            "Multi-start optimization is not enabled. "
            "Set multistart=True in NLSQConfig."
        )
    return _fit_multistart(model, c2_data, phi_angle, config, weights, use_nlsq_library)


def _compute_per_angle_chi2(
    residuals: np.ndarray,
    c2_matrix: np.ndarray,
    n_params: int,
) -> tuple[float, float]:
    """Compute per-angle cost and noise-normalised reduced chi-squared.

    Joint fits produce one aggregated cost and chi2 for all angles. This
    helper reconstructs the per-angle statistics so each NLSQResult carries
    its own diagnostics rather than a copy of the joint value.

    Args:
        residuals: Flat off-diagonal residual vector from compute_residuals,
            length n*(n-1).
        c2_matrix: Per-angle experimental C2 matrix, shape (n, n).
        n_params: Number of varying physics parameters.

    Returns:
        ``(per_angle_cost, reduced_chi_squared)`` where ``per_angle_cost``
        is ``0.5*SSR`` and ``reduced_chi_squared`` is noise-normalised
        (target ≈ 1.0 for a good fit; MSE fallback when noise is degenerate).
    """
    ssr = float(np.sum(residuals**2))
    per_angle_cost = 0.5 * ssr

    n_matrix = c2_matrix.shape[0]
    n_valid = (n_matrix - 1) * (
        n_matrix - 2
    )  # valid off-diagonal non-boundary residuals
    n_dof = max(n_valid - n_params, 1)

    # Far-lag photon-noise estimate — same formula as _fit_local
    c2_np = np.asarray(c2_matrix)
    row_idx = np.arange(n_matrix)
    lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
    far_vals = c2_np[lag_mat >= n_matrix // 2]
    sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

    if sigma2_noise > 1e-12:
        reduced_chi2 = ssr / (sigma2_noise * n_dof)
    else:
        reduced_chi2 = ssr / n_dof  # MSE fallback

    return per_angle_cost, reduced_chi2


def _anti_degen_dict_from_config(config: NLSQConfig) -> dict[str, Any]:
    """Project ``NLSQConfig`` fields onto the ``AntiDegeneracyConfig`` schema.

    Mirrors the shape consumed by
    :meth:`AntiDegeneracyController.from_config` so the joint-fit code paths
    can construct a controller from the active ``NLSQConfig`` without
    duplicating the projection logic.

    Returns a nested ``dict`` matching the YAML structure under
    ``optimization.nlsq.anti_degeneracy``.
    """
    return {
        "enable": config.enable_anti_degeneracy,
        "per_angle_mode": config.per_angle_mode,
        "fourier_order": config.fourier_order,
        "fourier_auto_threshold": config.fourier_auto_threshold,
        "constant_scaling_threshold": getattr(config, "constant_scaling_threshold", 3),
        "hierarchical": {
            "enable": config.enable_hierarchical,
            "max_outer_iterations": config.hierarchical_max_outer_iterations,
            "outer_tolerance": config.hierarchical_outer_tolerance,
            "physical_max_iterations": config.hierarchical_physical_max_iterations,
            "per_angle_max_iterations": config.hierarchical_per_angle_max_iterations,
        },
        "regularization": {
            "mode": config.regularization_mode,
            "lambda": config.group_variance_lambda,
            "target_cv": config.regularization_target_cv,
            "target_contribution": config.regularization_target_contribution,
            "max_cv": config.regularization_max_cv,
        },
        "gradient_monitoring": {
            "enable": config.enable_gradient_monitoring,
            "ratio_threshold": config.gradient_ratio_threshold,
            "consecutive_triggers": config.gradient_consecutive_triggers,
            "response": config.gradient_collapse_response,
        },
    }


def _build_anti_degen_controller(
    *,
    config: NLSQConfig,
    n_phi: int,
    phi_angles: np.ndarray,
    n_physical: int,
) -> Any | None:
    """Construct an ``AntiDegeneracyController`` when any active layer is requested.

    Sub-PR C3: broadened again to also build whenever L4 gradient-collapse
    monitoring is enabled.  L4 is observation-only (the monitor records
    per-iteration gradient norms and exposes a summary via
    ``controller.monitor.get_summary()``) and is meaningful even for the
    fixed-constant path where no per-angle scaling block exists.

    Sub-PR C2: broadened from the C1 marker-only guard.  The controller is
    built whenever anti-degeneracy is enabled and EITHER L2 hierarchical
    OR L3 regularization OR L4 gradient monitoring is requested.  L1
    (Fourier) is dispatched through a separate joint-fit path and does
    not require a controller here.

    Returns ``None`` when no active defense layer is requested, when the
    fit is single-angle, or when controller construction fails.  Callers
    are responsible for deriving the L2 marker, L3 callbacks, and L4
    monitor summary from the returned controller.
    """
    if not config.enable_anti_degeneracy:
        return None
    if not (
        config.enable_hierarchical
        or config.regularization_mode != "none"
        or config.enable_gradient_monitoring
    ):
        return None
    if n_phi <= 1:
        return None

    try:
        from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        return AntiDegeneracyController.from_config(
            config_dict=_anti_degen_dict_from_config(config),
            n_phi=n_phi,
            phi_angles=np.deg2rad(np.asarray(phi_angles, dtype=np.float64)),
            n_physical=n_physical,
            per_angle_scaling=True,
        )
    except Exception as exc:  # noqa: BLE001 — controller failure must not derail fit
        logger.warning(
            "Anti-degeneracy controller construction skipped: %s",
            exc,
        )
        return None


def _hierarchical_marker_from_controller(
    controller: Any | None,
) -> dict[str, Any] | None:
    """Derive the L2 hierarchical-request marker dict from a built controller.

    Sub-PR C1 marker (now plumbed through C2's broadened controller).
    Returns ``None`` if the controller did not enable hierarchical mode,
    otherwise returns the dict embedded in
    ``result.metadata["hierarchical_config"]``.

    ``HierarchicalFitter`` is a single-angle, stage-based parameter-
    unfreezing strategy (transport → velocity → fraction → all) that
    requires an ``NLSQAdapterBase`` at construction time — it is **not**
    the multi-phi outer/inner alternation implied by ``max_outer_iterations``.
    Wiring it directly into the joint residual path is out of scope here,
    so this is observe-only.

    TODO(C1-followup): replace this marker with an active outer/inner loop
    once the controller learns to drive the joint residual via callbacks
    (see ``create_nlsq_callbacks`` comment in
    ``anti_degeneracy_controller.py``).
    """
    if controller is None or not getattr(controller, "use_hierarchical", False):
        return None

    cfg_dict = dict(controller._hierarchical_config_dict)
    logger.info(
        "L2 hierarchical requested but not actively wired (see C1 follow-up); "
        "marker: max_outer_iterations=%d, outer_tolerance=%.1e",
        cfg_dict.get("max_outer_iterations", -1),
        cfg_dict.get("outer_tolerance", float("nan")),
    )
    return {
        "requested": True,
        "active": False,
        "max_outer_iterations": int(cfg_dict.get("max_outer_iterations", 0)),
        "outer_tolerance": float(cfg_dict.get("outer_tolerance", 0.0)),
        "physical_max_iterations": int(cfg_dict.get("physical_max_iterations", 0)),
        "per_angle_max_iterations": int(cfg_dict.get("per_angle_max_iterations", 0)),
    }


def _fit_joint_averaged_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> list[NLSQResult]:
    """Joint multi-angle fit with AVERAGED contrast/offset scaling.

    Homodyne parity for ``per_angle_mode="auto"`` (when n_phi >= threshold):
    per-angle quantile estimates are computed first, averaged to one
    contrast and one offset, and those two scalars are optimized jointly
    with the 14 physics parameters (16 total).

    This is distinct from :func:`_fit_joint_fixed_constant_multi_phi`
    (added in Sub-PR B), which implements the true homodyne
    ``"constant"`` semantics by FREEZING per-angle β,o and optimizing
    only the 14 physics parameters.
    """
    from heterodyne.config.parameter_registry import SCALING_PARAMS
    from heterodyne.core.scaling_utils import compute_averaged_scaling

    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    physics_initial = np.asarray(param_manager.get_initial_values(), dtype=np.float64)
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    t = model.t
    q = model.q
    dt = model.dt

    t1_mesh, t2_mesh = np.meshgrid(np.asarray(t), np.asarray(t), indexing="ij")
    n_time_points = t1_mesh.size
    c2_flat = []
    t1_flat = []
    t2_flat = []
    phi_indices = []
    for i in range(n_phi):
        c2_flat.append(np.asarray(c2_data[i], dtype=np.float64).reshape(-1))
        t1_flat.append(t1_mesh.reshape(-1))
        t2_flat.append(t2_mesh.reshape(-1))
        phi_indices.append(np.full(n_time_points, i, dtype=np.int32))

    contrast_bounds = (
        SCALING_PARAMS["contrast"].min_bound,
        SCALING_PARAMS["contrast"].max_bound,
    )
    offset_bounds = (
        SCALING_PARAMS["offset"].min_bound,
        SCALING_PARAMS["offset"].max_bound,
    )

    logger.info("=" * 60)
    logger.info("AUTO AVERAGED SCALING: Computing per-angle scaling from quantiles")
    logger.info("=" * 60)
    avg_contrast, avg_offset, contrast_per_angle, offset_per_angle = (
        compute_averaged_scaling(
            c2_data=np.concatenate(c2_flat),
            t1=np.concatenate(t1_flat),
            t2=np.concatenate(t2_flat),
            phi_indices=np.concatenate(phi_indices),
            n_phi=n_phi,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
            log=logger,
        )
    )

    x0 = np.concatenate([physics_initial, [avg_contrast, avg_offset]])
    lb = np.concatenate([physics_lower, [contrast_bounds[0], offset_bounds[0]]])
    ub = np.concatenate([physics_upper, [contrast_bounds[1], offset_bounds[1]]])
    joint_param_names = [*varying_names, "contrast", "offset"]

    logger.info(
        "Joint auto averaged fit: %d physical + 2 averaged scaling = %d total params, %d angles",
        n_physics_varying,
        len(x0),
        n_phi,
    )

    # Anti-degeneracy controller construction (Sub-PR C1 + C2).
    # Built whenever EITHER L2 hierarchical OR L3 regularization is requested.
    anti_degen_controller = _build_anti_degen_controller(
        config=config,
        n_phi=n_phi,
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_physical=n_physics_varying,
    )
    hierarchical_marker = _hierarchical_marker_from_controller(anti_degen_controller)

    c2_data_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_batch = (
        jnp.asarray(weights, dtype=jnp.float64)
        if weights is not None
        else jnp.ones_like(c2_data_batch)
    )
    if weights_batch.ndim == 2:
        weights_batch = jnp.broadcast_to(weights_batch, c2_data_batch.shape)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)
    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces this
    # closure; np.asarray() on a traced result raises TracerArrayConversionError.
    def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        physics_varying = x[:n_physics_varying]
        contrast = x[n_physics_varying]
        offset = x[n_physics_varying + 1]

        full_jax = fixed_values_jax.at[varying_indices_jax].set(
            jnp.asarray(physics_varying, dtype=jnp.float64)
        )
        contrasts_jax = jnp.full((n_phi,), contrast, dtype=jnp.float64)
        offsets_jax = jnp.full((n_phi,), offset, dtype=jnp.float64)
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    # L3 adaptive regularization wiring (Sub-PR C2).
    # When regularization_mode != "none", append a single Tikhonov penalty
    # row to the residual vector so the active fit penalises per-angle
    # scaling variance via the controller's loss-augmentation callback.
    if (
        anti_degen_controller is not None
        and getattr(anti_degen_controller, "regularizer", None) is not None
        and config.regularization_mode != "none"
    ):
        callbacks = anti_degen_controller.create_nlsq_callbacks()
        loss_aug = callbacks.get("loss_augmentation")

        if loss_aug is not None:
            _inner_residual_fn = joint_residual_fn

            def joint_residual_fn_with_penalty(x: np.ndarray) -> Any:  # type: ignore[return-value]
                """Residual + appended penalty row from controller's loss_aug."""
                base = _inner_residual_fn(x)
                base_np = np.asarray(base)
                penalty_value = float(loss_aug(np.asarray(x), base_np))
                penalty_row = float(np.sqrt(max(2.0 * penalty_value, 0.0)))
                return jnp.concatenate(
                    [
                        jnp.asarray(base, dtype=jnp.float64),
                        jnp.array([penalty_row], dtype=jnp.float64),
                    ]
                )

            joint_residual_fn = joint_residual_fn_with_penalty
            logger.info(
                "L3 adaptive regularization active (mode=%s, lambda=%.4e)",
                config.regularization_mode,
                config.group_variance_lambda,
            )

    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=len(x0),
    )

    joint_result: NLSQResult | None = None
    if HAS_ADAPTERS:
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
            )
            if not joint_result.success:
                raise RuntimeError(
                    f"Joint adapter returned success=False: {joint_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Joint auto averaged NLSQAdapter failed, falling back to NLSQWrapper: %s",
                adapter_exc,
            )
            joint_result = None

    if joint_result is None and HAS_WRAPPER:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for joint auto averaged multi-angle fit."
        )

    # L4 gradient collapse monitor wiring (Sub-PR C3).
    monitor_summary: dict[str, Any] = {}
    if (
        anti_degen_controller is not None
        and getattr(anti_degen_controller, "monitor", None) is not None
        and config.enable_gradient_monitoring
    ):
        # TODO(L4-followup): The current adapter loop does not expose a
        # per-iteration gradient callback to the controller, so
        # monitor.get_summary() is observation-passive — it returns empty
        # history.  To activate L4 we need either (a) the NLSQAdapter to
        # call back into the controller via iteration_callback from
        # create_nlsq_callbacks(), or (b) post-fit gradient-norm sampling
        # driven by the joint solve's iteration trace.  The metadata key
        # 'gradient_monitor' is currently set only as a placeholder to
        # surface that L4 was REQUESTED.
        try:
            monitor_summary = dict(anti_degen_controller.monitor.get_summary() or {})
        except (AttributeError, TypeError) as exc:
            logger.debug("L4 monitor summary unavailable: %s", exc)
            monitor_summary = {}
        if monitor_summary:
            logger.info("L4 gradient monitor summary: %s", monitor_summary)

    fitted_all = np.asarray(joint_result.parameters, dtype=np.float64)
    fitted_physics = fitted_all[:n_physics_varying]
    fitted_contrast = float(fitted_all[n_physics_varying])
    fitted_offset = float(fitted_all[n_physics_varying + 1])

    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling"):
        model.scaling.contrast[:] = fitted_contrast
        model.scaling.offset[:] = fitted_offset

    wall_time = time.perf_counter() - t_start

    results: list[NLSQResult] = []
    for i, phi in enumerate(phi_angles):
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            float(phi),
            contrast=fitted_contrast,
            offset=fitted_offset,
        )
        residuals = np.asarray(
            compute_residuals(
                jnp.asarray(full_fitted),
                t,
                q,
                dt,
                float(phi),
                c2_data_batch[i],
                weights_batch[i],
                contrast=fitted_contrast,
                offset=fitted_offset,
            )
        )
        per_angle_cost, per_angle_chi2 = _compute_per_angle_chi2(
            residuals, np.asarray(c2_data_batch[i]), n_physics_varying
        )

        result = NLSQResult(
            parameters=fitted_physics.copy(),
            parameter_names=varying_names,
            uncertainties=(
                joint_result.uncertainties[:n_physics_varying].copy()
                if joint_result.uncertainties is not None
                else None
            ),
            covariance=(
                joint_result.covariance[:n_physics_varying, :n_physics_varying].copy()
                if joint_result.covariance is not None
                else None
            ),
            residuals=residuals,
            final_cost=per_angle_cost,
            reduced_chi_squared=per_angle_chi2,
            success=bool(joint_result.success),
            message=str(joint_result.message),
            n_iterations=joint_result.n_iterations,
            n_function_evals=joint_result.n_function_evals,
            convergence_reason=joint_result.convergence_reason,
            fitted_correlation=np.asarray(fitted_c2),
            wall_time_seconds=joint_result.wall_time_seconds,
            metadata={
                "phi_angle": float(phi),
                "contrast": fitted_contrast,
                "offset": fitted_offset,
                "contrast_initial_quantile": float(contrast_per_angle[i]),
                "offset_initial_quantile": float(offset_per_angle[i]),
                "contrast_initial_average": avg_contrast,
                "offset_initial_average": avg_offset,
                "optimizer": "joint_auto_averaged",
                "n_angles_joint": n_phi,
                "wall_time_total": wall_time,
                **(
                    {"hierarchical_config": hierarchical_marker}
                    if hierarchical_marker is not None
                    else {}
                ),
                **({"gradient_monitor": monitor_summary} if monitor_summary else {}),
            },
        )
        results.append(result)

    logger.info(
        "Joint auto averaged fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles",
        joint_result.success,
        joint_result.final_cost or 0.0,
        joint_result.n_function_evals or 0,
        wall_time,
        n_phi,
    )

    return results


def _fit_joint_fixed_constant_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> list[NLSQResult]:
    """Joint multi-angle fit with FROZEN per-angle contrast/offset.

    Homodyne parity for ``per_angle_mode="constant"``: per-angle β(φ) and
    o(φ) are estimated once from quantile analysis (5 %/95 % of g2 per
    angle) and held fixed.  The optimizer fits only the 14 physics
    parameters.

    Distinct from :func:`_fit_joint_averaged_multi_phi`, which AVERAGES
    per-angle estimates to a single scalar β̄, ō and optimizes them
    jointly with physics (16 params total — the homodyne ``"auto"``
    behaviour).
    """
    from heterodyne.config.parameter_registry import SCALING_PARAMS
    from heterodyne.core.quantile_scaling import compute_per_angle_quantile_scaling

    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    physics_initial = np.asarray(param_manager.get_initial_values(), dtype=np.float64)
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    contrast_bounds = (
        SCALING_PARAMS["contrast"].min_bound,
        SCALING_PARAMS["contrast"].max_bound,
    )
    offset_bounds = (
        SCALING_PARAMS["offset"].min_bound,
        SCALING_PARAMS["offset"].max_bound,
    )

    # Per-angle quantile estimates (the FREEZE step).
    g2_flat_parts = []
    phi_idx_parts = []
    for i in range(n_phi):
        flat = np.asarray(c2_data[i], dtype=np.float64).reshape(-1)
        g2_flat_parts.append(flat)
        phi_idx_parts.append(np.full(flat.size, i, dtype=np.int32))

    quantile = compute_per_angle_quantile_scaling(
        g2=np.concatenate(g2_flat_parts),
        phi_indices=np.concatenate(phi_idx_parts),
        n_phi=n_phi,
        contrast_bounds=contrast_bounds,
        offset_bounds=offset_bounds,
    )
    contrast_per_angle = quantile.contrast_per_angle
    offset_per_angle = quantile.offset_per_angle

    logger.info("=" * 60)
    logger.info(
        "FIXED CONSTANT scaling (homodyne 'constant' parity): "
        "per-angle β,o frozen from quantile, n_phi=%d",
        n_phi,
    )
    logger.info(
        "  contrast range: [%.4f, %.4f]; offset range: [%.4f, %.4f]",
        float(contrast_per_angle.min()),
        float(contrast_per_angle.max()),
        float(offset_per_angle.min()),
        float(offset_per_angle.max()),
    )
    logger.info("=" * 60)

    # Optimizer sees only the physics parameters; β,o enter the residual
    # closure as captured constants.
    x0 = physics_initial.copy()
    lb = np.asarray(physics_lower, dtype=np.float64).copy()
    ub = np.asarray(physics_upper, dtype=np.float64).copy()
    joint_param_names = list(varying_names)

    logger.info(
        "Joint fixed-constant fit: %d physics params (frozen β,o), %d angles",
        n_physics_varying,
        n_phi,
    )

    # L4 gradient collapse monitor wiring (Sub-PR C3).
    # Fixed-constant has no per-angle scaling block, so L2/L3 don't apply
    # (per_angle_scaling=False), but L4 is observation-only and meaningful
    # for the physics-only optimization.
    anti_degen_controller: Any = None
    if config.enable_anti_degeneracy and config.enable_gradient_monitoring:
        try:
            from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
                AntiDegeneracyController,
            )

            anti_degen_controller = AntiDegeneracyController.from_config(
                config_dict=_anti_degen_dict_from_config(config),
                n_phi=n_phi,
                phi_angles=np.deg2rad(np.asarray(phi_angles, dtype=np.float64)),
                n_physical=n_physics_varying,
                per_angle_scaling=False,  # fixed-constant has no per-angle block
            )
        except Exception as exc:  # noqa: BLE001 — never derail the fit
            logger.warning("Fixed-constant L4 controller construction skipped: %s", exc)
            anti_degen_controller = None

    t = model.t
    q = model.q
    dt = model.dt
    c2_data_list = [jnp.asarray(c2_data[i], dtype=jnp.float64) for i in range(n_phi)]
    weights_list: list[jnp.ndarray | None] = []
    for i in range(n_phi):
        if weights is not None and weights.ndim == 3:
            weights_list.append(jnp.asarray(weights[i], dtype=jnp.float64))
        elif weights is not None:
            weights_list.append(jnp.asarray(weights, dtype=jnp.float64))
        else:
            weights_list.append(None)
    c2_data_batch = jnp.stack(c2_data_list, axis=0)
    weights_batch = jnp.stack(
        [
            (w if w is not None else jnp.ones_like(c2_data_list[i]))
            for i, w in enumerate(weights_list)
        ],
        axis=0,
    )

    contrast_per_angle_j = jnp.asarray(contrast_per_angle, dtype=jnp.float64)
    offset_per_angle_j = jnp.asarray(offset_per_angle, dtype=jnp.float64)

    def joint_residual_fn(physics_only: np.ndarray) -> np.ndarray:
        """Residual closure: β,o are FIXED per-angle constants."""
        full = jnp.asarray(physics_only, dtype=jnp.float64)
        all_res = []
        for i, phi in enumerate(phi_angles):
            res_i = compute_residuals(
                full,
                t,
                q,
                dt,
                float(phi),
                c2_data_batch[i],
                weights_batch[i],
                contrast=float(contrast_per_angle_j[i]),
                offset=float(offset_per_angle_j[i]),
            )
            all_res.append(np.asarray(res_i).reshape(-1))
        return np.concatenate(all_res)

    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=len(x0),
    )

    joint_result: NLSQResult | None = None
    if HAS_ADAPTERS:
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
            )
            if not joint_result.success:
                raise RuntimeError(
                    f"Fixed-constant joint adapter returned "
                    f"success=False: {joint_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Fixed-constant NLSQAdapter failed, falling back to wrapper: %s",
                adapter_exc,
            )
            joint_result = None

    if joint_result is None and HAS_WRAPPER:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for fixed-constant joint multi-angle fit."
        )

    # L4 gradient collapse monitor wiring (Sub-PR C3).
    monitor_summary: dict[str, Any] = {}
    if (
        anti_degen_controller is not None
        and getattr(anti_degen_controller, "monitor", None) is not None
        and config.enable_gradient_monitoring
    ):
        # TODO(L4-followup): The current adapter loop does not expose a
        # per-iteration gradient callback to the controller, so
        # monitor.get_summary() is observation-passive — it returns empty
        # history.  To activate L4 we need either (a) the NLSQAdapter to
        # call back into the controller via iteration_callback from
        # create_nlsq_callbacks(), or (b) post-fit gradient-norm sampling
        # driven by the joint solve's iteration trace.  The metadata key
        # 'gradient_monitor' is currently set only as a placeholder to
        # surface that L4 was REQUESTED.
        try:
            monitor_summary = dict(anti_degen_controller.monitor.get_summary() or {})
        except (AttributeError, TypeError) as exc:
            logger.debug("L4 monitor summary unavailable: %s", exc)
            monitor_summary = {}
        if monitor_summary:
            logger.info("L4 gradient monitor summary: %s", monitor_summary)

    fitted_physics = np.asarray(joint_result.parameters, dtype=np.float64)
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling"):
        # Surface the frozen per-angle β,o on the model for downstream viz.
        try:
            model.scaling.contrast[:] = contrast_per_angle
            model.scaling.offset[:] = offset_per_angle
        except (ValueError, TypeError):
            # Length mismatch (e.g. scaling was a scalar holder): replace.
            model.scaling.contrast = np.asarray(contrast_per_angle, dtype=np.float64)
            model.scaling.offset = np.asarray(offset_per_angle, dtype=np.float64)

    wall_time = time.perf_counter() - t_start

    results: list[NLSQResult] = []
    for i, phi in enumerate(phi_angles):
        # Post-review backfill: populate fitted_correlation, residuals,
        # reduced_chi_squared per-angle (parity with _fit_joint_averaged_multi_phi).
        # KEY DIFFERENCE from averaged path: β,o are per-angle frozen constants,
        # not averaged scalars.
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            float(phi),
            contrast=float(contrast_per_angle[i]),
            offset=float(offset_per_angle[i]),
        )
        residuals_i = np.asarray(
            compute_residuals(
                jnp.asarray(full_fitted),
                t,
                q,
                dt,
                float(phi),
                c2_data_batch[i],
                weights_batch[i],
                contrast=float(contrast_per_angle[i]),
                offset=float(offset_per_angle[i]),
            )
        )
        per_angle_cost_i, per_angle_chi2_i = _compute_per_angle_chi2(
            residuals_i, np.asarray(c2_data_batch[i]), n_physics_varying
        )

        per_angle_result = NLSQResult(
            parameters=fitted_physics.copy(),
            parameter_names=list(varying_names),
            success=joint_result.success,
            message=joint_result.message,
            covariance=joint_result.covariance,
            residuals=residuals_i,
            final_cost=per_angle_cost_i,
            reduced_chi_squared=per_angle_chi2_i,
            fitted_correlation=np.asarray(fitted_c2),
            n_iterations=joint_result.n_iterations,
            n_function_evals=joint_result.n_function_evals,
            wall_time_seconds=wall_time,
            metadata={
                **dict(joint_result.metadata or {}),
                "phi_angle": float(phi),
                "per_angle_mode_actual": "fixed_constant",
                "contrast_fixed": float(contrast_per_angle[i]),
                "offset_fixed": float(offset_per_angle[i]),
                "optimizer": "joint_fixed_constant",
                **({"gradient_monitor": monitor_summary} if monitor_summary else {}),
            },
        )
        results.append(per_angle_result)
    return results


def _fit_joint_cmaes_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> list[NLSQResult]:
    """Joint multi-angle CMA-ES with NLSQ warm-start and auto-skip.

    This mirrors homodyne's CMA-ES procedure at the orchestration level:
    first run the joint NLSQ path, optionally skip global search when the
    warm-start is already good, otherwise run CMA-ES and keep the lower-cost
    result.
    """
    from heterodyne.config.parameter_registry import SCALING_PARAMS
    from heterodyne.optimization.nlsq.cmaes_wrapper import CMAESConfig

    use_fixed_constant = _use_fixed_constant_scaling_mode(config, len(phi_angles))
    use_averaged_constant = _use_averaged_constant_scaling_mode(config, len(phi_angles))
    # legacy alias retained for downstream CMA-ES setup (residual closure,
    # bounds construction, result reconstruction) which all share the
    # averaged-style search-space layout regardless of warmstart path.
    use_constant = use_fixed_constant or use_averaged_constant
    fourier = (
        None
        if use_constant
        else _build_fourier_reparameterizer(
            phi_angles,
            config,
        )
    )

    if use_fixed_constant:
        warmstart_label = "fixed-constant (per-angle β,o frozen)"
    elif use_averaged_constant:
        warmstart_label = "averaged constant (β̄,ō jointly fit)"
    else:
        warmstart_label = "fourier/independent"

    logger.info("=" * 60)
    logger.info("CMA-ES GLOBAL OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("Analysis mode: %s", config.analysis_mode)
    logger.info(
        "Anti-degeneracy scaling mode: %s%s",
        warmstart_label,
        f" ({config.per_angle_mode})",
    )

    if use_fixed_constant:
        warmstart_results = _fit_joint_fixed_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=config,
            weights=weights,
        )
    elif use_averaged_constant:
        warmstart_results = _fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=config,
            weights=weights,
        )
    else:
        warmstart_results = _fit_joint_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=config,
            weights=weights,
            fourier=fourier,
        )

    first = warmstart_results[0]
    warmstart_cost = (
        float(first.final_cost) if first.final_cost is not None else float("inf")
    )
    warmstart_reduced_chi2 = (
        float(first.reduced_chi_squared)
        if first.reduced_chi_squared is not None
        else float("inf")
    )

    logger.info(
        "[CMA-ES] NLSQ warm-start succeeded: cost=%.4e, reduced chi2=%.4f",
        warmstart_cost,
        warmstart_reduced_chi2,
    )

    auto_skip = bool(getattr(config, "cmaes_warmstart_auto_skip", True))
    skip_threshold = float(getattr(config, "cmaes_warmstart_skip_threshold", 5.0))
    if auto_skip and warmstart_reduced_chi2 < skip_threshold:
        logger.info(
            "[CMA-ES] Auto-skip: NLSQ warm-start reduced chi2=%.4f < threshold=%.1f. "
            "Skipping CMA-ES global search.",
            warmstart_reduced_chi2,
            skip_threshold,
        )
        for result in warmstart_results:
            result.metadata["optimizer"] = "joint_cmaes_warmstart_auto_skip"
            result.metadata["cmaes_skipped"] = True
            result.metadata["warmstart_reduced_chi2"] = warmstart_reduced_chi2
        return warmstart_results

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    physics_lower, physics_upper = param_manager.get_bounds()
    if use_constant:
        contrast_bounds = (
            SCALING_PARAMS["contrast"].min_bound,
            SCALING_PARAMS["contrast"].max_bound,
        )
        offset_bounds = (
            SCALING_PARAMS["offset"].min_bound,
            SCALING_PARAMS["offset"].max_bound,
        )
        scaling_lower = np.array(
            [contrast_bounds[0], offset_bounds[0]],
            dtype=np.float64,
        )
        scaling_upper = np.array(
            [contrast_bounds[1], offset_bounds[1]],
            dtype=np.float64,
        )
        scaling_initial = np.array(
            [
                float(first.metadata.get("contrast", 0.3)),
                float(first.metadata.get("offset", 1.0)),
            ],
            dtype=np.float64,
        )
        scaling_names = ["contrast", "offset"]
    else:
        assert fourier is not None
        contrast_initial = np.array(
            [
                float(result.metadata.get("contrast", 0.3))
                for result in warmstart_results
            ],
            dtype=np.float64,
        )
        offset_initial = np.array(
            [float(result.metadata.get("offset", 1.0)) for result in warmstart_results],
            dtype=np.float64,
        )
        scaling_initial = fourier.per_angle_to_fourier(
            contrast_initial,
            offset_initial,
        )
        scaling_lower, scaling_upper = fourier.get_bounds()
        scaling_names = fourier.get_coefficient_labels()

    bounds = (
        np.concatenate([physics_lower, scaling_lower]),
        np.concatenate([physics_upper, scaling_upper]),
    )
    initial_params = np.concatenate(
        [np.asarray(first.parameters, dtype=np.float64), scaling_initial]
    )
    parameter_names = [*varying_names, *scaling_names]

    c2_data_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_batch = (
        jnp.asarray(weights, dtype=jnp.float64)
        if weights is not None
        else jnp.ones_like(c2_data_batch)
    )
    if weights_batch.ndim == 2:
        weights_batch = jnp.broadcast_to(weights_batch, c2_data_batch.shape)

    t = model.t
    q = model.q
    dt = model.dt
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)
    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # CMA-ES phase searches in averaged-style space (14 physics + 2 averaged β̄,ō);
    # fixed-constant warmstart hands off seeds in this space.
    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces this
    # closure; np.asarray() on a traced result raises TracerArrayConversionError.
    def residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        physics_varying = x[:n_physics_varying]
        full_jax = fixed_values_jax.at[varying_indices_jax].set(
            jnp.asarray(physics_varying, dtype=jnp.float64)
        )
        scaling_params = x[n_physics_varying:]
        if use_constant:
            contrast = scaling_params[0]
            offset = scaling_params[1]
            contrasts_jax = jnp.full((n_phi,), contrast, dtype=jnp.float64)
            offsets_jax = jnp.full((n_phi,), offset, dtype=jnp.float64)
        else:
            assert fourier is not None
            contrast_arr, offset_arr = fourier.fourier_to_per_angle(scaling_params)
            contrasts_jax = jnp.asarray(contrast_arr, dtype=jnp.float64)
            offsets_jax = jnp.asarray(offset_arr, dtype=jnp.float64)
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    def objective_fn(x: np.ndarray) -> float:
        residuals = residual_fn(x)
        return float(0.5 * np.sum(residuals**2))

    logger.info("[CMA-ES] Phase 2: Running CMA-ES global optimization...")
    n_time = int(c2_data_batch.shape[-1])
    n_off_diagonal_data = int(n_phi * (n_time - 1) * (n_time - 2))
    restart_strategy = getattr(config, "cmaes_restart_strategy", "bipop")
    max_restarts = getattr(config, "cmaes_max_restarts", 9)
    # Warmstart is always active in this path: BIPOP large-population restarts
    # are incoherent with a tight initial sigma derived from the NLSQ solution.
    if restart_strategy == "bipop":
        restart_strategy = "none"
        max_restarts = 0
        logger.debug(
            "[CMA-ES] Warm-start active: overriding restart_strategy='bipop' -> 'none' "
            "(BIPOP large-population restarts are incoherent with small sigma_warmstart)"
        )
    cmaes_result = fit_with_cmaes(  # pyright: ignore[reportPossiblyUnbound]
        objective_fn=objective_fn,
        initial_params=initial_params,
        bounds=bounds,
        parameter_names=parameter_names,
        config=CMAESConfig(
            sigma0=config.cmaes_sigma0,
            popsize=config.cmaes_population_size,
            maxiter=config.cmaes_max_iterations,
            tolx=config.cmaes_tolx,
            tolfun=config.cmaes_tolfun,
            diagonal_filtering=getattr(config, "cmaes_diagonal_filtering", "none"),
            restart_strategy=restart_strategy,
            max_restarts=max_restarts,
        ),
        residual_fn=residual_fn,
        n_data=n_off_diagonal_data,
        anti_degeneracy=getattr(config, "cmaes_anti_degeneracy", False),
    )

    cmaes_cost = (
        float(cmaes_result.final_cost)
        if cmaes_result.final_cost is not None
        else float("inf")
    )
    if warmstart_cost <= cmaes_cost:
        logger.info(
            "[CMA-ES] NLSQ warm-start result is better: NLSQ cost=%.4e < CMA-ES cost=%.4e. "
            "Using NLSQ solution.",
            warmstart_cost,
            cmaes_cost,
        )
        for result in warmstart_results:
            result.metadata["optimizer"] = "joint_cmaes_warmstart"
            result.metadata["cmaes_cost"] = cmaes_cost
            result.metadata["nlsq_warmstart_cost"] = warmstart_cost
        return warmstart_results

    logger.info(
        "[CMA-ES] CMA-ES result is better: CMA-ES cost=%.4e <= NLSQ cost=%.4e",
        cmaes_cost,
        warmstart_cost,
    )
    fitted = np.asarray(cmaes_result.parameters, dtype=np.float64)
    fitted_physics = fitted[:n_physics_varying]
    fitted_scaling = fitted[n_physics_varying:]
    if use_constant:
        fitted_contrast = np.full(n_phi, float(fitted_scaling[0]), dtype=np.float64)
        fitted_offset = np.full(n_phi, float(fitted_scaling[1]), dtype=np.float64)
    else:
        assert fourier is not None
        fitted_contrast, fitted_offset = fourier.fourier_to_per_angle(fitted_scaling)
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling"):
        model.scaling.contrast[:] = fitted_contrast
        model.scaling.offset[:] = fitted_offset

    results: list[NLSQResult] = []
    for i, phi in enumerate(phi_angles):
        contrast_i = float(fitted_contrast[i])
        offset_i = float(fitted_offset[i])
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            float(phi),
            contrast=contrast_i,
            offset=offset_i,
        )
        residuals = np.asarray(
            compute_residuals(
                jnp.asarray(full_fitted),
                t,
                q,
                dt,
                float(phi),
                c2_data_batch[i],
                weights_batch[i],
                contrast=contrast_i,
                offset=offset_i,
            )
        )
        metadata = {
            "phi_angle": float(phi),
            "contrast": contrast_i,
            "offset": offset_i,
            "optimizer": "joint_cmaes",
            "n_angles_joint": n_phi,
            "cmaes_cost": cmaes_cost,
            "nlsq_warmstart_cost": warmstart_cost,
        }
        if use_constant:
            metadata["anti_degeneracy_mode"] = "constant_averaged"
        else:
            assert fourier is not None
            metadata.update(
                {
                    "anti_degeneracy_mode": fourier.config.mode,
                    "fourier_mode": fourier.config.mode,
                    "fourier_order": fourier.order,
                    "fourier_coeffs": fitted_scaling.tolist(),
                    "fourier_n_coeffs": fourier.n_coeffs,
                    "fourier_reduction": fourier.get_diagnostics()["reduction_ratio"],
                }
            )
        results.append(
            NLSQResult(
                parameters=fitted_physics.copy(),
                parameter_names=varying_names,
                uncertainties=(
                    cmaes_result.uncertainties[:n_physics_varying].copy()
                    if cmaes_result.uncertainties is not None
                    else None
                ),
                covariance=(
                    cmaes_result.covariance[
                        :n_physics_varying, :n_physics_varying
                    ].copy()
                    if cmaes_result.covariance is not None
                    else None
                ),
                residuals=residuals,
                final_cost=cmaes_result.final_cost,
                reduced_chi_squared=cmaes_result.reduced_chi_squared,
                success=bool(cmaes_result.success),
                message=str(cmaes_result.message),
                n_iterations=cmaes_result.n_iterations,
                n_function_evals=cmaes_result.n_function_evals,
                convergence_reason=cmaes_result.convergence_reason,
                fitted_correlation=np.asarray(fitted_c2),
                wall_time_seconds=cmaes_result.wall_time_seconds,
                metadata=metadata,
            )
        )

    return results


def _use_fixed_constant_scaling_mode(config: NLSQConfig, n_phi: int) -> bool:
    """True when per-angle β,o must be FROZEN (homodyne `constant` parity)."""
    del n_phi  # explicit mode is threshold-independent
    return config.per_angle_mode == "constant"


def _use_averaged_constant_scaling_mode(config: NLSQConfig, n_phi: int) -> bool:
    """True when β,o are averaged across angles and OPTIMIZED (homodyne `auto`)."""
    constant_threshold = max(int(getattr(config, "constant_scaling_threshold", 3)), 1)
    return config.per_angle_mode == "auto" and n_phi >= constant_threshold


def _use_constant_scaling_mode(config: NLSQConfig, n_phi: int) -> bool:
    """Legacy union predicate; retained for backward compatibility.

    True if EITHER the fixed-constant or averaged-constant path applies.
    New code should call the specific predicate; this wrapper exists so
    legacy importers do not break.
    """
    return _use_fixed_constant_scaling_mode(
        config, n_phi
    ) or _use_averaged_constant_scaling_mode(config, n_phi)


def _run_input_validation(
    *,
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
) -> None:
    """Run :class:`InputValidator` against the fit inputs (non-strict).

    Bounds, finiteness, and initial-param-within-bounds checks. Any
    failures are logged at WARNING; the fit is not blocked. Strict
    enforcement is opt-in via the validator constructor and is not
    used here so the validation never breaks an otherwise-healthy fit.
    Failures inside the validator itself are themselves caught — input
    validation is an observer, not part of the hot path.
    """
    try:
        from heterodyne.optimization.nlsq.validation import InputValidator

        initial = np.asarray(model.param_manager.get_initial_values(), dtype=np.float64)
        lower, upper = model.param_manager.get_bounds()
        bounds = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        data = np.asarray(c2_data, dtype=np.float64)
        report = InputValidator(strict_mode=False).validate(
            data=data,
            initial_params=initial,
            bounds=bounds,
        )
        if not report.is_valid:
            logger.warning(
                "Input validation flagged %d issue(s) (continuing fit)",
                len(report.issues),
            )
    except Exception as exc:  # noqa: BLE001 — observer must never block
        logger.debug("Input validation skipped: %s", exc)


def _run_result_validation(
    *,
    result: NLSQResult,
    model: HeterodyneModel,
) -> None:
    """Run :class:`ResultValidator` against the optimization result (non-strict)."""
    try:
        from heterodyne.optimization.nlsq.validation import ResultValidator

        if result.parameters is None:
            return
        params = np.asarray(result.parameters, dtype=np.float64)
        lower, upper = model.param_manager.get_bounds()
        bounds = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        cov = (
            np.asarray(result.covariance, dtype=np.float64)
            if result.covariance is not None
            else None
        )
        ResultValidator(strict_mode=False).validate_all(
            params=params,
            covariance=cov,
            bounds=bounds,
            chi_squared=result.final_cost,
        )
    except Exception as exc:  # noqa: BLE001 — observer must never block
        logger.debug("Result validation skipped: %s", exc)


def _log_anti_degeneracy_diagnostics(
    *,
    config: NLSQConfig | None,
    phi_angles: np.ndarray,
) -> None:
    """Construct an ``AntiDegeneracyController`` and log its diagnostics.

    The controller's mode-resolution (auto → fourier / auto_averaged /
    individual / fixed_constant) and group-variance indices are computed
    here and emitted at DEBUG level. The class is the ported homodyne
    4-layer defense; heterodyne's actual joint-fit path lives in
    ``_fit_joint_multi_phi``, but the controller is exercised here as
    an active observer so its parity behaviour is verified on every
    multi-angle run.

    Failures (import errors, missing config fields, malformed phi
    arrays) are caught and logged at WARNING — the diagnostics path
    must never derail an otherwise-healthy fit.
    """
    if config is None or len(phi_angles) <= 1:
        return

    try:
        from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        ad_config_dict: dict[str, Any] = {
            "enable": True,
            "per_angle_mode": getattr(config, "per_angle_mode", "auto"),
            "constant_scaling_threshold": int(
                getattr(config, "constant_scaling_threshold", 3)
            ),
            "fourier_auto_threshold": int(
                getattr(config, "fourier_auto_threshold", 10)
            ),
            "fourier_order": int(getattr(config, "fourier_order", 1)),
        }
        controller = AntiDegeneracyController.from_config(
            config_dict=ad_config_dict,
            n_phi=len(phi_angles),
            phi_angles=np.deg2rad(phi_angles.astype(np.float64)),
            per_angle_scaling=True,
        )
        diag = controller.get_diagnostics()
        logger.debug(
            "AntiDegeneracyController diagnostics: mode=%s, n_phi=%d, "
            "per_angle_mode_actual=%s, group_indices=%s",
            diag.get("per_angle_mode"),
            diag.get("n_phi"),
            diag.get("per_angle_mode_actual"),
            controller.get_group_variance_indices(),
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must not block fits
        logger.warning("Anti-degeneracy diagnostics skipped: %s", exc)


def _build_fourier_reparameterizer(phi_angles: np.ndarray, config: NLSQConfig) -> Any:
    """Build the Fourier/independent reparameterizer for fallback paths."""
    from heterodyne.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )

    return FourierReparameterizer(
        np.deg2rad(phi_angles.astype(np.float64)),
        FourierReparamConfig(
            mode=config.per_angle_mode,
            fourier_order=config.fourier_order,
            auto_threshold=config.fourier_auto_threshold,
        ),
    )


def _fit_joint_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    fourier: Any,
) -> list[NLSQResult]:
    """Joint multi-angle fit with Fourier-parameterized scaling.

    The optimizer parameter vector is:
        [physics_varying_params | fourier_contrast_coeffs | fourier_offset_coeffs]

    The residual function evaluates all angles, using the Fourier basis to
    convert coefficients → per-angle contrast/offset at each evaluation.

    This is the heterodyne equivalent of homodyne's AntiDegeneracyController
    joint-fit path.
    """
    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    # Physics parameter initial values and bounds
    physics_initial = param_manager.get_initial_values()
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    # Fourier coefficient initial values and bounds
    scaling = model.scaling
    contrast_init = float(scaling.contrast[0]) if len(scaling.contrast) > 0 else 0.5
    offset_init = float(scaling.offset[0]) if len(scaling.offset) > 0 else 1.0
    fourier_initial = fourier.get_initial_coefficients(contrast_init, offset_init)
    fourier_lower, fourier_upper = fourier.get_bounds()

    # Combined parameter vector
    x0 = np.concatenate([physics_initial, fourier_initial])
    lb = np.concatenate([physics_lower, fourier_lower])
    ub = np.concatenate([physics_upper, fourier_upper])

    logger.info(
        "Joint multi-angle fit: %d physics + %d Fourier = %d total params, %d angles",
        n_physics_varying,
        fourier.n_coeffs,
        len(x0),
        n_phi,
    )

    # Anti-degeneracy controller construction (Sub-PR C1 + C2).
    # Built whenever EITHER L2 hierarchical OR L3 regularization is requested.
    anti_degen_controller = _build_anti_degen_controller(
        config=config,
        n_phi=n_phi,
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_physical=n_physics_varying,
    )
    hierarchical_marker = _hierarchical_marker_from_controller(anti_degen_controller)

    # Pre-convert data to JAX arrays (outside closure — constants)
    t, q, dt = model.t, model.q, model.dt
    c2_data_list = [jnp.asarray(c2_data[i], dtype=jnp.float64) for i in range(n_phi)]
    weights_list: list[jnp.ndarray | None] = []
    for i in range(n_phi):
        if weights is not None and weights.ndim == 3:
            weights_list.append(jnp.asarray(weights[i], dtype=jnp.float64))
        elif weights is not None:
            weights_list.append(jnp.asarray(weights, dtype=jnp.float64))
        else:
            weights_list.append(None)

    # Pre-stack batched arrays for compute_multi_angle_residuals.
    # weights_list entries may be None (unweighted) — materialise ones_like
    # so the stacked weights_batch is always a concrete (n_phi, N, N) array.
    c2_data_batch = jnp.stack(c2_data_list, axis=0)  # (n_phi, N, N)
    weights_batch = jnp.stack(
        [
            (w if w is not None else jnp.ones_like(c2_data_list[i]))
            for i, w in enumerate(weights_list)
        ],
        axis=0,
    )  # (n_phi, N, N)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)  # (n_phi,)

    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    def joint_residual_fn(x: np.ndarray) -> np.ndarray:
        """Compute concatenated residuals across all angles via vmap.

        Routes through ``compute_multi_angle_residuals`` (jit + vmap) to
        replace the previous n_phi serial kernel dispatches with a single
        batched XLA call.  Fourier reparameterization is preserved: the
        combined parameter vector is split into physics and Fourier parts,
        and ``fourier.fourier_to_per_angle`` converts coefficients to
        per-angle contrast/offset arrays before the batched residual call.
        """
        # Split combined vector
        physics_varying = x[:n_physics_varying]
        fourier_coeffs = x[n_physics_varying:]

        # Reconstruct full physics parameter array (immutable JAX scatter)
        varying_jax = jnp.asarray(physics_varying, dtype=jnp.float64)
        full_jax = fixed_values_jax.at[varying_indices_jax].set(varying_jax)

        # Convert Fourier coefficients → per-angle contrast/offset
        contrast_arr, offset_arr = fourier.fourier_to_per_angle(fourier_coeffs)
        contrasts_jax = jnp.asarray(contrast_arr, dtype=jnp.float64)  # (n_phi,)
        offsets_jax = jnp.asarray(offset_arr, dtype=jnp.float64)  # (n_phi,)

        # Single batched vmap call — eliminates n_phi serial dispatches
        return np.asarray(
            compute_multi_angle_residuals(
                full_jax,
                t,
                q,
                dt,
                phi_angles_jax,
                c2_data_batch,
                weights_batch,
                contrasts_jax,
                offsets_jax,
            )
        )

    # L3 adaptive regularization wiring (Sub-PR C2).
    # When regularization_mode != "none", append a single Tikhonov penalty
    # row to the residual vector so the active Fourier/individual fit
    # penalises per-angle scaling variance via the controller's
    # loss-augmentation callback.
    if (
        anti_degen_controller is not None
        and getattr(anti_degen_controller, "regularizer", None) is not None
        and config.regularization_mode != "none"
    ):
        callbacks = anti_degen_controller.create_nlsq_callbacks()
        loss_aug = callbacks.get("loss_augmentation")

        if loss_aug is not None:
            _inner_residual_fn = joint_residual_fn

            def joint_residual_fn_with_penalty(x: np.ndarray) -> np.ndarray:
                """Residual + appended penalty row from controller's loss_aug."""
                base = _inner_residual_fn(x)
                base_np = np.asarray(base)
                penalty_value = float(loss_aug(np.asarray(x), base_np))
                penalty_row = float(np.sqrt(max(2.0 * penalty_value, 0.0)))
                return np.concatenate([base_np, [penalty_row]])

            joint_residual_fn = joint_residual_fn_with_penalty
            logger.info(
                "L3 adaptive regularization active (mode=%s, lambda=%.4e)",
                config.regularization_mode,
                config.group_variance_lambda,
            )

    # Run optimization via NLSQAdapter (primary) with NLSQWrapper fallback
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
    )

    joint_result: NLSQResult | None = None
    joint_param_names = list(varying_names) + [
        f"fourier_{i}" for i in range(len(fourier_initial))
    ]

    if HAS_ADAPTERS:
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
            )
            if not joint_result.success:
                raise RuntimeError(
                    f"Joint adapter returned success=False: {joint_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Joint NLSQAdapter failed, falling back to NLSQWrapper: %s", adapter_exc
            )
            joint_result = None

    if joint_result is None and HAS_WRAPPER:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)  # pyright: ignore[reportPossiblyUnbound]
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for joint multi-angle fit. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    # L4 gradient collapse monitor wiring (Sub-PR C3).
    monitor_summary: dict[str, Any] = {}
    if (
        anti_degen_controller is not None
        and getattr(anti_degen_controller, "monitor", None) is not None
        and config.enable_gradient_monitoring
    ):
        # TODO(L4-followup): The current adapter loop does not expose a
        # per-iteration gradient callback to the controller, so
        # monitor.get_summary() is observation-passive — it returns empty
        # history.  To activate L4 we need either (a) the NLSQAdapter to
        # call back into the controller via iteration_callback from
        # create_nlsq_callbacks(), or (b) post-fit gradient-norm sampling
        # driven by the joint solve's iteration trace.  The metadata key
        # 'gradient_monitor' is currently set only as a placeholder to
        # surface that L4 was REQUESTED.
        try:
            monitor_summary = dict(anti_degen_controller.monitor.get_summary() or {})
        except (AttributeError, TypeError) as exc:
            logger.debug("L4 monitor summary unavailable: %s", exc)
            monitor_summary = {}
        if monitor_summary:
            logger.info("L4 gradient monitor summary: %s", monitor_summary)

    # Extract results
    fitted_params_full = joint_result.parameters
    fitted_physics = fitted_params_full[:n_physics_varying]
    fitted_fourier = fitted_params_full[n_physics_varying:]
    fitted_contrast, fitted_offset = fourier.fourier_to_per_angle(fitted_fourier)

    # Update model with fitted physics parameters
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)

    # Update model scaling
    if len(scaling.contrast) == n_phi:
        scaling.contrast[:] = fitted_contrast
        scaling.offset[:] = fitted_offset

    wall_time = time.perf_counter() - t_start

    # Build per-angle NLSQResult objects
    results: list[NLSQResult] = []
    for i in range(n_phi):
        # Compute fitted correlation for this angle
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            float(phi_angles[i]),
            contrast=float(fitted_contrast[i]),
            offset=float(fitted_offset[i]),
        )

        _residuals_i = np.asarray(
            compute_residuals(
                jnp.asarray(full_fitted),
                t,
                q,
                dt,
                float(phi_angles[i]),
                c2_data_list[i],
                weights_list[i],
                contrast=float(fitted_contrast[i]),
                offset=float(fitted_offset[i]),
            )
        )
        _per_cost_i, _per_chi2_i = _compute_per_angle_chi2(
            _residuals_i, np.asarray(c2_data_list[i]), n_physics_varying
        )
        result = NLSQResult(
            parameters=fitted_physics.copy(),
            parameter_names=list(varying_names),
            residuals=_residuals_i,
            final_cost=_per_cost_i,
            reduced_chi_squared=_per_chi2_i,
            success=bool(joint_result.success),
            message=str(joint_result.message),
            n_function_evals=int(joint_result.n_function_evals or 0),
            fitted_correlation=np.asarray(fitted_c2),
            metadata={
                "phi_angle": float(phi_angles[i]),
                "contrast": float(fitted_contrast[i]),
                "offset": float(fitted_offset[i]),
                "optimizer": "joint_fourier",
                "fourier_mode": fourier.config.mode,
                "fourier_order": fourier.order,
                "fourier_coeffs": fitted_fourier.tolist(),
                "fourier_n_coeffs": fourier.n_coeffs,
                "fourier_reduction": fourier.get_diagnostics()["reduction_ratio"],
                "n_angles_joint": n_phi,
                "wall_time_total": wall_time,
                **(
                    {"hierarchical_config": hierarchical_marker}
                    if hierarchical_marker is not None
                    else {}
                ),
                **({"gradient_monitor": monitor_summary} if monitor_summary else {}),
            },
        )
        results.append(result)

    logger.info(
        "Joint multi-angle fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles",
        joint_result.success,
        joint_result.final_cost or 0.0,
        joint_result.n_function_evals or 0,
        wall_time,
        n_phi,
    )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_global_optimization(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
) -> NLSQResult | None:
    """Attempt CMA-ES or multi-start if configured.

    Returns the result if a global method was selected, or ``None`` to
    fall through to local optimization.
    """
    # CMA-ES has highest priority
    if getattr(config, "enable_cmaes", False):
        if HAS_CMAES:
            logger.info("CMA-ES enabled, delegating to fit_nlsq_cmaes")
            return _fit_cmaes(model, c2_data, phi_angle, config, weights)
        logger.warning(
            "[CMA-ES] Enabled in config but not available (cma not installed). "
            "Install with: uv add cma. "
            "Falling back to multi-start or local optimization."
        )

    # Multi-start is second priority
    if getattr(config, "multistart", False):
        if HAS_MULTISTART:
            logger.info("Multi-start enabled, delegating to fit_nlsq_multistart")
            return _fit_multistart(
                model,
                c2_data,
                phi_angle,
                config,
                weights,
                use_nlsq_library,
            )
        logger.warning(
            "Multi-start enabled in config but multistart module not available. "
            "Falling back to local optimization."
        )

    return None


def _fit_cmaes(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
) -> NLSQResult:
    """Run CMA-ES global optimization with NLSQ warm-start and two-phase comparison.

    Implements fixes #1, #5, #6, #7 from homodyne parity:

    - **Phase 1 (Fix #1)**: Run local NLSQ refinement to get a warm-start point.
    - **Phase 2**: Run CMA-ES using the NLSQ result as initial guess.
    - **Phase 3 (Fix #7)**: Compare NLSQ vs CMA-ES by reduced chi-squared,
      keep the better result.
    - **Fix #5**: Classify result quality as good/marginal/poor.
    - **Fix #6**: Optionally apply anti-degeneracy penalty weights.
    """
    from heterodyne.optimization.nlsq.cmaes_wrapper import CMAESConfig
    from heterodyne.optimization.nlsq.validation.fit_quality import classify_fit_quality

    param_manager = model.param_manager

    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )
    t, q, dt = model.t, model.q, model.dt
    n_data = c2_jax.size
    contrast_val, offset_val = model.scaling.get_for_angle(0)

    def objective_fn(varying_params: np.ndarray) -> float:
        full_params = np.array(param_manager.get_full_values())
        for i, idx in enumerate(param_manager.varying_indices):
            full_params[idx] = varying_params[i]
        residuals = compute_residuals(
            jnp.asarray(full_params),
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast_val,
            offset_val,
        )
        return float(0.5 * jnp.sum(residuals**2))

    residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights, contrast_val, offset_val
    )

    # ------------------------------------------------------------------
    # Phase 1 (Fix #1): NLSQ warm-start
    # ------------------------------------------------------------------
    nlsq_result: NLSQResult | None = None
    cmaes_x0 = initial_varying

    try:
        logger.info("CMA-ES Phase 1: NLSQ warm-start refinement")
        nlsq_result = _fit_local(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library=config.use_nlsq_library,
        )
        if nlsq_result.success:
            cmaes_x0 = nlsq_result.parameters.copy()
            logger.info(
                "NLSQ warm-start succeeded: cost=%.6e, chi2_red=%.4f",
                nlsq_result.final_cost or float("inf"),
                nlsq_result.reduced_chi_squared or float("inf"),
            )
        else:
            logger.warning(
                "NLSQ warm-start failed (%s), using raw initial params for CMA-ES",
                nlsq_result.message,
            )
    except (ValueError, RuntimeError, ImportError) as e:
        logger.warning(
            "NLSQ warm-start raised %s: %s — proceeding with raw p0",
            type(e).__name__,
            e,
        )

    # Ensure model parameters are reset for CMA-ES (NLSQ may have modified them)
    model.set_params(param_manager.expand_varying_to_full(initial_varying))

    # ------------------------------------------------------------------
    # Phase 2: CMA-ES global optimization
    # ------------------------------------------------------------------
    logger.info("CMA-ES Phase 2: global search (warm-started)")

    cmaes_config = CMAESConfig(
        sigma0=config.cmaes_sigma0,
        popsize=config.cmaes_population_size,
        maxiter=config.cmaes_max_iterations,
        tolx=config.cmaes_tolx,
        tolfun=config.cmaes_tolfun,
        diagonal_filtering=getattr(config, "cmaes_diagonal_filtering", "none"),
    )

    cmaes_result = fit_with_cmaes(  # pyright: ignore[reportPossiblyUnbound]
        objective_fn=objective_fn,
        initial_params=cmaes_x0,
        bounds=(lower_bounds, upper_bounds),
        parameter_names=param_manager.varying_names,
        config=cmaes_config,
        residual_fn=residual_fn,
        n_data=n_data,
        anti_degeneracy=getattr(config, "cmaes_anti_degeneracy", False),
    )

    # ------------------------------------------------------------------
    # Phase 3 (Fix #7): Compare NLSQ vs CMA-ES, keep the better result
    # ------------------------------------------------------------------
    nlsq_cost = (
        float(nlsq_result.final_cost)
        if (nlsq_result and nlsq_result.success and nlsq_result.final_cost is not None)
        else float("inf")
    )
    cmaes_cost = (
        float(cmaes_result.final_cost)
        if (cmaes_result.success and cmaes_result.final_cost is not None)
        else float("inf")
    )

    if nlsq_cost <= cmaes_cost and nlsq_result is not None and nlsq_result.success:
        result = nlsq_result
        winner = "nlsq"
        logger.info(
            "Phase 3: NLSQ wins (cost=%.6e vs CMA-ES=%.6e)",
            nlsq_cost,
            cmaes_cost,
        )
    else:
        result = cmaes_result
        winner = "cmaes"
        logger.info(
            "Phase 3: CMA-ES wins (cost=%.6e vs NLSQ=%.6e)",
            cmaes_cost,
            nlsq_cost,
        )

    # ------------------------------------------------------------------
    # Post-fit: update model, classify quality (Fix #5)
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted), t, q, dt, phi_angle, contrast_val, offset_val
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # Apply same chi2 correction as _fit_local (DOF + σ² normalization)
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = (n_matrix - 1) * (n_matrix - 2)
        n_dof_valid = max(n_valid - len(param_manager.varying_names), 1)
        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_vals = c2_np[lag_mat >= n_matrix // 2]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0
        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            result.reduced_chi_squared = ssr / (sigma2_noise * n_dof_valid)

    quality_flag = classify_fit_quality(result.reduced_chi_squared)
    result.metadata["optimizer"] = "cmaes"
    result.metadata["cmaes_winner"] = winner
    result.metadata["cmaes_cost"] = cmaes_cost
    result.metadata["nlsq_warmstart_cost"] = nlsq_cost
    result.metadata["quality_flag"] = quality_flag

    _log_result(result)
    return result


def _fit_multistart(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
) -> NLSQResult:
    """Run multi-start optimization."""
    param_manager = model.param_manager
    varying_names = param_manager.varying_names

    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    contrast_val, offset_val = model.scaling.get_for_angle(0)

    # Build residual function
    residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights, contrast_val, offset_val
    )

    # Select adapter
    adapter = _select_adapter(varying_names, use_nlsq_library)

    # Build multistart config
    ms_config = MultiStartConfig(  # pyright: ignore[reportPossiblyUnbound]
        n_starts=getattr(config, "multistart_n", 10),
        seed=getattr(config, "multistart_seed", None),
    )
    optimizer = MultiStartOptimizer(adapter=adapter, config=ms_config)  # pyright: ignore[reportPossiblyUnbound]

    multi_result = optimizer.fit(
        residual_fn=residual_fn,
        initial_params=initial_varying,
        bounds=(lower_bounds, upper_bounds),
        config=config,
    )

    result = multi_result.to_nlsq_result()

    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            model.t,
            model.q,
            model.dt,
            phi_angle,
            contrast_val,
            offset_val,
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    result.metadata["optimizer"] = "multistart"
    _log_result(result)
    return result


def _fit_local(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
) -> NLSQResult:
    """Run local (single-start) optimization with adapter/wrapper fallback.

    Tries adapter first; on failure falls back to wrapper with progressive
    recovery.
    """
    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_varying = param_manager.n_varying

    logger.info("Fitting %d parameters: %s", n_varying, varying_names)

    # Memory-aware strategy selection
    if HAS_MEMORY:
        n_data_est = np.asarray(c2_data).size
        decision = select_nlsq_strategy(n_data_est, n_varying)  # pyright: ignore[reportPossiblyUnbound]
        if decision.strategy in (NLSQStrategy.LARGE, NLSQStrategy.STREAMING):  # pyright: ignore[reportPossiblyUnbound]
            logger.warning(
                "Estimated peak memory (%.2f GB) exceeds threshold (%.2f GB). "
                "Fit may fail with OOM.",
                decision.peak_memory_gb,
                decision.threshold_gb,
            )

    # Get initial values and bounds
    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    # Convert data to JAX arrays
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )

    if weights_jax is not None and weights_jax.shape != c2_jax.shape:
        raise ValueError(
            f"Weights shape {weights_jax.shape} does not match data shape {c2_jax.shape}"
        )

    # Capture constants
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices)
    n_data = c2_jax.size
    t, q, dt = model.t, model.q, model.dt

    # Per-angle scaling — fixed during local optimization (constant mode parity)
    contrast_val, offset_val = model.scaling.get_for_angle(0)

    # Build residual functions
    def jax_residual_fn(x: jnp.ndarray, *varying_params: float) -> jnp.ndarray:
        """Pure JAX residual function for nlsq tracing."""
        varying_array = jnp.array(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_array)
        return compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast_val,
            offset_val,
        )

    numpy_residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights, contrast_val, offset_val
    )

    # ------------------------------------------------------------------
    # Adapter → wrapper fallback chain
    # ------------------------------------------------------------------
    adapter_error: Exception | None = None
    fallback_occurred = False
    result: NLSQResult | None = None

    if use_nlsq_library and HAS_ADAPTERS:
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)  # pyright: ignore[reportPossiblyUnbound]
            logger.debug("Using NLSQAdapter (CurveFit class) for optimization")
            logger.debug("Attempting optimization with NLSQAdapter")

            result = adapter.fit_jax(
                jax_residual_fn=jax_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
                n_data=n_data,
            )

            if result.success:
                logger.info("NLSQAdapter optimization succeeded")
            else:
                raise RuntimeError(f"Adapter returned success=False: {result.message}")

        except (ValueError, RuntimeError, TypeError, ImportError, OSError) as e:
            adapter_error = e
            logger.warning("NLSQAdapter failed, falling back to wrapper: %s", e)
            fallback_occurred = True
            result = None

    # Wrapper fallback (or primary if use_nlsq_library=False)
    if result is None and HAS_WRAPPER:
        try:
            wrapper = NLSQWrapper(parameter_names=varying_names)  # pyright: ignore[reportPossiblyUnbound]
            logger.debug("Attempting optimization with NLSQWrapper")

            result = wrapper.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )

            if fallback_occurred:
                logger.info("NLSQWrapper fallback optimization succeeded")
            else:
                logger.info("NLSQWrapper optimization succeeded")

        except (ValueError, RuntimeError, TypeError, MemoryError) as wrapper_error:
            logger.error(
                "Both NLSQAdapter and NLSQWrapper failed: adapter=%s, wrapper=%s",
                adapter_error,
                wrapper_error,
            )
            result = NLSQResult(
                parameters=initial_varying,
                parameter_names=varying_names,
                success=False,
                message=f"All optimizers failed. Adapter: {adapter_error}; "
                f"Wrapper: {wrapper_error}",
            )

    if result is None:
        raise ImportError(
            "No NLSQ optimization backend available. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    # ------------------------------------------------------------------
    # Post-fit: compute fitted correlation, update model
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            phi_angle,
            contrast_val,
            offset_val,
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # ------------------------------------------------------------------
    # Post-fit: correct reduced chi-squared
    #
    # The raw chi2 from adapter.fit_jax is SSR / (N² − n_params), where
    # SSR = Σ r² over the full N×N residual vector.  Two corrections:
    #
    #   1. DOF: the N diagonal residuals are forced to 0 by the
    #      non_diagonal mask in compute_residuals — they should be
    #      excluded from the degrees-of-freedom count.
    #      n_valid = N*(N−1) instead of N².
    #
    #   2. σ² normalization: without dividing by measurement noise,
    #      chi2 = MSE ≪ 1 for normalized C2 data (C2 ~ 1, residuals ~ 5%).
    #      We estimate σ²_noise from the far-lag plateau of the C2 matrix
    #      (|t2−t1| ≥ N/2), where correlations have fully decayed and
    #      the remaining variance is photon-counting noise.
    #
    # chi2_corrected = SSR / (σ²_noise × n_dof_valid)  →  ~1 for good fits
    # ------------------------------------------------------------------
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = (n_matrix - 1) * (n_matrix - 2)  # exclude diagonal + t=0 boundary
        n_dof_valid = max(n_valid - n_varying, 1)

        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_mask = lag_mat >= n_matrix // 2  # diagonal (lag=0) not included
        far_vals = c2_np[far_mask]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            chi2_corrected = ssr / (sigma2_noise * n_dof_valid)
            logger.debug(
                "chi2 correction: σ²_noise=%.4e  n_valid=%d  SSR=%.4e  "
                "raw_chi2=%.4g → chi2_corrected=%.4f",
                sigma2_noise,
                n_valid,
                ssr,
                result.reduced_chi_squared or float("nan"),
                chi2_corrected,
            )
            result.reduced_chi_squared = chi2_corrected
        else:
            logger.warning(
                "chi2 noise estimate near-zero (σ²=%.2e); "
                "reporting uncorrected MSE chi2",
                sigma2_noise,
            )

    result.metadata["fallback_occurred"] = fallback_occurred
    if adapter_error is not None:
        result.metadata["adapter_error"] = str(adapter_error)
    result.metadata["optimizer"] = "local"
    result.metadata["wall_time_total"] = time.perf_counter() - t_start

    _log_result(result)
    return result


def _make_numpy_residual_fn(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    weights: np.ndarray | jnp.ndarray | None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> Any:
    """Create a numpy residual function closed over model/data.

    Returns a callable ``(varying_params: np.ndarray) -> np.ndarray``.

    Hot-path optimisation: ``fixed_values`` and ``varying_indices`` are
    pre-captured as JAX device arrays at construction time so each call
    only performs a single ``jnp.asarray`` (for the incoming numpy vector)
    and one ``jnp.ndarray.at[].set()`` scatter instead of a Python loop
    plus a full host copy.
    """
    param_manager = model.param_manager
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )
    t, q, dt = model.t, model.q, model.dt

    # Pre-capture as JAX device arrays — allocated once, reused every call.
    # NOTE: fixed_values snapshot is taken at construction time. Do not mutate
    # param_manager between construction and optimizer completion.
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    def residual_fn(varying_params: np.ndarray) -> np.ndarray:
        varying_jax = jnp.asarray(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_jax)
        residuals = compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast,
            offset,
        )
        return np.asarray(residuals)

    return residual_fn


def _select_adapter(
    varying_names: list[str],
    use_nlsq_library: bool,
) -> Any:
    """Select the appropriate adapter backend.

    Returns NLSQAdapter when the nlsq library is available and requested,
    otherwise falls back to NLSQWrapper (memory-tier routing).
    """
    if use_nlsq_library and HAS_ADAPTERS:
        try:
            return NLSQAdapter(parameter_names=varying_names)  # pyright: ignore[reportPossiblyUnbound]
        except ImportError:
            logger.warning("nlsq library not available, falling back to NLSQWrapper")
    if HAS_WRAPPER:
        return NLSQWrapper(parameter_names=varying_names)  # pyright: ignore[reportPossiblyUnbound]
    raise ImportError("No NLSQ adapter available")


def _log_result(result: NLSQResult) -> None:
    """Log optimization results summary."""
    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    status = "SUCCESS" if result.success else "FAILED"
    logger.info("Status: %s", status)
    logger.info("Message: %s", result.message)

    if result.final_cost is not None:
        logger.info("Final cost: %.6e", result.final_cost)
    if result.reduced_chi_squared is not None:
        logger.info("Reduced χ²: %.4f", result.reduced_chi_squared)
    if result.wall_time_seconds is not None:
        logger.info("Wall time: %.2f s", result.wall_time_seconds)

    if result.success:
        n_params = len(result.parameters)
        logger.info("Fitted parameters:")
        logger.info("  Physical parameters:")
        for name, val in zip(result.parameter_names, result.parameters, strict=True):
            unc_val = result.get_uncertainty(name)
            if unc_val is not None:
                logger.info("    %s: %.6g ± %.3g", name, val, unc_val)
            else:
                logger.info("    %s: %.6g", name, val)
        logger.info(
            "  Total parameters: %d physical + 2 scaling = %d",
            n_params,
            n_params + 2,
        )

    logger.info("=" * 60)
