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

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.core.heterodyne_model import HeterodyneModel

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — gated for graceful degradation
# ---------------------------------------------------------------------------

try:
    from heterodyne.optimization.nlsq.adapter import NLSQAdapter, NLSQWrapper

    HAS_ADAPTERS = True
    HAS_WRAPPER = True
except ImportError:
    HAS_ADAPTERS = False
    HAS_WRAPPER = False

try:
    from heterodyne.optimization.nlsq.multistart import (
        MultiStartConfig,
        MultiStartOptimizer,
    )

    HAS_MULTISTART = True
except ImportError:
    HAS_MULTISTART = False

try:
    from heterodyne.optimization.nlsq.cmaes_wrapper import (
        CMAES_AVAILABLE,
        fit_with_cmaes,
    )

    HAS_CMAES = CMAES_AVAILABLE
except ImportError:
    HAS_CMAES = False

try:
    from heterodyne.optimization.nlsq.memory import NLSQStrategy, select_nlsq_strategy

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
    # Global optimization selection (CMA-ES → multi-start → local)
    # ------------------------------------------------------------------
    if not _skip_global_selection:
        global_result = _try_global_optimization(
            model, c2_data, phi_angle, config, weights, use_nlsq_library,
        )
        if global_result is not None:
            return global_result

    # ------------------------------------------------------------------
    # Local optimization
    # ------------------------------------------------------------------
    return _fit_local(model, c2_data, phi_angle, config, weights, use_nlsq_library)


def fit_nlsq_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float] | np.ndarray,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> list[NLSQResult]:
    """Fit model to correlation data at multiple phi angles.

    Two modes of operation controlled by ``config.per_angle_mode``:

    **Joint fit** (``"fourier"``, ``"independent"``, or ``"auto"``
    with multiple angles):
        All angles are fit simultaneously in a single optimization.
        In ``"fourier"`` mode, the optimizer vector is
        ``[physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]``,
        where the Fourier basis constrains smooth angular variation.
        In ``"independent"`` mode, each angle has its own contrast/offset
        (2*n_phi scaling parameters), all optimized jointly.

    **Sequential mode** (single angle or fallback):
        Angles are fit one at a time with warm-starting.

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
    # Determine whether to use joint Fourier fit
    # ------------------------------------------------------------------
    use_joint = False
    if config is not None and len(phi_angles) > 1:
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
            use_joint = fourier.use_fourier or (
                config.per_angle_mode == "independent" and len(phi_angles) > 1
            )
        except ImportError:
            logger.warning(
                "fourier_reparam not available, falling back to sequential fits"
            )

    if use_joint:
        return _fit_joint_multi_phi(
            model, c2_data, phi_angles, config, weights, fourier,
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

    # Pre-convert data to JAX arrays
    t, q, dt = model.t, model.q, model.dt
    c2_data_list = [jnp.asarray(c2_data[i], dtype=jnp.float64) for i in range(n_phi)]
    weights_list = []
    for i in range(n_phi):
        if weights is not None and weights.ndim == 3:
            weights_list.append(jnp.asarray(weights[i], dtype=jnp.float64))
        elif weights is not None:
            weights_list.append(jnp.asarray(weights, dtype=jnp.float64))
        else:
            weights_list.append(None)

    fixed_values = param_manager.get_full_values().copy()
    varying_indices = param_manager.varying_indices

    def joint_residual_fn(x: np.ndarray) -> np.ndarray:
        """Compute concatenated residuals across all angles."""
        # Split combined vector
        physics_varying = x[:n_physics_varying]
        fourier_coeffs = x[n_physics_varying:]

        # Reconstruct full physics parameter array
        full_params = fixed_values.copy()
        for j, idx in enumerate(varying_indices):
            full_params[idx] = physics_varying[j]
        full_jax = jnp.asarray(full_params)

        # Convert Fourier coefficients → per-angle contrast/offset
        contrast_arr, offset_arr = fourier.fourier_to_per_angle(fourier_coeffs)

        # Compute residuals per angle and concatenate
        all_residuals = []
        for i in range(n_phi):
            residuals_i = compute_residuals(
                full_jax, t, q, dt, float(phi_angles[i]),
                c2_data_list[i], weights_list[i],
                contrast=float(contrast_arr[i]),
                offset=float(offset_arr[i]),
            )
            all_residuals.append(np.asarray(residuals_i))

        return np.concatenate(all_residuals)

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
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)
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
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)
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
            jnp.asarray(full_fitted), t, q, dt,
            float(phi_angles[i]),
            contrast=float(fitted_contrast[i]),
            offset=float(fitted_offset[i]),
        )

        result = NLSQResult(
            parameters=fitted_physics.copy(),
            parameter_names=list(varying_names),
            residuals=np.asarray(
                compute_residuals(
                    jnp.asarray(full_fitted), t, q, dt, float(phi_angles[i]),
                    c2_data_list[i], weights_list[i],
                    contrast=float(fitted_contrast[i]),
                    offset=float(fitted_offset[i]),
                )
            ),
            final_cost=joint_result.final_cost,
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
            logger.info("CMA-ES enabled, delegating to fit_with_cmaes")
            return _fit_cmaes(model, c2_data, phi_angle, config, weights)
        logger.warning(
            "CMA-ES enabled in config but not available (cma not installed). "
            "Install with: uv add cma. Falling back."
        )

    # Multi-start is second priority
    if getattr(config, "multistart", False):
        if HAS_MULTISTART:
            logger.info("Multi-start enabled, delegating to multi-start optimizer")
            return _fit_multistart(
                model, c2_data, phi_angle, config, weights, use_nlsq_library,
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
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    t, q, dt = model.t, model.q, model.dt
    n_data = c2_jax.size

    def objective_fn(varying_params: np.ndarray) -> float:
        full_params = np.array(param_manager.get_full_values())
        for i, idx in enumerate(param_manager.varying_indices):
            full_params[idx] = varying_params[i]
        residuals = compute_residuals(
            jnp.asarray(full_params), t, q, dt, phi_angle, c2_jax, weights_jax,
        )
        return float(0.5 * jnp.sum(residuals ** 2))

    residual_fn = _make_numpy_residual_fn(model, c2_data, phi_angle, weights)

    # ------------------------------------------------------------------
    # Phase 1 (Fix #1): NLSQ warm-start
    # ------------------------------------------------------------------
    nlsq_result: NLSQResult | None = None
    cmaes_x0 = initial_varying

    try:
        logger.info("CMA-ES Phase 1: NLSQ warm-start refinement")
        nlsq_result = _fit_local(
            model, c2_data, phi_angle, config, weights,
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
        logger.warning("NLSQ warm-start raised %s: %s — proceeding with raw p0", type(e).__name__, e)

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

    cmaes_result = fit_with_cmaes(
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
    nlsq_cost = float(nlsq_result.final_cost) if (nlsq_result and nlsq_result.success and nlsq_result.final_cost is not None) else float("inf")
    cmaes_cost = float(cmaes_result.final_cost) if (cmaes_result.success and cmaes_result.final_cost is not None) else float("inf")

    if nlsq_cost <= cmaes_cost and nlsq_result is not None and nlsq_result.success:
        result = nlsq_result
        winner = "nlsq"
        logger.info(
            "Phase 3: NLSQ wins (cost=%.6e vs CMA-ES=%.6e)", nlsq_cost, cmaes_cost,
        )
    else:
        result = cmaes_result
        winner = "cmaes"
        logger.info(
            "Phase 3: CMA-ES wins (cost=%.6e vs NLSQ=%.6e)", cmaes_cost, nlsq_cost,
        )

    # ------------------------------------------------------------------
    # Post-fit: update model, classify quality (Fix #5)
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(jnp.asarray(full_fitted), t, q, dt, phi_angle)
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

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

    # Build residual function
    residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights,
    )

    # Select adapter
    adapter = _select_adapter(varying_names, use_nlsq_library)

    # Build multistart config
    ms_config = MultiStartConfig(
        n_starts=getattr(config, "multistart_n", 10),
        seed=getattr(config, "multistart_seed", None),
    )
    optimizer = MultiStartOptimizer(adapter=adapter, config=ms_config)

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
            jnp.asarray(full_fitted), model.t, model.q, model.dt, phi_angle,
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
        decision = select_nlsq_strategy(n_data_est, n_varying)
        if decision.strategy in (NLSQStrategy.LARGE, NLSQStrategy.STREAMING):
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
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None

    if weights_jax is not None and weights_jax.shape != c2_jax.shape:
        raise ValueError(
            f"Weights shape {weights_jax.shape} does not match data shape {c2_jax.shape}"
        )

    # Capture constants
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices)
    n_data = c2_jax.size
    t, q, dt = model.t, model.q, model.dt

    # Build residual functions
    def jax_residual_fn(x: jnp.ndarray, *varying_params: float) -> jnp.ndarray:
        """Pure JAX residual function for nlsq tracing."""
        varying_array = jnp.array(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_array)
        return compute_residuals(
            full_params, t, q, dt, phi_angle, c2_jax, weights_jax,
        )

    numpy_residual_fn = _make_numpy_residual_fn(model, c2_data, phi_angle, weights)

    # ------------------------------------------------------------------
    # Adapter → wrapper fallback chain
    # ------------------------------------------------------------------
    adapter_error: Exception | None = None
    fallback_occurred = False
    result: NLSQResult | None = None

    if use_nlsq_library and HAS_ADAPTERS:
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQAdapter (JAX)")

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
            wrapper = NLSQWrapper(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQWrapper")

            result = wrapper.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )

            if fallback_occurred:
                logger.info("NLSQWrapper fallback succeeded")
            else:
                logger.info("NLSQWrapper optimization succeeded")

        except (ValueError, RuntimeError, TypeError, MemoryError) as wrapper_error:
            logger.error(
                "Both adapter and wrapper failed: adapter=%s, wrapper=%s",
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
            jnp.asarray(full_fitted), t, q, dt, phi_angle,
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

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
) -> Any:
    """Create a numpy residual function closed over model/data.

    Returns a callable ``(varying_params: np.ndarray) -> np.ndarray``.
    """
    param_manager = model.param_manager
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    t, q, dt = model.t, model.q, model.dt

    def residual_fn(varying_params: np.ndarray) -> np.ndarray:
        full_params = param_manager.get_full_values().copy()
        for i, idx in enumerate(param_manager.varying_indices):
            full_params[idx] = varying_params[i]
        residuals = compute_residuals(
            jnp.asarray(full_params), t, q, dt, phi_angle, c2_jax, weights_jax,
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
            return NLSQAdapter(parameter_names=varying_names)
        except ImportError:
            logger.warning("nlsq library not available, falling back to NLSQWrapper")
    if HAS_WRAPPER:
        return NLSQWrapper(parameter_names=varying_names)
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
        for name, val in zip(result.parameter_names, result.parameters, strict=True):
            unc_val = result.get_uncertainty(name)
            if unc_val is not None:
                logger.info("  %s: %.6g ± %.3g", name, val, unc_val)
            else:
                logger.info("  %s: %.6g", name, val)

    logger.info("=" * 60)
