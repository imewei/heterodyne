"""Configuration for NLSQ optimization in the heterodyne analysis pipeline.

This module defines the full configuration hierarchy for non-linear least squares
fitting of heterodyne XPCS correlation functions:

- ``HybridRecoveryConfig``  — progressive retry / fallback parameters
- ``NLSQValidationConfig``  — post-fit validation thresholds
- ``NLSQConfig``             — master configuration dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Safe type-conversion utilities
# ---------------------------------------------------------------------------

_SENTINEL = object()


def safe_float(value: Any, default: float) -> float:
    """Convert *value* to float, returning *default* on failure.

    Args:
        value: Arbitrary input that should be numeric.
        default: Fallback value when conversion fails.

    Returns:
        ``float(value)`` on success, *default* otherwise.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "safe_float: could not convert %r to float, using default %s",
            value,
            default,
        )
        return default


def safe_int(value: Any, default: int) -> int:
    """Convert *value* to int, returning *default* on failure.

    Args:
        value: Arbitrary input that should be integral.
        default: Fallback value when conversion fails.

    Returns:
        ``int(value)`` on success, *default* otherwise.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "safe_int: could not convert %r to int, using default %s",
            value,
            default,
        )
        return default


# ---------------------------------------------------------------------------
# HybridRecoveryConfig
# ---------------------------------------------------------------------------

_VALID_WORKFLOWS: frozenset[str] = frozenset({"auto", "auto_global", "hpc"})
_VALID_GOALS: frozenset[str] = frozenset(
    {"fast", "robust", "quality", "memory_efficient"}
)
_VALID_ANALYSIS_MODES: frozenset[str] = frozenset(
    {"static_ref", "static_both", "two_component"}
)
_VALID_NLSQ_STABILITY: frozenset[str] = frozenset({"auto", "check", "off"})


@dataclass
class HybridRecoveryConfig:
    """Progressive retry / fallback parameters for NLSQ recovery.

    When a fit fails to converge, the optimizer retries with progressively
    more aggressive regularisation and a smaller trust region.  Each attempt
    *k* (0-based) applies the following scaling to the baseline settings:

    - learning rate  : ``lr_decay  ** k``
    - regularisation : ``lambda_growth ** k``
    - trust radius   : ``trust_decay ** k``

    Attributes:
        max_retries: Maximum number of recovery attempts before giving up.
        lr_decay: Multiplicative factor applied to the learning rate per retry
            (< 1 shrinks the effective step size).
        lambda_growth: Multiplicative factor applied to the regularisation
            strength per retry (> 1 increases damping).
        trust_decay: Multiplicative factor applied to the trust-region radius
            per retry (< 1 tightens the constraint).
        perturb_scale: Standard deviation of the Gaussian perturbation added
            to the starting parameters before each retry, expressed as a
            fraction of the parameter range.
    """

    max_retries: int = 3
    lr_decay: float = 0.5
    lambda_growth: float = 10.0
    trust_decay: float = 0.5
    perturb_scale: float = 0.1
    log_retries: bool = True

    def get_retry_settings(self, attempt: int) -> dict[str, float]:
        """Return scaled optimiser settings for a given retry attempt.

        Args:
            attempt: 0-based retry index.  ``attempt=0`` returns the
                unscaled baseline (scale factor = 1).

        Returns:
            Dictionary with keys ``"lr_scale"``, ``"lambda_scale"``, and
            ``"trust_radius_scale"``, each a multiplicative factor to apply
            to the corresponding optimiser hyperparameter.
        """
        if attempt < 0:
            raise ValueError(f"attempt must be >= 0, got {attempt}")
        return {
            "lr_scale": self.lr_decay**attempt,
            "lambda_scale": self.lambda_growth**attempt,
            "trust_radius_scale": self.trust_decay**attempt,
        }


# ---------------------------------------------------------------------------
# NLSQValidationConfig
# ---------------------------------------------------------------------------


@dataclass
class NLSQValidationConfig:
    """Thresholds used when validating post-fit quality metrics.

    Attributes:
        chi2_warn_low: Reduced chi-squared below this value triggers a
            warning (possible over-fitting or under-estimated errors).
        chi2_warn_high: Reduced chi-squared above this value triggers a
            warning (possible under-fitting or under-estimated model).
        chi2_fail_high: Reduced chi-squared above this value is treated as a
            hard failure.
        max_relative_uncertainty: Maximum acceptable relative uncertainty
            (sigma / ``|param|``) for any fitted parameter.  Value of 1.0 means
            100 %.
        correlation_warn: Off-diagonal correlation coefficient magnitude above
            this threshold triggers a collinearity warning.
    """

    # Reduced chi-squared thresholds
    chi2_warn_low: float = 0.5
    chi2_warn_high: float = 2.0
    chi2_fail_high: float = 10.0

    # Uncertainty validation
    max_relative_uncertainty: float = 1.0  # 100 %

    # Correlation threshold for parameters
    correlation_warn: float = 0.95


# ---------------------------------------------------------------------------
# NLSQConfig
# ---------------------------------------------------------------------------


@dataclass
class NLSQConfig:
    """Master configuration for NLSQ fitting of heterodyne XPCS data.

    The heterodyne model has 14 parameters organised into two-component
    (signal + background) correlation functions.  This configuration covers
    the full pipeline: solver hyperparameters, multi-start, streaming /
    chunking, recovery on failure, and post-fit diagnostics.

    Attributes:
        max_iterations: Maximum number of optimiser iterations per fit.
        tolerance: Convergence tolerance for the cost function.
        method: Trust-region algorithm variant passed to the
            ``nlsq CurveFit optimizer``.  Note that ``dogbox`` is coerced
            to ``trf`` by the strategy layer.
        multistart: Whether to run multi-start optimisation to avoid local
            minima.
        multistart_n: Number of random starting points when *multistart* is
            enabled.
        verbose: Verbosity level forwarded to the solver (0 = silent,
            1 = summary, 2 = detailed).
        use_jac: Whether to supply an analytic Jacobian to the solver.
        x_scale: Parameter scaling strategy.  ``"jac"`` uses the Jacobian
            diagonal; a list of floats provides explicit per-parameter scales.
        ftol: Relative tolerance on the cost function change.
        xtol: Relative tolerance on the parameter step norm.
        gtol: Absolute tolerance on the projected gradient norm.
        loss: Robust loss function kernel.
        diff_step: Finite-difference step size.  ``None`` selects the
            solver default.
        max_nfev: Hard cap on function evaluations.  ``None`` is unlimited.
        chunk_size: Number of q-points per processing chunk.  ``None`` means
            auto-select based on available memory.

        workflow: High-level workflow preset.  One of ``"auto"``,
            ``"auto_global"``, ``"hpc"``.
        goal: Optimisation goal preset controlling the balance between speed,
            robustness, and solution quality.  One of ``"fast"``,
            ``"robust"``, ``"quality"``, ``"memory_efficient"``.

        enable_streaming: Process data in a streaming fashion (chunk-by-chunk)
            rather than loading all q-points at once.
        streaming_chunk_size: Number of q-points per streaming chunk when
            *enable_streaming* is ``True``.
        enable_stratified: Use stratified sampling across q-point subsets.
        target_chunk_size: Target number of data points per stratified chunk.

        enable_recovery: Automatically retry failed fits with more aggressive
            regularisation (see *recovery_config*).
        max_recovery_attempts: Maximum retries before a fit is declared failed.
        recovery_config: Per-retry scaling parameters.

        enable_diagnostics: Emit structured convergence / quality diagnostics
            after each fit.
        enable_anti_degeneracy: Apply anti-degeneracy constraints to prevent
            parameter collapse (e.g. two identical relaxation modes).

        x_scale_map: Per-parameter scale overrides keyed by parameter name.
            Entries here are merged into (and override) the default
            Jacobian-based scaling.
        loss_weights: Per-data-point loss weights.  ``None`` uses uniform
            weighting.
        loss_scale: Global scale factor applied to the loss function value
            before passing to the solver.
        tr_solver: Trust-region sub-problem solver override (``"exact"``,
            ``"lsmr"``, or ``None`` for solver default).
        step_bound: Upper bound on the step norm relative to the trust radius.
            ``0.0`` defers to the solver default.

        use_nlsq_library: Prefer the ``nlsq`` library over the scipy fallback.
        n_params: Number of model parameters.  Fixed at 14 for heterodyne.
        analysis_mode: Which physical model variant to use.  One of
            ``"static_ref"`` (reference beam treated as static background),
            ``"static_both"`` (both beams treated as static),
            ``"two_component"`` (full two-component model, default).

        validation: Post-fit validation thresholds.
    """

    # ------------------------------------------------------------------
    # Existing / core solver fields
    # ------------------------------------------------------------------

    max_iterations: int = 1000
    tolerance: float = 1e-8
    method: Literal["trf", "lm", "dogbox"] = "trf"
    multistart: bool = False
    multistart_n: int = 10
    verbose: int = 1
    use_jac: bool = True
    x_scale: str | list[float] = "jac"
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    trust_region_scale: float = 1.0

    # Advanced solver options
    diff_step: float | None = None
    max_nfev: int | None = None

    # Memory management
    chunk_size: int | None = None  # None for auto

    # ------------------------------------------------------------------
    # Workflow / goal presets
    # ------------------------------------------------------------------

    workflow: str = "auto"
    goal: str = "robust"

    # ------------------------------------------------------------------
    # Streaming and stratified sampling
    # ------------------------------------------------------------------

    enable_streaming: bool = False
    streaming_chunk_size: int = 50000
    enable_stratified: bool = False
    target_chunk_size: int = 10000

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    enable_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_config: HybridRecoveryConfig = field(default_factory=HybridRecoveryConfig)

    # ------------------------------------------------------------------
    # Diagnostics and anti-degeneracy
    # ------------------------------------------------------------------

    enable_diagnostics: bool = True
    enable_anti_degeneracy: bool = True

    # ------------------------------------------------------------------
    # Loss and scaling overrides
    # ------------------------------------------------------------------

    x_scale_map: dict[str, float] = field(default_factory=dict)
    loss_weights: list[float] | None = None
    loss_scale: float = 1.0
    tr_solver: str | None = None
    step_bound: float = 0.0

    # ------------------------------------------------------------------
    # Fourier reparameterization for per-angle scaling
    # ------------------------------------------------------------------

    per_angle_mode: Literal[
        "individual", "constant", "fourier", "auto", "independent"
    ] = "auto"
    fourier_order: int = 2
    fourier_auto_threshold: int = 6

    # ------------------------------------------------------------------
    # Hierarchical optimization
    # ------------------------------------------------------------------

    enable_hierarchical: bool = False
    hierarchical_max_outer_iterations: int = 20
    hierarchical_inner_tolerance: float = 1e-6
    hierarchical_outer_tolerance: float = 1e-4
    # Homodyne-parity hierarchical fields
    hierarchical_physical_max_iterations: int = 100
    hierarchical_per_angle_max_iterations: int = 50

    # ------------------------------------------------------------------
    # Adaptive regularization
    # ------------------------------------------------------------------

    regularization_mode: Literal["none", "tikhonov", "adaptive"] = "none"
    group_variance_lambda: float = 0.01
    regularization_target_cv: float = 0.5
    # Homodyne-parity regularization fields
    regularization_target_contribution: float = 0.10  # 10% of MSE contribution
    regularization_max_cv: float = 0.20  # 20% max variation
    regularization_auto_tune_lambda: bool = True

    # ------------------------------------------------------------------
    # Gradient collapse detection
    # ------------------------------------------------------------------

    enable_gradient_monitoring: bool = False
    gradient_ratio_threshold: float = 100.0
    gradient_consecutive_triggers: int = 3
    # Homodyne-parity gradient collapse response field
    gradient_collapse_response: str = (
        "hierarchical"  # "warn", "hierarchical", "reset", "abort"
    )

    # ------------------------------------------------------------------
    # CMA-ES global search (legacy heterodyne fields)
    # ------------------------------------------------------------------

    enable_cmaes: bool = False
    cmaes_sigma0: float = 0.3
    cmaes_max_iterations: int = 1000
    cmaes_population_size: int | None = None
    cmaes_tolx: float = 1e-6
    cmaes_tolfun: float = 1e-8
    cmaes_diagonal_filtering: str = "remove"
    cmaes_anti_degeneracy: bool = False
    cmaes_warmstart_auto_skip: bool = True
    cmaes_warmstart_skip_threshold: float = 5.0
    cmaes_restart_strategy: str = "bipop"
    cmaes_max_restarts: int = 9

    # ------------------------------------------------------------------
    # CMA-ES global search (homodyne-parity fields, NLSQ v0.6.4+)
    # ------------------------------------------------------------------

    cmaes_preset: str = "cmaes"  # "cmaes-fast", "cmaes", "cmaes-global"
    cmaes_max_generations: int | None = None  # None = use preset + adaptive scaling
    cmaes_popsize: int | None = None  # Population size (None = auto)
    cmaes_sigma: float = 0.5  # Initial step size (fraction of search range)
    cmaes_sigma_warmstart: float = 0.05  # Reduced sigma for warm-start mode
    cmaes_tol_fun: float = 1e-8  # Function value tolerance for convergence
    cmaes_tol_x: float = 1e-8  # Parameter tolerance for convergence
    cmaes_population_batch_size: int | None = None  # Memory batching (None = auto)
    cmaes_data_chunk_size: int | None = None  # Data streaming (None = auto)
    cmaes_refine_with_nlsq: bool = True  # Refine CMA-ES solution with NLSQ TRF
    cmaes_auto_select: bool = (
        True  # Auto-select CMA-ES vs multi-start based on scale ratio
    )
    cmaes_scale_threshold: float = 1000.0  # Scale ratio threshold for auto-selection
    cmaes_memory_limit_gb: float = 8.0  # Memory limit for auto-configuration
    # Post-CMA-ES NLSQ TRF Refinement
    cmaes_refinement_workflow: str = "auto"  # "auto", "standard", "streaming"
    cmaes_refinement_ftol: float = 1e-10
    cmaes_refinement_xtol: float = 1e-10
    cmaes_refinement_gtol: float = 1e-10
    cmaes_refinement_max_nfev: int = 500
    cmaes_refinement_loss: str = "linear"  # "linear", "soft_l1", "huber"
    # CMA-ES Parameter Normalization
    cmaes_normalize: bool = True  # Enable bounds-based normalization
    cmaes_normalization_epsilon: float = 1e-12  # Prevent division by zero

    # ------------------------------------------------------------------
    # Fit Quality Validation (homodyne-parity fields, v2.16.0)
    # ------------------------------------------------------------------

    enable_quality_validation: bool = True  # Enable post-fit quality checks
    quality_reduced_chi_squared_threshold: float = 10.0  # Warn if χ²_red > threshold
    quality_warn_on_max_restarts: bool = True  # Warn if CMA-ES didn't converge
    quality_warn_on_bounds_hit: bool = True  # Warn if physical params at bounds
    quality_warn_on_convergence_failure: bool = True  # Warn if optimization failed
    quality_bounds_tolerance: float = 1e-9  # Tolerance for "at bounds" detection

    # ------------------------------------------------------------------
    # Progress and logging settings
    # ------------------------------------------------------------------

    enable_progress_bar: bool = True  # Show tqdm progress bar during fitting
    log_iteration_interval: int = 10  # Log every N iterations (for verbose >= 2)

    # ------------------------------------------------------------------
    # Hybrid streaming optimizer (legacy heterodyne fields)
    # ------------------------------------------------------------------

    hybrid_enable: bool = False
    hybrid_warmup_fraction: float = 0.1
    hybrid_normalization: bool = True
    hybrid_method: Literal["lbfgs", "gauss_newton"] = "gauss_newton"
    hybrid_lbfgs_memory: int = 10
    hybrid_convergence_window: int = 5
    hybrid_convergence_threshold: float = 1e-6
    hybrid_max_phases: int = 4

    # ------------------------------------------------------------------
    # Hybrid streaming optimizer (homodyne-parity fields, v2.6.0+)
    # ------------------------------------------------------------------

    enable_hybrid_streaming: bool = True
    hybrid_normalize: bool = True
    hybrid_normalization_strategy: str = "auto"  # 'auto', 'bounds', 'p0', 'none'
    hybrid_warmup_iterations: int = 200
    hybrid_max_warmup_iterations: int = 500
    hybrid_warmup_learning_rate: float = 0.001
    hybrid_gauss_newton_max_iterations: int = 100
    hybrid_gauss_newton_tol: float = 1e-8
    hybrid_chunk_size: int = 10000
    hybrid_trust_region_initial: float = 1.0
    hybrid_regularization_factor: float = 1e-10
    hybrid_enable_checkpoints: bool = True
    hybrid_checkpoint_frequency: int = 100
    hybrid_validate_numerics: bool = True

    # 4-Layer Defense Strategy for L-BFGS Warmup (v2.8.0 / NLSQ 0.3.6)
    # Layer 1: Warm Start Detection
    hybrid_enable_warm_start_detection: bool = True
    hybrid_warm_start_threshold: float = 0.01  # Skip if loss/variance < this
    # Layer 2: Adaptive Learning Rate
    hybrid_enable_adaptive_warmup_lr: bool = True
    hybrid_warmup_lr_refinement: float = 1e-6  # LR for good starts
    hybrid_warmup_lr_careful: float = 1e-5  # LR for moderate starts
    # Layer 3: Cost-Increase Guard
    hybrid_enable_cost_guard: bool = True
    hybrid_cost_increase_tolerance: float = 0.05  # Abort if loss increases >5%
    # Layer 4: Step Clipping
    hybrid_enable_step_clipping: bool = True
    hybrid_max_warmup_step_size: float = 0.1  # Max step in normalized units

    # ------------------------------------------------------------------
    # Multi-start extensions (legacy heterodyne fields)
    # ------------------------------------------------------------------

    sampling_strategy: Literal["lhs", "sobol", "random"] = "lhs"
    screen_keep_fraction: float = 0.5
    refine_top_k: int = 3

    # ------------------------------------------------------------------
    # Multi-start optimization settings (homodyne-parity fields, v2.6.0)
    # ------------------------------------------------------------------

    enable_multi_start: bool = False  # Default OFF - user opt-in
    multi_start_n_starts: int = 10
    multi_start_seed: int = 42
    multi_start_sampling_strategy: str = (
        "latin_hypercube"  # 'latin_hypercube' or 'random'
    )
    multi_start_n_workers: int = 0  # 0 = auto (min of n_starts, cpu_count)
    multi_start_use_screening: bool = True
    multi_start_screen_keep_fraction: float = 0.5
    multi_start_refine_top_k: int = 3
    multi_start_refinement_ftol: float = 1e-12
    multi_start_degeneracy_threshold: float = 0.1

    # ------------------------------------------------------------------
    # Scaling threshold
    # ------------------------------------------------------------------

    constant_scaling_threshold: int = 3

    # ------------------------------------------------------------------
    # Backend and model identity
    # ------------------------------------------------------------------

    use_nlsq_library: bool = True
    n_params: int = 14  # heterodyne: 14 parameters
    analysis_mode: str = "two_component"

    # ------------------------------------------------------------------
    # NLSQ package integration (mirrors homodyne wrapper.py)
    # ------------------------------------------------------------------

    nlsq_stability: str = "auto"  # 'auto', 'check', or 'off'
    nlsq_rescale_data: bool = False  # xdata is indices, not physical
    nlsq_x_scale: str | np.ndarray = "jac"  # trust-region scaling
    nlsq_memory_fraction: float = 0.75  # fraction of RAM for NLSQ
    nlsq_memory_fallback_gb: float = 16.0  # fallback if detection fails

    # ------------------------------------------------------------------
    # Post-fit validation
    # ------------------------------------------------------------------

    validation: NLSQValidationConfig = field(default_factory=NLSQValidationConfig)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate invariants that must hold immediately after construction."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.multistart_n < 1:
            raise ValueError("multistart_n must be >= 1")
        if self.streaming_chunk_size < 1:
            raise ValueError("streaming_chunk_size must be >= 1")
        if self.target_chunk_size < 1:
            raise ValueError("target_chunk_size must be >= 1")
        if self.max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be >= 0")
        if self.loss_scale <= 0:
            raise ValueError("loss_scale must be positive")
        if self.hierarchical_max_outer_iterations < 1:
            raise ValueError("hierarchical_max_outer_iterations must be >= 1")
        if self.gradient_consecutive_triggers < 1:
            raise ValueError("gradient_consecutive_triggers must be >= 1")
        if self.cmaes_sigma0 <= 0:
            raise ValueError("cmaes_sigma0 must be > 0")
        if self.cmaes_diagonal_filtering not in ("remove", "none"):
            raise ValueError(
                f"cmaes_diagonal_filtering must be 'remove' or 'none', "
                f"got {self.cmaes_diagonal_filtering!r}"
            )
        if self.cmaes_warmstart_skip_threshold <= 0:
            raise ValueError("cmaes_warmstart_skip_threshold must be > 0")
        if self.cmaes_restart_strategy not in ("bipop", "none"):
            raise ValueError(
                f"cmaes_restart_strategy must be 'bipop' or 'none', "
                f"got {self.cmaes_restart_strategy!r}"
            )
        if self.cmaes_max_restarts < 0:
            raise ValueError("cmaes_max_restarts must be >= 0")
        if not (0 < self.hybrid_warmup_fraction < 1):
            raise ValueError("hybrid_warmup_fraction must be in (0, 1)")
        if not (0 < self.screen_keep_fraction <= 1):
            raise ValueError("screen_keep_fraction must be in (0, 1]")
        if self.refine_top_k < 1:
            raise ValueError("refine_top_k must be >= 1")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of configuration error strings.

        An empty list means the configuration is consistent.  Callers should
        treat a non-empty list as a hard error before launching a fit.

        Returns:
            List of human-readable error strings, one per violation found.
        """
        errors: list[str] = []

        if self.workflow not in _VALID_WORKFLOWS:
            errors.append(
                f"workflow={self.workflow!r} is not valid; "
                f"must be one of {sorted(_VALID_WORKFLOWS)}"
            )

        if self.goal not in _VALID_GOALS:
            errors.append(
                f"goal={self.goal!r} is not valid; "
                f"must be one of {sorted(_VALID_GOALS)}"
            )

        if self.tolerance <= 0:
            errors.append(f"tolerance={self.tolerance} must be > 0")

        if self.streaming_chunk_size <= 0:
            errors.append(
                f"streaming_chunk_size={self.streaming_chunk_size} must be > 0"
            )

        if self.analysis_mode not in _VALID_ANALYSIS_MODES:
            errors.append(
                f"analysis_mode={self.analysis_mode!r} is not valid; "
                f"must be one of {sorted(_VALID_ANALYSIS_MODES)}"
            )

        valid_per_angle_modes = ("independent", "fourier", "auto")
        if self.per_angle_mode not in valid_per_angle_modes:
            errors.append(
                f"per_angle_mode={self.per_angle_mode!r} is not valid; "
                f"must be one of {valid_per_angle_modes}"
            )
        if self.fourier_order < 1:
            errors.append(f"fourier_order={self.fourier_order} must be >= 1")
        if self.fourier_auto_threshold < 1:
            errors.append(
                f"fourier_auto_threshold={self.fourier_auto_threshold} must be >= 1"
            )

        valid_regularization_modes = ("none", "tikhonov", "adaptive")
        if self.regularization_mode not in valid_regularization_modes:
            errors.append(
                f"regularization_mode={self.regularization_mode!r} is not valid; "
                f"must be one of {valid_regularization_modes}"
            )

        valid_hybrid_methods = ("lbfgs", "gauss_newton")
        if self.hybrid_method not in valid_hybrid_methods:
            errors.append(
                f"hybrid_method={self.hybrid_method!r} is not valid; "
                f"must be one of {valid_hybrid_methods}"
            )

        valid_sampling_strategies = ("lhs", "sobol", "random")
        if self.sampling_strategy not in valid_sampling_strategies:
            errors.append(
                f"sampling_strategy={self.sampling_strategy!r} is not valid; "
                f"must be one of {valid_sampling_strategies}"
            )

        if self.nlsq_stability not in _VALID_NLSQ_STABILITY:
            errors.append(
                f"nlsq_stability={self.nlsq_stability!r} is not valid; "
                f"must be one of {sorted(_VALID_NLSQ_STABILITY)}"
            )

        if not (0 < self.nlsq_memory_fraction <= 1):
            errors.append(
                f"nlsq_memory_fraction={self.nlsq_memory_fraction} must be in (0, 1]"
            )

        if self.nlsq_memory_fallback_gb <= 0:
            errors.append(
                f"nlsq_memory_fallback_gb={self.nlsq_memory_fallback_gb} must be > 0"
            )

        # Advisory warnings — not errors, but worth surfacing at validate() time
        if self.gtol < 1e-7 and self.loss != "linear":
            logger.warning(
                "NLSQConfig: gtol=%.2e is very tight for loss=%r. "
                "Robust loss landscapes are harder — consider gtol >= 1e-6 "
                "to avoid premature max_nfev exhaustion.",
                self.gtol,
                self.loss,
            )

        if self.max_nfev is None:
            logger.debug(
                "NLSQConfig: max_nfev=None — nlsq defaults to 100×n_params "
                "(e.g. 1400 for 14 params). Set explicitly to override.",
            )

        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> NLSQConfig:
        """Construct an ``NLSQConfig`` from a plain dictionary.

        Nested sub-dictionaries under ``"recovery"`` and ``"validation"``
        are automatically parsed into their respective dataclasses.
        Unrecognised top-level keys are logged as warnings and ignored.

        Args:
            config: Flat or nested configuration dictionary, e.g. loaded from
                a YAML file.

        Returns:
            Fully populated ``NLSQConfig`` instance.
        """
        known_scalar_fields: dict[str, str] = {
            # Core solver
            "max_iterations": "int",
            "tolerance": "float",
            "method": "str",
            "multistart": "bool",
            "multistart_n": "int",
            "verbose": "int",
            "use_jac": "bool",
            "x_scale": "passthrough",  # str or list — handled separately
            "ftol": "float",
            "xtol": "float",
            "gtol": "float",
            "loss": "str",
            "diff_step": "float_or_none",
            "max_nfev": "int_or_none",
            "chunk_size": "int_or_none",
            # Workflow / goal
            "workflow": "str",
            "goal": "str",
            # Streaming / stratified
            "enable_streaming": "bool",
            "streaming_chunk_size": "int",
            "enable_stratified": "bool",
            "target_chunk_size": "int",
            # Recovery
            "enable_recovery": "bool",
            "max_recovery_attempts": "int",
            # Diagnostics
            "enable_diagnostics": "bool",
            "enable_anti_degeneracy": "bool",
            # Loss / scaling
            "loss_weights": "passthrough",  # list[float] | None
            "loss_scale": "float",
            "tr_solver": "str_or_none",
            "step_bound": "float",
            # Fourier reparameterization
            "per_angle_mode": "str",
            "fourier_order": "int",
            "fourier_auto_threshold": "int",
            # Hierarchical optimization
            "enable_hierarchical": "bool",
            "hierarchical_max_outer_iterations": "int",
            "hierarchical_inner_tolerance": "float",
            "hierarchical_outer_tolerance": "float",
            # Adaptive regularization
            "regularization_mode": "str",
            "group_variance_lambda": "float",
            "regularization_target_cv": "float",
            # Gradient collapse detection
            "enable_gradient_monitoring": "bool",
            "gradient_ratio_threshold": "float",
            "gradient_consecutive_triggers": "int",
            # Hierarchical optimization (homodyne-parity)
            "hierarchical_physical_max_iterations": "int",
            "hierarchical_per_angle_max_iterations": "int",
            # Adaptive regularization (homodyne-parity)
            "regularization_target_contribution": "float",
            "regularization_max_cv": "float",
            "regularization_auto_tune_lambda": "bool",
            # Gradient collapse detection (homodyne-parity)
            "gradient_collapse_response": "str",
            # CMA-ES global search (legacy heterodyne fields)
            "enable_cmaes": "bool",
            "cmaes_sigma0": "float",
            "cmaes_max_iterations": "int",
            "cmaes_population_size": "int_or_none",
            "cmaes_tolx": "float",
            "cmaes_tolfun": "float",
            "cmaes_diagonal_filtering": "str",
            "cmaes_anti_degeneracy": "bool",
            "cmaes_warmstart_auto_skip": "bool",
            "cmaes_warmstart_skip_threshold": "float",
            "cmaes_restart_strategy": "str",
            "cmaes_max_restarts": "int",
            # CMA-ES global search (homodyne-parity fields)
            "cmaes_preset": "str",
            "cmaes_max_generations": "int_or_none",
            "cmaes_popsize": "int_or_none",
            "cmaes_sigma": "float",
            "cmaes_sigma_warmstart": "float",
            "cmaes_tol_fun": "float",
            "cmaes_tol_x": "float",
            "cmaes_population_batch_size": "int_or_none",
            "cmaes_data_chunk_size": "int_or_none",
            "cmaes_refine_with_nlsq": "bool",
            "cmaes_auto_select": "bool",
            "cmaes_scale_threshold": "float",
            "cmaes_memory_limit_gb": "float",
            "cmaes_refinement_workflow": "str",
            "cmaes_refinement_ftol": "float",
            "cmaes_refinement_xtol": "float",
            "cmaes_refinement_gtol": "float",
            "cmaes_refinement_max_nfev": "int",
            "cmaes_refinement_loss": "str",
            "cmaes_normalize": "bool",
            "cmaes_normalization_epsilon": "float",
            # Progress and logging (homodyne-parity)
            "enable_progress_bar": "bool",
            "log_iteration_interval": "int",
            # Hybrid streaming optimizer (legacy heterodyne fields)
            "hybrid_enable": "bool",
            "hybrid_warmup_fraction": "float",
            "hybrid_normalization": "bool",
            "hybrid_method": "str",
            "hybrid_lbfgs_memory": "int",
            "hybrid_convergence_window": "int",
            "hybrid_convergence_threshold": "float",
            "hybrid_max_phases": "int",
            # Hybrid streaming optimizer (homodyne-parity fields)
            "enable_hybrid_streaming": "bool",
            "hybrid_normalize": "bool",
            "hybrid_normalization_strategy": "str",
            "hybrid_warmup_iterations": "int",
            "hybrid_max_warmup_iterations": "int",
            "hybrid_warmup_learning_rate": "float",
            "hybrid_gauss_newton_max_iterations": "int",
            "hybrid_gauss_newton_tol": "float",
            "hybrid_chunk_size": "int",
            "hybrid_trust_region_initial": "float",
            "hybrid_regularization_factor": "float",
            "hybrid_enable_checkpoints": "bool",
            "hybrid_checkpoint_frequency": "int",
            "hybrid_validate_numerics": "bool",
            "hybrid_enable_warm_start_detection": "bool",
            "hybrid_warm_start_threshold": "float",
            "hybrid_enable_adaptive_warmup_lr": "bool",
            "hybrid_warmup_lr_refinement": "float",
            "hybrid_warmup_lr_careful": "float",
            "hybrid_enable_cost_guard": "bool",
            "hybrid_cost_increase_tolerance": "float",
            "hybrid_enable_step_clipping": "bool",
            "hybrid_max_warmup_step_size": "float",
            # Multi-start extensions (legacy heterodyne fields)
            "sampling_strategy": "str",
            "screen_keep_fraction": "float",
            "refine_top_k": "int",
            # Multi-start optimization (homodyne-parity fields)
            "enable_multi_start": "bool",
            "multi_start_n_starts": "int",
            "multi_start_seed": "int",
            "multi_start_sampling_strategy": "str",
            "multi_start_n_workers": "int",
            "multi_start_use_screening": "bool",
            "multi_start_screen_keep_fraction": "float",
            "multi_start_refine_top_k": "int",
            "multi_start_refinement_ftol": "float",
            "multi_start_degeneracy_threshold": "float",
            # Fit quality validation (homodyne-parity fields)
            "enable_quality_validation": "bool",
            "quality_reduced_chi_squared_threshold": "float",
            "quality_warn_on_max_restarts": "bool",
            "quality_warn_on_bounds_hit": "bool",
            "quality_warn_on_convergence_failure": "bool",
            "quality_bounds_tolerance": "float",
            # Scaling threshold
            "constant_scaling_threshold": "int",
            # Backend / model
            "use_nlsq_library": "bool",
            "n_params": "int",
            "analysis_mode": "str",
            # NLSQ package integration
            "nlsq_stability": "str",
            "nlsq_rescale_data": "bool",
            "nlsq_x_scale": "passthrough",  # str or np.ndarray
            "nlsq_memory_fraction": "float",
            "nlsq_memory_fallback_gb": "float",
            # Loss function scale (homodyne-parity)
            "trust_region_scale": "float",
        }

        normalized_config = dict(config)

        def _set_from_nested(field_name: str, value: Any) -> None:
            if value is not _SENTINEL and field_name not in normalized_config:
                normalized_config[field_name] = value

        raw_anti_degeneracy = config.get("anti_degeneracy")
        if isinstance(raw_anti_degeneracy, dict):
            _set_from_nested(
                "per_angle_mode",
                raw_anti_degeneracy.get("per_angle_mode", _SENTINEL),
            )
            _set_from_nested(
                "fourier_order",
                raw_anti_degeneracy.get("fourier_order", _SENTINEL),
            )
            _set_from_nested(
                "fourier_auto_threshold",
                raw_anti_degeneracy.get("fourier_auto_threshold", _SENTINEL),
            )
            _set_from_nested(
                "constant_scaling_threshold",
                raw_anti_degeneracy.get("constant_scaling_threshold", _SENTINEL),
            )

            hierarchical = raw_anti_degeneracy.get("hierarchical")
            if isinstance(hierarchical, dict):
                _set_from_nested(
                    "enable_hierarchical", hierarchical.get("enable", _SENTINEL)
                )
                _set_from_nested(
                    "hierarchical_max_outer_iterations",
                    hierarchical.get("max_outer_iterations", _SENTINEL),
                )
                _set_from_nested(
                    "hierarchical_inner_tolerance",
                    hierarchical.get("inner_tolerance", _SENTINEL),
                )
                _set_from_nested(
                    "hierarchical_outer_tolerance",
                    hierarchical.get("outer_tolerance", _SENTINEL),
                )
                # Homodyne-parity hierarchical fields
                _set_from_nested(
                    "hierarchical_physical_max_iterations",
                    hierarchical.get("physical_max_iterations", _SENTINEL),
                )
                _set_from_nested(
                    "hierarchical_per_angle_max_iterations",
                    hierarchical.get("per_angle_max_iterations", _SENTINEL),
                )
            elif hierarchical is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.hierarchical must be a "
                    "dict, got %r — ignoring",
                    type(hierarchical).__name__,
                )

            regularization = raw_anti_degeneracy.get("regularization")
            if isinstance(regularization, dict):
                _set_from_nested(
                    "regularization_mode", regularization.get("mode", _SENTINEL)
                )
                _set_from_nested(
                    "group_variance_lambda", regularization.get("lambda", _SENTINEL)
                )
                _set_from_nested(
                    "regularization_target_cv",
                    regularization.get("target_cv", _SENTINEL),
                )
                # Homodyne-parity regularization fields
                _set_from_nested(
                    "regularization_target_contribution",
                    regularization.get("target_contribution", _SENTINEL),
                )
                _set_from_nested(
                    "regularization_max_cv",
                    regularization.get("max_cv", _SENTINEL),
                )
                _set_from_nested(
                    "regularization_auto_tune_lambda",
                    regularization.get("auto_tune_lambda", _SENTINEL),
                )
            elif regularization is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.regularization must be a "
                    "dict, got %r — ignoring",
                    type(regularization).__name__,
                )

            gradient_monitoring = raw_anti_degeneracy.get("gradient_monitoring")
            if isinstance(gradient_monitoring, dict):
                _set_from_nested(
                    "enable_gradient_monitoring",
                    gradient_monitoring.get("enable", _SENTINEL),
                )
                _set_from_nested(
                    "gradient_ratio_threshold",
                    gradient_monitoring.get("ratio_threshold", _SENTINEL),
                )
                _set_from_nested(
                    "gradient_consecutive_triggers",
                    gradient_monitoring.get("consecutive_triggers", _SENTINEL),
                )
                # Homodyne-parity gradient collapse response field
                _set_from_nested(
                    "gradient_collapse_response",
                    gradient_monitoring.get("response", _SENTINEL),
                )
            elif gradient_monitoring is not None:
                logger.warning(
                    "NLSQConfig.from_dict: anti_degeneracy.gradient_monitoring must "
                    "be a dict, got %r — ignoring",
                    type(gradient_monitoring).__name__,
                )
        elif raw_anti_degeneracy is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'anti_degeneracy' must be a dict, got %r — "
                "ignoring",
                type(raw_anti_degeneracy).__name__,
            )

        raw_cmaes = config.get("cmaes")
        if isinstance(raw_cmaes, dict):
            _set_from_nested("enable_cmaes", raw_cmaes.get("enable", _SENTINEL))
            _set_from_nested(
                "cmaes_sigma0",
                raw_cmaes.get("sigma", raw_cmaes.get("sigma0", _SENTINEL)),
            )
            _set_from_nested(
                "cmaes_max_iterations",
                raw_cmaes.get(
                    "max_generations",
                    raw_cmaes.get("max_iterations", _SENTINEL),
                ),
            )
            _set_from_nested(
                "cmaes_population_size",
                raw_cmaes.get("popsize", raw_cmaes.get("population_size", _SENTINEL)),
            )
            _set_from_nested(
                "cmaes_tolx", raw_cmaes.get("tol_x", raw_cmaes.get("tolx", _SENTINEL))
            )
            _set_from_nested(
                "cmaes_tolfun",
                raw_cmaes.get("tol_fun", raw_cmaes.get("tolfun", _SENTINEL)),
            )
            _set_from_nested(
                "cmaes_diagonal_filtering",
                raw_cmaes.get("diagonal_filtering", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_anti_degeneracy",
                raw_cmaes.get("anti_degeneracy", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_warmstart_auto_skip",
                raw_cmaes.get("warmstart_auto_skip", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_warmstart_skip_threshold",
                raw_cmaes.get("warmstart_skip_threshold", _SENTINEL),
            )
            # Homodyne-parity CMA-ES fields
            _set_from_nested("cmaes_preset", raw_cmaes.get("preset", _SENTINEL))
            _set_from_nested(
                "cmaes_max_generations",
                raw_cmaes.get("max_generations", _SENTINEL),
            )
            _set_from_nested("cmaes_popsize", raw_cmaes.get("popsize", _SENTINEL))
            _set_from_nested("cmaes_sigma", raw_cmaes.get("sigma", _SENTINEL))
            _set_from_nested(
                "cmaes_sigma_warmstart",
                raw_cmaes.get("sigma_warmstart", _SENTINEL),
            )
            _set_from_nested("cmaes_tol_fun", raw_cmaes.get("tol_fun", _SENTINEL))
            _set_from_nested("cmaes_tol_x", raw_cmaes.get("tol_x", _SENTINEL))
            _set_from_nested(
                "cmaes_population_batch_size",
                raw_cmaes.get("population_batch_size", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_data_chunk_size",
                raw_cmaes.get("data_chunk_size", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refine_with_nlsq",
                raw_cmaes.get("refine_with_nlsq", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_auto_select", raw_cmaes.get("auto_select", _SENTINEL)
            )
            _set_from_nested(
                "cmaes_scale_threshold",
                raw_cmaes.get("scale_threshold", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_memory_limit_gb",
                raw_cmaes.get("memory_limit_gb", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_workflow",
                raw_cmaes.get("refinement_workflow", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_ftol",
                raw_cmaes.get("refinement_ftol", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_xtol",
                raw_cmaes.get("refinement_xtol", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_gtol",
                raw_cmaes.get("refinement_gtol", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_max_nfev",
                raw_cmaes.get("refinement_max_nfev", _SENTINEL),
            )
            _set_from_nested(
                "cmaes_refinement_loss",
                raw_cmaes.get("refinement_loss", _SENTINEL),
            )
            _set_from_nested("cmaes_normalize", raw_cmaes.get("normalize", _SENTINEL))
            _set_from_nested(
                "cmaes_normalization_epsilon",
                raw_cmaes.get("normalization_epsilon", _SENTINEL),
            )
        elif raw_cmaes is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'cmaes' must be a dict, got %r — ignoring",
                type(raw_cmaes).__name__,
            )

        # Homodyne-parity: progress section
        raw_progress = config.get("progress")
        if isinstance(raw_progress, dict):
            _set_from_nested(
                "enable_progress_bar", raw_progress.get("enable", _SENTINEL)
            )
            _set_from_nested("verbose", raw_progress.get("verbose", _SENTINEL))
            _set_from_nested(
                "log_iteration_interval", raw_progress.get("log_interval", _SENTINEL)
            )
        elif raw_progress is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'progress' must be a dict, got %r — ignoring",
                type(raw_progress).__name__,
            )

        # Homodyne-parity: hybrid_streaming section
        raw_hybrid_streaming = config.get("hybrid_streaming")
        if isinstance(raw_hybrid_streaming, dict):
            _set_from_nested(
                "enable_hybrid_streaming",
                raw_hybrid_streaming.get("enable", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_normalize", raw_hybrid_streaming.get("normalize", _SENTINEL)
            )
            _set_from_nested(
                "hybrid_normalization_strategy",
                raw_hybrid_streaming.get("normalization_strategy", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_warmup_iterations",
                raw_hybrid_streaming.get("warmup_iterations", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_max_warmup_iterations",
                raw_hybrid_streaming.get("max_warmup_iterations", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_warmup_learning_rate",
                raw_hybrid_streaming.get("warmup_learning_rate", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_gauss_newton_max_iterations",
                raw_hybrid_streaming.get("gauss_newton_max_iterations", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_gauss_newton_tol",
                raw_hybrid_streaming.get("gauss_newton_tol", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_chunk_size", raw_hybrid_streaming.get("chunk_size", _SENTINEL)
            )
            _set_from_nested(
                "hybrid_trust_region_initial",
                raw_hybrid_streaming.get("trust_region_initial", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_regularization_factor",
                raw_hybrid_streaming.get("regularization_factor", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_enable_checkpoints",
                raw_hybrid_streaming.get("enable_checkpoints", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_checkpoint_frequency",
                raw_hybrid_streaming.get("checkpoint_frequency", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_validate_numerics",
                raw_hybrid_streaming.get("validate_numerics", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_enable_warm_start_detection",
                raw_hybrid_streaming.get("enable_warm_start_detection", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_warm_start_threshold",
                raw_hybrid_streaming.get("warm_start_threshold", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_enable_adaptive_warmup_lr",
                raw_hybrid_streaming.get("enable_adaptive_warmup_lr", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_warmup_lr_refinement",
                raw_hybrid_streaming.get("warmup_lr_refinement", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_warmup_lr_careful",
                raw_hybrid_streaming.get("warmup_lr_careful", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_enable_cost_guard",
                raw_hybrid_streaming.get("enable_cost_guard", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_cost_increase_tolerance",
                raw_hybrid_streaming.get("cost_increase_tolerance", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_enable_step_clipping",
                raw_hybrid_streaming.get("enable_step_clipping", _SENTINEL),
            )
            _set_from_nested(
                "hybrid_max_warmup_step_size",
                raw_hybrid_streaming.get("max_warmup_step_size", _SENTINEL),
            )
        elif raw_hybrid_streaming is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'hybrid_streaming' must be a dict, got %r — ignoring",
                type(raw_hybrid_streaming).__name__,
            )

        # Homodyne-parity: multi_start section
        raw_multi_start = config.get("multi_start")
        if isinstance(raw_multi_start, dict):
            _set_from_nested(
                "enable_multi_start", raw_multi_start.get("enable", _SENTINEL)
            )
            _set_from_nested(
                "multi_start_n_starts", raw_multi_start.get("n_starts", _SENTINEL)
            )
            _set_from_nested("multi_start_seed", raw_multi_start.get("seed", _SENTINEL))
            _set_from_nested(
                "multi_start_sampling_strategy",
                raw_multi_start.get("sampling_strategy", _SENTINEL),
            )
            _set_from_nested(
                "multi_start_n_workers", raw_multi_start.get("n_workers", _SENTINEL)
            )
            _set_from_nested(
                "multi_start_use_screening",
                raw_multi_start.get("use_screening", _SENTINEL),
            )
            _set_from_nested(
                "multi_start_screen_keep_fraction",
                raw_multi_start.get("screen_keep_fraction", _SENTINEL),
            )
            _set_from_nested(
                "multi_start_refine_top_k",
                raw_multi_start.get("refine_top_k", _SENTINEL),
            )
            _set_from_nested(
                "multi_start_refinement_ftol",
                raw_multi_start.get("refinement_ftol", _SENTINEL),
            )
            _set_from_nested(
                "multi_start_degeneracy_threshold",
                raw_multi_start.get("degeneracy_threshold", _SENTINEL),
            )
        elif raw_multi_start is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'multi_start' must be a dict, got %r — ignoring",
                type(raw_multi_start).__name__,
            )

        # Homodyne-parity: quality_validation section
        raw_quality = config.get("quality_validation")
        if isinstance(raw_quality, dict):
            _set_from_nested(
                "enable_quality_validation", raw_quality.get("enable", _SENTINEL)
            )
            _set_from_nested(
                "quality_reduced_chi_squared_threshold",
                raw_quality.get("reduced_chi_squared_threshold", _SENTINEL),
            )
            _set_from_nested(
                "quality_warn_on_max_restarts",
                raw_quality.get("warn_on_max_restarts", _SENTINEL),
            )
            _set_from_nested(
                "quality_warn_on_bounds_hit",
                raw_quality.get("warn_on_bounds_hit", _SENTINEL),
            )
            _set_from_nested(
                "quality_warn_on_convergence_failure",
                raw_quality.get("warn_on_convergence_failure", _SENTINEL),
            )
            _set_from_nested(
                "quality_bounds_tolerance",
                raw_quality.get("bounds_tolerance", _SENTINEL),
            )
        elif raw_quality is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'quality_validation' must be a dict, got %r — ignoring",
                type(raw_quality).__name__,
            )

        nested_keys = {
            "recovery",
            "validation",
            "x_scale_map",
            "anti_degeneracy",
            "cmaes",
            "progress",
            "hybrid_streaming",
            "multi_start",
            "quality_validation",
        }

        # Warn on unrecognised keys
        all_known = set(known_scalar_fields) | nested_keys
        for key in normalized_config:
            if key not in all_known:
                logger.warning(
                    "NLSQConfig.from_dict: unrecognised key %r — ignoring", key
                )

        kwargs: dict[str, Any] = {}

        # --- Parse scalar fields -----------------------------------------
        for field_name, kind in known_scalar_fields.items():
            raw = normalized_config.get(field_name, _SENTINEL)
            if raw is _SENTINEL:
                continue  # use dataclass default

            if kind == "float":
                kwargs[field_name] = safe_float(raw, 0.0)
            elif kind == "int":
                kwargs[field_name] = safe_int(raw, 0)
            elif kind == "bool":
                kwargs[field_name] = bool(raw)
            elif kind == "str":
                kwargs[field_name] = str(raw)
            elif kind == "float_or_none":
                kwargs[field_name] = None if raw is None else safe_float(raw, 0.0)
            elif kind == "int_or_none":
                kwargs[field_name] = None if raw is None else safe_int(raw, 0)
            elif kind == "str_or_none":
                kwargs[field_name] = None if raw is None else str(raw)
            elif kind == "passthrough":
                kwargs[field_name] = raw
            # no else branch needed — exhaustive set above

        # --- Parse x_scale_map -------------------------------------------
        raw_scale_map = normalized_config.get("x_scale_map")
        if isinstance(raw_scale_map, dict):
            kwargs["x_scale_map"] = {
                str(k): safe_float(v, 1.0) for k, v in raw_scale_map.items()
            }
        elif raw_scale_map is not None:
            logger.warning(
                "NLSQConfig.from_dict: x_scale_map must be a dict, got %r — ignoring",
                type(raw_scale_map).__name__,
            )

        # --- Parse nested recovery sub-dict ------------------------------
        raw_recovery = normalized_config.get("recovery")
        if isinstance(raw_recovery, dict):
            recovery = HybridRecoveryConfig(
                max_retries=safe_int(
                    raw_recovery.get("max_retries"), HybridRecoveryConfig.max_retries
                ),
                lr_decay=safe_float(
                    raw_recovery.get("lr_decay"), HybridRecoveryConfig.lr_decay
                ),
                lambda_growth=safe_float(
                    raw_recovery.get("lambda_growth"),
                    HybridRecoveryConfig.lambda_growth,
                ),
                trust_decay=safe_float(
                    raw_recovery.get("trust_decay"), HybridRecoveryConfig.trust_decay
                ),
                perturb_scale=safe_float(
                    raw_recovery.get("perturb_scale"),
                    HybridRecoveryConfig.perturb_scale,
                ),
            )
            kwargs["recovery_config"] = recovery
        elif raw_recovery is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'recovery' must be a dict, got %r — ignoring",
                type(raw_recovery).__name__,
            )

        # --- Parse nested validation sub-dict ----------------------------
        raw_validation = normalized_config.get("validation")
        if isinstance(raw_validation, dict):
            defaults = NLSQValidationConfig()
            validation = NLSQValidationConfig(
                chi2_warn_low=safe_float(
                    raw_validation.get("chi2_warn_low"), defaults.chi2_warn_low
                ),
                chi2_warn_high=safe_float(
                    raw_validation.get("chi2_warn_high"), defaults.chi2_warn_high
                ),
                chi2_fail_high=safe_float(
                    raw_validation.get("chi2_fail_high"), defaults.chi2_fail_high
                ),
                max_relative_uncertainty=safe_float(
                    raw_validation.get("max_relative_uncertainty"),
                    defaults.max_relative_uncertainty,
                ),
                correlation_warn=safe_float(
                    raw_validation.get("correlation_warn"), defaults.correlation_warn
                ),
            )
            kwargs["validation"] = validation
        elif raw_validation is not None:
            logger.warning(
                "NLSQConfig.from_dict: 'validation' must be a dict, got %r — ignoring",
                type(raw_validation).__name__,
            )

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration to a plain dictionary.

        Nested dataclasses are serialised as nested dicts, making the output
        suitable for round-tripping through YAML / JSON.

        Returns:
            Fully populated dictionary representation.
        """
        return {
            # Core solver
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "method": self.method,
            "multistart": self.multistart,
            "multistart_n": self.multistart_n,
            "verbose": self.verbose,
            "use_jac": self.use_jac,
            "x_scale": self.x_scale,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "loss": self.loss,
            "diff_step": self.diff_step,
            "max_nfev": self.max_nfev,
            "chunk_size": self.chunk_size,
            # Workflow / goal
            "workflow": self.workflow,
            "goal": self.goal,
            # Streaming / stratified
            "enable_streaming": self.enable_streaming,
            "streaming_chunk_size": self.streaming_chunk_size,
            "enable_stratified": self.enable_stratified,
            "target_chunk_size": self.target_chunk_size,
            # Recovery
            "enable_recovery": self.enable_recovery,
            "max_recovery_attempts": self.max_recovery_attempts,
            "recovery": {
                "max_retries": self.recovery_config.max_retries,
                "lr_decay": self.recovery_config.lr_decay,
                "lambda_growth": self.recovery_config.lambda_growth,
                "trust_decay": self.recovery_config.trust_decay,
                "perturb_scale": self.recovery_config.perturb_scale,
            },
            # Diagnostics
            "enable_diagnostics": self.enable_diagnostics,
            "enable_anti_degeneracy": self.enable_anti_degeneracy,
            # Loss / scaling
            "x_scale_map": dict(self.x_scale_map),
            "loss_weights": self.loss_weights,
            "loss_scale": self.loss_scale,
            "tr_solver": self.tr_solver,
            "step_bound": self.step_bound,
            # Fourier reparameterization
            "per_angle_mode": self.per_angle_mode,
            "fourier_order": self.fourier_order,
            "fourier_auto_threshold": self.fourier_auto_threshold,
            # Hierarchical optimization
            "enable_hierarchical": self.enable_hierarchical,
            "hierarchical_max_outer_iterations": self.hierarchical_max_outer_iterations,
            "hierarchical_inner_tolerance": self.hierarchical_inner_tolerance,
            "hierarchical_outer_tolerance": self.hierarchical_outer_tolerance,
            # Adaptive regularization
            "regularization_mode": self.regularization_mode,
            "group_variance_lambda": self.group_variance_lambda,
            "regularization_target_cv": self.regularization_target_cv,
            # Gradient collapse detection
            "enable_gradient_monitoring": self.enable_gradient_monitoring,
            "gradient_ratio_threshold": self.gradient_ratio_threshold,
            "gradient_consecutive_triggers": self.gradient_consecutive_triggers,
            # CMA-ES global search
            "enable_cmaes": self.enable_cmaes,
            "cmaes_sigma0": self.cmaes_sigma0,
            "cmaes_max_iterations": self.cmaes_max_iterations,
            "cmaes_population_size": self.cmaes_population_size,
            "cmaes_tolx": self.cmaes_tolx,
            "cmaes_tolfun": self.cmaes_tolfun,
            "cmaes_diagonal_filtering": self.cmaes_diagonal_filtering,
            "cmaes_anti_degeneracy": self.cmaes_anti_degeneracy,
            "cmaes_warmstart_auto_skip": self.cmaes_warmstart_auto_skip,
            "cmaes_warmstart_skip_threshold": self.cmaes_warmstart_skip_threshold,
            "cmaes_restart_strategy": self.cmaes_restart_strategy,
            "cmaes_max_restarts": self.cmaes_max_restarts,
            # Hybrid streaming optimizer
            "hybrid_enable": self.hybrid_enable,
            "hybrid_warmup_fraction": self.hybrid_warmup_fraction,
            "hybrid_normalization": self.hybrid_normalization,
            "hybrid_method": self.hybrid_method,
            "hybrid_lbfgs_memory": self.hybrid_lbfgs_memory,
            "hybrid_convergence_window": self.hybrid_convergence_window,
            "hybrid_convergence_threshold": self.hybrid_convergence_threshold,
            "hybrid_max_phases": self.hybrid_max_phases,
            # Multi-start extensions
            "sampling_strategy": self.sampling_strategy,
            "screen_keep_fraction": self.screen_keep_fraction,
            "refine_top_k": self.refine_top_k,
            # Scaling threshold
            "constant_scaling_threshold": self.constant_scaling_threshold,
            # Backend / model
            "use_nlsq_library": self.use_nlsq_library,
            "n_params": self.n_params,
            "analysis_mode": self.analysis_mode,
            # NLSQ package integration
            "nlsq_stability": self.nlsq_stability,
            "nlsq_rescale_data": self.nlsq_rescale_data,
            "nlsq_x_scale": self.nlsq_x_scale,
            "nlsq_memory_fraction": self.nlsq_memory_fraction,
            "nlsq_memory_fallback_gb": self.nlsq_memory_fallback_gb,
            # Validation
            "validation": {
                "chi2_warn_low": self.validation.chi2_warn_low,
                "chi2_warn_high": self.validation.chi2_warn_high,
                "chi2_fail_high": self.validation.chi2_fail_high,
                "max_relative_uncertainty": self.validation.max_relative_uncertainty,
                "correlation_warn": self.validation.correlation_warn,
            },
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> NLSQConfig:
        """Create NLSQConfig from YAML configuration file.

        This is the recommended single entry point for loading NLSQ
        configuration.  It reads the YAML file, extracts the
        ``optimization.nlsq`` section, and creates a validated
        ``NLSQConfig`` object.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Validated ``NLSQConfig`` instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML file is invalid or missing required
                sections.

        Examples:
            >>> config = NLSQConfig.from_yaml("heterodyne_config.yaml")
            >>> print(config.loss)
            soft_l1
        """
        from pathlib import Path

        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(path, encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        if full_config is None:
            full_config = {}

        # Extract optimization.nlsq section
        optimization = full_config.get("optimization", {})
        nlsq_config = optimization.get("nlsq", {})

        if not nlsq_config:
            logger.warning(
                "No optimization.nlsq section found in %s, using defaults",
                yaml_path,
            )

        return cls.from_dict(nlsq_config)

    def is_valid(self) -> bool:
        """Check if the configuration is valid.

        Returns:
            ``True`` if ``validate()`` returns an empty list, ``False``
            otherwise.
        """
        return len(self.validate()) == 0

    def to_workflow_kwargs(self) -> dict[str, Any]:
        """Convert settings to kwargs for NLSQ's ``curve_fit()``.

        Maps ``NLSQConfig`` settings to NLSQ 0.6.10+ ``curve_fit()``
        parameters.  Heterodyne uses ``curve_fit()`` directly rather than
        the unified ``fit()`` API.

        Returns:
            Dictionary of kwargs suitable for passing to ``curve_fit()``:
            ``ftol``, ``gtol``, ``xtol``, ``max_nfev``, ``loss``, and
            optionally ``goal``.

        Notes:
            NLSQ 0.6.3+ workflows: ``"auto"``, ``"auto_global"``,
            ``"hpc"``.  Old presets (``"streaming"``, ``"standard"``)
            were removed.  Heterodyne uses its own strategy selection for
            memory-aware dispatch, so ``workflow`` is not forwarded.
        """
        kwargs: dict[str, Any] = {}

        # goal can be passed to NLSQ's fit() API; omit the default value
        # to avoid overriding NLSQ's own default.
        if self.goal != "quality":
            kwargs["goal"] = self.goal

        # Convergence settings — directly supported by curve_fit()
        kwargs["ftol"] = self.ftol
        kwargs["gtol"] = self.gtol
        kwargs["xtol"] = self.xtol
        kwargs["max_nfev"] = self.max_iterations

        # Loss function
        kwargs["loss"] = self.loss

        return kwargs
