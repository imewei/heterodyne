"""Anti-Degeneracy Controller — Orchestrator for 4-Layer Defense System.

This module provides a clean interface for initializing and coordinating
the 4-layer anti-degeneracy defense system for NLSQ optimization.

The controller encapsulates:
- Layer 1: Fourier/Constant Reparameterization
- Layer 2: Hierarchical Optimization
- Layer 3: Adaptive CV-based Regularization
- Layer 4: Gradient Collapse Monitoring

Note: Layer 5 (Shear-Sensitivity Weighting) is intentionally absent.
Heterodyne uses a velocity-phase physics model; there is no shear sinc term
in the g2 formula, so shear-sensitivity weighting is not applicable (D3
divergence, spec §2).

The heterodyne model has 14 physics parameters (vs. 7 in homodyne) plus
2 per-angle scaling parameters (contrast, offset).

Ported from homodyne v2.9.0 with the following adaptations:
- Layer 5 removed entirely (shear_weighting)
- Support module class names adapted to heterodyne's existing implementations:
  * GradientMonitor (heterodyne) ← GradientCollapseMonitor (homodyne)
  * RegularizationConfig (heterodyne) ← AdaptiveRegularizationConfig (homodyne)
- n_physical=14 default throughout (heterodyne's 14-param model)
- is_laminar_flow kwarg absent: heterodyne has a single physics model

Version: 2.9.0-het
Author: Claude Code
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from heterodyne.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizer,
    RegularizationConfig,
)
from heterodyne.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from heterodyne.optimization.nlsq.gradient_monitor import GradientMonitor
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Known problematic parameter pairs in the 14-parameter heterodyne model
_KNOWN_DEGENERATE_PAIRS: list[tuple[str, str, str]] = [
    ("D0_ref", "D0_sample", "diffusion coefficient degeneracy"),
    ("f0", "f3", "fraction amplitude/baseline trading"),
    ("v0", "v_offset", "velocity magnitude/offset trading"),
    ("alpha_ref", "D0_ref", "exponent-prefactor compensation"),
    ("alpha_sample", "D0_sample", "exponent-prefactor compensation"),
]


@dataclass
class AntiDegeneracyConfig:
    """Configuration for the Anti-Degeneracy Defense System.

    Attributes
    ----------
    enable : bool
        Master switch for all anti-degeneracy defenses.
    per_angle_mode : str
        Mode for per-angle parameters: "individual", "constant", "fourier", or "auto".
    fourier_order : int
        Order of Fourier series (order=2 -> 5 coefficients per group).
    fourier_auto_threshold : int
        n_phi threshold for auto mode to switch to Fourier.
    constant_scaling_threshold : int
        n_phi threshold for auto mode to use constant scaling (n_phi >= threshold).
    hierarchical_enable : bool
        Enable hierarchical two-stage optimization.
    hierarchical_max_outer_iterations : int
        Maximum outer iterations for hierarchical optimization.
    hierarchical_outer_tolerance : float
        Convergence tolerance on physical parameter change.
    regularization_mode : str
        Regularization mode: "absolute", "relative", or "auto".
    regularization_lambda : float
        Base regularization strength.
    regularization_target_cv : float
        Target coefficient of variation (0-1).
    regularization_target_contribution : float
        Target regularization contribution to loss (0-1).
    regularization_max_cv : float
        Maximum allowed CV before hard constraint warning.
    gradient_monitoring_enable : bool
        Enable gradient collapse monitoring.
    gradient_ratio_threshold : float
        Ratio threshold for gradient collapse detection.
    gradient_consecutive_triggers : int
        Consecutive low-ratio iterations to confirm collapse.
    gradient_response_mode : str
        Response action: "warn", "hierarchical", "reset", "abort".

    Note: No shear_weighting fields — Layer 5 is D3-dropped in heterodyne.
    """

    enable: bool = True
    per_angle_mode: str = "auto"
    fourier_order: int = 2
    fourier_auto_threshold: int = 8
    constant_scaling_threshold: int = 3
    # Layer 2: Hierarchical Optimization
    hierarchical_enable: bool = True
    hierarchical_max_outer_iterations: int = 5
    hierarchical_outer_tolerance: float = 1e-4
    hierarchical_physical_max_iterations: int = 100
    hierarchical_per_angle_max_iterations: int = 50
    # Layer 3: Adaptive CV-based Regularization
    regularization_mode: str = "relative"
    regularization_lambda: float = 1.0
    regularization_target_cv: float = 0.10
    regularization_target_contribution: float = 0.10
    regularization_max_cv: float = 0.20
    # Layer 4: Gradient Collapse Monitoring
    gradient_monitoring_enable: bool = True
    gradient_ratio_threshold: float = 0.01
    gradient_consecutive_triggers: int = 5
    gradient_response_mode: str = "hierarchical"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AntiDegeneracyConfig:
        """Create config from nested dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with structure::

                {
                    "enable": bool,
                    "per_angle_mode": str,
                    "fourier_order": int,
                    "fourier_auto_threshold": int,
                    "constant_scaling_threshold": int,
                    "hierarchical": {
                        "enable": bool,
                        "max_outer_iterations": int,
                        "outer_tolerance": float,
                    },
                    "regularization": {
                        "enable": bool,
                        "mode": str,
                        "lambda": float,
                        "target_cv": float,
                        "target_contribution": float,
                        "max_cv": float,
                    },
                    "gradient_monitoring": {
                        "enable": bool,
                        "ratio_threshold": float,
                        "consecutive_triggers": int,
                        "response": str,
                    },
                }

        Returns
        -------
        AntiDegeneracyConfig
            Validated configuration object.
        """
        hierarchical = config_dict.get("hierarchical", {})
        regularization = config_dict.get("regularization", {})
        gradient_monitoring = config_dict.get("gradient_monitoring", {})

        return cls(
            enable=bool(config_dict.get("enable", True)),
            per_angle_mode=str(config_dict.get("per_angle_mode", "auto")),
            fourier_order=int(config_dict.get("fourier_order", 2)),
            fourier_auto_threshold=int(config_dict.get("fourier_auto_threshold", 8)),
            constant_scaling_threshold=int(
                config_dict.get("constant_scaling_threshold", 3)
            ),
            # Layer 2
            hierarchical_enable=bool(hierarchical.get("enable", True)),
            hierarchical_max_outer_iterations=int(
                hierarchical.get("max_outer_iterations", 5)
            ),
            hierarchical_outer_tolerance=float(
                hierarchical.get("outer_tolerance", 1e-4)
            ),
            hierarchical_physical_max_iterations=int(
                hierarchical.get("physical_max_iterations", 100)
            ),
            hierarchical_per_angle_max_iterations=int(
                hierarchical.get("per_angle_max_iterations", 50)
            ),
            # Layer 3
            regularization_mode=str(regularization.get("mode", "relative")),
            regularization_lambda=float(regularization.get("lambda", 1.0)),
            regularization_target_cv=float(regularization.get("target_cv", 0.10)),
            regularization_target_contribution=float(
                regularization.get("target_contribution", 0.10)
            ),
            regularization_max_cv=float(regularization.get("max_cv", 0.20)),
            # Layer 4
            gradient_monitoring_enable=bool(gradient_monitoring.get("enable", True)),
            gradient_ratio_threshold=float(
                gradient_monitoring.get("ratio_threshold", 0.01)
            ),
            gradient_consecutive_triggers=int(
                gradient_monitoring.get("consecutive_triggers", 5)
            ),
            gradient_response_mode=str(
                gradient_monitoring.get("response", "hierarchical")
            ),
        )


@dataclass
class AntiDegeneracyController:
    """Orchestrator for the 4-Layer Anti-Degeneracy Defense System.

    This controller provides a clean interface for initializing and
    coordinating all anti-degeneracy components for heterodyne NLSQ.

    Attributes
    ----------
    config : AntiDegeneracyConfig
        Configuration for the defense system.
    n_phi : int
        Number of phi angles.
    n_physical : int
        Number of physical parameters (14 for heterodyne).
    phi_angles : np.ndarray
        Array of phi angles in radians.
    fourier : FourierReparameterizer | None
        Layer 1: Fourier reparameterization component.
    regularizer : AdaptiveRegularizer | None
        Layer 3: Adaptive regularization component.
    monitor : GradientMonitor | None
        Layer 4: Gradient collapse monitoring component.
    per_angle_mode_actual : str
        Actual mode used ("auto_averaged", "fourier", "individual", or "disabled").

    Note: No ``shear_weighter`` field — Layer 5 is D3-dropped in heterodyne.
    The ``use_shear_weighting`` property always returns False.
    Layer 2 (Hierarchical) uses lazy construction at fit time; see
    ``use_hierarchical`` property and ``_hierarchical_config_dict``.
    """

    config: AntiDegeneracyConfig
    n_phi: int
    n_physical: int
    phi_angles: np.ndarray
    fourier: FourierReparameterizer | None = None
    regularizer: AdaptiveRegularizer | None = None
    monitor: GradientMonitor | None = None
    per_angle_mode_actual: str = "disabled"
    # Fixed per-angle quantile estimates for constant mode
    _fixed_contrast_per_angle: np.ndarray | None = field(default=None, repr=False)
    _fixed_offset_per_angle: np.ndarray | None = field(default=None, repr=False)
    _is_initialized: bool = field(default=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        n_phi: int,
        phi_angles: np.ndarray,
        n_physical: int = 14,
        per_angle_scaling: bool = True,
    ) -> AntiDegeneracyController:
        """Create controller from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Anti-degeneracy configuration dictionary.
        n_phi : int
            Number of phi angles.
        phi_angles : np.ndarray
            Array of phi angles in radians.
        n_physical : int
            Number of physical parameters (14 for heterodyne).
        per_angle_scaling : bool
            Whether per-angle scaling is enabled.

        Returns
        -------
        AntiDegeneracyController
            Initialized controller with all components.

        Note
        ----
        The ``is_laminar_flow`` kwarg from homodyne is not present here because
        heterodyne has a single physics model (always the full 14-param model).
        """
        config = AntiDegeneracyConfig.from_dict(config_dict)

        controller = cls(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=phi_angles,
        )

        # Only initialize if enabled and per-angle scaling is active
        if config.enable and per_angle_scaling:
            controller._initialize_components()

        return controller

    def _initialize_components(self) -> None:
        """Initialize all 4 layers of the defense system."""
        config = self.config

        # Determine actual per-angle mode with auto-selection logic
        # Mirrors homodyne v2.18.0 semantics:
        #   - auto (n_phi >= constant_scaling_threshold): "auto_averaged" → optimized averaged scaling
        #   - auto (n_phi < threshold): "individual" → per-angle scaling
        #   - constant (explicit): "fixed_constant" → FIXED per-angle scaling
        #   - fourier (explicit or auto >= fourier_auto_threshold): "fourier"
        if config.per_angle_mode == "auto":
            if self.n_phi >= config.fourier_auto_threshold:
                self.per_angle_mode_actual = "fourier"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY: Auto-selected 'fourier' mode")
                logger.info(
                    f"  Reason: n_phi ({self.n_phi}) >= "
                    f"fourier_auto_threshold ({config.fourier_auto_threshold})"
                )
                logger.info("=" * 60)
            elif self.n_phi >= config.constant_scaling_threshold:
                self.per_angle_mode_actual = "auto_averaged"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY: Auto-selected 'auto_averaged' mode")
                logger.info(
                    f"  Reason: n_phi ({self.n_phi}) >= "
                    f"constant_scaling_threshold ({config.constant_scaling_threshold})"
                )
                logger.info("  Behavior: Quantile estimates -> AVERAGED -> OPTIMIZED")
                logger.info("  Parameters: 14 physical + 2 averaged scaling = 16 total")
                logger.info("=" * 60)
            else:
                self.per_angle_mode_actual = "individual"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY: Auto-selected 'individual' mode")
                logger.info(
                    f"  Reason: n_phi ({self.n_phi}) < "
                    f"constant_scaling_threshold ({config.constant_scaling_threshold})"
                )
                logger.info(
                    f"  Parameters: 14 physical + {2 * self.n_phi} per-angle = "
                    f"{14 + 2 * self.n_phi} total"
                )
                logger.info("=" * 60)
        elif config.per_angle_mode == "constant":
            self.per_angle_mode_actual = "fixed_constant"
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY: Using explicit 'constant' mode -> fixed_constant"
            )
            logger.info(f"  n_phi: {self.n_phi}")
            logger.info(
                "  Behavior: Quantile estimates -> per-angle FIXED (not optimized)"
            )
            logger.info("=" * 60)
        elif config.per_angle_mode == "fourier":
            self.per_angle_mode_actual = "fourier"
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Using explicit 'fourier' mode")
            logger.info(f"  n_phi: {self.n_phi}, Fourier order: {config.fourier_order}")
            logger.info("=" * 60)
        else:
            # "individual" or unknown → individual
            self.per_angle_mode_actual = "individual"
            logger.info("=" * 60)
            logger.info(
                f"ANTI-DEGENERACY: Using per_angle_mode='{config.per_angle_mode}' -> individual"
            )
            logger.info("=" * 60)

        # Determine if we use "constant" style (fixed_constant or auto_averaged)
        use_constant = self.per_angle_mode_actual in ("fixed_constant", "auto_averaged")

        # Layer 1: Fourier Reparameterization (only if fourier mode)
        if self.per_angle_mode_actual == "fourier":
            fourier_config = FourierReparamConfig(
                mode="fourier",
                fourier_order=config.fourier_order,
                auto_threshold=config.fourier_auto_threshold,
            )
            self.fourier = FourierReparameterizer(self.phi_angles, fourier_config)
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 1 - Fourier Reparameterization")
            logger.info(f"  Mode: {self.per_angle_mode_actual}")
            logger.info(f"  n_phi: {self.n_phi}, Fourier order: {config.fourier_order}")
            logger.info(
                f"  Parameter reduction: {2 * self.n_phi} -> {self.fourier.n_coeffs}"
            )
            logger.info("=" * 60)

        # Layer 2: Hierarchical Optimization
        # Note: HierarchicalFitter in heterodyne requires an adapter at construction
        # time (unlike HierarchicalOptimizer in homodyne).  We store the config
        # parameters and a flag; the strategy or adapter instantiates the fitter
        # lazily when it has an adapter handle.  Active-loop integration is via
        # create_nlsq_callbacks().
        if config.hierarchical_enable:
            self._hierarchical_enabled = True
            self._hierarchical_config_dict: dict[str, Any] = {
                "max_outer_iterations": config.hierarchical_max_outer_iterations,
                "outer_tolerance": config.hierarchical_outer_tolerance,
                "physical_max_iterations": config.hierarchical_physical_max_iterations,
                "per_angle_max_iterations": config.hierarchical_per_angle_max_iterations,
            }
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 2 - Hierarchical Optimization")
            logger.info("  Enabled: True (lazy construction at fit time)")
            logger.info(
                f"  Max outer iterations: {config.hierarchical_max_outer_iterations}"
            )
            logger.info(f"  Outer tolerance: {config.hierarchical_outer_tolerance}")
            logger.info("=" * 60)
        else:
            self._hierarchical_enabled = False
            self._hierarchical_config_dict = {}

        # Layer 3: Adaptive Regularization
        # Compute per-angle parameter group indices for regularization.
        # In the optimization parameter vector layout:
        #   [contrast_params... | offset_params... | physics_params...]
        # where the per-angle section has n_per_angle_params entries total.
        if use_constant:
            # auto_averaged: [contrast_avg(1), offset_avg(1)]
            group_indices: list[tuple[int, int]] = [(0, 1), (1, 2)]
        elif self.fourier is not None:
            # Fourier mode: [contrast_coeffs(k), offset_coeffs(k)]
            k = self.fourier.n_coeffs_per_param
            group_indices = [(0, k), (k, 2 * k)]
        else:
            # Individual: [contrast_n_phi, offset_n_phi]
            group_indices = [(0, self.n_phi), (self.n_phi, 2 * self.n_phi)]

        reg_config = RegularizationConfig(
            lambda_init=config.regularization_lambda,
        )
        self.regularizer = AdaptiveRegularizer(config=reg_config)
        # Store group indices for use in callbacks and diagnostics
        self._reg_group_indices: list[tuple[int, int]] = group_indices
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY: Layer 3 - Adaptive Regularization")
        logger.info(f"  Mode: {config.regularization_mode}")
        logger.info(f"  lambda_base: {config.regularization_lambda}")
        logger.info(f"  target_cv: {config.regularization_target_cv}")
        logger.info("=" * 60)

        # Layer 4: Gradient Collapse Monitoring
        if config.gradient_monitoring_enable:
            self.monitor = GradientMonitor(
                parameter_names=None  # will be set when param names are known
            )
            self._gradient_consecutive_triggers = config.gradient_consecutive_triggers
            self._gradient_ratio_threshold = config.gradient_ratio_threshold
            self._gradient_trigger_count = 0
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 4 - Gradient Collapse Monitor")
            logger.info("  Enabled: True")
            logger.info(f"  Ratio threshold: {config.gradient_ratio_threshold}")
            logger.info(f"  Response mode: {config.gradient_response_mode}")
            logger.info("=" * 60)

        self._is_initialized = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Check if the defense system is enabled and initialized."""
        return self._is_initialized and self.config.enable

    @property
    def use_fourier(self) -> bool:
        """Check if Fourier reparameterization is active."""
        return self.fourier is not None

    @property
    def use_constant(self) -> bool:
        """Check if constant scaling mode is active.

        Returns True for both "auto_averaged" and "fixed_constant" modes.
        Both use constant-style parameter mapping (fewer params than 2×n_phi).
        """
        return self.per_angle_mode_actual in ("fixed_constant", "auto_averaged")

    @property
    def use_fixed_scaling(self) -> bool:
        """Check if fixed (non-optimized) per-angle scaling is active."""
        return self.per_angle_mode_actual == "fixed_constant"

    @property
    def use_averaged_scaling(self) -> bool:
        """Check if averaged (optimized) scaling is active (auto_averaged mode)."""
        return self.per_angle_mode_actual == "auto_averaged"

    @property
    def use_hierarchical(self) -> bool:
        """Check if hierarchical optimization is active."""
        return getattr(self, "_hierarchical_enabled", False)

    @property
    def use_shear_weighting(self) -> bool:
        """Layer 5 — always False in heterodyne (D3-dropped, no shear physics)."""
        return False

    @property
    def n_per_angle_params(self) -> int:
        """Get the number of per-angle parameters in the optimization vector.

        Returns:
        - fixed_constant: 0 (scaling is FIXED, not optimized)
        - auto_averaged: 2 (one contrast, one offset — OPTIMIZED)
        - fourier: n_coeffs (Fourier coefficients — OPTIMIZED)
        - individual: 2 * n_phi (independent per-angle — OPTIMIZED)
        """
        if self.use_fixed_scaling:
            return 0
        if self.use_averaged_scaling:
            return 2
        if self.fourier:
            return self.fourier.n_coeffs
        return 2 * self.n_phi

    @property
    def has_fixed_per_angle_scaling(self) -> bool:
        """Check if fixed per-angle scaling estimates are available."""
        return (
            self._fixed_contrast_per_angle is not None
            and self._fixed_offset_per_angle is not None
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def transform_params_to_fourier(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Transform per-angle parameters to Fourier coefficients.

        Parameters
        ----------
        params : np.ndarray
            Full parameter array: [contrast_per_angle... | offset_per_angle... | physical...]

        Returns
        -------
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]
            (transformed_params, (contrast_coeffs, offset_coeffs)) if Fourier active,
            otherwise (params, None).
        """
        if self.fourier is None:
            return params, None

        # Extract per-angle params (first 2*n_phi entries by convention)
        n_per_angle = 2 * self.n_phi
        contrast_per_angle = params[: self.n_phi]
        offset_per_angle = params[self.n_phi : n_per_angle]
        physical = params[n_per_angle:]

        contrast_coeffs = self.fourier.to_fourier(contrast_per_angle)
        offset_coeffs = self.fourier.to_fourier(offset_per_angle)

        transformed = np.concatenate([contrast_coeffs, offset_coeffs, physical])
        return transformed, (contrast_coeffs, offset_coeffs)

    def create_nlsq_callbacks(self) -> dict[str, Any]:
        """Create callbacks for NLSQ's CurveFit integration.

        Returns a dict of callbacks compatible with the heterodyne optimizer loop:
        - 'loss_augmentation': Callable for regularization penalty
        - 'iteration_callback': Callable for gradient monitoring

        Returns
        -------
        dict
            Dictionary of callbacks, empty if controller is not enabled.
        """
        if not self.is_enabled:
            return {}

        callbacks: dict[str, Any] = {}

        # Layer 3: Regularization callback
        if self.regularizer is not None:

            def loss_augmentation(params: np.ndarray, residuals: np.ndarray) -> float:
                """Add regularization penalty to loss.

                The joint parameter vector is laid out as
                ``[physics | per_angle_scaling]`` (see ``core.py``:
                _fit_joint_averaged_multi_phi / _fit_joint_multi_phi).
                Penalize the trailing per-angle block, NOT the leading
                physics block — slicing from the head would regularize
                physics parameters, which is incorrect.

                Post-Codex-review fix: previously sliced ``params[:n_per]``
                which inadvertently penalized the first n_per physics
                parameters.  Now slices ``params[n_physical:n_physical+n_per]``.
                """
                del residuals  # signature contract, not used by this regularizer
                lambda_val = self.regularizer.current_lambda  # type: ignore[union-attr]
                n_per = self.n_per_angle_params
                n_physical = self.n_physical
                if n_per > 0 and len(params) >= n_physical + n_per:
                    per_angle = params[n_physical : n_physical + n_per]
                    reg_term = lambda_val * float(np.var(per_angle))
                    return reg_term
                return 0.0

            callbacks["loss_augmentation"] = loss_augmentation

        # Layer 4: Gradient monitoring callback
        if self.monitor is not None:
            trigger_count: list[int] = [0]

            def iteration_callback(
                iteration: int,
                params: np.ndarray,
                cost: float,
                gradient: np.ndarray | None = None,
            ) -> None:
                """Monitor gradients for collapse detection."""
                if gradient is not None:
                    assert self.monitor is not None
                    self.monitor.record(iteration, gradient)
                    if self.monitor.check_vanishing(
                        threshold=self._gradient_ratio_threshold
                    ):
                        trigger_count[0] += 1
                        if trigger_count[0] >= self._gradient_consecutive_triggers:
                            logger.warning(
                                "ANTI-DEGENERACY Layer 4: Gradient collapse detected "
                                "at iteration %d (cost=%.4e)",
                                iteration,
                                cost,
                            )

            callbacks["iteration_callback"] = iteration_callback

        logger.debug("Created NLSQ callbacks: %s", list(callbacks.keys()))
        return callbacks

    def create_hybrid_streaming_config_kwargs(self) -> dict[str, Any]:
        """Create kwargs for NLSQ's HybridStreamingConfig.

        Returns kwargs that can be used to configure NLSQ's
        AdaptiveHybridStreamingOptimizer with anti-degeneracy features.

        Returns
        -------
        dict
            Configuration kwargs, empty if not enabled.
        """
        if not self.is_enabled:
            return {}

        kwargs: dict[str, Any] = {}

        # Group variance regularization
        reg_indices = getattr(self, "_reg_group_indices", [])
        if reg_indices and self.regularizer:
            kwargs["enable_group_variance_regularization"] = True
            kwargs["group_variance_lambda"] = self.regularizer.current_lambda
            kwargs["group_variance_indices"] = reg_indices

        logger.debug("Created HybridStreamingConfig kwargs: %s", list(kwargs.keys()))
        return kwargs

    def get_group_variance_indices(self) -> list[tuple[int, int]]:
        """Get parameter group indices for variance regularization.

        Returns the stored group indices computed during initialization
        from the parameter vector layout:
        [contrast_params... | offset_params... | physics_params...]

        Returns
        -------
        list of tuple[int, int]
            [(start, end), ...] for each scaling parameter group.
            Empty list when controller is not initialized.
        """
        return getattr(self, "_reg_group_indices", [])

    def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostics from all components.

        Returns
        -------
        dict
            Nested diagnostics from all 4 layers (no Layer 5).
        """
        diag: dict[str, Any] = {
            "version": "2.9.0-het",
            "enabled": self.is_enabled,
            "per_angle_mode": self.config.per_angle_mode,
            "per_angle_mode_actual": self.per_angle_mode_actual,
            "use_constant": self.use_constant,
            "use_fixed_scaling": self.use_fixed_scaling,
            "use_averaged_scaling": self.use_averaged_scaling,
            "use_fourier": self.use_fourier,
            "use_shear_weighting": False,  # Layer 5 always absent
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
            "n_per_angle_params": self.n_per_angle_params,
            "n_total_params": self.n_physical + self.n_per_angle_params,
            "has_fixed_per_angle_scaling": self.has_fixed_per_angle_scaling,
        }

        if self.has_fixed_per_angle_scaling:
            assert self._fixed_contrast_per_angle is not None
            assert self._fixed_offset_per_angle is not None
            diag["fixed_scaling"] = {
                "contrast_mean": float(np.nanmean(self._fixed_contrast_per_angle)),
                "contrast_std": float(np.nanstd(self._fixed_contrast_per_angle)),
                "offset_mean": float(np.nanmean(self._fixed_offset_per_angle)),
                "offset_std": float(np.nanstd(self._fixed_offset_per_angle)),
            }

        if self.fourier:
            diag["fourier"] = self.fourier.get_diagnostics()

        if self.regularizer:
            diag["regularization"] = {
                "lambda": self.regularizer.current_lambda,
                "group_indices": getattr(self, "_reg_group_indices", []),
            }

        if self.monitor:
            diag["gradient_monitor"] = self.monitor.get_summary()

        return diag

    def reset_monitor(self) -> None:
        """Reset the gradient collapse monitor state."""
        if self.monitor is not None:
            self.monitor._history.clear()

    def get_shear_weights(self) -> np.ndarray | None:
        """Layer 5 stub — always returns None in heterodyne (D3-dropped).

        Returns
        -------
        None
            Always None; heterodyne has no shear physics.
        """
        return None

    def get_fixed_per_angle_scaling(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get fixed per-angle contrast and offset estimates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            (contrast_per_angle, offset_per_angle) arrays if available,
            otherwise None.
        """
        if not self.has_fixed_per_angle_scaling:
            return None
        assert self._fixed_contrast_per_angle is not None
        assert self._fixed_offset_per_angle is not None
        return (
            self._fixed_contrast_per_angle.copy(),
            self._fixed_offset_per_angle.copy(),
        )

    def compute_fixed_per_angle_scaling(
        self,
        stratified_data: Any,
        contrast_bounds: tuple[float, float] = (0.0, 1.0),
        offset_bounds: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        """Compute and store fixed per-angle contrast/offset from quantiles.

        This method uses physics-informed quantile analysis to estimate
        contrast and offset for each phi angle independently.

        In "constant" mode:
        1. Computes N contrast + N offset values from quantile estimation
        2. These are averaged to 1 contrast + 1 offset for optimization
        3. The individual per-angle estimates are stored for diagnostics

        Parameters
        ----------
        stratified_data : StratifiedData
            Data containing per-angle g2_flat, phi_flat, t1_flat, t2_flat arrays.
        contrast_bounds : tuple[float, float]
            Valid bounds for contrast parameter.
        offset_bounds : tuple[float, float]
            Valid bounds for offset parameter.
        """
        if not self.use_constant:
            logger.warning(
                "compute_fixed_per_angle_scaling called but not in constant mode; "
                "estimates will be stored but may not be used"
            )

        logger.info("=" * 60)
        logger.info("CONSTANT MODE: Computing fixed per-angle scaling from quantiles")
        logger.info(f"  n_phi: {self.n_phi}")
        logger.info("=" * 60)

        # Compute per-angle contrast/offset from quantile analysis of g2 data
        contrast_per_angle = np.zeros(self.n_phi)
        offset_per_angle = np.zeros(self.n_phi)

        # Group data by phi angle using stratified_data
        phi_flat = np.asarray(stratified_data.phi_flat)
        g2_flat = np.asarray(stratified_data.g2_flat)
        unique_phis = np.sort(np.unique(phi_flat))

        for i, phi_val in enumerate(unique_phis[: self.n_phi]):
            mask = np.isclose(phi_flat, phi_val)
            g2_vals = g2_flat[mask]
            if len(g2_vals) == 0:
                contrast_per_angle[i] = 0.5
                offset_per_angle[i] = 1.0
                continue

            # Physics-informed quantile: g2 baseline ≈ offset, contrast = range
            q_low = float(np.nanquantile(g2_vals, 0.05))
            q_high = float(np.nanquantile(g2_vals, 0.95))
            contrast = float(
                np.clip(q_high - q_low, contrast_bounds[0], contrast_bounds[1])
            )
            offset = float(np.clip(q_low, offset_bounds[0], offset_bounds[1]))
            contrast_per_angle[i] = contrast
            offset_per_angle[i] = offset

        self._fixed_contrast_per_angle = contrast_per_angle
        self._fixed_offset_per_angle = offset_per_angle

        logger.info(
            "Fixed scaling computed: contrast=[%.3f±%.3f], offset=[%.3f±%.3f]",
            float(np.mean(contrast_per_angle)),
            float(np.std(contrast_per_angle)),
            float(np.mean(offset_per_angle)),
            float(np.std(offset_per_angle)),
        )


# ---------------------------------------------------------------------------
# Legacy compatibility: DegeneracyCheck result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DegeneracyCheck:
    """Result of a post-fit degeneracy diagnostic check.

    This dataclass is retained for backward compatibility with code that
    uses the passive-diagnostic API (correlation/bound/plateau checks on
    a completed NLSQResult).  It is NOT part of the active orchestrator
    loop — the active controller operates via ``create_nlsq_callbacks()``
    during optimization, not via post-fit ``check()``.

    Attributes:
        is_degenerate: Whether a degeneracy was detected.
        affected_params: Names of parameters involved.
        message: Human-readable description of the issue.
        suggested_action: Recommended remediation step.
    """

    is_degenerate: bool
    affected_params: list[str] = field(default_factory=list)
    message: str = ""
    suggested_action: str = ""


# ---------------------------------------------------------------------------
# Legacy compatibility: GradientCollapseDetector standalone class
#
# This class is heterodyne-specific (no homodyne equivalent) and is
# retained as a standalone Jacobian-norm-based collapse detector.
# It is distinct from the active controller's Layer 4 (GradientMonitor),
# which tracks per-parameter gradient ratios during the optimizer loop.
# Classified as D3 (heterodyne-only extra class).
# ---------------------------------------------------------------------------


class GradientCollapseDetector:
    """Detect Jacobian collapse during iterative optimization.

    Tracks the Frobenius norm of the Jacobian across successive calls.
    When the norm stays below ``threshold`` for ``window`` consecutive
    calls the optimizer has effectively lost gradient information and a
    re-start or regularization bump should be considered.

    This is a heterodyne-specific standalone class (D3, no homodyne equivalent).
    See ``AntiDegeneracyController`` Layer 4 for the active orchestrator's
    gradient monitoring.

    Parameters
    ----------
    threshold : float
        Frobenius norm below which a call counts as "collapsed".
    window : int
        Number of consecutive sub-threshold calls required before
        :meth:`update` returns ``True``.
    """

    def __init__(self, threshold: float = 1e-8, window: int = 5) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if window < 1:
            raise ValueError("window must be >= 1")
        self._threshold = threshold
        self._window = window
        self._history: deque[float] = deque(maxlen=window)

    def update(self, jacobian: np.ndarray) -> bool:
        """Record the Frobenius norm of *jacobian* and check for collapse.

        Args:
            jacobian: Jacobian matrix of any shape.

        Returns:
            ``True`` if the norm has been below ``threshold`` for at least
            ``window`` consecutive calls; ``False`` otherwise.
        """
        norm = float(np.linalg.norm(jacobian, ord="fro"))
        self._history.append(norm)
        logger.debug("GradientCollapseDetector: Frobenius norm = %.4e", norm)

        if len(self._history) < self._window:
            return False

        collapsed = all(n < self._threshold for n in self._history)
        if collapsed:
            logger.warning(
                "GradientCollapseDetector: Jacobian Frobenius norm has been "
                "below %.2e for %d consecutive calls — gradient collapse detected",
                self._threshold,
                self._window,
            )
        return collapsed

    def reset(self) -> None:
        """Clear accumulated norm history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Legacy compatibility: function-style helpers
#
# These functions are heterodyne-specific (no direct homodyne equivalents).
# In homodyne, equivalent logic lives inside AntiDegeneracyController methods.
# Retained as thin standalone helpers for backward compatibility (D3).
# ---------------------------------------------------------------------------


def suggest_regularization(
    degeneracy_checks: list[DegeneracyCheck],
    base_lambda: float = 1e-4,
) -> float:
    """Suggest a Tikhonov regularization strength based on detected degeneracies.

    The returned value scales linearly with the number of degenerate checks,
    leaving ``base_lambda`` unchanged when no degeneracy is present.

    Args:
        degeneracy_checks: Results from one or more degeneracy checks.
        base_lambda: Baseline regularization strength.

    Returns:
        Suggested regularization coefficient >= ``base_lambda``.
    """
    severity = sum(1 for d in degeneracy_checks if d.is_degenerate)
    if severity == 0:
        return base_lambda
    suggested = base_lambda * (1 + severity)
    logger.debug(
        "suggest_regularization: %d degenerate checks → lambda=%.4e",
        severity,
        suggested,
    )
    return suggested


def compute_effective_lambda(
    base_lambda: float,
    iteration: int,
    decay_rate: float = 0.95,
) -> float:
    """Compute decaying regularization strength.

    Applies an exponential schedule so that regularization pressure is
    strongest early in the optimization and relaxes as the solver converges:

        lambda_eff = base_lambda * decay_rate^iteration

    Args:
        base_lambda: Initial regularization coefficient.
        iteration: Current iteration index (0-based).
        decay_rate: Multiplicative decay per iteration; must be in (0, 1].

    Returns:
        Effective regularization coefficient for this iteration.

    Raises:
        ValueError: If ``decay_rate`` is outside (0, 1].
    """
    if not 0 < decay_rate <= 1:
        raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    effective = base_lambda * (decay_rate**iteration)
    logger.debug(
        "compute_effective_lambda: iter=%d, base=%.4e, decay=%.4f → lambda=%.4e",
        iteration,
        base_lambda,
        decay_rate,
        effective,
    )
    return effective


def detect_hierarchical_trigger(
    degeneracy_checks: list[DegeneracyCheck],
    cost_history: list[float],
) -> bool:
    """Decide whether to invoke the hierarchical fitting strategy.

    Returns ``True`` when *both* of the following conditions hold:

    1. At least one degeneracy check in *degeneracy_checks* reports
       ``is_degenerate=True``.
    2. The last three entries in *cost_history* are within 1 % relative
       change of one another (cost plateau).

    Args:
        degeneracy_checks: Sequence of :class:`DegeneracyCheck` results.
        cost_history: Ordered sequence of scalar cost values across
            optimization iterations.

    Returns:
        ``True`` if both degeneracy and cost plateau are detected;
        ``False`` otherwise.
    """
    # Condition 1: at least one degeneracy detected
    any_degenerate = any(d.is_degenerate for d in degeneracy_checks)
    if not any_degenerate:
        return False

    # Condition 2: cost plateau over the last 3 entries
    if len(cost_history) < 3:
        return False

    last_three = cost_history[-3:]
    ref = abs(last_three[0])
    denom = ref if ref > 1e-30 else 1e-30
    plateaued = all(abs(c - last_three[0]) / denom < 0.01 for c in last_three[1:])
    if not plateaued:
        return False

    logger.info(
        "detect_hierarchical_trigger: degeneracy + cost plateau detected "
        "(last 3 costs: %s) — hierarchical fitting recommended",
        [f"{c:.4e}" for c in last_three],
    )
    return True
