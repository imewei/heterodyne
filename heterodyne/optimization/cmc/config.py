"""Configuration for CMC (Consensus Monte Carlo) analysis.

This module defines CMCConfig, a comprehensive dataclass covering all aspects of
the heterodyne CMC pipeline: sharding strategy, backend selection, NUTS sampling
parameters, convergence validation thresholds, reparameterization, prior tempering,
shard combination, and run identification.

The heterodyne model has 14 free parameters (vs. 7 in homodyne). All auto-scaling
formulas account for this increased dimensionality.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of model parameters for heterodyne (14-parameter model).
_N_PARAMS_HETERODYNE: int = 14

#: Minimum ratio of data points to parameters required per shard.
_MIN_POINTS_PER_PARAM_DEFAULT: int = 1_500

#: Reference shard size used for adaptive sample-count scaling (10 K points → full).
_REFERENCE_SHARD_SIZE: int = 10_000

# Valid string literals for each enumerated field.
_VALID_ENABLE: frozenset[str] = frozenset({"auto", "always", "never"})
_VALID_PER_ANGLE_MODE: frozenset[str] = frozenset(
    {"auto", "constant", "constant_averaged", "individual"}
)
_VALID_SHARDING_STRATEGY: frozenset[str] = frozenset(
    {"stratified", "random", "contiguous"}
)
_VALID_BACKEND_NAME: frozenset[str] = frozenset(
    {"auto", "multiprocessing", "pjit", "cpu"}
)
_VALID_CHAIN_METHOD: frozenset[str] = frozenset({"parallel", "sequential"})
_VALID_INIT_STRATEGY: frozenset[str] = frozenset(
    {"init_to_median", "init_to_sample", "init_to_value"}
)
_VALID_COMBINATION_METHOD: frozenset[str] = frozenset(
    {"consensus_mc", "robust_consensus_mc", "weighted_gaussian", "simple_average"}
)


# ---------------------------------------------------------------------------
# CMCConfig
# ---------------------------------------------------------------------------


@dataclass
class CMCConfig:
    """Comprehensive configuration for Consensus Monte Carlo (CMC) analysis.

    CMC splits a large dataset into K shards, runs NUTS independently on each
    shard, then combines the resulting posteriors using a consensus algorithm.
    This dataclass controls every knob across the full pipeline.

    Parameters are grouped into logical sections, matching the structure of the
    ``to_dict`` / ``from_dict`` serialization format:

    - **enable** — master on/off switch and dataset-size gate.
    - **per_angle** — how to handle the phi (angle) dimension.
    - **sharding** — shard count, strategy, and size bounds.
    - **backend_config** — worker backend and checkpoint settings.
    - **per_shard_mcmc** — NUTS hyper-parameters and adaptive scaling.
    - **validation** — convergence thresholds and abort conditions.
    - **nlsq** — NLSQ warm-start and prior-width configuration.
    - **prior_tempering** — scale priors by 1/K for shard consistency.
    - **combination** — posterior combination algorithm and success criteria.
    - **timeout** — per-shard and heartbeat time limits.
    - **reparameterization** — parameter transforms and bimodality guards.
    - **run_id** — optional identifier for checkpoint namespacing.

    Attributes
    ----------
    enable:
        Master switch. ``"auto"`` enables CMC when ``n_points >= min_points_for_cmc``.
        ``True`` / ``"always"`` forces CMC regardless of dataset size.
        ``False`` / ``"never"`` disables CMC entirely.
    min_points_for_cmc:
        Minimum number of data points required before CMC is activated under
        ``enable="auto"``. Below this threshold the pipeline falls back to full
        NUTS.
    per_angle_mode:
        Strategy for handling the angle (phi) dimension.
        ``"auto"`` selects automatically based on ``n_phi`` and
        ``constant_scaling_threshold``.
    constant_scaling_threshold:
        Minimum number of phi angles required before switching from
        ``"constant"`` to ``"individual"`` mode when ``per_angle_mode="auto"``.
    sharding_strategy:
        How to partition data across shards: ``"stratified"`` preserves angle
        distributions, ``"random"`` shuffles globally, ``"contiguous"`` uses
        contiguous memory blocks.
    num_shards:
        Number of shards ``K``. ``"auto"`` derives ``K`` from dataset size,
        phi count, and ``min_points_per_shard`` / ``min_points_per_param``.
    max_points_per_shard:
        Upper bound on shard size. ``"auto"`` disables the cap.
    min_points_per_shard:
        Lower bound on shard size; prevents degenerate under-determined shards.
    min_points_per_param:
        Minimum ratio of points-to-parameters per shard (heterodyne default: 14).
    backend_name:
        Worker backend. ``"auto"`` selects based on available CPU devices
        and core count.
    enable_checkpoints:
        Persist intermediate shard results to ``checkpoint_dir``.
    checkpoint_dir:
        Directory for shard checkpoint files.
    chain_method:
        Whether to run chains in ``"parallel"`` or ``"sequential"`` order within
        each shard worker.
    num_warmup:
        Number of NUTS warm-up (burn-in) steps per chain.
    num_samples:
        Number of posterior draws per chain after warm-up.
    num_chains:
        Number of independent MCMC chains per shard.
    target_accept_prob:
        Target acceptance probability for the dual-averaging NUTS step-size
        adaptation (must be in ``[0.5, 0.99]``).
    max_tree_depth:
        Maximum binary tree depth for NUTS leapfrog integration.
    seed:
        Base random seed. ``None`` uses a non-deterministic seed.
    dense_mass:
        Use a dense (full-covariance) mass matrix. Expensive but more
        accurate for highly correlated posteriors.
    init_strategy:
        NUTS initialisation strategy.
    adaptive_sampling:
        Scale ``num_warmup`` / ``num_samples`` down proportionally when shard
        size is below ``_REFERENCE_SHARD_SIZE``.
    min_warmup:
        Floor on adaptive warm-up count.
    min_samples:
        Floor on adaptive sample count.
    max_r_hat:
        Maximum acceptable Gelman-Rubin statistic; chains with ``R-hat >
        max_r_hat`` are flagged as not converged.
    min_ess:
        Minimum effective sample size per parameter.
    min_bfmi:
        Minimum Bayesian Fraction of Missing Information (energy diagnostic).
    max_divergence_rate:
        Maximum fraction of divergent transitions before a shard is rejected.
    require_nlsq_warmstart:
        Abort if an NLSQ warm-start was requested but unavailable.
    max_parameter_cv:
        Maximum allowed coefficient of variation across chains for any
        parameter; guards against pathological multi-modal posteriors.
    heterogeneity_abort:
        Abort the entire CMC run if shards produce incompatible posteriors
        (detected via KL divergence or parameter-CV checks).
    use_nlsq_warmstart:
        Initialise each shard's NUTS chains from the NLSQ MAP estimate.
    use_nlsq_informed_priors:
        Centre Gaussian priors on NLSQ estimates scaled by
        ``nlsq_prior_width_factor``.
    nlsq_prior_width_factor:
        Scale factor applied to NLSQ parameter uncertainties when constructing
        informed priors.
    prior_tempering:
        Divide log-prior by ``K`` (number of shards) so that the combined
        posterior approximates the full-data prior exactly.
    combination_method:
        Algorithm used to combine shard posteriors.
    min_success_rate:
        Minimum fraction of shards that must converge; run fails below this.
    min_success_rate_warning:
        Fraction below which a warning is emitted even if the run succeeds.
    per_shard_timeout:
        Wall-clock seconds allowed per shard before it is cancelled.
    heartbeat_timeout:
        Seconds of silence from a worker before it is declared dead.
    use_reparam:
        Apply parameter reparameterisations (e.g. log-transforms) in NumPyro.
    reparameterization_d_total:
        Reparameterise ``d_total = d_fast + d_slow`` as an unconstrained sum.
    reparameterization_log_gamma:
        Reparameterise ``gamma`` on a log scale to enforce positivity.
    bimodal_min_weight:
        Minimum mixture weight for the minor mode in bimodal posteriors; below
        this the minor mode is discarded.
    bimodal_min_separation:
        Minimum normalised distance between modes to declare bimodality.
    run_id:
        Optional string identifier for this CMC run, used in checkpoint paths
        and log messages.
    """

    # ------------------------------------------------------------------
    # 1. Enable & dataset-size gate
    # ------------------------------------------------------------------

    enable: bool | str = "auto"
    min_points_for_cmc: int = 100_000

    # ------------------------------------------------------------------
    # 2. Per-angle mode
    # ------------------------------------------------------------------

    per_angle_mode: str = "auto"
    constant_scaling_threshold: int = 3

    # ------------------------------------------------------------------
    # 3. Sharding
    # ------------------------------------------------------------------

    sharding_strategy: str = "random"
    num_shards: int | str = "auto"
    max_points_per_shard: int | str = "auto"
    min_points_per_shard: int = 10_000
    min_points_per_param: int = _MIN_POINTS_PER_PARAM_DEFAULT

    # ------------------------------------------------------------------
    # 4. Backend
    # ------------------------------------------------------------------

    backend_name: str = "auto"
    enable_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/cmc"
    chain_method: str = "parallel"

    # ------------------------------------------------------------------
    # 5. Sampling
    # ------------------------------------------------------------------

    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    seed: int | None = None
    dense_mass: bool = False
    init_strategy: str = "init_to_median"
    adaptive_sampling: bool = True
    min_warmup: int = 100
    min_samples: int = 200

    # ------------------------------------------------------------------
    # 6. Validation thresholds
    # ------------------------------------------------------------------

    max_r_hat: float = 1.1
    min_ess: int = 100
    min_bfmi: float = 0.3
    max_divergence_rate: float = 0.10
    require_nlsq_warmstart: bool = False
    max_parameter_cv: float = 1.0
    heterogeneity_abort: bool = True

    # ------------------------------------------------------------------
    # 7. NLSQ-informed priors
    # ------------------------------------------------------------------

    use_nlsq_warmstart: bool = True
    use_nlsq_informed_priors: bool = True
    nlsq_prior_width_factor: float = 2.0

    # ------------------------------------------------------------------
    # 8. Prior tempering
    # ------------------------------------------------------------------

    prior_tempering: bool = True

    # ------------------------------------------------------------------
    # 9. Combination
    # ------------------------------------------------------------------

    combination_method: str = "robust_consensus_mc"
    min_success_rate: float = 0.90
    min_success_rate_warning: float = 0.80

    # ------------------------------------------------------------------
    # 10. Timeout
    # ------------------------------------------------------------------

    per_shard_timeout: int = 3600
    heartbeat_timeout: int = 600

    # ------------------------------------------------------------------
    # 11. Reparameterization
    # ------------------------------------------------------------------

    use_reparam: bool = True
    reparameterization_d_total: bool = True
    reparameterization_log_gamma: bool = True
    bimodal_min_weight: float = 0.2
    bimodal_min_separation: float = 0.5

    # ------------------------------------------------------------------
    # 12. Run identification
    # ------------------------------------------------------------------

    run_id: str | None = None

    # ------------------------------------------------------------------
    # Private: accumulated validation errors (not shown in repr)
    # ------------------------------------------------------------------

    _validation_errors: list[str] = field(default_factory=list, repr=False)

    # ==================================================================
    # Post-init
    # ==================================================================

    def __post_init__(self) -> None:
        """Normalise string-valued enable flag and log construction."""
        # Coerce boolean True/False to canonical string forms.
        if self.enable is True:
            self.enable = "always"
        elif self.enable is False:
            self.enable = "never"

        # Coerce removed "gpu" backend to "auto" (heterodyne is CPU-only)
        if self.backend_name == "gpu":
            import warnings

            warnings.warn(
                "backend_name='gpu' is not supported; heterodyne is CPU-only. "
                "Falling back to 'auto'.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.backend_name = "auto"

        logger.debug(
            "CMCConfig constructed: enable=%s num_shards=%s backend=%s",
            self.enable,
            self.num_shards,
            self.backend_name,
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(self) -> list[str]:
        """Run comprehensive field validation and return a list of error strings.

        Returns
        -------
        list[str]
            Empty list when the configuration is valid; one entry per violation
            otherwise.  Does *not* raise — callers decide how to handle errors.
        """
        errors: list[str] = []

        # ---- enable ---------------------------------------------------
        enable_str = str(self.enable).lower()
        if enable_str not in _VALID_ENABLE:
            errors.append(
                f"enable={self.enable!r} is not valid; "
                f"must be one of {sorted(_VALID_ENABLE)!r} or a bool."
            )

        # ---- min_points_for_cmc ---------------------------------------
        if self.min_points_for_cmc < 1:
            errors.append(f"min_points_for_cmc={self.min_points_for_cmc} must be >= 1.")

        # ---- per_angle_mode -------------------------------------------
        if self.per_angle_mode not in _VALID_PER_ANGLE_MODE:
            errors.append(
                f"per_angle_mode={self.per_angle_mode!r} is not valid; "
                f"must be one of {sorted(_VALID_PER_ANGLE_MODE)!r}."
            )

        if self.constant_scaling_threshold < 1:
            errors.append(
                f"constant_scaling_threshold={self.constant_scaling_threshold} "
                "must be >= 1."
            )

        # ---- sharding -------------------------------------------------
        if self.sharding_strategy not in _VALID_SHARDING_STRATEGY:
            errors.append(
                f"sharding_strategy={self.sharding_strategy!r} is not valid; "
                f"must be one of {sorted(_VALID_SHARDING_STRATEGY)!r}."
            )

        if isinstance(self.num_shards, int) and self.num_shards < 1:
            errors.append(
                f"num_shards={self.num_shards} must be >= 1 when set explicitly."
            )
        elif isinstance(self.num_shards, str) and self.num_shards != "auto":
            errors.append(f"num_shards={self.num_shards!r} must be an int or 'auto'.")

        if isinstance(self.max_points_per_shard, int):
            if self.max_points_per_shard < 1:
                errors.append(
                    f"max_points_per_shard={self.max_points_per_shard} must be >= 1."
                )
        elif (
            isinstance(self.max_points_per_shard, str)
            and self.max_points_per_shard != "auto"
        ):
            errors.append(
                f"max_points_per_shard={self.max_points_per_shard!r} must be an int "
                "or 'auto'."
            )

        if self.min_points_per_shard < 1:
            errors.append(
                f"min_points_per_shard={self.min_points_per_shard} must be >= 1."
            )

        if self.min_points_per_param < 1:
            errors.append(
                f"min_points_per_param={self.min_points_per_param} must be >= 1."
            )

        # Cross-check: if both are explicit integers, min <= max.
        if isinstance(self.min_points_per_shard, int) and isinstance(
            self.max_points_per_shard, int
        ):
            if self.min_points_per_shard > self.max_points_per_shard:
                errors.append(
                    f"min_points_per_shard={self.min_points_per_shard} exceeds "
                    f"max_points_per_shard={self.max_points_per_shard}."
                )

        # ---- backend --------------------------------------------------
        if self.backend_name not in _VALID_BACKEND_NAME:
            errors.append(
                f"backend_name={self.backend_name!r} is not valid; "
                f"must be one of {sorted(_VALID_BACKEND_NAME)!r}."
            )

        if self.chain_method not in _VALID_CHAIN_METHOD:
            errors.append(
                f"chain_method={self.chain_method!r} is not valid; "
                f"must be one of {sorted(_VALID_CHAIN_METHOD)!r}."
            )

        # ---- sampling -------------------------------------------------
        if self.num_warmup < 1:
            errors.append(f"num_warmup={self.num_warmup} must be >= 1.")

        if self.num_samples < 1:
            errors.append(f"num_samples={self.num_samples} must be >= 1.")

        if self.num_chains < 1:
            errors.append(f"num_chains={self.num_chains} must be >= 1.")

        if not (0.5 <= self.target_accept_prob <= 0.99):
            errors.append(
                f"target_accept_prob={self.target_accept_prob} must be in [0.5, 0.99]."
            )

        if self.max_tree_depth < 1:
            errors.append(f"max_tree_depth={self.max_tree_depth} must be >= 1.")

        if self.init_strategy not in _VALID_INIT_STRATEGY:
            errors.append(
                f"init_strategy={self.init_strategy!r} is not valid; "
                f"must be one of {sorted(_VALID_INIT_STRATEGY)!r}."
            )

        if self.min_warmup < 1:
            errors.append(f"min_warmup={self.min_warmup} must be >= 1.")

        if self.min_samples < 1:
            errors.append(f"min_samples={self.min_samples} must be >= 1.")

        if self.min_warmup > self.num_warmup:
            errors.append(
                f"min_warmup={self.min_warmup} must be <= num_warmup={self.num_warmup}."
            )

        if self.min_samples > self.num_samples:
            errors.append(
                f"min_samples={self.min_samples} must be <= num_samples={self.num_samples}."
            )

        # ---- validation thresholds ------------------------------------
        if self.max_r_hat <= 1.0:
            errors.append(
                f"max_r_hat={self.max_r_hat} must be > 1.0 "
                "(R-hat is always >= 1 by definition; threshold must exceed 1.0)."
            )

        if self.min_ess < 1:
            errors.append(f"min_ess={self.min_ess} must be >= 1.")

        if not (0.0 < self.min_bfmi <= 1.0):
            errors.append(f"min_bfmi={self.min_bfmi} must be in (0, 1].")

        if not (0.0 <= self.max_divergence_rate <= 1.0):
            errors.append(
                f"max_divergence_rate={self.max_divergence_rate} must be in [0, 1]."
            )

        if self.max_parameter_cv <= 0.0:
            errors.append(f"max_parameter_cv={self.max_parameter_cv} must be > 0.")

        # ---- NLSQ priors ----------------------------------------------
        if self.nlsq_prior_width_factor <= 0.0:
            errors.append(
                f"nlsq_prior_width_factor={self.nlsq_prior_width_factor} must be > 0."
            )

        # ---- combination ----------------------------------------------
        if self.combination_method not in _VALID_COMBINATION_METHOD:
            errors.append(
                f"combination_method={self.combination_method!r} is not valid; "
                f"must be one of {sorted(_VALID_COMBINATION_METHOD)!r}."
            )

        if not (0.0 < self.min_success_rate <= 1.0):
            errors.append(
                f"min_success_rate={self.min_success_rate} must be in (0, 1]."
            )

        if not (0.0 < self.min_success_rate_warning <= 1.0):
            errors.append(
                f"min_success_rate_warning={self.min_success_rate_warning} "
                "must be in (0, 1]."
            )

        if self.min_success_rate_warning > self.min_success_rate:
            errors.append(
                f"min_success_rate_warning={self.min_success_rate_warning} "
                f"must be <= min_success_rate={self.min_success_rate}."
            )

        # ---- timeout --------------------------------------------------
        if self.per_shard_timeout < 1:
            errors.append(
                f"per_shard_timeout={self.per_shard_timeout} must be >= 1 second."
            )

        if self.heartbeat_timeout < 1:
            errors.append(
                f"heartbeat_timeout={self.heartbeat_timeout} must be >= 1 second."
            )

        if self.heartbeat_timeout > self.per_shard_timeout:
            errors.append(
                f"heartbeat_timeout={self.heartbeat_timeout} must be <= "
                f"per_shard_timeout={self.per_shard_timeout}."
            )

        # ---- reparameterization ---------------------------------------
        if not (0.0 < self.bimodal_min_weight < 0.5):
            errors.append(
                f"bimodal_min_weight={self.bimodal_min_weight} must be in (0, 0.5); "
                "it represents the minor mixture component weight."
            )

        if self.bimodal_min_separation <= 0.0:
            errors.append(
                f"bimodal_min_separation={self.bimodal_min_separation} must be > 0."
            )

        # Cache for is_valid() fast path.
        self._validation_errors = errors

        if errors:
            logger.warning(
                "CMCConfig validation found %d error(s): %s",
                len(errors),
                "; ".join(errors),
            )

        return errors

    def is_valid(self) -> bool:
        """Return True if the configuration passes all validation checks.

        Equivalent to ``len(self.validate()) == 0``.

        Returns
        -------
        bool
        """
        return len(self.validate()) == 0

    # ==================================================================
    # Runtime queries
    # ==================================================================

    def should_enable_cmc(self, n_points: int) -> bool:
        """Decide whether to run CMC given the dataset size.

        Parameters
        ----------
        n_points:
            Total number of data points in the dataset.

        Returns
        -------
        bool
            ``True`` if CMC should run for this dataset.
        """
        enable_str = str(self.enable).lower()

        if enable_str in {"always", "true"}:
            logger.debug("CMC enabled unconditionally (enable=%r).", self.enable)
            return True

        if enable_str in {"never", "false"}:
            logger.debug("CMC disabled unconditionally (enable=%r).", self.enable)
            return False

        # "auto" branch — gate on minimum dataset size.
        if n_points >= self.min_points_for_cmc:
            logger.debug(
                "CMC auto-enabled: n_points=%d >= min_points_for_cmc=%d.",
                n_points,
                self.min_points_for_cmc,
            )
            return True

        logger.info(
            "CMC auto-disabled: n_points=%d < min_points_for_cmc=%d.",
            n_points,
            self.min_points_for_cmc,
        )
        return False

    def get_num_shards(
        self,
        n_points: int,
        n_phi: int,
        n_params: int = _N_PARAMS_HETERODYNE,
    ) -> int:
        """Compute the number of shards K for a given dataset.

        When ``num_shards`` is an explicit integer it is returned directly
        (clamped to >= 1).  When ``"auto"``, K is derived as:

        1. Start from ``max(n_phi, 2)`` — at least as many shards as phi angles.
        2. Apply the ``min_points_per_shard`` lower bound:
           ``K <= n_points // min_points_per_shard``.
        3. Apply the ``min_points_per_param`` constraint:
           ``K <= n_points // (n_params * min_points_per_param)``.
        4. Apply the ``max_points_per_shard`` upper bound when set:
           ``K >= ceil(n_points / max_points_per_shard)``.
        5. Clamp to ``[1, n_points]``.

        Parameters
        ----------
        n_points:
            Total number of data points.
        n_phi:
            Number of distinct phi (azimuthal angle) bins.
        n_params:
            Number of free model parameters (default: 14 for heterodyne).

        Returns
        -------
        int
            Number of shards K >= 1.
        """
        if isinstance(self.num_shards, int):
            k = max(1, self.num_shards)
            logger.debug("Using explicit num_shards=%d.", k)
            return k

        # Auto computation.
        if n_points < 1:
            logger.warning(
                "get_num_shards called with n_points=%d; returning K=1.", n_points
            )
            return 1

        # Lower bound from phi structure: at least one shard per angle group.
        k_from_phi = max(n_phi, 2)

        # Upper bound from min shard size.
        k_max_from_min_size = n_points // max(self.min_points_per_shard, 1)

        # Upper bound from points-per-parameter constraint.
        min_shard_size_for_params = n_params * self.min_points_per_param
        k_max_from_params = n_points // max(min_shard_size_for_params, 1)

        k_upper = min(k_max_from_min_size, k_max_from_params)

        # Lower bound from max_points_per_shard cap.
        if isinstance(self.max_points_per_shard, int):
            k_min_from_max_size = math.ceil(
                n_points / max(self.max_points_per_shard, 1)
            )
        else:
            k_min_from_max_size = 1

        # Combine: start from phi suggestion, respect all bounds.
        k = max(k_from_phi, k_min_from_max_size)
        k = min(k, max(k_upper, 1))
        k = max(k, 1)

        logger.debug(
            "Auto num_shards: n_points=%d n_phi=%d n_params=%d → K=%d "
            "(k_from_phi=%d k_max_size=%d k_max_params=%d k_min_cap=%d).",
            n_points,
            n_phi,
            n_params,
            k,
            k_from_phi,
            k_max_from_min_size,
            k_max_from_params,
            k_min_from_max_size,
        )
        return k

    def get_adaptive_sample_counts(
        self,
        shard_size: int,
        n_params: int = _N_PARAMS_HETERODYNE,
    ) -> tuple[int, int]:
        """Scale warmup and sample counts for a given shard size.

        When ``adaptive_sampling=False`` the configured ``num_warmup`` and
        ``num_samples`` are returned unchanged.

        The scaling law is:

        .. code-block:: text

            scale = clamp(shard_size / reference_size, 0, 1)
            warmup = max(min_warmup, round(num_warmup * scale))
            samples = max(min_samples, round(num_samples * scale))

        where ``reference_size = _REFERENCE_SHARD_SIZE`` (10 000 points) is the
        shard size at which the full configured counts are used.  Larger shards
        are *not* scaled up beyond the configured maximum; the formula saturates
        at ``scale = 1``.

        A secondary check ensures a minimum of ``n_params`` samples are drawn
        (ESS cannot exceed ``num_samples * num_chains``).

        Parameters
        ----------
        shard_size:
            Number of data points in this shard.
        n_params:
            Number of model parameters (default: 14).

        Returns
        -------
        tuple[int, int]
            ``(warmup, samples)`` after adaptive scaling.
        """
        if not self.adaptive_sampling or shard_size >= _REFERENCE_SHARD_SIZE:
            return self.num_warmup, self.num_samples

        scale = max(0.0, shard_size / _REFERENCE_SHARD_SIZE)

        warmup = max(self.min_warmup, round(self.num_warmup * scale))
        samples = max(self.min_samples, round(self.num_samples * scale))

        # Absolute floor: at least n_params samples per chain.
        samples = max(samples, n_params)

        logger.debug(
            "Adaptive sampling: shard_size=%d scale=%.3f → warmup=%d samples=%d.",
            shard_size,
            scale,
            warmup,
            samples,
        )
        return warmup, samples

    def get_effective_per_angle_mode(
        self,
        n_phi: int,
        nlsq_per_angle_mode: str | None = None,
        has_nlsq_warmstart: bool = False,
    ) -> str:
        """Resolve the effective per-angle mode for a concrete dataset.

        Resolution logic (in priority order):

        1. If ``per_angle_mode != "auto"`` the configured value is returned
           directly (no override from NLSQ).
        2. If ``per_angle_mode == "auto"``:

           a. If ``has_nlsq_warmstart`` and ``nlsq_per_angle_mode`` is one of
              the valid non-auto modes, inherit it from NLSQ.
           b. Else if ``n_phi >= constant_scaling_threshold`` → ``"individual"``.
           c. Else → ``"constant"``.

        Parameters
        ----------
        n_phi:
            Number of distinct phi (azimuthal angle) bins in the dataset.
        nlsq_per_angle_mode:
            The per-angle mode resolved by the preceding NLSQ fit, if any.
        has_nlsq_warmstart:
            Whether a valid NLSQ warm-start is available for this run.

        Returns
        -------
        str
            Resolved per-angle mode (never ``"auto"``).
        """
        if self.per_angle_mode != "auto":
            logger.debug("Per-angle mode fixed to %r (not auto).", self.per_angle_mode)
            return self.per_angle_mode

        # --- Auto resolution ---
        valid_non_auto = _VALID_PER_ANGLE_MODE - {"auto"}

        if (
            has_nlsq_warmstart
            and nlsq_per_angle_mode is not None
            and nlsq_per_angle_mode in valid_non_auto
        ):
            logger.debug(
                "Per-angle mode auto-resolved to %r from NLSQ warm-start.",
                nlsq_per_angle_mode,
            )
            return nlsq_per_angle_mode

        if n_phi >= self.constant_scaling_threshold:
            resolved = "individual"
        else:
            resolved = "constant"

        logger.debug(
            "Per-angle mode auto-resolved to %r (n_phi=%d threshold=%d).",
            resolved,
            n_phi,
            self.constant_scaling_threshold,
        )
        return resolved

    # ==================================================================
    # Serialization
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration to a nested dictionary.

        The returned structure uses the same section names expected by
        ``from_dict``, making round-trips lossless.

        Returns
        -------
        dict[str, Any]
            Nested dictionary representation of the config.
        """
        return {
            "enable": self.enable,
            "min_points_for_cmc": self.min_points_for_cmc,
            "run_id": self.run_id,
            "per_angle": {
                "per_angle_mode": self.per_angle_mode,
                "constant_scaling_threshold": self.constant_scaling_threshold,
            },
            "sharding": {
                "sharding_strategy": self.sharding_strategy,
                "num_shards": self.num_shards,
                "max_points_per_shard": self.max_points_per_shard,
                "min_points_per_shard": self.min_points_per_shard,
                "min_points_per_param": self.min_points_per_param,
            },
            "backend_config": {
                "backend_name": self.backend_name,
                "enable_checkpoints": self.enable_checkpoints,
                "checkpoint_dir": self.checkpoint_dir,
                "chain_method": self.chain_method,
            },
            "per_shard_mcmc": {
                "num_warmup": self.num_warmup,
                "num_samples": self.num_samples,
                "num_chains": self.num_chains,
                "target_accept_prob": self.target_accept_prob,
                "max_tree_depth": self.max_tree_depth,
                "seed": self.seed,
                "dense_mass": self.dense_mass,
                "init_strategy": self.init_strategy,
                "adaptive_sampling": self.adaptive_sampling,
                "min_warmup": self.min_warmup,
                "min_samples": self.min_samples,
            },
            "validation": {
                "max_r_hat": self.max_r_hat,
                "min_ess": self.min_ess,
                "min_bfmi": self.min_bfmi,
                "max_divergence_rate": self.max_divergence_rate,
                "require_nlsq_warmstart": self.require_nlsq_warmstart,
                "max_parameter_cv": self.max_parameter_cv,
                "heterogeneity_abort": self.heterogeneity_abort,
            },
            "nlsq": {
                "use_nlsq_warmstart": self.use_nlsq_warmstart,
                "use_nlsq_informed_priors": self.use_nlsq_informed_priors,
                "nlsq_prior_width_factor": self.nlsq_prior_width_factor,
            },
            "prior_tempering": self.prior_tempering,
            "combination": {
                "combination_method": self.combination_method,
                "min_success_rate": self.min_success_rate,
                "min_success_rate_warning": self.min_success_rate_warning,
            },
            "timeout": {
                "per_shard_timeout": self.per_shard_timeout,
                "heartbeat_timeout": self.heartbeat_timeout,
            },
            "reparameterization": {
                "use_reparam": self.use_reparam,
                "reparameterization_d_total": self.reparameterization_d_total,
                "reparameterization_log_gamma": self.reparameterization_log_gamma,
                "bimodal_min_weight": self.bimodal_min_weight,
                "bimodal_min_separation": self.bimodal_min_separation,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CMCConfig:
        """Construct a CMCConfig from a (possibly nested) dictionary.

        Recognised top-level keys and sections:

        - ``enable``, ``min_points_for_cmc``, ``run_id`` — top-level scalars.
        - ``prior_tempering`` — top-level scalar.
        - ``per_angle`` — maps to ``per_angle_mode``, ``constant_scaling_threshold``.
        - ``sharding`` — maps to the five sharding fields.
        - ``backend_config`` — maps to the four backend fields.
        - ``per_shard_mcmc`` — maps to the eleven sampling fields.
        - ``validation`` — maps to the seven validation-threshold fields.
        - ``nlsq`` — maps to the three NLSQ-prior fields.
        - ``combination`` — maps to the three combination fields.
        - ``timeout`` — maps to the two timeout fields.
        - ``reparameterization`` — maps to the five reparam fields.

        Flat (non-nested) dictionaries are also accepted for backward
        compatibility: any key that matches a field name directly is used as-is.

        Unrecognised top-level keys emit a ``warnings.warn`` so that
        configuration typos surface immediately.

        Parameters
        ----------
        config_dict:
            Parsed YAML / JSON dictionary.

        Returns
        -------
        CMCConfig
            Fully constructed configuration instance.
        """
        kwargs: dict[str, Any] = {}

        # --- Helpers ---------------------------------------------------

        def _extract_section(section_key: str) -> dict[str, Any]:
            val = config_dict.get(section_key, {})
            if not isinstance(val, dict):
                warnings.warn(
                    f"CMCConfig.from_dict: section {section_key!r} is not a dict "
                    f"(got {type(val).__name__!r}); ignoring.",
                    stacklevel=3,
                )
                return {}
            return val

        def _pick(
            target_field: str,
            source: dict[str, Any],
            source_key: str | None = None,
        ) -> None:
            key = source_key if source_key is not None else target_field
            if key in source:
                kwargs[target_field] = source[key]

        # --- Top-level scalars -----------------------------------------
        _pick("enable", config_dict)
        _pick("min_points_for_cmc", config_dict)
        _pick("run_id", config_dict)
        _pick("prior_tempering", config_dict)

        # --- per_angle section -----------------------------------------
        per_angle = _extract_section("per_angle")
        _pick("per_angle_mode", per_angle)
        _pick("constant_scaling_threshold", per_angle)
        # Flat fallback
        _pick("per_angle_mode", config_dict)
        _pick("constant_scaling_threshold", config_dict)

        # --- sharding section ------------------------------------------
        sharding = _extract_section("sharding")
        _pick("sharding_strategy", sharding)
        _pick("num_shards", sharding)
        _pick("max_points_per_shard", sharding)
        _pick("min_points_per_shard", sharding)
        _pick("min_points_per_param", sharding)
        # Flat fallback
        for _f in (
            "sharding_strategy",
            "num_shards",
            "max_points_per_shard",
            "min_points_per_shard",
            "min_points_per_param",
        ):
            _pick(_f, config_dict)

        # --- backend_config section ------------------------------------
        backend = _extract_section("backend_config")
        _pick("backend_name", backend)
        _pick("enable_checkpoints", backend)
        _pick("checkpoint_dir", backend)
        _pick("chain_method", backend)
        # Flat fallback
        for _f in (
            "backend_name",
            "enable_checkpoints",
            "checkpoint_dir",
            "chain_method",
        ):
            _pick(_f, config_dict)

        # Coerce removed "gpu" backend to "auto" (heterodyne is CPU-only)
        if kwargs.get("backend_name") == "gpu":
            warnings.warn(
                "backend_name='gpu' is not supported; heterodyne is CPU-only. "
                "Falling back to 'auto'.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["backend_name"] = "auto"

        # --- per_shard_mcmc section ------------------------------------
        mcmc = _extract_section("per_shard_mcmc")
        _pick("num_warmup", mcmc)
        _pick("num_samples", mcmc)
        _pick("num_chains", mcmc)
        _pick("target_accept_prob", mcmc)
        # Accept legacy key name from the old config schema.
        if "target_accept" in mcmc and "target_accept_prob" not in kwargs:
            kwargs["target_accept_prob"] = mcmc["target_accept"]
        _pick("max_tree_depth", mcmc)
        _pick("seed", mcmc)
        _pick("dense_mass", mcmc)
        _pick("init_strategy", mcmc)
        _pick("adaptive_sampling", mcmc)
        _pick("min_warmup", mcmc)
        _pick("min_samples", mcmc)
        # Flat fallbacks (including legacy target_accept)
        for _f in (
            "num_warmup",
            "num_samples",
            "num_chains",
            "target_accept_prob",
            "max_tree_depth",
            "seed",
            "dense_mass",
            "init_strategy",
            "adaptive_sampling",
            "min_warmup",
            "min_samples",
        ):
            _pick(_f, config_dict)
        if "target_accept" in config_dict and "target_accept_prob" not in kwargs:
            kwargs["target_accept_prob"] = config_dict["target_accept"]

        # --- validation section ----------------------------------------
        validation = _extract_section("validation")
        _pick("max_r_hat", validation)
        # Accept legacy key name.
        if "r_hat_threshold" in validation and "max_r_hat" not in kwargs:
            kwargs["max_r_hat"] = validation["r_hat_threshold"]
        _pick("min_ess", validation)
        _pick("min_bfmi", validation)
        _pick("max_divergence_rate", validation)
        _pick("require_nlsq_warmstart", validation)
        _pick("max_parameter_cv", validation)
        _pick("heterogeneity_abort", validation)
        # Flat fallbacks
        for _f in (
            "max_r_hat",
            "min_ess",
            "min_bfmi",
            "max_divergence_rate",
            "require_nlsq_warmstart",
            "max_parameter_cv",
            "heterogeneity_abort",
        ):
            _pick(_f, config_dict)
        if "r_hat_threshold" in config_dict and "max_r_hat" not in kwargs:
            kwargs["max_r_hat"] = config_dict["r_hat_threshold"]

        # --- nlsq section ----------------------------------------------
        nlsq = _extract_section("nlsq")
        _pick("use_nlsq_warmstart", nlsq)
        _pick("use_nlsq_informed_priors", nlsq)
        _pick("nlsq_prior_width_factor", nlsq)
        # Accept legacy key name.
        if "prior_width_factor" in nlsq and "nlsq_prior_width_factor" not in kwargs:
            kwargs["nlsq_prior_width_factor"] = nlsq["prior_width_factor"]
        # Flat fallbacks
        for _f in (
            "use_nlsq_warmstart",
            "use_nlsq_informed_priors",
            "nlsq_prior_width_factor",
        ):
            _pick(_f, config_dict)
        if (
            "prior_width_factor" in config_dict
            and "nlsq_prior_width_factor" not in kwargs
        ):
            kwargs["nlsq_prior_width_factor"] = config_dict["prior_width_factor"]

        # --- combination section ---------------------------------------
        combination = _extract_section("combination")
        _pick("combination_method", combination)
        _pick("min_success_rate", combination)
        _pick("min_success_rate_warning", combination)
        # Flat fallbacks
        for _f in (
            "combination_method",
            "min_success_rate",
            "min_success_rate_warning",
        ):
            _pick(_f, config_dict)

        # --- timeout section -------------------------------------------
        timeout = _extract_section("timeout")
        _pick("per_shard_timeout", timeout)
        _pick("heartbeat_timeout", timeout)
        # Flat fallbacks
        for _f in ("per_shard_timeout", "heartbeat_timeout"):
            _pick(_f, config_dict)

        # --- reparameterization section --------------------------------
        reparam = _extract_section("reparameterization")
        _pick("use_reparam", reparam)
        _pick("reparameterization_d_total", reparam)
        _pick("reparameterization_log_gamma", reparam)
        _pick("bimodal_min_weight", reparam)
        _pick("bimodal_min_separation", reparam)
        # Flat fallbacks
        for _f in (
            "use_reparam",
            "reparameterization_d_total",
            "reparameterization_log_gamma",
            "bimodal_min_weight",
            "bimodal_min_separation",
        ):
            _pick(_f, config_dict)

        # --- Warn on unrecognised top-level keys -----------------------
        _known_top_level: frozenset[str] = frozenset(
            {
                "enable",
                "min_points_for_cmc",
                "run_id",
                "prior_tempering",
                "per_angle",
                "sharding",
                "backend_config",
                "per_shard_mcmc",
                "validation",
                "nlsq",
                "combination",
                "timeout",
                "reparameterization",
                # Legacy flat keys accepted above.
                "per_angle_mode",
                "constant_scaling_threshold",
                "sharding_strategy",
                "num_shards",
                "max_points_per_shard",
                "min_points_per_shard",
                "min_points_per_param",
                "backend_name",
                "enable_checkpoints",
                "checkpoint_dir",
                "chain_method",
                "num_warmup",
                "num_samples",
                "num_chains",
                "target_accept_prob",
                "target_accept",
                "max_tree_depth",
                "seed",
                "dense_mass",
                "init_strategy",
                "adaptive_sampling",
                "min_warmup",
                "min_samples",
                "max_r_hat",
                "r_hat_threshold",
                "min_ess",
                "min_bfmi",
                "max_divergence_rate",
                "require_nlsq_warmstart",
                "max_parameter_cv",
                "heterogeneity_abort",
                "use_nlsq_warmstart",
                "use_nlsq_informed_priors",
                "nlsq_prior_width_factor",
                "prior_width_factor",
                "combination_method",
                "min_success_rate",
                "min_success_rate_warning",
                "per_shard_timeout",
                "heartbeat_timeout",
                "use_reparam",
                "reparameterization_d_total",
                "reparameterization_log_gamma",
                "bimodal_min_weight",
                "bimodal_min_separation",
            }
        )

        unknown = sorted(set(config_dict.keys()) - _known_top_level)
        if unknown:
            warnings.warn(
                f"CMCConfig.from_dict: unrecognised key(s) {unknown!r} will be ignored.",
                stacklevel=2,
            )

        logger.debug(
            "CMCConfig.from_dict: constructed with %d kwargs from %d input keys.",
            len(kwargs),
            len(config_dict),
        )
        return cls(**kwargs)
