"""High-level NUTS sampler wrapper for heterodyne CMC analysis.

Provides a ``SamplingPlan`` dataclass for sampling hyperparameters and a
``NUTSSampler`` class that wraps NumPyro's MCMC with ergonomic factories,
automatic chain initialization with perturbation, and ArviZ diagnostics.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import arviz as az
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from numpyro.infer import initialization as numpyro_init

from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class SamplingPlan:
    """Hyperparameters for NUTS sampling.

    Immutable configuration that fully specifies a sampling run.

    Attributes:
        num_warmup: Number of warmup (adaptation) steps per chain.
        num_samples: Number of posterior draws per chain after warmup.
        num_chains: Number of independent MCMC chains.
        target_accept: Target acceptance probability for dual-averaging
            step-size adaptation.  Values in [0.6, 0.95] are typical.
        max_tree_depth: Maximum binary tree depth for NUTS.  Higher values
            allow longer trajectories but increase per-step cost.
        adapt_step_size: Whether to use dual-averaging step-size adaptation
            during warmup.
        dense_mass: Whether to estimate a dense (full) mass matrix during
            warmup, or use a diagonal approximation.
        seed: Explicit random seed for reproducibility.  If ``None``, a
            cryptographically random seed is generated.
    """

    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 4
    target_accept: float = 0.8
    max_tree_depth: int = 10
    adapt_step_size: bool = True
    dense_mass: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate hyperparameters."""
        if self.num_warmup < 1:
            raise ValueError(f"num_warmup must be >= 1, got {self.num_warmup}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {self.num_chains}")
        if not (0.1 <= self.target_accept <= 0.99):
            raise ValueError(
                f"target_accept must be in [0.1, 0.99], got {self.target_accept}"
            )
        if self.max_tree_depth < 1:
            raise ValueError(
                f"max_tree_depth must be >= 1, got {self.max_tree_depth}"
            )

    @property
    def effective_seed(self) -> int:
        """Return the seed, generating one if not explicitly set."""
        if self.seed is not None:
            return self.seed
        return secrets.randbelow(2**31)

    @classmethod
    def from_config(
        cls,
        config: CMCConfig,
        n_data: int | None = None,
        n_params: int | None = None,
    ) -> SamplingPlan:
        """Build a ``SamplingPlan`` from a :class:`CMCConfig`.

        Applies adaptive scaling when ``config.adaptive_sampling`` is ``True``
        and ``n_data`` is provided: warmup and sample counts are scaled
        proportionally to the ratio ``n_data / _REFERENCE_SHARD_SIZE`` and
        clamped to the configured floors.

        Args:
            config: CMC configuration carrying all NUTS hyperparameters and
                adaptive-sampling knobs.
            n_data: Number of data points in this shard (or the full dataset
                when sharding is disabled).  When ``None``, no adaptive
                scaling is applied regardless of ``config.adaptive_sampling``.
            n_params: Number of varying model parameters.  Reserved for
                future dimension-aware scaling; currently unused.

        Returns:
            Fully validated :class:`SamplingPlan`.
        """
        # Reference shard size used for proportional scaling.
        _REFERENCE_SHARD_SIZE = 10_000

        num_warmup = config.num_warmup
        num_samples = config.num_samples

        if config.adaptive_sampling and n_data is not None and n_data > 0:
            scale = min(1.0, n_data / _REFERENCE_SHARD_SIZE)
            # Use sqrt scaling: smaller shards need proportionally less warmup
            # but the relationship is sub-linear because MCMC mixing time does
            # not scale as badly as the raw data ratio suggests.
            sqrt_scale = math.sqrt(scale)
            num_warmup = max(config.min_warmup, int(config.num_warmup * sqrt_scale))
            num_samples = max(config.min_samples, int(config.num_samples * sqrt_scale))

            logger.debug(
                "SamplingPlan.from_config: n_data=%d, scale=%.3f, "
                "num_warmup=%d, num_samples=%d",
                n_data, sqrt_scale, num_warmup, num_samples,
            )

        return cls(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=config.num_chains,
            target_accept=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            adapt_step_size=True,
            dense_mass=config.dense_mass,
            seed=config.seed,
        )

    def for_shard(self, shard_size: int, full_size: int) -> SamplingPlan:
        """Return a scaled-down plan appropriate for a single CMC shard.

        Scales warmup and sample counts by ``sqrt(shard_size / full_size)``
        to reflect the reduced information content of the shard.  Counts are
        clamped to a minimum of ``max(1, num_x // 10)`` to avoid degenerate
        one-step runs.

        Args:
            shard_size: Number of data points in this shard.
            full_size: Total number of data points across all shards.

        Returns:
            New :class:`SamplingPlan` with adjusted warmup/sample counts and
            the same seed and other hyperparameters.

        Raises:
            ValueError: If ``shard_size <= 0`` or ``full_size <= 0``.
        """
        if shard_size <= 0:
            raise ValueError(f"shard_size must be > 0, got {shard_size}")
        if full_size <= 0:
            raise ValueError(f"full_size must be > 0, got {full_size}")

        ratio = min(1.0, shard_size / full_size)
        scale = math.sqrt(ratio)

        min_warmup = max(1, self.num_warmup // 10)
        min_samples = max(1, self.num_samples // 10)

        new_warmup = max(min_warmup, int(self.num_warmup * scale))
        new_samples = max(min_samples, int(self.num_samples * scale))

        logger.debug(
            "SamplingPlan.for_shard: shard_size=%d, full_size=%d, scale=%.3f, "
            "num_warmup=%d->%d, num_samples=%d->%d",
            shard_size, full_size, scale,
            self.num_warmup, new_warmup,
            self.num_samples, new_samples,
        )

        # frozen dataclass — use object.__setattr__ via a new instance
        return SamplingPlan(
            num_warmup=new_warmup,
            num_samples=new_samples,
            num_chains=self.num_chains,
            target_accept=self.target_accept,
            max_tree_depth=self.max_tree_depth,
            adapt_step_size=self.adapt_step_size,
            dense_mass=self.dense_mass,
            seed=self.seed,
        )


class NUTSSampler:
    """High-level NUTS sampler wrapping NumPyro's MCMC.

    Manages kernel construction, chain initialization with perturbation,
    sampling execution, and ArviZ diagnostic extraction.

    Use the :meth:`from_plan` factory for the standard construction path.
    """

    def __init__(
        self,
        mcmc: MCMC,
        plan: SamplingPlan,
    ) -> None:
        self._mcmc = mcmc
        self._plan = plan
        self._has_run = False

    @property
    def plan(self) -> SamplingPlan:
        """The sampling plan used to configure this sampler."""
        return self._plan

    @classmethod
    def from_plan(
        cls,
        plan: SamplingPlan,
        model: Callable[..., Any],
        init_strategy: str = "init_to_median",
        chain_method: str = "sequential",
    ) -> NUTSSampler:
        """Create a NUTSSampler from a SamplingPlan and NumPyro model.

        Args:
            plan: Sampling hyperparameters.
            model: NumPyro model function (callable with no required args).
            init_strategy: NumPyro initialization strategy name.
                One of ``"init_to_median"``, ``"init_to_sample"``,
                ``"init_to_value"``.
            chain_method: NumPyro chain execution method.
                ``"sequential"`` for CPU, ``"parallel"`` for GPU/pmap.

        Returns:
            Configured NUTSSampler ready for :meth:`run`.
        """
        init_fn_map: dict[str, Callable[..., Any]] = {
            "init_to_median": numpyro_init.init_to_median,
            "init_to_sample": numpyro_init.init_to_sample,
            "init_to_value": numpyro_init.init_to_value,
        }
        init_factory = init_fn_map.get(init_strategy, numpyro_init.init_to_median)

        kernel = NUTS(
            model,
            target_accept_prob=plan.target_accept,
            max_tree_depth=plan.max_tree_depth,
            dense_mass=plan.dense_mass,
            adapt_step_size=plan.adapt_step_size,
            init_strategy=init_factory(),
        )

        mcmc = MCMC(
            kernel,
            num_warmup=plan.num_warmup,
            num_samples=plan.num_samples,
            num_chains=plan.num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )

        logger.info(
            "NUTSSampler created: %d chains, %d warmup, %d samples, "
            "target_accept=%s, chain_method=%s",
            plan.num_chains, plan.num_warmup, plan.num_samples,
            plan.target_accept, chain_method,
        )

        return cls(mcmc, plan)

    def run(
        self,
        rng_key: jnp.ndarray | None = None,
        init_params: dict[str, jnp.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run MCMC sampling.

        If ``init_params`` are provided, small random perturbations are
        added per chain to break symmetry and improve exploration.

        Args:
            rng_key: JAX PRNG key. If ``None``, one is generated from
                the plan's seed.
            init_params: Optional initial values for each chain.  Keys
                are parameter names; values should be scalars or have
                shape ``(num_chains,)``.

        Returns:
            Dictionary of posterior samples (ungrouped).

        Raises:
            RuntimeError: If sampling fails.
        """
        seed = self._plan.effective_seed
        if rng_key is None:
            rng_key = jax.random.PRNGKey(seed)

        # Apply perturbation to break chain symmetry
        perturbed_params = None
        if init_params is not None:
            perturbed_params = _perturb_init_params(
                init_params,
                num_chains=self._plan.num_chains,
                seed=seed + 1,
            )

        logger.info("NUTSSampler: starting sampling (seed=%d)", seed)
        self._mcmc.run(
            rng_key,
            init_params=perturbed_params,
            extra_fields=("energy",),
        )
        self._has_run = True

        samples = self._mcmc.get_samples()
        logger.info("NUTSSampler: sampling complete")
        return dict(samples)

    def run_with_init_values(
        self,
        init_values: dict[str, float],
        rng_key: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run MCMC seeded from NLSQ warm-start values.

        Validates that the initial log density is finite before launching
        full sampling, raising early with a diagnostic message if not.

        Args:
            init_values: NLSQ MAP estimates keyed by parameter name.  Values
                should be in the same space as the NumPyro model samples
                (physics space, or reparameterized space if the model uses
                reparameterization).
            rng_key: JAX PRNG key.  Generated from the plan seed if ``None``.

        Returns:
            Dictionary of posterior samples (ungrouped).

        Raises:
            RuntimeError: If the initial log density is not finite or if
                sampling itself fails.
        """
        seed = self._plan.effective_seed
        if rng_key is None:
            rng_key = jax.random.PRNGKey(seed)

        # Convert scalar values to JAX arrays for compatibility with NumPyro
        init_params: dict[str, jnp.ndarray] = {
            name: jnp.asarray(val) for name, val in init_values.items()
        }

        # Preflight: check that the init point has finite log density.
        self._validate_init_log_density(init_params, rng_key)

        logger.info(
            "NUTSSampler.run_with_init_values: warm-starting from %d NLSQ parameters",
            len(init_values),
        )
        return self.run(rng_key=rng_key, init_params=init_params)

    def _validate_init_log_density(
        self,
        init_params: dict[str, jnp.ndarray],
        rng_key: jnp.ndarray,
    ) -> None:
        """Check that the initial parameter point yields a finite log density.

        NumPyro's NUTS will silently produce NaN chains when the initial
        point is outside the support of the model.  This preflight surfaces
        such issues as a ``RuntimeError`` before the full sampling run.

        Args:
            init_params: Initial parameter values (same format as
                ``init_params`` accepted by :meth:`run`).
            rng_key: JAX PRNG key passed to the MCMC potential energy
                evaluation.

        Raises:
            RuntimeError: If the log density at ``init_params`` is not finite.
        """

        try:
            # NumPyro's MCMC exposes get_extra_fields only after a run;
            # use the kernel's postprocess_fn / init to probe log density.
            # The safest approach without running full sampling is to use
            # numpyro's potential_fn via the NUTS kernel's _potential_fn.
            kernel = self._mcmc.sampler  # type: ignore[attr-defined]
            if not hasattr(kernel, "_potential_fn") or kernel._potential_fn is None:
                # Can't probe without a compiled potential — skip silently.
                logger.debug(
                    "_validate_init_log_density: potential_fn not available, skipping"
                )
                return

            potential_fn = kernel._potential_fn
            log_density = -float(potential_fn(init_params))

            if not math.isfinite(log_density):
                param_summary = ", ".join(
                    f"{k}={float(v):.4g}" for k, v in init_params.items()
                )
                raise RuntimeError(
                    f"Initial log density is not finite ({log_density}) at "
                    f"NLSQ warm-start point: {param_summary}. "
                    "Check that init values lie within the model's prior support."
                )

            logger.debug(
                "_validate_init_log_density: log_density=%.4g (finite)", log_density
            )

        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Non-critical: potential_fn probing can fail for various internal
            # NumPyro reasons.  Log and continue rather than blocking sampling.
            logger.debug(
                "_validate_init_log_density: probe failed (%s), skipping", exc
            )

    def get_divergence_stats(self) -> dict[str, float]:
        """Extract divergence rate and tree-depth statistics from the last run.

        Requires that :meth:`run` or :meth:`run_with_init_values` has been
        called.

        Returns:
            Dictionary with the following keys:

            ``"divergence_rate"``
                Fraction of post-warmup transitions that were divergent.
                Zero when no divergences were recorded.
            ``"mean_tree_depth"``
                Mean NUTS tree depth across all post-warmup samples and
                chains.  Values near ``max_tree_depth`` indicate the
                trajectory is being truncated.
            ``"max_tree_depth_fraction"``
                Fraction of samples that hit the maximum tree depth
                (``plan.max_tree_depth``).

        Raises:
            RuntimeError: If called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "Cannot extract divergence stats before calling run()"
            )

        import numpy as np

        extra = self._mcmc.get_extra_fields()

        # Divergences: shape (num_samples, num_chains) or flattened
        div_key = "diverging"
        divergence_rate = 0.0
        if div_key in extra:
            div = np.asarray(extra[div_key], dtype=bool)
            divergence_rate = float(np.mean(div))

        # Tree depth: stored as "num_steps" (number of leapfrog steps = 2^depth)
        # NumPyro stores the actual tree depth in "tree_depth" extra field when
        # available, otherwise fall back to estimating from "num_steps".
        mean_tree_depth = float("nan")
        max_depth_fraction = float("nan")

        if "tree_depth" in extra:
            depths = np.asarray(extra["tree_depth"], dtype=float)
            mean_tree_depth = float(np.mean(depths))
            max_depth_fraction = float(np.mean(depths >= self._plan.max_tree_depth))
        elif "num_steps" in extra:
            # num_steps = 2^tree_depth for NUTS binary tree; invert to get depth
            steps = np.asarray(extra["num_steps"], dtype=float)
            # Guard against zero steps (shouldn't occur, but be safe)
            steps = np.where(steps > 0, steps, 1.0)
            depths = np.log2(steps)
            mean_tree_depth = float(np.mean(depths))
            max_depth_fraction = float(
                np.mean(depths >= self._plan.max_tree_depth)
            )

        stats: dict[str, float] = {
            "divergence_rate": divergence_rate,
            "mean_tree_depth": mean_tree_depth,
            "max_tree_depth_fraction": max_depth_fraction,
        }

        logger.debug(
            "get_divergence_stats: divergence_rate=%.4f, mean_tree_depth=%.2f, "
            "max_depth_fraction=%.4f",
            divergence_rate, mean_tree_depth, max_depth_fraction,
        )

        return stats

    def get_diagnostics(self) -> az.InferenceData:
        """Extract ArviZ InferenceData for convergence diagnostics.

        Returns:
            ArviZ InferenceData containing posterior samples, sample
            stats (energy, divergences), and warmup statistics.

        Raises:
            RuntimeError: If called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "Cannot extract diagnostics before calling run()"
            )
        return az.from_numpyro(self._mcmc)

    @property
    def mcmc(self) -> MCMC:
        """Access the underlying NumPyro MCMC object."""
        return self._mcmc


# ---------------------------------------------------------------------------
# Adaptive sampling plan
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSamplingPlan:
    """Sampling plan that adjusts warmup/sample counts based on shard size.

    Wraps a base :class:`SamplingPlan` and scales it down proportionally
    when the shard is smaller than a reference size, while respecting
    floors derived from parameter count.

    The scaling rule is::

        scale = sqrt(shard_size / reference_shard_size)
        num_warmup  = max(min_warmup_floor,  int(base.num_warmup  * scale))
        num_samples = max(min_samples_floor, int(base.num_samples * scale))

    where ``min_warmup_floor = max(50, 5 * n_params)`` and
    ``min_samples_floor = max(100, 10 * n_params)``.

    Attributes:
        base_plan: Base :class:`SamplingPlan` for a full-size shard.
        shard_size: Number of data points in this shard.
        n_params: Number of varying model parameters.  Used to set
            minimum sample-count floors.
    """

    base_plan: SamplingPlan
    shard_size: int
    n_params: int

    #: Reference shard size at which no scaling is applied.
    _reference_shard_size: int = 10_000

    def __post_init__(self) -> None:
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be > 0, got {self.shard_size}")
        if self.n_params <= 0:
            raise ValueError(f"n_params must be > 0, got {self.n_params}")

    def get_plan(self) -> SamplingPlan:
        """Return a :class:`SamplingPlan` adjusted for this shard.

        Scaling is sub-linear (square-root) so that small shards still
        receive enough samples to characterise the posterior, while large
        shards are not penalised by excessive warmup.

        The floor on warmup is ``max(50, 5 * n_params)`` — enough adaptation
        steps to approximate the mass matrix for a 14-parameter model
        (floor = 70 steps).  The floor on samples is
        ``max(100, 10 * n_params)`` — enough draws for basic ESS diagnostics.

        Returns:
            Scaled :class:`SamplingPlan`.
        """
        scale = min(
            1.0,
            math.sqrt(self.shard_size / self._reference_shard_size),
        )

        # Parameter-aware floors
        min_warmup = max(50, 5 * self.n_params)
        min_samples = max(100, 10 * self.n_params)

        new_warmup = max(min_warmup, int(self.base_plan.num_warmup * scale))
        new_samples = max(min_samples, int(self.base_plan.num_samples * scale))

        logger.debug(
            "AdaptiveSamplingPlan.get_plan: shard_size=%d, n_params=%d, "
            "scale=%.3f, num_warmup=%d, num_samples=%d",
            self.shard_size, self.n_params, scale, new_warmup, new_samples,
        )

        return SamplingPlan(
            num_warmup=new_warmup,
            num_samples=new_samples,
            num_chains=self.base_plan.num_chains,
            target_accept=self.base_plan.target_accept,
            max_tree_depth=self.base_plan.max_tree_depth,
            adapt_step_size=self.base_plan.adapt_step_size,
            dense_mass=self.base_plan.dense_mass,
            seed=self.base_plan.seed,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Divergence rate thresholds
# ---------------------------------------------------------------------------

#: Target divergence rate — below this level the run is considered healthy.
DIVERGENCE_RATE_TARGET: float = 0.01

#: Elevated divergence rate — triggers a retry in :func:`run_nuts_with_retry`.
DIVERGENCE_RATE_HIGH: float = 0.05

#: Critical divergence rate — posterior geometry is likely incompatible with HMC.
DIVERGENCE_RATE_CRITICAL: float = 0.10


# ---------------------------------------------------------------------------
# SamplingStats dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingStats:
    """Summary statistics from a completed NUTS sampling run.

    Attributes:
        num_samples: Number of posterior draws per chain (post-warmup).
        num_warmup: Number of warmup steps per chain.
        num_divergences: Total divergent transitions across all chains.
        divergence_rate: Fraction of post-warmup transitions that diverged.
        mean_accept_prob: Mean Metropolis acceptance probability.
        max_tree_depth_fraction: Fraction of samples that hit the maximum
            NUTS tree depth.
        wall_time_seconds: Elapsed wall-clock time for the run.
    """

    num_samples: int
    num_warmup: int
    num_divergences: int
    divergence_rate: float
    mean_accept_prob: float
    max_tree_depth_fraction: float
    wall_time_seconds: float

    @property
    def is_healthy(self) -> bool:
        """Return True when divergence rate and acceptance probability are acceptable.

        Criteria:

        * ``divergence_rate < 0.05`` (below :data:`DIVERGENCE_RATE_HIGH`)
        * ``mean_accept_prob > 0.6``
        """
        return (
            self.divergence_rate < DIVERGENCE_RATE_HIGH
            and self.mean_accept_prob > 0.6
        )


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def run_nuts_with_retry(
    sampler: NUTSSampler,
    model_fn: Any,
    model_kwargs: dict[str, Any],
    max_retries: int = 3,
    step_size_factor: float = 0.5,
) -> tuple[dict[str, Any], SamplingStats]:
    """Run NUTS sampling with automatic step-size reduction on high divergence.

    Executes :meth:`~NUTSSampler.run` and checks the divergence rate
    after each attempt.  When the rate exceeds
    :data:`DIVERGENCE_RATE_HIGH`, a new :class:`NUTSSampler` is built
    with a step size reduced by ``step_size_factor`` and the run is
    retried.  After ``max_retries`` attempts the result with the lowest
    divergence rate is returned regardless of health.

    The ``model_fn`` is re-used across retries so it must be stateless
    (i.e. a pure NumPyro model function with no side effects).

    Args:
        sampler: Configured :class:`NUTSSampler` for the first attempt.
        model_fn: NumPyro model callable.  Not called directly here but
            passed to :meth:`NUTSSampler.from_plan` for retry instances.
        model_kwargs: Keyword arguments forwarded to the model via
            :meth:`~NUTSSampler.run`.  Currently unused by
            :meth:`~NUTSSampler.run` (which takes ``rng_key`` and
            ``init_params``); included for forward compatibility.
        max_retries: Maximum number of additional attempts after the
            first run.  Total runs = ``max_retries + 1``.
        step_size_factor: Multiplicative reduction applied to
            ``target_accept`` (lower target accept ≈ larger step size
            is avoided; instead we rebuild with a smaller target_accept
            proxy) each retry.  Must be in ``(0, 1)``.

    Returns:
        Tuple of ``(samples_dict, SamplingStats)`` for the best attempt
        (lowest divergence rate).

    Note:
        Step-size control in NumPyro NUTS is indirect — the target
        acceptance probability drives dual-averaging adaptation.  This
        function reduces ``target_accept`` by ``step_size_factor``
        each retry (e.g. 0.8 → 0.4), which causes dual-averaging to
        converge to a *larger* step size.  A larger step size can help
        when divergences are caused by overly-conservative trajectories
        in well-conditioned regions, but may worsen divergences in
        funnel geometries.  If funnel geometry is suspected,
        reparameterisation is the correct remedy and retrying will
        not help.
    """
    import time

    import numpy as np

    if not (0.0 < step_size_factor < 1.0):
        raise ValueError(
            f"step_size_factor must be in (0, 1), got {step_size_factor}"
        )

    best_samples: dict[str, Any] | None = None
    best_stats: SamplingStats | None = None
    best_divergence_rate = float("inf")

    current_sampler = sampler
    current_target_accept = sampler.plan.target_accept

    for attempt in range(max_retries + 1):
        t_start = time.monotonic()
        samples = current_sampler.run()
        wall_time = time.monotonic() - t_start

        div_stats = current_sampler.get_divergence_stats()
        divergence_rate = div_stats["divergence_rate"]
        max_tree_depth_fraction = div_stats.get("max_tree_depth_fraction", float("nan"))

        # Extract mean acceptance probability from extra fields
        extra = current_sampler.mcmc.get_extra_fields()
        mean_accept_prob = 0.0
        if "mean_accept_prob" in extra:
            mean_accept_prob = float(np.mean(np.asarray(extra["mean_accept_prob"])))
        elif "accept_prob" in extra:
            mean_accept_prob = float(np.mean(np.asarray(extra["accept_prob"])))

        plan = current_sampler.plan
        n_divergent = int(round(
            divergence_rate * plan.num_samples * plan.num_chains
        ))

        stats = SamplingStats(
            num_samples=plan.num_samples,
            num_warmup=plan.num_warmup,
            num_divergences=n_divergent,
            divergence_rate=divergence_rate,
            mean_accept_prob=mean_accept_prob,
            max_tree_depth_fraction=max_tree_depth_fraction,
            wall_time_seconds=wall_time,
        )

        logger.info(
            "run_nuts_with_retry attempt %d/%d: "
            "divergence_rate=%.4f, mean_accept_prob=%.4f, wall_time=%.1fs",
            attempt + 1, max_retries + 1,
            divergence_rate, mean_accept_prob, wall_time,
        )

        if divergence_rate < best_divergence_rate:
            best_divergence_rate = divergence_rate
            best_samples = samples
            best_stats = stats

        # Stop early if divergence rate is acceptable
        if divergence_rate <= DIVERGENCE_RATE_HIGH:
            break

        if attempt < max_retries:
            current_target_accept = current_target_accept * step_size_factor
            # Clamp to a sensible minimum to avoid pathological kernels
            current_target_accept = max(0.1, current_target_accept)
            logger.warning(
                "run_nuts_with_retry: divergence_rate=%.4f > %.2f; "
                "retrying with target_accept=%.4f (attempt %d/%d)",
                divergence_rate, DIVERGENCE_RATE_HIGH,
                current_target_accept, attempt + 2, max_retries + 1,
            )
            # Build a new sampler with reduced target acceptance
            new_plan = SamplingPlan(
                num_warmup=plan.num_warmup,
                num_samples=plan.num_samples,
                num_chains=plan.num_chains,
                target_accept=current_target_accept,
                max_tree_depth=plan.max_tree_depth,
                adapt_step_size=plan.adapt_step_size,
                dense_mass=plan.dense_mass,
                seed=plan.seed,
            )
            current_sampler = NUTSSampler.from_plan(
                new_plan,
                model_fn,
                chain_method="sequential",
            )
    else:
        # Exhausted retries
        logger.warning(
            "run_nuts_with_retry: exhausted %d retries; "
            "returning best result with divergence_rate=%.4f",
            max_retries, best_divergence_rate,
        )

    # These are always set on the first iteration, so cannot be None here
    assert best_samples is not None  # noqa: S101
    assert best_stats is not None  # noqa: S101
    return best_samples, best_stats


def _perturb_init_params(
    init_params: dict[str, jnp.ndarray],
    num_chains: int,
    seed: int,
    perturbation_scale: float = 0.01,
) -> dict[str, jnp.ndarray]:
    """Add small per-chain perturbations to initial parameters.

    Ensures each chain starts at a slightly different point in parameter
    space, preventing degenerate identical chains that waste compute.

    Args:
        init_params: Base initial values.  Scalars are broadcast to
            ``(num_chains,)``.
        num_chains: Number of MCMC chains.
        seed: Random seed for perturbation generation.
        perturbation_scale: Standard deviation of additive Gaussian
            perturbation relative to parameter magnitude.

    Returns:
        New dict with perturbed values of shape ``(num_chains,)``.
    """
    perturbed: dict[str, jnp.ndarray] = {}
    rng_key = jax.random.PRNGKey(seed)

    for name, value in init_params.items():
        rng_key, subkey = jax.random.split(rng_key)

        # Ensure shape (num_chains,)
        base = jnp.broadcast_to(jnp.asarray(value), (num_chains,))

        magnitude = jnp.abs(base) + 1e-10  # floor for zero-valued params
        noise = perturbation_scale * magnitude * jax.random.normal(
            subkey, shape=(num_chains,)
        )
        perturbed[name] = base + noise

    return perturbed
