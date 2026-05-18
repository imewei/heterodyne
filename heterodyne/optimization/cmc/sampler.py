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

from heterodyne.optimization.cmc.config import effective_warmup_floor
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from heterodyne.optimization.cmc.config import CMCConfig

logger = get_logger(__name__)

#: Below this shard size, ``SamplingPlan.for_shard`` downgrades a ``"parallel"``
#: chain method to ``"sequential"`` so JIT and warmup costs are shared across
#: chains.  Mirrors the homodyne small-shard policy.
_SMALL_SHARD_CHAIN_THRESHOLD: int = 100


# ---------------------------------------------------------------------------
# Adapter state diagnostics — homodyne CMC parity
# ---------------------------------------------------------------------------
#
# These helpers introspect the NumPyro ``last_state`` / ``adapt_state``
# objects so callers can log the adapted step size, inverse mass matrix,
# and accept-prob trajectory at the end of a run.  They are robust to
# the multiple shapes NumPyro uses (scalar vs per-chain dense matrix vs
# dict-of-arrays) and never raise.


def _summarize_inverse_mass_matrix(inv_mass: Any) -> str:
    """Return a compact textual summary of the adapted inverse mass matrix.

    Handles scalar, 1-D diagonal, 2-D dense, ``(n_chains, dim, dim)``
    per-chain dense, dict-of-arrays, and list/tuple-of-per-chain forms.
    Reports diagonal min/max, condition number, and dimensionality so
    callers can flag mass-matrix pathologies (near-singular, exploding
    condition numbers) at glance.
    """
    import numpy as _np

    def _one(mat: Any) -> str:
        if isinstance(mat, dict):
            keys = list(mat.keys())
            if not keys:
                return "dict(empty)"
            first = mat[keys[0]]
            return f"dict(keys={len(keys)}) first[{keys[0]}]: {_one(first)}"
        try:
            arr = _np.asarray(mat)
        except Exception:  # noqa: BLE001
            return f"type={type(mat).__name__}"
        if arr.ndim == 0:
            try:
                return f"scalar={float(arr):.3g}"
            except Exception:  # noqa: BLE001
                return f"scalar(type={type(arr.item()).__name__})"
        if arr.ndim == 1:
            diag = arr[_np.isfinite(arr)]
            if diag.size == 0:
                return f"diag(dim={arr.size}) all-nonfinite"
            dmin = float(_np.min(diag))
            dmax = float(_np.max(diag))
            cond = float(dmax / dmin) if dmin > 0 else float("inf")
            return f"diag(dim={arr.size}) min={dmin:.3g} max={dmax:.3g} cond~{cond:.3g}"
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            diag = _np.diag(arr)
            diag = diag[_np.isfinite(diag)]
            if diag.size == 0:
                return f"dense(dim={arr.shape[0]}) diag all-nonfinite"
            dmin = float(_np.min(diag))
            dmax = float(_np.max(diag))
            try:
                cond = float(_np.linalg.cond(arr))
            except Exception:  # noqa: BLE001
                cond = float("nan")
            return (
                f"dense(dim={arr.shape[0]}) diag[min={dmin:.3g}, max={dmax:.3g}] "
                f"cond={cond:.3g}"
            )
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
            n_chains = arr.shape[0]
            dim = arr.shape[1]
            parts = [_one(arr[i]) for i in range(min(n_chains, 2))]
            more = "" if n_chains <= 2 else f" (+{n_chains - 2} more)"
            return f"per-chain dense(dim={dim})[{', '.join(parts)}]{more}"
        return f"array(shape={arr.shape}, ndim={arr.ndim})"

    if isinstance(inv_mass, list | tuple):
        parts = [_one(m) for m in inv_mass[:2]]
        more = "" if len(inv_mass) <= 2 else f" (+{len(inv_mass) - 2} more)"
        return f"per-chain[{', '.join(parts)}]{more}"
    return _one(inv_mass)


def _extract_adapt_states(last_state: Any) -> list[Any]:
    """Return a list of NumPyro per-chain ``adapt_state`` objects.

    Empty when ``last_state`` is ``None`` or the adapt_state attribute is
    not present (e.g. when NUTS adaptation never ran).
    """
    if last_state is None:
        return []
    if hasattr(last_state, "adapt_state"):
        return [last_state.adapt_state]
    if isinstance(last_state, list | tuple):
        return [item.adapt_state for item in last_state if hasattr(item, "adapt_state")]
    return []


def _extract_step_sizes(adapt_states: list[Any]) -> list[float]:
    """Pull the final adapted ``step_size`` out of each adapt_state."""
    step_sizes: list[float] = []
    for adapt_state in adapt_states:
        if adapt_state is None:
            continue
        if hasattr(adapt_state, "step_size"):
            try:
                step_sizes.append(float(adapt_state.step_size))
                continue
            except Exception:  # noqa: S110 — robust fallback for adapt_state variants
                pass
        if isinstance(adapt_state, dict) and "step_size" in adapt_state:
            try:
                step_sizes.append(float(adapt_state["step_size"]))
            except Exception:  # noqa: S110 — robust fallback
                pass
    return step_sizes


def _log_array_stats(run_logger: Any, *, name: str, arr: Any) -> None:
    """Emit a single-line stat summary for an MCMC extra-field array."""
    import numpy as _np

    try:
        a = _np.asarray(arr)
    except Exception:  # noqa: BLE001
        return
    if a.size == 0:
        return
    finite = _np.isfinite(a)
    if not _np.any(finite):
        run_logger.info(f"{name} stats: all non-finite, shape={a.shape}")
        return
    run_logger.info(
        f"{name} stats: "
        f"min={float(_np.min(a[finite])):.3g}, "
        f"median={float(_np.median(a[finite])):.3g}, "
        f"max={float(_np.max(a[finite])):.3g}, "
        f"mean={float(_np.mean(a[finite])):.3g}, "
        f"std={float(_np.std(a[finite])):.3g}, "
        f"finite={float(_np.mean(finite)):.1%}, shape={a.shape}"
    )


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
    dense_mass: bool = True
    chain_method: str = "sequential"
    seed: int | None = None
    #: Rule 12 escape hatch propagated from :class:`CMCConfig.fast_warmup`.
    #: When True, ``for_shard`` and ``AdaptiveSamplingPlan`` skip the dense-mass
    #: warmup floor. CI / pytest fast-mode only — not for production posteriors.
    fast_warmup: bool = False

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
            raise ValueError(f"max_tree_depth must be >= 1, got {self.max_tree_depth}")
        if self.chain_method not in ("sequential", "parallel", "vectorized"):
            raise ValueError(
                f"chain_method must be 'sequential', 'parallel', or 'vectorized', "
                f"got {self.chain_method!r}"
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
                n_data,
                sqrt_scale,
                num_warmup,
                num_samples,
            )

        # Rule 12: dense-mass NUTS requires the configured warmup floor.
        num_warmup = effective_warmup_floor(
            num_warmup,
            dense_mass=config.dense_mass,
            fast_warmup=getattr(config, "fast_warmup", False),
        )

        return cls(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=config.num_chains,
            target_accept=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            adapt_step_size=True,
            dense_mass=config.dense_mass,
            chain_method=config.chain_method,
            seed=config.seed,
            fast_warmup=getattr(config, "fast_warmup", False),
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
        # Rule 12: per-shard scaling must not slip below the dense-mass floor.
        # Honour ``self.fast_warmup`` so CI fast-mode propagated from
        # CMCConfig.fast_warmup keeps its opt-out.
        new_warmup = effective_warmup_floor(
            new_warmup, dense_mass=self.dense_mass, fast_warmup=self.fast_warmup
        )

        # Homodyne CMC parity: shards with very few points cannot amortise
        # the parallel-chain dispatch overhead.  Fall back to sequential
        # chains so each shard's chains share JIT state and warmup costs.
        effective_chain_method = self.chain_method
        if (
            shard_size < _SMALL_SHARD_CHAIN_THRESHOLD
            and self.chain_method == "parallel"
        ):
            logger.info(
                "SamplingPlan.for_shard: shard_size=%d < %d; switching chain_method "
                "from 'parallel' to 'sequential' for amortised JIT cost.",
                shard_size,
                _SMALL_SHARD_CHAIN_THRESHOLD,
            )
            effective_chain_method = "sequential"

        logger.debug(
            "SamplingPlan.for_shard: shard_size=%d, full_size=%d, scale=%.3f, "
            "num_warmup=%d->%d, num_samples=%d->%d, chain_method=%s",
            shard_size,
            full_size,
            scale,
            self.num_warmup,
            new_warmup,
            self.num_samples,
            new_samples,
            effective_chain_method,
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
            chain_method=effective_chain_method,
            seed=self.seed,
            fast_warmup=self.fast_warmup,
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
        chain_method: str | None = None,
    ) -> NUTSSampler:
        """Create a NUTSSampler from a SamplingPlan and NumPyro model.

        Args:
            plan: Sampling hyperparameters.
            model: NumPyro model function (callable with no required args).
            init_strategy: NumPyro initialization strategy name.
                One of ``"init_to_median"``, ``"init_to_sample"``,
                ``"init_to_value"``.
            chain_method: NumPyro chain execution method.
                ``"sequential"`` for single device, ``"parallel"`` for multi-device.

        Returns:
            Configured NUTSSampler ready for :meth:`run`.
        """
        effective_chain_method = (
            chain_method if chain_method is not None else plan.chain_method
        )

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
            chain_method=effective_chain_method,
            progress_bar=True,
        )

        logger.info(
            "NUTSSampler created: %d chains, %d warmup, %d samples, "
            "target_accept=%s, chain_method=%s",
            plan.num_chains,
            plan.num_warmup,
            plan.num_samples,
            plan.target_accept,
            effective_chain_method,
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
        # Homodyne CMC parity: capture per-step accept_prob, num_steps, and
        # potential_energy alongside divergence/energy so downstream
        # diagnostics (BFMI, tree-depth analysis, accept-prob stats) work.
        self._mcmc.run(
            rng_key,
            init_params=perturbed_params,
            extra_fields=(
                "energy",
                "diverging",
                "accept_prob",
                "num_steps",
                "potential_energy",
            ),
        )
        # Block until JAX lazy evaluation completes so wall_time_seconds
        # reflects true compute time, not deferred device_get() overhead.
        jax.block_until_ready(self._mcmc.last_state)
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
            # Use numpyro.infer.util.log_density to evaluate the model's
            # unnormalized log-joint at init_params without running full NUTS.
            # This works before any sampling and is the correct API — relying
            # on kernel._potential_fn was wrong because it is None pre-run.
            from numpyro.infer.util import log_density as _numpyro_log_density

            kernel = self._mcmc.sampler  # type: ignore[attr-defined]
            model_fn = kernel.model  # type: ignore[attr-defined]
            if model_fn is None:
                logger.debug(
                    "_validate_init_log_density: model not available on kernel, skipping"
                )
                return

            log_density_val, _ = _numpyro_log_density(model_fn, (), {}, init_params)
            log_density = float(log_density_val)

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
            logger.debug("_validate_init_log_density: probe failed (%s), skipping", exc)

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
            raise RuntimeError("Cannot extract divergence stats before calling run()")

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
            max_depth_fraction = float(np.mean(depths >= self._plan.max_tree_depth))

        stats: dict[str, float] = {
            "divergence_rate": divergence_rate,
            "mean_tree_depth": mean_tree_depth,
            "max_tree_depth_fraction": max_depth_fraction,
        }

        logger.debug(
            "get_divergence_stats: divergence_rate=%.4f, mean_tree_depth=%.2f, "
            "max_depth_fraction=%.4f",
            divergence_rate,
            mean_tree_depth,
            max_depth_fraction,
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
            raise RuntimeError("Cannot extract diagnostics before calling run()")
        return az.from_numpyro(self._mcmc)

    def log_adapter_diagnostics(self, run_logger: Any | None = None) -> None:
        """Log NUTS adapter state at INFO level — homodyne CMC parity helper.

        Reports the adapted ``step_size`` per chain, a compact summary of
        the adapted inverse mass matrix, and per-step ``accept_prob`` /
        ``num_steps`` / ``potential_energy`` statistics when the
        corresponding extra fields are present.

        Args:
            run_logger: Logger to emit through.  ``None`` uses this
                module's logger.

        Raises:
            RuntimeError: If called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "Cannot extract adapter diagnostics before calling run()"
            )

        log_inst = run_logger if run_logger is not None else logger

        last_state = getattr(self._mcmc, "last_state", None)
        adapt_states = _extract_adapt_states(last_state)
        step_sizes = _extract_step_sizes(adapt_states)
        if step_sizes:
            log_inst.info(
                "Adapted step sizes (per chain): %s",
                ", ".join(f"{s:.3g}" for s in step_sizes),
            )

        if adapt_states:
            first = adapt_states[0]
            inv_mass = getattr(first, "inverse_mass_matrix", None)
            if inv_mass is None and isinstance(first, dict):
                inv_mass = first.get("inverse_mass_matrix")
            if inv_mass is not None:
                log_inst.info(
                    "Inverse mass matrix: %s",
                    _summarize_inverse_mass_matrix(inv_mass),
                )

        extra = self._mcmc.get_extra_fields()
        for field in ("accept_prob", "num_steps", "potential_energy"):
            if field in extra:
                _log_array_stats(log_inst, name=field, arr=extra[field])

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
        # Rule 12: floor=70 here is too low for dense-mass NUTS on 14 params.
        new_warmup = effective_warmup_floor(
            new_warmup,
            dense_mass=self.base_plan.dense_mass,
            fast_warmup=self.base_plan.fast_warmup,
        )

        logger.debug(
            "AdaptiveSamplingPlan.get_plan: shard_size=%d, n_params=%d, "
            "scale=%.3f, num_warmup=%d, num_samples=%d",
            self.shard_size,
            self.n_params,
            scale,
            new_warmup,
            new_samples,
        )

        return SamplingPlan(
            num_warmup=new_warmup,
            num_samples=new_samples,
            num_chains=self.base_plan.num_chains,
            target_accept=self.base_plan.target_accept,
            max_tree_depth=self.base_plan.max_tree_depth,
            adapt_step_size=self.base_plan.adapt_step_size,
            dense_mass=self.base_plan.dense_mass,
            chain_method=self.base_plan.chain_method,
            seed=self.base_plan.seed,
            fast_warmup=self.base_plan.fast_warmup,
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
            self.divergence_rate < DIVERGENCE_RATE_HIGH and self.mean_accept_prob > 0.6
        )


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def run_nuts_with_retry(
    sampler: NUTSSampler,
    model_fn: Any,
    model_kwargs: dict[str, Any],
    max_retries: int = 3,
    target_accept_increment: float = 0.05,
    *,
    step_size_factor: float | None = None,
) -> tuple[dict[str, Any], SamplingStats]:
    """Run NUTS sampling with automatic step-size reduction on high divergence.

    Executes :meth:`~NUTSSampler.run` and checks the divergence rate
    after each attempt.  When the rate exceeds
    :data:`DIVERGENCE_RATE_HIGH`, a new :class:`NUTSSampler` is built
    with ``target_accept`` RAISED by ``target_accept_increment`` (which
    drives dual averaging toward a SMALLER step size — the
    mathematically correct response to high divergence rate) and the
    run is retried.  After ``max_retries`` attempts the result with the
    lowest divergence rate is returned regardless of health.  Mirrors
    homodyne ``run_nuts_with_retry`` (sampler.py:1311-1314).

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
        target_accept_increment: Additive increase applied to
            ``target_accept`` each retry (e.g. 0.80 → 0.85 → 0.90 →
            0.95).  Raising the target acceptance LOWERS the
            dual-averaging step size, which is the mathematically
            correct response to high divergence rates.  Must be in
            ``(0, 0.5)``.  The target is clamped at ``0.99`` to avoid
            pathological tiny step sizes.
        step_size_factor: DEPRECATED keyword-only alias.  Earlier
            versions multiplied ``target_accept`` by this factor
            (with factor ``< 1``) on retry, which drove dual averaging
            toward a LARGER step size — the OPPOSITE of what high
            divergence rate calls for.  When supplied,
            ``target_accept_increment`` is derived as
            ``(1 - step_size_factor) * 0.1`` so legacy callers see a
            corrected mathematical direction.

    Returns:
        Tuple of ``(samples_dict, SamplingStats)`` for the best attempt
        (lowest divergence rate).
    """
    import time

    import numpy as np

    if step_size_factor is not None:
        # Legacy alias: map a multiplicative factor < 1 to a positive
        # increment so the direction is correct.  E.g. legacy
        # step_size_factor=0.5 ⇒ increment 0.05.
        target_accept_increment = max(0.01, (1.0 - step_size_factor) * 0.1)
        logger.warning(
            "run_nuts_with_retry: 'step_size_factor' is deprecated and its "
            "original direction was inverted; mapped to "
            "target_accept_increment=%.4f",
            target_accept_increment,
        )

    if not (0.0 < target_accept_increment < 0.5):
        raise ValueError(
            f"target_accept_increment must be in (0, 0.5), got "
            f"{target_accept_increment}"
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
        n_divergent = int(round(divergence_rate * plan.num_samples * plan.num_chains))

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
            attempt + 1,
            max_retries + 1,
            divergence_rate,
            mean_accept_prob,
            wall_time,
        )

        if divergence_rate < best_divergence_rate:
            best_divergence_rate = divergence_rate
            best_samples = samples
            best_stats = stats

        # Stop early if divergence rate is acceptable
        if divergence_rate <= DIVERGENCE_RATE_HIGH:
            break

        if attempt < max_retries:
            # Raising target_accept drives dual averaging toward a SMALLER
            # step size, which is the mathematically correct response to
            # high divergence rate.  The previous implementation reduced
            # target_accept (silently making divergence WORSE).
            current_target_accept = min(
                0.99, current_target_accept + target_accept_increment
            )
            logger.warning(
                "run_nuts_with_retry: divergence_rate=%.4f > %.2f; "
                "retrying with target_accept=%.4f (attempt %d/%d)",
                divergence_rate,
                DIVERGENCE_RATE_HIGH,
                current_target_accept,
                attempt + 2,
                max_retries + 1,
            )
            # Build a new sampler with raised target acceptance
            new_plan = SamplingPlan(
                num_warmup=plan.num_warmup,
                num_samples=plan.num_samples,
                num_chains=plan.num_chains,
                target_accept=current_target_accept,
                max_tree_depth=plan.max_tree_depth,
                adapt_step_size=plan.adapt_step_size,
                dense_mass=plan.dense_mass,
                chain_method=plan.chain_method,
                seed=plan.seed,
            )
            current_sampler = NUTSSampler.from_plan(
                new_plan,
                model_fn,
            )
    else:
        # Exhausted retries
        logger.warning(
            "run_nuts_with_retry: exhausted %d retries; "
            "returning best result with divergence_rate=%.4f",
            max_retries,
            best_divergence_rate,
        )

    # These are always set on the first iteration, so cannot be None here
    assert best_samples is not None  # noqa: S101
    assert best_stats is not None  # noqa: S101
    return best_samples, best_stats


def _compute_mcmc_safe_d0_component(
    d0: float | None,
    alpha: float | None,
    d_offset: float | None,
    *,
    q: float,
    dt: float,
    time_grid: Any | None,
    target_g1: float = 0.5,
    g1_threshold: float = 0.1,
) -> tuple[float, float] | None:
    """Detect vanishing-gradient pathology for a single transport component.

    The heterodyne signal is a mixture of two single-exponential
    contributions :math:`g_1 = f \\exp(-q^2 J_s) + (1-f) \\exp(-q^2 J_r)`.
    Each component can independently kill the gradient if its diffusion
    integral grows large; this helper inspects one component's
    ``(D0, alpha, D_offset)`` triple and returns a rescaled
    ``(D0', D_offset')`` pair when ``g1`` falls below ``g1_threshold``.

    Mirrors homodyne ``_compute_mcmc_safe_d0`` (sampler.py:143-278) but
    is adapted for heterodyne's two-channel architecture by being called
    twice (once per ``ref``/``sample`` channel).

    Args:
        d0: Diffusion prefactor (e.g. ``D0_ref`` or ``D0_sample``).
        alpha: Power-law exponent on the diffusion term.
        d_offset: Additive offset to the diffusion rate.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        time_grid: Time grid for integration.  ``None`` → return ``None``.
        target_g1: ``g1`` value the scaling targets when adjustment fires.
        g1_threshold: ``g1`` floor below which an adjustment fires.

    Returns:
        ``(new_d0, new_d_offset)`` when an MCMC-safe scaling is required;
        ``None`` when the original values are safe or cannot be evaluated.
    """
    import numpy as _np

    if d0 is None or alpha is None or d_offset is None:
        return None
    if time_grid is None or len(time_grid) < 2:
        return None
    if not (_np.isfinite(d0) and _np.isfinite(alpha) and _np.isfinite(d_offset)):
        return None

    try:
        epsilon = 1e-10
        time_safe = _np.asarray(time_grid) + epsilon
        # Homodyne parity: use a gradient-safe floor for the integrand.
        # jnp.maximum would zero the gradient under the floor; np.maximum
        # is safe here because we evaluate at concrete values outside the
        # JAX tracer.
        d_grid = d0 * (time_safe**alpha) + d_offset
        d_grid = _np.maximum(d_grid, 1e-10)

        if len(d_grid) > 1:
            trap_avg = 0.5 * (d_grid[:-1] + d_grid[1:])
            cumsum = _np.concatenate([[0.0], _np.cumsum(trap_avg)])
        else:
            cumsum = _np.cumsum(d_grid)

        n = len(cumsum)
        integral_estimate = abs(cumsum[3 * n // 4] - cumsum[n // 4])

        prefactor = q**2 * dt
        log_g1 = -prefactor * integral_estimate
        log_g1_clipped = max(log_g1, -700.0)
        g1_estimate = _np.exp(log_g1_clipped)

        if g1_estimate >= g1_threshold:
            return None

        target_log_g1 = _np.log(target_g1)
        target_integral = -target_log_g1 / prefactor
        scale_factor = (
            target_integral / integral_estimate if integral_estimate > 0 else 0.01
        )
        new_d0 = max(float(d0 * scale_factor), 1.0)
        new_d_offset = max(float(d_offset * scale_factor), -1e6)
        return new_d0, new_d_offset
    except Exception:  # noqa: BLE001 — diagnostics helper must not crash MCMC
        return None


def compute_mcmc_safe_initial_values(
    initial_values: dict[str, float] | None,
    *,
    q: float,
    dt: float,
    time_grid: Any | None,
    target_g1: float = 0.5,
    g1_threshold: float = 0.1,
) -> dict[str, float] | None:
    """Detect/repair vanishing-g1 initial parameters for heterodyne NUTS.

    Inspects both the reference and sample transport components and, when
    either drives ``g1`` below ``g1_threshold`` at a typical lag, rescales
    its ``(D0, D_offset)`` pair so that ``g1 ≈ target_g1``.  Returns
    ``None`` when no adjustment is needed so callers can short-circuit.

    Args:
        initial_values: NLSQ warm-start dictionary.  Expected keys include
            ``D0_ref``, ``alpha_ref``, ``D_offset_ref``, ``D0_sample``,
            ``alpha_sample``, ``D_offset_sample``.
        q: Wavevector magnitude (Å⁻¹).
        dt: Lag-time step (s).
        time_grid: Time grid for integration.
        target_g1: Target ``g1`` value when rescaling.
        g1_threshold: Threshold below which rescaling fires.

    Returns:
        A new dictionary with adjusted ``D0_*`` / ``D_offset_*`` values
        when adjustment is required, else ``None``.
    """
    if initial_values is None:
        return None

    adjusted: dict[str, float] = dict(initial_values)
    any_change = False
    for prefix in ("ref", "sample"):
        d0_key = f"D0_{prefix}"
        a_key = f"alpha_{prefix}"
        off_key = f"D_offset_{prefix}"
        result = _compute_mcmc_safe_d0_component(
            adjusted.get(d0_key),
            adjusted.get(a_key),
            adjusted.get(off_key),
            q=q,
            dt=dt,
            time_grid=time_grid,
            target_g1=target_g1,
            g1_threshold=g1_threshold,
        )
        if result is not None:
            new_d0, new_off = result
            logger.warning(
                "compute_mcmc_safe_initial_values: %s pathology — "
                "scaling %s %.4g -> %.4g, %s %.4g -> %.4g for MCMC stability",
                prefix,
                d0_key,
                adjusted[d0_key],
                new_d0,
                off_key,
                adjusted[off_key],
                new_off,
            )
            adjusted[d0_key] = new_d0
            adjusted[off_key] = new_off
            any_change = True

    return adjusted if any_change else None


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
    param_names = list(init_params.keys())
    subkeys = jax.random.split(rng_key, num=len(param_names))

    for i, name in enumerate(param_names):
        value = init_params[name]
        # Ensure shape (num_chains,)
        base = jnp.broadcast_to(jnp.asarray(value), (num_chains,))

        magnitude = jnp.abs(base) + 1e-10  # floor for zero-valued params
        noise = (
            perturbation_scale
            * magnitude
            * jax.random.normal(subkeys[i], shape=(num_chains,))
        )
        perturbed[name] = base + noise

    return perturbed
