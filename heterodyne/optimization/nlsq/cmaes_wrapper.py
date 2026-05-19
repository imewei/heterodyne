"""CMA-ES optimization wrapper for heterodyne parameter fitting.

Provides a derivative-free global optimizer as an alternative to
gradient-based NLSQ methods.  Useful when the cost landscape is
multi-modal or the Jacobian is unreliable.

The ``cma`` package is an optional dependency.  If it is not installed,
importing this module succeeds but :meth:`CMAESWrapper.fit` raises
:class:`ImportError` at call time.

Parity note
-----------
This module matches the homodyne ``cmaes_wrapper`` public API:

- :class:`CMAESWrapperConfig` (v2.16.0 normalization + refinement settings)
- :class:`CMAESResult` (parameters, covariance, chi_squared, …)
- :class:`CMAESWrapper` (compute_scale_ratio, is_available, should_use_cmaes, fit)

Heterodyne-specific additions (not in homodyne):

- :class:`CMAESConfig` — low-level ``cma`` library settings
- :func:`fit_with_cmaes` — objective-based convenience (returns NLSQResult)
- :func:`build_anti_degeneracy_objective`
- :func:`normalize_to_unit_cube` / :func:`denormalize_from_unit_cube`
- :func:`compute_adaptive_cmaes_params`
- :func:`adjust_covariance_for_bounds`

D3 divergence: heterodyne uses the ``cma`` library backend instead of
NLSQ's evosax backend (CPU-only; NLSQ evosax not required).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heterodyne.optimization.nlsq.result_builder import build_result_from_arrays
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.optimization.nlsq.config import NLSQConfig
    from heterodyne.optimization.nlsq.results import NLSQResult

logger = get_logger(__name__)

try:
    import cma as _cma  # noqa: F401  # type: ignore[import-untyped]

    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False

#: Public availability flag consumed by ``core.py``.
CMAES_AVAILABLE: bool = _HAS_CMA

# Skip L-M refinement when CMA-ES chi2 exceeds this multiple of the
# warm-start chi2 — the comparison in core.py will discard it anyway.
_REFINEMENT_SKIP_CHI2_RATIO = 10.0


# =============================================================================
# Bounds summary helper (homodyne parity)
# =============================================================================


def _format_bounds_summary(bounds: tuple[np.ndarray, np.ndarray]) -> str:
    """Format bounds summary for logging.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Lower and upper bounds as (lower, upper) arrays.

    Returns
    -------
    str
        Human-readable bounds summary.
    """
    lower, upper = bounds
    ranges = upper - lower
    n_params = len(lower)

    valid_ranges = ranges[ranges > 0]
    if len(valid_ranges) == 0:
        return f"{n_params} params (no valid ranges)"

    min_range = np.min(valid_ranges)
    max_range = np.max(valid_ranges)

    return f"{n_params} params, range=[{min_range:.2e}, {max_range:.2e}]"


# =============================================================================
# Parameter Normalization Utilities (v2.16.0 — homodyne parity)
# =============================================================================
# These functions implement bounds-based normalization to improve CMA-ES
# convergence for multi-scale problems (e.g., D₀ ~ 1e4 vs alpha near 1).


def _compute_normalization_factors(
    bounds: tuple[np.ndarray, np.ndarray],
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalization factors for bounds-based normalization.

    Normalization: x_norm = (x - lower) / (upper - lower)

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Lower and upper bounds as (lower, upper) arrays.
    epsilon : float
        Small value to prevent division by zero for fixed parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (scale, offset, range) where:
        - scale = 1 / (upper - lower + epsilon) for safe division
        - offset = lower (subtracted before scaling)
        - range = upper - lower (for covariance adjustment)
    """
    lower, upper = bounds
    param_range = upper - lower

    safe_range = np.where(param_range > epsilon, param_range, 1.0)
    scale = 1.0 / safe_range

    return scale, lower, param_range


def _normalize_params(
    params: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Normalize parameters from physical space to [0, 1] space.

    Parameters
    ----------
    params : np.ndarray
        Parameters in physical space.
    scale : np.ndarray
        Scale factors (1 / range).
    offset : np.ndarray
        Offset values (lower bounds).

    Returns
    -------
    np.ndarray
        Parameters normalized to [0, 1] space.
    """
    return (params - offset) * scale


def _denormalize_params(
    params_norm: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Denormalize parameters from [0, 1] space back to physical space.

    Parameters
    ----------
    params_norm : np.ndarray
        Parameters in normalized [0, 1] space.
    scale : np.ndarray
        Scale factors (1 / range).
    offset : np.ndarray
        Offset values (lower bounds).

    Returns
    -------
    np.ndarray
        Parameters in physical space.
    """
    return params_norm / scale + offset


def _normalize_bounds(
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize bounds to [0, 1] space.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Physical bounds as (lower, upper) arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized bounds (zeros, ones).
    """
    n_params = len(bounds[0])
    return (np.zeros(n_params), np.ones(n_params))


def _adjust_covariance_for_normalization(
    covariance: np.ndarray | None,
    param_range: np.ndarray,
) -> np.ndarray | None:
    """Transform covariance from normalized to physical space.

    Uses Jacobian scaling: Cov_phys = J @ Cov_norm @ J.T
    where J is diagonal with elements = param_range.

    Parameters
    ----------
    covariance : np.ndarray | None
        Covariance matrix in normalized space.
    param_range : np.ndarray
        Parameter ranges (upper - lower bounds).

    Returns
    -------
    np.ndarray | None
        Covariance matrix in physical space, or None if input is None.
    """
    if covariance is None:
        return None

    jacobian_outer = np.outer(param_range, param_range)
    return covariance * jacobian_outer


# =============================================================================
# CMAESConfig (heterodyne-specific: cma library settings)
# =============================================================================


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimization via the ``cma`` library.

    Attributes:
        sigma0: Initial step-size (standard deviation) for the search
            distribution.  A good default is ~1/4 of the expected parameter
            range.
        popsize: Population size.  ``None`` lets cma choose automatically.
        maxiter: Maximum number of CMA-ES generations.
        tolx: Termination tolerance on parameter changes.
        tolfun: Termination tolerance on cost function changes.
        seed: Random seed for reproducibility.
        restart_strategy: Restart strategy for CMA-ES.  ``"bipop"`` alternates
            large/small population restarts for robust global search.  ``"none"``
            runs a single CMA-ES run with no restarts.  Callers should override
            to ``"none"`` when a warmstart sigma is already small (warmstart mode),
            because BIPOP large-population restarts are incoherent with a tight
            initial distribution.
        max_restarts: Maximum number of BIPOP restarts.  Ignored when
            ``restart_strategy="none"``.
    """

    sigma0: float = 0.5
    popsize: int | None = None
    maxiter: int = 1000
    tolx: float = 1e-11
    tolfun: float = 1e-11
    seed: int = 42
    diagonal_filtering: str = "none"
    restart_strategy: str = "bipop"
    max_restarts: int = 9

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1")
        if self.popsize is not None and self.popsize < 2:
            raise ValueError("popsize must be >= 2 when specified")
        if self.diagonal_filtering not in ("remove", "none"):
            raise ValueError(
                f"diagonal_filtering must be 'remove' or 'none', "
                f"got {self.diagonal_filtering!r}"
            )
        if self.restart_strategy not in ("bipop", "none"):
            raise ValueError(
                f"restart_strategy must be 'bipop' or 'none', "
                f"got {self.restart_strategy!r}"
            )
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")


# =============================================================================
# CMAESWrapperConfig (homodyne parity)
# =============================================================================


@dataclass
class CMAESWrapperConfig:
    """Configuration for :class:`CMAESWrapper`.

    Mirrors homodyne's ``CMAESWrapperConfig`` for structural parity.

    Attributes
    ----------
    preset : str
        CMA-ES preset: "cmaes-fast" (50 gen), "cmaes" (100 gen),
        "cmaes-global" (200 gen).
    max_generations : int | None
        Maximum CMA-ES generations. None = use preset default.
    popsize : int | None
        Population size (None = auto from 4+3*ln(n)).
    sigma : float
        Initial step size as fraction of search range (0, 1].
    sigma_warmstart : float
        Reduced sigma for warm-start (local refinement).
    tol_fun : float
        Function value tolerance for convergence.
    tol_x : float
        Parameter tolerance for convergence.
    restart_strategy : str
        Restart strategy: "none" or "bipop".
    max_restarts : int
        Maximum number of BIPOP restarts.
    refine_with_nlsq : bool
        Whether to refine CMA-ES solution with NLSQ TRF.
    normalize : bool
        Enable bounds-based parameter normalization.
    normalization_epsilon : float
        Prevent division by zero in normalization.
    """

    # CMA-ES global search settings
    preset: str = "cmaes"
    max_generations: int | None = None
    popsize: int | None = None
    sigma: float = 0.5
    sigma_warmstart: float = 0.05
    tol_fun: float = 1e-8
    tol_x: float = 1e-8
    restart_strategy: str = "bipop"
    max_restarts: int = 9

    # NLSQ TRF refinement settings (post-CMA-ES)
    refine_with_nlsq: bool = True
    refinement_ftol: float = 1e-10
    refinement_xtol: float = 1e-10
    refinement_gtol: float = 1e-10
    refinement_max_nfev: int = 500
    refinement_loss: str = "linear"

    # Parameter normalization (v2.16.0)
    normalize: bool = True
    normalization_epsilon: float = 1e-12

    @classmethod
    def from_nlsq_config(cls, config: NLSQConfig) -> CMAESWrapperConfig:
        """Create :class:`CMAESWrapperConfig` from :class:`NLSQConfig`.

        Parameters
        ----------
        config : NLSQConfig
            NLSQ configuration object.

        Returns
        -------
        CMAESWrapperConfig
            CMA-ES wrapper configuration.
        """
        return cls(
            preset=getattr(config, "cmaes_preset", "cmaes"),
            max_generations=getattr(config, "cmaes_max_generations", None),
            popsize=getattr(config, "cmaes_popsize", None),
            sigma=getattr(config, "cmaes_sigma", 0.5),
            sigma_warmstart=getattr(config, "cmaes_sigma_warmstart", 0.05),
            tol_fun=getattr(config, "cmaes_tol_fun", 1e-8),
            tol_x=getattr(config, "cmaes_tol_x", 1e-8),
            restart_strategy=getattr(config, "cmaes_restart_strategy", "bipop"),
            max_restarts=getattr(config, "cmaes_max_restarts", 9),
            refine_with_nlsq=getattr(config, "cmaes_refine_with_nlsq", True),
            refinement_ftol=getattr(config, "cmaes_refinement_ftol", 1e-10),
            refinement_xtol=getattr(config, "cmaes_refinement_xtol", 1e-10),
            refinement_gtol=getattr(config, "cmaes_refinement_gtol", 1e-10),
            refinement_max_nfev=getattr(config, "cmaes_refinement_max_nfev", 500),
            refinement_loss=getattr(config, "cmaes_refinement_loss", "linear"),
            normalize=getattr(config, "cmaes_normalize", True),
            normalization_epsilon=getattr(config, "cmaes_normalization_epsilon", 1e-12),
        )


# =============================================================================
# CMAESResult (homodyne parity)
# =============================================================================


@dataclass
class CMAESResult:
    """Result from CMA-ES optimization.

    Matches homodyne's ``CMAESResult`` for structural parity.

    Attributes
    ----------
    parameters : np.ndarray
        Optimized parameter values.
    covariance : np.ndarray | None
        Parameter covariance matrix (if computed).
    chi_squared : float
        Final chi-squared value.
    success : bool
        Whether optimization converged successfully.
    diagnostics : dict
        CMA-ES diagnostics (generations, evaluations, etc.).
    method_used : str
        Method used: "cmaes" or "multi-start".
    nlsq_refined : bool
        Whether result was refined with NLSQ L-M.
    message : str
        Convergence message.
    """

    parameters: np.ndarray
    covariance: np.ndarray | None
    chi_squared: float
    success: bool
    diagnostics: dict = field(default_factory=dict)
    method_used: str = "cmaes"
    nlsq_refined: bool = False
    message: str = ""


# =============================================================================
# CMAESWrapper (homodyne parity API + heterodyne cma backend)
# =============================================================================


class CMAESWrapper:
    """Wrapper around ``cma.fmin2`` for bounded parameter optimization.

    Public API matches homodyne's ``CMAESWrapper``:

    - :meth:`is_available` — check if ``cma`` library is installed
    - :meth:`compute_scale_ratio` — max/min bound range ratio
    - :meth:`should_use_cmaes` — scale-ratio based selection
    - :meth:`fit` — run CMA-ES, return :class:`CMAESResult`

    Heterodyne-specific internal method:

    - :meth:`fit_with_objective` — objective-fn based fit, returns NLSQResult

    Parameters
    ----------
    config : CMAESWrapperConfig | None
        CMA-ES wrapper configuration.  Uses defaults if None.
    parameter_names : list[str] | None
        Ordered parameter names (used in returned NLSQResult from
        :meth:`fit_with_objective`).
    """

    def __init__(
        self,
        config: CMAESWrapperConfig | None = None,
        parameter_names: list[str] | None = None,
    ) -> None:
        self.config = config or CMAESWrapperConfig()
        self._parameter_names = parameter_names or []

    # ------------------------------------------------------------------
    # Homodyne parity properties / methods
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Check if CMA-ES is available (``cma`` library installed)."""
        return CMAES_AVAILABLE

    def compute_scale_ratio(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Compute scale ratio from parameter bounds.

        The scale ratio is the ratio of the largest to smallest parameter
        range. High scale ratios (> 1000) indicate multi-scale problems
        where CMA-ES excels.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Lower and upper bounds as (lower, upper) arrays.

        Returns
        -------
        float
            Scale ratio (max_range / min_range).
        """
        lower, upper = bounds
        ranges = upper - lower

        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) < 2:
            return 1.0

        min_range = float(np.min(valid_ranges))
        max_range = float(np.max(valid_ranges))

        return max_range / min_range if min_range > 0 else 1.0

    def should_use_cmaes(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        scale_threshold: float = 1000.0,
    ) -> bool:
        """Determine if CMA-ES should be used based on scale ratio.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper) arrays.
        scale_threshold : float
            Scale ratio threshold for CMA-ES selection. Default: 1000.

        Returns
        -------
        bool
            True if CMA-ES should be used.
        """
        if not self.is_available:
            logger.info(
                "[CMA-ES] Method unavailable: 'cma' package not installed. "
                "Install with: uv add cma"
            )
            return False

        scale_ratio = self.compute_scale_ratio(bounds)
        should_use = scale_ratio >= scale_threshold
        bounds_summary = _format_bounds_summary(bounds)

        if should_use:
            logger.info(
                "[CMA-ES] Auto-selected: scale_ratio=%.2e (threshold=%.2e), %s",
                scale_ratio,
                scale_threshold,
                bounds_summary,
            )
        else:
            logger.debug(
                "[CMA-ES] Not selected: scale_ratio=%.2e < threshold=%.2e",
                scale_ratio,
                scale_threshold,
            )

        return should_use

    def fit(
        self,
        model_func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None = None,
        warmstart_chi2: float | None = None,
    ) -> CMAESResult:
        """Run CMA-ES global optimization.

        Matches homodyne's ``CMAESWrapper.fit`` signature exactly.
        Uses the ``cma`` library backend (heterodyne CPU-only path).

        Parameters
        ----------
        model_func : Callable
            Model function: ``y = f(x, *params)``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data to fit.
        p0 : np.ndarray
            Initial parameter guess.
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper).
        sigma : np.ndarray | None
            Data uncertainties (optional).
        warmstart_chi2 : float | None
            Chi-squared from NLSQ warm-start. If provided and CMA-ES chi2
            exceeds ``_REFINEMENT_SKIP_CHI2_RATIO`` × this value, refinement
            is skipped.  Also triggers ``sigma_warmstart`` instead of ``sigma``
            for the CMA-ES search.

        Returns
        -------
        CMAESResult
            Optimization result with parameters, covariance, diagnostics.

        Raises
        ------
        ImportError
            If the ``cma`` package is not installed.
        """
        if not _HAS_CMA:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install it with: uv add cma"
            )
        import cma  # noqa: F811  # type: ignore[import-untyped]

        p0 = np.asarray(p0, dtype=np.float64)
        lower, upper = (
            np.asarray(bounds[0], dtype=np.float64),
            np.asarray(bounds[1], dtype=np.float64),
        )
        n_params = len(p0)
        scale_ratio = self.compute_scale_ratio(bounds)
        bounds_summary = _format_bounds_summary(bounds)

        logger.info(
            "[CMA-ES] Optimization starting: n_params=%d, n_data=%d",
            n_params,
            len(ydata),
        )
        logger.info(
            "[CMA-ES] Problem characteristics: scale_ratio=%.2e, %s",
            scale_ratio,
            bounds_summary,
        )

        # Select sigma based on warm-start state (v2.20.0)
        warmstart_active = warmstart_chi2 is not None and warmstart_chi2 < float("inf")
        if warmstart_active:
            effective_sigma = self.config.sigma_warmstart
            logger.info(
                "[CMA-ES] Warm-start active: using sigma_warmstart=%.3f "
                "(default sigma=%.3f)",
                effective_sigma,
                self.config.sigma,
            )
        else:
            effective_sigma = self.config.sigma

        # Build cma opts
        preset_generations = {
            "cmaes-fast": 50,
            "cmaes": 100,
            "cmaes-global": 200,
        }
        max_gen = preset_generations.get(
            self.config.preset, self.config.max_generations or 100
        )

        opts: dict[str, Any] = {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxiter": max_gen,
            "tolx": self.config.tol_x,
            "tolfun": self.config.tol_fun,
            "verbose": -9,
        }
        if self.config.popsize is not None:
            opts["popsize"] = self.config.popsize

        # Override BIPOP -> none when warm-start is active (v2.21.0)
        restart_strategy = self.config.restart_strategy
        max_restarts = self.config.max_restarts
        if warmstart_active and restart_strategy == "bipop":
            restart_strategy = "none"
            max_restarts = 0
            logger.info(
                "[CMA-ES] Warm-start: overriding restart_strategy='bipop' -> 'none'"
            )

        # Build objective from model_func
        def _objective(x: np.ndarray) -> float:
            pred = model_func(xdata, *x)
            residuals = np.asarray(ydata) - np.asarray(pred)
            if sigma is not None:
                residuals = residuals / np.asarray(sigma)
            return float(np.sum(residuals**2))

        logger.info("[CMA-ES] Global search phase starting...")
        t0 = time.perf_counter()

        if restart_strategy == "bipop" and max_restarts > 0:
            logger.info("[CMA-ES] BIPOP: max_restarts=%d", max_restarts)
            xbest, es = cma.fmin2(
                _objective,
                p0.tolist(),
                effective_sigma,
                opts,
                restarts=max_restarts,
                bipop=True,
            )
            best_x = np.asarray(xbest, dtype=np.float64)
        else:
            es = cma.CMAEvolutionStrategy(p0.tolist(), effective_sigma, opts)
            es.optimize(_objective)
            best_x = np.asarray(es.result.xbest, dtype=np.float64)

        search_duration = time.perf_counter() - t0
        cmaes_chi_squared = float(es.result.fbest)
        n_evals = int(es.result.evaluations)
        n_iters = int(es.result.iterations)
        stop_conds = es.stop()
        convergence_reason = "; ".join(f"{k}={v}" for k, v in stop_conds.items())

        logger.info(
            "[CMA-ES] Global search completed: chi2=%.4e, "
            "generations=%d, evals=%d, wall=%.2fs",
            cmaes_chi_squared,
            n_iters,
            n_evals,
            search_duration,
        )

        diagnostics: dict[str, Any] = {
            "generations": n_iters,
            "evaluations": n_evals,
            "convergence_reason": convergence_reason,
            "global_search_time_s": search_duration,
        }

        # Early-exit: skip refinement if CMA-ES chi2 is much worse than warm-start
        skip_refinement = False
        if (
            warmstart_chi2 is not None
            and cmaes_chi_squared > _REFINEMENT_SKIP_CHI2_RATIO * warmstart_chi2
        ):
            skip_refinement = True
            logger.warning(
                "[CMA-ES] Skipping refinement: CMA-ES chi2=%.4e is >%.0fx "
                "worse than warm-start chi2=%.4e.",
                cmaes_chi_squared,
                _REFINEMENT_SKIP_CHI2_RATIO,
                warmstart_chi2,
            )

        # Optional NLSQ refinement (uses nlsq.curve_fit if available)
        best_params = best_x
        best_covariance: np.ndarray | None = None
        best_chi_squared = cmaes_chi_squared
        nlsq_refined = False

        if self.config.refine_with_nlsq and not skip_refinement:
            try:
                from nlsq import curve_fit as _nlsq_curve_fit

                popt, pcov = _nlsq_curve_fit(
                    f=model_func,
                    xdata=xdata,
                    ydata=ydata,
                    p0=best_x,
                    sigma=sigma,
                    bounds=bounds,
                    ftol=self.config.refinement_ftol,
                    xtol=self.config.refinement_xtol,
                    gtol=self.config.refinement_gtol,
                    max_nfev=self.config.refinement_max_nfev,
                    loss=self.config.refinement_loss,
                )
                popt_arr = np.asarray(popt, dtype=np.float64)
                refined_pred = model_func(xdata, *popt_arr)
                refined_res = np.asarray(ydata) - np.asarray(refined_pred)
                if sigma is not None:
                    refined_res = refined_res / np.asarray(sigma)
                refined_chi2 = float(np.sum(refined_res**2))

                best_params = popt_arr
                best_covariance = (
                    np.asarray(pcov, dtype=np.float64) if pcov is not None else None
                )
                best_chi_squared = refined_chi2
                nlsq_refined = True

                diagnostics["cmaes_chi_squared"] = cmaes_chi_squared
                diagnostics["refined_chi_squared"] = refined_chi2
                logger.info(
                    "[CMA-ES] Refinement improved chi2: %.4e -> %.4e",
                    cmaes_chi_squared,
                    refined_chi2,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[CMA-ES] Refinement failed: %s. Using CMA-ES result.", exc
                )
                diagnostics["refinement_failed"] = True

        converged_reasons = {"tol_fun", "tol_x", "tol_fun_hist", "ftarget"}
        raw_reason = (
            convergence_reason.split("=")[0] if "=" in convergence_reason else ""
        )
        success = raw_reason in converged_reasons or nlsq_refined

        return CMAESResult(
            parameters=best_params,
            covariance=best_covariance,
            chi_squared=best_chi_squared,
            success=success,
            diagnostics=diagnostics,
            method_used="cmaes",
            nlsq_refined=nlsq_refined,
            message=f"CMA-ES: {convergence_reason}",
        )

    # ------------------------------------------------------------------
    # Heterodyne-specific: objective-based fit → NLSQResult
    # ------------------------------------------------------------------

    def fit_with_objective(
        self,
        objective_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        *,
        residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        n_data: int | None = None,
        parameter_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NLSQResult:
        """Run CMA-ES optimization from a scalar objective function.

        Heterodyne-specific; not in homodyne.  Used by :func:`fit_with_cmaes`.

        Args:
            objective_fn: Scalar objective ``f(x) -> float`` to minimize.
            x0: Initial guess, shape ``(n_params,)``.
            bounds: ``(lower, upper)`` arrays, each shape ``(n_params,)``.
            residual_fn: Optional residual function for building a full
                :class:`NLSQResult` with covariance information.
            n_data: Number of data points (for reduced chi-squared).
            parameter_names: Override parameter names for this call.
            metadata: Extra metadata attached to the result.

        Returns:
            An :class:`NLSQResult` populated from the CMA-ES solution.

        Raises:
            ImportError: If the ``cma`` package is not installed.
        """
        if not _HAS_CMA:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install it with: uv add cma"
            )
        import cma  # noqa: F811  # type: ignore[import-untyped]

        # Build a CMAESConfig from CMAESWrapperConfig for cma opts
        preset_generations = {
            "cmaes-fast": 50,
            "cmaes": 100,
            "cmaes-global": 200,
        }
        max_gen = preset_generations.get(
            self.config.preset, self.config.max_generations or 100
        )

        names = parameter_names or self._parameter_names
        if not names:
            names = [f"p{i}" for i in range(len(x0))]

        x0 = np.asarray(x0, dtype=np.float64)
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)

        opts: dict[str, Any] = {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxiter": max_gen,
            "tolx": self.config.tol_x,
            "tolfun": self.config.tol_fun,
            "verbose": -9,
        }
        if self.config.popsize is not None:
            opts["popsize"] = self.config.popsize

        logger.info(
            "Starting CMA-ES (objective): sigma=%.3g, maxiter=%d, n_params=%d",
            self.config.sigma,
            max_gen,
            len(x0),
        )

        t0 = time.perf_counter()
        if self.config.restart_strategy == "bipop" and self.config.max_restarts > 0:
            logger.info("CMA-ES BIPOP: max_restarts=%d", self.config.max_restarts)
            xbest, es = cma.fmin2(
                objective_fn,
                x0.tolist(),
                self.config.sigma,
                opts,
                restarts=self.config.max_restarts,
                bipop=True,
            )
            best_x = np.asarray(xbest, dtype=np.float64)
        else:
            es = cma.CMAEvolutionStrategy(x0.tolist(), self.config.sigma, opts)
            es.optimize(objective_fn)
            best_x = np.asarray(es.result.xbest, dtype=np.float64)
        wall_time = time.perf_counter() - t0

        best_cost = float(es.result.fbest)
        n_evals = int(es.result.evaluations)
        n_iters = int(es.result.iterations)

        logger.info(
            "CMA-ES finished: cost=%.6e, evals=%d, iters=%d, wall=%.2fs",
            best_cost,
            n_evals,
            n_iters,
            wall_time,
        )

        if residual_fn is not None:
            residuals = np.asarray(residual_fn(best_x), dtype=np.float64)
        else:
            residuals = np.array([np.sqrt(max(best_cost, 0.0))], dtype=np.float64)

        effective_n_data = n_data if n_data is not None else max(len(residuals), 1)
        stop_reason = "; ".join(f"{k}={v}" for k, v in es.stop().items())
        success = best_cost < np.inf and not np.isnan(best_cost)

        result_metadata: dict[str, Any] = {
            "optimizer": "CMA-ES",
            "stop_conditions": es.stop(),
        }
        if metadata:
            result_metadata.update(metadata)

        return build_result_from_arrays(
            parameters=best_x,
            parameter_names=names,
            residuals=residuals,
            n_data=effective_n_data,
            success=success,
            message=stop_reason if stop_reason else "CMA-ES completed",
            n_iterations=n_iters,
            n_function_evals=n_evals,
            wall_time=wall_time,
            metadata=result_metadata,
        )


# =============================================================================
# Coordinate transformation utilities (heterodyne-specific)
# =============================================================================


def normalize_to_unit_cube(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Map parameter vector from physical bounds to the unit hypercube [0, 1].

    Args:
        x: Parameter vector of shape ``(n_params,)``.
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        Transformed array of shape ``(n_params,)`` with values in [0, 1].
        Dimensions where ``lower == upper`` are mapped to 0.0.

    Raises:
        ValueError: If arrays have mismatched shapes.
    """
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (x.shape == lower.shape == upper.shape):
        raise ValueError(
            f"x, lower, and upper must have the same shape; "
            f"got {x.shape}, {lower.shape}, {upper.shape}"
        )
    span = upper - lower
    safe_span = np.where(span > 0, span, 1.0)
    x_norm = np.where(span > 0, (x - lower) / safe_span, 0.0)
    return x_norm


def denormalize_from_unit_cube(
    x_norm: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Map parameter vector from the unit hypercube [0, 1] to physical bounds.

    This is the inverse of :func:`normalize_to_unit_cube`.

    Args:
        x_norm: Normalized parameter vector of shape ``(n_params,)``
            with values in [0, 1].
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        De-normalized array of shape ``(n_params,)`` in physical units.

    Raises:
        ValueError: If arrays have mismatched shapes.
    """
    x_norm = np.asarray(x_norm, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (x_norm.shape == lower.shape == upper.shape):
        raise ValueError(
            f"x_norm, lower, and upper must have the same shape; "
            f"got {x_norm.shape}, {lower.shape}, {upper.shape}"
        )
    return lower + x_norm * (upper - lower)


def compute_adaptive_cmaes_params(
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[int, int]:
    """Compute adaptive population size and max generations from bound ranges.

    Scales popsize from 50 to 200 and max_generations from 200 to 500 based
    on the ratio between the largest and smallest parameter ranges.

    Args:
        bounds: ``(lower, upper)`` arrays, each shape ``(n_params,)``.

    Returns:
        ``(popsize, max_generations)`` tuple.
    """
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    spans = upper - lower
    active_spans = spans[spans > 0]
    if len(active_spans) < 2:
        return 50, 200

    scale_ratio = float(active_spans.max() / active_spans.min())

    log_ratio = np.log10(max(scale_ratio, 1.0))
    t = min(log_ratio / 6.0, 1.0)
    popsize = int(50 + t * 150)
    max_generations = int(200 + t * 300)

    logger.debug(
        "Adaptive CMA-ES: scale_ratio=%.1f, popsize=%d, max_generations=%d",
        scale_ratio,
        popsize,
        max_generations,
    )
    return popsize, max_generations


def build_anti_degeneracy_objective(
    base_objective: Callable[[np.ndarray], float],
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_names: list[str],
    penalty_weight: float = 0.01,
) -> Callable[[np.ndarray], float]:
    """Wrap an objective with penalty terms for known degenerate parameter pairs.

    For each known degenerate pair (e.g. v0/v_offset, D0_ref/D0_sample),
    a soft penalty encourages separation of their normalized values.

    Args:
        base_objective: Original scalar objective ``f(x) -> float``.
        bounds: ``(lower, upper)`` arrays.
        parameter_names: Parameter names matching the indices in ``x``.
        penalty_weight: Strength of the penalty terms relative to the
            base cost.

    Returns:
        Wrapped objective function with degeneracy penalties.
    """
    from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
        _KNOWN_DEGENERATE_PAIRS,
    )

    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    spans = upper - lower
    safe_spans = np.where(spans > 0, spans, 1.0)

    name_to_idx = {n: i for i, n in enumerate(parameter_names)}
    active_pairs: list[tuple[int, int]] = []
    for name_a, name_b, _desc in _KNOWN_DEGENERATE_PAIRS:
        if name_a in name_to_idx and name_b in name_to_idx:
            active_pairs.append((name_to_idx[name_a], name_to_idx[name_b]))

    if not active_pairs:
        return base_objective

    def penalized_objective(x: np.ndarray) -> float:
        cost = base_objective(x)
        x_norm = (x - lower) / safe_spans
        penalty = 0.0
        for idx_a, idx_b in active_pairs:
            diff = abs(x_norm[idx_a] - x_norm[idx_b])
            penalty += 1.0 / (diff + 0.01)
        return cost + penalty_weight * penalty

    return penalized_objective


def fit_with_cmaes(
    objective_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_names: list[str] | None = None,
    *,
    config: CMAESConfig | None = None,
    residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_data: int | None = None,
    anti_degeneracy: bool = False,
    metadata: dict[str, Any] | None = None,
) -> NLSQResult:
    """Convenience function to run CMA-ES from a scalar objective.

    Applies adaptive population sizing and max_generations based on the
    parameter scale ratio when the user hasn't explicitly set these values.

    Args:
        objective_fn: Scalar objective ``f(x) -> float`` to minimize.
        initial_params: Initial guess, shape ``(n_params,)``.
        bounds: ``(lower, upper)`` arrays.
        parameter_names: Ordered parameter names.
        config: Low-level :class:`CMAESConfig`; ``None`` for defaults.
        residual_fn: Optional residual function for covariance estimation.
        n_data: Number of data points (for reduced chi-squared).
        anti_degeneracy: Apply degeneracy-penalty weighting to objective.
        metadata: Extra metadata attached to the result.

    Returns:
        An :class:`NLSQResult` from the CMA-ES solution.
    """
    cfg = config or CMAESConfig()

    # Adaptive population sizing
    if cfg.popsize is None or cfg.maxiter == CMAESConfig.maxiter:
        adaptive_pop, adaptive_gen = compute_adaptive_cmaes_params(bounds)
        if cfg.popsize is None:
            cfg = CMAESConfig(
                sigma0=cfg.sigma0,
                popsize=adaptive_pop,
                maxiter=cfg.maxiter
                if cfg.maxiter != CMAESConfig.maxiter
                else adaptive_gen,
                tolx=cfg.tolx,
                tolfun=cfg.tolfun,
                seed=cfg.seed,
                diagonal_filtering=cfg.diagonal_filtering,
            )
        elif cfg.maxiter == CMAESConfig.maxiter:
            cfg = CMAESConfig(
                sigma0=cfg.sigma0,
                popsize=cfg.popsize,
                maxiter=adaptive_gen,
                tolx=cfg.tolx,
                tolfun=cfg.tolfun,
                seed=cfg.seed,
                diagonal_filtering=cfg.diagonal_filtering,
            )

    names = parameter_names or [f"p{i}" for i in range(len(initial_params))]
    effective_objective = objective_fn
    if anti_degeneracy:
        effective_objective = build_anti_degeneracy_objective(
            objective_fn,
            bounds,
            names,
        )

    # Convert CMAESConfig → CMAESWrapperConfig for the wrapper
    wrapper_cfg = CMAESWrapperConfig(
        sigma=cfg.sigma0,
        tol_fun=cfg.tolfun,
        tol_x=cfg.tolx,
        restart_strategy=cfg.restart_strategy,
        max_restarts=cfg.max_restarts,
        refine_with_nlsq=False,  # no model_func available here
    )
    if cfg.popsize is not None:
        wrapper_cfg.popsize = cfg.popsize

    wrapper = CMAESWrapper(config=wrapper_cfg, parameter_names=names)
    return wrapper.fit_with_objective(
        objective_fn=effective_objective,
        x0=initial_params,
        bounds=bounds,
        residual_fn=residual_fn,
        n_data=n_data,
        parameter_names=names,
        metadata=metadata,
    )


def adjust_covariance_for_bounds(
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Scale a covariance matrix to account for parameter bounds ranges.

    When parameters have very different dynamic ranges, a unit-cube
    covariance should be scaled to physical space.  The adjustment is:

        cov_adjusted[i, j] = cov[i, j] * (upper[i] - lower[i])
                                        * (upper[j] - lower[j])

    Args:
        cov: Covariance matrix of shape ``(n_params, n_params)``.
        lower: Lower bound array of shape ``(n_params,)``.
        upper: Upper bound array of shape ``(n_params,)``.

    Returns:
        Adjusted covariance matrix of shape ``(n_params, n_params)``.

    Raises:
        ValueError: If array shapes are inconsistent.
    """
    cov = np.asarray(cov, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    n = len(lower)
    if cov.shape != (n, n):
        raise ValueError(
            f"cov must be square with side length matching bounds ({n}), "
            f"got shape {cov.shape}"
        )
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError(
            f"lower and upper must have shape ({n},), "
            f"got {lower.shape} and {upper.shape}"
        )

    spans = upper - lower
    scale = np.outer(spans, spans)
    result = cov * scale

    min_eig = float(np.linalg.eigvalsh(result).min())
    if min_eig < 0:
        logger.warning(
            "adjust_covariance_for_bounds: result is not PSD "
            "(min eigenvalue=%.2e). Downstream consumers may need "
            "to apply a PSD projection.",
            min_eig,
        )

    return result
