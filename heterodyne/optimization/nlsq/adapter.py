"""NLSQ adapter using the nlsq library (CurveFit)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from heterodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model cache — avoids re-JIT-compiling CurveFit for identical problem shapes
# ---------------------------------------------------------------------------
_MODEL_CACHE_MAX_SIZE = 8


@dataclass(frozen=True)
class ModelCacheKey:
    """Cache key for CurveFit instances."""

    n_data: int
    n_params: int


@dataclass
class CachedModel:
    """A cached CurveFit instance with usage stats."""

    fitter: object  # nlsq.CurveFit
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    n_hits: int = 0


_model_cache: dict[ModelCacheKey, CachedModel] = {}
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}


def get_or_create_fitter(n_data: int, n_params: int) -> tuple[object, bool]:
    """Get a CurveFit instance from cache or create a new one.

    Args:
        n_data: Number of data points (flength).
        n_params: Number of parameters.

    Returns:
        Tuple of (CurveFit fitter, cache_hit: bool).
    """
    from nlsq import CurveFit

    key = ModelCacheKey(n_data=n_data, n_params=n_params)

    if key in _model_cache:
        _model_cache[key].last_accessed = time.monotonic()
        _model_cache[key].n_hits += 1
        _cache_stats["hits"] += 1
        return _model_cache[key].fitter, True

    _cache_stats["misses"] += 1

    # Evict oldest entry if cache is full
    if len(_model_cache) >= _MODEL_CACHE_MAX_SIZE:
        oldest_key = min(_model_cache, key=lambda k: _model_cache[k].last_accessed)
        del _model_cache[oldest_key]

    fitter = CurveFit(flength=float(n_data))
    _model_cache[key] = CachedModel(fitter=fitter)
    return fitter, False


def clear_model_cache() -> None:
    """Clear the CurveFit model cache."""
    _model_cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


def get_cache_stats() -> dict[str, int]:
    """Get cache hit/miss statistics."""
    return {**_cache_stats, "size": len(_model_cache)}


class NLSQAdapter(NLSQAdapterBase):
    """Adapter for the nlsq library's CurveFit optimizer.

    Uses JAX-accelerated nonlinear least squares from the nlsq package.
    """

    def __init__(self, parameter_names: list[str]) -> None:
        """Initialize adapter.

        Args:
            parameter_names: Names of parameters being optimized
        """
        self._parameter_names = parameter_names

    @property
    def name(self) -> str:
        return "nlsq.CurveFit"

    def supports_bounds(self) -> bool:
        return True

    def supports_jacobian(self) -> bool:
        return True

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run NLSQ optimization using nlsq library.

        Note: For JAX-traced optimization, use fit_jax() instead.
        This method falls back to scipy for numpy-based residual functions.

        Args:
            residual_fn: Function computing residuals (numpy)
            initial_params: Starting parameter values
            bounds: (lower, upper) bound arrays
            config: Optimization configuration
            jacobian_fn: Optional analytic Jacobian function

        Returns:
            NLSQResult with fit results
        """
        # Fall back to scipy since the residual_fn is numpy-based
        # The nlsq library requires pure JAX functions for tracing
        logger.debug("NLSQAdapter delegating to ScipyNLSQAdapter for numpy residual_fn")
        scipy_adapter = ScipyNLSQAdapter(parameter_names=self._parameter_names)
        return scipy_adapter.fit(
            residual_fn=residual_fn,
            initial_params=initial_params,
            bounds=bounds,
            config=config,
            jacobian_fn=jacobian_fn,
        )

    def fit_jax(
        self,
        jax_residual_fn: Callable,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        n_data: int,
    ) -> NLSQResult:
        """Run NLSQ optimization using nlsq library with JAX-traced function.

        This method accepts a pure JAX function that nlsq can trace.

        Args:
            jax_residual_fn: JAX-compatible function (x, *params) -> residuals
            initial_params: Starting parameter values
            bounds: (lower, upper) bound arrays
            config: Optimization configuration
            n_data: Number of data points (for flength)

        Returns:
            NLSQResult with fit results
        """
        start_time = time.perf_counter()

        lower_bounds, upper_bounds = bounds
        n_params = len(initial_params)

        # Validate bounds
        initial_params = np.clip(initial_params, lower_bounds, upper_bounds)

        logger.info(f"Starting NLSQ fit (JAX) with {n_params} parameters")

        try:
            # Create xdata and ydata for nlsq API
            xdata = np.arange(n_data, dtype=np.float64)
            ydata = np.zeros(n_data, dtype=np.float64)  # Target is zero residuals

            # Get or create CurveFit instance (cached by shape)
            fitter, cache_hit = get_or_create_fitter(n_data, n_params)
            if cache_hit:
                logger.debug(f"CurveFit cache hit for shape ({n_data}, {n_params})")

            # Run optimization
            fitted_params, covariance = fitter.curve_fit(
                f=jax_residual_fn,
                xdata=xdata,
                ydata=ydata,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=config.method if config.method != "dogbox" else "trf",
            )

            # Compute final residuals and cost (using the JAX function)
            import jax.numpy as jnp
            final_residuals_jax = jax_residual_fn(jnp.arange(n_data), *fitted_params)
            final_residuals = np.asarray(final_residuals_jax)
            final_cost = 0.5 * np.sum(final_residuals ** 2)

            # Degrees of freedom
            n_dof = n_data - n_params
            reduced_chi2 = final_cost / n_dof if n_dof > 0 else None

            # Get uncertainties from covariance
            uncertainties = None
            if covariance is not None:
                try:
                    with np.errstate(invalid='raise'):
                        uncertainties = np.sqrt(np.diag(covariance))
                except Exception:
                    logger.warning("Could not extract uncertainties from covariance")

            wall_time = time.perf_counter() - start_time

            return NLSQResult(
                parameters=np.asarray(fitted_params),
                parameter_names=self._parameter_names,
                success=True,
                message="Optimization converged",
                uncertainties=uncertainties,
                covariance=np.asarray(covariance) if covariance is not None else None,
                final_cost=final_cost,
                reduced_chi_squared=reduced_chi2,
                n_iterations=0,
                n_function_evals=0,
                convergence_reason="tolerance",
                residuals=final_residuals,
                wall_time_seconds=wall_time,
            )

        except Exception as e:
            logger.error(f"NLSQ optimization failed: {e}")
            wall_time = time.perf_counter() - start_time

            return NLSQResult(
                parameters=initial_params,
                parameter_names=self._parameter_names,
                success=False,
                message=str(e),
                wall_time_seconds=wall_time,
            )


class ScipyNLSQAdapter(NLSQAdapterBase):
    """Fallback adapter using scipy.optimize.least_squares."""

    def __init__(self, parameter_names: list[str]) -> None:
        """Initialize adapter.

        Args:
            parameter_names: Names of parameters being optimized
        """
        self._parameter_names = parameter_names

    @property
    def name(self) -> str:
        return "scipy.optimize.least_squares"

    def supports_bounds(self) -> bool:
        return True

    def supports_jacobian(self) -> bool:
        return True

    def fit(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        config: NLSQConfig,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> NLSQResult:
        """Run optimization using scipy.

        Args:
            residual_fn: Function computing residuals
            initial_params: Starting values
            bounds: Bound arrays
            config: Configuration
            jacobian_fn: Optional Jacobian

        Returns:
            NLSQResult
        """
        from scipy.optimize import least_squares

        start_time = time.perf_counter()

        lower_bounds, upper_bounds = bounds
        initial_params = np.clip(initial_params, lower_bounds, upper_bounds)

        logger.info(f"Starting scipy NLSQ with {len(initial_params)} parameters")

        try:
            # Prepare jacobian argument
            jac = jacobian_fn if config.use_jac and jacobian_fn else '2-point'

            result = least_squares(
                residual_fn,
                initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=config.method,
                jac=jac,
                ftol=config.ftol,
                xtol=config.xtol,
                gtol=config.gtol,
                max_nfev=config.max_nfev or config.max_iterations * len(initial_params),
                verbose=2 if config.verbose > 1 else 0,
                loss=config.loss,
            )

            fitted_params = result.x
            final_residuals = result.fun
            final_cost = result.cost

            # Degrees of freedom
            n_data = len(final_residuals)
            n_params = len(fitted_params)
            n_dof = n_data - n_params
            reduced_chi2 = 2 * final_cost / n_dof if n_dof > 0 else None

            # Compute covariance from Jacobian
            covariance = None
            uncertainties = None
            if result.jac is not None:
                try:
                    # J^T J approximates the Hessian
                    jtj = result.jac.T @ result.jac
                    # Covariance = (J^T J)^-1 * s^2 where s^2 is residual variance
                    s2 = 2 * final_cost / n_dof if n_dof > 0 else 1.0
                    cond = np.linalg.cond(jtj)
                    if cond > 1e12:
                        logger.warning(
                            f"Ill-conditioned J^T J (cond={cond:.2e}), "
                            f"using pseudo-inverse"
                        )
                        covariance = np.linalg.pinv(jtj) * s2
                    else:
                        covariance = np.linalg.inv(jtj) * s2
                    uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                except np.linalg.LinAlgError:
                    logger.warning("Could not compute covariance matrix")

            wall_time = time.perf_counter() - start_time

            return NLSQResult(
                parameters=fitted_params,
                parameter_names=self._parameter_names,
                success=result.success,
                message=result.message,
                uncertainties=uncertainties,
                covariance=covariance,
                final_cost=final_cost,
                reduced_chi_squared=reduced_chi2,
                n_iterations=result.njev if hasattr(result, 'njev') else 0,
                n_function_evals=result.nfev,
                convergence_reason=str(result.status),
                residuals=final_residuals,
                jacobian=result.jac,
                wall_time_seconds=wall_time,
            )

        except Exception as e:
            logger.error(f"Scipy optimization failed: {e}")
            wall_time = time.perf_counter() - start_time

            return NLSQResult(
                parameters=initial_params,
                parameter_names=self._parameter_names,
                success=False,
                message=str(e),
                wall_time_seconds=wall_time,
            )
