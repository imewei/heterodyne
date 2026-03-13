"""Smoke tests verifying that the 6 strategy files use nlsq, not scipy.

Each test class covers one strategy and checks:
1. No ``scipy.optimize`` import remains in the module.
2. The correct nlsq symbol is imported.
3. A basic smoke test using a mocked HeterodyneModel confirms the strategy
   can be constructed and its ``fit()`` method is callable without raising.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _fake_param_manager(n_params: int = 3) -> Any:
    """Build a minimal ParameterManager-like namespace."""
    vals = np.linspace(0.5, 1.5, n_params)
    lo = np.zeros(n_params)
    hi = np.full(n_params, 10.0)
    pm = SimpleNamespace(
        get_initial_values=lambda: vals.copy(),
        get_bounds=lambda: (lo.copy(), hi.copy()),
        get_full_values=lambda: vals.copy(),
        varying_indices=list(range(n_params)),
        varying_names=[f"p{i}" for i in range(n_params)],
        expand_varying_to_full=lambda x: np.asarray(x),
    )
    return pm


def _fake_model(n_params: int = 3, n_t: int = 5) -> Any:
    """Build a minimal HeterodyneModel-like namespace."""
    t = jnp.linspace(0.001, 1.0, n_t)
    model = SimpleNamespace(
        param_manager=_fake_param_manager(n_params),
        t=t,
        q=0.01,
        dt=0.001,
        set_params=lambda x: None,
    )
    return model


def _fake_c2(n_t: int = 5) -> np.ndarray:
    """Return a small synthetic correlation matrix."""
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 1.1, size=(n_t, n_t)).astype(np.float64)


def _fake_nlsq_result(n_params: int = 3, n_data: int = 25) -> Any:
    """Build a plausible CurveFitResult-like SimpleNamespace."""
    return SimpleNamespace(
        x=np.ones(n_params, dtype=np.float64),
        fun=np.zeros(n_data, dtype=np.float64),
        jac=np.zeros((n_data, n_params), dtype=np.float64),
        cost=0.0,
        success=True,
        message="converged",
        nfev=10,
        njev=2,
        status=1,
    )


def _fake_config(
    method: str = "trf",
    use_jac: bool = False,
    max_nfev: int | None = None,
    max_iterations: int = 50,
    chunk_size: int | None = None,
    target_chunk_size: int | None = None,
) -> Any:
    return SimpleNamespace(
        method=method,
        use_jac=use_jac,
        max_nfev=max_nfev,
        max_iterations=max_iterations,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        loss="linear",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Helper: check no scipy.optimize import in module source
# ---------------------------------------------------------------------------


def _assert_no_scipy_import(module_path: str) -> None:
    """Assert that a module file does not contain 'scipy.optimize' imports."""
    spec = importlib.util.find_spec(module_path)
    assert spec is not None, f"Cannot locate module {module_path}"
    assert spec.origin is not None
    with open(spec.origin) as fh:
        source = fh.read()
    assert "scipy.optimize" not in source, (
        f"{module_path} still contains 'scipy.optimize'. "
        "All scipy optimizer calls must be replaced with nlsq."
    )


def _assert_nlsq_import(module_path: str, symbol: str) -> None:
    """Assert that a module source contains an nlsq import for the given symbol."""
    spec = importlib.util.find_spec(module_path)
    assert spec is not None
    assert spec.origin is not None
    with open(spec.origin) as fh:
        source = fh.read()
    assert any(
        "from nlsq import" in line and symbol in line for line in source.splitlines()
    ), f"{module_path} does not import '{symbol}' from nlsq."


# ---------------------------------------------------------------------------
# ResidualStrategy
# ---------------------------------------------------------------------------


class TestResidualStrategyNlsq:
    """Verify ResidualStrategy uses nlsq.CurveFit."""

    MODULE = "heterodyne.optimization.nlsq.strategies.residual"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit(self) -> None:
        _assert_nlsq_import(self.MODULE, "CurveFit")

    def test_smoke_fit(self) -> None:
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(use_jac=False)

        fitter_mock = MagicMock()
        fitter_mock.curve_fit.return_value = _fake_nlsq_result(
            n_params=3, n_data=c2.size
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.CurveFit",
                return_value=fitter_mock,
            ),
        ):
            strategy = ResidualStrategy(use_analytic_jac=False)
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "residual"
        assert sr.result.success is True

    def test_dogbox_coercion(self) -> None:
        """dogbox method should be coerced to trf with a warning."""
        from heterodyne.optimization.nlsq.strategies.residual import ResidualStrategy

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(method="dogbox", use_jac=False)

        fitter_mock = MagicMock()
        fitter_mock.curve_fit.return_value = _fake_nlsq_result(
            n_params=3, n_data=c2.size
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual.CurveFit",
                return_value=fitter_mock,
            ),
        ):
            strategy = ResidualStrategy(use_analytic_jac=False)
            _sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        # The call should succeed and internally use trf
        _, call_kwargs = fitter_mock.curve_fit.call_args
        assert call_kwargs.get("method") in (None, "trf")


# ---------------------------------------------------------------------------
# JITStrategy
# ---------------------------------------------------------------------------


class TestJITStrategyNlsq:
    """Verify JITStrategy uses nlsq.CurveFit."""

    MODULE = "heterodyne.optimization.nlsq.strategies.jit_strategy"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit(self) -> None:
        _assert_nlsq_import(self.MODULE, "CurveFit")

    def test_smoke_fit_with_fallback(self) -> None:
        """JITStrategy falls back gracefully to ResidualStrategy on JIT failure."""
        from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(use_jac=False)

        # Force the compilation step to raise so fallback is exercised
        with patch.object(
            JITStrategy,
            "_get_or_compile",
            side_effect=RuntimeError("forced compile failure"),
        ):
            # The fallback calls ResidualStrategy.fit; mock that as well
            fitter_mock = MagicMock()
            fitter_mock.curve_fit.return_value = _fake_nlsq_result(
                n_params=3, n_data=c2.size
            )
            with (
                patch(
                    "heterodyne.optimization.nlsq.strategies.residual.compute_residuals",
                    return_value=jnp.zeros(c2.size),
                ),
                patch(
                    "heterodyne.optimization.nlsq.strategies.residual.compute_c2_heterodyne",
                    return_value=jnp.ones_like(jnp.asarray(c2)),
                ),
                patch(
                    "heterodyne.optimization.nlsq.strategies.residual.CurveFit",
                    return_value=fitter_mock,
                ),
            ):
                strategy = JITStrategy()
                sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.metadata.get("fallback") is True
        assert sr.metadata.get("original_strategy") == "jit"

    def test_dogbox_coercion(self) -> None:
        """dogbox method inside JITStrategy compile path is coerced to trf."""
        from heterodyne.optimization.nlsq.strategies.jit_strategy import JITStrategy

        # We only verify the coercion logic: patch compile to succeed then mock fitter.
        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(method="dogbox", use_jac=False)

        fitter_mock = MagicMock()
        fitter_mock.curve_fit.return_value = _fake_nlsq_result(
            n_params=3, n_data=c2.size
        )

        dummy_jit_fn = lambda varying: jnp.zeros(c2.size)  # noqa: E731

        with (
            patch.object(
                JITStrategy,
                "_get_or_compile",
                return_value=(dummy_jit_fn, None),
            ),
            patch.object(JITStrategy, "_last_compile_time", 0.0, create=True),
            patch(
                "heterodyne.optimization.nlsq.strategies.jit_strategy.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.jit_strategy.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.jit_strategy.CurveFit",
                return_value=fitter_mock,
            ),
        ):
            strategy = JITStrategy()
            strategy._last_compile_time = 0.0  # initialise before fit
            _sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        _, call_kwargs = fitter_mock.curve_fit.call_args
        assert call_kwargs.get("method") in (None, "trf")


# ---------------------------------------------------------------------------
# ResidualJITStrategy
# ---------------------------------------------------------------------------


class TestResidualJITStrategyNlsq:
    """Verify ResidualJITStrategy uses nlsq.CurveFit."""

    MODULE = "heterodyne.optimization.nlsq.strategies.residual_jit"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit(self) -> None:
        _assert_nlsq_import(self.MODULE, "CurveFit")

    def test_smoke_fit(self) -> None:
        from heterodyne.optimization.nlsq.strategies.residual_jit import (
            ResidualJITStrategy,
        )

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config()

        fitter_mock = MagicMock()
        fitter_mock.curve_fit.return_value = _fake_nlsq_result(
            n_params=3, n_data=c2.size
        )

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.residual_jit.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual_jit.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.residual_jit.CurveFit",
                return_value=fitter_mock,
            ),
        ):
            strategy = ResidualJITStrategy()
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "residual_jit"
        assert sr.result.success is True


# ---------------------------------------------------------------------------
# ChunkedStrategy
# ---------------------------------------------------------------------------


class TestChunkedStrategyNlsq:
    """Verify ChunkedStrategy uses nlsq.curve_fit_large."""

    MODULE = "heterodyne.optimization.nlsq.strategies.chunked"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit_large(self) -> None:
        _assert_nlsq_import(self.MODULE, "curve_fit_large")

    def test_smoke_fit(self) -> None:
        from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(chunk_size=10)

        mock_result = _fake_nlsq_result(n_params=3, n_data=c2.size)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.curve_fit_large",
                return_value=mock_result,
            ),
        ):
            strategy = ChunkedStrategy(chunk_size=10)
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "chunked"
        assert sr.result.success is True

    def test_failure_path_marks_partial_failure(self) -> None:
        """When compute_residuals raises, result.success is False."""
        from heterodyne.optimization.nlsq.strategies.chunked import ChunkedStrategy

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(chunk_size=10)

        mock_result = _fake_nlsq_result(n_params=3, n_data=c2.size)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.compute_residuals",
                side_effect=RuntimeError("deliberate test failure"),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.chunked.curve_fit_large",
                return_value=mock_result,
            ),
        ):
            strategy = ChunkedStrategy(chunk_size=10)
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "chunked"
        assert sr.result.success is False
        assert sr.result.metadata["partial_failure"] is True


# ---------------------------------------------------------------------------
# OutOfCoreStrategy
# ---------------------------------------------------------------------------


class TestOutOfCoreStrategyNlsq:
    """Verify OutOfCoreStrategy uses nlsq.curve_fit_large."""

    MODULE = "heterodyne.optimization.nlsq.strategies.out_of_core"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit_large(self) -> None:
        _assert_nlsq_import(self.MODULE, "curve_fit_large")

    def test_smoke_fit(self) -> None:
        from heterodyne.optimization.nlsq.strategies.out_of_core import (
            OutOfCoreStrategy,
        )

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config()

        mock_result = _fake_nlsq_result(n_params=3, n_data=c2.size)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.out_of_core.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.out_of_core.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.out_of_core.curve_fit_large",
                return_value=mock_result,
            ),
        ):
            strategy = OutOfCoreStrategy(chunk_size=10)
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "out_of_core"
        assert sr.result.success is True


# ---------------------------------------------------------------------------
# StratifiedLSStrategy
# ---------------------------------------------------------------------------


class TestStratifiedLSStrategyNlsq:
    """Verify StratifiedLSStrategy uses nlsq.curve_fit_large."""

    MODULE = "heterodyne.optimization.nlsq.strategies.stratified_ls"

    def test_no_scipy_import(self) -> None:
        _assert_no_scipy_import(self.MODULE)

    def test_uses_curve_fit_large(self) -> None:
        _assert_nlsq_import(self.MODULE, "curve_fit_large")

    def test_smoke_fit(self) -> None:
        from heterodyne.optimization.nlsq.strategies.stratified_ls import (
            StratifiedLSStrategy,
        )

        n_t = 5
        model = _fake_model(n_params=3, n_t=n_t)
        c2 = _fake_c2(n_t=n_t)
        config = _fake_config(target_chunk_size=10)

        mock_result = _fake_nlsq_result(n_params=3, n_data=c2.size)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.stratified_ls.compute_residuals",
                return_value=jnp.zeros(c2.size),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.stratified_ls.compute_c2_heterodyne",
                return_value=jnp.ones_like(jnp.asarray(c2)),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.stratified_ls.curve_fit_large",
                return_value=mock_result,
            ),
        ):
            strategy = StratifiedLSStrategy()
            sr = strategy.fit(model, c2, phi_angle=0.0, config=config)

        assert sr.strategy_name == "stratified_ls"
        assert sr.result.success is True
