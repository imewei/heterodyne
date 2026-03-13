"""Unit tests for HybridStreamingStrategy.

Verifies:
- No scipy.optimize imports in the source module.
- AdaptiveHybridStreamingOptimizer (nlsq) import present.
- End-to-end smoke test with mocked optimizer.
- Graceful fallback to nlsq.curve_fit_large when streaming unavailable.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "heterodyne/optimization/nlsq/strategies/hybrid_streaming.py"
)
_MODULE_NAME = "heterodyne.optimization.nlsq.strategies.hybrid_streaming"


def _source() -> str:
    """Return the source text of the strategy module."""
    return _MODULE_PATH.read_text(encoding="utf-8")


def _ast_imports(source: str) -> list[str]:
    """Return all dotted module names that appear in import statements."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _make_model(n_params: int = 4) -> Any:
    """Build a minimal HeterodyneModel mock."""
    pm = MagicMock()
    pm.get_initial_values.return_value = np.ones(n_params) * 0.5
    pm.get_bounds.return_value = (
        np.zeros(n_params),
        np.ones(n_params),
    )
    pm.get_full_values.return_value = np.ones(n_params) * 0.5
    pm.varying_indices = list(range(n_params))
    pm.varying_names = [f"p{i}" for i in range(n_params)]
    pm.expand_varying_to_full.side_effect = lambda x: np.asarray(x)

    model = MagicMock()
    model.param_manager = pm
    model.t = np.linspace(0.001, 1.0, 8)
    model.q = np.array([0.01])
    model.dt = float(model.t[1] - model.t[0])
    return model


def _make_config(**overrides: Any) -> Any:
    """Build a minimal NLSQConfig mock for hybrid_streaming fields."""
    cfg = MagicMock()
    cfg.hybrid_normalization = True
    cfg.hybrid_method = "lbfgs"
    cfg.hybrid_warmup_fraction = 0.1
    cfg.hybrid_lbfgs_memory = 10
    cfg.max_iterations = 200
    cfg.method = "trf"
    cfg.ftol = 1e-6
    cfg.xtol = 1e-6
    cfg.gtol = 1e-6
    cfg.loss = "linear"
    cfg.verbose = 0
    cfg.streaming_chunk_size = 1000
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_streaming_dict_result(n_params: int = 4, n_data: int = 64) -> dict[str, Any]:
    """Build a dict that mimics AdaptiveHybridStreamingOptimizer.fit() output."""
    rng = np.random.default_rng(0)
    return {
        "x": rng.uniform(0.0, 1.0, n_params),
        "pcov": np.eye(n_params) * 1e-4,
        "fun": rng.normal(0, 0.01, n_data),
        "success": True,
        "message": "Converged",
        "streaming_diagnostics": {"epochs": 5, "best_loss": 0.001},
        "nfev": 120,
    }


# ---------------------------------------------------------------------------
# Test 1: No scipy.optimize in source
# ---------------------------------------------------------------------------


class TestNoScipyImport:
    """Verify that scipy.optimize is not imported by the strategy module."""

    def test_no_scipy_import_in_source(self) -> None:
        """AST-level check: scipy.optimize must not appear in any import."""
        imports = _ast_imports(_source())
        scipy_imports = [m for m in imports if m.startswith("scipy.optimize")]
        assert scipy_imports == [], (
            f"Found scipy.optimize imports: {scipy_imports}. "
            "hybrid_streaming.py must not import from scipy.optimize."
        )

    def test_no_scipy_optimize_string_in_source(self) -> None:
        """Textual guard: the string 'scipy.optimize' must not appear at all."""
        source = _source()
        assert "scipy.optimize" not in source, (
            "The literal string 'scipy.optimize' was found in hybrid_streaming.py."
        )


# ---------------------------------------------------------------------------
# Test 2: nlsq import present in source
# ---------------------------------------------------------------------------


class TestUsesStreamingOptimizer:
    """Verify that the nlsq package is imported in the strategy module."""

    def test_nlsq_import_present(self) -> None:
        """At least one 'nlsq' import must be present in the source."""
        imports = _ast_imports(_source())
        nlsq_imports = [m for m in imports if m == "nlsq" or m.startswith("nlsq.")]
        assert nlsq_imports, (
            "No 'nlsq' imports found in hybrid_streaming.py. "
            "AdaptiveHybridStreamingOptimizer must be imported from nlsq."
        )

    def test_adaptive_hybrid_streaming_optimizer_referenced(self) -> None:
        """The class name must appear in the source text."""
        assert "AdaptiveHybridStreamingOptimizer" in _source(), (
            "AdaptiveHybridStreamingOptimizer not found in hybrid_streaming.py source."
        )

    def test_hybrid_streaming_config_referenced(self) -> None:
        """HybridStreamingConfig must appear in the source text."""
        assert "HybridStreamingConfig" in _source(), (
            "HybridStreamingConfig not found in hybrid_streaming.py source."
        )


# ---------------------------------------------------------------------------
# Test 3: Smoke test — mocked end-to-end fit
# ---------------------------------------------------------------------------


class TestSmokeFit:
    """End-to-end smoke test with all heavy dependencies mocked."""

    def _run_fit(
        self,
        streaming_result: dict[str, Any] | None = None,
        n_params: int = 4,
        n_rows: int = 8,
    ) -> Any:
        """Execute HybridStreamingStrategy.fit() with fully mocked dependencies."""
        from heterodyne.optimization.nlsq.strategies.hybrid_streaming import (
            HybridStreamingStrategy,
        )

        n_data = n_rows * n_rows
        if streaming_result is None:
            streaming_result = _make_streaming_dict_result(n_params, n_data)

        model = _make_model(n_params)
        config = _make_config()
        c2_data = np.random.default_rng(42).uniform(1.0, 1.5, (n_rows, n_rows))

        # Mock the streaming optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.fit.return_value = streaming_result

        mock_optimizer_cls = MagicMock(return_value=mock_optimizer)
        mock_config_cls = MagicMock()

        # Mock JAX backend calls so no real physics is computed
        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".AdaptiveHybridStreamingOptimizer",
                mock_optimizer_cls,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".HybridStreamingConfig",
                mock_config_cls,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".STREAMING_AVAILABLE",
                True,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_residuals",
                return_value=np.zeros(n_data),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_c2_heterodyne",
                return_value=np.ones((n_rows, n_rows)),
            ),
        ):
            strategy = HybridStreamingStrategy()
            return strategy.fit(model, c2_data, phi_angle=0.0, config=config)

    def test_returns_strategy_result(self) -> None:
        """fit() must return a StrategyResult instance."""
        from heterodyne.optimization.nlsq.strategies.base import StrategyResult

        result = self._run_fit()
        assert isinstance(result, StrategyResult)

    def test_strategy_name(self) -> None:
        """strategy_name in StrategyResult must be 'hybrid_streaming'."""
        result = self._run_fit()
        assert result.strategy_name == "hybrid_streaming"

    def test_nlsq_result_has_parameters(self) -> None:
        """NLSQResult must carry fitted parameters of correct length."""
        n_params = 4
        result = self._run_fit(n_params=n_params)
        assert result.result.parameters is not None
        assert len(result.result.parameters) == n_params

    def test_nlsq_result_success(self) -> None:
        """NLSQResult.success must be True when optimizer reports success."""
        result = self._run_fit()
        assert result.result.success is True

    def test_nlsq_result_has_parameter_names(self) -> None:
        """NLSQResult.parameter_names must be populated."""
        result = self._run_fit()
        assert len(result.result.parameter_names) == 4

    def test_metadata_strategy_key(self) -> None:
        """Metadata must carry 'strategy' = 'hybrid_streaming'."""
        result = self._run_fit()
        assert result.metadata.get("strategy") == "hybrid_streaming"

    def test_wall_time_recorded(self) -> None:
        """wall_time_seconds must be a non-negative float."""
        result = self._run_fit()
        wt = result.result.wall_time_seconds
        assert wt is not None and wt >= 0.0

    def test_optimizer_fit_called_once(self) -> None:
        """The streaming optimizer's fit() must be called exactly once."""
        from heterodyne.optimization.nlsq.strategies.hybrid_streaming import (
            HybridStreamingStrategy,
        )

        n_params = 4
        n_rows = 8
        n_data = n_rows * n_rows
        streaming_result = _make_streaming_dict_result(n_params, n_data)
        model = _make_model(n_params)
        config = _make_config()
        c2_data = np.random.default_rng(42).uniform(1.0, 1.5, (n_rows, n_rows))

        mock_optimizer = MagicMock()
        mock_optimizer.fit.return_value = streaming_result
        mock_optimizer_cls = MagicMock(return_value=mock_optimizer)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".AdaptiveHybridStreamingOptimizer",
                mock_optimizer_cls,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".HybridStreamingConfig",
                MagicMock(),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".STREAMING_AVAILABLE",
                True,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_residuals",
                return_value=np.zeros(n_data),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_c2_heterodyne",
                return_value=np.ones((n_rows, n_rows)),
            ),
        ):
            HybridStreamingStrategy().fit(model, c2_data, phi_angle=0.0, config=config)

        mock_optimizer.fit.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Fallback to curve_fit_large when streaming unavailable
# ---------------------------------------------------------------------------


class TestStreamingUnavailableFallback:
    """When STREAMING_AVAILABLE=False, must fall back to nlsq.curve_fit_large."""

    def test_fallback_uses_curve_fit_large(self) -> None:
        """curve_fit_large must be called when streaming is unavailable."""
        from heterodyne.optimization.nlsq.strategies.hybrid_streaming import (
            HybridStreamingStrategy,
        )

        n_params = 4
        n_rows = 8
        n_data = n_rows * n_rows
        model = _make_model(n_params)
        config = _make_config()
        c2_data = np.random.default_rng(7).uniform(1.0, 1.5, (n_rows, n_rows))

        # curve_fit_large returns (popt, pcov)
        popt = np.ones(n_params) * 0.6
        pcov = np.eye(n_params) * 1e-3
        fallback_result = (popt, pcov)

        mock_curve_fit_large = MagicMock(return_value=fallback_result)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".STREAMING_AVAILABLE",
                False,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".curve_fit_large",
                mock_curve_fit_large,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_residuals",
                return_value=np.zeros(n_data),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_c2_heterodyne",
                return_value=np.ones((n_rows, n_rows)),
            ),
        ):
            strategy = HybridStreamingStrategy()
            result = strategy.fit(model, c2_data, phi_angle=0.0, config=config)

        mock_curve_fit_large.assert_called_once()
        from heterodyne.optimization.nlsq.strategies.base import StrategyResult

        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "hybrid_streaming"

    def test_fallback_result_has_correct_parameters(self) -> None:
        """Fallback result must contain the popt from curve_fit_large."""
        from heterodyne.optimization.nlsq.strategies.hybrid_streaming import (
            HybridStreamingStrategy,
        )

        n_params = 4
        n_rows = 8
        n_data = n_rows * n_rows
        model = _make_model(n_params)
        config = _make_config()
        c2_data = np.random.default_rng(7).uniform(1.0, 1.5, (n_rows, n_rows))

        popt = np.linspace(0.1, 0.4, n_params)
        pcov = np.eye(n_params) * 5e-4
        fallback_result = (popt, pcov)

        with (
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".STREAMING_AVAILABLE",
                False,
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".curve_fit_large",
                MagicMock(return_value=fallback_result),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_residuals",
                return_value=np.zeros(n_data),
            ),
            patch(
                "heterodyne.optimization.nlsq.strategies.hybrid_streaming"
                ".compute_c2_heterodyne",
                return_value=np.ones((n_rows, n_rows)),
            ),
        ):
            result = HybridStreamingStrategy().fit(
                model, c2_data, phi_angle=0.0, config=config
            )

        np.testing.assert_allclose(result.result.parameters, popt)
