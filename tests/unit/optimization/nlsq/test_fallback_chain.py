"""Unit tests for fallback_chain.py — NLSQ-native routing.

Tests cover:
- Descending fallback order: STREAMING → LARGE → STANDARD → None
- handle_nlsq_result() normalization for all 4 return formats
- get_fallback_strategy() logic including memory-error skip
- CHUNKED not present in enum
- execute_optimization_with_fallback() routing
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.optimization.nlsq.fallback_chain import (
    OptimizationStrategy,
    _FALLBACK_ORDER,
    _is_memory_error,
    execute_optimization_with_fallback,
    get_fallback_strategy,
    handle_nlsq_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_popt(n: int = 3) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def _make_pcov(n: int = 3) -> np.ndarray:
    return np.eye(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Fallback order
# ---------------------------------------------------------------------------


def test_fallback_order() -> None:
    """_FALLBACK_ORDER must be STREAMING → LARGE → STANDARD (descending)."""
    assert _FALLBACK_ORDER == [
        OptimizationStrategy.STREAMING,
        OptimizationStrategy.LARGE,
        OptimizationStrategy.STANDARD,
    ]


# ---------------------------------------------------------------------------
# 2. handle_nlsq_result — (popt, pcov) tuple
# ---------------------------------------------------------------------------


def test_handle_nlsq_result_tuple() -> None:
    """(popt, pcov) 2-tuple returns (popt, pcov, {})."""
    popt = _make_popt()
    pcov = _make_pcov()
    got_popt, got_pcov, got_info = handle_nlsq_result((popt, pcov))
    np.testing.assert_array_equal(got_popt, popt)
    np.testing.assert_array_equal(got_pcov, pcov)
    assert got_info == {}


# ---------------------------------------------------------------------------
# 3. handle_nlsq_result — (popt, pcov, info) triple
# ---------------------------------------------------------------------------


def test_handle_nlsq_result_triple() -> None:
    """(popt, pcov, info_dict) 3-tuple returns (popt, pcov, info)."""
    popt = _make_popt()
    pcov = _make_pcov()
    info = {"nfev": 42, "message": "converged"}
    got_popt, got_pcov, got_info = handle_nlsq_result((popt, pcov, info))
    np.testing.assert_array_equal(got_popt, popt)
    np.testing.assert_array_equal(got_pcov, pcov)
    assert got_info["nfev"] == 42
    assert got_info["message"] == "converged"


def test_handle_nlsq_result_triple_non_dict_info() -> None:
    """(popt, pcov, non_dict_info) wraps info under 'raw_info'."""
    popt = _make_popt()
    pcov = _make_pcov()
    info_obj = SimpleNamespace(nfev=5)
    got_popt, got_pcov, got_info = handle_nlsq_result((popt, pcov, info_obj))
    np.testing.assert_array_equal(got_popt, popt)
    assert "raw_info" in got_info


# ---------------------------------------------------------------------------
# 4. handle_nlsq_result — streaming dict
# ---------------------------------------------------------------------------


def test_handle_nlsq_result_dict() -> None:
    """Dict with 'x' key (StreamingOptimizer output) is normalized."""
    popt = _make_popt()
    pcov = _make_pcov()
    result_dict: dict[str, Any] = {
        "x": popt,
        "pcov": pcov,
        "success": True,
        "message": "ok",
        "streaming_diagnostics": {"epochs": 3},
    }
    got_popt, got_pcov, got_info = handle_nlsq_result(result_dict)
    np.testing.assert_array_equal(got_popt, popt)
    np.testing.assert_array_equal(got_pcov, pcov)
    assert got_info.get("success") is True
    assert "streaming_diagnostics" in got_info


def test_handle_nlsq_result_dict_popt_key() -> None:
    """Dict with 'popt' key (alternative naming) is normalized."""
    popt = _make_popt()
    result_dict: dict[str, Any] = {"popt": popt}
    got_popt, got_pcov, got_info = handle_nlsq_result(result_dict)
    np.testing.assert_array_equal(got_popt, popt)
    assert got_pcov is None


def test_handle_nlsq_result_dict_missing_keys_raises() -> None:
    """Dict with neither 'x' nor 'popt' raises TypeError."""
    with pytest.raises(TypeError, match="neither 'x' nor 'popt'"):
        handle_nlsq_result({"foo": 1, "bar": 2})


# ---------------------------------------------------------------------------
# 5. handle_nlsq_result — object with .x / .pcov
# ---------------------------------------------------------------------------


def test_handle_nlsq_result_object() -> None:
    """Object with .x and .pcov attributes is normalized."""
    popt = _make_popt()
    pcov = _make_pcov()
    obj = SimpleNamespace(x=popt, pcov=pcov, message="done", nfev=10)
    got_popt, got_pcov, got_info = handle_nlsq_result(obj)
    np.testing.assert_array_equal(got_popt, popt)
    np.testing.assert_array_equal(got_pcov, pcov)
    assert got_info.get("message") == "done"
    assert got_info.get("nfev") == 10


def test_handle_nlsq_result_object_popt_attr() -> None:
    """Object with .popt attribute (no .x) is normalized."""
    popt = _make_popt()
    obj = SimpleNamespace(popt=popt)
    got_popt, got_pcov, got_info = handle_nlsq_result(obj)
    np.testing.assert_array_equal(got_popt, popt)
    assert got_pcov is None


# ---------------------------------------------------------------------------
# 6. handle_nlsq_result — bad type
# ---------------------------------------------------------------------------


def test_handle_nlsq_result_bad_type() -> None:
    """Unrecognized result type raises TypeError."""
    with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
        handle_nlsq_result(12345)  # type: ignore[arg-type]


def test_handle_nlsq_result_bad_tuple_length() -> None:
    """Tuple with length != 2 or 3 raises TypeError."""
    with pytest.raises(TypeError, match="Unexpected tuple length"):
        handle_nlsq_result((np.ones(3),))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. get_fallback_strategy — STREAMING → LARGE
# ---------------------------------------------------------------------------


def test_get_fallback_strategy_streaming() -> None:
    """STREAMING falls back to LARGE."""
    nxt = get_fallback_strategy(OptimizationStrategy.STREAMING)
    assert nxt == OptimizationStrategy.LARGE


def test_get_fallback_strategy_large() -> None:
    """LARGE falls back to STANDARD."""
    nxt = get_fallback_strategy(OptimizationStrategy.LARGE)
    assert nxt == OptimizationStrategy.STANDARD


# ---------------------------------------------------------------------------
# 8. get_fallback_strategy — STANDARD → None
# ---------------------------------------------------------------------------


def test_get_fallback_strategy_standard() -> None:
    """STANDARD has no fallback (end of chain)."""
    nxt = get_fallback_strategy(OptimizationStrategy.STANDARD)
    assert nxt is None


# ---------------------------------------------------------------------------
# 9. Memory error skip
# ---------------------------------------------------------------------------


def test_memory_error_skip() -> None:
    """Memory error from LARGE skips directly to STREAMING."""
    err = MemoryError("out of memory")
    nxt = get_fallback_strategy(OptimizationStrategy.LARGE, error=err)
    assert nxt == OptimizationStrategy.STREAMING


def test_memory_error_skip_from_standard() -> None:
    """Memory error from STANDARD (already last) → None."""
    err = MemoryError("oom")
    nxt = get_fallback_strategy(OptimizationStrategy.STANDARD, error=err)
    assert nxt is None


def test_memory_error_keyword_detection() -> None:
    """RuntimeError with 'out of memory' text is detected as memory error."""
    err = RuntimeError("CUDA out of memory allocate 1GB")
    assert _is_memory_error(err) is True


def test_non_memory_error_normal_fallback() -> None:
    """Non-memory error from STREAMING falls back to LARGE (normal chain)."""
    err = ValueError("convergence failure")
    nxt = get_fallback_strategy(OptimizationStrategy.STREAMING, error=err)
    assert nxt == OptimizationStrategy.LARGE


# ---------------------------------------------------------------------------
# 10. CHUNKED not in enum
# ---------------------------------------------------------------------------


def test_no_chunked_strategy() -> None:
    """CHUNKED must not appear in OptimizationStrategy enum."""
    names = {m.name for m in OptimizationStrategy}
    assert "CHUNKED" not in names


def test_enum_values_match_nlsq_strategy() -> None:
    """String values match NLSQStrategy enum values."""
    assert OptimizationStrategy.STANDARD.value == "standard"
    assert OptimizationStrategy.LARGE.value == "large"
    assert OptimizationStrategy.STREAMING.value == "streaming"


# ---------------------------------------------------------------------------
# 11. execute_optimization_with_fallback — integration smoke tests
# ---------------------------------------------------------------------------


def _make_mock_nlsq_result() -> NLSQResult_type:  # type: ignore[valid-type]
    """Return a minimal mock NLSQResult."""
    from heterodyne.optimization.nlsq.results import NLSQResult

    return NLSQResult(
        parameters=_make_popt(),
        parameter_names=["a", "b", "c"],
        success=True,
        message="ok",
    )


def test_execute_uses_select_nlsq_strategy(tmp_path: Any) -> None:
    """execute_optimization_with_fallback uses select_nlsq_strategy for auto-selection."""
    from heterodyne.optimization.nlsq.memory import NLSQStrategy, StrategyDecision

    mock_model = MagicMock()
    mock_model.parameter_manager.n_varying = 3
    mock_model.parameter_manager.get_initial_params.return_value = np.ones(3)
    mock_model.parameter_manager.get_parameter_names.return_value = ["a", "b", "c"]

    c2_data = np.random.default_rng(0).random((10, 10))

    decision = StrategyDecision(
        strategy=NLSQStrategy.STANDARD,
        threshold_gb=16.0,
        peak_memory_gb=0.001,
        reason="fits in memory",
    )

    mock_result = _make_mock_nlsq_result()

    with (
        patch(
            "heterodyne.optimization.nlsq.fallback_chain.select_nlsq_strategy",
            return_value=decision,
        ) as mock_select,
        patch(
            "heterodyne.optimization.nlsq.fallback_chain.build_result_from_nlsq",
            return_value=mock_result,
        ),
        patch(
            "heterodyne.optimization.nlsq.fallback_chain._run_strategy",
        ) as mock_run,
    ):
        mock_run.return_value = (np.ones(3), np.eye(3), {})

        result = execute_optimization_with_fallback(
            mock_model,
            c2_data,
            phi_angle=0.0,
            config=MagicMock(),
        )

    mock_select.assert_called_once()
    assert result is mock_result


def test_execute_explicit_start_strategy_skips_select() -> None:
    """Providing start_strategy skips memory-based auto-selection."""
    mock_model = MagicMock()
    mock_model.parameter_manager.n_varying = 3
    mock_model.parameter_manager.get_initial_params.return_value = np.ones(3)
    mock_model.parameter_manager.get_parameter_names.return_value = ["a", "b", "c"]

    c2_data = np.zeros((5, 5))
    mock_result = _make_mock_nlsq_result()

    with (
        patch(
            "heterodyne.optimization.nlsq.fallback_chain.select_nlsq_strategy",
        ) as mock_select,
        patch(
            "heterodyne.optimization.nlsq.fallback_chain.build_result_from_nlsq",
            return_value=mock_result,
        ),
        patch(
            "heterodyne.optimization.nlsq.fallback_chain._run_strategy",
            return_value=(np.ones(3), np.eye(3), {}),
        ),
    ):
        execute_optimization_with_fallback(
            mock_model,
            c2_data,
            phi_angle=0.0,
            config=MagicMock(),
            start_strategy=OptimizationStrategy.STANDARD,
        )

    mock_select.assert_not_called()


def test_execute_falls_back_on_failure() -> None:
    """When _run_strategy raises on STREAMING, chain falls back to LARGE."""
    mock_model = MagicMock()
    mock_model.parameter_manager.n_varying = 3
    mock_model.parameter_manager.get_initial_params.return_value = np.ones(3)
    mock_model.parameter_manager.get_parameter_names.return_value = ["a", "b", "c"]

    c2_data = np.zeros((5, 5))
    mock_result = _make_mock_nlsq_result()

    call_count = {"n": 0}

    def side_effect(
        strategy: OptimizationStrategy, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        call_count["n"] += 1
        if strategy == OptimizationStrategy.STREAMING:
            raise RuntimeError("streaming unavailable")
        return (np.ones(3), np.eye(3), {})

    with (
        patch(
            "heterodyne.optimization.nlsq.fallback_chain.build_result_from_nlsq",
            return_value=mock_result,
        ),
        patch(
            "heterodyne.optimization.nlsq.fallback_chain._run_strategy",
            side_effect=side_effect,
        ),
    ):
        result = execute_optimization_with_fallback(
            mock_model,
            c2_data,
            phi_angle=0.0,
            config=MagicMock(),
            start_strategy=OptimizationStrategy.STREAMING,
        )

    assert call_count["n"] == 2  # STREAMING failed, LARGE succeeded
    assert result is mock_result


def test_execute_all_strategies_fail_raises() -> None:
    """RuntimeError raised when all strategies in fallback chain fail."""
    mock_model = MagicMock()
    mock_model.parameter_manager.n_varying = 3
    mock_model.parameter_manager.get_initial_params.return_value = np.ones(3)
    mock_model.parameter_manager.get_parameter_names.return_value = ["a", "b", "c"]

    c2_data = np.zeros((5, 5))

    with (
        patch(
            "heterodyne.optimization.nlsq.fallback_chain._run_strategy",
            side_effect=RuntimeError("always fails"),
        ),
    ):
        with pytest.raises(RuntimeError, match="All optimization strategies failed"):
            execute_optimization_with_fallback(
                mock_model,
                c2_data,
                phi_angle=0.0,
                config=MagicMock(),
                start_strategy=OptimizationStrategy.STREAMING,
            )
