"""Tests for core.py dual-adapter routing.

Verifies:
1. _fit_local uses NLSQAdapter as primary
2. _fit_local falls back to NLSQWrapper when adapter fails
3. No scipy.optimize.least_squares import in core.py
4. _fit_joint_multi_phi uses NLSQAdapter (not scipy)
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_success_result(param_names: list[str]) -> NLSQResult:
    """Return a minimal successful NLSQResult for mocking."""
    return NLSQResult(
        parameters=np.ones(len(param_names), dtype=np.float64),
        parameter_names=param_names,
        success=True,
        message="converged",
        final_cost=1e-6,
        n_function_evals=10,
        metadata={},
    )


def _make_failed_result(param_names: list[str]) -> NLSQResult:
    """Return a minimal failed NLSQResult for mocking."""
    return NLSQResult(
        parameters=np.zeros(len(param_names), dtype=np.float64),
        parameter_names=param_names,
        success=False,
        message="Adapter returned success=False: test failure",
        metadata={},
    )


def _make_mock_model(n_params: int = 3) -> MagicMock:
    """Build a minimal mock HeterodyneModel for _fit_local tests."""
    param_names = [f"p{i}" for i in range(n_params)]
    model = MagicMock()

    pm = MagicMock()
    pm.varying_names = param_names
    pm.n_varying = n_params
    pm.varying_indices = list(range(n_params))
    pm.get_initial_values.return_value = np.ones(n_params, dtype=np.float64)
    pm.get_bounds.return_value = (
        np.zeros(n_params, dtype=np.float64),
        np.full(n_params, 10.0, dtype=np.float64),
    )
    pm.get_full_values.return_value = np.ones(n_params, dtype=np.float64)
    pm.expand_varying_to_full.side_effect = lambda x: np.asarray(x, dtype=np.float64)

    model.param_manager = pm
    model.t = np.linspace(0, 1, 5)
    model.q = np.array([0.1])
    model.dt = 0.01
    model.scaling = MagicMock()
    model.scaling.get_for_angle.return_value = (0.5, 1.0)

    return model


# ---------------------------------------------------------------------------
# Test 1: _fit_local tries NLSQAdapter first
# ---------------------------------------------------------------------------


class TestFitLocalAdapterPrimary:
    """NLSQAdapter is tried before NLSQWrapper in _fit_local."""

    @pytest.mark.unit
    def test_fit_local_uses_nlsq_adapter(self) -> None:
        """When use_nlsq_library=True, NLSQAdapter.fit_jax is called first."""
        from heterodyne.optimization.nlsq import core as core_mod

        model = _make_mock_model(n_params=2)
        c2_data = np.ones((3, 3), dtype=np.float64)
        config = NLSQConfig()
        param_names = model.param_manager.varying_names

        success_result = _make_success_result(param_names)

        with (
            patch.object(core_mod, "HAS_ADAPTERS", True),
            patch.object(core_mod, "HAS_WRAPPER", True),
            patch.object(core_mod, "HAS_MEMORY", False),
            patch("heterodyne.optimization.nlsq.core.NLSQAdapter") as MockAdapter,
            patch("heterodyne.optimization.nlsq.core.NLSQWrapper") as MockWrapper,
            patch("heterodyne.optimization.nlsq.core.compute_c2_heterodyne") as mock_c2,
        ):
            adapter_instance = MagicMock()
            adapter_instance.fit_jax.return_value = success_result
            MockAdapter.return_value = adapter_instance

            mock_c2.return_value = np.zeros((3, 3))

            result = core_mod._fit_local(
                model=model,
                c2_data=c2_data,
                phi_angle=0.0,
                config=config,
                weights=None,
                use_nlsq_library=True,
            )

        # Adapter was constructed and called
        MockAdapter.assert_called_once_with(parameter_names=param_names)
        adapter_instance.fit_jax.assert_called_once()

        # Wrapper was never called since adapter succeeded
        MockWrapper.assert_not_called()

        assert result.success is True


# ---------------------------------------------------------------------------
# Test 2: _fit_local falls back to NLSQWrapper on adapter failure
# ---------------------------------------------------------------------------


class TestFitLocalFallback:
    """NLSQWrapper is used when NLSQAdapter raises or returns success=False."""

    @pytest.mark.unit
    def test_fit_local_fallback_to_wrapper_on_runtime_error(self) -> None:
        """If NLSQAdapter.fit_jax raises RuntimeError, NLSQWrapper is used."""
        from heterodyne.optimization.nlsq import core as core_mod

        model = _make_mock_model(n_params=2)
        c2_data = np.ones((3, 3), dtype=np.float64)
        config = NLSQConfig()
        param_names = model.param_manager.varying_names

        wrapper_result = _make_success_result(param_names)
        wrapper_result.metadata["fallback_from"] = "adapter_error"

        with (
            patch.object(core_mod, "HAS_ADAPTERS", True),
            patch.object(core_mod, "HAS_WRAPPER", True),
            patch.object(core_mod, "HAS_MEMORY", False),
            patch("heterodyne.optimization.nlsq.core.NLSQAdapter") as MockAdapter,
            patch("heterodyne.optimization.nlsq.core.NLSQWrapper") as MockWrapper,
            patch("heterodyne.optimization.nlsq.core.compute_c2_heterodyne") as mock_c2,
        ):
            adapter_instance = MagicMock()
            adapter_instance.fit_jax.side_effect = RuntimeError("adapter exploded")
            MockAdapter.return_value = adapter_instance

            wrapper_instance = MagicMock()
            wrapper_instance.fit.return_value = wrapper_result
            MockWrapper.return_value = wrapper_instance

            mock_c2.return_value = np.zeros((3, 3))

            result = core_mod._fit_local(
                model=model,
                c2_data=c2_data,
                phi_angle=0.0,
                config=config,
                weights=None,
                use_nlsq_library=True,
            )

        # Adapter was tried
        adapter_instance.fit_jax.assert_called_once()
        # Wrapper was used as fallback
        MockWrapper.assert_called_once_with(parameter_names=param_names)
        wrapper_instance.fit.assert_called_once()

        assert result.success is True
        assert result.metadata.get("fallback_occurred") is True

    @pytest.mark.unit
    def test_fit_local_fallback_to_wrapper_on_success_false(self) -> None:
        """If NLSQAdapter returns success=False, NLSQWrapper is used as fallback."""
        from heterodyne.optimization.nlsq import core as core_mod

        model = _make_mock_model(n_params=2)
        c2_data = np.ones((3, 3), dtype=np.float64)
        config = NLSQConfig()
        param_names = model.param_manager.varying_names

        failed_adapter_result = _make_failed_result(param_names)
        wrapper_result = _make_success_result(param_names)

        with (
            patch.object(core_mod, "HAS_ADAPTERS", True),
            patch.object(core_mod, "HAS_WRAPPER", True),
            patch.object(core_mod, "HAS_MEMORY", False),
            patch("heterodyne.optimization.nlsq.core.NLSQAdapter") as MockAdapter,
            patch("heterodyne.optimization.nlsq.core.NLSQWrapper") as MockWrapper,
            patch("heterodyne.optimization.nlsq.core.compute_c2_heterodyne") as mock_c2,
        ):
            adapter_instance = MagicMock()
            adapter_instance.fit_jax.return_value = failed_adapter_result
            MockAdapter.return_value = adapter_instance

            wrapper_instance = MagicMock()
            wrapper_instance.fit.return_value = wrapper_result
            MockWrapper.return_value = wrapper_instance

            mock_c2.return_value = np.zeros((3, 3))

            result = core_mod._fit_local(
                model=model,
                c2_data=c2_data,
                phi_angle=0.0,
                config=config,
                weights=None,
                use_nlsq_library=True,
            )

        # The adapter raised a RuntimeError from success=False, triggering fallback
        wrapper_instance.fit.assert_called_once()
        assert result.success is True
        assert result.metadata.get("fallback_occurred") is True


# ---------------------------------------------------------------------------
# Test 3: No scipy.optimize.least_squares import in core.py
# ---------------------------------------------------------------------------


class TestNoScipyLeastSquares:
    """core.py must not import or call scipy.optimize.least_squares."""

    @pytest.mark.unit
    def test_no_scipy_least_squares_import(self) -> None:
        """core.py must not contain a scipy.optimize import."""
        core_path = (
            Path(__file__).parents[4]
            / "heterodyne"
            / "optimization"
            / "nlsq"
            / "core.py"
        )
        source = core_path.read_text(encoding="utf-8")

        # Direct text check — catches any form of import
        assert "from scipy.optimize import least_squares" not in source, (
            "core.py still imports least_squares from scipy.optimize"
        )
        assert "scipy.optimize.least_squares" not in source, (
            "core.py still references scipy.optimize.least_squares"
        )

    @pytest.mark.unit
    def test_no_scipy_nlsq_adapter_reference(self) -> None:
        """core.py must not reference ScipyNLSQAdapter."""
        core_path = (
            Path(__file__).parents[4]
            / "heterodyne"
            / "optimization"
            / "nlsq"
            / "core.py"
        )
        source = core_path.read_text(encoding="utf-8")

        assert "ScipyNLSQAdapter" not in source, (
            "core.py still references ScipyNLSQAdapter"
        )

    @pytest.mark.unit
    def test_no_scipy_import_via_ast(self) -> None:
        """AST parse of core.py must not have scipy.optimize imports."""
        core_path = (
            Path(__file__).parents[4]
            / "heterodyne"
            / "optimization"
            / "nlsq"
            / "core.py"
        )
        source = core_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "scipy.optimize" in module:
                    imported_names = [alias.name for alias in node.names]
                    assert "least_squares" not in imported_names, (
                        f"AST found 'from {module} import least_squares'"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "scipy" not in (alias.name or ""), (
                        f"AST found bare 'import {alias.name}'"
                    )


# ---------------------------------------------------------------------------
# Test 4: _fit_joint_multi_phi uses NLSQAdapter (not scipy)
# ---------------------------------------------------------------------------


class TestFitJointMultiPhi:
    """_fit_joint_multi_phi must use NLSQAdapter, not scipy."""

    @pytest.mark.unit
    def test_fit_joint_multi_phi_uses_nlsq_adapter(self) -> None:
        """NLSQAdapter.fit is called in _fit_joint_multi_phi, not least_squares."""
        from heterodyne.optimization.nlsq import core as core_mod

        n_params = 2
        n_phi = 2
        model = _make_mock_model(n_params=n_params)
        _param_names = model.param_manager.varying_names

        # c2_data shape (n_phi, N, N)
        c2_data = np.ones((n_phi, 4, 4), dtype=np.float64)
        phi_angles = np.array([0.0, 45.0])
        config = NLSQConfig()

        # Combine physics + fourier params
        n_fourier_coeffs = 4
        joint_param_count = n_params + n_fourier_coeffs
        joint_params = np.ones(joint_param_count, dtype=np.float64)
        joint_result = _make_success_result([f"p{i}" for i in range(joint_param_count)])
        joint_result = NLSQResult(
            parameters=joint_params,
            parameter_names=[f"p{i}" for i in range(joint_param_count)],
            success=True,
            message="converged",
            final_cost=1e-6,
            n_function_evals=10,
            metadata={},
        )

        # Mock fourier reparameterizer
        fourier = MagicMock()
        fourier.n_coeffs = n_fourier_coeffs
        fourier.get_initial_coefficients.return_value = np.zeros(
            n_fourier_coeffs, dtype=np.float64
        )
        fourier.get_bounds.return_value = (
            np.zeros(n_fourier_coeffs, dtype=np.float64),
            np.ones(n_fourier_coeffs, dtype=np.float64),
        )
        fourier.fourier_to_per_angle.return_value = (
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
        )
        fourier.config.mode = "fourier"
        fourier.order = 2
        fourier.get_diagnostics.return_value = {"reduction_ratio": 0.5}

        model.scaling.contrast = np.array([0.5, 0.5])
        model.scaling.offset = np.array([1.0, 1.0])

        with (
            patch.object(core_mod, "HAS_ADAPTERS", True),
            patch.object(core_mod, "HAS_WRAPPER", True),
            patch("heterodyne.optimization.nlsq.core.NLSQAdapter") as MockAdapter,
            patch("heterodyne.optimization.nlsq.core.NLSQWrapper") as MockWrapper,
            patch("heterodyne.optimization.nlsq.core.compute_c2_heterodyne") as mock_c2,
            patch("heterodyne.optimization.nlsq.core.compute_residuals") as mock_res,
        ):
            adapter_instance = MagicMock()
            adapter_instance.fit.return_value = joint_result
            MockAdapter.return_value = adapter_instance

            mock_c2.return_value = np.zeros((4, 4))
            mock_res.return_value = np.zeros(16)

            results = core_mod._fit_joint_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=None,
                fourier=fourier,
            )

        # NLSQAdapter.fit was called (not scipy)
        MockAdapter.assert_called_once()
        adapter_instance.fit.assert_called_once()

        # NLSQWrapper was not needed
        MockWrapper.assert_not_called()

        assert len(results) == n_phi

    @pytest.mark.unit
    def test_fit_joint_multi_phi_fallback_to_wrapper(self) -> None:
        """NLSQWrapper is used when NLSQAdapter fails in _fit_joint_multi_phi."""
        from heterodyne.optimization.nlsq import core as core_mod

        n_params = 2
        n_phi = 2
        model = _make_mock_model(n_params=n_params)

        c2_data = np.ones((n_phi, 4, 4), dtype=np.float64)
        phi_angles = np.array([0.0, 45.0])
        config = NLSQConfig()

        n_fourier_coeffs = 4
        joint_param_count = n_params + n_fourier_coeffs
        joint_params = np.ones(joint_param_count, dtype=np.float64)
        joint_result = NLSQResult(
            parameters=joint_params,
            parameter_names=[f"p{i}" for i in range(joint_param_count)],
            success=True,
            message="wrapper converged",
            final_cost=1e-5,
            n_function_evals=20,
            metadata={},
        )

        fourier = MagicMock()
        fourier.n_coeffs = n_fourier_coeffs
        fourier.get_initial_coefficients.return_value = np.zeros(
            n_fourier_coeffs, dtype=np.float64
        )
        fourier.get_bounds.return_value = (
            np.zeros(n_fourier_coeffs, dtype=np.float64),
            np.ones(n_fourier_coeffs, dtype=np.float64),
        )
        fourier.fourier_to_per_angle.return_value = (
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
        )
        fourier.config.mode = "fourier"
        fourier.order = 2
        fourier.get_diagnostics.return_value = {"reduction_ratio": 0.5}

        model.scaling.contrast = np.array([0.5, 0.5])
        model.scaling.offset = np.array([1.0, 1.0])

        with (
            patch.object(core_mod, "HAS_ADAPTERS", True),
            patch.object(core_mod, "HAS_WRAPPER", True),
            patch("heterodyne.optimization.nlsq.core.NLSQAdapter") as MockAdapter,
            patch("heterodyne.optimization.nlsq.core.NLSQWrapper") as MockWrapper,
            patch("heterodyne.optimization.nlsq.core.compute_c2_heterodyne") as mock_c2,
            patch("heterodyne.optimization.nlsq.core.compute_residuals") as mock_res,
        ):
            adapter_instance = MagicMock()
            adapter_instance.fit.side_effect = RuntimeError("nlsq backend error")
            MockAdapter.return_value = adapter_instance

            wrapper_instance = MagicMock()
            wrapper_instance.fit.return_value = joint_result
            MockWrapper.return_value = wrapper_instance

            mock_c2.return_value = np.zeros((4, 4))
            mock_res.return_value = np.zeros(16)

            results = core_mod._fit_joint_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=None,
                fourier=fourier,
            )

        # Adapter was attempted
        adapter_instance.fit.assert_called_once()
        # Wrapper picked up the fallback
        wrapper_instance.fit.assert_called_once()

        assert len(results) == n_phi


# ---------------------------------------------------------------------------
# Test 5: Module import — NLSQAdapter and NLSQWrapper come from adapter.py
# ---------------------------------------------------------------------------


class TestCoreImportStructure:
    """core.py must import from adapter, not from wrapper or scipy."""

    @pytest.mark.unit
    def test_has_adapters_is_true_when_adapter_available(self) -> None:
        """HAS_ADAPTERS is True when adapter module is importable."""
        from heterodyne.optimization.nlsq import core as core_mod

        # Both flags must be set together from adapter.py import
        assert core_mod.HAS_ADAPTERS is True
        assert core_mod.HAS_WRAPPER is True

    @pytest.mark.unit
    def test_nlsq_adapter_class_is_from_adapter_module(self) -> None:
        """NLSQAdapter in core namespace must be the class from adapter.py."""
        from heterodyne.optimization.nlsq import core as core_mod
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        assert core_mod.NLSQAdapter is NLSQAdapter

    @pytest.mark.unit
    def test_nlsq_wrapper_class_is_from_adapter_module(self) -> None:
        """NLSQWrapper in core namespace must be the class from adapter.py."""
        from heterodyne.optimization.nlsq import core as core_mod
        from heterodyne.optimization.nlsq.adapter import NLSQWrapper

        assert core_mod.NLSQWrapper is NLSQWrapper
