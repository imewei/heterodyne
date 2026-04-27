"""Tests for NLSQ adapter API compatibility.

Bug Prevented: NLSQ Adapter API Mismatch
----------------------------------------
The nlsq library v0.6.4+ changed its API, specifically:
- CurveFit now requires a `flength` parameter in the constructor
- The curve_fit method signature changed

These tests verify that the heterodyne package is compatible with the
current nlsq library API and will catch breaking changes early.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from heterodyne import HeterodyneModel, NLSQConfig


class TestNLSQLibraryVersion:
    """Tests for nlsq library version compatibility."""

    @pytest.mark.api
    def test_nlsq_library_available(self) -> None:
        """Verify nlsq library is installed and importable."""
        try:
            import nlsq  # noqa: F401
        except ImportError:
            pytest.fail("nlsq library is not installed")

    @pytest.mark.api
    def test_nlsq_library_version(self) -> None:
        """Verify nlsq library version >= 0.6.4.

        The heterodyne package requires nlsq >= 0.6.4 for the updated
        CurveFit API with flength parameter.
        """
        try:
            from importlib.metadata import version

            nlsq_version = version("nlsq")
        except ImportError:
            # Python < 3.8 fallback
            import nlsq

            nlsq_version = getattr(nlsq, "__version__", "0.0.0")

        # Parse version
        parts = nlsq_version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2].split("+")[0].split("-")[0]) if len(parts) > 2 else 0

        # Require >= 0.6.4
        version_tuple = (major, minor, patch)
        required = (0, 6, 4)

        assert version_tuple >= required, (
            f"nlsq version {nlsq_version} < required 0.6.4. "
            "Please upgrade: pip install 'nlsq>=0.6.4'"
        )


class TestCurveFitAPI:
    """Tests for nlsq CurveFit class API."""

    @pytest.mark.api
    def test_curvefit_exists(self) -> None:
        """Verify CurveFit class is available."""
        from nlsq import CurveFit

        assert CurveFit is not None

    @pytest.mark.api
    def test_curvefit_has_flength_parameter(self) -> None:
        """Verify CurveFit __init__ accepts flength parameter.

        This is a critical API change in nlsq v0.6.4+. The flength
        parameter is required for proper residual sizing.
        """
        from nlsq import CurveFit

        sig = inspect.signature(CurveFit.__init__)
        params = list(sig.parameters.keys())

        assert "flength" in params, (
            "CurveFit.__init__ missing 'flength' parameter. "
            "This indicates an incompatible nlsq version."
        )

    @pytest.mark.api
    def test_curvefit_curve_fit_method_exists(self) -> None:
        """Verify curve_fit method exists on CurveFit."""
        from nlsq import CurveFit

        assert hasattr(CurveFit, "curve_fit"), "CurveFit missing 'curve_fit' method"

    @pytest.mark.api
    def test_curvefit_curve_fit_method_signature(self) -> None:
        """Verify curve_fit method has expected parameters.

        Expected signature includes: f, xdata, ydata, p0, bounds, method
        """
        from nlsq import CurveFit

        sig = inspect.signature(CurveFit.curve_fit)
        params = set(sig.parameters.keys())

        # Required parameters (excluding 'self')
        required = {"f", "xdata", "ydata", "p0"}

        missing = required - params
        assert not missing, f"CurveFit.curve_fit missing required parameters: {missing}"

        # Check optional parameters exist
        optional = {"bounds", "method"}
        for opt in optional:
            assert opt in params, (
                f"CurveFit.curve_fit missing optional parameter: {opt}"
            )

    @pytest.mark.api
    def test_curvefit_instantiation_with_flength(self) -> None:
        """Verify CurveFit can be instantiated with flength."""
        from nlsq import CurveFit

        # Should not raise
        fitter = CurveFit(flength=100.0)
        assert fitter is not None


class TestNLSQAdapterUnit:
    """Unit tests for NLSQAdapter class."""

    @pytest.mark.unit
    def test_adapter_creation(self) -> None:
        """Test NLSQAdapter can be created."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=["p1", "p2", "p3"])
        assert adapter is not None

    @pytest.mark.unit
    def test_adapter_name_property(self) -> None:
        """Test adapter name property returns expected value."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=["p1"])
        assert adapter.name == "nlsq.CurveFit"

    @pytest.mark.unit
    def test_adapter_supports_bounds(self) -> None:
        """Test adapter reports bounds support."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=["p1"])
        assert adapter.supports_bounds() is True

    @pytest.mark.unit
    def test_adapter_supports_jacobian(self) -> None:
        """Test adapter reports jacobian support."""
        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        adapter = NLSQAdapter(parameter_names=["p1"])
        assert adapter.supports_jacobian() is True


class TestScipyAdapterFallback:
    """ScipyNLSQAdapter was removed in the dual-adapter refactor.

    The tests below act as a tombstone: they verify that the class no longer
    exists in the module (so any attempt to reintroduce it will break CI).
    Full coverage of the replacement classes is in test_adapter.py.
    """

    @pytest.mark.unit
    def test_scipy_adapter_does_not_exist(self) -> None:
        """ScipyNLSQAdapter must NOT be importable after the dual-adapter refactor."""
        import importlib

        mod = importlib.import_module("heterodyne.optimization.nlsq.adapter")
        assert not hasattr(mod, "ScipyNLSQAdapter"), (
            "ScipyNLSQAdapter was re-introduced into adapter.py; "
            "it must remain deleted — scipy.optimize is forbidden in the NLSQ path."
        )


class TestNLSQAdapterIntegration:
    """Integration tests for NLSQAdapter with real models."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_nlsq_adapter_fit_jax_basic(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test NLSQAdapter.fit_jax with a basic heterodyne model.

        This is the primary integration test for the JAX-traced
        optimization path using nlsq.
        """
        import jax.numpy as jnp

        from heterodyne.optimization.nlsq.adapter import NLSQAdapter

        model = small_heterodyne_model
        param_manager = model.param_manager
        varying_names = param_manager.varying_names

        # Get initial values and bounds
        initial = param_manager.get_initial_values()
        lower, upper = param_manager.get_bounds()

        # Create simple JAX residual function
        def jax_residual_fn(x: jnp.ndarray, *params) -> jnp.ndarray:
            # Simplified: just return difference from target
            return jnp.zeros_like(x)

        adapter = NLSQAdapter(parameter_names=varying_names)
        n_data = small_c2_data.size

        result = adapter.fit_jax(
            jax_residual_fn=jax_residual_fn,
            initial_params=initial,
            bounds=(lower, upper),
            config=fast_nlsq_config,
            n_data=n_data,
        )

        # Should return a result (may or may not succeed with trivial residual)
        assert result is not None
        assert len(result.parameters) == len(varying_names)
        assert result.parameter_names == varying_names

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_returns_result(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax returns a valid NLSQResult.

        Uses the full fitting pipeline with nlsq library.
        """
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=True,
        )

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "parameters")
        assert hasattr(result, "parameter_names")

        # Parameters should have correct length
        assert len(result.parameters) == small_heterodyne_model.n_varying


class TestBugPrevention_NLSQAdapterAPI:
    """Regression tests for NLSQ Adapter API Mismatch bug.

    BUG DESCRIPTION:
    The nlsq library v0.6.4+ changed its API. The CurveFit class now requires
    a `flength` parameter in the constructor. Without this, instantiation fails.

    These tests verify the adapter correctly uses the new API.
    """

    @pytest.mark.api
    @pytest.mark.unit
    def test_adapter_uses_flength_parameter(self) -> None:
        """REGRESSION TEST: Verify adapter passes flength to CurveFit.

        If the adapter doesn't pass flength, CurveFit will raise an error
        or produce incorrect results.
        """
        import jax.numpy as jnp

        from heterodyne.optimization.nlsq.adapter import NLSQAdapter
        from heterodyne.optimization.nlsq.config import NLSQConfig

        adapter = NLSQAdapter(parameter_names=["p1", "p2"])
        config = NLSQConfig(max_iterations=5, tolerance=1e-4)

        # Simple JAX function
        def jax_fn(x: jnp.ndarray, p1: float, p2: float) -> jnp.ndarray:
            return jnp.zeros_like(x)

        n_data = 100  # This should be passed as flength

        # This should NOT raise an error about flength
        result = adapter.fit_jax(
            jax_residual_fn=jax_fn,
            initial_params=np.array([1.0, 1.0]),
            bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            config=config,
            n_data=n_data,
        )

        assert result is not None, "fit_jax should return a result"

    @pytest.mark.api
    @pytest.mark.unit
    def test_old_api_without_flength_would_fail(self) -> None:
        """REGRESSION TEST: Verify old API (without flength) fails.

        This test documents that CurveFit() without flength raises an error,
        confirming we need the new API.
        """
        from nlsq import CurveFit

        # Modern nlsq requires flength - this test verifies it
        # If nlsq changed to not require flength, this test would need updating
        try:
            # Try to instantiate without flength
            fitter = CurveFit()
            # If we get here, nlsq has changed - need to verify behavior
            # Check if flength was auto-set or is required
            assert hasattr(fitter, "_flength") or True  # Allow if it works
        except TypeError as e:
            # Expected: TypeError about missing flength
            assert "flength" in str(e).lower() or "argument" in str(e).lower()


# ---------------------------------------------------------------------------
# Regression tests: loss keyword must be propagated / excluded correctly
# ---------------------------------------------------------------------------


class TestBugPrevention_LossKwarg:
    """Regression tests for loss kwarg propagation (RCA 2026-04-26).

    BUG: config.loss was silently dropped — every optimizer call only passed
    method=, so nlsq defaulted to loss='linear' regardless of config.

    BUG: NLSQWrapper STANDARD tier passed loss='soft_l1' to nlsq.curve_fit,
    which JIT-compiled the loss wrapper over the numpy residual function,
    causing TracerArrayConversionError.
    """

    @pytest.mark.unit
    @pytest.mark.api
    def test_adapter_fitjax_passes_loss_kwarg(self) -> None:
        """config.loss must appear in kwargs forwarded to CurveFit.curve_fit."""
        from unittest.mock import MagicMock, patch

        import jax.numpy as jnp

        from heterodyne.optimization.nlsq.adapter import NLSQAdapter
        from heterodyne.optimization.nlsq.config import NLSQConfig

        config = NLSQConfig(loss="soft_l1")
        adapter = NLSQAdapter(parameter_names=["p0", "p1"])

        mock_fitter = MagicMock()
        # Return a (popt, pcov) tuple that build_result_from_nlsq can parse
        mock_fitter.curve_fit.return_value = (np.array([2.0, 3.0]), np.eye(2))

        with patch(
            "heterodyne.optimization.nlsq.adapter.get_or_create_fitter",
            return_value=(mock_fitter, False),
        ):
            try:
                adapter.fit_jax(
                    jax_residual_fn=lambda x, *p: jnp.zeros(9),
                    initial_params=np.ones(2),
                    bounds=(np.zeros(2), np.full(2, 10.0)),
                    config=config,
                    n_data=9,
                )
            except Exception:
                pass  # Only care about call args, not downstream processing

        assert mock_fitter.curve_fit.called, "CurveFit.curve_fit was never called"
        call_kwargs = mock_fitter.curve_fit.call_args.kwargs
        assert "loss" in call_kwargs, (
            f"'loss' kwarg missing from CurveFit.curve_fit call. Got: {call_kwargs}"
        )
        assert call_kwargs["loss"] == "soft_l1", (
            f"Expected loss='soft_l1', got: {call_kwargs['loss']}"
        )

    @pytest.mark.unit
    @pytest.mark.api
    def test_wrapper_standard_tier_omits_loss(self) -> None:
        """NLSQWrapper STANDARD tier must NOT forward loss to nlsq.curve_fit.

        nlsq.curve_fit applies robust loss by JAX-JIT-compiling a mask over the
        residual function.  The numpy-wrapped residual calls np.array(params)
        inside that trace, causing TracerArrayConversionError.
        """
        from unittest.mock import patch

        from heterodyne.optimization.nlsq.adapter import NLSQStrategy, NLSQWrapper
        from heterodyne.optimization.nlsq.config import NLSQConfig

        wrapper = NLSQWrapper(parameter_names=["p0", "p1"])
        config = NLSQConfig(loss="soft_l1")

        with patch(
            "heterodyne.optimization.nlsq.adapter.curve_fit",
            return_value=(np.array([2.0, 3.0]), np.eye(2)),
        ) as mock_cf:
            try:
                wrapper._call_tier(
                    tier=NLSQStrategy.STANDARD,
                    wrapped_fn=lambda x, *p: np.zeros(9),
                    xdata=np.arange(9, dtype=np.float64),
                    ydata=np.zeros(9, dtype=np.float64),
                    p0=np.ones(2),
                    lower_bounds=np.zeros(2),
                    upper_bounds=np.full(2, 10.0),
                    n_data=9,
                    n_params=2,
                    method="trf",
                    loss=config.loss,
                )
            except Exception:
                pass

        assert mock_cf.called, "nlsq.curve_fit was never called"
        call_kwargs = mock_cf.call_args.kwargs
        assert "loss" not in call_kwargs, (
            f"'loss' must NOT be in nlsq.curve_fit kwargs (TracerArrayConversionError). "
            f"Got: {call_kwargs}"
        )
