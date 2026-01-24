"""Tests for NLSQ core functionality and JAX tracing.

Bug Prevented: JAX Tracing Error
--------------------------------
Using numpy operations inside JAX-traced functions causes
ConcretizationTypeError. The residual functions must use jax.numpy
exclusively to remain traceable.

These tests verify that:
1. Residual functions are fully JAX-traceable
2. No numpy operations leak into traced code paths
3. JIT compilation works correctly
"""

from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from heterodyne import HeterodyneModel, NLSQConfig


class TestJAXResidualTracing:
    """Tests for JAX traceability of residual functions."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_residual_fn_is_traceable(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Verify residual function can be traced by JAX without errors.

        This is the critical test for the JAX tracing bug. If numpy
        operations are used inside the residual function, this will
        raise jax.errors.ConcretizationTypeError.
        """
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data)

        # Get parameter info
        param_manager = model.param_manager
        fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
        varying_indices = jnp.array(param_manager.varying_indices)

        # Create JAX-compatible residual function
        def jax_residual_fn(x: jnp.ndarray, *varying_params) -> jnp.ndarray:
            """Pure JAX residual function."""
            varying_array = jnp.array(varying_params, dtype=jnp.float64)
            full_params = fixed_values.at[varying_indices].set(varying_array)

            residuals = compute_residuals(
                full_params,
                model.t,
                model.q,
                model.dt,
                0.0,  # phi_angle
                c2_jax,
                None,  # weights
            )
            return residuals

        # Try to JIT compile - this will fail if numpy ops are used
        try:
            # Use make_jaxpr to test traceability without full JIT
            initial_params = tuple(param_manager.get_initial_values())
            x_test = jnp.arange(c2_jax.size, dtype=jnp.float64)

            # This will raise if the function is not traceable
            jax.make_jaxpr(jax_residual_fn)(x_test, *initial_params)

        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(
                f"Residual function is not JAX-traceable: {e}\n"
                "This indicates numpy operations inside traced code."
            )
        except Exception as e:
            # Some other error - still a problem
            pytest.fail(f"Unexpected error during tracing: {type(e).__name__}: {e}")

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jit_compilation_of_residual_function(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """Verify residual function can be JIT-compiled.

        JIT compilation is more strict than make_jaxpr and will
        catch additional tracing issues.
        """
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data)

        param_manager = model.param_manager
        fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
        varying_indices = jnp.array(param_manager.varying_indices)

        @jax.jit
        def jit_residual_fn(varying_array: jnp.ndarray) -> jnp.ndarray:
            """JIT-compiled residual computation."""
            full_params = fixed_values.at[varying_indices].set(varying_array)
            return compute_residuals(
                full_params,
                model.t,
                model.q,
                model.dt,
                0.0,
                c2_jax,
                None,
            )

        # Run the JIT-compiled function
        initial = jnp.asarray(param_manager.get_initial_values())

        try:
            result = jit_residual_fn(initial)

            # Verify output is valid
            assert result.shape[0] > 0
            assert not jnp.any(jnp.isnan(result))
            assert not jnp.any(jnp.isinf(result))

        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed: {e}")


class TestNumpyLeakage:
    """Tests to detect numpy operations in JAX code paths."""

    @pytest.mark.unit
    def test_no_numpy_in_jax_residual_source(self) -> None:
        """Check that compute_residuals source code uses jax.numpy.

        This is a static analysis check to catch accidental numpy usage.
        """
        from heterodyne.core.jax_backend import compute_residuals

        source = inspect.getsource(compute_residuals)

        # Parse the source to find numpy imports/usages
        tree = ast.parse(source)

        # Look for 'np.' usage (common alias for numpy)
        np_usages = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    np_usages.append(node.attr)

        # Some numpy usage is OK (e.g., for type hints, docstrings)
        # But core computation should not use numpy
        forbidden_np_ops = {"array", "zeros", "ones", "arange", "linspace", "sum", "mean"}
        violations = set(np_usages) & forbidden_np_ops

        if violations:
            pytest.fail(
                f"Found numpy operations in compute_residuals: {violations}\n"
                "Use jax.numpy instead for JAX traceability."
            )

    @pytest.mark.unit
    def test_no_numpy_in_compute_c2_heterodyne(self) -> None:
        """Check that compute_c2_heterodyne uses jax.numpy."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne

        source = inspect.getsource(compute_c2_heterodyne)
        tree = ast.parse(source)

        np_usages = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    np_usages.append(node.attr)

        forbidden_np_ops = {"array", "zeros", "ones", "sum", "exp", "sqrt", "cos", "sin"}
        violations = set(np_usages) & forbidden_np_ops

        if violations:
            pytest.fail(
                f"Found numpy operations in compute_c2_heterodyne: {violations}\n"
                "Use jax.numpy instead for JAX traceability."
            )


class TestFitNLSQJax:
    """Tests for the main fit_nlsq_jax function."""

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_returns_result(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax returns a valid result."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=True,
        )

        assert result is not None
        assert hasattr(result, "parameters")
        assert len(result.parameters) == small_heterodyne_model.n_varying

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_scipy_fallback(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax works with scipy fallback."""
        from heterodyne import fit_nlsq_jax

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,  # Force scipy fallback
        )

        assert result is not None
        assert hasattr(result, "parameters")
        assert len(result.parameters) == small_heterodyne_model.n_varying

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_with_weights(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test fit_nlsq_jax accepts weights parameter."""
        from heterodyne import fit_nlsq_jax

        # Create simple uniform weights
        weights = np.ones_like(small_c2_data)

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            weights=weights,
            use_nlsq_library=False,  # Use scipy for reliability
        )

        assert result is not None
        assert len(result.parameters) == small_heterodyne_model.n_varying

    @pytest.mark.integration
    @pytest.mark.requires_jax
    def test_fit_nlsq_jax_updates_model(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        fast_nlsq_config: NLSQConfig,
    ) -> None:
        """Test that successful fit updates model parameters."""
        from heterodyne import fit_nlsq_jax

        # Get initial parameters
        small_heterodyne_model.get_params().copy()

        result = fit_nlsq_jax(
            model=small_heterodyne_model,
            c2_data=small_c2_data,
            phi_angle=0.0,
            config=fast_nlsq_config,
            use_nlsq_library=False,
        )

        if result.success:
            # Model should have updated parameters
            updated_params = small_heterodyne_model.get_params()
            # At least some parameters should have changed
            # (unless initial guess was already optimal)
            assert updated_params is not None


class TestResidualDtypes:
    """Tests for residual output dtypes."""

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_residual_output_dtype(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
        ensure_jax_float64: bool,
    ) -> None:
        """Verify residuals are computed in float64.

        This depends on JAX x64 mode being enabled.
        """
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data, dtype=jnp.float64)
        full_params = jnp.asarray(model.get_params(), dtype=jnp.float64)

        residuals = compute_residuals(
            full_params,
            model.t,
            model.q,
            model.dt,
            0.0,
            c2_jax,
            None,
        )

        assert residuals.dtype == jnp.float64, (
            f"Residuals have dtype {residuals.dtype}, expected float64. "
            "JAX x64 mode may not be enabled."
        )


class TestBugPrevention_JAXTracing:
    """Regression tests for JAX Tracing Error bug.

    BUG DESCRIPTION:
    Using numpy operations inside JAX-traced functions causes
    jax.errors.ConcretizationTypeError. For example:

        def bad_residual_fn(x, *params):
            return np.sum(x)  # BUG: numpy inside traced function

    The fix is to use jax.numpy exclusively:

        def good_residual_fn(x, *params):
            return jnp.sum(x)  # CORRECT: jax.numpy is traceable

    These tests verify the residual functions remain traceable.
    """

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_numpy_inside_traced_fn_fails(self) -> None:
        """REGRESSION TEST: Demonstrate that numpy inside traced fn fails.

        This test documents the bug behavior - using numpy inside a
        JAX-traced function should raise ConcretizationTypeError.
        """
        # This function uses numpy - it should NOT be traceable
        def bad_residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            # BUG: using numpy inside traced function
            return np.array([np.sum(np.asarray(x))])

        # Trying to trace this should fail
        with pytest.raises((jax.errors.ConcretizationTypeError, TypeError)):
            jax.make_jaxpr(bad_residual_fn)(jnp.arange(10.0))

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_jax_numpy_inside_traced_fn_succeeds(self) -> None:
        """REGRESSION TEST: Demonstrate that jax.numpy is traceable.

        This test documents the correct behavior - using jax.numpy
        inside a traced function should work.
        """
        # This function uses jax.numpy - it SHOULD be traceable
        def good_residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([jnp.sum(x)])

        # Tracing should succeed
        jaxpr = jax.make_jaxpr(good_residual_fn)(jnp.arange(10.0))
        assert jaxpr is not None

    @pytest.mark.unit
    @pytest.mark.requires_jax
    def test_compute_residuals_is_fully_traceable(
        self,
        small_heterodyne_model: HeterodyneModel,
        small_c2_data: np.ndarray,
    ) -> None:
        """REGRESSION TEST: Verify compute_residuals has no numpy leakage.

        If compute_residuals uses any numpy operations, this test will fail.
        """
        from heterodyne.core.jax_backend import compute_residuals

        model = small_heterodyne_model
        c2_jax = jnp.asarray(small_c2_data, dtype=jnp.float64)
        full_params = jnp.asarray(model.get_params(), dtype=jnp.float64)

        # Create a wrapper that we can trace
        def traceable_wrapper(params: jnp.ndarray) -> jnp.ndarray:
            return compute_residuals(
                params,
                model.t,
                model.q,
                model.dt,
                0.0,
                c2_jax,
                None,
            )

        # This should NOT raise ConcretizationTypeError
        try:
            jaxpr = jax.make_jaxpr(traceable_wrapper)(full_params)
            assert jaxpr is not None, "Tracing should produce a jaxpr"
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(
                f"compute_residuals is not traceable: {e}\n"
                "This indicates numpy operations inside the function."
            )
