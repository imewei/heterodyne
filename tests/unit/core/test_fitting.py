"""Tests for heterodyne.core.fitting — UnifiedHeterodyneEngine + JAX solvers."""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.core.fitting import (
    DatasetSize,
    FitResult,
    ParameterSpace,
    ScaledFittingEngine,
    UnifiedHeterodyneEngine,
    solve_least_squares_chunked_jax,
    solve_least_squares_general_jax,
    solve_least_squares_jax,
)

# ---------------------------------------------------------------------------
# ParameterSpace
# ---------------------------------------------------------------------------


class TestParameterSpace:
    """Tests for ParameterSpace dataclass."""

    def test_parameter_space_defaults(self) -> None:
        """Bounds and priors are populated from registry."""
        ps = ParameterSpace()
        bounds = ps.get_param_bounds()
        priors = ps.get_param_priors()
        assert len(bounds) == 14
        assert len(priors) == 14
        # Each entry is a 2-tuple
        for b in bounds:
            assert len(b) == 2
            assert b[0] < b[1]
        for p in priors:
            assert len(p) == 2
            assert p[1] > 0  # std > 0

    def test_parameter_space_get_bounds(self) -> None:
        """Returns correct count (14 physics params) with sensible values."""
        ps = ParameterSpace()
        bounds = ps.get_param_bounds()
        assert len(bounds) == 14
        # First param is D0_ref: (100, 1e6)
        assert bounds[0] == (100.0, 1e6)

    def test_parameter_space_scaling_bounds(self) -> None:
        """Scaling bounds use task-specified defaults."""
        ps = ParameterSpace()
        assert ps.contrast_bounds == (0.0, 10.0)
        assert ps.offset_bounds == (-1.0, 1.0)

    def test_parameter_space_scaling_priors(self) -> None:
        """Scaling priors use task-specified defaults."""
        ps = ParameterSpace()
        assert ps.contrast_prior == (1.0, 0.5)
        assert ps.offset_prior == (0.0, 0.25)

    def test_parameter_space_config_manager_override(self) -> None:
        """config_manager can be set for bound override."""
        ps = ParameterSpace(config_manager={"custom": True})
        assert ps.config_manager is not None


# ---------------------------------------------------------------------------
# DatasetSize
# ---------------------------------------------------------------------------


class TestDatasetSize:
    """Tests for DatasetSize classification."""

    def test_small(self) -> None:
        assert DatasetSize.categorize(500_000) == DatasetSize.SMALL

    def test_medium(self) -> None:
        assert DatasetSize.categorize(5_000_000) == DatasetSize.MEDIUM

    def test_large(self) -> None:
        assert DatasetSize.categorize(25_000_000) == DatasetSize.LARGE

    def test_boundary_small_medium(self) -> None:
        assert DatasetSize.categorize(999_999) == DatasetSize.SMALL
        assert DatasetSize.categorize(1_000_000) == DatasetSize.MEDIUM

    def test_boundary_medium_large(self) -> None:
        # Mirrors homodyne: <10M is MEDIUM, >=10M is LARGE
        assert DatasetSize.categorize(9_999_999) == DatasetSize.MEDIUM
        assert DatasetSize.categorize(10_000_000) == DatasetSize.LARGE


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


class TestFitResult:
    """Tests for FitResult dataclass."""

    def test_get_summary(self) -> None:
        """Dict structure has all expected keys."""
        result = FitResult(
            params=np.array([1.0, 2.0, 3.0]),
            contrast=0.8,
            offset=1.0,
            chi_squared=10.0,
            reduced_chi_squared=1.1,
            degrees_of_freedom=9,
        )
        summary = result.get_summary()
        assert "parameters" in summary
        assert "errors" in summary
        assert "fit_quality" in summary
        assert "convergence" in summary
        assert summary["parameters"]["contrast"] == 0.8
        assert summary["parameters"]["offset"] == 1.0
        assert summary["fit_quality"]["chi_squared"] == 10.0
        assert summary["convergence"]["converged"] is True

    def test_fit_result_defaults(self) -> None:
        """Default optional fields are sensible."""
        result = FitResult(
            params=np.zeros(3),
            contrast=1.0,
            offset=0.0,
            chi_squared=0.0,
            reduced_chi_squared=0.0,
            degrees_of_freedom=0,
        )
        assert result.param_errors is None
        assert result.contrast_error is None
        assert result.offset_error is None
        assert result.converged is True
        assert result.computation_time == 0.0


# ---------------------------------------------------------------------------
# JAX solvers
# ---------------------------------------------------------------------------


class TestSolveLeastSquaresJAX:
    """Tests for solve_least_squares_jax (batch 2x2 normal equations)."""

    def test_identity(self) -> None:
        """theory=data -> contrast~1, offset~0."""
        import jax.numpy as jnp

        theory = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        data = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        contrast, offset = solve_least_squares_jax(theory, data)
        np.testing.assert_allclose(float(contrast[0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(offset[0]), 0.0, atol=1e-6)

    def test_scaling(self) -> None:
        """Known contrast/offset recovery."""
        import jax.numpy as jnp

        theory = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        # data = 2.0 * theory + 0.5
        data = 2.0 * theory + 0.5
        contrast, offset = solve_least_squares_jax(theory, data)
        np.testing.assert_allclose(float(contrast[0]), 2.0, atol=1e-6)
        np.testing.assert_allclose(float(offset[0]), 0.5, atol=1e-6)

    def test_batch(self) -> None:
        """Multiple angles processed simultaneously."""
        import jax.numpy as jnp

        theory = jnp.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])
        data = jnp.array([
            [1.5, 3.0, 4.5],  # contrast=1.5, offset=0
            [2.0, 4.0, 6.0],  # contrast=2.0, offset=0
        ])
        contrast, offset = solve_least_squares_jax(theory, data)
        np.testing.assert_allclose(float(contrast[0]), 1.5, atol=1e-6)
        np.testing.assert_allclose(float(contrast[1]), 2.0, atol=1e-6)

    def test_singular_matrix(self) -> None:
        """Singular case (constant theory) returns safe defaults."""
        import jax.numpy as jnp

        theory = jnp.array([[1.0, 1.0, 1.0, 1.0]])
        data = jnp.array([[2.0, 2.0, 2.0, 2.0]])
        contrast, offset = solve_least_squares_jax(theory, data)
        # Should not crash; returns fallback values
        assert np.isfinite(float(contrast[0]))
        assert np.isfinite(float(offset[0]))


class TestSolveLeastSquaresGeneralJAX:
    """Tests for solve_least_squares_general_jax (N-param linear regression)."""

    def test_simple_regression(self) -> None:
        """2-param linear fit recovers known coefficients."""
        import jax.numpy as jnp

        # y = 3*x + 1
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 1.0
        A = jnp.column_stack([x, jnp.ones_like(x)])
        params = solve_least_squares_general_jax(A, y)
        np.testing.assert_allclose(float(params[0]), 3.0, atol=1e-6)
        np.testing.assert_allclose(float(params[1]), 1.0, atol=1e-6)

    def test_overdetermined(self) -> None:
        """More data points than parameters."""
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        x = jnp.linspace(0, 10, 100)
        noise = jnp.array(rng.normal(0, 0.01, 100))
        y = 2.5 * x - 0.3 + noise
        A = jnp.column_stack([x, jnp.ones_like(x)])
        params = solve_least_squares_general_jax(A, y)
        np.testing.assert_allclose(float(params[0]), 2.5, atol=0.05)
        np.testing.assert_allclose(float(params[1]), -0.3, atol=0.1)


class TestSolveLeastSquaresChunkedJAX:
    """Tests for solve_least_squares_chunked_jax (scan-based accumulation)."""

    def test_matches_non_chunked(self) -> None:
        """Chunked result matches non-chunked for same data."""
        import jax.numpy as jnp

        rng = np.random.default_rng(123)
        theory_full = jnp.array(rng.uniform(0.5, 2.0, (1, 100)))
        data_full = 1.5 * theory_full + 0.3

        # Non-chunked
        c_ref, o_ref = solve_least_squares_jax(theory_full, data_full)

        # Chunked: reshape to (n_chunks, chunk_size)
        theory_chunks = theory_full.reshape(10, 10)
        data_chunks = data_full.reshape(10, 10)
        c_chunked, o_chunked = solve_least_squares_chunked_jax(
            theory_chunks, data_chunks
        )

        np.testing.assert_allclose(float(c_chunked), float(c_ref[0]), atol=1e-6)
        np.testing.assert_allclose(float(o_chunked), float(o_ref[0]), atol=1e-6)


# ---------------------------------------------------------------------------
# UnifiedHeterodyneEngine
# ---------------------------------------------------------------------------


class TestUnifiedHeterodyneEngine:
    """Tests for UnifiedHeterodyneEngine."""

    def test_default_construction(self) -> None:
        """Engine constructs with default parameter space."""
        engine = UnifiedHeterodyneEngine()
        assert engine.parameter_space is not None
        assert isinstance(engine.parameter_space, ParameterSpace)

    def test_backward_compat_alias(self) -> None:
        """ScaledFittingEngine is an alias for UnifiedHeterodyneEngine."""
        assert ScaledFittingEngine is UnifiedHeterodyneEngine

    def test_estimate_scaling(self) -> None:
        """Contrast/offset estimation via JAX solver."""
        engine = UnifiedHeterodyneEngine()
        rng = np.random.default_rng(42)
        theory = rng.uniform(0.5, 2.0, 200)
        data = 0.8 * theory + 0.1
        contrast, offset = engine.estimate_scaling_parameters(data, theory)
        np.testing.assert_allclose(contrast, 0.8, atol=0.01)
        np.testing.assert_allclose(offset, 0.1, atol=0.01)

    def test_validate_inputs_empty(self) -> None:
        """Raises on empty data."""
        engine = UnifiedHeterodyneEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.validate_inputs(
                data=np.array([]),
                sigma=np.array([]),
                t=np.array([1.0]),
                phi=np.array([0.0]),
                q=1.0,
            )

    def test_validate_inputs_negative_q(self) -> None:
        """Raises on q<=0."""
        engine = UnifiedHeterodyneEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.validate_inputs(
                data=np.array([1.0]),
                sigma=np.array([0.1]),
                t=np.array([1.0]),
                phi=np.array([0.0]),
                q=-1.0,
            )

    def test_validate_inputs_sigma_shape_mismatch(self) -> None:
        """Raises when sigma shape != data shape."""
        engine = UnifiedHeterodyneEngine()
        with pytest.raises(ValueError, match="shape"):
            engine.validate_inputs(
                data=np.array([1.0, 2.0]),
                sigma=np.array([0.1]),
                t=np.array([1.0]),
                phi=np.array([0.0]),
                q=1.0,
            )

    def test_detect_dataset_size(self) -> None:
        """Dataset size detection delegates to DatasetSize."""
        engine = UnifiedHeterodyneEngine()
        small = engine.detect_dataset_size(np.zeros(100))
        assert small == DatasetSize.SMALL
        large = engine.detect_dataset_size(np.zeros(30_000_000))
        assert large == DatasetSize.LARGE

    def test_get_parameter_info(self) -> None:
        """Parameter info dict has expected structure."""
        engine = UnifiedHeterodyneEngine()
        info = engine.get_parameter_info()
        assert "parameter_count" in info
        assert info["parameter_count"] == 14
        assert "physical_bounds" in info
        assert "scaling_bounds" in info
        assert "scaling_priors" in info
