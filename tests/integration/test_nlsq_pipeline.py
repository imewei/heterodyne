"""Integration tests for the NLSQ analysis pipeline.

Tests Jacobian computation, condition numbers, sensitivity analysis,
gradient noise estimation, and the HierarchicalResult container.
All tests use pure NumPy residual functions — no real model required.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.nlsq.jacobian import (
    analyze_parameter_sensitivity,
    compute_jacobian_condition_number,
    compute_numerical_jacobian,
    estimate_gradient_noise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_residual(x: np.ndarray) -> np.ndarray:
    """Simple linear residual: r_i = x_i - target_i."""
    target = np.array([1.0, 2.0, 3.0])
    return x - target


def _quadratic_residual(x: np.ndarray) -> np.ndarray:
    """Quadratic residual over 10 data points: r_i = (x[0] - i)^2 + x[1]*i."""
    n = 10
    t = np.arange(n, dtype=np.float64)
    return (x[0] - t) ** 2 + x[1] * t


# ---------------------------------------------------------------------------
# compute_numerical_jacobian
# ---------------------------------------------------------------------------


class TestComputeNumericalJacobian:
    """Tests for compute_numerical_jacobian."""

    def test_jacobian_shape_linear(self) -> None:
        """Jacobian of a linear residual has shape (n_residuals, n_params)."""
        params = np.array([1.0, 2.0, 3.0])
        jac = compute_numerical_jacobian(_linear_residual, params)
        assert jac.shape == (3, 3)

    def test_jacobian_linear_is_identity(self) -> None:
        """Jacobian of r(x) = x - c is the identity matrix."""
        params = np.array([1.5, 2.5, 0.5])
        jac = compute_numerical_jacobian(_linear_residual, params)
        np.testing.assert_allclose(jac, np.eye(3), atol=1e-6)

    def test_jacobian_shape_quadratic(self) -> None:
        """Jacobian of quadratic residual has shape (n_data, n_params)."""
        params = np.array([1.0, 0.5])
        jac = compute_numerical_jacobian(_quadratic_residual, params)
        assert jac.shape == (10, 2)

    def test_jacobian_all_finite(self) -> None:
        """Jacobian values are all finite for a well-behaved function."""
        params = np.array([2.0, -1.0])
        jac = compute_numerical_jacobian(_quadratic_residual, params)
        assert np.all(np.isfinite(jac))

    def test_jacobian_custom_step_sizes(self) -> None:
        """Custom step sizes produce a valid (finite) Jacobian."""
        params = np.array([1.0, 2.0, 3.0])
        steps = np.array([1e-5, 1e-5, 1e-5])
        jac = compute_numerical_jacobian(_linear_residual, params, step_sizes=steps)
        np.testing.assert_allclose(jac, np.eye(3), atol=1e-4)

    def test_jacobian_step_size_shape_mismatch(self) -> None:
        """Mismatched step_sizes shape raises ValueError."""
        params = np.array([1.0, 2.0, 3.0])
        bad_steps = np.array([1e-5, 1e-5])
        with pytest.raises(ValueError, match="step_sizes"):
            compute_numerical_jacobian(_linear_residual, params, step_sizes=bad_steps)


# ---------------------------------------------------------------------------
# compute_jacobian_condition_number
# ---------------------------------------------------------------------------


class TestComputeJacobianConditionNumber:
    """Tests for compute_jacobian_condition_number."""

    def test_condition_number_identity_jacobian(self) -> None:
        """Condition number of the identity Jacobian is 1."""
        jac = np.eye(4)
        cond = compute_jacobian_condition_number(jac)
        assert np.isfinite(cond)
        assert cond == pytest.approx(1.0, rel=1e-6)

    def test_condition_number_returns_finite_float(self) -> None:
        """Returns a finite float for a generic well-conditioned Jacobian."""
        params = np.array([1.0, 0.5])
        jac = compute_numerical_jacobian(_quadratic_residual, params)
        cond = compute_jacobian_condition_number(jac)
        assert isinstance(cond, float)
        assert np.isfinite(cond)

    def test_condition_number_ill_conditioned(self) -> None:
        """Ill-conditioned Jacobian gives a large condition number."""
        # Columns differ by factor of 1e10 → JtJ is very ill-conditioned
        jac = np.array([[1.0, 1e10], [1.0, 1e10 + 1.0]])
        cond = compute_jacobian_condition_number(jac)
        assert cond > 1e10


# ---------------------------------------------------------------------------
# analyze_parameter_sensitivity
# ---------------------------------------------------------------------------


class TestAnalyzeParameterSensitivity:
    """Tests for analyze_parameter_sensitivity."""

    def test_sensitivity_returns_correct_keys(self) -> None:
        """Output dict has one key per parameter name."""
        jac = np.eye(3)
        names = ["p0", "p1", "p2"]
        sensitivity = analyze_parameter_sensitivity(jac, names)
        assert set(sensitivity.keys()) == set(names)

    def test_sensitivity_values_are_l2_column_norms(self) -> None:
        """Sensitivity equals the L2 norm of each Jacobian column."""
        jac = np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 2.0]])
        names = ["a", "b"]
        sensitivity = analyze_parameter_sensitivity(jac, names)
        assert sensitivity["a"] == pytest.approx(5.0)  # sqrt(9 + 16)
        assert sensitivity["b"] == pytest.approx(2.0)

    def test_sensitivity_zero_column_gives_zero(self) -> None:
        """An all-zero Jacobian column gives sensitivity 0."""
        jac = np.zeros((5, 2))
        jac[:, 0] = 1.0  # only first column is non-zero
        names = ["active", "inactive"]
        sensitivity = analyze_parameter_sensitivity(jac, names)
        assert sensitivity["inactive"] == pytest.approx(0.0)

    def test_sensitivity_name_length_mismatch(self) -> None:
        """Wrong number of names raises ValueError."""
        jac = np.eye(3)
        with pytest.raises(ValueError, match="param_names"):
            analyze_parameter_sensitivity(jac, param_names=["only_one"])

    def test_sensitivity_all_values_non_negative(self) -> None:
        """All sensitivity values are non-negative (they are norms)."""
        rng = np.random.default_rng(7)
        jac = rng.standard_normal((10, 4))
        names = [f"p{i}" for i in range(4)]
        sensitivity = analyze_parameter_sensitivity(jac, names)
        assert all(v >= 0 for v in sensitivity.values())


# ---------------------------------------------------------------------------
# estimate_gradient_noise
# ---------------------------------------------------------------------------


class TestEstimateGradientNoise:
    """Tests for estimate_gradient_noise."""

    def test_gradient_noise_returns_expected_keys(self) -> None:
        """Output dict contains the three documented keys."""
        rng = np.random.default_rng(1)
        jac = rng.standard_normal((20, 3))
        r = rng.standard_normal(20)
        result = estimate_gradient_noise(jac, r)
        assert "mean_noise_ratio" in result
        assert "max_noise_ratio" in result
        assert "noisy_params_fraction" in result

    def test_gradient_noise_values_are_finite(self) -> None:
        """All returned values are finite floats."""
        rng = np.random.default_rng(2)
        jac = rng.standard_normal((15, 4))
        r = rng.standard_normal(15)
        result = estimate_gradient_noise(jac, r)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite"

    def test_gradient_noise_max_ge_mean(self) -> None:
        """max_noise_ratio is always >= mean_noise_ratio."""
        rng = np.random.default_rng(3)
        jac = rng.standard_normal((20, 5))
        r = rng.standard_normal(20)
        result = estimate_gradient_noise(jac, r)
        assert result["max_noise_ratio"] >= result["mean_noise_ratio"]

    def test_gradient_noise_fraction_in_unit_interval(self) -> None:
        """noisy_params_fraction is in [0, 1]."""
        rng = np.random.default_rng(4)
        jac = rng.standard_normal((30, 6))
        r = rng.standard_normal(30)
        result = estimate_gradient_noise(jac, r)
        assert 0.0 <= result["noisy_params_fraction"] <= 1.0

    def test_gradient_noise_residual_shape_mismatch(self) -> None:
        """Mismatched residual length raises ValueError."""
        jac = np.ones((10, 3))
        r_bad = np.ones(7)  # wrong length
        with pytest.raises(ValueError, match="residuals"):
            estimate_gradient_noise(jac, r_bad)

    def test_gradient_noise_zero_residuals_stable(self) -> None:
        """Zero residuals do not cause NaN or division errors."""
        jac = np.ones((8, 2))
        r = np.zeros(8)
        result = estimate_gradient_noise(jac, r)
        for val in result.values():
            assert np.isfinite(val)


# ---------------------------------------------------------------------------
# HierarchicalResult (via HierarchicalFitter internals)
# ---------------------------------------------------------------------------


class TestHierarchicalResultCreation:
    """Tests for HierarchicalConfig creation and validation."""

    def test_hierarchical_config_default_creation(self) -> None:
        """HierarchicalConfig can be created with default stages."""
        from heterodyne.optimization.nlsq.hierarchical import HierarchicalConfig

        config = HierarchicalConfig()
        assert len(config.stages) > 0
        assert all("name" in s for s in config.stages)
        assert all("groups" in s for s in config.stages)

    def test_hierarchical_config_stage_names(self) -> None:
        """Default stages include 'transport', 'velocity', 'fraction', 'all'."""
        from heterodyne.optimization.nlsq.hierarchical import HierarchicalConfig

        config = HierarchicalConfig()
        names = [s["name"] for s in config.stages]
        assert "transport" in names
        assert "all" in names

    def test_hierarchical_config_invalid_group(self) -> None:
        """Stage with unknown group name raises ValueError."""
        from heterodyne.optimization.nlsq.hierarchical import HierarchicalConfig

        with pytest.raises(ValueError, match="unknown group"):
            HierarchicalConfig(
                stages=[{"name": "bad", "groups": ["nonexistent_group"]}]
            )

    def test_hierarchical_config_missing_name(self) -> None:
        """Stage without 'name' key raises ValueError."""
        from heterodyne.optimization.nlsq.hierarchical import HierarchicalConfig

        with pytest.raises(ValueError, match="missing required 'name'"):
            HierarchicalConfig(stages=[{"groups": ["reference"]}])
