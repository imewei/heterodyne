"""Tests for heterodyne.optimization.gradient_diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from heterodyne.optimization.gradient_diagnostics import (
    GradientHealth,
    compute_gradient_norm,
    compute_gradient_norms,
    compute_optimal_x_scale,
    compute_per_parameter_sensitivity,
    diagnose_gradient_imbalance,
    diagnose_gradients,
    suggest_step_sizes,
)


# ---------------------------------------------------------------------------
# GradientHealth dataclass
# ---------------------------------------------------------------------------


class TestGradientHealth:
    def test_construction_healthy(self) -> None:
        h = GradientHealth(is_healthy=True)
        assert h.is_healthy is True
        assert h.issues == []
        assert h.metrics == {}

    def test_construction_unhealthy(self) -> None:
        h = GradientHealth(
            is_healthy=False,
            issues=["vanishing"],
            metrics={"gradient_norm": 1e-15},
        )
        assert h.is_healthy is False
        assert len(h.issues) == 1
        assert h.metrics["gradient_norm"] == pytest.approx(1e-15)


# ---------------------------------------------------------------------------
# compute_gradient_norm
# ---------------------------------------------------------------------------


class TestComputeGradientNorm:
    def test_identity_matrix(self) -> None:
        jac = np.eye(3)
        norm = compute_gradient_norm(jac)
        assert norm == pytest.approx(np.sqrt(3.0))

    def test_all_zeros(self) -> None:
        jac = np.zeros((5, 3))
        assert compute_gradient_norm(jac) == pytest.approx(0.0)

    def test_single_element(self) -> None:
        jac = np.array([[7.0]])
        assert compute_gradient_norm(jac) == pytest.approx(7.0)

    def test_positive(self) -> None:
        rng = np.random.default_rng(42)
        jac = rng.standard_normal((10, 4))
        assert compute_gradient_norm(jac) > 0.0


# ---------------------------------------------------------------------------
# compute_per_parameter_sensitivity
# ---------------------------------------------------------------------------


class TestComputePerParameterSensitivity:
    def test_diagonal_jacobian(self) -> None:
        jac = np.diag([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]
        sens = compute_per_parameter_sensitivity(jac, names)
        assert sens["a"] == pytest.approx(1.0)
        assert sens["b"] == pytest.approx(2.0)
        assert sens["c"] == pytest.approx(3.0)

    def test_length_mismatch_raises(self) -> None:
        jac = np.eye(3)
        with pytest.raises(ValueError):
            compute_per_parameter_sensitivity(jac, ["a", "b"])

    def test_single_param(self) -> None:
        jac = np.array([[3.0], [4.0]])
        sens = compute_per_parameter_sensitivity(jac, ["x"])
        assert sens["x"] == pytest.approx(5.0)  # sqrt(9 + 16)


# ---------------------------------------------------------------------------
# suggest_step_sizes
# ---------------------------------------------------------------------------


class TestSuggestStepSizes:
    def test_inverse_proportionality(self) -> None:
        """Parameters with larger column norms should get smaller steps."""
        jac = np.diag([10.0, 1000.0])
        names = ["slow", "fast"]
        steps = suggest_step_sizes(jac, names)
        assert steps["fast"] < steps["slow"]

    def test_clipping_floor(self) -> None:
        """Huge column norm should clip step to floor."""
        jac = np.array([[1e20]])
        steps = suggest_step_sizes(jac, ["x"])
        assert steps["x"] >= 1e-12

    def test_clipping_ceiling(self) -> None:
        """Very small column norm should clip step to ceiling."""
        jac = np.array([[1e-20]])
        steps = suggest_step_sizes(jac, ["x"])
        assert steps["x"] <= 1e-2

    def test_all_zeros(self) -> None:
        """Zero Jacobian column should produce the max step (ceiling)."""
        jac = np.zeros((3, 1))
        steps = suggest_step_sizes(jac, ["x"])
        assert steps["x"] == pytest.approx(1e-2)


# ---------------------------------------------------------------------------
# diagnose_gradients
# ---------------------------------------------------------------------------


class TestDiagnoseGradients:
    def test_healthy_jacobian(self) -> None:
        jac = np.eye(3)
        res = np.ones(3)
        names = ["a", "b", "c"]
        report = diagnose_gradients(jac, res, names)
        assert report.is_healthy is True
        assert report.issues == []
        assert "gradient_norm" in report.metrics

    def test_nan_in_jacobian(self) -> None:
        jac = np.array([[1.0, float("nan")], [0.0, 1.0]])
        res = np.ones(2)
        report = diagnose_gradients(jac, res, ["a", "b"])
        assert report.is_healthy is False
        assert any("NaN" in msg for msg in report.issues)

    def test_inf_in_jacobian(self) -> None:
        jac = np.array([[float("inf")]])
        res = np.ones(1)
        report = diagnose_gradients(jac, res, ["x"])
        assert report.is_healthy is False
        assert any("Inf" in msg for msg in report.issues)

    def test_nan_in_residuals(self) -> None:
        jac = np.eye(2)
        res = np.array([1.0, float("nan")])
        report = diagnose_gradients(jac, res, ["a", "b"])
        assert report.is_healthy is False
        assert any("Residuals" in msg and "NaN" in msg for msg in report.issues)

    def test_vanishing_gradient(self) -> None:
        jac = np.eye(2) * 1e-14
        res = np.ones(2)
        report = diagnose_gradients(jac, res, ["a", "b"])
        assert report.is_healthy is False
        assert any("Vanishing" in msg for msg in report.issues)

    def test_exploding_gradient(self) -> None:
        jac = np.eye(2) * 1e14
        res = np.ones(2)
        report = diagnose_gradients(jac, res, ["a", "b"])
        assert report.is_healthy is False
        assert any("Exploding" in msg for msg in report.issues)

    def test_imbalanced_sensitivity(self) -> None:
        """Large sensitivity ratio should be flagged."""
        jac = np.diag([1e-6, 1e6])
        res = np.ones(2)
        report = diagnose_gradients(jac, res, ["small", "big"])
        assert any("Imbalanced" in msg for msg in report.issues)
        assert "sensitivity_ratio" in report.metrics
        assert report.metrics["sensitivity_ratio"] > 1e6


# ---------------------------------------------------------------------------
# compute_gradient_norms (JAX autodiff)
# ---------------------------------------------------------------------------


class TestComputeGradientNorms:
    def test_simple_quadratic(self) -> None:
        """For r(p) = p, L = sum(p^2), grad_i = 2*p_i."""

        def residual_fn(p: jnp.ndarray) -> jnp.ndarray:
            return p

        params = jnp.array([3.0, 4.0])
        names = ["a", "b"]
        norms = compute_gradient_norms(residual_fn, params, names)
        # dL/da = 2*3 = 6, dL/db = 2*4 = 8
        assert norms["a"] == pytest.approx(6.0, rel=1e-5)
        assert norms["b"] == pytest.approx(8.0, rel=1e-5)

    def test_single_param(self) -> None:
        def residual_fn(p: jnp.ndarray) -> jnp.ndarray:
            return p * 2.0

        params = jnp.array([1.0])
        norms = compute_gradient_norms(residual_fn, params, ["x"])
        # r = 2p, L = 4p^2, dL/dp = 8p = 8
        assert norms["x"] == pytest.approx(8.0, rel=1e-5)

    def test_norms_are_nonnegative(self) -> None:
        def residual_fn(p: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(p)

        params = jnp.array([1.0, -1.0, 0.5])
        norms = compute_gradient_norms(residual_fn, params, ["a", "b", "c"])
        for v in norms.values():
            assert v >= 0.0

    def test_zero_residual(self) -> None:
        """Constant-zero residual -> zero gradients."""

        def residual_fn(p: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros_like(p)

        params = jnp.array([1.0, 2.0])
        norms = compute_gradient_norms(residual_fn, params, ["a", "b"])
        for v in norms.values():
            assert v == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# compute_optimal_x_scale
# ---------------------------------------------------------------------------


class TestComputeOptimalXScale:
    def test_uniform_norms_yield_unit_scale(self) -> None:
        """When all gradient norms are equal, scale ~ 1.0."""
        norms = {"a": 10.0, "b": 10.0, "c": 10.0}
        scales = compute_optimal_x_scale(norms)
        for v in scales.values():
            assert v == pytest.approx(1.0, rel=0.01)

    def test_inverse_proportional(self) -> None:
        """Larger gradient norm -> smaller x_scale."""
        norms = {"slow": 1.0, "fast": 100.0}
        scales = compute_optimal_x_scale(norms)
        assert scales["fast"] < scales["slow"]

    def test_zero_norm_gets_unit_scale(self) -> None:
        norms = {"a": 0.0, "b": 5.0}
        scales = compute_optimal_x_scale(norms)
        assert scales["a"] == 1.0

    def test_all_zero_norms(self) -> None:
        norms = {"a": 0.0, "b": 0.0}
        scales = compute_optimal_x_scale(norms)
        # All zero -> fallback to 1.0
        assert scales["a"] == 1.0
        assert scales["b"] == 1.0

    def test_clipping_bounds(self) -> None:
        norms = {"tiny": 1e-20, "huge": 1e20}
        scales = compute_optimal_x_scale(
            norms, min_scale=1e-8, max_scale=1e2,
        )
        for v in scales.values():
            assert 1e-8 <= v <= 1e2

    def test_custom_baseline_params(self) -> None:
        norms = {"a": 1.0, "b": 100.0, "c": 10.0}
        scales = compute_optimal_x_scale(norms, baseline_params=["c"])
        # Baseline is norm of c=10, so scale_c ~ 1.0
        assert scales["c"] == pytest.approx(1.0, rel=0.01)

    def test_safety_factor(self) -> None:
        norms = {"a": 10.0, "b": 10.0}
        scales_1x = compute_optimal_x_scale(norms, safety_factor=1.0)
        scales_2x = compute_optimal_x_scale(norms, safety_factor=2.0)
        assert scales_2x["a"] == pytest.approx(2.0 * scales_1x["a"], rel=0.01)

    def test_baseline_params_missing(self) -> None:
        """Baseline params not in gradient_norms -> falls back to median."""
        norms = {"a": 1.0, "b": 100.0}
        scales = compute_optimal_x_scale(norms, baseline_params=["missing"])
        # Should not raise; falls back to median of nonzero norms
        assert len(scales) == 2


# ---------------------------------------------------------------------------
# diagnose_gradient_imbalance
# ---------------------------------------------------------------------------


class TestDiagnoseGradientImbalance:
    def test_balanced(self) -> None:
        norms = {"a": 5.0, "b": 6.0, "c": 5.5}
        result = diagnose_gradient_imbalance(norms, threshold=10.0)
        assert result["imbalance_detected"] is False
        assert result["recommendations"] is None
        assert result["max_ratio"] == pytest.approx(6.0 / 5.0, rel=1e-6)

    def test_imbalanced(self) -> None:
        norms = {"slow": 0.01, "fast": 1000.0}
        result = diagnose_gradient_imbalance(norms, threshold=10.0)
        assert result["imbalance_detected"] is True
        assert result["max_ratio"] == pytest.approx(1e5, rel=1e-3)
        assert result["recommendations"] is not None
        assert "slow" in result["recommendations"]
        assert "fast" in result["recommendations"]

    def test_single_nonzero(self) -> None:
        """Fewer than 2 nonzero norms -> no imbalance."""
        norms = {"a": 5.0, "b": 0.0}
        result = diagnose_gradient_imbalance(norms)
        assert result["imbalance_detected"] is False
        assert result["max_ratio"] == 1.0

    def test_all_zero(self) -> None:
        norms = {"a": 0.0, "b": 0.0}
        result = diagnose_gradient_imbalance(norms)
        assert result["imbalance_detected"] is False

    def test_custom_threshold(self) -> None:
        norms = {"a": 1.0, "b": 5.0}
        # ratio = 5.0, threshold = 3.0 -> imbalanced
        result = diagnose_gradient_imbalance(norms, threshold=3.0)
        assert result["imbalance_detected"] is True

    def test_gradient_norms_in_output(self) -> None:
        norms = {"x": 1.0, "y": 2.0}
        result = diagnose_gradient_imbalance(norms)
        assert result["gradient_norms"] is norms
