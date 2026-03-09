"""Unit tests for heterodyne.viz.diagnostics module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from heterodyne.viz.diagnostics import (
    DiagonalOverlayResult,
    compute_diagonal_overlay_stats,
    plot_convergence_trace,
    plot_diagonal_overlay,
    plot_pair_correlation,
    plot_parameter_sensitivity,
    plot_residual_histogram,
    plot_residual_map,
    plot_trace_posterior,
    plot_weight_map,
)


@pytest.fixture(autouse=True)
def _close_figures() -> None:  # type: ignore[misc]
    """Close all matplotlib figures after each test."""
    yield  # type: ignore[misc]
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def times() -> np.ndarray:
    return np.linspace(0, 10, 20)


@pytest.fixture()
def square_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 20))


@pytest.fixture()
def samples_1d() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "alpha": rng.standard_normal(200),
        "beta": rng.standard_normal(200),
        "gamma": rng.standard_normal(200),
    }


@pytest.fixture()
def samples_2d() -> dict[str, np.ndarray]:
    """Multi-chain samples: shape (n_chains, n_draws)."""
    rng = np.random.default_rng(1)
    return {
        "alpha": rng.standard_normal((3, 100)),
        "beta": rng.standard_normal((3, 100)),
    }


# ---------------------------------------------------------------------------
# plot_diagonal_overlay
# ---------------------------------------------------------------------------


class TestPlotDiagonalOverlay:
    def test_returns_axes(self, times: np.ndarray, square_matrix: np.ndarray) -> None:
        ax = plot_diagonal_overlay(square_matrix, square_matrix * 0.9, times)
        assert ax is not None
        assert ax.get_xlabel() == "Time"
        assert ax.get_title() == "Diagonal Correction Overlay"

    def test_accepts_existing_axes(
        self, times: np.ndarray, square_matrix: np.ndarray
    ) -> None:
        _, provided_ax = plt.subplots()
        ax = plot_diagonal_overlay(square_matrix, square_matrix, times, ax=provided_ax)
        assert ax is provided_ax

    def test_legend_has_two_entries(
        self, times: np.ndarray, square_matrix: np.ndarray
    ) -> None:
        ax = plot_diagonal_overlay(square_matrix, square_matrix * 0.9, times)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Original" in labels
        assert "Corrected" in labels


# ---------------------------------------------------------------------------
# plot_residual_map
# ---------------------------------------------------------------------------


class TestPlotResidualMap:
    def test_returns_axes(self, times: np.ndarray, square_matrix: np.ndarray) -> None:
        ax = plot_residual_map(square_matrix, times)
        assert ax is not None
        assert ax.get_title() == "Residual Map"

    def test_accepts_existing_axes(
        self, times: np.ndarray, square_matrix: np.ndarray
    ) -> None:
        _, provided_ax = plt.subplots()
        ax = plot_residual_map(square_matrix, times, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# plot_weight_map
# ---------------------------------------------------------------------------


class TestPlotWeightMap:
    def test_returns_axes(self, times: np.ndarray, square_matrix: np.ndarray) -> None:
        ax = plot_weight_map(square_matrix, times)
        assert ax is not None
        assert ax.get_title() == "Weight Map"

    def test_accepts_existing_axes(
        self, times: np.ndarray, square_matrix: np.ndarray
    ) -> None:
        _, provided_ax = plt.subplots()
        ax = plot_weight_map(square_matrix, times, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# plot_convergence_trace
# ---------------------------------------------------------------------------


class TestPlotConvergenceTrace:
    def test_returns_axes(self) -> None:
        losses = np.exp(-np.linspace(0, 3, 50))
        ax = plot_convergence_trace(losses)
        assert ax is not None
        assert ax.get_title() == "Convergence Trace"
        assert ax.get_yscale() == "log"

    def test_linear_scale_when_requested(self) -> None:
        losses = np.exp(-np.linspace(0, 3, 50))
        ax = plot_convergence_trace(losses, log_scale=False)
        assert ax.get_yscale() == "linear"

    def test_linear_scale_when_losses_contain_non_positive(self) -> None:
        losses = np.array([1.0, 0.5, -0.1, 0.2])
        ax = plot_convergence_trace(losses, log_scale=True)
        # Should stay linear because not all losses > 0
        assert ax.get_yscale() == "linear"

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        losses = np.ones(10)
        ax = plot_convergence_trace(losses, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# plot_trace_posterior
# ---------------------------------------------------------------------------


class TestPlotTracePosterior:
    def test_returns_figure_1d_samples(
        self, samples_1d: dict[str, np.ndarray]
    ) -> None:
        fig = plot_trace_posterior(samples_1d)
        assert isinstance(fig, plt.Figure)
        # 3 params, 2 columns each
        axes = fig.get_axes()
        assert len(axes) == 6

    def test_returns_figure_2d_samples(
        self, samples_2d: dict[str, np.ndarray]
    ) -> None:
        fig = plot_trace_posterior(samples_2d)
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 4  # 2 params x 2 columns

    def test_subset_param_names(self, samples_1d: dict[str, np.ndarray]) -> None:
        fig = plot_trace_posterior(samples_1d, param_names=["alpha"])
        axes = fig.get_axes()
        assert len(axes) == 2  # 1 param x 2 columns

    def test_custom_figsize(self, samples_1d: dict[str, np.ndarray]) -> None:
        fig = plot_trace_posterior(samples_1d, figsize=(8, 6))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8)
        assert h == pytest.approx(6)


# ---------------------------------------------------------------------------
# plot_pair_correlation
# ---------------------------------------------------------------------------


class TestPlotPairCorrelation:
    def test_returns_axes(self, samples_1d: dict[str, np.ndarray]) -> None:
        ax = plot_pair_correlation(samples_1d)
        assert ax is not None
        assert ax.get_title() == "Parameter Correlation"

    def test_subset_param_names(self, samples_1d: dict[str, np.ndarray]) -> None:
        ax = plot_pair_correlation(samples_1d, param_names=["alpha", "beta"])
        assert ax is not None

    def test_accepts_existing_axes(self, samples_1d: dict[str, np.ndarray]) -> None:
        _, provided_ax = plt.subplots()
        ax = plot_pair_correlation(samples_1d, ax=provided_ax)
        assert ax is provided_ax

    def test_correlation_matrix_diag_is_one(
        self, samples_1d: dict[str, np.ndarray]
    ) -> None:
        """The diagonal of the correlation image should be 1.0."""
        ax = plot_pair_correlation(samples_1d)
        images = ax.get_images()
        assert len(images) == 1
        data = images[0].get_array()
        n = len(samples_1d)
        for i in range(n):
            assert data[i, i] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# plot_residual_histogram
# ---------------------------------------------------------------------------


class TestPlotResidualHistogram:
    def test_returns_axes(self) -> None:
        rng = np.random.default_rng(7)
        residuals = rng.standard_normal((10, 10))
        ax = plot_residual_histogram(residuals)
        assert ax is not None
        assert ax.get_title() == "Residual Distribution"

    def test_handles_nan_values(self) -> None:
        residuals = np.array([1.0, 2.0, np.nan, -1.0, np.inf, 0.5])
        ax = plot_residual_histogram(residuals)
        assert ax is not None

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        residuals = np.ones(50)
        ax = plot_residual_histogram(residuals, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# plot_parameter_sensitivity
# ---------------------------------------------------------------------------


class TestPlotParameterSensitivity:
    def test_returns_axes(self) -> None:
        sensitivity = {"D0_ref": 1.2, "alpha_ref": 0.3, "v0": 5.0}
        ax = plot_parameter_sensitivity(sensitivity)
        assert ax is not None
        assert ax.get_title() == "Parameter Sensitivity"

    def test_accepts_existing_axes(self) -> None:
        _, provided_ax = plt.subplots()
        sensitivity = {"a": 1.0}
        ax = plot_parameter_sensitivity(sensitivity, ax=provided_ax)
        assert ax is provided_ax


# ---------------------------------------------------------------------------
# DiagonalOverlayResult dataclass
# ---------------------------------------------------------------------------


class TestDiagonalOverlayResult:
    def test_fields(self) -> None:
        result = DiagonalOverlayResult(
            phi_index=0,
            raw_diagonal=np.array([1.0, 2.0]),
            solver_diagonal=np.array([1.1, 2.1]),
            posthoc_diagonal=np.array([1.05, 2.05]),
            raw_variance=0.25,
            solver_variance=0.25,
            posthoc_variance=0.001,
            solver_rmse=0.1,
            posthoc_rmse=0.05,
        )
        assert result.phi_index == 0
        assert result.raw_variance == 0.25
        assert result.solver_rmse == 0.1
        assert result.posthoc_rmse == 0.05
        np.testing.assert_array_equal(result.raw_diagonal, [1.0, 2.0])


# ---------------------------------------------------------------------------
# compute_diagonal_overlay_stats
# ---------------------------------------------------------------------------


class TestComputeDiagonalOverlayStats:
    @pytest.fixture()
    def c2_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create (n_phi, N, N) test arrays."""
        rng = np.random.default_rng(99)
        n_phi, n = 2, 10
        c2_exp = rng.standard_normal((n_phi, n, n))
        c2_solver = c2_exp + 0.01 * rng.standard_normal((n_phi, n, n))
        c2_posthoc = c2_exp + 0.001 * rng.standard_normal((n_phi, n, n))
        return c2_exp, c2_solver, c2_posthoc

    def test_basic_stats(
        self,
        c2_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        c2_exp, c2_solver, c2_posthoc = c2_arrays
        result = compute_diagonal_overlay_stats(c2_exp, c2_solver, c2_posthoc)
        assert result.phi_index == 0
        assert result.raw_diagonal.shape == (10,)
        assert result.solver_diagonal.shape == (10,)
        assert result.posthoc_diagonal.shape == (10,)
        assert result.raw_variance >= 0
        assert result.solver_variance >= 0
        assert result.posthoc_variance >= 0
        assert result.solver_rmse >= 0
        assert result.posthoc_rmse >= 0

    def test_custom_phi_index(
        self,
        c2_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        c2_exp, c2_solver, c2_posthoc = c2_arrays
        result = compute_diagonal_overlay_stats(
            c2_exp, c2_solver, c2_posthoc, phi_index=1
        )
        assert result.phi_index == 1

    def test_solver_none_raises(
        self,
        c2_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        c2_exp, _, c2_posthoc = c2_arrays
        with pytest.raises(ValueError, match="c2_solver must not be None"):
            compute_diagonal_overlay_stats(c2_exp, None, c2_posthoc)

    def test_identical_matrices_give_zero_rmse(self) -> None:
        c2 = np.eye(5).reshape(1, 5, 5)
        result = compute_diagonal_overlay_stats(c2, c2, c2)
        assert result.solver_rmse == pytest.approx(0.0, abs=1e-15)
        assert result.posthoc_rmse == pytest.approx(0.0, abs=1e-15)

    def test_posthoc_rmse_smaller_than_solver(
        self,
        c2_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Post-hoc correction should yield a smaller RMSE (by construction)."""
        c2_exp, c2_solver, c2_posthoc = c2_arrays
        result = compute_diagonal_overlay_stats(c2_exp, c2_solver, c2_posthoc)
        assert result.posthoc_rmse < result.solver_rmse
