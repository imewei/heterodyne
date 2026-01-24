"""Tests for multi-start optimization module.

Tests MultiStartOptimizer and Latin Hypercube Sampling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.multistart import (
    MultiStartOptimizer,
    MultiStartResult,
)
from heterodyne.optimization.nlsq.results import NLSQResult

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create mock NLSQ adapter."""
    adapter = MagicMock()
    adapter.name = "mock"
    return adapter


@pytest.fixture
def nlsq_config() -> NLSQConfig:
    """Create NLSQ configuration."""
    return NLSQConfig(max_iterations=100, tolerance=1e-6)


@pytest.fixture
def simple_residual_fn():
    """Simple quadratic residual function."""
    def residual(params):
        return params - np.array([1.0, 2.0, 3.0])
    return residual


@pytest.fixture
def simple_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Simple bounds for 3-parameter problem."""
    return (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))


@pytest.fixture
def initial_params() -> np.ndarray:
    """Initial parameters."""
    return np.array([5.0, 5.0, 5.0])


# ============================================================================
# MultiStartOptimizer Initialization Tests
# ============================================================================


class TestMultiStartOptimizerInit:
    """Tests for MultiStartOptimizer initialization."""

    @pytest.mark.unit
    def test_init_with_defaults(self, mock_adapter: MagicMock) -> None:
        """Initialize with default values."""
        optimizer = MultiStartOptimizer(mock_adapter)
        assert optimizer._n_starts == 10
        assert optimizer._adapter is mock_adapter

    @pytest.mark.unit
    def test_init_with_custom_n_starts(self, mock_adapter: MagicMock) -> None:
        """Initialize with custom number of starts."""
        optimizer = MultiStartOptimizer(mock_adapter, n_starts=20)
        assert optimizer._n_starts == 20

    @pytest.mark.unit
    def test_init_with_seed_reproducible(self, mock_adapter: MagicMock) -> None:
        """Same seed produces reproducible results."""
        opt1 = MultiStartOptimizer(mock_adapter, n_starts=5, seed=42)
        opt2 = MultiStartOptimizer(mock_adapter, n_starts=5, seed=42)

        bounds = (np.zeros(3), np.ones(3))
        initial = np.array([0.5, 0.5, 0.5])

        points1 = opt1.generate_starting_points(initial, bounds)
        points2 = opt2.generate_starting_points(initial, bounds)

        assert_allclose(points1, points2)

    @pytest.mark.unit
    def test_init_different_seeds_different_results(
        self, mock_adapter: MagicMock
    ) -> None:
        """Different seeds produce different results."""
        opt1 = MultiStartOptimizer(mock_adapter, n_starts=5, seed=42)
        opt2 = MultiStartOptimizer(mock_adapter, n_starts=5, seed=123)

        bounds = (np.zeros(3), np.ones(3))
        initial = np.array([0.5, 0.5, 0.5])

        points1 = opt1.generate_starting_points(initial, bounds)
        points2 = opt2.generate_starting_points(initial, bounds)

        # First point is the same (initial), but LHS points differ
        assert_allclose(points1[0], points2[0])  # Both are initial
        assert not np.allclose(points1[1:], points2[1:])  # LHS points differ


# ============================================================================
# generate_starting_points Tests
# ============================================================================


class TestGenerateStartingPoints:
    """Tests for generate_starting_points method."""

    @pytest.mark.unit
    def test_first_point_is_initial(
        self,
        mock_adapter: MagicMock,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """First starting point is the user-provided initial guess."""
        optimizer = MultiStartOptimizer(mock_adapter, n_starts=5, seed=42)
        points = optimizer.generate_starting_points(initial_params, simple_bounds)

        assert_allclose(points[0], initial_params)

    @pytest.mark.unit
    def test_correct_number_of_points(
        self,
        mock_adapter: MagicMock,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """Generates correct number of starting points."""
        n_starts = 7
        optimizer = MultiStartOptimizer(mock_adapter, n_starts=n_starts, seed=42)
        points = optimizer.generate_starting_points(initial_params, simple_bounds)

        assert points.shape[0] == n_starts
        assert points.shape[1] == len(initial_params)

    @pytest.mark.unit
    def test_points_within_bounds(
        self,
        mock_adapter: MagicMock,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """All generated points are within bounds."""
        optimizer = MultiStartOptimizer(mock_adapter, n_starts=20, seed=42)
        points = optimizer.generate_starting_points(initial_params, simple_bounds)

        lower, upper = simple_bounds
        for i in range(len(points)):
            assert np.all(points[i] >= lower), f"Point {i} below lower bound"
            assert np.all(points[i] <= upper), f"Point {i} above upper bound"

    @pytest.mark.unit
    def test_single_start_only_initial(
        self,
        mock_adapter: MagicMock,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """With n_starts=1, only initial point is returned."""
        optimizer = MultiStartOptimizer(mock_adapter, n_starts=1, seed=42)
        points = optimizer.generate_starting_points(initial_params, simple_bounds)

        assert points.shape[0] == 1
        assert_allclose(points[0], initial_params)


# ============================================================================
# Latin Hypercube Sampling Tests
# ============================================================================


class TestLatinHypercubeSampling:
    """Tests for _latin_hypercube_sample method."""

    @pytest.mark.unit
    def test_lhs_correct_shape(self, mock_adapter: MagicMock) -> None:
        """LHS produces correct shape."""
        optimizer = MultiStartOptimizer(mock_adapter, seed=42)
        lower = np.zeros(5)
        upper = np.ones(5)

        samples = optimizer._latin_hypercube_sample(10, 5, lower, upper)

        assert len(samples) == 10
        assert all(s.shape == (5,) for s in samples)

    @pytest.mark.unit
    def test_lhs_within_bounds(self, mock_adapter: MagicMock) -> None:
        """LHS samples are within bounds."""
        optimizer = MultiStartOptimizer(mock_adapter, seed=42)
        lower = np.array([0.0, -5.0, 10.0])
        upper = np.array([10.0, 5.0, 20.0])

        samples = optimizer._latin_hypercube_sample(50, 3, lower, upper)

        for sample in samples:
            assert np.all(sample >= lower)
            assert np.all(sample <= upper)

    @pytest.mark.unit
    def test_lhs_covers_space(self, mock_adapter: MagicMock) -> None:
        """LHS samples cover parameter space reasonably well."""
        optimizer = MultiStartOptimizer(mock_adapter, seed=42)
        lower = np.zeros(2)
        upper = np.ones(2)

        samples = optimizer._latin_hypercube_sample(100, 2, lower, upper)
        samples_array = np.array(samples)

        # Check coverage: each quadrant should have some samples
        n_quadrant1 = np.sum((samples_array[:, 0] < 0.5) & (samples_array[:, 1] < 0.5))
        n_quadrant2 = np.sum((samples_array[:, 0] >= 0.5) & (samples_array[:, 1] < 0.5))
        n_quadrant3 = np.sum((samples_array[:, 0] < 0.5) & (samples_array[:, 1] >= 0.5))
        n_quadrant4 = np.sum((samples_array[:, 0] >= 0.5) & (samples_array[:, 1] >= 0.5))

        # Each quadrant should have at least some samples (not exact for random)
        assert n_quadrant1 > 5
        assert n_quadrant2 > 5
        assert n_quadrant3 > 5
        assert n_quadrant4 > 5


# ============================================================================
# fit Method Tests
# ============================================================================


class TestMultiStartFit:
    """Tests for fit method."""

    @pytest.mark.unit
    def test_fit_returns_multistart_result(
        self,
        mock_adapter: MagicMock,
        nlsq_config: NLSQConfig,
        simple_residual_fn,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """fit() returns MultiStartResult."""
        # Configure mock to return successful result
        mock_result = NLSQResult(
            parameters=np.array([1.0, 2.0, 3.0]),
            parameter_names=["a", "b", "c"],
            success=True,
            message="Converged",
            final_cost=0.0,
        )
        mock_adapter.fit.return_value = mock_result

        optimizer = MultiStartOptimizer(mock_adapter, n_starts=3, seed=42)
        result = optimizer.fit(
            residual_fn=simple_residual_fn,
            initial_params=initial_params,
            bounds=simple_bounds,
            config=nlsq_config,
        )

        assert isinstance(result, MultiStartResult)
        assert result.n_total == 3
        assert result.best_result is not None

    @pytest.mark.unit
    def test_fit_selects_best_result(
        self,
        mock_adapter: MagicMock,
        nlsq_config: NLSQConfig,
        simple_residual_fn,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """fit() selects result with lowest cost."""
        # Configure mock to return different costs for each start
        results = [
            NLSQResult(
                parameters=np.array([1.0, 2.0, 3.0]),
                parameter_names=["a", "b", "c"],
                success=True,
                message="Converged",
                final_cost=0.5,  # Medium cost
            ),
            NLSQResult(
                parameters=np.array([1.1, 2.1, 3.1]),
                parameter_names=["a", "b", "c"],
                success=True,
                message="Converged",
                final_cost=0.1,  # Best cost
            ),
            NLSQResult(
                parameters=np.array([0.9, 1.9, 2.9]),
                parameter_names=["a", "b", "c"],
                success=True,
                message="Converged",
                final_cost=1.0,  # Worst cost
            ),
        ]
        mock_adapter.fit.side_effect = results

        optimizer = MultiStartOptimizer(mock_adapter, n_starts=3, seed=42)
        result = optimizer.fit(
            residual_fn=simple_residual_fn,
            initial_params=initial_params,
            bounds=simple_bounds,
            config=nlsq_config,
        )

        # Best result should be the one with lowest cost
        assert result.best_result.final_cost == 0.1
        assert_allclose(result.best_result.parameters, [1.1, 2.1, 3.1])

    @pytest.mark.unit
    def test_fit_counts_successful_runs(
        self,
        mock_adapter: MagicMock,
        nlsq_config: NLSQConfig,
        simple_residual_fn,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """fit() correctly counts successful runs."""
        # Mix of successful and failed results
        results = [
            NLSQResult(
                parameters=np.array([1.0, 2.0, 3.0]),
                parameter_names=["a", "b", "c"],
                success=True,
                message="Converged",
                final_cost=0.1,
            ),
            NLSQResult(
                parameters=np.array([0.0, 0.0, 0.0]),
                parameter_names=["a", "b", "c"],
                success=False,  # Failed
                message="Did not converge",
                final_cost=None,
            ),
            NLSQResult(
                parameters=np.array([1.1, 2.1, 3.1]),
                parameter_names=["a", "b", "c"],
                success=True,
                message="Converged",
                final_cost=0.2,
            ),
        ]
        mock_adapter.fit.side_effect = results

        optimizer = MultiStartOptimizer(mock_adapter, n_starts=3, seed=42)
        result = optimizer.fit(
            residual_fn=simple_residual_fn,
            initial_params=initial_params,
            bounds=simple_bounds,
            config=nlsq_config,
        )

        assert result.n_successful == 2
        assert result.n_total == 3

    @pytest.mark.unit
    def test_fit_uses_last_result_when_all_fail(
        self,
        mock_adapter: MagicMock,
        nlsq_config: NLSQConfig,
        simple_residual_fn,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """fit() uses last result when all runs fail."""
        # All failed results
        results = [
            NLSQResult(
                parameters=np.array([0.0, 0.0, 0.0]),
                parameter_names=["a", "b", "c"],
                success=False,
                message="Failed 1",
                final_cost=None,
            ),
            NLSQResult(
                parameters=np.array([1.0, 1.0, 1.0]),
                parameter_names=["a", "b", "c"],
                success=False,
                message="Failed 2",
                final_cost=None,
            ),
        ]
        mock_adapter.fit.side_effect = results

        optimizer = MultiStartOptimizer(mock_adapter, n_starts=2, seed=42)
        result = optimizer.fit(
            residual_fn=simple_residual_fn,
            initial_params=initial_params,
            bounds=simple_bounds,
            config=nlsq_config,
        )

        # Should use last result
        assert result.best_result.message == "Failed 2"
        assert result.n_successful == 0

    @pytest.mark.unit
    def test_fit_passes_jacobian_fn(
        self,
        mock_adapter: MagicMock,
        nlsq_config: NLSQConfig,
        simple_residual_fn,
        initial_params: np.ndarray,
        simple_bounds: tuple,
    ) -> None:
        """fit() passes jacobian_fn to adapter."""
        mock_result = NLSQResult(
            parameters=np.array([1.0, 2.0, 3.0]),
            parameter_names=["a", "b", "c"],
            success=True,
            message="Converged",
            final_cost=0.1,
        )
        mock_adapter.fit.return_value = mock_result

        def jacobian_fn(params):
            return np.eye(3)

        optimizer = MultiStartOptimizer(mock_adapter, n_starts=2, seed=42)
        optimizer.fit(
            residual_fn=simple_residual_fn,
            initial_params=initial_params,
            bounds=simple_bounds,
            config=nlsq_config,
            jacobian_fn=jacobian_fn,
        )

        # Check jacobian_fn was passed
        call_args = mock_adapter.fit.call_args_list[0]
        assert call_args.kwargs.get("jacobian_fn") is jacobian_fn


# ============================================================================
# MultiStartResult Tests
# ============================================================================


class TestMultiStartResult:
    """Tests for MultiStartResult dataclass."""

    @pytest.mark.unit
    def test_dataclass_fields(self) -> None:
        """MultiStartResult has expected fields."""
        best_result = NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["a"],
            success=True,
            message="OK",
        )
        result = MultiStartResult(
            best_result=best_result,
            all_results=[best_result],
            n_successful=1,
            n_total=1,
        )

        assert result.best_result is best_result
        assert result.all_results == [best_result]
        assert result.n_successful == 1
        assert result.n_total == 1
