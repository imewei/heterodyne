"""Tests for CMC sharding: ShardGrid, precompute_shard_grid, compute_c2_elementwise.

Covers:
- ShardGrid creation via precompute_shard_grid
- ShardGrid has correct n_pairs count
- ShardGrid indices are valid (within array bounds)
- compute_c2_elementwise produces finite output
- Shard grid with different time grid sizes
- Shard grid indices match expected time pairs
- CMCConfig sharding fields: num_shards, sharding_strategy, min_points_per_shard
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.core.physics_cmc import (
    ShardGrid,
    compute_c2_elementwise,
    precompute_shard_grid,
    precompute_shard_grid_from_matrix,
)
from heterodyne.optimization.cmc.config import CMCConfig
from tests.factories.config_factory import make_cmc_config

# ============================================================================
# Helpers
# ============================================================================


def _make_time_grid(n: int, dt: float = 0.1) -> jnp.ndarray:
    """Create a simple linear time grid."""
    return jnp.arange(n, dtype=jnp.float64) * dt


def _make_default_params() -> jnp.ndarray:
    """Return a 14-parameter array with physically reasonable defaults."""
    return jnp.array(
        [
            1e4,  # D0_ref
            0.5,  # alpha_ref
            0.0,  # D_offset_ref
            1e4,  # D0_sample
            0.5,  # alpha_sample
            0.0,  # D_offset_sample
            1e3,  # v0
            0.5,  # beta
            0.0,  # v_offset
            0.5,  # f0
            0.01,  # f1
            0.5,  # f2
            0.0,  # f3
            0.0,  # phi0
        ],
        dtype=jnp.float64,
    )


# ============================================================================
# ShardGrid creation
# ============================================================================


@pytest.mark.unit
class TestShardGridCreation:
    """Tests for precompute_shard_grid and ShardGrid structure."""

    def test_basic_creation(self) -> None:
        """ShardGrid is created successfully from valid inputs."""
        time_grid = _make_time_grid(10)
        t1 = jnp.array([0.0, 0.1, 0.2])
        t2 = jnp.array([0.3, 0.5, 0.9])

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert isinstance(sg, ShardGrid)
        assert sg.n_pairs == 3

    def test_n_pairs_matches_input(self) -> None:
        """n_pairs equals the number of (t1, t2) pairs provided."""
        time_grid = _make_time_grid(20)
        n = 15
        t1 = jnp.linspace(0.0, 1.0, n)
        t2 = jnp.linspace(0.5, 1.5, n)

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert sg.n_pairs == n

    def test_indices_within_bounds(self) -> None:
        """idx1 and idx2 are within valid range [0, N-1]."""
        n_grid = 50
        time_grid = _make_time_grid(n_grid)
        # Include times beyond grid edges to test clipping
        t1 = jnp.array([-0.5, 0.0, 2.5, 10.0])
        t2 = jnp.array([0.0, 1.0, 3.0, 20.0])

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert jnp.all(sg.idx1 >= 0)
        assert jnp.all(sg.idx1 < n_grid)
        assert jnp.all(sg.idx2 >= 0)
        assert jnp.all(sg.idx2 < n_grid)

    def test_time_grid_preserved(self) -> None:
        """ShardGrid stores the original time_grid."""
        time_grid = _make_time_grid(10)
        t1 = jnp.array([0.1])
        t2 = jnp.array([0.5])

        sg = precompute_shard_grid(time_grid, t1, t2)

        np.testing.assert_allclose(sg.time_grid, time_grid)

    def test_indices_match_searchsorted(self) -> None:
        """Indices match expected searchsorted+clip results."""
        time_grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        t1 = jnp.array([0.5, 2.5])
        t2 = jnp.array([1.5, 3.5])

        sg = precompute_shard_grid(time_grid, t1, t2)

        # searchsorted gives insertion point, clipped to [0, N-1]
        expected_idx1 = jnp.clip(jnp.searchsorted(time_grid, t1), 0, 4)
        expected_idx2 = jnp.clip(jnp.searchsorted(time_grid, t2), 0, 4)

        np.testing.assert_array_equal(sg.idx1, expected_idx1)
        np.testing.assert_array_equal(sg.idx2, expected_idx2)


# ============================================================================
# ShardGrid from matrix
# ============================================================================


@pytest.mark.unit
class TestShardGridFromMatrix:
    """Tests for precompute_shard_grid_from_matrix."""

    def test_upper_triangle_pairs(self) -> None:
        """From-matrix creates upper-triangle pairs for a shard block."""
        time_grid = _make_time_grid(10)
        shard_start, shard_end = 2, 5
        shard_size = shard_end - shard_start

        sg = precompute_shard_grid_from_matrix(time_grid, shard_start, shard_end)

        # Upper triangle of a 3x3 block has 3+2+1 = 6 pairs
        expected_n_pairs = shard_size * (shard_size + 1) // 2
        assert sg.n_pairs == expected_n_pairs

    def test_full_grid_as_single_shard(self) -> None:
        """Using entire time grid as one shard."""
        n = 8
        time_grid = _make_time_grid(n)

        sg = precompute_shard_grid_from_matrix(time_grid, 0, n)

        expected_n_pairs = n * (n + 1) // 2
        assert sg.n_pairs == expected_n_pairs


# ============================================================================
# Different time grid sizes
# ============================================================================


@pytest.mark.unit
class TestDifferentGridSizes:
    """ShardGrid with various time grid sizes."""

    @pytest.mark.parametrize("n_grid", [5, 10, 50, 100])
    def test_various_grid_sizes(self, n_grid: int) -> None:
        """ShardGrid works with different grid sizes."""
        time_grid = _make_time_grid(n_grid)
        n_pairs = min(n_grid, 20)
        rng = np.random.default_rng(42)
        t_vals = rng.choice(np.asarray(time_grid), size=n_pairs, replace=True)
        t1 = jnp.array(sorted(t_vals[:n_pairs]))
        t2 = jnp.array(sorted(t_vals[:n_pairs]))

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert sg.n_pairs == n_pairs
        assert jnp.all(sg.idx1 >= 0)
        assert jnp.all(sg.idx1 < n_grid)
        assert jnp.all(sg.idx2 >= 0)
        assert jnp.all(sg.idx2 < n_grid)

    def test_single_point_grid(self) -> None:
        """Grid with a single time point."""
        time_grid = jnp.array([1.0])
        t1 = jnp.array([1.0])
        t2 = jnp.array([1.0])

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert sg.n_pairs == 1
        assert int(sg.idx1[0]) == 0
        assert int(sg.idx2[0]) == 0


# ============================================================================
# compute_c2_elementwise
# ============================================================================


@pytest.mark.unit
class TestComputeC2Elementwise:
    """Tests for compute_c2_elementwise."""

    def test_output_finite(self) -> None:
        """compute_c2_elementwise produces all-finite output."""
        n = 20
        time_grid = _make_time_grid(n, dt=0.1)
        sg = precompute_shard_grid_from_matrix(time_grid, 0, n)
        params = _make_default_params()

        c2 = compute_c2_elementwise(
            params,
            sg,
            q=0.01,
            dt=0.1,
            phi_angle=0.0,
            contrast=0.5,
            offset=1.0,
        )

        assert c2.shape == (sg.n_pairs,)
        assert jnp.all(jnp.isfinite(c2))

    def test_output_shape_matches_n_pairs(self) -> None:
        """Output length matches n_pairs in the ShardGrid."""
        n = 15
        time_grid = _make_time_grid(n, dt=0.05)
        sg = precompute_shard_grid_from_matrix(time_grid, 0, n)
        params = _make_default_params()

        c2 = compute_c2_elementwise(
            params,
            sg,
            q=0.01,
            dt=0.05,
            phi_angle=0.0,
        )

        expected_pairs = n * (n + 1) // 2
        assert c2.shape == (expected_pairs,)

    def test_contrast_zero_returns_offset(self) -> None:
        """With contrast=0, c2 should equal the offset everywhere."""
        n = 10
        time_grid = _make_time_grid(n, dt=0.1)
        sg = precompute_shard_grid_from_matrix(time_grid, 0, n)
        params = _make_default_params()

        offset_val = 1.5
        c2 = compute_c2_elementwise(
            params,
            sg,
            q=0.01,
            dt=0.1,
            phi_angle=0.0,
            contrast=0.0,
            offset=offset_val,
        )

        np.testing.assert_allclose(c2, offset_val, atol=1e-10)

    def test_different_phi_angles(self) -> None:
        """Different phi angles produce different results."""
        n = 10
        time_grid = _make_time_grid(n, dt=0.1)
        sg = precompute_shard_grid_from_matrix(time_grid, 0, n)
        params = _make_default_params()

        c2_phi0 = compute_c2_elementwise(
            params,
            sg,
            q=0.01,
            dt=0.1,
            phi_angle=0.0,
        )
        c2_phi45 = compute_c2_elementwise(
            params,
            sg,
            q=0.01,
            dt=0.1,
            phi_angle=45.0,
        )

        # Different phi angles should generally produce different c2
        assert not jnp.allclose(c2_phi0, c2_phi45)


# ============================================================================
# CMCConfig sharding fields
# ============================================================================


@pytest.mark.unit
class TestCMCConfigShardingFields:
    """Tests for CMCConfig sharding-related fields."""

    def test_default_num_shards(self) -> None:
        """Default num_shards is 'auto'."""
        config = CMCConfig()
        assert config.num_shards == "auto"

    def test_explicit_num_shards(self) -> None:
        """Can set num_shards to an explicit integer."""
        config = make_cmc_config(num_shards=4)
        assert config.num_shards == 4

    def test_invalid_num_shards_detected(self) -> None:
        """num_shards=0 is caught by validate()."""
        config = make_cmc_config(num_shards=0)
        errors = config.validate()
        assert any("num_shards" in e for e in errors)

    def test_default_sharding_strategy(self) -> None:
        """Default sharding_strategy is 'random'."""
        config = CMCConfig()
        assert config.sharding_strategy == "random"

    def test_valid_sharding_strategies(self) -> None:
        """All valid sharding strategies are accepted."""
        for strategy in ["stratified", "random", "contiguous"]:
            config = make_cmc_config(sharding_strategy=strategy)
            assert config.sharding_strategy == strategy

    def test_invalid_sharding_strategy_detected(self) -> None:
        """Invalid sharding_strategy is caught by validate()."""
        config = make_cmc_config(sharding_strategy="invalid_strategy")
        errors = config.validate()
        assert any("sharding_strategy" in e for e in errors)

    def test_default_min_points_per_shard(self) -> None:
        """Default min_points_per_shard is 10_000."""
        config = CMCConfig()
        assert config.min_points_per_shard == 10_000

    def test_custom_min_points_per_shard(self) -> None:
        """Can set min_points_per_shard to a custom value."""
        config = make_cmc_config(min_points_per_shard=5000)
        assert config.min_points_per_shard == 5000

    def test_num_shards_negative_detected(self) -> None:
        """Negative num_shards is caught by validate()."""
        config = make_cmc_config(num_shards=-1)
        errors = config.validate()
        assert any("num_shards" in e for e in errors)

    def test_num_shards_invalid_string_detected(self) -> None:
        """Non-'auto' string for num_shards is caught by validate()."""
        config = make_cmc_config(num_shards="manual")
        errors = config.validate()
        assert any("num_shards" in e for e in errors)
