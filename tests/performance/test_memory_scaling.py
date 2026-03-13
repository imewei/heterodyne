"""Memory scaling benchmark tests for heterodyne XPCS package.

Verifies that:
- N x N matrix operations scale quadratically with input size.
- The memory-aware NLSQ strategy selector responds correctly to
  different dataset scales.
- The adapter cache stores and releases entries properly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.core.physics_utils import create_time_integral_matrix, trapezoid_cumsum
from heterodyne.optimization.nlsq.adapter import (
    clear_model_cache,
    get_cache_stats,
    get_or_create_fitter,
)
from heterodyne.optimization.nlsq.memory import (
    NLSQStrategy,
    StrategyDecision,
    select_nlsq_strategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cumsum(n: int) -> jnp.ndarray:
    """Build a cumulative-sum vector of length *n* from a simple rate."""
    t = jnp.linspace(0.0, 1.0, n)
    dt = float(t[1] - t[0])
    rate = jnp.ones(n)  # constant unit rate
    return trapezoid_cumsum(rate, dt)


def _default_params() -> jnp.ndarray:
    """14-element canonical parameter vector with safe defaults."""
    return jnp.array(
        [
            1e4,
            1.0,
            0.0,  # D0_ref, alpha_ref, D_offset_ref
            1e4,
            1.0,
            0.0,  # D0_sample, alpha_sample, D_offset_sample
            1e3,
            1.0,
            0.0,  # v0, beta, v_offset
            0.5,
            0.0,
            0.0,
            0.5,  # f0, f1, f2, f3
            0.0,  # phi0
        ]
    )


# ===========================================================================
# TestMatrixMemoryScaling
# ===========================================================================


@pytest.mark.performance
class TestMatrixMemoryScaling:
    """Verify N x N matrix memory footprint scales quadratically."""

    sizes = [16, 32, 64, 128]

    def test_time_integral_matrix_scales_quadratically(self) -> None:
        """Output shape is (n, n) and memory ratio ~4x between doubling sizes."""
        nbytes_list: list[int] = []
        for n in self.sizes:
            cumsum = _make_cumsum(n)
            matrix = create_time_integral_matrix(cumsum)
            assert matrix.shape == (n, n)
            nbytes_list.append(int(np.asarray(matrix).nbytes))

        # Between consecutive doublings the ratio should be ~4x.
        for i in range(len(self.sizes) - 1):
            ratio = nbytes_list[i + 1] / nbytes_list[i]
            expected = (self.sizes[i + 1] / self.sizes[i]) ** 2
            assert ratio == pytest.approx(expected, rel=0.01), (
                f"Memory ratio {self.sizes[i]}->{self.sizes[i + 1]}: "
                f"got {ratio:.2f}, expected {expected:.2f}"
            )

    @pytest.mark.parametrize("n_times", [8, 16, 32, 64])
    def test_c2_output_size_matches_input(self, n_times: int) -> None:
        """compute_c2_heterodyne returns (n_times, n_times) matrix."""
        params = _default_params()
        t = jnp.linspace(0.0, 1.0, n_times)
        dt = float(t[1] - t[0])
        q = 0.01
        phi_angle = 0.0

        c2 = compute_c2_heterodyne(params, t, q, dt, phi_angle)
        assert c2.shape == (n_times, n_times)


# ===========================================================================
# TestMemoryStrategySelection
# ===========================================================================


@pytest.mark.performance
class TestMemoryStrategySelection:
    """Verify strategy selection logic responds to dataset scale."""

    def test_small_dataset_selects_standard(self) -> None:
        """Small problems (100 points x 14 params) should use STANDARD."""
        decision = select_nlsq_strategy(n_points=100, n_params=14)
        assert decision.strategy is NLSQStrategy.STANDARD

    def test_large_dataset_selects_large_or_streaming(self) -> None:
        """Very large problems should NOT select STANDARD."""
        # 1 billion points x 14 params — clearly exceeds any threshold.
        decision = select_nlsq_strategy(n_points=1_000_000_000, n_params=14)
        assert decision.strategy is not NLSQStrategy.STANDARD

    def test_strategy_decision_has_required_fields(self) -> None:
        """StrategyDecision exposes strategy, threshold_gb, peak_memory_gb, reason."""
        decision = select_nlsq_strategy(n_points=50, n_params=10)
        assert isinstance(decision, StrategyDecision)
        assert isinstance(decision.strategy, NLSQStrategy)
        assert isinstance(decision.threshold_gb, float)
        assert isinstance(decision.peak_memory_gb, float)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_peak_memory_estimate_scales_with_data(self) -> None:
        """Larger datasets produce higher peak_memory_gb estimates."""
        small = select_nlsq_strategy(n_points=100, n_params=14)
        large = select_nlsq_strategy(n_points=10_000, n_params=14)
        assert large.peak_memory_gb > small.peak_memory_gb


# ===========================================================================
# TestNLSQCacheMemory
# ===========================================================================


@pytest.mark.performance
class TestNLSQCacheMemory:
    """Verify adapter cache stores entries and releases them on clear."""

    def setup_method(self) -> None:
        """Start each test with a clean cache."""
        clear_model_cache()

    def teardown_method(self) -> None:
        """Restore clean state after each test."""
        clear_model_cache()

    def test_cache_stores_entries(self) -> None:
        """After get_or_create_fitter, cache size is non-zero."""
        _fitter, cache_hit = get_or_create_fitter(
            n_data=50,
            n_params=14,
            phi_angles=(0.0,),
            scaling_mode="auto",
        )
        assert not cache_hit
        stats = get_cache_stats()
        assert stats["size"] > 0
        assert stats["misses"] > 0

    def test_clear_cache_releases(self) -> None:
        """After clear_model_cache(), cache stats show zero entries."""
        get_or_create_fitter(
            n_data=50,
            n_params=14,
            phi_angles=(0.0,),
            scaling_mode="auto",
        )
        assert get_cache_stats()["size"] > 0

        clear_model_cache()
        stats = get_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
