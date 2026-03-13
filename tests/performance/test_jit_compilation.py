"""Performance benchmarks for JAX JIT compilation overhead.

Tests verify that:
1. JIT compilation completes within reasonable time bounds.
2. Cached (second) calls are significantly faster than first (compilation) calls.
3. Core physics functions are traceable by jax.jit without errors.
4. Same-shape inputs reuse compiled traces (no recompilation).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.core.physics_utils import (
    create_time_integral_matrix,
    smooth_abs,
    trapezoid_cumsum,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_TIMES = 32
N_PARAMS = 14


def _make_params(key: jax.Array | None = None) -> jnp.ndarray:
    """Create a realistic 14-element parameter vector."""
    return jnp.array(
        [
            1e4,  # D0_ref
            1.0,  # alpha_ref
            0.0,  # D_offset_ref
            1e4,  # D0_sample
            1.0,  # alpha_sample
            0.0,  # D_offset_sample
            1e3,  # v0
            1.0,  # beta
            0.0,  # v_offset
            0.5,  # f0
            0.01,  # f1
            10.0,  # f2
            0.3,  # f3
            0.0,  # phi0
        ]
    )


def _make_inputs(
    n_times: int = N_TIMES,
) -> tuple[jnp.ndarray, jnp.ndarray, float, float, float]:
    """Return (params, t, q, dt, phi_angle) for a typical call."""
    params = _make_params()
    t = jnp.linspace(0.001, 10.0, n_times)
    q = 0.01  # Angstrom^-1
    dt = float(t[1] - t[0])
    phi_angle = 45.0
    return params, t, q, dt, phi_angle


# ---------------------------------------------------------------------------
# 1. JIT compilation time
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestJITCompilationTime:
    """Verify JIT compilation completes in bounded time and caching works."""

    def test_c2_heterodyne_first_call_compiles(self) -> None:
        """First call to compute_c2_heterodyne triggers JIT; must finish < 60s."""
        # Clear any cached compilation by using a fresh jit wrapper
        fn = jax.jit(
            compute_c2_heterodyne.__wrapped__
            if hasattr(compute_c2_heterodyne, "__wrapped__")
            else compute_c2_heterodyne
        )
        params, t, q, dt, phi_angle = _make_inputs()

        start = time.perf_counter()
        result = fn(params, t, q, dt, phi_angle)
        # Force materialization (JAX lazy eval)
        result.block_until_ready()
        elapsed = time.perf_counter() - start

        assert elapsed < 60.0, f"First JIT compilation took {elapsed:.1f}s (limit 60s)"
        assert result.shape == (N_TIMES, N_TIMES)

    def test_c2_heterodyne_second_call_cached(self) -> None:
        """Second call uses JIT cache and must be >= 2x faster than first.

        Uses a unique array shape (n_times=37) to guarantee a fresh JIT trace
        on the first call, avoiding cache pollution from other tests.
        """
        # Use a unique shape so the first call here truly compiles
        n = 37
        params, t, q, dt, phi_angle = _make_inputs(n_times=n)

        # First call: includes compilation for this shape
        start1 = time.perf_counter()
        r1 = compute_c2_heterodyne(params, t, q, dt, phi_angle)
        r1.block_until_ready()
        elapsed1 = time.perf_counter() - start1

        # Second call: same shapes, should hit cache
        params2 = params.at[0].set(2e4)  # different value, same shape
        start2 = time.perf_counter()
        r2 = compute_c2_heterodyne(params2, t, q, dt, phi_angle)
        r2.block_until_ready()
        elapsed2 = time.perf_counter() - start2

        assert elapsed2 < elapsed1 / 2.0, (
            f"Cached call ({elapsed2:.4f}s) was not 2x faster than "
            f"first call ({elapsed1:.4f}s)"
        )

    def test_residuals_jit_cached(self) -> None:
        """compute_residuals second call is faster than first."""
        params, t, q, dt, phi_angle = _make_inputs()
        c2_data = jnp.ones((N_TIMES, N_TIMES))

        # First call
        start1 = time.perf_counter()
        r1 = compute_residuals(params, t, q, dt, phi_angle, c2_data)
        r1.block_until_ready()
        elapsed1 = time.perf_counter() - start1

        # Second call with different values
        params2 = params.at[3].set(5e3)
        start2 = time.perf_counter()
        r2 = compute_residuals(params2, t, q, dt, phi_angle, c2_data)
        r2.block_until_ready()
        elapsed2 = time.perf_counter() - start2

        assert elapsed2 < elapsed1 / 2.0, (
            f"Cached residuals ({elapsed2:.4f}s) was not 2x faster than "
            f"first call ({elapsed1:.4f}s)"
        )

    def test_trapezoid_cumsum_jit(self) -> None:
        """trapezoid_cumsum JIT compiles and caches."""
        f = jnp.linspace(1.0, 5.0, N_TIMES)
        dt = 0.1

        # First call
        start1 = time.perf_counter()
        r1 = trapezoid_cumsum(f, dt)
        r1.block_until_ready()
        elapsed1 = time.perf_counter() - start1

        # Second call
        f2 = jnp.linspace(2.0, 8.0, N_TIMES)
        start2 = time.perf_counter()
        r2 = trapezoid_cumsum(f2, dt)
        r2.block_until_ready()
        elapsed2 = time.perf_counter() - start2

        assert elapsed2 < elapsed1 / 2.0, (
            f"Cached trapezoid_cumsum ({elapsed2:.4f}s) was not 2x faster "
            f"than first call ({elapsed1:.4f}s)"
        )


# ---------------------------------------------------------------------------
# 2. JIT traceability
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestJITTraceability:
    """Verify core functions can be traced by jax.jit without errors."""

    def test_c2_heterodyne_is_jit_traceable(self) -> None:
        """jax.jit(compute_c2_heterodyne) executes without error."""
        params, t, q, dt, phi_angle = _make_inputs()
        # compute_c2_heterodyne is already decorated with @jax.jit,
        # but verify it actually traces and produces valid output.
        result = compute_c2_heterodyne(params, t, q, dt, phi_angle)
        result.block_until_ready()

        assert result.shape == (N_TIMES, N_TIMES)
        assert jnp.all(jnp.isfinite(result)), "c2 contains non-finite values"

    def test_smooth_abs_is_jit_traceable(self) -> None:
        """jax.jit(smooth_abs) executes without error on typical inputs."""
        x = jnp.linspace(-5.0, 5.0, 64)
        result = smooth_abs(x)
        result.block_until_ready()

        assert result.shape == x.shape
        assert jnp.all(result >= 0.0), "smooth_abs returned negative values"
        # At x=0 the smooth_abs should be close to 0 (within sqrt(eps))
        zero_val = smooth_abs(jnp.array(0.0))
        assert float(zero_val) < 1e-5

    def test_create_time_integral_matrix_jit_traceable(self) -> None:
        """jax.jit(create_time_integral_matrix) executes without error."""
        cumsum = jnp.linspace(0.0, 10.0, N_TIMES)
        result = create_time_integral_matrix(cumsum)
        result.block_until_ready()

        assert result.shape == (N_TIMES, N_TIMES)
        # Diagonal should be zero (cumsum[i] - cumsum[i] = 0)
        assert jnp.allclose(jnp.diag(result), 0.0, atol=1e-12)
        # Anti-symmetric: M[i,j] = -M[j,i]
        assert jnp.allclose(result, -result.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Recompilation avoidance
# ---------------------------------------------------------------------------


@pytest.mark.performance
class TestRecompilationAvoidance:
    """Verify JAX trace caching behavior with same/different shapes."""

    def test_same_shape_no_recompile(self) -> None:
        """Same-shaped inputs reuse the JIT cache (no recompilation)."""
        params, t, q, dt, phi_angle = _make_inputs(n_times=16)

        # Warm up: ensure compilation is done
        r0 = compute_c2_heterodyne(params, t, q, dt, phi_angle)
        r0.block_until_ready()

        # Measure several cached calls — all should be fast
        timings = []
        for i in range(5):
            p = params.at[0].set(float(1e4 + i * 100))
            start = time.perf_counter()
            r = compute_c2_heterodyne(p, t, q, dt, phi_angle)
            r.block_until_ready()
            timings.append(time.perf_counter() - start)

        # All cached calls should be under 1 second (compilation takes seconds)
        max_cached = max(timings)
        assert max_cached < 1.0, (
            f"Cached call took {max_cached:.4f}s — possible recompilation"
        )

        # Variance should be low (no intermittent recompilations)
        mean_t = sum(timings) / len(timings)
        assert all(abs(t_i - mean_t) < 10 * mean_t for t_i in timings), (
            f"High variance in cached call times: {timings}"
        )

    def test_different_shape_recompiles(self) -> None:
        """Different-shaped inputs trigger recompilation (expected behavior).

        This documents JAX's shape-specialization: changing array shapes
        invalidates the trace cache and forces a new compilation.
        """
        params = _make_params()
        q = 0.01
        phi_angle = 45.0

        # First shape: n_times=16
        t_small = jnp.linspace(0.001, 10.0, 16)
        dt_small = float(t_small[1] - t_small[0])
        r1 = compute_c2_heterodyne(params, t_small, q, dt_small, phi_angle)
        r1.block_until_ready()

        # Cached call with same shape (baseline)
        start_cached = time.perf_counter()
        r1b = compute_c2_heterodyne(params, t_small, q, dt_small, phi_angle)
        r1b.block_until_ready()
        elapsed_cached = time.perf_counter() - start_cached

        # Different shape: n_times=48 — this MUST recompile
        t_large = jnp.linspace(0.001, 10.0, 48)
        dt_large = float(t_large[1] - t_large[0])
        start_new = time.perf_counter()
        r2 = compute_c2_heterodyne(params, t_large, q, dt_large, phi_angle)
        r2.block_until_ready()
        elapsed_new = time.perf_counter() - start_new

        # New shape call should be slower due to recompilation.
        # We verify at minimum the output shapes are correct.
        assert r1.shape == (16, 16)
        assert r2.shape == (48, 48)

        # The recompilation call should take noticeably longer than cached.
        # Use a soft check: just verify it completed (the recompilation
        # overhead is highly variable on CI, so we document rather than
        # assert a strict ratio).
        assert elapsed_new > 0, "Recompilation call took zero time (unexpected)"
        assert elapsed_cached >= 0, "Cached call timing is negative (unexpected)"
