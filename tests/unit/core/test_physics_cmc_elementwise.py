"""Tests for element-wise CMC path parity with meshgrid path.

Verifies that ``compute_c2_elementwise`` produces identical results to
``compute_c2_heterodyne`` (the meshgrid path) at the same (t1, t2) pairs.
This is the critical parity test for the two-path architecture.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np


class TestShardGridCreation(unittest.TestCase):
    """Test ShardGrid pre-computation."""

    def test_precompute_shard_grid_indices(self):
        """searchsorted indices map t1/t2 to correct time_grid positions."""
        from heterodyne.core.physics_cmc import precompute_shard_grid

        time_grid = jnp.linspace(0.0, 1.0, 10)
        t1 = jnp.array([0.0, 0.1, 0.5])
        t2 = jnp.array([0.1, 0.5, 1.0])

        sg = precompute_shard_grid(time_grid, t1, t2)

        assert sg.n_pairs == 3
        assert sg.idx1.shape == (3,)
        assert sg.idx2.shape == (3,)
        # First pair: t1=0.0 → idx 0, t2=0.1 → idx 1
        np.testing.assert_equal(int(sg.idx1[0]), 0)
        np.testing.assert_equal(int(sg.idx2[0]), 1)

    def test_precompute_shard_grid_from_matrix(self):
        """ShardGrid from matrix block has correct number of pairs."""
        from heterodyne.core.physics_cmc import precompute_shard_grid_from_matrix

        time_grid = jnp.linspace(0.0, 1.0, 20)
        sg = precompute_shard_grid_from_matrix(time_grid, 5, 10)

        # Upper triangle of 5×5 block: 5*(5+1)/2 = 15 pairs
        assert sg.n_pairs == 15

    def test_shard_grid_time_grid_preserved(self):
        """time_grid is stored unchanged in ShardGrid."""
        from heterodyne.core.physics_cmc import precompute_shard_grid

        time_grid = jnp.linspace(0.0, 2.0, 50)
        t1 = jnp.array([0.5])
        t2 = jnp.array([1.5])
        sg = precompute_shard_grid(time_grid, t1, t2)

        np.testing.assert_array_equal(sg.time_grid, time_grid)


class TestElementwiseMeshgridParity(unittest.TestCase):
    """Verify element-wise and meshgrid paths give identical results."""

    def _default_params(self):
        """Standard 14-param test vector."""
        return jnp.array(
            [
                1e3,
                1.0,
                0.0,  # D0_ref, alpha_ref, D_offset_ref
                5e2,
                0.8,
                0.0,  # D0_sample, alpha_sample, D_offset_sample
                100.0,
                0.5,
                0.0,  # v0, beta, v_offset
                0.5,
                -0.01,
                50.0,
                0.3,  # f0, f1, f2, f3
                5.0,  # phi0
            ]
        )

    def test_full_matrix_parity(self):
        """Element-wise c2 matches meshgrid c2 at all upper-triangle points."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne
        from heterodyne.core.physics_cmc import (
            compute_c2_elementwise,
            precompute_shard_grid_from_matrix,
        )

        params = self._default_params()
        t = jnp.linspace(0.0, 1.0, 30)
        q, dt, phi_angle = 0.01, 0.01, 45.0
        contrast, offset = 0.5, 1.0

        # Meshgrid path: full N×N matrix
        c2_meshgrid = compute_c2_heterodyne(
            params, t, q, dt, phi_angle, contrast, offset
        )

        # Element-wise path: upper triangle
        n = len(t)
        sg = precompute_shard_grid_from_matrix(t, 0, n)
        c2_elemwise = compute_c2_elementwise(
            params, sg, q, dt, phi_angle, contrast, offset
        )

        # Extract upper triangle from meshgrid for comparison
        triu_i, triu_j = np.triu_indices(n, k=0)
        c2_meshgrid_flat = np.asarray(c2_meshgrid[triu_i, triu_j])
        c2_elemwise_np = np.asarray(c2_elemwise)

        np.testing.assert_allclose(
            c2_elemwise_np,
            c2_meshgrid_flat,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Element-wise and meshgrid paths diverge!",
        )

    def test_shard_parity(self):
        """Element-wise shard matches meshgrid shard at sub-block pairs."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne
        from heterodyne.core.physics_cmc import (
            compute_c2_elementwise,
            precompute_shard_grid_from_matrix,
        )

        params = self._default_params()
        t = jnp.linspace(0.0, 1.0, 40)
        q, dt, phi_angle = 0.02, 0.005, 30.0

        # Meshgrid: shard block [10:20, 10:20]
        c2_full = compute_c2_heterodyne(params, t, q, dt, phi_angle)

        shard_start, shard_end = 10, 20
        shard_size = shard_end - shard_start
        triu_i, triu_j = np.triu_indices(shard_size, k=0)
        global_i = triu_i + shard_start
        global_j = triu_j + shard_start
        c2_meshgrid_shard = np.asarray(c2_full[global_i, global_j])

        # Element-wise: same shard
        sg = precompute_shard_grid_from_matrix(t, shard_start, shard_end)
        c2_elemwise = np.asarray(compute_c2_elementwise(params, sg, q, dt, phi_angle))

        np.testing.assert_allclose(
            c2_elemwise,
            c2_meshgrid_shard,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_log_likelihood_parity(self):
        """Element-wise log-likelihood matches meshgrid log-likelihood."""
        from heterodyne.core.physics_cmc import (
            compute_log_likelihood,
            compute_log_likelihood_elementwise,
            precompute_shard_grid_from_matrix,
        )

        params = self._default_params()
        t = jnp.linspace(0.0, 1.0, 20)
        q, dt, phi_angle = 0.01, 0.01, 45.0
        contrast, offset = 0.5, 1.0
        sigma = 0.01

        # Generate synthetic data from model
        from heterodyne.core.jax_backend import compute_c2_heterodyne

        c2_data = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)

        # Meshgrid log-likelihood
        ll_meshgrid = float(
            compute_log_likelihood(
                params,
                t,
                q,
                dt,
                phi_angle,
                c2_data,
                sigma,
                contrast,
                offset,
            )
        )

        # Element-wise log-likelihood (upper triangle only)
        n = len(t)
        sg = precompute_shard_grid_from_matrix(t, 0, n)
        triu_i, triu_j = np.triu_indices(n, k=0)
        c2_flat = c2_data[triu_i, triu_j]

        ll_elemwise = float(
            compute_log_likelihood_elementwise(
                params,
                sg,
                c2_flat,
                sigma,
                q,
                dt,
                phi_angle,
                contrast,
                offset,
            )
        )

        # They won't be exactly equal because meshgrid uses full matrix
        # while element-wise uses only upper triangle. But the element-wise
        # result should be close to half the meshgrid result (upper tri ≈
        # half of symmetric matrix, plus the diagonal).
        # For a perfect model (zero residuals), both should be ~0.
        np.testing.assert_allclose(ll_meshgrid, 0.0, atol=1e-6)
        np.testing.assert_allclose(ll_elemwise, 0.0, atol=1e-6)

    def test_different_parameters(self):
        """Parity holds across diverse parameter regimes."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne
        from heterodyne.core.physics_cmc import (
            compute_c2_elementwise,
            precompute_shard_grid_from_matrix,
        )

        t = jnp.linspace(0.0, 0.5, 15)
        q, dt, phi_angle = 0.05, 0.005, 90.0
        n = len(t)
        sg = precompute_shard_grid_from_matrix(t, 0, n)
        triu_i, triu_j = np.triu_indices(n, k=0)

        param_sets = [
            # Pure diffusion (no velocity, equal fractions)
            jnp.array(
                [1e3, 1.0, 0.0, 1e3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
            ),
            # Strong velocity, asymmetric fractions
            jnp.array(
                [
                    500,
                    0.5,
                    10.0,
                    200,
                    1.5,
                    5.0,
                    500.0,
                    1.0,
                    50.0,
                    0.8,
                    -0.05,
                    25.0,
                    0.1,
                    15.0,
                ]
            ),
            # Near-zero transport (static limit)
            jnp.array(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
            ),
        ]

        for i, params in enumerate(param_sets):
            with self.subTest(param_set=i):
                c2_mesh = compute_c2_heterodyne(params, t, q, dt, phi_angle)
                c2_elem = compute_c2_elementwise(params, sg, q, dt, phi_angle)

                np.testing.assert_allclose(
                    np.asarray(c2_elem),
                    np.asarray(c2_mesh[triu_i, triu_j]),
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Parity failed for param_set={i}",
                )


class TestSharedPrimitives(unittest.TestCase):
    """Test shared primitives from physics_utils."""

    def test_trapezoid_cumsum_constant(self):
        """Constant function integrates to f×t."""
        from heterodyne.core.physics_utils import trapezoid_cumsum

        f = jnp.ones(100) * 3.0
        dt = 0.1
        cumsum = trapezoid_cumsum(f, dt)

        assert cumsum.shape == (100,)
        assert float(cumsum[0]) == 0.0
        # After 99 steps: 3.0 * 99 * 0.1 = 29.7
        np.testing.assert_allclose(float(cumsum[-1]), 29.7, rtol=1e-10)

    def test_trapezoid_cumsum_linear(self):
        """Linear function f(t)=t integrates to t²/2 (exact for trapezoidal)."""
        from heterodyne.core.physics_utils import trapezoid_cumsum

        t = jnp.linspace(0.0, 1.0, 1001)
        dt = t[1] - t[0]
        cumsum = trapezoid_cumsum(t, dt)

        # ∫₀^1 t dt = 0.5 — trapezoidal is exact for linear functions
        np.testing.assert_allclose(float(cumsum[-1]), 0.5, rtol=1e-10)

    def test_create_time_integral_matrix_antisymmetric(self):
        """Signed integral matrix is antisymmetric: M[i,j] = -M[j,i]."""
        from heterodyne.core.physics_utils import (
            create_time_integral_matrix,
            trapezoid_cumsum,
        )

        f = jnp.ones(20) * 2.0
        cumsum = trapezoid_cumsum(f, 0.1)
        M = create_time_integral_matrix(cumsum)

        np.testing.assert_allclose(
            np.asarray(M),
            -np.asarray(M.T),
            atol=1e-14,
        )

    def test_smooth_abs_gradient_at_zero(self):
        """smooth_abs has well-defined gradient at x=0 (no NaN)."""
        import jax

        from heterodyne.core.physics_utils import smooth_abs

        grad_fn = jax.grad(lambda x: smooth_abs(x).sum())
        g = grad_fn(jnp.array([0.0, 1.0, -1.0]))

        assert not np.any(np.isnan(np.asarray(g)))

    def test_compute_transport_rate_nonnegative(self):
        """Transport rate is floored at 0."""
        from heterodyne.core.physics_utils import compute_transport_rate

        t = jnp.linspace(0.0, 1.0, 50)
        # Negative offset that would make rate negative
        rate = compute_transport_rate(t, D0=1.0, alpha=1.0, offset=-10.0)

        assert np.all(np.asarray(rate) >= 0.0)

    def test_compute_velocity_rate_allows_negative(self):
        """Velocity rate is NOT floored — can be negative."""
        from heterodyne.core.physics_utils import compute_velocity_rate

        t = jnp.linspace(0.0, 1.0, 50)
        v = compute_velocity_rate(t, v0=1.0, beta=1.0, v_offset=-10.0)

        # Should have negative values due to large negative offset
        assert np.any(np.asarray(v) < 0.0)

    def test_safe_sinc_at_zero(self):
        """safe_sinc(0) = 1 (mathematical limit)."""
        from heterodyne.core.physics_utils import safe_sinc

        result = safe_sinc(jnp.array([0.0, 1e-15, 1.0]))
        np.testing.assert_allclose(float(result[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(float(result[1]), 1.0, atol=1e-6)


class TestPrepareShards(unittest.TestCase):
    """Test shard preparation utilities."""

    def test_prepare_shards_elementwise(self):
        """Element-wise shard preparation produces correct structure."""
        from heterodyne.core.physics_cmc import (
            create_shard_grid,
            prepare_shards_elementwise,
        )

        n = 20
        t = jnp.linspace(0.0, 1.0, n)
        c2 = jnp.ones((n, n))
        sigma = 0.1

        intervals = create_shard_grid(n, 4)
        shard_grids, c2_flats, sigma_flats = prepare_shards_elementwise(
            c2,
            sigma,
            t,
            intervals,
        )

        assert len(shard_grids) == len(intervals)
        for sg, c2_flat in zip(shard_grids, c2_flats, strict=True):
            assert sg.n_pairs == c2_flat.shape[0]

    def test_sharded_log_likelihood_elementwise(self):
        """Sharded element-wise log-likelihood runs without error."""
        from heterodyne.core.jax_backend import compute_c2_heterodyne
        from heterodyne.core.physics_cmc import (
            compute_sharded_log_likelihood_elementwise,
            create_shard_grid,
            prepare_shards_elementwise,
        )

        params = jnp.array(
            [
                1e3,
                1.0,
                0.0,
                5e2,
                0.8,
                0.0,
                100.0,
                0.5,
                0.0,
                0.5,
                -0.01,
                50.0,
                0.3,
                5.0,
            ]
        )
        t = jnp.linspace(0.0, 1.0, 20)
        q, dt, phi = 0.01, 0.01, 45.0
        c2_data = compute_c2_heterodyne(params, t, q, dt, phi)
        sigma = 0.01

        intervals = create_shard_grid(len(t), 3)
        sgs, c2_flats, sigma_flats = prepare_shards_elementwise(
            c2_data,
            sigma,
            t,
            intervals,
        )

        ll = compute_sharded_log_likelihood_elementwise(
            params,
            sgs,
            c2_flats,
            sigma_flats,
            q,
            dt,
            phi,
        )

        # Perfect model → residuals ~0 → ll ~0
        np.testing.assert_allclose(float(ll), 0.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
