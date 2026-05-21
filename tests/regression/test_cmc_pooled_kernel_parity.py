"""Numerical parity between vmap+gather and pooled c2 kernels (Phase 4).

Phase 4 of the joint multi-phi CMC refactor replaces the
``compute_c2_heterodyne_multiphi`` (vmap-over-phi-then-gather) hot path
in ``_heterodyne_pooled_likelihood`` with a true pooled-data physics
function (``compute_c2_heterodyne_pooled``) that computes c2 directly at
the pooled ``(phi, t1, t2)`` points without ever materializing the
``(n_phi, N, N)`` intermediate.

This test pins the invariant that both paths agree to within float64
operation-order rounding noise, mirroring the existing meshgrid-vs-
elementwise parity contract in ``test_cmc_kernel_parity.py``.  Any future
physics change must touch the unified kernel only, and any drift between
the vmap reference path and the pooled production path trips a test.
"""

from __future__ import annotations

import os

# Rule 8: x64 before first jax import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from heterodyne.core.jax_backend import (  # noqa: E402
    compute_c2_heterodyne_multiphi,
    compute_c2_heterodyne_pooled,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_Q = 0.0054  # Å⁻¹
_DT = 0.001  # s
_N_TIMES = 60
_N_PHI = 3
_N_POOLED = 400
_ATOL = 1.0e-12
_RTOL = 1.0e-10


def _time_grid() -> jnp.ndarray:
    return jnp.arange(_N_TIMES, dtype=jnp.float64) * _DT


def _random_params(rng: np.random.Generator) -> jnp.ndarray:
    """Sample a physics-plausible 14-parameter vector within bounds.

    Mirrors typical NLSQ posterior regions — D0 / v0 sampled on log-scale,
    exponents in their narrow physical ranges, fractions in (0, 1).
    """
    return jnp.asarray(
        [
            float(10 ** rng.uniform(3.0, 5.0)),  # D0_ref
            float(rng.uniform(-0.5, 0.5)),  # alpha_ref
            float(rng.uniform(-0.5, 0.5)),  # D_offset_ref
            float(10 ** rng.uniform(3.0, 5.0)),  # D0_sample
            float(rng.uniform(-0.5, 0.5)),  # alpha_sample
            float(rng.uniform(-0.5, 0.5)),  # D_offset_sample
            float(10 ** rng.uniform(2.0, 3.5)),  # v0
            float(rng.uniform(-0.5, 0.5)),  # beta
            float(rng.uniform(-10.0, 10.0)),  # v_offset
            float(rng.uniform(0.2, 0.8)),  # f0
            float(rng.uniform(-1.0, 1.0)),  # f1
            float(rng.uniform(-0.05, 0.05)),  # f2
            float(rng.uniform(-0.1, 0.1)),  # f3
            float(rng.uniform(-30.0, 30.0)),  # phi0
        ],
        dtype=jnp.float64,
    )


def _build_pooled_indices(
    rng: np.random.Generator,
    n_pooled: int,
    n_times: int,
    n_phi: int,
    *,
    include_boundary: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Draw random (idx1, idx2, phi_indices) for pooled-point lookup.

    When ``include_boundary=False`` the indices avoid the t=0 row/column
    (homodyne contract for the likelihood support).  When True, some
    indices land on 0 so we can check the kernel still agrees there — the
    boundary mask is applied outside the kernel (in the likelihood).
    """
    lo = 0 if include_boundary else 1
    idx1 = rng.integers(lo, n_times, size=n_pooled).astype(np.int64)
    idx2 = rng.integers(lo, n_times, size=n_pooled).astype(np.int64)
    phi_indices = rng.integers(0, n_phi, size=n_pooled).astype(np.int64)
    return jnp.asarray(idx1), jnp.asarray(idx2), jnp.asarray(phi_indices)


def _gather_vmap_reference(
    params: jnp.ndarray,
    t: jnp.ndarray,
    phi_unique: jnp.ndarray,
    contrast_arr: jnp.ndarray,
    offset_arr: jnp.ndarray,
    idx1: jnp.ndarray,
    idx2: jnp.ndarray,
    phi_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Vmap+gather reference: builds (n_phi, N, N) stack then gathers."""
    c2_stack = compute_c2_heterodyne_multiphi(
        params, t, _Q, _DT, phi_unique, contrast_arr, offset_arr
    )
    return c2_stack[phi_indices, idx1, idx2]


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


class TestPooledKernelParity:
    """The pooled kernel must agree with the vmap+gather reference path to
    within float64 operation-order rounding noise.

    Together with ``TestKernelParityBaseline`` (meshgrid vs elementwise),
    this locks in the Phase 4 invariant: all three evaluation strategies
    (meshgrid / elementwise / pooled) share the same physics math.
    """

    def test_pooled_matches_vmap_gather_basic(self) -> None:
        rng = np.random.default_rng(20260521)
        params = _random_params(rng)
        t = _time_grid()
        phi_unique = jnp.asarray(
            sorted(rng.uniform(-90.0, 90.0, size=_N_PHI)), dtype=jnp.float64
        )
        contrast_arr = jnp.asarray(rng.uniform(0.6, 1.0, size=_N_PHI))
        offset_arr = jnp.asarray(rng.uniform(0.9, 1.1, size=_N_PHI))
        idx1, idx2, phi_indices = _build_pooled_indices(
            rng, _N_POOLED, _N_TIMES, _N_PHI
        )

        c2_ref = np.asarray(
            _gather_vmap_reference(
                params,
                t,
                phi_unique,
                contrast_arr,
                offset_arr,
                idx1,
                idx2,
                phi_indices,
            )
        )
        c2_pooled = np.asarray(
            compute_c2_heterodyne_pooled(
                params,
                t,
                _Q,
                _DT,
                idx1,
                idx2,
                phi_indices,
                phi_unique,
                contrast_arr,
                offset_arr,
            )
        )

        assert c2_pooled.shape == (_N_POOLED,)
        assert np.allclose(c2_ref, c2_pooled, atol=_ATOL, rtol=_RTOL), (
            "pooled kernel deviates from vmap+gather reference: "
            f"max abs diff = {np.max(np.abs(c2_ref - c2_pooled)):.3e}"
        )

    def test_pooled_matches_vmap_gather_across_seeds(self) -> None:
        """Sweep multiple random fixtures — each must agree to atol/rtol."""
        for seed in (1, 7, 42, 2026, 99999):
            rng = np.random.default_rng(seed)
            params = _random_params(rng)
            t = _time_grid()
            phi_unique = jnp.asarray(
                sorted(rng.uniform(-90.0, 90.0, size=_N_PHI)), dtype=jnp.float64
            )
            contrast_arr = jnp.asarray(rng.uniform(0.6, 1.0, size=_N_PHI))
            offset_arr = jnp.asarray(rng.uniform(0.9, 1.1, size=_N_PHI))
            idx1, idx2, phi_indices = _build_pooled_indices(
                rng, _N_POOLED, _N_TIMES, _N_PHI
            )

            c2_ref = np.asarray(
                _gather_vmap_reference(
                    params,
                    t,
                    phi_unique,
                    contrast_arr,
                    offset_arr,
                    idx1,
                    idx2,
                    phi_indices,
                )
            )
            c2_pooled = np.asarray(
                compute_c2_heterodyne_pooled(
                    params,
                    t,
                    _Q,
                    _DT,
                    idx1,
                    idx2,
                    phi_indices,
                    phi_unique,
                    contrast_arr,
                    offset_arr,
                )
            )
            diff = float(np.max(np.abs(c2_ref - c2_pooled)))
            assert np.allclose(c2_ref, c2_pooled, atol=_ATOL, rtol=_RTOL), (
                f"seed={seed}: pooled vs vmap+gather max abs diff = {diff:.3e} "
                f"exceeds atol={_ATOL:.0e}, rtol={_RTOL:.0e}"
            )

    def test_pooled_boundary_agreement(self) -> None:
        """Both paths agree even at t=0 boundary indices.

        The pooled kernel does not apply the boundary mask itself — that
        is the likelihood's responsibility.  This test verifies that the
        bare kernel still agrees with the vmap+gather reference at the
        boundary, so any future masking change is a likelihood-layer
        concern, not a kernel-layer one.
        """
        rng = np.random.default_rng(20260521 + 1)
        params = _random_params(rng)
        t = _time_grid()
        phi_unique = jnp.asarray(
            sorted(rng.uniform(-90.0, 90.0, size=_N_PHI)), dtype=jnp.float64
        )
        contrast_arr = jnp.asarray(rng.uniform(0.6, 1.0, size=_N_PHI))
        offset_arr = jnp.asarray(rng.uniform(0.9, 1.1, size=_N_PHI))
        idx1, idx2, phi_indices = _build_pooled_indices(
            rng, _N_POOLED, _N_TIMES, _N_PHI, include_boundary=True
        )
        # Ensure we actually hit the boundary
        assert (
            int(np.min(np.asarray(idx1))) == 0 or int(np.min(np.asarray(idx2))) == 0
        ), "fixture failed to draw a boundary index — re-seed or widen the draw"

        c2_ref = np.asarray(
            _gather_vmap_reference(
                params,
                t,
                phi_unique,
                contrast_arr,
                offset_arr,
                idx1,
                idx2,
                phi_indices,
            )
        )
        c2_pooled = np.asarray(
            compute_c2_heterodyne_pooled(
                params,
                t,
                _Q,
                _DT,
                idx1,
                idx2,
                phi_indices,
                phi_unique,
                contrast_arr,
                offset_arr,
            )
        )
        assert np.allclose(c2_ref, c2_pooled, atol=_ATOL, rtol=_RTOL), (
            "pooled kernel disagrees with vmap+gather at the t=0 boundary: "
            f"max abs diff = {np.max(np.abs(c2_ref - c2_pooled)):.3e}"
        )


class TestPooledKernelGradientParity:
    """Gradient parity: ``jax.grad(sum(c2))`` w.r.t. params must match.

    A gradient mismatch would mean the pooled and vmap paths give the
    same forward value but disagree on backward — the kind of silent
    physics bug that would shift NUTS leapfrog trajectories.
    """

    def _grad_sum(
        self,
        kernel_fn,
        params,
        *args,
    ):
        def _scalar(p):
            return jnp.sum(kernel_fn(p, *args))

        return jax.value_and_grad(_scalar)(params)

    def test_gradient_matches_vmap_gather(self) -> None:
        rng = np.random.default_rng(20260521 + 2)
        params = _random_params(rng)
        t = _time_grid()
        phi_unique = jnp.asarray(
            sorted(rng.uniform(-90.0, 90.0, size=_N_PHI)), dtype=jnp.float64
        )
        contrast_arr = jnp.asarray(rng.uniform(0.6, 1.0, size=_N_PHI))
        offset_arr = jnp.asarray(rng.uniform(0.9, 1.1, size=_N_PHI))
        idx1, idx2, phi_indices = _build_pooled_indices(
            rng, _N_POOLED, _N_TIMES, _N_PHI
        )

        def vmap_gather_kernel(p):
            return _gather_vmap_reference(
                p,
                t,
                phi_unique,
                contrast_arr,
                offset_arr,
                idx1,
                idx2,
                phi_indices,
            )

        def pooled_kernel(p):
            return compute_c2_heterodyne_pooled(
                p,
                t,
                _Q,
                _DT,
                idx1,
                idx2,
                phi_indices,
                phi_unique,
                contrast_arr,
                offset_arr,
            )

        val_ref, grad_ref = jax.value_and_grad(
            lambda p: jnp.sum(vmap_gather_kernel(p))
        )(params)
        val_new, grad_new = jax.value_and_grad(lambda p: jnp.sum(pooled_kernel(p)))(
            params
        )

        assert np.allclose(
            np.asarray(val_ref), np.asarray(val_new), atol=_ATOL, rtol=_RTOL
        ), (
            f"forward value mismatch: ref={float(val_ref):.6e}, "
            f"pooled={float(val_new):.6e}"
        )
        grad_ref_np = np.asarray(grad_ref)
        grad_new_np = np.asarray(grad_new)
        # Same tolerance as the forward parity: the gather and the pooled
        # kernel are bit-equivalent up to JAX op-ordering noise, so the VJP
        # they emit is bit-equivalent too. If this ever needs loosening it's
        # a signal that the kernels have drifted, not a tolerance issue.
        assert np.allclose(grad_ref_np, grad_new_np, atol=_ATOL, rtol=_RTOL), (
            "gradient mismatch: "
            f"max abs diff = {float(np.max(np.abs(grad_ref_np - grad_new_np))):.3e}, "
            f"max rel diff = "
            f"{float(np.max(np.abs((grad_ref_np - grad_new_np) / (np.abs(grad_ref_np) + 1e-30)))):.3e}"
        )


# ---------------------------------------------------------------------------
# Memory-shape proof (documentary, not budget)
# ---------------------------------------------------------------------------


class TestPooledKernelMemoryShape:
    """Document — in a runnable form — the memory-shape difference between
    the pooled production path and the vmap reference path.

    We use ``jax.eval_shape`` so no buffers are actually allocated; the
    test is fast and deterministic, and serves as a contract that the
    pooled path does not regress to materializing the (n_phi, N, N) stack.
    """

    def test_pooled_output_shape_is_n_total(self) -> None:
        rng = np.random.default_rng(0)
        params = _random_params(rng)
        t = _time_grid()
        phi_unique = jnp.asarray([0.0, 45.0, 90.0], dtype=jnp.float64)
        contrast_arr = jnp.ones((_N_PHI,))
        offset_arr = jnp.ones((_N_PHI,))
        idx1, idx2, phi_indices = _build_pooled_indices(
            rng, _N_POOLED, _N_TIMES, _N_PHI
        )

        out_shape = jax.eval_shape(
            compute_c2_heterodyne_pooled,
            params,
            t,
            _Q,
            _DT,
            idx1,
            idx2,
            phi_indices,
            phi_unique,
            contrast_arr,
            offset_arr,
        )
        assert out_shape.shape == (_N_POOLED,), (
            f"pooled output shape regressed from (n_total,) to {out_shape.shape}"
        )

    def test_vmap_reference_output_shape_is_nphi_n_n(self) -> None:
        rng = np.random.default_rng(0)
        params = _random_params(rng)
        t = _time_grid()
        phi_unique = jnp.asarray([0.0, 45.0, 90.0], dtype=jnp.float64)
        contrast_arr = jnp.ones((_N_PHI,))
        offset_arr = jnp.ones((_N_PHI,))

        out_shape = jax.eval_shape(
            compute_c2_heterodyne_multiphi,
            params,
            t,
            _Q,
            _DT,
            phi_unique,
            contrast_arr,
            offset_arr,
        )
        assert out_shape.shape == (_N_PHI, _N_TIMES, _N_TIMES), (
            "vmap reference output shape changed — if this is intentional, "
            "update the parity-test shape contract here."
        )
