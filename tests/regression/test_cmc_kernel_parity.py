"""Numerical parity between meshgrid and elementwise c2 kernels (Tier 4 G1).

Tier 4 baseline measurement on main: do ``compute_c2_heterodyne`` (meshgrid)
and ``compute_c2_elementwise`` (ShardGrid) produce numerically identical c2
values at matching (i, j) coordinate pairs?

If the answer is yes (max abs diff ≤ 1e-10 across all gauntlet scenarios),
a unified-kernel refactor that wires both call sites through a single core
function is feasible.  If the answer is no, the kernels are already
divergent on main; the unification would force one path to change its
output, breaking either NLSQ or CMC.  In that case, the Tier 4 deliverable
is a contract-doc + this regression test pinning the current divergence.

The 5 gauntlet scenarios cover:
    1. NLSQ single-angle (n_phi=1, individual mode)
    2. NLSQ multi-angle averaged (n_phi=4, auto)
    3. NLSQ multi-angle individual (n_phi=4, individual mode)
    4. CMC single-shard (one shard covers the whole grid)
    5. CMC multi-shard (3 shards over the same grid)
"""

from __future__ import annotations

import os

# Rule 8: x64 before first jax import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from heterodyne.core.jax_backend import compute_c2_heterodyne  # noqa: E402
from heterodyne.core.physics_cmc import (  # noqa: E402
    compute_c2_elementwise,
    precompute_shard_grid_from_matrix,
)

# Realistic 14-parameter heterodyne fixture (physics defaults, not pathological).
_PARAMS = jnp.asarray(
    [
        1.0e4,
        0.0,
        0.0,  # D0_ref, alpha_ref, D_offset_ref
        1.0e4,
        0.0,
        0.0,  # D0_sample, alpha_sample, D_offset_sample
        1.0e3,
        0.0,
        0.0,  # v0, beta, v_offset
        0.5,
        0.0,
        0.0,
        0.0,  # f0, f1, f2, f3
        0.0,  # phi0
    ],
    dtype=jnp.float64,
)

_Q = 0.0054  # Å⁻¹
_DT = 0.001  # s
_N_TIMES = 100  # 100×100 grid is plenty to expose float64 drift
_TOLERANCE = 1.0e-10


def _time_grid() -> jnp.ndarray:
    return jnp.arange(_N_TIMES, dtype=jnp.float64) * _DT


def _meshgrid_c2(phi_angle: float, contrast: float, offset: float) -> jnp.ndarray:
    t = _time_grid()
    return compute_c2_heterodyne(_PARAMS, t, _Q, _DT, phi_angle, contrast, offset)


def _elementwise_c2(
    phi_angle: float,
    contrast: float,
    offset: float,
    shard_start: int,
    shard_end: int,
) -> jnp.ndarray:
    """Element-wise upper-triangular c2 for shard rows [shard_start, shard_end)."""
    t = _time_grid()
    grid = precompute_shard_grid_from_matrix(t, shard_start, shard_end)
    return compute_c2_elementwise(_PARAMS, grid, _Q, _DT, phi_angle, contrast, offset)


def _diff_meshgrid_vs_elementwise(
    phi_angle: float, contrast: float, offset: float
) -> tuple[float, tuple[int, int]]:
    """Compare full-matrix meshgrid vs upper-triangular elementwise.

    Returns ``(max_abs_diff, (i*, j*))`` for the worst-case pair.
    """
    c2_mesh = np.asarray(_meshgrid_c2(phi_angle, contrast, offset))
    c2_elem = np.asarray(_elementwise_c2(phi_angle, contrast, offset, 0, _N_TIMES))

    # Element-wise result is in upper-triangular (i<=j) order matching
    # precompute_shard_grid_from_matrix's np.triu_indices(N, k=0).
    triu_i, triu_j = np.triu_indices(_N_TIMES, k=0)
    mesh_triu = c2_mesh[triu_i, triu_j]
    abs_diff = np.abs(mesh_triu - c2_elem)
    arg = int(np.argmax(abs_diff))
    return float(abs_diff[arg]), (int(triu_i[arg]), int(triu_j[arg]))


# ---------------------------------------------------------------------------
# Gauntlet scenarios
# ---------------------------------------------------------------------------


class TestKernelParityBaseline:
    """Records the meshgrid-vs-elementwise divergence on main for each scenario.

    The threshold is ``_TOLERANCE = 1e-10``.  Scenarios that hold ≤ this
    tolerance make a unified-kernel refactor feasible.  Any scenario that
    exceeds it on main is a pre-existing divergence; the Tier 4 deliverable
    becomes the contract doc rather than a kernel merge.
    """

    def test_s1_nlsq_single_angle(self) -> None:
        diff, (i, j) = _diff_meshgrid_vs_elementwise(
            phi_angle=0.0, contrast=1.0, offset=0.0
        )
        assert diff <= _TOLERANCE, (
            f"scenario 1 (NLSQ single-angle): max abs diff = {diff:.3e} "
            f"at (i*, j*) = ({i}, {j}); tolerance = {_TOLERANCE:.0e}"
        )

    def test_s2_nlsq_multi_angle_averaged(self) -> None:
        # Same scaling, multiple phi angles — averaged mode.
        for phi in (0.0, 45.0, 90.0, 135.0):
            diff, (i, j) = _diff_meshgrid_vs_elementwise(
                phi_angle=phi, contrast=1.0, offset=0.0
            )
            assert diff <= _TOLERANCE, (
                f"scenario 2 (NLSQ averaged) at phi={phi}: max abs diff = "
                f"{diff:.3e} at (i*, j*) = ({i}, {j})"
            )

    def test_s3_nlsq_multi_angle_individual(self) -> None:
        # Per-angle contrast and offset (individual mode).
        per_angle = [
            (0.0, 1.0, 0.0),
            (45.0, 0.85, 0.05),
            (90.0, 0.72, 0.1),
            (135.0, 0.95, -0.03),
        ]
        for phi, contrast, offset in per_angle:
            diff, (i, j) = _diff_meshgrid_vs_elementwise(
                phi_angle=phi, contrast=contrast, offset=offset
            )
            assert diff <= _TOLERANCE, (
                f"scenario 3 (NLSQ individual) at phi={phi}, "
                f"contrast={contrast}, offset={offset}: max abs diff = "
                f"{diff:.3e} at (i*, j*) = ({i}, {j})"
            )

    def test_s4_cmc_single_shard(self) -> None:
        # One ShardGrid covering the entire N×N — same as scenario 1
        # except the test asserts shape preservation (n_pairs = N(N+1)/2).
        t = _time_grid()
        grid = precompute_shard_grid_from_matrix(t, 0, _N_TIMES)
        expected_pairs = _N_TIMES * (_N_TIMES + 1) // 2
        assert grid.n_pairs == expected_pairs
        c2_elem = np.asarray(_elementwise_c2(0.0, 1.0, 0.0, 0, _N_TIMES))
        assert c2_elem.shape == (expected_pairs,)

        c2_mesh = np.asarray(_meshgrid_c2(0.0, 1.0, 0.0))
        triu_i, triu_j = np.triu_indices(_N_TIMES, k=0)
        diff = float(np.max(np.abs(c2_mesh[triu_i, triu_j] - c2_elem)))
        assert diff <= _TOLERANCE, (
            f"scenario 4 (CMC single-shard): max abs diff = {diff:.3e}"
        )

    def test_s5_cmc_multi_shard(self) -> None:
        # Split the N×N grid into 3 diagonal-block shards and compare each
        # shard's elementwise result against the same-rows-and-columns
        # slice of the meshgrid result.
        t = _time_grid()
        c2_mesh = np.asarray(_meshgrid_c2(0.0, 1.0, 0.0))
        starts = (0, _N_TIMES // 3, 2 * _N_TIMES // 3)
        ends = (_N_TIMES // 3, 2 * _N_TIMES // 3, _N_TIMES)
        for shard_start, shard_end in zip(starts, ends, strict=True):
            grid = precompute_shard_grid_from_matrix(t, shard_start, shard_end)
            c2_elem = np.asarray(
                compute_c2_elementwise(_PARAMS, grid, _Q, _DT, 0.0, 1.0, 0.0)
            )
            shard_size = shard_end - shard_start
            triu_i, triu_j = np.triu_indices(shard_size, k=0)
            mesh_triu = c2_mesh[triu_i + shard_start, triu_j + shard_start]
            diff = float(np.max(np.abs(mesh_triu - c2_elem)))
            assert diff <= _TOLERANCE, (
                f"scenario 5 (CMC multi-shard [{shard_start}:{shard_end}]): "
                f"max abs diff = {diff:.3e}"
            )


# ---------------------------------------------------------------------------
# Unified-vs-shim parity (Tier 4 G1 merge invariant)
# ---------------------------------------------------------------------------


class TestUnifiedKernelParity:
    """The legacy ``compute_c2_*`` shims must produce results bit-identical
    to ``compute_c2_unified`` with the matching ``eval_strategy``.

    Together with :class:`TestKernelParityBaseline` this locks in the G1
    invariant: the unified kernel and both shims agree, so any future
    physics change must touch the unified kernel only, and any drift trips
    a test.
    """

    def test_unified_meshgrid_matches_shim_s1(self) -> None:
        from heterodyne.core.physics_kernel import compute_c2_unified

        t = _time_grid()
        c2_shim = np.asarray(_meshgrid_c2(0.0, 1.0, 0.0))
        c2_unified = np.asarray(
            compute_c2_unified(
                _PARAMS,
                _Q,
                _DT,
                0.0,
                1.0,
                0.0,
                eval_strategy="meshgrid",
                t=t,
            )
        )
        diff = float(np.max(np.abs(c2_shim - c2_unified)))
        assert diff == 0.0, (
            f"unified meshgrid != shim on s1 fixture: max abs diff = {diff:.3e}"
        )

    def test_unified_meshgrid_matches_shim_s2_multi_phi(self) -> None:
        from heterodyne.core.physics_kernel import compute_c2_unified

        t = _time_grid()
        for phi in (0.0, 45.0, 90.0, 135.0):
            c2_shim = np.asarray(_meshgrid_c2(phi, 1.0, 0.0))
            c2_unified = np.asarray(
                compute_c2_unified(
                    _PARAMS,
                    _Q,
                    _DT,
                    phi,
                    1.0,
                    0.0,
                    eval_strategy="meshgrid",
                    t=t,
                )
            )
            diff = float(np.max(np.abs(c2_shim - c2_unified)))
            assert diff == 0.0, (
                f"unified meshgrid != shim at phi={phi}: max abs diff = {diff:.3e}"
            )

    def test_unified_meshgrid_matches_shim_s3_individual(self) -> None:
        from heterodyne.core.physics_kernel import compute_c2_unified

        t = _time_grid()
        per_angle = [
            (0.0, 1.0, 0.0),
            (45.0, 0.85, 0.05),
            (90.0, 0.72, 0.1),
            (135.0, 0.95, -0.03),
        ]
        for phi, contrast, offset in per_angle:
            c2_shim = np.asarray(_meshgrid_c2(phi, contrast, offset))
            c2_unified = np.asarray(
                compute_c2_unified(
                    _PARAMS,
                    _Q,
                    _DT,
                    phi,
                    contrast,
                    offset,
                    eval_strategy="meshgrid",
                    t=t,
                )
            )
            diff = float(np.max(np.abs(c2_shim - c2_unified)))
            assert diff == 0.0, (
                f"unified meshgrid != shim at phi={phi}, contrast={contrast}, "
                f"offset={offset}: max abs diff = {diff:.3e}"
            )

    def test_unified_elementwise_matches_shim_s4_single_shard(self) -> None:
        from heterodyne.core.physics_kernel import compute_c2_unified

        t = _time_grid()
        grid = precompute_shard_grid_from_matrix(t, 0, _N_TIMES)
        c2_shim = np.asarray(_elementwise_c2(0.0, 1.0, 0.0, 0, _N_TIMES))
        c2_unified = np.asarray(
            compute_c2_unified(
                _PARAMS,
                _Q,
                _DT,
                0.0,
                1.0,
                0.0,
                eval_strategy="elementwise",
                shard_grid=grid,
            )
        )
        diff = float(np.max(np.abs(c2_shim - c2_unified)))
        assert diff == 0.0, (
            f"unified elementwise != shim on s4 (single-shard): "
            f"max abs diff = {diff:.3e}"
        )

    def test_unified_elementwise_matches_shim_s5_multi_shard(self) -> None:
        from heterodyne.core.physics_kernel import compute_c2_unified

        t = _time_grid()
        starts = (0, _N_TIMES // 3, 2 * _N_TIMES // 3)
        ends = (_N_TIMES // 3, 2 * _N_TIMES // 3, _N_TIMES)
        for shard_start, shard_end in zip(starts, ends, strict=True):
            grid = precompute_shard_grid_from_matrix(t, shard_start, shard_end)
            c2_shim = np.asarray(
                compute_c2_elementwise(_PARAMS, grid, _Q, _DT, 0.0, 1.0, 0.0)
            )
            c2_unified = np.asarray(
                compute_c2_unified(
                    _PARAMS,
                    _Q,
                    _DT,
                    0.0,
                    1.0,
                    0.0,
                    eval_strategy="elementwise",
                    shard_grid=grid,
                )
            )
            diff = float(np.max(np.abs(c2_shim - c2_unified)))
            assert diff == 0.0, (
                f"unified elementwise != shim on s5 shard [{shard_start}:"
                f"{shard_end}]: max abs diff = {diff:.3e}"
            )
