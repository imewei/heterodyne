"""Shared physics kernel for heterodyne two-time correlation.

Codex/Gemini G1: prior to this module, the meshgrid path
(``jax_backend.compute_c2_heterodyne``) and the element-wise path
(``physics_cmc.compute_c2_elementwise``) implemented the same heterodyne
physics in two structurally-different functions.  Any change to the model
had to be made in both places, and any drift between them was a silent
physics bug.

This module factors the **physics** out of the **evaluation strategy**.
:func:`compute_c2_unified` is one JIT-compiled function whose
``eval_strategy`` static-argument selects:

- ``"meshgrid"``: full N×N two-time matrix (NLSQ path, ``compute_residuals``).
- ``"elementwise"``: O(n_pairs) lookup using a :class:`ShardGrid` (CMC path).

Both strategies consume the same primitives from
:mod:`heterodyne.core.physics_utils` (``compute_transport_rate``,
``compute_velocity_rate``, ``trapezoid_cumsum``, ``smooth_abs``,
``smooth_clip``, ``safe_exp``, ``create_time_integral_matrix``) and apply
the same gradient-safe clips, so the outputs agree to within float64
operation-order rounding noise (≤ 1e-15 on the canonical regression
fixture; ≤ 1e-10 on every gauntlet scenario in
``tests/regression/test_cmc_kernel_parity.py``).

The legacy ``compute_c2_heterodyne`` and ``compute_c2_elementwise`` are
retained as thin shims that pin ``eval_strategy`` and forward.  No caller
needs to change.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp

from heterodyne.core.physics_utils import (
    compute_transport_rate,
    compute_velocity_rate,
    create_time_integral_matrix,
    safe_exp,
    smooth_abs,
    smooth_clip,
    trapezoid_cumsum,
)

if TYPE_CHECKING:
    # Import for typing only — physics_cmc imports this module at runtime, so
    # a non-TYPE_CHECKING import would create a cycle.
    from heterodyne.core.physics_cmc import ShardGrid


EvalStrategy = Literal["meshgrid", "elementwise"]


def _half_transport_meshgrid(
    t: jnp.ndarray,
    D0: jnp.ndarray,
    alpha: jnp.ndarray,
    D_offset: jnp.ndarray,
    q: float,
    dt: float,
) -> jnp.ndarray:
    """Full ``(N, N)`` half-transport matrix: exp(-½ q² |∫J dt|).

    The ``jnp.exp(jnp.clip(...))`` idiom is intentional: clipping the
    exponent argument (not the post-exp value) is an overflow guard, not a
    gradient-killing physical clip — gradients are already vanishingly
    small at the cap.  See ``test_no_unguarded_clip_in_physics_module``.
    """
    rate = compute_transport_rate(t, D0, alpha, D_offset)
    cumsum = trapezoid_cumsum(rate, dt)
    J_integral = smooth_abs(create_time_integral_matrix(cumsum))
    return jnp.exp(jnp.clip(-0.5 * q * q * J_integral, -700.0, 0.0))


def _half_transport_elementwise(
    shard_grid: ShardGrid,
    D0: jnp.ndarray,
    alpha: jnp.ndarray,
    D_offset: jnp.ndarray,
    q: float,
    dt: float,
) -> jnp.ndarray:
    """Per-pair half-transport ``(n_pairs,)`` via cumsum-index lookup.

    Same ``jnp.exp(jnp.clip(...))`` overflow-guard pattern as the meshgrid
    sibling (see allow-list in ``test_no_unguarded_clip_in_physics_module``).
    """
    rate = compute_transport_rate(shard_grid.time_grid, D0, alpha, D_offset)
    cumsum = trapezoid_cumsum(rate, dt)
    integral = smooth_abs(cumsum[shard_grid.idx2] - cumsum[shard_grid.idx1])
    return jnp.exp(jnp.clip(-0.5 * q * q * integral, -700.0, 0.0))


def _velocity_integral_meshgrid(
    t: jnp.ndarray,
    v0: jnp.ndarray,
    beta: jnp.ndarray,
    v_offset: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Full ``(N, N)`` signed velocity integral."""
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    v_cumsum = trapezoid_cumsum(velocity, dt)
    return create_time_integral_matrix(v_cumsum)


def _velocity_integral_elementwise(
    shard_grid: ShardGrid,
    v0: jnp.ndarray,
    beta: jnp.ndarray,
    v_offset: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Per-pair signed velocity integral ``(n_pairs,)``."""
    velocity = compute_velocity_rate(shard_grid.time_grid, v0, beta, v_offset)
    v_cumsum = trapezoid_cumsum(velocity, dt)
    return v_cumsum[shard_grid.idx2] - v_cumsum[shard_grid.idx1]


def _fraction(
    t_vals: jnp.ndarray,
    f0: jnp.ndarray,
    f1: jnp.ndarray,
    f2: jnp.ndarray,
    f3: jnp.ndarray,
) -> jnp.ndarray:
    """Sample-fraction f_s(t), smoothly bounded to [0, 1].

    Mirrors the implementation in both legacy kernels exactly so that
    the unified-vs-shim parity guarantee holds: ``safe_exp`` caps the
    exponent and ``smooth_clip`` preserves Jacobian gradient at the
    boundary (Rule 7).  ``jnp.clip`` would zero the gradient.
    """
    return smooth_clip(f0 * safe_exp(f1 * (t_vals - f2)) + f3, 0.0, 1.0)


@partial(jax.jit, static_argnames=("eval_strategy",))
def compute_c2_unified(
    params: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float = 1.0,
    offset: float = 1.0,
    *,
    eval_strategy: EvalStrategy,
    t: jnp.ndarray | None = None,
    shard_grid: Any = None,
) -> jnp.ndarray:
    """Two-component heterodyne c2 via the shared kernel.

    The math is identical to both legacy kernels — they are now thin shims
    pinning ``eval_strategy`` and forwarding here.  The two paths share
    every primitive call, every clip range, every order of arithmetic
    operations that's structurally common; they differ only where the
    output shape forces a different reduction (outer products in meshgrid
    vs element-wise products in elementwise).

    Args:
        params: 14-parameter array.
        q: Scattering wavevector magnitude.
        dt: Time step.
        phi_angle: Detector phi angle (degrees).
        contrast: Speckle contrast (β).
        offset: Baseline offset.
        eval_strategy: ``"meshgrid"`` for full ``(N, N)`` output (NLSQ);
            ``"elementwise"`` for per-pair ``(n_pairs,)`` (CMC sharded).
        t: Time array shape ``(N,)`` — required when ``eval_strategy="meshgrid"``.
        shard_grid: :class:`ShardGrid` pytree — required when
            ``eval_strategy="elementwise"``.

    Returns:
        For ``eval_strategy="meshgrid"``: ``(N, N)`` array.
        For ``eval_strategy="elementwise"``: ``(n_pairs,)`` array.

    Raises:
        ValueError: Wrong combination of strategy-input arguments.
    """
    if eval_strategy == "meshgrid":
        if t is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='meshgrid', ...) requires t"
            )
        return _compute_c2_meshgrid(params, t, q, dt, phi_angle, contrast, offset)
    if eval_strategy == "elementwise":
        if shard_grid is None:
            raise ValueError(
                "compute_c2_unified(eval_strategy='elementwise', ...) "
                "requires shard_grid"
            )
        return _compute_c2_elementwise(
            params, shard_grid, q, dt, phi_angle, contrast, offset
        )
    raise ValueError(
        f"eval_strategy must be 'meshgrid' or 'elementwise', got {eval_strategy!r}"
    )


def _compute_c2_meshgrid(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """Meshgrid evaluation — produces ``(N, N)`` matrix."""
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    half_tr_ref = _half_transport_meshgrid(t, D0_ref, alpha_ref, D_offset_ref, q, dt)
    half_tr_sample = _half_transport_meshgrid(
        t, D0_sample, alpha_sample, D_offset_sample, q, dt
    )

    f_sample = _fraction(t, f0, f1, f2, f3)
    f_ref = 1.0 - f_sample

    v_integral = _velocity_integral_meshgrid(t, v0, beta, v_offset, dt)

    total_phi = phi_angle + phi0
    phi_rad = jnp.deg2rad(total_phi)
    phase = q * jnp.cos(phi_rad) * v_integral

    f_ref_matrix = f_ref[:, None] * f_ref[None, :]
    f_sample_matrix = f_sample[:, None] * f_sample[None, :]
    f_cross_vec = f_ref * f_sample
    f_cross_matrix = f_cross_vec[:, None] * f_cross_vec[None, :]

    ref_term = f_ref_matrix**2 * half_tr_ref**2
    sample_term = f_sample_matrix**2 * half_tr_sample**2
    cross_term = 2.0 * f_cross_matrix * half_tr_ref * half_tr_sample * jnp.cos(phase)

    norm_1 = f_sample**2 + f_ref**2
    normalization = norm_1[:, None] * norm_1[None, :]

    return offset + contrast * (ref_term + sample_term + cross_term) / jnp.where(
        normalization > 1e-10, normalization, 1e-10
    )


def _compute_c2_elementwise(
    params: jnp.ndarray,
    shard_grid: ShardGrid,
    q: float,
    dt: float,
    phi_angle: float,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """Element-wise evaluation — produces ``(n_pairs,)`` array."""
    D0_ref, alpha_ref, D_offset_ref = params[0], params[1], params[2]
    D0_sample, alpha_sample, D_offset_sample = params[3], params[4], params[5]
    v0, beta, v_offset = params[6], params[7], params[8]
    f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
    phi0 = params[13]

    half_tr_ref = _half_transport_elementwise(
        shard_grid, D0_ref, alpha_ref, D_offset_ref, q, dt
    )
    half_tr_sample = _half_transport_elementwise(
        shard_grid, D0_sample, alpha_sample, D_offset_sample, q, dt
    )

    v_integral = _velocity_integral_elementwise(shard_grid, v0, beta, v_offset, dt)

    total_phi = phi_angle + phi0
    phi_rad = jnp.deg2rad(total_phi)
    phase = q * jnp.cos(phi_rad) * v_integral

    t1_vals = shard_grid.time_grid[shard_grid.idx1]
    t2_vals = shard_grid.time_grid[shard_grid.idx2]
    f_sample_1 = _fraction(t1_vals, f0, f1, f2, f3)
    f_sample_2 = _fraction(t2_vals, f0, f1, f2, f3)
    f_ref_1 = 1.0 - f_sample_1
    f_ref_2 = 1.0 - f_sample_2

    f_ref_prod = f_ref_1 * f_ref_2
    f_sample_prod = f_sample_1 * f_sample_2
    f_cross_1 = f_ref_1 * f_sample_1
    f_cross_2 = f_ref_2 * f_sample_2
    f_cross_prod = f_cross_1 * f_cross_2

    ref_term = f_ref_prod**2 * half_tr_ref**2
    sample_term = f_sample_prod**2 * half_tr_sample**2
    cross_term = 2.0 * f_cross_prod * half_tr_ref * half_tr_sample * jnp.cos(phase)

    norm_1 = f_sample_1**2 + f_ref_1**2
    norm_2 = f_sample_2**2 + f_ref_2**2
    _norm_prod = norm_1 * norm_2
    normalization = jnp.where(_norm_prod > 1e-10, _norm_prod, 1e-10)

    return offset + contrast * (ref_term + sample_term + cross_term) / normalization
