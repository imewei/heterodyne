"""JAX-accelerated computational backend for heterodyne correlation.

This module provides JIT-compiled functions for computing the two-component
heterodyne correlation function using the integral formulation (PNAS Eq. S-95).
All functions are designed to be stateless and compatible with JAX
transformations (jit, vmap, grad).

The correlation is computed as
``c2 = offset + contrast × [ref + sample + cross] / f²``,
where transport terms use the integral of the rate J(t):
``half_tr[i,j] = exp(-½q² × |∫ J_rate(t') dt'|)``.

The 14 model parameters in canonical order are
D0_ref, alpha_ref, D_offset_ref,
D0_sample, alpha_sample, D_offset_sample,
v0, beta, v_offset, f0, f1, f2, f3, phi0.

Three evaluation strategies are exposed (all dispatch through
:func:`heterodyne.core.physics_kernel.compute_c2_unified`):

- :func:`compute_c2_heterodyne` — full ``(N, N)`` meshgrid output for NLSQ.
- ``compute_c2_elementwise`` (re-exported from :mod:`heterodyne.core.physics_cmc`)
  — sharded per-pair ``(n_pairs,)`` output for CMC.
- :func:`compute_c2_heterodyne_pooled` — per-pooled-point ``(n_total,)``
  output for joint multi-phi CMC (Phase 4; replaces the older
  vmap+gather pattern in :func:`compute_c2_heterodyne_multiphi`).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

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
    pass


@partial(jax.jit, static_argnames=("n_times",))
def compute_transport_jit(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    n_times: int,
) -> jnp.ndarray:
    """JIT-compiled pointwise transport coefficient computation.

    J(t) = D0 * t^alpha + offset

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses compute_transport_integral_matrix for the
        integral formulation (PNAS Eq. S-95). Retained for test
        compatibility and 1D visualization helpers.

    Args:
        t: Time array
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Constant offset
        n_times: Number of time points (static for JIT)

    Returns:
        Transport coefficient array
    """
    # Use jnp.where instead of jnp.maximum to preserve gradients at the t=0
    # floor (jnp.maximum zeros the gradient when t < 1e-10).
    t_safe = jnp.where(t > 1e-10, t, 1e-10)
    t_power = jnp.power(t_safe, alpha)
    t_power = jnp.where(t > 0, t_power, 0.0)
    return D0 * t_power + offset


@jax.jit
def compute_g1_transport(
    J: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """JIT-compiled pointwise g1 correlation from transport coefficient.

    g1(t) = exp(-q² * J(t))

    .. deprecated::
        Pointwise approximation — not used in production correlation.
        Production code uses the integral formulation via
        compute_transport_integral_matrix. Retained for test
        compatibility and 1D visualization helpers.

    Args:
        J: Transport coefficient array
        q: Scattering wavevector

    Returns:
        g1 correlation array
    """
    return jnp.exp(-q * q * J)


@jax.jit
def compute_fraction_jit(
    t: jnp.ndarray,
    f0: float,
    f1: float,
    f2: float,
    f3: float,
) -> jnp.ndarray:
    """JIT-compiled sample fraction computation.

    f_s(t) = f0 * exp(f1 * (t - f2)) + f3, clipped to [0, 1]

    Args:
        t: Time array
        f0: Amplitude
        f1: Exponential rate
        f2: Time shift
        f3: Baseline

    Returns:
        Fraction array in [0, 1]
    """
    # ``safe_exp`` + ``smooth_clip`` preserve gradient at the [0, 1] boundary
    # so NLSQ Jacobian descent does not stall when f(t) saturates (CLAUDE.md
    # rule #7 — gradient-safe floors).  Mirrors physics_cmc.py to keep the
    # element-wise and meshgrid paths bit-equivalent.
    fraction = f0 * safe_exp(f1 * (t - f2)) + f3
    return smooth_clip(fraction, 0.0, 1.0)


@jax.jit
def compute_velocity_integral_matrix(
    t: jnp.ndarray,
    v0: float,
    beta: float,
    v_offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled velocity integral matrix (NLSQ meshgrid path).

    Computes M[i,j] = ∫_{t_i}^{t_j} v(t') dt'
    where v(t) = v0 * t^beta + v_offset

    Uses shared ``trapezoid_cumsum`` → ``create_time_integral_matrix``
    pipeline for O(N) efficiency and O(dt²) accuracy.
    The velocity integral is *signed* (not absolute-valued) because it
    feeds into the phase factor ``cos(q cos(φ) ∫v dt)``.

    Args:
        t: Time array, shape (N,)
        v0: Velocity prefactor
        beta: Velocity exponent
        v_offset: Velocity offset
        dt: Time step

    Returns:
        Signed integral matrix, shape (N, N)
    """
    velocity = compute_velocity_rate(t, v0, beta, v_offset)
    cumsum = trapezoid_cumsum(velocity, dt)
    return create_time_integral_matrix(cumsum)


@jax.jit
def compute_transport_integral_matrix(
    t: jnp.ndarray,
    D0: float,
    alpha: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled transport integral matrix (NLSQ meshgrid path).

    Computes ``M[i,j] = |∫_{t_i}^{t_j} J_rate(t') dt'|``
    where J_rate(t) = D0 * t^alpha + offset

    Uses shared ``compute_transport_rate`` → ``trapezoid_cumsum`` →
    ``create_time_integral_matrix`` → ``smooth_abs`` pipeline.

    Args:
        t: Time array, shape (N,)
        D0: Transport prefactor
        alpha: Transport exponent
        offset: Transport rate offset
        dt: Time step

    Returns:
        Transport integral matrix, shape (N, N)
    """
    J_rate = compute_transport_rate(t, D0, alpha, offset)
    cumsum = trapezoid_cumsum(J_rate, dt)
    diff = create_time_integral_matrix(cumsum)
    return smooth_abs(diff)


def compute_c2_heterodyne(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    phi_angle: float | jnp.ndarray,
    contrast: float | jnp.ndarray = 1.0,
    offset: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """JIT-compiled two-time heterodyne correlation (meshgrid path).

    Thin shim around :func:`heterodyne.core.physics_kernel.compute_c2_unified`
    with ``eval_strategy="meshgrid"``.  Codex/Gemini G1: the physics math
    lives in the unified kernel; this function preserves the legacy import
    path for all NLSQ call sites (``compute_residuals``, ``compute_jacobian``,
    multi-angle stratified fits, etc.) without behavioural change.

    Args:
        params: Parameter array of shape ``(14,)`` in canonical order:
            ``[D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample,
            D_offset_sample, v0, beta, v_offset, f0, f1, f2, f3, phi0]``.
        t: Time array, shape (N,)
        q: Scattering wavevector magnitude
        dt: Time step
        phi_angle: Detector phi angle (degrees)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Correlation matrix c2, shape (N, N).
    """
    # Local import to avoid a cycle: physics_kernel.py imports nothing from
    # this module, but importing at module load would chain into JAX init
    # before some downstream test helpers expect it.
    from heterodyne.core.physics_kernel import compute_c2_unified

    return compute_c2_unified(  # type: ignore[no-any-return]
        params,
        q,
        dt,
        phi_angle,
        contrast,
        offset,
        eval_strategy="meshgrid",
        t=t,
    )


def compute_c2_heterodyne_multiphi(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    phi_unique: jnp.ndarray,
    contrast_arr: jnp.ndarray,
    offset_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Joint multi-phi c2 evaluator (vmap reference path — not used in CMC hot path).

    Vmap wrapper over :func:`compute_c2_heterodyne` that evaluates the
    heterodyne two-component model for every angle in ``phi_unique`` with
    the matching per-angle ``contrast_arr`` / ``offset_arr``. Returns a
    stacked tensor that can be gathered by ``(phi_indices, i1, i2)`` to
    recover c2 at each pooled (t1, t2, phi) tuple.

    .. note::
        The joint multi-phi CMC likelihood (Phase 4) now calls
        :func:`compute_c2_heterodyne_pooled` directly, which computes c2
        at the pooled points without ever materializing the
        ``(n_phi, N, N)`` stack this function returns. This helper is
        retained as a vmap reference path for parity tests
        (``tests/regression/test_cmc_pooled_kernel_parity.py``) and ad-hoc
        diagnostics.

    Args:
        params: Parameter array of shape ``(14,)`` (same canonical order as
            ``compute_c2_heterodyne``).
        t: Time grid array, shape ``(N,)``.
        q: Scattering wavevector magnitude.
        dt: Time step.
        phi_unique: Unique phi angles, shape ``(n_phi,)``.
        contrast_arr: Per-angle contrast values, shape ``(n_phi,)``.
        offset_arr: Per-angle offset values, shape ``(n_phi,)``.

    Returns:
        Stacked correlation, shape ``(n_phi, N, N)``. ``c2[i, j, k]`` is the
        model at phi=phi_unique[i], t1=t[j], t2=t[k] with contrast=contrast_arr[i],
        offset=offset_arr[i].
    """

    def _single(
        phi: jnp.ndarray, contrast: jnp.ndarray, offset: jnp.ndarray
    ) -> jnp.ndarray:
        return compute_c2_heterodyne(params, t, q, dt, phi, contrast, offset)

    return jax.vmap(_single, in_axes=(0, 0, 0))(phi_unique, contrast_arr, offset_arr)


def compute_c2_heterodyne_pooled(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float | jnp.ndarray,
    dt: float | jnp.ndarray,
    idx1: jnp.ndarray,
    idx2: jnp.ndarray,
    phi_indices: jnp.ndarray,
    phi_unique: jnp.ndarray,
    contrast_arr: jnp.ndarray,
    offset_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Pooled-data c2 evaluator for joint multi-phi CMC (homodyne parity).

    Thin shim around :func:`heterodyne.core.physics_kernel.compute_c2_unified`
    with ``eval_strategy="pooled"``.  Computes c2 directly at every pooled
    ``(phi_indices[k], idx1[k], idx2[k])`` tuple — never materializes the
    ``(n_phi, N, N)`` stack that the vmap+gather path required.  For
    N=1000, n_phi=4 this eliminates a ~256 MB float64 intermediate at every
    NUTS leapfrog step.

    Phase 4 of the joint multi-phi CMC refactor — replaces the per-step
    materialise-then-gather pattern in
    :func:`heterodyne.optimization.cmc.model._heterodyne_pooled_likelihood`.

    Args:
        params: Parameter array of shape ``(14,)`` (same canonical order as
            :func:`compute_c2_heterodyne`).
        t: Time grid array, shape ``(N,)``.
        q: Scattering wavevector magnitude.
        dt: Time step.
        idx1: Per-pooled-point index into ``t`` for the first time
            coordinate, shape ``(n_total,)``. Typically built via
            ``np.searchsorted(t, t1_flat)``.
        idx2: Per-pooled-point index into ``t`` for the second time
            coordinate, shape ``(n_total,)``.
        phi_indices: Per-pooled-point index into ``phi_unique``, shape
            ``(n_total,)``.
        phi_unique: Sorted unique phi angles, shape ``(n_phi,)``.
        contrast_arr: Per-angle contrast values, shape ``(n_phi,)``.
        offset_arr: Per-angle offset values, shape ``(n_phi,)``.

    Returns:
        Pooled correlation values, shape ``(n_total,)``. ``c2[k]`` is the
        model at ``phi=phi_unique[phi_indices[k]]``, ``t1=t[idx1[k]]``,
        ``t2=t[idx2[k]]`` with the matching per-angle scaling.
    """
    # Local import to avoid a cycle: physics_kernel.py imports nothing from
    # this module, but importing at module load would chain into JAX init
    # before some downstream test helpers expect it. Same pattern as
    # ``compute_c2_heterodyne`` above.
    from heterodyne.core.physics_kernel import compute_c2_unified

    return compute_c2_unified(  # type: ignore[no-any-return]
        params,
        q,
        dt,
        eval_strategy="pooled",
        time_grid=t,
        idx1=idx1,
        idx2=idx2,
        phi_unique=phi_unique,
        phi_indices=phi_indices,
        contrast_arr=contrast_arr,
        offset_arr=offset_arr,
    )


def compute_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute weighted residuals between model and data.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Flattened residual array
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )


@jax.jit
def _compute_residuals_jit(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """JIT-compiled residuals computation (always receives weights).

    Two boundary exclusions are applied at residual construction (not via
    data truncation, which would shorten the model's time grid and break
    viz/NPZ shape parity):

    1. The t=0 row ``(0, j)`` and t=0 column ``(i, 0)`` are zeroed.  The
       first frame holds the correlator's raw output; the model evaluates
       cleanly at t=0 (``g1(0,0)=1 ⇒ c2 = offset + contrast``), but the
       experimental boundary is not used in chi-square fitting.
    2. The diagonal ``t1==t2`` is excluded (homodyne parity): corrected
       diagonal values are interpolated estimates, not real physics.

    The returned vector keeps shape ``n_time * (n_time - 1)`` (off-diagonal
    pairs) for JIT-cache reuse; boundary entries contribute zero to the
    chi-square sum, giving an effective fit support of ``(n_time-1) *
    (n_time-2)`` real residuals.
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    n_time = c2_data.shape[0]
    indices = jnp.arange(n_time)
    boundary_mask = (indices[:, None] > 0) & (indices[None, :] > 0)
    residuals = (
        (c2_model - c2_data) * jnp.sqrt(weights) * boundary_mask.astype(c2_model.dtype)
    )
    non_diagonal = ~jnp.eye(n_time, dtype=bool)
    rows, cols = jnp.nonzero(non_diagonal, size=n_time * (n_time - 1))
    return residuals[rows, cols]


# Jacobian of residuals with respect to parameters (for NLSQ).
# jacfwd (forward-mode) does 14 JVP passes for 14 parameters,
# whereas jacobian (reverse-mode) would do ~N² backward passes
# (one per residual element).  For the XPCS use-case with N=200-500
# this is ~8,900x cheaper at N=500 (125K residuals vs 14 params).
_compute_residuals_jacobian_jit = jax.jit(jax.jacfwd(_compute_residuals_jit, argnums=0))


@jax.jit
def compute_chi_squared(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """JIT-compiled chi-squared computation.

    chi² = sum((c2_model - c2_data)² × weights)

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Weights (1/uncertainty²)
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        Chi-squared scalar
    """
    c2_model = compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)
    n_time = c2_data.shape[0]
    indices = jnp.arange(n_time)
    boundary_mask = (indices[:, None] > 0) & (indices[None, :] > 0)
    return jnp.sum(
        (c2_model - c2_data) ** 2 * weights * boundary_mask.astype(c2_model.dtype)
    )


def batch_chi_squared(
    params_batch: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray,
    contrast: float = 1.0,
    offset: float = 1.0,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Vectorized chi-squared over a batch of parameter sets.

    Uses ``jax.vmap`` for efficient parallel evaluation.  For large batches
    or large time grids, ``chunk_size`` limits simultaneous N×N allocations
    to prevent XLA memory exhaustion (each vmap'd evaluation allocates
    multiple N×N intermediate matrices).

    Args:
        params_batch: Parameter sets, shape ``(n_sets, 14)``.
        t: Time array, shape ``(N,)``.
        q: Scattering wavevector.
        dt: Time step.
        phi_angle: Detector phi angle.
        c2_data: Experimental data.
        weights: Weights.
        contrast: Speckle contrast.
        offset: Baseline offset.
        chunk_size: Max batch elements to vmap simultaneously.  ``None``
            (default) auto-selects based on time-grid size: ``max(1,
            200 // (N // 100))`` to keep peak memory under ~1.6 GB.

    Returns:
        Chi-squared values, shape ``(n_sets,)``.
    """
    n_sets = params_batch.shape[0]
    n_times = t.shape[0]

    def single_chi2(params: jnp.ndarray) -> jnp.ndarray:
        return compute_chi_squared(  # type: ignore[no-any-return]
            params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
        )

    if chunk_size is None:
        # Auto-select: each evaluation creates ~12 N×N float64 matrices
        # (half_tr, cumsum, integral matrix, ref/sample/cross terms, plus
        # XLA intermediates) → ~96 N² bytes per evaluation.
        # Target peak ≈ 1.6 GB → chunk_size ≈ 1.6e9 / (96 * N²).
        matrix_bytes = 96 * n_times * n_times
        chunk_size = max(1, int(1.6e9 / max(matrix_bytes, 1)))

    if n_sets <= chunk_size:
        return jax.vmap(single_chi2)(params_batch)

    # Chunked evaluation to bound peak memory
    chunks = []
    for start in range(0, n_sets, chunk_size):
        chunk = params_batch[start : start + chunk_size]
        chunks.append(jax.vmap(single_chi2)(chunk))
    return jnp.concatenate(chunks)


@jax.jit
def compute_multi_angle_residuals(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angles: jnp.ndarray,
    c2_data_batch: jnp.ndarray,
    weights_batch: jnp.ndarray,
    contrasts: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled residuals for multiple phi angles simultaneously.

    Args:
        params: Parameter array, shape (14,)
        t: Time array, shape (N,)
        q: Scattering wavevector
        dt: Time step
        phi_angles: Phi angles, shape (n_phi,)
        c2_data_batch: Experimental data, shape (n_phi, N, N)
        weights_batch: Weights, shape (n_phi, N, N)
        contrasts: Per-angle contrasts, shape (n_phi,)
        offsets: Per-angle offsets, shape (n_phi,)

    Returns:
        Stacked flattened residuals, shape (n_phi × N × (N-1),)
    """

    def single_angle_residual(
        phi: jnp.ndarray,
        c2_exp: jnp.ndarray,
        w: jnp.ndarray,
        c: jnp.ndarray,
        o: jnp.ndarray,
    ) -> jnp.ndarray:
        # Match _compute_residuals_jit: apply BOTH the t=0 boundary mask
        # and the diagonal exclusion before flattening. Previously this
        # path only excluded the diagonal, so joint multi-phi fits used
        # the t=0 row/col while single-phi fits did not, silently
        # changing the chi-square support between the two code paths.
        c2_model = compute_c2_heterodyne(params, t, q, dt, phi, c, o)
        n_time = c2_exp.shape[0]
        indices = jnp.arange(n_time)
        boundary_mask = (indices[:, None] > 0) & (indices[None, :] > 0)
        residuals = (
            (c2_model - c2_exp) * jnp.sqrt(w) * boundary_mask.astype(c2_model.dtype)
        )
        non_diagonal = ~jnp.eye(n_time, dtype=bool)
        rows, cols = jnp.nonzero(non_diagonal, size=n_time * (n_time - 1))
        return residuals[rows, cols]

    compute_all = jax.vmap(single_angle_residual, in_axes=(0, 0, 0, 0, 0))
    residuals_batch = compute_all(
        phi_angles, c2_data_batch, weights_batch, contrasts, offsets
    )
    return residuals_batch.ravel()


# Gradient of chi-squared with respect to parameters
compute_chi_squared_grad = jax.jit(jax.grad(compute_chi_squared, argnums=0))

# Hessian of chi-squared (for uncertainty estimation).
# Forward-over-reverse (jacfwd ∘ grad) is preferred over hessian()
# (reverse-over-reverse) for a (14,14) output: it runs 14 JVP passes
# over the gradient graph rather than 14 backward passes over 14
# backward passes, giving a ~14x reduction in graph size on CPU.
compute_chi_squared_hessian = jax.jit(
    jax.jacfwd(jax.grad(compute_chi_squared, argnums=0), argnums=0)
)


def compute_residuals_jacobian(
    params: jnp.ndarray,
    t: jnp.ndarray,
    q: float,
    dt: float,
    phi_angle: float,
    c2_data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute Jacobian of residuals with respect to parameters.

    Args:
        params: Parameter array, shape (14,)
        t: Time array
        q: Scattering wavevector
        dt: Time step
        phi_angle: Detector phi angle
        c2_data: Experimental correlation data
        weights: Optional weights (1/uncertainty²)
        contrast: Speckle contrast (beta), default 1.0
        offset: Baseline offset, default 1.0

    Returns:
        Jacobian matrix
    """
    if weights is None:
        weights = jnp.ones_like(c2_data)
    return _compute_residuals_jacobian_jit(  # type: ignore[no-any-return]
        params, t, q, dt, phi_angle, c2_data, weights, contrast, offset
    )
