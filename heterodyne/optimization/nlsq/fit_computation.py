"""Fit Computation Utilities for Heterodyne NLSQ Results.

Provides functions for computing theoretical fits from NLSQ optimization
results, including batch computation, per-angle scaling, and parameter
extraction.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.core.jax_backend import compute_c2_heterodyne
from heterodyne.core.models import ANALYSIS_MODES
from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


def compute_c2_batch(
    params: jnp.ndarray,
    t: jnp.ndarray,
    phi_angles: jnp.ndarray,
    q: float,
    dt: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute c2 for multiple phi angles via jax.vmap.

    Args:
        params: 14-parameter array
        t: Time array, shape (N,)
        phi_angles: Phi angles in degrees, shape (n_phi,)
        q: Scattering wavevector
        dt: Time step
        contrast: Speckle contrast
        offset: Baseline offset

    Returns:
        c2 matrices, shape (n_phi, N, N)
    """

    def compute_single_angle(phi_val: jnp.ndarray) -> jnp.ndarray:
        return compute_c2_heterodyne(params, t, q, dt, phi_val, contrast, offset)

    compute_all = jax.vmap(compute_single_angle)
    return compute_all(phi_angles)  # type: ignore[no-any-return]


def compute_c2_batch_with_per_angle_scaling(
    params: jnp.ndarray,
    t: jnp.ndarray,
    phi_angles: jnp.ndarray,
    q: float,
    dt: float,
    contrasts: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute c2 with per-angle contrast/offset via jax.vmap.

    Args:
        params: 14-parameter array
        t: Time array, shape (N,)
        phi_angles: Phi angles in degrees, shape (n_phi,)
        q: Scattering wavevector
        dt: Time step
        contrasts: Per-angle contrasts, shape (n_phi,)
        offsets: Per-angle offsets, shape (n_phi,)

    Returns:
        c2 matrices, shape (n_phi, N, N)
    """

    def compute_single(
        phi_val: jnp.ndarray,
        contrast_val: jnp.ndarray,
        offset_val: jnp.ndarray,
    ) -> jnp.ndarray:
        return compute_c2_heterodyne(
            params, t, q, dt, phi_val, contrast_val, offset_val
        )

    compute_all = jax.vmap(compute_single, in_axes=(0, 0, 0))
    return compute_all(phi_angles, contrasts, offsets)  # type: ignore[no-any-return]


def solve_lstsq_batch(
    theory_batch: jnp.ndarray,
    exp_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch least-squares for per-angle contrast/offset.

    For each angle, solves: exp = contrast * theory + offset
    via linear least squares.

    Args:
        theory_batch: Theory values flattened, shape (n_phi, n_elements)
        exp_batch: Experimental values flattened, shape (n_phi, n_elements)

    Returns:
        Tuple of (contrasts, offsets), each shape (n_phi,)
    """

    def solve_single(
        theory_flat: jnp.ndarray,
        exp_flat: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        A = jnp.column_stack([theory_flat, jnp.ones_like(theory_flat)])
        solution, _, _, _ = jnp.linalg.lstsq(A, exp_flat, rcond=None)
        return solution[0], solution[1]

    solve_all = jax.vmap(solve_single, in_axes=(0, 0))
    contrasts, offsets = solve_all(theory_batch, exp_batch)
    return contrasts, offsets  # type: ignore[return-value]


def get_physical_param_count(analysis_mode: str) -> int:
    """Get number of physical parameters for an analysis mode.

    Args:
        analysis_mode: One of "static_ref", "static_both", "two_component"

    Returns:
        Number of physical parameters

    Raises:
        ValueError: If mode is unknown
    """
    if analysis_mode not in ANALYSIS_MODES:
        valid = ", ".join(sorted(ANALYSIS_MODES))
        raise ValueError(
            f"Unknown analysis_mode: '{analysis_mode}'. Expected one of: {valid}"
        )
    return len(ANALYSIS_MODES[analysis_mode])


def normalize_analysis_mode(
    mode: str | None,
    n_params: int,
    n_angles: int,
) -> str:
    """Resolve analysis mode, inferring from parameter counts if needed.

    Args:
        mode: Explicit mode string or None
        n_params: Total number of parameters
        n_angles: Number of phi angles

    Returns:
        Normalized mode string
    """
    if mode:
        mode_lower = mode.lower()
        if mode_lower in ANALYSIS_MODES:
            return mode_lower

    # Infer from parameter counts
    for candidate_mode, param_names in ANALYSIS_MODES.items():
        n_phys = len(param_names)
        # Check per-angle layout: 2*n_angles scaling + n_phys physics
        if n_params in {n_phys + 2, 2 * n_angles + n_phys}:
            return candidate_mode

    logger.debug(
        "Unable to infer analysis_mode from params=%s angles=%s; defaulting to two_component",
        n_params,
        n_angles,
    )
    return "two_component"


def extract_parameters_from_result(
    parameters: np.ndarray,
    n_angles: int,
    analysis_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Extract contrast, offset, and physical parameters from result.

    Handles both per-angle and scalar parameter layouts.

    Args:
        parameters: Full parameter array from optimization
        n_angles: Number of phi angles
        analysis_mode: Analysis mode string

    Returns:
        Tuple of (contrasts, offsets, physical_params, scalar_expansion_used)

    Raises:
        ValueError: If parameter count is invalid
    """
    n_params = len(parameters)
    n_physical = get_physical_param_count(analysis_mode)
    expected_per_angle = 2 * n_angles + n_physical

    scalar_expansion = False

    if n_params == expected_per_angle:
        # Per-angle layout: [contrast_0..N, offset_0..N, physical...]
        contrasts = parameters[:n_angles]
        offsets = parameters[n_angles : 2 * n_angles]
        physical_params = parameters[2 * n_angles :]
    elif n_params == (n_physical + 2):
        # Scalar layout: [contrast, offset, physical...]
        logger.warning(
            "Scalar contrast/offset (param count %d). Expanding to %d angles.",
            n_params,
            n_angles,
        )
        scalar_expansion = True
        contrasts = np.full(n_angles, float(parameters[0]))
        offsets = np.full(n_angles, float(parameters[1]))
        physical_params = parameters[2:]
    elif n_params == n_physical:
        # Physics-only layout (no scaling params)
        logger.info("Physics-only parameters (%d). Using default scaling.", n_params)
        scalar_expansion = True
        contrasts = np.ones(n_angles)
        offsets = np.ones(n_angles)
        physical_params = parameters
    else:
        raise ValueError(
            f"Parameter count mismatch! Expected {expected_per_angle} "
            f"(2\u00d7{n_angles} scaling + {n_physical} physical) or "
            f"{n_physical + 2} (scalar) or {n_physical} (physics-only), "
            f"got {n_params}."
        )

    return contrasts, offsets, physical_params, scalar_expansion


def compute_theoretical_fits(
    result: Any,
    data: dict[str, Any],
    metadata: dict[str, Any],
    *,
    analysis_mode: str | None = None,
    include_solver_surface: bool = True,
) -> dict[str, Any]:
    """Compute theoretical fits with per-angle least-squares scaling.

    Generates theoretical correlation functions using optimized parameters,
    then applies per-angle scaling via least squares.

    Args:
        result: NLSQ result with .parameters attribute
        data: Experimental data dict with keys:
            - phi_angles_list: phi angles (degrees)
            - c2_exp: experimental c2, shape (n_phi, N, N)
            - t: time array, shape (N,)
        metadata: Dict with keys: dt, q
        analysis_mode: Optional mode override
        include_solver_surface: Include solver-scaled surface in output

    Returns:
        Dictionary with keys:
        - c2_theoretical_raw: Raw theory, shape (n_phi, N, N)
        - c2_theoretical_scaled: Post-hoc lstsq scaled, shape (n_phi, N, N)
        - c2_solver_scaled: Solver surface (if requested)
        - per_angle_scaling: Post-hoc lstsq scaling, shape (n_phi, 2)
        - per_angle_scaling_solver: Solver scaling, shape (n_phi, 2)
        - residuals: Exp minus scaled theory
        - scalar_per_angle_expansion: Whether scalar expansion was used
    """
    phi_angles = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])
    t = np.asarray(data["t"])
    n_angles = len(phi_angles)

    n_params = len(result.parameters)

    # Normalize analysis mode
    normalized_mode = normalize_analysis_mode(
        analysis_mode or getattr(result, "analysis_mode", None),
        n_params,
        n_angles,
    )

    # Extract parameters
    fitted_contrasts, fitted_offsets, physical_params, scalar_expansion = (
        extract_parameters_from_result(result.parameters, n_angles, normalized_mode)
    )

    logger.info("Per-angle scaling: %d angles, mode=%s", n_angles, normalized_mode)

    # Extract metadata
    dt = metadata.get("dt")
    if dt is None:
        raise ValueError(
            "dt (time step) is required for compute_theoretical_fits() "
            "but was not found in metadata."
        )
    dt = float(dt)
    q = metadata.get("q")
    if q is None:
        raise ValueError("q (wavevector) is required but was not found")
    q = float(q)

    # Convert to JAX arrays
    t_jax = jnp.array(t)
    phi_jax = jnp.array(phi_angles)
    params_jax = jnp.array(physical_params)

    # Pad physics params to 14 if using a reduced mode
    if len(physical_params) < 14:
        mode_names = ANALYSIS_MODES[normalized_mode]
        full_defaults = np.array(
            [
                1e4,
                0.0,
                0.0,  # ref transport: D0_ref, alpha_ref, D_offset_ref
                1e4,
                0.0,
                0.0,  # sample transport: D0_sample, alpha_sample, D_offset_sample
                1e3,
                0.0,
                0.0,  # velocity: v0, beta, v_offset
                0.5,
                0.0,
                0.0,
                0.0,  # fraction: f0, f1, f2, f3
                0.0,  # phi0
            ],
            dtype=np.float64,
        )
        for i, name in enumerate(mode_names):
            idx = ALL_PARAM_NAMES.index(name)
            full_defaults[idx] = physical_params[i]
        params_jax = jnp.array(full_defaults)

    # Compute RAW theory for all angles (contrast=1, offset=1 to expose shape)
    c2_theoretical_raw = compute_c2_batch(
        params=params_jax,
        t=t_jax,
        phi_angles=phi_jax,
        q=q,
        dt=dt,
        contrast=1.0,
        offset=1.0,
    )
    c2_theoretical_raw = np.asarray(c2_theoretical_raw)

    # Compute solver surface with fitted per-angle scaling if requested
    c2_solver_surface: np.ndarray | None = None
    if include_solver_surface:
        c2_solver_surface = np.asarray(
            compute_c2_batch_with_per_angle_scaling(
                params=params_jax,
                t=t_jax,
                phi_angles=phi_jax,
                q=q,
                dt=dt,
                contrasts=jnp.array(fitted_contrasts),
                offsets=jnp.array(fitted_offsets),
            )
        )

    # Batch least-squares scaling: solve exp = contrast * theory + offset per angle
    # Flatten to (n_angles, N*N)
    n_elements = c2_theoretical_raw.shape[1] * c2_theoretical_raw.shape[2]
    theory_flat = jnp.array(c2_theoretical_raw.reshape(n_angles, n_elements))
    exp_flat = jnp.array(c2_exp.reshape(n_angles, n_elements))

    contrasts_lstsq, offsets_lstsq = solve_lstsq_batch(theory_flat, exp_flat)
    contrasts_lstsq = np.asarray(contrasts_lstsq)
    offsets_lstsq = np.asarray(offsets_lstsq)

    # Apply post-hoc scaling: c2_scaled[i] = contrast[i] * c2_raw[i] + offset[i]
    c2_theoretical_fitted = (
        contrasts_lstsq[:, None, None] * c2_theoretical_raw
        + offsets_lstsq[:, None, None]
    )

    per_angle_scaling = np.column_stack((contrasts_lstsq, offsets_lstsq))
    solver_scaling = np.column_stack((fitted_contrasts, fitted_offsets))

    logger.debug(
        "Batch lstsq - contrasts: mean=%.4f, offsets: mean=%.4f",
        np.nanmean(contrasts_lstsq),
        np.nanmean(offsets_lstsq),
    )

    residuals = c2_exp - c2_theoretical_fitted

    logger.info("Computed theoretical fits for %d angles", n_angles)

    return {
        "c2_theoretical_raw": c2_theoretical_raw,
        "c2_theoretical_scaled": c2_theoretical_fitted,
        "c2_solver_scaled": c2_solver_surface,
        "per_angle_scaling": per_angle_scaling,
        "per_angle_scaling_solver": solver_scaling,
        "residuals": residuals,
        "scalar_per_angle_expansion": scalar_expansion,
    }
