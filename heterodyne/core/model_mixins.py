"""Reusable model behavior mixins for heterodyne correlation analysis.

Provides convenience wrappers around the stateless functions in
:mod:`heterodyne.core.theory` and :mod:`heterodyne.core.jax_backend`,
organized by physical domain (transport, fraction, velocity).

Each mixin expects the consuming class to provide ``t`` (time array),
``q`` (wavevector), and ``dt`` (time step) attributes — matching the
:class:`~heterodyne.core.heterodyne_model.HeterodyneModel` interface.

These mixins do **not** reimplement physics; they delegate to the
existing JIT-compiled functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from heterodyne.core.theory import (
    compute_cross_term_phase,
    compute_fraction,
    compute_normalization_factor,
    compute_time_integral_matrix,
    compute_transport_coefficient,
    compute_transport_integral_matrix,
    compute_velocity_field,
)
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class TransportMixin:
    """Mixin for transport coefficient and integral computations.

    Requires the consuming class to have:
        - ``t``: jnp.ndarray — time array, shape (N,)
        - ``q``: float — scattering wavevector
        - ``dt``: float — time step
    """

    t: jnp.ndarray
    q: float
    dt: float

    def compute_transport_rate(
        self,
        D0: float,
        alpha: float,
        offset: float,
    ) -> jnp.ndarray:
        """Compute pointwise transport rate J(t) = D0 * t^alpha + offset.

        Args:
            D0: Transport prefactor.
            alpha: Transport exponent.
            offset: Constant offset.

        Returns:
            Transport rate array, shape (N,).
        """
        return compute_transport_coefficient(self.t, D0, alpha, offset)

    def compute_transport_integral(
        self,
        D0: float,
        alpha: float,
        offset: float,
    ) -> jnp.ndarray:
        """Compute transport integral matrix |integral from t_i to t_j of J(t') dt'|.

        Uses cumsum for O(N) efficiency.

        Args:
            D0: Transport prefactor.
            alpha: Transport exponent.
            offset: Transport rate offset.

        Returns:
            Transport integral matrix, shape (N, N), non-negative and symmetric.
        """
        return compute_transport_integral_matrix(
            self.t, D0, alpha, offset, self.dt
        )

    def compute_half_transport(
        self,
        D0: float,
        alpha: float,
        offset: float,
    ) -> jnp.ndarray:
        """Compute half-transport matrix exp(-0.5 * q^2 * integral of J).

        This is the fundamental building block for the two-time correlation:
        self-terms use half_tr^2 to recover exp(-q^2 * integral of J), and
        cross-terms multiply half_tr_ref * half_tr_sample.

        Args:
            D0: Transport prefactor.
            alpha: Transport exponent.
            offset: Transport rate offset.

        Returns:
            Half-transport matrix, shape (N, N), values in (0, 1].
        """
        J_integral = self.compute_transport_integral(D0, alpha, offset)
        q2 = self.q * self.q
        log_half_tr = jnp.clip(-0.5 * q2 * J_integral, -700.0, 0.0)
        return jnp.exp(log_half_tr)


class FractionMixin:
    """Mixin for sample fraction computations.

    Requires the consuming class to have:
        - ``t``: jnp.ndarray — time array, shape (N,)
    """

    t: jnp.ndarray

    def compute_fraction_evolution(
        self,
        f0: float,
        f1: float,
        f2: float,
        f3: float,
    ) -> jnp.ndarray:
        """Compute sample fraction f_s(t) = f0 * exp(f1 * (t - f2)) + f3.

        Result is clipped to [0, 1] for physical validity.

        Args:
            f0: Fraction amplitude.
            f1: Exponential rate.
            f2: Time shift.
            f3: Baseline offset.

        Returns:
            Sample fraction array, shape (N,), values in [0, 1].
        """
        return compute_fraction(self.t, f0, f1, f2, f3)

    def compute_fraction_matrices(
        self,
        f_sample: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute fraction outer-product matrices for correlation terms.

        Constructs the four fraction matrices needed by the correlation:
            - f_ref_mat[i,j]    = f_r(t_i) * f_r(t_j)
            - f_sample_mat[i,j] = f_s(t_i) * f_s(t_j)
            - f_cross_mat[i,j]  = (f_r*f_s)(t_i) * (f_r*f_s)(t_j)
            - normalization[i,j]= (f_s^2+f_r^2)(t_i) * (f_s^2+f_r^2)(t_j)

        Args:
            f_sample: Sample fraction array, shape (N,), values in [0, 1].

        Returns:
            Tuple of (f_ref_mat, f_sample_mat, f_cross_mat, normalization),
            each shape (N, N).
        """
        f_ref = 1.0 - f_sample

        f_ref_mat = f_ref[:, None] * f_ref[None, :]
        f_sample_mat = f_sample[:, None] * f_sample[None, :]

        f_cross_vec = f_ref * f_sample
        f_cross_mat = f_cross_vec[:, None] * f_cross_vec[None, :]

        normalization = compute_normalization_factor(f_sample, f_sample)

        return f_ref_mat, f_sample_mat, f_cross_mat, normalization


class VelocityMixin:
    """Mixin for velocity field and flow computations.

    Requires the consuming class to have:
        - ``t``: jnp.ndarray — time array, shape (N,)
        - ``q``: float — scattering wavevector
        - ``dt``: float — time step
    """

    t: jnp.ndarray
    q: float
    dt: float

    def compute_velocity_field(
        self,
        v0: float,
        beta: float,
        v_offset: float,
    ) -> jnp.ndarray:
        """Compute velocity field v(t) = v0 * t^beta + v_offset.

        Args:
            v0: Velocity prefactor.
            beta: Velocity exponent.
            v_offset: Constant velocity offset.

        Returns:
            Velocity array, shape (N,).
        """
        return compute_velocity_field(self.t, v0, beta, v_offset)

    def compute_velocity_integral(
        self,
        v0: float,
        beta: float,
        v_offset: float,
    ) -> jnp.ndarray:
        """Compute velocity integral matrix integral from t_i to t_j of v(t') dt'.

        Uses cumsum broadcasting for O(N) efficiency.

        Args:
            v0: Velocity prefactor.
            beta: Velocity exponent.
            v_offset: Constant velocity offset.

        Returns:
            Velocity integral matrix, shape (N, N).
        """
        velocity = compute_velocity_field(self.t, v0, beta, v_offset)
        return compute_time_integral_matrix(velocity, self.dt)

    def compute_phase_factor(
        self,
        v_integral: jnp.ndarray,
        phi: float,
    ) -> jnp.ndarray:
        """Compute cross-term phase factor cos(q * cos(phi) * integral of v dt).

        Args:
            v_integral: Velocity integral matrix, shape (N, N).
            phi: Total flow angle (phi_angle + phi0) in degrees.

        Returns:
            Phase factor matrix cos(phase), shape (N, N).
        """
        phase = compute_cross_term_phase(v_integral, self.q, phi)
        return jnp.cos(phase)
