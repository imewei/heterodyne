"""Fourier Reparameterization for Per-Angle Scaling Parameters.

This module replaces n_phi independent per-angle contrast/offset values
with truncated Fourier series, dramatically reducing structural degeneracy
in joint multi-angle fits.

Adapted from homodyne Anti-Degeneracy Defense System.  The Fourier basis is
model-agnostic (captures smooth phi-variation); here it regularises the
angle-dependent velocity phase term cos(q·cos(φ)·∫v dt) rather than the
homodyne shear sinc term.

Mathematical Formulation
------------------------
contrast(phi) = c0 + sum_k[ck*cos(k*phi) + sk*sin(k*phi)]    for k=1..order
offset(phi)   = o0 + sum_k[ok*cos(k*phi) + tk*sin(k*phi)]    for k=1..order

For order=2:
- Contrast: 5 coefficients [c0, c1, s1, c2, s2]
- Offset: 5 coefficients [o0, o1, t1, o2, t2]
- Total: 10 Fourier coefficients vs 2*n_phi independent params

Parameter Count Comparison::

    n_phi | Independent | Fourier (order=2) | Reduction
    ------|-------------|-------------------|----------
      2   |     4       |        4          |    0%
      3   |     6       |        6          |    0%
     10   |    20       |       10          |   50%
     23   |    46       |       10          |   78%
    100   |   200       |       10          |   95%

Note: For n_phi <= 2*(order+1), independent mode is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from heterodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FourierReparamConfig:
    """Configuration for Fourier reparameterization.

    Attributes:
        mode: Per-angle parameter mode:
            - "independent": Use n_phi independent contrast/offset values
            - "fourier": Use truncated Fourier series
            - "auto": Use Fourier when n_phi > auto_threshold
        fourier_order: Number of Fourier harmonics. Default 2.
            order=2 gives 5 coefficients per parameter (c0, c1, s1, c2, s2).
        auto_threshold: Use Fourier when n_phi > this threshold in auto mode.
        c0_bounds: Bounds for mean contrast coefficient.
        ck_bounds: Bounds for harmonic contrast amplitudes.
        o0_bounds: Bounds for mean offset coefficient.
        ok_bounds: Bounds for harmonic offset amplitudes.
    """

    mode: Literal["independent", "fourier", "auto"] = "auto"
    fourier_order: int = 2
    auto_threshold: int = 6

    # Bounds for Fourier coefficients
    c0_bounds: tuple[float, float] = (0.01, 1.0)  # Mean contrast
    ck_bounds: tuple[float, float] = (-0.2, 0.2)  # Harmonic amplitudes
    o0_bounds: tuple[float, float] = (0.5, 1.5)  # Mean offset
    ok_bounds: tuple[float, float] = (-0.3, 0.3)  # Harmonic amplitudes

    @classmethod
    def from_dict(cls, config_dict: dict) -> FourierReparamConfig:
        """Create config from dictionary."""
        return cls(
            mode=config_dict.get("per_angle_mode", "auto"),
            fourier_order=config_dict.get("fourier_order", 2),
            auto_threshold=config_dict.get("fourier_auto_threshold", 6),
            c0_bounds=tuple(config_dict.get("c0_bounds", (0.01, 1.0))),
            ck_bounds=tuple(config_dict.get("ck_bounds", (-0.2, 0.2))),
            o0_bounds=tuple(config_dict.get("o0_bounds", (0.5, 1.5))),
            ok_bounds=tuple(config_dict.get("ok_bounds", (-0.3, 0.3))),
        )


class FourierReparameterizer:
    """Handles conversion between Fourier coefficients and per-angle values.

    Core functionality:
    1. Convert per-angle values to Fourier coefficients (initialization)
    2. Convert Fourier coefficients to per-angle values (model evaluation)
    3. Compute Jacobian for covariance transformation

    The Fourier basis ensures smooth variation of contrast/offset with angle,
    preventing the optimizer from using per-angle parameters to absorb
    angle-dependent physical signals.

    Parameters
    ----------
    phi_angles : np.ndarray
        Unique phi angles in radians, shape (n_phi,).
    config : FourierReparamConfig
        Fourier configuration.

    Attributes
    ----------
    n_phi : int
        Number of unique phi angles.
    n_coeffs : int
        Total number of Fourier coefficients (contrast + offset).
    n_coeffs_per_param : int
        Coefficients per parameter type (contrast or offset).
    use_fourier : bool
        Whether Fourier mode is active.

    Examples
    --------
    >>> phi_angles = np.linspace(-np.pi, np.pi, 23)
    >>> config = FourierReparamConfig(mode="fourier", fourier_order=2)
    >>> fourier = FourierReparameterizer(phi_angles, config)
    >>> contrast = np.full(23, 0.3)
    >>> offset = np.full(23, 1.0)
    >>> fourier_coeffs = fourier.per_angle_to_fourier(contrast, offset)
    >>> contrast_out, offset_out = fourier.fourier_to_per_angle(fourier_coeffs)
    """

    def __init__(self, phi_angles: np.ndarray, config: FourierReparamConfig) -> None:
        self.phi_angles = np.asarray(phi_angles, dtype=np.float64)
        self.config = config
        self.n_phi = len(phi_angles)
        self._basis_matrix: np.ndarray | None = None

        # Determine effective mode
        self.use_fourier = self._determine_mode()

        if self.use_fourier:
            # Number of coefficients per parameter: c0 + order*(ck, sk)
            self.n_coeffs_per_param = 1 + 2 * config.fourier_order
            self.n_coeffs = 2 * self.n_coeffs_per_param  # contrast + offset

            # Precompute Fourier basis matrix
            self._basis_matrix = self._compute_basis_matrix()

            # Explicit rcond for lstsq stability
            self._rcond = (
                max(self.n_phi, self.n_coeffs_per_param) * np.finfo(np.float64).eps
            )

            logger.info(
                "Fourier reparameterization enabled: "
                "%d coefficients for %d angles "
                "(order=%d, rcond=%.2e)",
                self.n_coeffs,
                self.n_phi,
                config.fourier_order,
                self._rcond,
            )
        else:
            # Independent mode: n_phi per parameter
            self.n_coeffs_per_param = self.n_phi
            self.n_coeffs = 2 * self.n_phi
            self._basis_matrix = None
            self._rcond = 0.0

            logger.info(
                "Independent per-angle mode: %d parameters for %d angles",
                self.n_coeffs,
                self.n_phi,
            )

    def _determine_mode(self) -> bool:
        """Determine whether to use Fourier mode."""
        min_angles = 1 + 2 * self.config.fourier_order

        if self.config.mode == "fourier":
            if self.n_phi < min_angles:
                logger.warning(
                    "Fourier mode requested but n_phi=%d < min_angles=%d. "
                    "Falling back to independent mode.",
                    self.n_phi,
                    min_angles,
                )
                return False
            return True

        elif self.config.mode == "independent":
            return False

        else:  # auto
            use_fourier = (
                self.n_phi > self.config.auto_threshold and self.n_phi >= min_angles
            )
            if use_fourier:
                logger.debug(
                    "Auto mode: using Fourier (n_phi=%d > threshold=%d)",
                    self.n_phi,
                    self.config.auto_threshold,
                )
            else:
                logger.debug(
                    "Auto mode: using independent (n_phi=%d <= threshold=%d)",
                    self.n_phi,
                    self.config.auto_threshold,
                )
            return use_fourier

    def _compute_basis_matrix(self) -> np.ndarray:
        """Compute Fourier basis matrix B where values = B @ coeffs.

        Returns:
            Basis matrix of shape (n_phi, n_coeffs_per_param).
        """
        order = self.config.fourier_order

        B = np.zeros((self.n_phi, self.n_coeffs_per_param))
        B[:, 0] = 1.0  # c0 term (constant)

        col = 1
        for k in range(1, order + 1):
            B[:, col] = np.cos(k * self.phi_angles)  # ck term
            B[:, col + 1] = np.sin(k * self.phi_angles)  # sk term
            col += 2

        return B

    def get_basis_matrix(self) -> np.ndarray | None:
        """Get the Fourier basis matrix.

        Returns:
            Basis matrix of shape (n_phi, n_coeffs_per_param) if Fourier mode,
            None if independent mode. Satisfies: per_angle_values = B @ coeffs.
        """
        return self._basis_matrix

    @property
    def order(self) -> int:
        """Fourier order (number of harmonics)."""
        return self.config.fourier_order

    def fourier_to_per_angle(
        self,
        fourier_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Fourier coefficients to per-angle contrast/offset.

        Args:
            fourier_coeffs: Shape (n_coeffs,) =
                [c0,c1,s1,c2,s2,...,o0,o1,t1,o2,t2,...].

        Returns:
            (contrast, offset) each of shape (n_phi,).

        Raises:
            ValueError: If fourier_coeffs has wrong shape.
        """
        fourier_coeffs = np.asarray(fourier_coeffs, dtype=np.float64)
        if fourier_coeffs.ndim != 1:
            raise ValueError(
                f"fourier_coeffs must be 1D array, got shape {fourier_coeffs.shape}"
            )
        if len(fourier_coeffs) != self.n_coeffs:
            raise ValueError(
                f"Expected {self.n_coeffs} Fourier coefficients, "
                f"got {len(fourier_coeffs)}"
            )

        if not self.use_fourier:
            # Independent mode: first half is contrast, second half is offset
            contrast = fourier_coeffs[: self.n_phi].copy()
            offset = fourier_coeffs[self.n_phi :].copy()
            return contrast, offset

        n_half = self.n_coeffs_per_param
        contrast_coeffs = fourier_coeffs[:n_half]
        offset_coeffs = fourier_coeffs[n_half:]

        contrast = self._basis_matrix @ contrast_coeffs
        offset = self._basis_matrix @ offset_coeffs

        return contrast, offset

    def per_angle_to_fourier(
        self,
        contrast: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        """Convert per-angle values to Fourier coefficients.

        Uses least squares fitting when n_phi > n_coeffs_per_param.

        Args:
            contrast: Per-angle contrast values, shape (n_phi,).
            offset: Per-angle offset values, shape (n_phi,).

        Returns:
            Fourier coefficients, shape (n_coeffs,).
        """
        contrast = np.asarray(contrast, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)

        if len(contrast) != self.n_phi:
            raise ValueError(
                f"Expected {self.n_phi} contrast values, got {len(contrast)}"
            )
        if len(offset) != self.n_phi:
            raise ValueError(f"Expected {self.n_phi} offset values, got {len(offset)}")

        if not self.use_fourier:
            return np.concatenate([contrast, offset])

        # Least squares: B @ coeffs = values
        contrast_coeffs, residuals_c, _, _ = np.linalg.lstsq(
            self._basis_matrix,
            contrast,
            rcond=float(self._rcond),
        )
        offset_coeffs, residuals_o, _, _ = np.linalg.lstsq(
            self._basis_matrix,
            offset,
            rcond=float(self._rcond),
        )

        if len(residuals_c) > 0 and residuals_c[0] > 0.01:
            rms_c = np.sqrt(residuals_c[0] / self.n_phi)
            logger.debug("Fourier fit residual (contrast): %.4f", rms_c)
        if len(residuals_o) > 0 and residuals_o[0] > 0.01:
            rms_o = np.sqrt(residuals_o[0] / self.n_phi)
            logger.debug("Fourier fit residual (offset): %.4f", rms_o)

        return np.concatenate([contrast_coeffs, offset_coeffs])

    def get_jacobian_transform(self) -> np.ndarray:
        """Get Jacobian of transformation: d(per_angle)/d(fourier).

        Used for covariance transformation back to per-angle space:
            Cov_per_angle = J @ Cov_fourier @ J.T

        Returns:
            Jacobian matrix of shape (2*n_phi, n_coeffs).
        """
        if not self.use_fourier:
            return np.eye(self.n_coeffs)

        n_half = self.n_coeffs_per_param
        jacobian = np.zeros((2 * self.n_phi, self.n_coeffs))

        # d(contrast_i)/d(contrast_coeffs)
        jacobian[: self.n_phi, :n_half] = self._basis_matrix

        # d(offset_i)/d(offset_coeffs)
        jacobian[self.n_phi :, n_half:] = self._basis_matrix

        return jacobian

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds for Fourier coefficients.

        Returns:
            (lower, upper) each of shape (n_coeffs,).
        """
        if not self.use_fourier:
            contrast_lower = np.full(self.n_phi, self.config.c0_bounds[0])
            contrast_upper = np.full(self.n_phi, self.config.c0_bounds[1])
            offset_lower = np.full(self.n_phi, self.config.o0_bounds[0])
            offset_upper = np.full(self.n_phi, self.config.o0_bounds[1])

            lower = np.concatenate([contrast_lower, offset_lower])
            upper = np.concatenate([contrast_upper, offset_upper])
            return lower, upper

        n_half = self.n_coeffs_per_param

        lower = np.zeros(self.n_coeffs)
        upper = np.zeros(self.n_coeffs)

        # Contrast coefficients
        lower[0] = self.config.c0_bounds[0]  # c0 (mean contrast)
        upper[0] = self.config.c0_bounds[1]
        for i in range(1, n_half):
            lower[i] = self.config.ck_bounds[0]  # ck, sk harmonics
            upper[i] = self.config.ck_bounds[1]

        # Offset coefficients
        lower[n_half] = self.config.o0_bounds[0]  # o0 (mean offset)
        upper[n_half] = self.config.o0_bounds[1]
        for i in range(n_half + 1, self.n_coeffs):
            lower[i] = self.config.ok_bounds[0]  # ok, tk harmonics
            upper[i] = self.config.ok_bounds[1]

        return lower, upper

    def get_initial_coefficients(
        self,
        contrast_init: float | np.ndarray,
        offset_init: float | np.ndarray,
    ) -> np.ndarray:
        """Get initial Fourier coefficients from initial values.

        Args:
            contrast_init: Initial contrast (scalar for uniform, array for per-angle).
            offset_init: Initial offset (scalar for uniform, array for per-angle).

        Returns:
            Initial Fourier coefficients, shape (n_coeffs,).
        """
        if np.isscalar(contrast_init):
            contrast = np.full(self.n_phi, float(contrast_init))
        else:
            contrast = np.asarray(contrast_init)

        if np.isscalar(offset_init):
            offset = np.full(self.n_phi, float(offset_init))
        else:
            offset = np.asarray(offset_init)

        return self.per_angle_to_fourier(contrast, offset)

    def get_coefficient_labels(self) -> list[str]:
        """Get parameter labels for Fourier coefficients.

        Returns:
            List of parameter labels, length n_coeffs.
        """
        if not self.use_fourier:
            labels = [f"contrast[{i}]" for i in range(self.n_phi)]
            labels += [f"offset[{i}]" for i in range(self.n_phi)]
            return labels

        labels = ["contrast_c0"]
        for k in range(1, self.config.fourier_order + 1):
            labels.append(f"contrast_c{k}")
            labels.append(f"contrast_s{k}")

        labels.append("offset_c0")
        for k in range(1, self.config.fourier_order + 1):
            labels.append(f"offset_c{k}")
            labels.append(f"offset_s{k}")

        return labels

    def to_fourier(self, per_angle_values: np.ndarray) -> np.ndarray:
        """Convert a single per-angle array to Fourier coefficients.

        Convenience method for one parameter group at a time.

        Args:
            per_angle_values: Per-angle values, shape (n_phi,).

        Returns:
            Fourier coefficients, shape (n_coeffs_per_param,).
        """
        per_angle_values = np.asarray(per_angle_values, dtype=np.float64)
        if len(per_angle_values) != self.n_phi:
            raise ValueError(
                f"Expected {self.n_phi} values, got {len(per_angle_values)}"
            )

        if not self.use_fourier:
            return per_angle_values.copy()

        coeffs, _, _, _ = np.linalg.lstsq(
            self._basis_matrix,
            per_angle_values,
            rcond=self._rcond,
        )
        return coeffs

    def from_fourier(self, fourier_coeffs: np.ndarray) -> np.ndarray:
        """Convert Fourier coefficients to per-angle values for one group.

        Args:
            fourier_coeffs: Fourier coefficients, shape (n_coeffs_per_param,).

        Returns:
            Per-angle values, shape (n_phi,).
        """
        fourier_coeffs = np.asarray(fourier_coeffs, dtype=np.float64)
        if len(fourier_coeffs) != self.n_coeffs_per_param:
            raise ValueError(
                f"Expected {self.n_coeffs_per_param} coefficients, "
                f"got {len(fourier_coeffs)}"
            )

        if not self.use_fourier:
            return fourier_coeffs.copy()

        return self._basis_matrix @ fourier_coeffs

    def get_diagnostics(self) -> dict:
        """Get Fourier reparameterization diagnostics."""
        return {
            "use_fourier": self.use_fourier,
            "mode": self.config.mode,
            "n_phi": self.n_phi,
            "n_coeffs": self.n_coeffs,
            "n_coeffs_per_param": self.n_coeffs_per_param,
            "fourier_order": (self.config.fourier_order if self.use_fourier else None),
            "reduction_ratio": (
                self.n_coeffs / (2 * self.n_phi) if self.use_fourier else 1.0
            ),
        }
