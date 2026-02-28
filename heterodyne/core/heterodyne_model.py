"""Main heterodyne model wrapper class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from heterodyne.config.parameter_manager import ParameterManager
from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.core.jax_backend import compute_c2_heterodyne, compute_residuals
from heterodyne.core.models import TwoComponentModel
from heterodyne.core.physics_factors import PhysicsFactors, create_physics_factors

if TYPE_CHECKING:
    pass


@dataclass
class HeterodyneModel:
    """Main heterodyne correlation model with stateful parameter management.

    This class provides a convenient interface for:
    - Managing model parameters through ParameterManager
    - Computing correlation matrices
    - Computing residuals for fitting
    - Accessing pre-computed physics factors

    Example:
        >>> model = HeterodyneModel.from_config(config)
        >>> c2 = model.compute_correlation(phi_angle=45.0)
        >>> residuals = model.compute_residuals(c2_data, phi_angle=45.0)
    """

    # Core model
    _model: TwoComponentModel = field(default_factory=TwoComponentModel)

    # Parameter management
    param_manager: ParameterManager = field(default_factory=ParameterManager)

    # Physics factors (pre-computed from config)
    _factors: PhysicsFactors | None = field(default=None)

    # Cached time array
    _t: jnp.ndarray | None = field(default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HeterodyneModel:
        """Create model from configuration dictionary.

        Args:
            config: Configuration with temporal, scattering, and parameters sections

        Returns:
            Configured HeterodyneModel
        """
        param_manager = ParameterManager.from_config(config)

        temporal = config.get("temporal", {})
        scattering = config.get("scattering", {})

        factors = create_physics_factors(
            n_times=int(temporal.get("time_length", 1000)),
            dt=float(temporal.get("dt", 1.0)),
            q=float(scattering.get("wavevector_q", 0.01)),
            phi_angle=0.0,
            t_start=float(temporal.get("t_start", 0.0)),
        )

        return cls(
            _model=TwoComponentModel(),
            param_manager=param_manager,
            _factors=factors,
            _t=factors.t,
        )

    @property
    def n_params(self) -> int:
        """Total number of model parameters (14)."""
        return 14

    @property
    def n_varying(self) -> int:
        """Number of varying parameters."""
        return self.param_manager.n_varying

    @property
    def param_names(self) -> tuple[str, ...]:
        """All parameter names in canonical order."""
        return ALL_PARAM_NAMES

    @property
    def varying_names(self) -> list[str]:
        """Names of varying parameters."""
        return self.param_manager.varying_names

    @property
    def q(self) -> float:
        """Scattering wavevector magnitude."""
        if self._factors is None:
            raise ValueError("Physics factors not initialized")
        return self._factors.q

    @property
    def dt(self) -> float:
        """Time step."""
        if self._factors is None:
            raise ValueError("Physics factors not initialized")
        return self._factors.dt

    @property
    def t(self) -> jnp.ndarray:
        """Time array."""
        if self._t is None:
            raise ValueError("Time array not initialized")
        return self._t

    @property
    def n_times(self) -> int:
        """Number of time points."""
        if self._factors is None:
            raise ValueError("Physics factors not initialized")
        return self._factors.n_times

    def get_params(self) -> np.ndarray:
        """Get current full parameter array.

        Returns:
            Array of shape (14,)
        """
        return self.param_manager.get_full_values()

    def get_params_dict(self) -> dict[str, float]:
        """Get current parameters as dictionary."""
        return self.param_manager.get_parameter_dict()

    def set_params(self, params: np.ndarray | dict[str, float]) -> None:
        """Set parameter values.

        Args:
            params: Either array of shape (14,) or dict with param names
        """
        self.param_manager.update_values(params)

    def compute_correlation(
        self,
        phi_angle: float = 0.0,
        params: np.ndarray | None = None,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute two-time correlation matrix.

        Args:
            phi_angle: Detector phi angle (degrees)
            params: Optional parameter array (uses stored values if None)
            contrast: Speckle contrast (beta), default 1.0
            offset: Baseline offset, default 1.0

        Returns:
            Correlation matrix c2(t1, t2), shape (N, N)
        """
        if params is None:
            params = self.get_params()

        return compute_c2_heterodyne(  # type: ignore[no-any-return]
            jnp.asarray(params),
            self.t,
            self.q,
            self.dt,
            phi_angle,
            contrast,
            offset,
        )

    def compute_residuals(
        self,
        c2_data: np.ndarray | jnp.ndarray,
        phi_angle: float = 0.0,
        params: np.ndarray | None = None,
        weights: np.ndarray | jnp.ndarray | None = None,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute residuals between model and data.

        Args:
            c2_data: Experimental correlation data
            phi_angle: Detector phi angle
            params: Optional parameter array
            weights: Optional weights (1/sigma²)
            contrast: Speckle contrast (beta), default 1.0
            offset: Baseline offset, default 1.0

        Returns:
            Flattened residual array
        """
        if params is None:
            params = self.get_params()

        return compute_residuals(
            jnp.asarray(params),
            self.t,
            self.q,
            self.dt,
            phi_angle,
            jnp.asarray(c2_data),
            jnp.asarray(weights) if weights is not None else None,
            contrast,
            offset,
        )

    def compute_g1_reference(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute reference g1 correlation.

        Args:
            params: Optional parameter array

        Returns:
            g1_ref array, shape (N,)
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_g1_reference(params, self.t, self.q)

    def compute_g1_sample(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute sample g1 correlation.

        Args:
            params: Optional parameter array

        Returns:
            g1_sample array, shape (N,)
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_g1_sample(params, self.t, self.q)

    def compute_fraction(self, params: np.ndarray | None = None) -> jnp.ndarray:
        """Compute sample fraction evolution.

        Args:
            params: Optional parameter array

        Returns:
            f_sample array, shape (N,)
        """
        if params is None:
            params = self.get_params()
        return self._model.compute_fraction(params, self.t)

    def create_residual_function(
        self,
        c2_data: np.ndarray | jnp.ndarray,
        phi_angle: float,
        weights: np.ndarray | jnp.ndarray | None = None,
    ) -> Any:
        """Create a residual function for optimization.

        Returns a function that takes varying parameters and returns residuals.

        Args:
            c2_data: Experimental correlation data
            phi_angle: Detector phi angle
            weights: Optional weights

        Returns:
            Callable that maps varying params -> residuals
        """
        c2_jax = jnp.asarray(c2_data)
        weights_jax = (
            jnp.asarray(weights) if weights is not None
            else jnp.ones_like(c2_jax)
        )
        t = self.t
        q = self.q
        dt = self.dt

        varying_idx_jax = jnp.array(self.param_manager.varying_indices)
        fixed_values_jax = jnp.array(self.param_manager.get_full_values())

        @jax.jit
        def residual_fn(varying_params: jnp.ndarray) -> jnp.ndarray:
            # Reconstruct full params
            full_params = fixed_values_jax.at[varying_idx_jax].set(varying_params)

            return compute_residuals(
                full_params, t, q, dt, phi_angle, c2_jax, weights_jax
            )

        return residual_fn

    def summary(self) -> str:
        """Return summary of model configuration.

        Returns:
            Multi-line summary string
        """
        lines = [
            "HeterodyneModel Summary",
            "=" * 40,
            f"Time points: {self.n_times}",
            f"Time step: {self.dt}",
            f"Wavevector q: {self.q}",
            f"Total params: {self.n_params}",
            f"Varying params: {self.n_varying}",
            "",
            "Current Parameters:",
            "-" * 40,
        ]

        params = self.get_params_dict()
        for name in ALL_PARAM_NAMES:
            vary = "vary" if name in self.varying_names else "fixed"
            lines.append(f"  {name:18s}: {params[name]:12.4e} ({vary})")

        return "\n".join(lines)
