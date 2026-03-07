"""Parameter transforms between model space and optimizer space.

Handles:
- Expanding varying parameters back to full 14-parameter arrays
- Compressing full arrays down to varying-only subsets
- Log-space reparameterization for parameters like D0, v0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_manager import ParameterManager

logger = get_logger(__name__)


class ParameterTransform:
    """Bidirectional transform between model and optimizer parameter spaces.

    The optimizer sees a reduced vector of only the varying parameters,
    optionally in log-space for parameters where that improves conditioning
    (e.g. D0_ref, D0_sample, v0).

    Usage::

        transform = ParameterTransform(param_manager, use_log=True)
        x0 = transform.to_optimizer(full_params)       # (n_varying,)
        full = transform.to_model(x_opt)                # (14,)
        lo, hi = transform.get_optimizer_bounds()        # transformed bounds
    """

    def __init__(
        self,
        param_manager: ParameterManager,
        use_log: bool = False,
    ) -> None:
        """Initialize transform.

        Args:
            param_manager: ParameterManager with vary flags and bounds
            use_log: Apply log-transform to parameters with log_space=True
        """
        self._pm = param_manager
        self._use_log = use_log

        # Cache indices and log-space flags for varying params
        self._varying_indices = param_manager.varying_indices
        self._varying_names = param_manager.varying_names

        self._log_mask = np.zeros(len(self._varying_indices), dtype=bool)
        if use_log:
            for i, name in enumerate(self._varying_names):
                try:
                    info = DEFAULT_REGISTRY[name]
                    if info.log_space and info.min_bound > 0:
                        self._log_mask[i] = True
                except KeyError:
                    pass

        n_log = int(np.sum(self._log_mask))
        if n_log > 0:
            log_names = [
                self._varying_names[i]
                for i in range(len(self._varying_names))
                if self._log_mask[i]
            ]
            logger.debug(
                "Log-space transform active for %d parameters: %s",
                n_log,
                log_names,
            )

    @property
    def n_varying(self) -> int:
        """Number of varying parameters in optimizer space."""
        return len(self._varying_indices)

    @property
    def varying_names(self) -> list[str]:
        """Names of varying parameters in order."""
        return self._varying_names

    @property
    def log_mask(self) -> np.ndarray:
        """Boolean mask: True for log-transformed parameters."""
        return self._log_mask

    def to_optimizer(self, full_params: np.ndarray) -> np.ndarray:
        """Transform full model parameters to optimizer space.

        Args:
            full_params: Array of shape (14,) in model space

        Returns:
            Array of shape (n_varying,) in optimizer space
        """
        varying = np.array([full_params[i] for i in self._varying_indices])
        if self._use_log and np.any(self._log_mask):
            varying = np.where(
                self._log_mask,
                np.log(np.maximum(varying, 1e-30)),
                varying,
            )
        return varying

    def to_model(self, optimizer_params: np.ndarray) -> np.ndarray:
        """Transform optimizer parameters back to full model space.

        Args:
            optimizer_params: Array of shape (n_varying,) in optimizer space

        Returns:
            Array of shape (14,) in model space
        """
        varying = np.asarray(optimizer_params, dtype=np.float64)
        if self._use_log and np.any(self._log_mask):
            varying = np.where(self._log_mask, np.exp(varying), varying)

        return self._pm.expand_varying_to_full(varying)

    def get_optimizer_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds in optimizer space.

        Returns:
            (lower, upper) each of shape (n_varying,), log-transformed if active
        """
        lower, upper = self._pm.get_bounds()
        if self._use_log and np.any(self._log_mask):
            lower = np.where(
                self._log_mask,
                np.log(np.maximum(lower, 1e-30)),
                lower,
            )
            upper = np.where(
                self._log_mask,
                np.log(np.maximum(upper, 1e-30)),
                upper,
            )
        return lower, upper

    def get_optimizer_x0(self) -> np.ndarray:
        """Get initial guess in optimizer space.

        Returns:
            Array of shape (n_varying,) — initial varying values, log-transformed if active
        """
        full = self._pm.get_full_values()
        return self.to_optimizer(full)

    def jacobian_correction(self, optimizer_params: np.ndarray) -> np.ndarray:
        """Compute diagonal Jacobian correction for log-transform.

        When parameters are log-transformed, the chain rule gives
        d(model)/d(log_x) = x * d(model)/d(x). This returns the
        scaling factors (x for log-params, 1 otherwise).

        Args:
            optimizer_params: Current optimizer-space parameters

        Returns:
            Array of shape (n_varying,) with scaling factors
        """
        if not self._use_log or not np.any(self._log_mask):
            return np.ones(self.n_varying)

        varying = np.asarray(optimizer_params, dtype=np.float64)
        model_vals = np.where(self._log_mask, np.exp(varying), varying)
        return np.where(self._log_mask, model_vals, 1.0)


def compress_to_varying(
    full_params: np.ndarray,
    vary_flags: dict[str, bool],
) -> np.ndarray:
    """Extract varying parameters from a full 14-parameter array.

    Args:
        full_params: Shape (14,)
        vary_flags: Dict mapping parameter names to vary booleans

    Returns:
        Array of varying parameter values
    """
    indices = [
        i for i, name in enumerate(ALL_PARAM_NAMES)
        if vary_flags.get(name, True)
    ]
    return full_params[indices]


def expand_to_full(
    varying_params: np.ndarray,
    fixed_values: np.ndarray,
    vary_flags: dict[str, bool],
) -> np.ndarray:
    """Expand varying parameters into full 14-parameter array.

    Args:
        varying_params: Values for varying parameters
        fixed_values: Full 14-parameter array with fixed values
        vary_flags: Dict mapping parameter names to vary booleans

    Returns:
        Full 14-parameter array with varying values inserted
    """
    full = fixed_values.copy()
    vary_idx = 0
    for i, name in enumerate(ALL_PARAM_NAMES):
        if vary_flags.get(name, True):
            full[i] = varying_params[vary_idx]
            vary_idx += 1
    return full
