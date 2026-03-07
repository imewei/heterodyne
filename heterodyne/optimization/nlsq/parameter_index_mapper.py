"""Bidirectional index mapper between parameter spaces.

Maps indices among three representations of the heterodyne model's
parameter vector:

1. **Full** -- all 14 parameters in canonical order.
2. **Varying** -- only the parameters that are free in optimization.
3. **Optimizer** -- varying parameters, optionally log-transformed.

Useful for interpreting Jacobian columns, extracting covariance sub-
matrices, and translating between human-readable names and numeric
indices in each space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heterodyne.config.parameter_names import ALL_PARAM_NAMES
from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from heterodyne.config.parameter_manager import ParameterManager

logger = get_logger(__name__)


class ParameterIndexMapper:
    """Map parameter indices between full, varying, and optimizer spaces.

    Parameters
    ----------
    varying_names : list[str]
        Ordered names of the parameters that vary in optimization.
    varying_full_indices : list[int]
        Position of each varying parameter in the full 14-element array.
    log_mask : list[bool] | None
        Per-varying-parameter flag indicating log-transform in optimizer
        space.  ``None`` means no log-transforms.

    Examples
    --------
    >>> mapper = ParameterIndexMapper.build_from_manager(pm)
    >>> mapper.get_name(0)          # name of the first varying parameter
    'D0_ref'
    >>> mapper.varying_to_full(0)   # its index in the 14-element array
    0
    >>> mapper.full_to_varying(0)   # reverse lookup
    0
    """

    def __init__(
        self,
        varying_names: list[str],
        varying_full_indices: list[int],
        log_mask: list[bool] | None = None,
    ) -> None:
        if len(varying_names) != len(varying_full_indices):
            raise ValueError(
                f"Length mismatch: {len(varying_names)} names vs "
                f"{len(varying_full_indices)} indices"
            )

        self._varying_names = list(varying_names)
        self._varying_full_indices = list(varying_full_indices)
        self._log_mask = list(log_mask) if log_mask is not None else [False] * len(varying_names)

        # Build reverse lookup: full_index -> varying_index
        self._full_to_varying: dict[int, int] = {
            full_idx: vary_idx
            for vary_idx, full_idx in enumerate(self._varying_full_indices)
        }

        # Build name -> varying_index lookup
        self._name_to_varying: dict[str, int] = {
            name: i for i, name in enumerate(self._varying_names)
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_full(self) -> int:
        """Total number of model parameters (always 14)."""
        return len(ALL_PARAM_NAMES)

    @property
    def n_varying(self) -> int:
        """Number of varying parameters."""
        return len(self._varying_names)

    @property
    def varying_names(self) -> list[str]:
        """Names of varying parameters in order."""
        return list(self._varying_names)

    @property
    def varying_full_indices(self) -> list[int]:
        """Full-array indices of the varying parameters."""
        return list(self._varying_full_indices)

    @property
    def fixed_full_indices(self) -> list[int]:
        """Full-array indices of the fixed parameters."""
        varying_set = set(self._varying_full_indices)
        return [i for i in range(self.n_full) if i not in varying_set]

    @property
    def log_mask(self) -> list[bool]:
        """Per-varying-parameter log-transform flags."""
        return list(self._log_mask)

    # ------------------------------------------------------------------
    # Index conversions
    # ------------------------------------------------------------------

    def full_to_varying(self, full_idx: int) -> int:
        """Convert a full-array index to a varying-array index.

        Args:
            full_idx: Index in the 14-element array.

        Returns:
            Index in the varying-parameter array.

        Raises:
            KeyError: If the parameter at *full_idx* is not varying.
        """
        if full_idx not in self._full_to_varying:
            name = ALL_PARAM_NAMES[full_idx] if 0 <= full_idx < self.n_full else f"#{full_idx}"
            raise KeyError(f"Parameter '{name}' (full index {full_idx}) is not varying")
        return self._full_to_varying[full_idx]

    def varying_to_full(self, varying_idx: int) -> int:
        """Convert a varying-array index to a full-array index.

        Args:
            varying_idx: Index in the varying-parameter array.

        Returns:
            Index in the 14-element array.

        Raises:
            IndexError: If *varying_idx* is out of range.
        """
        if varying_idx < 0 or varying_idx >= self.n_varying:
            raise IndexError(
                f"varying_idx {varying_idx} out of range [0, {self.n_varying})"
            )
        return self._varying_full_indices[varying_idx]

    def get_name(self, varying_idx: int) -> str:
        """Get the parameter name for a varying-array index.

        Args:
            varying_idx: Index in the varying-parameter array.

        Returns:
            Parameter name string.

        Raises:
            IndexError: If *varying_idx* is out of range.
        """
        if varying_idx < 0 or varying_idx >= self.n_varying:
            raise IndexError(
                f"varying_idx {varying_idx} out of range [0, {self.n_varying})"
            )
        return self._varying_names[varying_idx]

    def name_to_varying(self, name: str) -> int:
        """Look up the varying-array index for a parameter name.

        Args:
            name: Parameter name.

        Returns:
            Index in the varying-parameter array.

        Raises:
            KeyError: If *name* is not a varying parameter.
        """
        if name not in self._name_to_varying:
            raise KeyError(f"Parameter '{name}' is not varying")
        return self._name_to_varying[name]

    def is_log_transformed(self, varying_idx: int) -> bool:
        """Check whether a varying parameter uses log-transform.

        Args:
            varying_idx: Index in the varying-parameter array.

        Returns:
            ``True`` if log-transformed in optimizer space.
        """
        if varying_idx < 0 or varying_idx >= self.n_varying:
            raise IndexError(
                f"varying_idx {varying_idx} out of range [0, {self.n_varying})"
            )
        return self._log_mask[varying_idx]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_manager(
        cls,
        pm: ParameterManager,
        use_log: bool = False,
    ) -> ParameterIndexMapper:
        """Construct a mapper from a :class:`ParameterManager`.

        Args:
            pm: Configured parameter manager with vary flags.
            use_log: Whether to include log-transform flags based on the
                default registry's ``log_space`` attribute.

        Returns:
            A new :class:`ParameterIndexMapper`.
        """
        varying_names = pm.varying_names
        varying_indices = pm.varying_indices

        log_mask: list[bool] = []
        if use_log:
            for name in varying_names:
                try:
                    info = DEFAULT_REGISTRY[name]
                    log_mask.append(info.log_space and info.min_bound > 0)
                except KeyError:
                    log_mask.append(False)
        else:
            log_mask = [False] * len(varying_names)

        logger.debug(
            "Built ParameterIndexMapper: %d varying out of %d total",
            len(varying_names),
            len(ALL_PARAM_NAMES),
        )

        return cls(
            varying_names=varying_names,
            varying_full_indices=varying_indices,
            log_mask=log_mask,
        )
