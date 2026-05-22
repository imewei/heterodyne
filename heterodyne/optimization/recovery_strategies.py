"""Deprecated location for NLSQ recovery strategies.

The implementation now lives in :mod:`heterodyne.optimization.nlsq.recovery`.
This shim re-exports the public names so that legacy imports continue to
work for one release; new code should import from the new path.
"""

from __future__ import annotations

import warnings

from heterodyne.optimization.nlsq.recovery import (
    RecoveryAction,
    RecoveryPlan,
    apply_recovery,
    diagnose_failure,
    suggest_fixed_parameters,
)

warnings.warn(
    "heterodyne.optimization.recovery_strategies is deprecated; "
    "import from heterodyne.optimization.nlsq.recovery instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RecoveryAction",
    "RecoveryPlan",
    "apply_recovery",
    "diagnose_failure",
    "suggest_fixed_parameters",
]
