"""Regression tests for deprecated import shims.

The two modules below were moved/retired during the Phase B-C
homodyne-parity consolidation. They are retained as compatibility
shims that emit ``DeprecationWarning`` so external consumers do not
fail with ``ModuleNotFoundError``. This test pins that contract.
"""

from __future__ import annotations

import importlib
import warnings


def test_recovery_strategies_shim_reexports_public_api() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module("heterodyne.optimization.recovery_strategies")

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    for name in (
        "RecoveryAction",
        "RecoveryPlan",
        "diagnose_failure",
        "apply_recovery",
        "suggest_fixed_parameters",
    ):
        assert hasattr(mod, name), f"shim missing {name}"

    from heterodyne.optimization.nlsq.recovery import (
        RecoveryAction as NewRecoveryAction,
    )

    assert mod.RecoveryAction is NewRecoveryAction


def test_numpy_gradients_module_still_importable() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module("heterodyne.core.numpy_gradients")

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    for name in (
        "compute_gradient_finite_diff",
        "compute_jacobian_finite_diff",
        "compute_hessian_finite_diff",
        "validate_gradient",
    ):
        assert hasattr(mod, name), f"numpy_gradients missing {name}"
