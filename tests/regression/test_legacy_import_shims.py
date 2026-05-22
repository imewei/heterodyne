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


_NUMPY_GRADIENTS_PUBLIC_NAMES = (
    "compute_gradient_finite_diff",
    "compute_jacobian_finite_diff",
    "compute_hessian_finite_diff",
    "validate_gradient",
)


def _reset_numpy_gradients_warned_set() -> None:
    """Clear the one-shot ``_warned_names`` so a test can re-observe the warning.

    The shim's lazy deprecation fires once per symbol per process; tests that
    exercise the warning must reset this state, otherwise earlier tests (or
    other modules in the same suite) would silence ours.
    """
    mod = importlib.import_module("heterodyne.core.numpy_gradients")
    type(mod)._warned_names.clear()  # type: ignore[attr-defined]


def test_numpy_gradients_bare_import_is_silent() -> None:
    """PEP 562 lazy contract: a bare import must NOT emit DeprecationWarning.

    The previous shim warned at module load time, which surfaced in unrelated
    test suites that happened to import this module via a transitive
    consumer. The new contract is: import is silent; warning fires only on
    external attribute access of a public symbol.
    """
    _reset_numpy_gradients_warned_set()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Reload to force the module body to re-execute — that's the load
        # path we want to verify is quiet.
        mod = importlib.import_module("heterodyne.core.numpy_gradients")
        importlib.reload(mod)

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation, (
        "Bare import of heterodyne.core.numpy_gradients must be silent under "
        f"PEP 562 lazy deprecation; got {len(deprecation)} DeprecationWarning(s): "
        f"{[str(w.message) for w in deprecation]}"
    )


def test_numpy_gradients_attribute_access_warns_once_per_symbol() -> None:
    """Accessing a public symbol externally fires DeprecationWarning once."""
    _reset_numpy_gradients_warned_set()
    mod = importlib.import_module("heterodyne.core.numpy_gradients")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in _NUMPY_GRADIENTS_PUBLIC_NAMES:
            assert hasattr(mod, name), f"numpy_gradients missing {name}"

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    warned_names = {
        str(w.message).split("heterodyne.core.numpy_gradients.")[1].split(" ")[0]
        for w in deprecation
    }
    assert warned_names == set(_NUMPY_GRADIENTS_PUBLIC_NAMES), (
        f"Expected one DeprecationWarning per pinned public name; got "
        f"{warned_names}. Each external access of a public symbol should "
        f"fire the warning the first time it happens."
    )


def test_numpy_gradients_warning_is_one_shot_per_symbol() -> None:
    """Second access of the same symbol must not re-warn (one-shot guard)."""
    _reset_numpy_gradients_warned_set()
    mod = importlib.import_module("heterodyne.core.numpy_gradients")

    # First access — fires the warning.
    _ = mod.compute_gradient_finite_diff

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = mod.compute_gradient_finite_diff
        _ = mod.compute_gradient_finite_diff

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation, (
        "Second access of an already-warned symbol must be silent; got "
        f"{len(deprecation)} DeprecationWarning(s) on repeat access."
    )
