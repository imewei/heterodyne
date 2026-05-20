"""Pinned-constant parity guard for per_angle_mode taxonomy.

Source of truth: https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html

If homodyne adds a fifth mode, this test will start failing — that is
intentional: the homodyne docs are the canonical reference and any new
mode there should be tracked in heterodyne.
"""

from __future__ import annotations

import typing

import pytest

# Pinned from homodyne anti-degeneracy theory page (revision dated 2026-05).
HOMODYNE_PER_ANGLE_MODES: frozenset[str] = frozenset(
    {"constant", "auto", "fourier", "individual"}
)


@pytest.mark.parity
def test_heterodyne_publishes_all_homodyne_modes() -> None:
    """Every homodyne mode must be present in heterodyne's public Literal."""
    from heterodyne.optimization.nlsq.config import NLSQConfig

    field_type = typing.get_type_hints(NLSQConfig)["per_angle_mode"]
    heterodyne_modes = set(typing.get_args(field_type))
    # "independent" is a deprecation alias; not part of the canonical set.
    canonical = heterodyne_modes - {"independent"}

    missing = HOMODYNE_PER_ANGLE_MODES - canonical
    assert not missing, (
        f"Heterodyne is missing homodyne mode(s): {missing}. "
        f"Update Literal in heterodyne/optimization/nlsq/config.py."
    )

    extra = canonical - HOMODYNE_PER_ANGLE_MODES
    assert not extra, (
        f"Heterodyne publishes mode(s) not in homodyne reference: {extra}. "
        f"Either update HOMODYNE_PER_ANGLE_MODES in this test (if homodyne "
        f"added them too) or remove from the heterodyne Literal."
    )


@pytest.mark.parity
def test_l5_shear_weighting_remains_disabled() -> None:
    """The controller's use_shear_weighting property must always be False."""
    import numpy as np

    from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
        AntiDegeneracyController,
    )

    controller = AntiDegeneracyController.from_config(
        config_dict={"enable": True, "per_angle_mode": "auto"},
        n_phi=5,
        phi_angles=np.zeros(5, dtype=np.float64),
        n_physical=14,
    )
    assert controller.use_shear_weighting is False, (
        "L5 (shear weighting) must never activate in heterodyne — the "
        "velocity-phase model has no shear sinc term to re-weight."
    )
