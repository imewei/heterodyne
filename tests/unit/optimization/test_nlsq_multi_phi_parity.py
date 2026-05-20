"""Parity tests for heterodyne multi-angle NLSQ orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult


@pytest.mark.unit
def test_auto_mode_uses_joint_constant_fit_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should use homodyne-style averaged scaling for 3+ angles."""
    import heterodyne.optimization.nlsq.core as core

    phi_angles = np.array([-5.0, 5.0, 90.0])
    c2_data = np.zeros((3, 4, 4), dtype=float)
    config = NLSQConfig(per_angle_mode="auto", constant_scaling_threshold=3)

    result = NLSQResult(
        parameters=np.array([1.0]),
        parameter_names=["D0_ref"],
        success=True,
        message="ok",
        metadata={},
    )
    calls: list[dict[str, object]] = []

    def fake_joint_constant(**kwargs):
        calls.append(kwargs)
        return [result, result, result]

    def fail_single_phi(**kwargs):
        raise AssertionError("auto constant mode should not fit angles sequentially")

    monkeypatch.setattr(
        core,
        "_fit_joint_averaged_multi_phi",
        fake_joint_constant,
        raising=False,
    )
    monkeypatch.setattr(core, "fit_nlsq_jax", fail_single_phi)

    results = core.fit_nlsq_multi_phi(
        model=MagicMock(),
        c2_data=c2_data,
        phi_angles=phi_angles,
        config=config,
    )

    assert len(calls) == 1
    assert calls[0]["config"] is config
    assert results == [result, result, result]


@pytest.mark.unit
def test_cmaes_enabled_multi_phi_uses_joint_cmaes_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMA-ES should be selected once for multi-angle fitting."""
    import heterodyne.optimization.nlsq.core as core

    phi_angles = np.array([-5.0, 5.0, 90.0])
    c2_data = np.zeros((3, 4, 4), dtype=float)
    config = NLSQConfig(
        per_angle_mode="auto",
        constant_scaling_threshold=3,
        enable_cmaes=True,
    )

    result = NLSQResult(
        parameters=np.array([1.0]),
        parameter_names=["D0_ref"],
        success=True,
        message="ok",
        metadata={},
    )
    calls: list[dict[str, object]] = []

    def fake_joint_cmaes(**kwargs):
        calls.append(kwargs)
        return [result, result, result]

    def fail_local_joint(**kwargs):
        raise AssertionError("CMA-ES should own the multi-angle path")

    monkeypatch.setattr(core, "HAS_CMAES", True)
    monkeypatch.setattr(
        core,
        "_fit_joint_cmaes_multi_phi",
        fake_joint_cmaes,
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "_fit_joint_averaged_multi_phi",
        fail_local_joint,
        raising=False,
    )

    results = core.fit_nlsq_multi_phi(
        model=MagicMock(),
        c2_data=c2_data,
        phi_angles=phi_angles,
        config=config,
    )

    assert len(calls) == 1
    assert calls[0]["config"] is config
    assert results == [result, result, result]


@pytest.mark.unit
def test_explicit_constant_mode_uses_joint_fixed_constant_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Homodyne parity: explicit constant mode uses the FROZEN per-angle path.

    Previously this test asserted that constant routed to
    _fit_joint_averaged_multi_phi (the averaged path) — that was the
    pre-parity behaviour.  Constant now routes to
    _fit_joint_fixed_constant_multi_phi (B3 dispatch split).
    """
    import heterodyne.optimization.nlsq.core as core

    phi_angles = np.array([-5.0, 5.0])
    c2_data = np.zeros((2, 4, 4), dtype=float)
    config = NLSQConfig(per_angle_mode="constant", constant_scaling_threshold=3)

    result = NLSQResult(
        parameters=np.array([1.0]),
        parameter_names=["D0_ref"],
        success=True,
        message="ok",
        metadata={},
    )
    calls: list[dict[str, object]] = []

    def fake_joint_fixed(**kwargs):
        calls.append(kwargs)
        return [result, result]

    def fail_averaged(**kwargs):
        raise AssertionError("constant mode must use FIXED per-angle β,o, not averaged")

    def fail_fourier(**kwargs):
        raise AssertionError("constant mode should not use Fourier joint fitting")

    monkeypatch.setattr(
        core,
        "_fit_joint_fixed_constant_multi_phi",
        fake_joint_fixed,
        raising=False,
    )
    monkeypatch.setattr(
        core, "_fit_joint_averaged_multi_phi", fail_averaged, raising=False
    )
    monkeypatch.setattr(core, "_fit_joint_multi_phi", fail_fourier, raising=False)

    results = core.fit_nlsq_multi_phi(
        model=MagicMock(),
        c2_data=c2_data,
        phi_angles=phi_angles,
        config=config,
    )

    assert len(calls) == 1
    assert results == [result, result]


@pytest.mark.unit
def test_joint_cmaes_uses_off_diagonal_data_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMA-ES reduced chi2 data count should match diagonal-filtered residuals."""
    import jax.numpy as jnp

    import heterodyne.optimization.nlsq.core as core

    config = NLSQConfig(
        enable_cmaes=True,
        per_angle_mode="constant",
        cmaes_warmstart_auto_skip=False,
    )
    c2_data = np.zeros((2, 3, 3), dtype=float)
    phi_angles = np.array([0.0, 90.0])
    warmstart = NLSQResult(
        parameters=np.array([1.0]),
        parameter_names=["D0_ref"],
        success=True,
        message="warm",
        final_cost=1.0,
        reduced_chi_squared=99.0,
        metadata={"contrast": 0.3, "offset": 1.0},
    )
    captured: dict[str, object] = {}

    param_manager = MagicMock()
    param_manager.varying_names = ["D0_ref"]
    param_manager.n_varying = 1
    param_manager.get_bounds.return_value = (np.array([0.0]), np.array([10.0]))
    param_manager.get_full_values.return_value = np.ones(14)
    param_manager.varying_indices = [0]

    model = MagicMock()
    model.param_manager = param_manager
    model.t = jnp.arange(1.0, 4.0)
    model.q = 0.01
    model.dt = 0.1

    def fake_warmstart(**kwargs):
        return [warmstart, warmstart]

    def fake_fit_with_cmaes(**kwargs):
        captured.update(kwargs)
        return NLSQResult(
            parameters=np.array([1.0, 0.3, 1.0]),
            parameter_names=["D0_ref", "contrast", "offset"],
            success=True,
            message="cmaes",
            final_cost=2.0,
        )

    # Post-review parity fix: per_angle_mode="constant" now routes CMA-ES
    # warmstart through _fit_joint_fixed_constant_multi_phi (was incorrectly
    # using _fit_joint_averaged_multi_phi via the legacy union predicate).
    monkeypatch.setattr(core, "_fit_joint_fixed_constant_multi_phi", fake_warmstart)
    monkeypatch.setattr(core, "fit_with_cmaes", fake_fit_with_cmaes)

    results = core._fit_joint_cmaes_multi_phi(
        model=model,
        c2_data=c2_data,
        phi_angles=phi_angles,
        config=config,
        weights=None,
    )

    assert captured["n_data"] == 2 * 3 * (3 - 1)
    assert results == [warmstart, warmstart]


@pytest.mark.unit
def test_joint_cmaes_independent_mode_runs_global_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMA-ES should not fall back to plain NLSQ for non-auto scaling modes."""
    import jax.numpy as jnp

    import heterodyne.optimization.nlsq.core as core

    config = NLSQConfig(
        enable_cmaes=True,
        per_angle_mode="independent",
        cmaes_warmstart_auto_skip=False,
    )
    c2_data = np.zeros((2, 3, 3), dtype=float)
    phi_angles = np.array([0.0, 90.0])
    warmstart_results = [
        NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["D0_ref"],
            success=True,
            message="warm",
            final_cost=1.0,
            reduced_chi_squared=99.0,
            metadata={"contrast": 0.25, "offset": 0.95},
        ),
        NLSQResult(
            parameters=np.array([1.0]),
            parameter_names=["D0_ref"],
            success=True,
            message="warm",
            final_cost=1.0,
            reduced_chi_squared=99.0,
            metadata={"contrast": 0.35, "offset": 1.05},
        ),
    ]
    captured: dict[str, object] = {}

    param_manager = MagicMock()
    param_manager.varying_names = ["D0_ref"]
    param_manager.n_varying = 1
    param_manager.get_bounds.return_value = (np.array([0.0]), np.array([10.0]))
    param_manager.get_full_values.return_value = np.ones(14)
    param_manager.varying_indices = [0]

    model = MagicMock()
    model.param_manager = param_manager
    model.t = jnp.arange(1.0, 4.0)
    model.q = 0.01
    model.dt = 0.1

    def fake_warmstart(**kwargs):
        return warmstart_results

    def fake_fit_with_cmaes(**kwargs):
        captured.update(kwargs)
        return NLSQResult(
            parameters=np.array([1.0, 0.25, 0.35, 0.95, 1.05]),
            parameter_names=[
                "D0_ref",
                "contrast[0]",
                "contrast[1]",
                "offset[0]",
                "offset[1]",
            ],
            success=True,
            message="cmaes",
            final_cost=2.0,
        )

    monkeypatch.setattr(core, "_fit_joint_multi_phi", fake_warmstart)
    monkeypatch.setattr(core, "fit_with_cmaes", fake_fit_with_cmaes)

    results = core._fit_joint_cmaes_multi_phi(
        model=model,
        c2_data=c2_data,
        phi_angles=phi_angles,
        config=config,
        weights=None,
    )

    assert captured["n_data"] == 2 * 3 * (3 - 1)
    assert captured["parameter_names"] == [
        "D0_ref",
        "contrast[0]",
        "contrast[1]",
        "offset[0]",
        "offset[1]",
    ]
    assert np.allclose(
        np.asarray(captured["initial_params"]), [1.0, 0.25, 0.35, 0.95, 1.05]
    )
    assert results == warmstart_results


@pytest.mark.unit
def test_multi_angle_residuals_exclude_diagonal() -> None:
    """Joint multi-angle residuals should remove corrected diagonal points."""
    import jax.numpy as jnp

    from heterodyne.core.jax_backend import compute_multi_angle_residuals

    params = jnp.array(
        [
            1.0e4,
            -0.5,
            10.0,
            8.0e3,
            -0.4,
            8.0,
            100.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.0,
        ],
        dtype=jnp.float64,
    )
    residuals = compute_multi_angle_residuals(
        params=params,
        t=jnp.arange(1.0, 4.0),
        q=0.001,
        dt=0.1,
        phi_angles=jnp.array([0.0, 90.0]),
        c2_data_batch=jnp.zeros((2, 3, 3), dtype=jnp.float64),
        weights_batch=jnp.ones((2, 3, 3), dtype=jnp.float64),
        contrasts=jnp.ones(2, dtype=jnp.float64),
        offsets=jnp.ones(2, dtype=jnp.float64),
    )

    assert np.asarray(residuals).shape == (2 * 3 * (3 - 1),)
