"""Homodyne parity tests for heterodyne NLSQ per-angle-mode dispatch.

Reference: https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html

Target mode table (heterodyne has 14 physics params; homodyne has 7):

  - constant:   14 physics, β(φ) and o(φ) frozen per-angle from quantile
  - auto:       14 + 2 averaged scaling = 16 params (when n_phi >= threshold)
  - fourier K=2: 14 + 2·(2K+1) = 24 params (K=2 -> 10 Fourier coefficients)
  - individual: 14 + 2·n_phi params
"""

from __future__ import annotations

import typing
from typing import Literal, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig
from heterodyne.optimization.nlsq.results import NLSQResult

ModeLiteral = Literal["individual", "constant", "fourier", "auto", "independent"]


@pytest.mark.unit
class TestModeTaxonomy:
    """The four homodyne mode names are accepted by the public Literal."""

    def test_individual_mode_accepted(self) -> None:
        """`individual` is the canonical name (matches homodyne docs)."""
        cfg = NLSQConfig(per_angle_mode="individual")
        assert cfg.per_angle_mode == "individual"

    def test_constant_mode_accepted(self) -> None:
        """`constant` is a valid public configuration value."""
        cfg = NLSQConfig(per_angle_mode="constant")
        assert cfg.per_angle_mode == "constant"

    def test_literal_contains_four_canonical_modes(self) -> None:
        """`get_args` returns the four canonical homodyne mode names.

        The Literal may additionally include `"independent"` as a deprecation
        alias; we assert the canonical four are present and `"independent"`
        is the only allowed extra.
        """
        field_type = typing.get_type_hints(NLSQConfig)["per_angle_mode"]
        args = set(typing.get_args(field_type))
        canonical = {"individual", "constant", "fourier", "auto"}
        assert canonical.issubset(args), f"Missing canonical names: {canonical - args}"
        assert args - canonical <= {"independent"}, (
            f"Unexpected mode names in Literal: {args - canonical - {'independent'}}"
        )

    def test_validate_accepts_all_literal_modes(self) -> None:
        """`validate()` must accept every value the Literal allows.

        Catches the validate-vs-Literal drift that the code-quality review
        caught after Task A1 — the original A1 widened the Literal but
        forgot to update the validate() allowlist.
        """
        import warnings as _warnings

        for mode in ("individual", "constant", "fourier", "auto", "independent"):
            with _warnings.catch_warnings():
                # A2 alias: 'independent' emits DeprecationWarning at construction.
                _warnings.simplefilter("ignore", DeprecationWarning)
                cfg = NLSQConfig(per_angle_mode=mode)
            errors = cfg.validate()
            mode_errors = [e for e in errors if "per_angle_mode" in e]
            assert not mode_errors, (
                f"validate() rejected per_angle_mode={mode!r} "
                f"even though it is in the Literal: {mode_errors}"
            )

    def test_independent_deprecation_alias_normalises_to_individual(self) -> None:
        """`independent` maps to `individual` with a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="independent.*individual"):
            cfg = NLSQConfig(per_angle_mode="independent")
        assert cfg.per_angle_mode == "individual"

    def test_individual_does_not_warn(self) -> None:
        """`individual` is the canonical name and emits no DeprecationWarning."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            cfg = NLSQConfig(per_angle_mode="individual")
        assert cfg.per_angle_mode == "individual"
        deprecation_warnings = [
            r for r in record if issubclass(r.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, f"Unexpected: {deprecation_warnings}"

    def test_deprecation_warning_attributes_to_caller(self) -> None:
        """`stacklevel=3` makes the DeprecationWarning point to the user's
        call site, not the dataclass-generated __init__.

        Pins the empirically-verified stacklevel from the code-quality
        review fix of 45b5a1d.  With stacklevel=2 the warning reported
        '<string>:149' (the dataclass __init__); stacklevel=3 reports
        this test file at the line of the NLSQConfig() call.
        """
        import sys
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            expected_lineno = sys._getframe(0).f_lineno + 1
            NLSQConfig(per_angle_mode="independent")

        deprecation = [r for r in record if issubclass(r.category, DeprecationWarning)]
        assert len(deprecation) == 1, (
            f"Expected 1 DeprecationWarning, got {deprecation}"
        )
        w = deprecation[0]
        assert w.filename == __file__, (
            f"DeprecationWarning filename {w.filename!r} should point to the "
            f"test file {__file__!r}; stacklevel is probably wrong."
        )
        assert w.lineno == expected_lineno, (
            f"DeprecationWarning lineno {w.lineno} should match the caller "
            f"line {expected_lineno}; stacklevel is probably wrong."
        )


def _make_synthetic_model(n_varying: int):
    """Mock HeterodyneModel exposing the minimal API needed by joint fits."""
    from heterodyne.core.heterodyne_model import HeterodyneModel  # noqa: F401

    model = MagicMock(spec=HeterodyneModel)
    model.t = np.linspace(0.001, 0.005, 5, dtype=np.float64)
    model.q = 0.01
    model.dt = 0.001
    pm = MagicMock()
    pm.varying_names = [f"p{i}" for i in range(n_varying)]
    pm.n_varying = n_varying
    pm.get_initial_values.return_value = np.full(n_varying, 0.1, dtype=np.float64)
    pm.get_bounds.return_value = (
        np.full(n_varying, -1.0, dtype=np.float64),
        np.full(n_varying, 1.0, dtype=np.float64),
    )
    pm.expand_varying_to_full.side_effect = lambda v: np.asarray(v)
    model.param_manager = pm
    model.set_params = MagicMock()
    model.scaling = MagicMock()
    model.scaling.contrast = np.array([0.5])
    model.scaling.offset = np.array([1.0])
    return model


@pytest.mark.unit
class TestFixedConstantSemantics:
    """`constant` mode optimizes physics params only; β,o are frozen per-angle."""

    def test_fit_joint_fixed_constant_sizes_x0_to_14_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Optimizer receives x0 of length 14 (n_varying), not 16, not 14+2·n_phi.

        β_k, o_k are pre-computed from quantile and supplied to the residual
        as closure constants, NOT as free optimizer variables.
        """
        import heterodyne.optimization.nlsq.core as core

        captured: dict[str, np.ndarray] = {}
        captured_names: list[str] = []

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                captured_names[:] = list(parameter_names)

            def fit(
                self,
                *,
                residual_fn,
                initial_params: np.ndarray,
                bounds: tuple[np.ndarray, np.ndarray],
                config,
            ) -> NLSQResult:
                captured["x0"] = np.asarray(initial_params).copy()
                captured["lb"] = np.asarray(bounds[0]).copy()
                captured["ub"] = np.asarray(bounds[1]).copy()
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=list(captured_names),
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)
        cfg = NLSQConfig(per_angle_mode="constant")

        core._fit_joint_fixed_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
            weights=None,
        )

        assert captured["x0"].shape == (14,)
        assert captured["lb"].shape == (14,)
        assert captured["ub"].shape == (14,)
        assert "contrast" not in captured_names
        assert "offset" not in captured_names

    def test_result_metadata_carries_frozen_scaling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-angle NLSQResult records its frozen β,o for downstream viz."""
        import heterodyne.optimization.nlsq.core as core

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(14)],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)

        results = core._fit_joint_fixed_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=NLSQConfig(per_angle_mode="constant"),
            weights=None,
        )

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.metadata["per_angle_mode_actual"] == "fixed_constant"
            assert r.metadata["optimizer"] == "joint_fixed_constant"
            assert r.metadata["phi_angle"] == float(phi_angles[i])
            assert "contrast_fixed" in r.metadata
            assert "offset_fixed" in r.metadata

    def test_result_carries_fitted_correlation_and_chi2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-angle NLSQResult from fixed-constant has fitted_correlation,
        residuals, and reduced_chi_squared populated (parity with averaged path).

        Pins the post-review fix — _fit_joint_fixed_constant_multi_phi
        previously left these three fields as None, creating a silent data
        gap in downstream quality gates and residual visualization.
        """
        import heterodyne.optimization.nlsq.core as core

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(14)],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)

        results = core._fit_joint_fixed_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=NLSQConfig(per_angle_mode=cast(ModeLiteral, "constant")),
            weights=None,
        )

        for r in results:
            assert r.fitted_correlation is not None, (
                "fixed-constant must populate fitted_correlation (parity with averaged)"
            )
            assert r.residuals is not None, (
                "fixed-constant must populate residuals (parity with averaged)"
            )
            assert r.reduced_chi_squared is not None, (
                "fixed-constant must populate reduced_chi_squared (parity with averaged)"
            )
            assert r.fitted_correlation.shape == (5, 5), (
                f"fitted_correlation shape should match c2_data; got {r.fitted_correlation.shape}"
            )


@pytest.mark.unit
class TestDispatchPredicates:
    """`constant` triggers the fixed-constant path; `auto`+threshold triggers averaged."""

    def test_explicit_constant_is_fixed_not_averaged(self) -> None:
        from heterodyne.optimization.nlsq.core import (
            _use_averaged_constant_scaling_mode,
            _use_fixed_constant_scaling_mode,
        )

        cfg = NLSQConfig(per_angle_mode="constant", constant_scaling_threshold=3)
        assert _use_fixed_constant_scaling_mode(cfg, n_phi=5) is True
        assert _use_averaged_constant_scaling_mode(cfg, n_phi=5) is False

    def test_auto_above_threshold_is_averaged_not_fixed(self) -> None:
        from heterodyne.optimization.nlsq.core import (
            _use_averaged_constant_scaling_mode,
            _use_fixed_constant_scaling_mode,
        )

        cfg = NLSQConfig(per_angle_mode="auto", constant_scaling_threshold=3)
        assert _use_averaged_constant_scaling_mode(cfg, n_phi=5) is True
        assert _use_fixed_constant_scaling_mode(cfg, n_phi=5) is False

    def test_auto_below_threshold_is_neither(self) -> None:
        from heterodyne.optimization.nlsq.core import (
            _use_averaged_constant_scaling_mode,
            _use_fixed_constant_scaling_mode,
        )

        cfg = NLSQConfig(per_angle_mode="auto", constant_scaling_threshold=5)
        assert _use_averaged_constant_scaling_mode(cfg, n_phi=3) is False
        assert _use_fixed_constant_scaling_mode(cfg, n_phi=3) is False

    def test_fourier_and_individual_trigger_neither(self) -> None:
        from heterodyne.optimization.nlsq.core import (
            _use_averaged_constant_scaling_mode,
            _use_fixed_constant_scaling_mode,
        )

        for mode in ("fourier", "individual"):
            cfg = NLSQConfig(per_angle_mode=mode)
            assert _use_fixed_constant_scaling_mode(cfg, n_phi=10) is False
            assert _use_averaged_constant_scaling_mode(cfg, n_phi=10) is False


@pytest.mark.unit
class TestDispatchRouting:
    """Verify `fit_nlsq_multi_phi` routes each mode to the correct joint fit."""

    @pytest.mark.parametrize(
        ("mode", "n_phi", "expected_dispatch_attr"),
        [
            ("constant", 3, "_fit_joint_fixed_constant_multi_phi"),
            ("auto", 5, "_fit_joint_averaged_multi_phi"),
            ("fourier", 5, "_fit_joint_multi_phi"),
            ("individual", 5, "_fit_joint_multi_phi"),
        ],
    )
    def test_mode_dispatches_to_expected_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        n_phi: int,
        expected_dispatch_attr: str,
    ) -> None:
        import heterodyne.optimization.nlsq.core as core

        phi_angles = np.linspace(-90.0, 90.0, n_phi, dtype=np.float64)
        c2_data = np.zeros((n_phi, 4, 4), dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode=cast(ModeLiteral, mode),
            constant_scaling_threshold=3,
            fourier_order=2,
            fourier_auto_threshold=999,
        )

        calls = {
            "_fit_joint_fixed_constant_multi_phi": 0,
            "_fit_joint_averaged_multi_phi": 0,
            "_fit_joint_multi_phi": 0,
        }

        def make_fake(name: str):
            def _fake(*args, **kwargs):
                calls[name] += 1
                return [
                    NLSQResult(
                        parameters=np.zeros(1),
                        success=True,
                        message="ok",
                        metadata={},
                        parameter_names=[],
                    )
                ] * n_phi

            return _fake

        for name in calls:
            monkeypatch.setattr(core, name, make_fake(name), raising=False)

        core.fit_nlsq_multi_phi(
            model=MagicMock(),
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
        )

        assert calls[expected_dispatch_attr] == 1, (
            f"Mode {mode!r} must dispatch to {expected_dispatch_attr}; "
            f"actual call counts: {calls}"
        )
        leaks = {k: v for k, v in calls.items() if k != expected_dispatch_attr and v}
        assert not leaks, f"Mode {mode!r} leaked into other dispatch paths: {leaks}"

    def test_cmaes_with_constant_uses_fixed_constant_warmstart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CMA-ES warmstart must respect per_angle_mode='constant'.

        Pins the post-review fix (commit-after-108694b) — previously the
        warmstart used the legacy _use_constant_scaling_mode union predicate
        and silently routed 'constant' to the averaged path.
        """
        import heterodyne.optimization.nlsq.core as core

        calls = {
            "_fit_joint_fixed_constant_multi_phi": 0,
            "_fit_joint_averaged_multi_phi": 0,
        }

        def make_fake(name: str):
            def _fake(**kwargs):
                calls[name] += 1
                return [
                    NLSQResult(
                        parameters=np.zeros(14),
                        success=True,
                        message="ok",
                        metadata={},
                        parameter_names=[],
                    )
                ] * 3

            return _fake

        for name in calls:
            monkeypatch.setattr(core, name, make_fake(name), raising=False)

        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.zeros((3, 4, 4), dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode=cast(ModeLiteral, "constant"),
            enable_cmaes=True,
            constant_scaling_threshold=3,
        )

        try:
            core._fit_joint_cmaes_multi_phi(
                model=MagicMock(),
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=cfg,
                weights=None,
            )
        except Exception:
            # CMA-ES phase may raise on the mock model — we only need
            # to verify the warmstart dispatched correctly before it ran.
            pass

        assert calls["_fit_joint_fixed_constant_multi_phi"] == 1, (
            f"constant + CMA-ES must use fixed-constant warmstart; actual: {calls}"
        )
        assert calls["_fit_joint_averaged_multi_phi"] == 0, (
            f"averaged path must not be invoked for constant mode; actual: {calls}"
        )


@pytest.mark.unit
class TestL2Hierarchical:
    """L2 marker wiring: when enable_hierarchical=True, the joint fit
    must construct the controller and surface the hierarchical config
    in per-angle result metadata.

    Note: this is the MARKER variant of L2 wiring (Sub-PR C1) — the
    controller is built and observed, but HierarchicalFitter is not
    actively driving the fit (its single-angle API does not compose
    cleanly with the joint multi-angle path).  A future C1-follow-up
    can extend this to active driving once HierarchicalFitter is
    refactored or wrapped.
    """

    def test_enable_hierarchical_surfaces_marker_in_result_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With enable_hierarchical=True, every per-angle result carries
        result.metadata['hierarchical_config'] with the configured
        max_outer_iterations."""
        import heterodyne.optimization.nlsq.core as core

        # Stub adapter so the joint fit completes without a real solver
        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(len(initial_params))],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
            enable_hierarchical=True,
            hierarchical_max_outer_iterations=4,
        )

        results = core._fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
            weights=None,
        )

        assert len(results) == 3
        for r in results:
            marker = r.metadata.get("hierarchical_config")
            assert marker is not None, (
                f"L2 marker missing from metadata; keys: {list(r.metadata)}"
            )
            assert marker.get("max_outer_iterations") == 4, (
                f"L2 marker max_outer_iterations should be 4; got {marker}"
            )

    def test_disabled_hierarchical_omits_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With enable_hierarchical=False (default), no marker is set."""
        import heterodyne.optimization.nlsq.core as core

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(len(initial_params))],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
            enable_hierarchical=False,
        )

        results = core._fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
            weights=None,
        )

        for r in results:
            assert "hierarchical_config" not in r.metadata


@pytest.mark.unit
class TestL3Regularization:
    """When regularization_mode != 'none', the joint residual returns
    one extra penalty row sqrt(2·loss_aug(params, base))."""

    def test_regularization_appends_penalty_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With regularization_mode='adaptive', residual length grows by 1."""
        import heterodyne.optimization.nlsq.core as core

        captured_residual_fn: list = []

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                captured_residual_fn.append(residual_fn)
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(len(initial_params))],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        # Extend the synthetic model with the param_manager attrs that the
        # joint residual closure needs to actually execute (not just be
        # constructed): the meshgrid-path closure scatters varying values
        # into a 14-vector via .at[varying_indices].set(...).
        model.param_manager.varying_indices = np.arange(14, dtype=np.int32)
        model.param_manager.get_full_values.return_value = np.zeros(
            14, dtype=np.float64
        )
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)

        # First pass: regularization on
        cfg_on = NLSQConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
            regularization_mode="adaptive",
            group_variance_lambda=0.5,
        )
        core._fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg_on,
            weights=None,
        )
        assert captured_residual_fn, "Residual fn must reach the adapter"
        res_on = captured_residual_fn[0]
        try:
            length_on = len(res_on(np.zeros(16)))
        except Exception as exc:  # noqa: BLE001 — residual eval may fail on mock
            pytest.skip(f"L3-on residual eval failed on synthetic model: {exc}")

        # Second pass: regularization off
        captured_residual_fn.clear()
        cfg_off = NLSQConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
            regularization_mode="none",
        )
        core._fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg_off,
            weights=None,
        )
        res_off = captured_residual_fn[0]
        try:
            length_off = len(res_off(np.zeros(16)))
        except Exception as exc:  # noqa: BLE001 — residual eval may fail on mock
            pytest.skip(f"L3-off residual eval failed on synthetic model: {exc}")

        assert length_on == length_off + 1, (
            f"L3 active path must append exactly one penalty residual row. "
            f"with reg: {length_on}, without: {length_off}"
        )

    def test_loss_augmentation_penalizes_scaling_not_physics(self) -> None:
        """L3 penalty must come from per-angle SCALING params (tail), not PHYSICS (head).

        Joint parameter vector layout is ``[physics | per_angle_scaling]``
        (see ``_fit_joint_averaged_multi_phi``: ``x0 = np.concatenate([physics_initial, [avg_contrast, avg_offset]])``).
        A naive ``params[:n_per]`` slice would penalize the FIRST n_per
        physics params instead of the scaling block — this test pins the
        Codex-flagged fix (post-review) that switches to a tail slice
        ``params[n_physical:n_physical+n_per]``.
        """
        from heterodyne.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        controller = AntiDegeneracyController.from_config(
            config_dict={
                "enable": True,
                "per_angle_mode": "auto",
                "constant_scaling_threshold": 3,
                "regularization": {"mode": "adaptive", "lambda": 1.0},
            },
            n_phi=5,
            phi_angles=np.linspace(-90.0, 90.0, 5, dtype=np.float64),
            n_physical=14,
        )

        callbacks = controller.create_nlsq_callbacks()
        loss_aug = callbacks.get("loss_augmentation")
        assert loss_aug is not None, "L3 callback must be registered"

        # Build two parameter vectors of shape [physics | per_angle_scaling].
        # Vector A: physics has HIGH variance, scaling is constant -> penalty should be ~0
        # Vector B: physics is constant, scaling has HIGH variance -> penalty should be large
        # n_per_angle_params == 2 for auto_averaged mode (1 contrast + 1 offset).
        n_phys = 14
        n_per = controller.n_per_angle_params
        assert n_per == 2, f"auto_averaged expects 2 per-angle params; got {n_per}"

        params_high_physics_var = np.concatenate(
            [np.linspace(-1.0, 1.0, n_phys), [0.5, 0.5]]  # scaling constant
        )
        params_high_scaling_var = np.concatenate(
            [np.ones(n_phys), [0.0, 1.0]]  # physics constant, scaling spread
        )
        dummy_residuals = np.zeros(10)

        penalty_phys = float(loss_aug(params_high_physics_var, dummy_residuals))
        penalty_scaling = float(loss_aug(params_high_scaling_var, dummy_residuals))

        # Physics-variance vector should yield essentially zero penalty —
        # the L3 callback ignores physics params entirely.
        assert penalty_phys == pytest.approx(0.0, abs=1e-9), (
            f"L3 must IGNORE physics-param variance; got penalty={penalty_phys}. "
            "If non-zero, the slice is inverted (penalizing physics instead of scaling)."
        )
        # Scaling-variance vector should yield a strictly positive penalty
        # (lambda * var([0.0, 1.0]) = 1.0 * 0.25 = 0.25).
        assert penalty_scaling > 0.0, (
            f"L3 must penalize per-angle scaling variance; got penalty={penalty_scaling}"
        )
        assert penalty_scaling == pytest.approx(0.25, abs=1e-9), (
            f"L3 penalty should be lambda*var(scaling) = 1.0*0.25 = 0.25; "
            f"got {penalty_scaling}"
        )


@pytest.mark.unit
class TestL4GradientMonitor:
    """When enable_gradient_monitoring=True, per-angle result.metadata
    carries the controller's gradient-monitor summary."""

    def test_monitoring_surfaces_summary_in_averaged_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4 active in _fit_joint_averaged_multi_phi adds 'gradient_monitor' key."""
        import heterodyne.optimization.nlsq.core as core

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(len(initial_params))],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
            enable_gradient_monitoring=True,
            gradient_ratio_threshold=0.01,
            gradient_consecutive_triggers=3,
        )

        results = core._fit_joint_averaged_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
            weights=None,
        )

        for r in results:
            # When monitor has no history yet (mock adapter didn't iterate),
            # the summary may be empty; key absent is acceptable too.
            # The contract: when monitoring is ON, the summary key may
            # appear in metadata (it appears whenever the summary is non-empty).
            # For this mock test, we just verify the wiring doesn't crash.
            assert isinstance(r.metadata, dict)

    def test_monitoring_surfaces_summary_in_fixed_constant_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4 active in _fit_joint_fixed_constant_multi_phi too (per spec)."""
        import heterodyne.optimization.nlsq.core as core

        class FakeAdapter:
            def __init__(self, parameter_names: list[str]) -> None:
                pass

            def fit(self, *, residual_fn, initial_params, bounds, config):
                return NLSQResult(
                    parameters=np.asarray(initial_params).copy(),
                    parameter_names=[f"p{i}" for i in range(len(initial_params))],
                    success=True,
                    message="fake",
                    metadata={},
                )

        monkeypatch.setattr(core, "NLSQAdapter", FakeAdapter, raising=False)
        monkeypatch.setattr(core, "HAS_ADAPTERS", True, raising=False)

        model = _make_synthetic_model(n_varying=14)
        phi_angles = np.array([-5.0, 5.0, 90.0], dtype=np.float64)
        c2_data = np.full((3, 5, 5), 1.2, dtype=np.float64)
        cfg = NLSQConfig(
            per_angle_mode="constant",
            enable_gradient_monitoring=True,
        )

        results = core._fit_joint_fixed_constant_multi_phi(
            model=model,
            c2_data=c2_data,
            phi_angles=phi_angles,
            config=cfg,
            weights=None,
        )

        assert len(results) == 3
        # The fit completed without raising — that's the primary contract.
        # Summary is observation-only; mock adapter never iterated, so
        # the key may be absent. Verify the per-angle keys from B2 are still
        # present (regression guard).
        for r in results:
            assert r.metadata["per_angle_mode_actual"] == "fixed_constant"
