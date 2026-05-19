"""Regression tests for the public warm-start clamp API (P2-a).

Originally tested :mod:`heterodyne.optimization.cmc.warmstart` directly.
Phase 4 PR 4 Task 4.4 (spec §7 Rule 1) absorbed warmstart.py into
:mod:`heterodyne.optimization.cmc.priors`; imports updated accordingly.
Python API users who call ``fit_cmc_jax`` outside the CLI get the same
boundary-clamp protection that ``optimization_runner`` provides.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.optimization.cmc.priors import (
    BOUNDARY_INTERIOR_MARGIN,
    clamp_params_to_interior,
    clamp_to_interior,
)
from heterodyne.optimization.nlsq.results import NLSQResult


def _make_result(values: dict[str, float]) -> NLSQResult:
    names = list(values.keys())
    params = np.asarray([values[n] for n in names], dtype=float)
    return NLSQResult(
        parameters=params,
        parameter_names=names,
        success=True,
        message="ok",
    )


class TestClampNLSQResult:
    def test_interior_params_untouched(self) -> None:
        # Pick values comfortably inside each parameter's bounds.
        result = _make_result(
            {"alpha_ref": 0.0, "f0": 0.5, "v_offset": 0.0, "phi0": 0.0}
        )
        clamped = clamp_to_interior(result)
        # `is` works because the implementation short-circuits to ``return
        # result`` when no parameter was moved.
        assert clamped is result

    def test_log_space_uses_geometric_margin(self) -> None:
        # D0_ref is log_space=True; min_bound > 0.  A value at min_bound
        # must be shifted to min_bound * (max/min)^margin, NOT to a linear
        # midpoint of the bounds.
        info = DEFAULT_REGISTRY["D0_ref"]
        assert info.log_space and info.min_bound > 0
        result = _make_result({"D0_ref": info.min_bound})
        clamped = clamp_to_interior(result)
        log_range = math.log(info.max_bound / info.min_bound)
        expected = info.min_bound * math.exp(BOUNDARY_INTERIOR_MARGIN * log_range)
        assert clamped is not result  # something was clamped
        got = float(clamped.parameters[0])
        # Allow 1e-9 absolute tolerance for float roundtrip.
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9), (
            f"geometric clamp mismatch: got {got}, expected {expected}"
        )

    def test_linear_parameter_uses_linear_margin(self) -> None:
        # alpha_ref is NOT log_space; bounds are e.g. [-5, 5]. A value at
        # min_bound must be shifted to min_bound + 0.05 * range.
        info = DEFAULT_REGISTRY["alpha_ref"]
        result = _make_result({"alpha_ref": info.min_bound})
        clamped = clamp_to_interior(result)
        span = info.max_bound - info.min_bound
        expected = info.min_bound + BOUNDARY_INTERIOR_MARGIN * span
        got = float(clamped.parameters[0])
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)

    def test_fixed_param_overrides_apply_before_bounds_clamp(self) -> None:
        # A stale NLSQResult value gets replaced with the fixed override
        # without going through the linear/geometric clamp first.
        result = _make_result({"phi0": 30.0})
        out = clamp_to_interior(result, fixed_param_overrides={"phi0": 7.5})
        assert float(out.parameters[0]) == 7.5


class TestClampToInterior:
    def test_raw_array_path_returns_clamped_and_names(self) -> None:
        info = DEFAULT_REGISTRY["alpha_ref"]
        params = np.asarray([info.min_bound], dtype=float)
        new_params, clamped_names = clamp_params_to_interior(params, ["alpha_ref"])
        assert clamped_names == ["alpha_ref"]
        span = info.max_bound - info.min_bound
        expected = info.min_bound + BOUNDARY_INTERIOR_MARGIN * span
        assert math.isclose(float(new_params[0]), expected, rel_tol=1e-9, abs_tol=1e-9)

    def test_raw_array_path_passes_unknown_names_through(self) -> None:
        params = np.asarray([0.42], dtype=float)
        new_params, clamped = clamp_params_to_interior(params, ["not_a_parameter"])
        assert clamped == []
        assert float(new_params[0]) == 0.42

    def test_raw_array_path_does_not_mutate_input(self) -> None:
        # Ensure the function returns a copy, not an alias.
        info = DEFAULT_REGISTRY["alpha_ref"]
        params = np.asarray([info.min_bound], dtype=float)
        original = params.copy()
        clamp_params_to_interior(params, ["alpha_ref"])
        np.testing.assert_array_equal(params, original)


class TestCLIShimRedirect:
    """The CLI's private ``_clamp_warmstart_to_interior`` must now forward
    to the public ``cmc.priors`` API (post Task 4.4 absorption) rather than
    carry its own copy.
    """

    def test_cli_imports_public_warmstart_module(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "heterodyne"
            / "cli"
            / "optimization_runner.py"
        )
        text = path.read_text()
        # Phase 4 PR 4 Task 4.4: warmstart.py absorbed into priors.py;
        # CLI now imports from priors instead of warmstart.
        assert "from heterodyne.optimization.cmc.priors import" in text, (
            "CLI must source the boundary clamp from the public "
            "heterodyne.optimization.cmc.priors module (post Task 4.4 absorption)"
        )
