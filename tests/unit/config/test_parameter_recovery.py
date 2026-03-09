"""Tests for parameter initial-value construction and bounds validation.

Verifies that build_init_values_dict correctly resolves values from
NLSQ estimates, registry prior_means, and registry defaults — and that
validate_initial_value_bounds correctly identifies out-of-range values.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
from heterodyne.optimization.cmc.priors import (
    build_init_values_dict,
    get_param_names_in_order,
    validate_initial_value_bounds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_varying_names() -> list[str]:
    return get_param_names_in_order()


# ---------------------------------------------------------------------------
# build_init_values_dict — prior_mean fallback
# ---------------------------------------------------------------------------


class TestPriorMeanRecovery:
    def test_returns_dict_for_all_varying_params(self) -> None:
        init = build_init_values_dict(fallback="prior_mean")
        varying = _all_varying_names()
        assert set(init.keys()) == set(varying)

    def test_prior_mean_used_when_available(self) -> None:
        init = build_init_values_dict(fallback="prior_mean")
        for name in _all_varying_names():
            info = DEFAULT_REGISTRY[name]
            if info.prior_mean is not None:
                expected = float(
                    max(info.min_bound, min(info.max_bound, info.prior_mean))
                )
                assert init[name] == pytest.approx(expected, rel=1e-9), (
                    f"{name}: expected prior_mean {info.prior_mean} "
                    f"(clamped to {expected}), got {init[name]}"
                )

    def test_fallback_to_default_when_prior_mean_is_none(self) -> None:
        init = build_init_values_dict(fallback="prior_mean")
        for name in _all_varying_names():
            info = DEFAULT_REGISTRY[name]
            if info.prior_mean is None:
                expected = float(
                    max(info.min_bound, min(info.max_bound, info.default))
                )
                assert init[name] == pytest.approx(expected, rel=1e-9), (
                    f"{name}: expected default {info.default} "
                    f"(clamped to {expected}), got {init[name]}"
                )

    def test_all_values_within_bounds(self) -> None:
        init = build_init_values_dict(fallback="prior_mean")
        for name, value in init.items():
            info = DEFAULT_REGISTRY[name]
            assert info.min_bound <= value <= info.max_bound, (
                f"{name}: value {value} outside [{info.min_bound}, {info.max_bound}]"
            )


# ---------------------------------------------------------------------------
# build_init_values_dict — default fallback
# ---------------------------------------------------------------------------


class TestDefaultRecovery:
    def test_returns_dict_for_all_varying_params(self) -> None:
        init = build_init_values_dict(fallback="default")
        varying = _all_varying_names()
        assert set(init.keys()) == set(varying)

    def test_values_match_registry_defaults(self) -> None:
        init = build_init_values_dict(fallback="default")
        for name in _all_varying_names():
            info = DEFAULT_REGISTRY[name]
            expected = float(
                max(info.min_bound, min(info.max_bound, info.default))
            )
            assert init[name] == pytest.approx(expected, rel=1e-9), (
                f"{name}: expected {expected}, got {init[name]}"
            )

    def test_all_values_within_bounds(self) -> None:
        init = build_init_values_dict(fallback="default")
        for name, value in init.items():
            info = DEFAULT_REGISTRY[name]
            assert info.min_bound <= value <= info.max_bound

    def test_invalid_fallback_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="fallback"):
            build_init_values_dict(fallback="bogus")


# ---------------------------------------------------------------------------
# build_init_values_dict — NLSQ override
# ---------------------------------------------------------------------------


class TestNLSQOverride:
    def test_nlsq_values_take_priority_over_prior_mean(self) -> None:
        # Pick a parameter that has a prior_mean and override it
        name = next(
            n
            for n in _all_varying_names()
            if DEFAULT_REGISTRY[n].prior_mean is not None
        )
        info = DEFAULT_REGISTRY[name]
        # Use a value that differs from prior_mean but is within bounds
        override_val = float(
            (info.min_bound + info.max_bound) / 2.0
        )
        init = build_init_values_dict(
            nlsq_values={name: override_val}, fallback="prior_mean"
        )
        assert init[name] == pytest.approx(override_val, rel=1e-9)

    def test_nlsq_values_take_priority_over_default(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        mid = float((info.min_bound + info.max_bound) / 2.0)
        init = build_init_values_dict(
            nlsq_values={name: mid}, fallback="default"
        )
        assert init[name] == pytest.approx(mid, rel=1e-9)

    def test_non_overridden_params_use_fallback(self) -> None:
        varying = _all_varying_names()
        first_name = varying[0]
        info = DEFAULT_REGISTRY[first_name]
        override_val = float((info.min_bound + info.max_bound) / 2.0)

        init = build_init_values_dict(
            nlsq_values={first_name: override_val}, fallback="default"
        )
        for name in varying[1:]:
            inf2 = DEFAULT_REGISTRY[name]
            expected = float(
                max(inf2.min_bound, min(inf2.max_bound, inf2.default))
            )
            assert init[name] == pytest.approx(expected, rel=1e-9)

    def test_out_of_bounds_nlsq_value_is_clamped(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        # Pass a value well above max_bound
        extreme = info.max_bound * 1e6 + 1.0
        init = build_init_values_dict(
            nlsq_values={name: extreme}, fallback="default"
        )
        assert init[name] == pytest.approx(info.max_bound, rel=1e-9)

    def test_empty_nlsq_dict_behaves_like_no_override(self) -> None:
        init_no_override = build_init_values_dict(fallback="default")
        init_empty = build_init_values_dict(nlsq_values={}, fallback="default")
        assert init_no_override == pytest.approx(init_empty, rel=1e-9)


# ---------------------------------------------------------------------------
# validate_initial_value_bounds
# ---------------------------------------------------------------------------


class TestParamBoundsValidation:
    def test_valid_values_return_empty_issues(self) -> None:
        # Use registry defaults — all guaranteed in-bounds
        init = {name: DEFAULT_REGISTRY[name].default for name in _all_varying_names()}
        issues = validate_initial_value_bounds(init)
        assert issues == {}

    def test_value_below_min_bound_reported(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        bad_val = info.min_bound - 1.0
        issues = validate_initial_value_bounds({name: bad_val})
        assert name in issues
        assert any("below" in msg for msg in issues[name])

    def test_value_above_max_bound_reported(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        bad_val = info.max_bound + 1.0
        issues = validate_initial_value_bounds({name: bad_val})
        assert name in issues
        assert any("above" in msg for msg in issues[name])

    def test_exactly_at_min_bound_is_valid(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        issues = validate_initial_value_bounds({name: info.min_bound})
        assert name not in issues

    def test_exactly_at_max_bound_is_valid(self) -> None:
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        issues = validate_initial_value_bounds({name: info.max_bound})
        assert name not in issues

    def test_unknown_parameter_is_skipped(self) -> None:
        issues = validate_initial_value_bounds({"nonexistent_param_xyz": 999.9})
        assert "nonexistent_param_xyz" not in issues

    def test_multiple_violations_all_reported(self) -> None:
        varying = _all_varying_names()
        init = {}
        violating = []
        for name in varying[:3]:
            info = DEFAULT_REGISTRY[name]
            init[name] = info.min_bound - 1.0
            violating.append(name)

        issues = validate_initial_value_bounds(init)
        for name in violating:
            assert name in issues

    def test_custom_param_specs_override_registry_bounds(self) -> None:
        # Use a tight custom spec that makes the default out-of-range
        name = _all_varying_names()[0]
        info = DEFAULT_REGISTRY[name]
        tight_spec = {name: {"min_bound": info.default + 1.0, "max_bound": info.default + 2.0}}
        issues = validate_initial_value_bounds(
            {name: info.default}, param_specs=tight_spec
        )
        assert name in issues

    def test_empty_init_dict_returns_empty_issues(self) -> None:
        issues = validate_initial_value_bounds({})
        assert issues == {}


# ---------------------------------------------------------------------------
# get_param_names_in_order
# ---------------------------------------------------------------------------


class TestGetParamNamesInOrder:
    def test_returns_list_of_strings(self) -> None:
        names = get_param_names_in_order()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_all_names_in_registry(self) -> None:
        names = get_param_names_in_order()
        for name in names:
            assert name in DEFAULT_REGISTRY

    def test_vary_flags_none_uses_registry_defaults(self) -> None:
        names_default = get_param_names_in_order()
        names_none = get_param_names_in_order(vary_flags=None)
        assert names_default == names_none

    def test_vary_flags_override_disables_param(self) -> None:
        all_names = get_param_names_in_order()
        first = all_names[0]
        names_reduced = get_param_names_in_order(vary_flags={first: False})
        assert first not in names_reduced
        assert len(names_reduced) == len(all_names) - 1

    def test_vary_flags_override_enables_param(self) -> None:
        # Find a param with vary_default=False if one exists, else skip
        fixed_params = [
            name for name in DEFAULT_REGISTRY
            if not DEFAULT_REGISTRY[name].vary_default
        ]
        if not fixed_params:
            pytest.skip("No parameters with vary_default=False in registry")
        name = fixed_params[0]
        names = get_param_names_in_order(vary_flags={name: True})
        assert name in names

    def test_result_count_matches_registry_vary_defaults(self) -> None:
        expected_count = sum(
            1 for name in DEFAULT_REGISTRY
            if DEFAULT_REGISTRY[name].vary_default
        )
        names = get_param_names_in_order()
        assert len(names) == expected_count
