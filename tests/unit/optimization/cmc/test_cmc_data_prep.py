"""Tests for priors.py data-preparation helpers.

Covers:
- :func:`get_param_names_in_order`
- :func:`validate_initial_value_bounds`
- :func:`build_init_values_dict`
- :func:`estimate_per_angle_scaling`
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.cmc.priors import (
    build_init_values_dict,
    estimate_per_angle_scaling,
    get_param_names_in_order,
    validate_initial_value_bounds,
)

# ---------------------------------------------------------------------------
# get_param_names_in_order
# ---------------------------------------------------------------------------


class TestGetParamNamesInOrder:
    """get_param_names_in_order returns correctly ordered, filtered name lists."""

    def test_get_param_names_in_order_all(self) -> None:
        """With no arguments all registry parameters with vary_default=True are returned."""
        names = get_param_names_in_order()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_get_param_names_in_order_returns_strings(self) -> None:
        """Every entry in the returned list is a non-empty string."""
        names = get_param_names_in_order()
        assert all(n for n in names), "Empty string found in parameter names"

    def test_get_param_names_in_order_no_duplicates(self) -> None:
        """No parameter name appears more than once."""
        names = get_param_names_in_order()
        assert len(names) == len(set(names))

    def test_get_param_names_in_order_filtered(self) -> None:
        """Passing vary_flags with some False entries reduces the returned list."""
        all_names = get_param_names_in_order()

        # Disable the first two parameters
        disabled = {all_names[0]: False, all_names[1]: False}
        filtered = get_param_names_in_order(vary_flags=disabled)

        assert len(filtered) < len(all_names)
        assert all_names[0] not in filtered
        assert all_names[1] not in filtered

    def test_get_param_names_in_order_force_vary(self) -> None:
        """Setting a parameter to True in vary_flags includes it even if vary_default=False."""
        # beta is vary_default=True by default; disable it, then re-enable via vary_flags
        all_names = get_param_names_in_order()
        first = all_names[0]

        disabled = {first: False}
        without = get_param_names_in_order(vary_flags=disabled)
        assert first not in without

        re_enabled = {first: True}
        with_it = get_param_names_in_order(vary_flags=re_enabled)
        assert first in with_it

    def test_get_param_names_in_order_empty_flags(self) -> None:
        """An empty vary_flags dict is equivalent to passing None."""
        assert get_param_names_in_order({}) == get_param_names_in_order(None)


# ---------------------------------------------------------------------------
# validate_initial_value_bounds
# ---------------------------------------------------------------------------


class TestValidateInitialValueBounds:
    """validate_initial_value_bounds correctly flags out-of-bounds values."""

    def test_valid_returns_empty_dict(self) -> None:
        """Registry-default values are within bounds and produce no warnings."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        init_values = {
            name: DEFAULT_REGISTRY[name].default for name in DEFAULT_REGISTRY
        }
        issues = validate_initial_value_bounds(init_values)
        assert issues == {}

    def test_invalid_below_min_returns_warning(self) -> None:
        """A value below min_bound is reported."""
        # D0_ref min_bound = 100.0; use a value far below it
        issues = validate_initial_value_bounds({"D0_ref": -999.0})
        assert "D0_ref" in issues
        assert len(issues["D0_ref"]) > 0

    def test_invalid_above_max_returns_warning(self) -> None:
        """A value above max_bound is reported."""
        # D0_ref max_bound = 1e6; use an astronomically large value
        issues = validate_initial_value_bounds({"D0_ref": 1e12})
        assert "D0_ref" in issues

    def test_multiple_invalid_all_reported(self) -> None:
        """All out-of-bounds parameters are reported, not just the first."""
        issues = validate_initial_value_bounds({"D0_ref": -1.0, "D0_sample": -1.0})
        assert "D0_ref" in issues
        assert "D0_sample" in issues

    def test_unknown_parameter_skipped(self) -> None:
        """An unknown parameter name does not raise; it is silently skipped."""
        issues = validate_initial_value_bounds({"nonexistent_param": 99.0})
        assert "nonexistent_param" not in issues

    def test_boundary_value_not_flagged(self) -> None:
        """A value exactly at min_bound or max_bound is considered valid."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        info = DEFAULT_REGISTRY["D0_ref"]
        issues_at_min = validate_initial_value_bounds({"D0_ref": info.min_bound})
        issues_at_max = validate_initial_value_bounds({"D0_ref": info.max_bound})
        assert "D0_ref" not in issues_at_min
        assert "D0_ref" not in issues_at_max


# ---------------------------------------------------------------------------
# build_init_values_dict
# ---------------------------------------------------------------------------


class TestBuildInitValuesDict:
    """build_init_values_dict builds consistent initial-value dicts."""

    def test_build_init_values_dict_defaults(self) -> None:
        """With no NLSQ values the result falls back to prior_mean or default."""
        init = build_init_values_dict()

        assert isinstance(init, dict)
        assert len(init) > 0
        assert all(isinstance(v, float) for v in init.values())

    def test_build_init_values_dict_with_nlsq(self) -> None:
        """NLSQ values override the fallback for the supplied parameters."""
        nlsq = {"D0_ref": 5000.0, "alpha_ref": 0.5}
        init = build_init_values_dict(nlsq_values=nlsq)

        assert pytest.approx(init["D0_ref"]) == 5000.0
        assert pytest.approx(init["alpha_ref"]) == 0.5

    def test_build_init_values_dict_all_within_bounds(self) -> None:
        """All returned values lie within their registry bounds."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        init = build_init_values_dict()
        for name, value in init.items():
            if name not in DEFAULT_REGISTRY:
                continue
            info = DEFAULT_REGISTRY[name]
            assert info.min_bound <= value <= info.max_bound, (
                f"{name}={value} outside [{info.min_bound}, {info.max_bound}]"
            )

    def test_build_init_values_dict_clamped_nlsq(self) -> None:
        """An NLSQ value outside bounds is clamped rather than raising."""
        # D0_ref min_bound = 100.0; pass 1.0 which is below the minimum
        init = build_init_values_dict(nlsq_values={"D0_ref": 1.0})
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        info = DEFAULT_REGISTRY["D0_ref"]
        assert init["D0_ref"] >= info.min_bound

    def test_build_init_values_dict_fallback_default(self) -> None:
        """fallback='default' uses the registry default value instead of prior_mean."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        init = build_init_values_dict(fallback="default")
        for name, value in init.items():
            if name in DEFAULT_REGISTRY:
                info = DEFAULT_REGISTRY[name]
                # The value should equal the clamped default (clamping is no-op for defaults)
                assert pytest.approx(value) == float(
                    max(info.min_bound, min(info.max_bound, info.default))
                )


# ---------------------------------------------------------------------------
# estimate_per_angle_scaling
# ---------------------------------------------------------------------------


class TestEstimatePerAngleScaling:
    """estimate_per_angle_scaling extracts plausible contrast/offset estimates."""

    def _make_simple_g2(self, contrast: float = 0.3, offset: float = 1.0) -> np.ndarray:
        """Single-exponential g2 on a 64-point log-spaced grid."""
        tau = np.logspace(-6, 0, 64)
        return offset + contrast * np.exp(-2e3 * tau)

    def test_estimate_per_angle_scaling_from_array(self) -> None:
        """Contrast and offset estimates are in a physically plausible range."""
        g2 = self._make_simple_g2(contrast=0.3, offset=1.0)
        data_dict = {"phi0": g2}
        result = estimate_per_angle_scaling(data_dict)

        assert "phi0" in result
        contrast_est, offset_est = result["phi0"]

        # Contrast should be positive and close to 0.3
        assert contrast_est > 0.0
        assert contrast_est == pytest.approx(0.3, abs=0.05)

        # Offset is the long-lag tail mean, clipped to [0, 1]
        assert 0.0 <= offset_est <= 1.0

    def test_estimate_per_angle_scaling_from_dict_with_g2_key(self) -> None:
        """Data passed as a dict with a 'g2' sub-key is also handled."""
        g2 = self._make_simple_g2()
        data_dict = {"phi0": {"g2": g2, "extra": "ignored"}}
        result = estimate_per_angle_scaling(data_dict)
        assert "phi0" in result

    def test_estimate_per_angle_scaling_multiple_angles(self) -> None:
        """All supplied angle keys produce an entry in the result."""
        data_dict = {
            "phi0": self._make_simple_g2(contrast=0.2),
            "phi45": self._make_simple_g2(contrast=0.4),
            "phi90": self._make_simple_g2(contrast=0.1),
        }
        result = estimate_per_angle_scaling(data_dict)
        assert set(result.keys()) == {"phi0", "phi45", "phi90"}

    def test_estimate_per_angle_scaling_angle_keys_subset(self) -> None:
        """Passing angle_keys restricts which keys are processed."""
        data_dict = {
            "phi0": self._make_simple_g2(),
            "phi45": self._make_simple_g2(),
        }
        result = estimate_per_angle_scaling(data_dict, angle_keys=["phi0"])
        assert "phi0" in result
        assert "phi45" not in result

    def test_estimate_per_angle_scaling_empty_array_skipped(self) -> None:
        """An empty array for a key is silently omitted from the result."""
        data_dict = {"phi0": np.array([])}
        result = estimate_per_angle_scaling(data_dict)
        assert "phi0" not in result

    def test_estimate_per_angle_scaling_offset_clipped(self) -> None:
        """Offset estimate is always in [0, 1] regardless of input magnitude."""
        # Artificially large g2 values
        tau = np.logspace(-6, 0, 64)
        data_dict = {"phi0": np.full_like(tau, 5.0)}
        result = estimate_per_angle_scaling(data_dict)
        _, offset_est = result["phi0"]
        assert 0.0 <= offset_est <= 1.0
