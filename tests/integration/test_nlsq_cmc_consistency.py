"""Cross-module consistency tests for NLSQ ↔ CMC interfaces.

Verifies that the canonical parameter ordering exposed by
`get_param_names_in_order` is consistent with the priors produced by
`build_default_priors`, and that `build_init_values_dict` keys form a
valid subset of the ordered name list.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Parameter name consistency
# ---------------------------------------------------------------------------


class TestParamNamesConsistency:
    """get_param_names_in_order must agree with build_default_priors keys."""

    def test_param_names_in_order_returns_list(self) -> None:
        """get_param_names_in_order returns a non-empty list of strings."""
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        names = get_param_names_in_order()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_param_names_no_duplicates(self) -> None:
        """Each parameter name appears exactly once in the ordered list."""
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        names = get_param_names_in_order()
        assert len(names) == len(set(names))

    def test_param_names_consistency_with_build_default_priors(self) -> None:
        """build_default_priors keys are a subset of get_param_names_in_order."""
        from heterodyne.config.parameter_space import ParameterSpace
        from heterodyne.optimization.cmc.priors import (
            build_default_priors,
            get_param_names_in_order,
        )

        names = set(get_param_names_in_order())
        # Use a default ParameterSpace to drive prior construction
        ps = ParameterSpace()
        priors = build_default_priors(ps)
        prior_keys = set(priors.keys())
        # Every prior key must be a known varying parameter
        assert prior_keys <= names, (
            f"Prior keys not in param_names: {prior_keys - names}"
        )

    def test_param_names_registry_coverage(self) -> None:
        """All varying params in the default registry appear in get_param_names_in_order."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        # ParameterRegistry supports __iter__ over names and __getitem__
        vary_defaults = {
            name for name in DEFAULT_REGISTRY if DEFAULT_REGISTRY[name].vary_default
        }
        ordered = set(get_param_names_in_order())
        assert vary_defaults == ordered, (
            f"Mismatch between registry vary_defaults and get_param_names_in_order.\n"
            f"In registry but not ordered: {vary_defaults - ordered}\n"
            f"In ordered but not registry: {ordered - vary_defaults}"
        )

    def test_param_names_vary_flags_override(self) -> None:
        """Passing vary_flags={"p": False} removes p from the returned list."""
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        all_names = get_param_names_in_order()
        if not all_names:
            pytest.skip("No varying parameters registered")

        # Freeze the first parameter
        first = all_names[0]
        reduced = get_param_names_in_order(vary_flags={first: False})
        assert first not in reduced
        # All other originally-varying params should still be present
        for name in all_names[1:]:
            assert name in reduced


# ---------------------------------------------------------------------------
# build_init_values_dict consistency
# ---------------------------------------------------------------------------


class TestInitValuesDictConsistency:
    """build_init_values_dict keys must be a subset of get_param_names_in_order."""

    def test_init_values_keys_subset_of_ordered_names(self) -> None:
        """Every key in build_init_values_dict is a recognised varying param."""
        from heterodyne.optimization.cmc.priors import (
            build_init_values_dict,
            get_param_names_in_order,
        )

        ordered = set(get_param_names_in_order())
        init = build_init_values_dict()
        init_keys = set(init.keys())
        assert init_keys <= ordered, (
            f"Init-values keys not in ordered names: {init_keys - ordered}"
        )

    def test_init_values_are_finite_floats(self) -> None:
        """All initial values are finite Python floats."""
        from heterodyne.optimization.cmc.priors import build_init_values_dict

        init = build_init_values_dict()
        for name, value in init.items():
            assert isinstance(value, float), (
                f"{name}: expected float, got {type(value)}"
            )
            assert np.isfinite(value), f"{name}: value {value} is not finite"

    def test_init_values_within_registry_bounds(self) -> None:
        """All initial values satisfy their registry bounds."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
        from heterodyne.optimization.cmc.priors import build_init_values_dict

        init = build_init_values_dict()
        for name, value in init.items():
            if name not in DEFAULT_REGISTRY:
                continue
            info = DEFAULT_REGISTRY[name]
            assert value >= info.min_bound, (
                f"{name}: value {value:.4e} < min_bound {info.min_bound:.4e}"
            )
            assert value <= info.max_bound, (
                f"{name}: value {value:.4e} > max_bound {info.max_bound:.4e}"
            )

    def test_init_values_with_nlsq_override(self) -> None:
        """NLSQ-provided values are used in preference to registry defaults."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
        from heterodyne.optimization.cmc.priors import (
            build_init_values_dict,
            get_param_names_in_order,
        )

        names = get_param_names_in_order()
        if not names:
            pytest.skip("No varying parameters")

        # Pick a parameter and override it with a custom value within bounds
        target_name = names[0]
        info = DEFAULT_REGISTRY[target_name]
        mid = (info.min_bound + info.max_bound) / 2.0
        # Guard against infinite bounds
        if not np.isfinite(mid):
            mid = info.default

        nlsq_override = {target_name: mid}
        init = build_init_values_dict(nlsq_values=nlsq_override)
        assert init[target_name] == pytest.approx(mid, rel=1e-10)

    def test_init_values_fallback_prior_mean(self) -> None:
        """fallback='prior_mean' uses registry prior_mean when available."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY
        from heterodyne.optimization.cmc.priors import (
            build_init_values_dict,
            get_param_names_in_order,
        )

        names = get_param_names_in_order()
        # Find a parameter that has a prior_mean set
        target = next(
            (n for n in names if DEFAULT_REGISTRY[n].prior_mean is not None),
            None,
        )
        if target is None:
            pytest.skip("No parameter with prior_mean set")

        init_pm = build_init_values_dict(fallback="prior_mean")
        init_def = build_init_values_dict(fallback="default")

        info = DEFAULT_REGISTRY[target]
        # With fallback="prior_mean", value should equal prior_mean (clamped)
        expected_pm = float(
            max(info.min_bound, min(info.max_bound, info.prior_mean))  # type: ignore[arg-type]
        )
        assert init_pm[target] == pytest.approx(expected_pm, rel=1e-9)

        # With fallback="default", value should equal default (clamped)
        expected_def = float(max(info.min_bound, min(info.max_bound, info.default)))
        assert init_def[target] == pytest.approx(expected_def, rel=1e-9)
