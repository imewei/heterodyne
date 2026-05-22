"""Tests for config/parameter_registry.py module.

Covers ParameterInfo, ParameterRegistry methods for improved coverage.
"""

from __future__ import annotations

import pytest

from heterodyne.config.parameter_names import (
    ALL_PARAM_NAMES,
    ALL_PARAM_NAMES_WITH_SCALING,
)
from heterodyne.config.parameter_registry import (
    DEFAULT_REGISTRY,
    SCALING_PARAMS,
    ParameterInfo,
    ParameterRegistry,
)

# ============================================================================
# Test ParameterInfo
# ============================================================================


class TestParameterInfo:
    """Tests for ParameterInfo dataclass."""

    @pytest.fixture
    def param_info(self) -> ParameterInfo:
        """Create a ParameterInfo for testing."""
        return ParameterInfo(
            name="test_param",
            default=1.0,
            min_bound=0.0,
            max_bound=10.0,
            description="A test parameter",
            unit="Å",
            group="test",
            vary_default=True,
        )

    def test_validate_value_within_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns True for value within bounds."""
        assert param_info.validate_value(5.0) is True

    def test_validate_value_at_lower_bound(self, param_info: ParameterInfo) -> None:
        """validate_value returns True at lower bound."""
        assert param_info.validate_value(0.0) is True

    def test_validate_value_at_upper_bound(self, param_info: ParameterInfo) -> None:
        """validate_value returns True at upper bound."""
        assert param_info.validate_value(10.0) is True

    def test_validate_value_below_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns False below lower bound."""
        assert param_info.validate_value(-1.0) is False

    def test_validate_value_above_bounds(self, param_info: ParameterInfo) -> None:
        """validate_value returns False above upper bound."""
        assert param_info.validate_value(11.0) is False

    def test_clip_value_within_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value returns same value if within bounds."""
        assert param_info.clip_value(5.0) == 5.0

    def test_clip_value_below_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value clips to lower bound."""
        assert param_info.clip_value(-5.0) == 0.0

    def test_clip_value_above_bounds(self, param_info: ParameterInfo) -> None:
        """clip_value clips to upper bound."""
        assert param_info.clip_value(15.0) == 10.0

    def test_frozen_dataclass(self, param_info: ParameterInfo) -> None:
        """ParameterInfo is frozen (immutable)."""
        with pytest.raises(AttributeError):
            param_info.default = 2.0  # type: ignore[misc]

    def test_new_metadata_fields(self) -> None:
        """ParameterInfo supports log_space, prior_mean/std, is_scaling."""
        info = ParameterInfo(
            name="test",
            default=100.0,
            min_bound=1.0,
            max_bound=1e5,
            description="test",
            log_space=True,
            prior_mean=50000.0,
            prior_std=25000.0,
            is_scaling=False,
        )
        assert info.log_space is True
        assert info.prior_mean == 50000.0
        assert info.prior_std == 25000.0
        assert info.is_scaling is False

    def test_default_metadata_fields(self, param_info: ParameterInfo) -> None:
        """New metadata fields have sensible defaults."""
        assert param_info.log_space is False
        assert param_info.prior_mean is None
        assert param_info.prior_std is None
        assert param_info.is_scaling is False


# ============================================================================
# Test ParameterRegistry
# ============================================================================


class TestParameterRegistry:
    """Tests for ParameterRegistry class."""

    @pytest.fixture
    def registry(self) -> ParameterRegistry:
        """Create a ParameterRegistry for testing."""
        return ParameterRegistry()

    def test_len(self, registry: ParameterRegistry) -> None:
        """len(registry) returns 16 (14 physics + 2 scaling)."""
        assert len(registry) == 16

    def test_getitem_valid_name(self, registry: ParameterRegistry) -> None:
        """__getitem__ returns ParameterInfo for valid name."""
        info = registry["D0_ref"]
        assert isinstance(info, ParameterInfo)
        assert info.name == "D0_ref"

    def test_getitem_invalid_name_raises(self, registry: ParameterRegistry) -> None:
        """__getitem__ raises KeyError for invalid name."""
        with pytest.raises(KeyError, match="Unknown parameter"):
            _ = registry["invalid_name"]

    def test_iter_yields_canonical_order(self, registry: ParameterRegistry) -> None:
        """__iter__ yields names in canonical order (including scaling)."""
        names = list(registry)
        assert names == list(ALL_PARAM_NAMES_WITH_SCALING)

    def test_get_defaults(self, registry: ParameterRegistry) -> None:
        """get_defaults returns dict of default values."""
        defaults = registry.get_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) == 16
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            assert name in defaults

    def test_get_bounds(self, registry: ParameterRegistry) -> None:
        """get_bounds returns lower and upper bounds lists."""
        lower, upper = registry.get_bounds()
        assert len(lower) == 16
        assert len(upper) == 16
        for lo, hi in zip(lower, upper, strict=True):
            assert lo <= hi

    def test_get_group_reference(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for reference group."""
        group = registry.get_group("reference")
        assert len(group) == 3
        assert all(isinstance(p, ParameterInfo) for p in group)
        names = [p.name for p in group]
        assert "D0_ref" in names
        assert "alpha_ref" in names
        assert "D_offset_ref" in names

    def test_get_group_sample(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for sample group."""
        group = registry.get_group("sample")
        assert len(group) == 3

    def test_get_group_velocity(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for velocity group."""
        group = registry.get_group("velocity")
        assert len(group) == 3

    def test_get_group_fraction(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for fraction group."""
        group = registry.get_group("fraction")
        assert len(group) == 4

    def test_get_group_angle(self, registry: ParameterRegistry) -> None:
        """get_group returns ParameterInfo list for angle group."""
        group = registry.get_group("angle")
        assert len(group) == 1
        assert group[0].name == "phi0"

    def test_get_group_invalid_raises(self, registry: ParameterRegistry) -> None:
        """get_group raises KeyError for invalid group name."""
        with pytest.raises(KeyError, match="Unknown group"):
            registry.get_group("invalid_group")

    def test_get_varying_indices_default_flags(
        self, registry: ParameterRegistry
    ) -> None:
        """get_varying_indices uses default vary flags when empty dict passed."""
        indices = registry.get_varying_indices({})
        assert len(indices) > 0
        assert all(isinstance(i, int) for i in indices)

    def test_get_varying_indices_custom_flags(
        self, registry: ParameterRegistry
    ) -> None:
        """get_varying_indices uses provided vary flags."""
        all_names = list(ALL_PARAM_NAMES_WITH_SCALING)
        # All fixed
        indices = registry.get_varying_indices(dict.fromkeys(all_names, False))
        assert len(indices) == 0

        # All varying
        indices = registry.get_varying_indices(dict.fromkeys(all_names, True))
        assert len(indices) == 16

    def test_scaling_params_have_is_scaling(self, registry: ParameterRegistry) -> None:
        """Contrast and offset have is_scaling=True."""
        assert registry["contrast"].is_scaling is True
        assert registry["offset"].is_scaling is True

    def test_physics_params_not_scaling(self, registry: ParameterRegistry) -> None:
        """Physics params have is_scaling=False."""
        for name in ALL_PARAM_NAMES:
            assert registry[name].is_scaling is False


# ============================================================================
# Test DEFAULT_REGISTRY module constant
# ============================================================================


class TestDefaultRegistry:
    """Tests for DEFAULT_REGISTRY module constant."""

    def test_default_registry_exists(self) -> None:
        """DEFAULT_REGISTRY is a ParameterRegistry instance."""
        assert isinstance(DEFAULT_REGISTRY, ParameterRegistry)

    def test_default_registry_has_all_params(self) -> None:
        """DEFAULT_REGISTRY contains all 16 parameters."""
        assert len(DEFAULT_REGISTRY) == 16

    def test_default_registry_d0_ref_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct D0_ref bounds."""
        info = DEFAULT_REGISTRY["D0_ref"]
        assert info.min_bound == 100.0
        assert info.max_bound == 1e6

    def test_default_registry_alpha_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct alpha bounds (widened to [-5, 5] to support sub-diffusive regimes)."""
        for name in ("alpha_ref", "alpha_sample"):
            info = DEFAULT_REGISTRY[name]
            assert info.min_bound == -5.0, f"{name} min_bound"
            assert info.max_bound == 5.0, f"{name} max_bound"

    def test_default_registry_fraction_bounds(self) -> None:
        """DEFAULT_REGISTRY has correct fraction parameter bounds."""
        info = DEFAULT_REGISTRY["f0"]
        assert info.min_bound == 0.0
        assert info.max_bound == 1.0

    def test_default_registry_vary_defaults(self) -> None:
        """DEFAULT_REGISTRY has sensible vary_default flags."""
        # Main parameters vary by default
        assert DEFAULT_REGISTRY["D0_ref"].vary_default is True

    def test_default_registry_v0_bounds(self) -> None:
        """DEFAULT_REGISTRY has non-negative v0 with log_space."""
        info = DEFAULT_REGISTRY["v0"]
        assert info.min_bound == 1e-6
        assert info.max_bound == 1e4
        assert info.log_space is True

    def test_default_registry_phi0_bounds(self) -> None:
        """DEFAULT_REGISTRY has phi0 bounds."""
        info = DEFAULT_REGISTRY["phi0"]
        assert info.min_bound == -10.0
        assert info.max_bound == 10.0

    def test_default_registry_units_angstroms(self) -> None:
        """All length units use Å (angstroms), not nm."""
        for name in ALL_PARAM_NAMES:
            info = DEFAULT_REGISTRY[name]
            assert "nm" not in info.unit, f"{name} still uses nm: {info.unit}"

    def test_default_values_within_bounds(self) -> None:
        """All default values are within their bounds."""
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            if name not in DEFAULT_REGISTRY._parameters:
                continue
            info = DEFAULT_REGISTRY[name]
            assert info.validate_value(info.default), (
                f"{name} default {info.default} outside [{info.min_bound}, {info.max_bound}]"
            )


# ============================================================================
# Test SCALING_PARAMS
# ============================================================================


class TestScalingParams:
    """Tests for SCALING_PARAMS module constant."""

    def test_scaling_params_has_contrast_and_offset(self) -> None:
        """SCALING_PARAMS contains contrast and offset."""
        assert "contrast" in SCALING_PARAMS
        assert "offset" in SCALING_PARAMS
        assert len(SCALING_PARAMS) == 2

    def test_contrast_bounds(self) -> None:
        """Contrast parameter has [0, 1] bounds."""
        info = SCALING_PARAMS["contrast"]
        assert info.min_bound == 0.0
        assert info.max_bound == 1.0
        assert info.is_scaling is True

    def test_offset_bounds(self) -> None:
        """Offset parameter has [0.5, 1.5] bounds."""
        info = SCALING_PARAMS["offset"]
        assert info.min_bound == 0.5
        assert info.max_bound == 1.5
        assert info.is_scaling is True


# ============================================================================
# Prior-sanity contract (deep-RCA Prevention 1)
# ============================================================================


class TestPriorSanity:
    """Contract tests that pin the registry priors against the geometric
    requirements of NUTS sampling.  The deep-RCA finding F2 was that tight
    log-space priors (D0 prior_std=5e3 → log-space std=0.40) collapse the
    posterior onto the registry mean and trigger divergence cascades; these
    tests ensure future edits cannot silently re-tighten the priors below
    the geometric threshold.

    Two checks per parameter:
      (a) linear-space prior_std/|prior_mean| > 0.10 — the prior must span
          at least ±10% of its centre, else it dominates over the data.
      (b) log-space std > 0.50 — log-scale parameters need ≥ half an
          e-fold of width to accommodate the empirical NLSQ posterior.

    A parameter passes if EITHER (a) holds in linear space OR (b) holds in
    log space (log_space=True).  Parameters centred on zero (prior_mean=0)
    are exempt from (a) and are checked only via their absolute width.
    """

    @pytest.mark.parametrize(
        "name",
        [n for n in DEFAULT_REGISTRY],  # noqa: C416 — registry is iterable, not dict-like
    )
    def test_prior_width_is_geometrically_adequate(self, name: str) -> None:
        import math

        info = DEFAULT_REGISTRY[name]
        if info.prior_mean is None or info.prior_std is None:
            pytest.skip(f"{name} has no registry prior")
        prior_mean = float(info.prior_mean)
        prior_std = float(info.prior_std)
        assert prior_std > 0.0, f"{name}: prior_std must be > 0"

        if info.log_space:
            # log-space std = log(1 + prior_std/prior_mean) for log-normal
            # approximation; require ≥ 0.5 (≈ half an e-fold)
            assert prior_mean > 0, (
                f"{name}: log_space=True but prior_mean={prior_mean} is non-positive"
            )
            log_std = math.log1p(prior_std / prior_mean)
            assert log_std > 0.5, (
                f"{name}: log-space std = {log_std:.3f} too tight for NUTS "
                f"(prior_mean={prior_mean:g}, prior_std={prior_std:g}); "
                "see deep-RCA F2 — D0 with std=5e3/mean=1e4 produced 0.40 "
                "which collapsed posteriors"
            )
        elif prior_mean != 0.0:
            ratio = prior_std / abs(prior_mean)
            assert ratio > 0.10, (
                f"{name}: prior_std/|prior_mean| = {ratio:.3f} too tight; "
                "registry prior would dominate over data likelihood"
            )
        else:
            # Zero-centred priors: just require nonzero absolute width.
            # Concrete adequacy depends on the parameter scale and is
            # enforced by integration tests, not this contract.
            assert prior_std > 0.0

    def test_dual_prior_specs_in_sync(self) -> None:
        """CLAUDE.md rule #9 — registry prior_mean/prior_std MUST equal
        the (loc, scale) tuple in parameter_space._DEFAULT_PRIOR_SPECS for
        every physical parameter that appears in both.  Drift between the
        two breaks the dual-prior contract and produces silently
        inconsistent CMC inference."""
        from heterodyne.config.parameter_space import _DEFAULT_PRIOR_SPECS

        # Forward direction: every registry entry with a defined prior must
        # have a matching entry in _DEFAULT_PRIOR_SPECS with equal (loc, scale).
        registry_priors: set[str] = set()
        for name in DEFAULT_REGISTRY:
            info = DEFAULT_REGISTRY[name]
            if info.prior_mean is None or info.prior_std is None:
                continue
            registry_priors.add(name)
            assert name in _DEFAULT_PRIOR_SPECS, (
                f"{name}: defined in DEFAULT_REGISTRY with "
                f"prior_mean={info.prior_mean!r}, prior_std={info.prior_std!r} "
                f"but missing from parameter_space._DEFAULT_PRIOR_SPECS "
                f"(CLAUDE.md rule #9 — one-sided drift breaks CMC priors)"
            )
            spec_loc, spec_scale = _DEFAULT_PRIOR_SPECS[name]
            assert float(info.prior_mean) == pytest.approx(float(spec_loc)), (
                f"{name}: registry prior_mean={info.prior_mean!r} != "
                f"parameter_space loc={spec_loc!r} (CLAUDE.md rule #9)"
            )
            assert float(info.prior_std) == pytest.approx(float(spec_scale)), (
                f"{name}: registry prior_std={info.prior_std!r} != "
                f"parameter_space scale={spec_scale!r} (CLAUDE.md rule #9)"
            )

        # Reverse direction: every _DEFAULT_PRIOR_SPECS entry must be backed
        # by a registry entry with the same numeric values. Drift in this
        # direction would silently broaden CMC priors beyond the registry.
        for name in _DEFAULT_PRIOR_SPECS:
            assert name in registry_priors, (
                f"{name}: present in parameter_space._DEFAULT_PRIOR_SPECS "
                f"but DEFAULT_REGISTRY[{name!r}] has prior_mean/prior_std=None. "
                f"Either remove from _DEFAULT_PRIOR_SPECS or populate the "
                f"registry entry (CLAUDE.md rule #9)."
            )
