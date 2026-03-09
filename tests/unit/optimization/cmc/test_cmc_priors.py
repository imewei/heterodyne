"""Tests for CMC prior construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpyro.distributions as dist
import pytest
from numpyro.distributions.truncated import TwoSidedTruncatedDistribution

from heterodyne.optimization.cmc.priors import (
    build_default_priors,
    build_log_space_priors,
    build_nlsq_informed_priors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockParamInfo:
    """Minimal stand-in for ParameterInfo with new metadata fields."""

    name: str
    default: float
    min_bound: float
    max_bound: float
    prior_mean: float | None = None
    prior_std: float | None = None
    log_space: bool = False
    vary_default: bool = True
    description: str = ""
    unit: str = ""
    group: str = ""


def _make_mock_registry(entries: dict[str, MockParamInfo]):
    """Create a mock registry that supports __getitem__."""
    reg = MagicMock()
    reg.__getitem__ = lambda self, key: entries[key]
    return reg


@dataclass
class MockParameterSpace:
    """Minimal stand-in for ParameterSpace."""

    varying_names: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class MockNLSQResult:
    """Minimal stand-in for NLSQResult."""

    parameter_names: list[str] = field(default_factory=list)
    _values: dict[str, float] = field(default_factory=dict)
    _uncertainties: dict[str, float | None] = field(default_factory=dict)

    def get_param(self, name: str) -> float:
        return self._values[name]

    def get_uncertainty(self, name: str) -> float | None:
        return self._uncertainties.get(name)


# ===========================================================================
# build_nlsq_informed_priors
# ===========================================================================

class TestBuildNLSQInformedPriors:

    @pytest.mark.unit
    def test_uses_nlsq_values_and_uncertainty(self) -> None:
        space = MockParameterSpace(
            varying_names=["D0_ref"],
            bounds={"D0_ref": (100.0, 1e5)},
            values={"D0_ref": 500.0},
        )
        nlsq = MockNLSQResult(
            parameter_names=["D0_ref"],
            _values={"D0_ref": 1234.0},
            _uncertainties={"D0_ref": 50.0},
        )

        priors = build_nlsq_informed_priors(nlsq, space, width_factor=2.0)

        assert "D0_ref" in priors
        p = priors["D0_ref"]
        assert isinstance(p, TwoSidedTruncatedDistribution)
        assert float(p.base_dist.loc) == pytest.approx(1234.0)
        assert float(p.base_dist.scale) == pytest.approx(100.0)  # 50 * 2

    @pytest.mark.unit
    def test_falls_back_to_space_value_when_nlsq_missing(self) -> None:
        space = MockParameterSpace(
            varying_names=["alpha_ref"],
            bounds={"alpha_ref": (-2.0, 2.0)},
            values={"alpha_ref": 0.5},
        )
        nlsq = MockNLSQResult(
            parameter_names=[],  # alpha_ref not in NLSQ
            _values={},
            _uncertainties={},
        )

        priors = build_nlsq_informed_priors(nlsq, space)
        p = priors["alpha_ref"]
        assert float(p.base_dist.loc) == pytest.approx(0.5)


# ===========================================================================
# build_default_priors
# ===========================================================================

class TestBuildDefaultPriors:

    @pytest.mark.unit
    def test_truncated_normal_when_prior_mean_and_std(self) -> None:
        info = MockParamInfo(
            name="D0_ref", default=500.0,
            min_bound=100.0, max_bound=1e5,
            prior_mean=50050.0, prior_std=24975.0,
        )
        registry = _make_mock_registry({"D0_ref": info})
        space = MockParameterSpace(
            varying_names=["D0_ref"],
            bounds={"D0_ref": (100.0, 1e5)},
        )

        priors = build_default_priors(space, registry=registry)
        p = priors["D0_ref"]
        assert isinstance(p, TwoSidedTruncatedDistribution)
        assert float(p.base_dist.loc) == pytest.approx(50050.0)

    @pytest.mark.unit
    def test_uniform_fallback_when_no_prior_std(self) -> None:
        info = MockParamInfo(
            name="alpha_ref", default=0.0,
            min_bound=-2.0, max_bound=2.0,
            prior_mean=None, prior_std=None,
        )
        registry = _make_mock_registry({"alpha_ref": info})
        space = MockParameterSpace(
            varying_names=["alpha_ref"],
            bounds={"alpha_ref": (-2.0, 2.0)},
        )

        priors = build_default_priors(space, registry=registry)
        p = priors["alpha_ref"]
        assert isinstance(p, dist.Uniform)


# ===========================================================================
# build_log_space_priors
# ===========================================================================

class TestBuildLogSpacePriors:

    @pytest.mark.unit
    def test_creates_lognormal_for_flagged_params(self) -> None:
        info = MockParamInfo(
            name="D0_ref", default=500.0,
            min_bound=100.0, max_bound=1e5,
            prior_mean=50050.0, prior_std=24975.0,
            log_space=True,
        )
        registry = _make_mock_registry({"D0_ref": info})

        priors = build_log_space_priors(["D0_ref"], registry=registry)
        assert "D0_ref" in priors
        assert isinstance(priors["D0_ref"], dist.LogNormal)

    @pytest.mark.unit
    def test_skips_non_log_space_params(self) -> None:
        info = MockParamInfo(
            name="alpha_ref", default=0.0,
            min_bound=-2.0, max_bound=2.0,
            log_space=False,
        )
        registry = _make_mock_registry({"alpha_ref": info})

        priors = build_log_space_priors(["alpha_ref"], registry=registry)
        assert "alpha_ref" not in priors

    @pytest.mark.unit
    def test_lognormal_median_matches_center(self) -> None:
        """The LogNormal median = exp(mu) should equal prior_mean."""
        center = 5000.0
        info = MockParamInfo(
            name="v0", default=center,
            min_bound=1e-6, max_bound=1e4,
            prior_mean=center, prior_std=2500.0,
            log_space=True,
        )
        registry = _make_mock_registry({"v0": info})

        priors = build_log_space_priors(["v0"], registry=registry)
        p = priors["v0"]
        # median of LogNormal(loc=mu, scale=sigma) = exp(mu)
        median = math.exp(float(p.loc))
        assert median == pytest.approx(center, rel=1e-6)

    @pytest.mark.unit
    def test_cv_to_sigma_conversion(self) -> None:
        """sigma = sqrt(log(1 + CV^2)) for CV = prior_std / prior_mean."""
        center = 1000.0
        std = 500.0  # CV = 0.5
        info = MockParamInfo(
            name="D0_ref", default=center,
            min_bound=100.0, max_bound=1e5,
            prior_mean=center, prior_std=std,
            log_space=True,
        )
        registry = _make_mock_registry({"D0_ref": info})

        priors = build_log_space_priors(["D0_ref"], registry=registry)
        expected_sigma = math.sqrt(math.log1p((std / center) ** 2))
        assert float(priors["D0_ref"].scale) == pytest.approx(expected_sigma, rel=1e-6)
