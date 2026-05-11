"""Tests for CMC prior construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions.truncated import TwoSidedTruncatedDistribution

from heterodyne.optimization.cmc.priors import (
    build_default_priors,
    build_log_space_priors,
    build_nlsq_informed_priors,
    estimate_contrast_offset_from_data,
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
            name="D0_ref",
            default=500.0,
            min_bound=100.0,
            max_bound=1e5,
            prior_mean=50050.0,
            prior_std=24975.0,
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
            name="alpha_ref",
            default=0.0,
            min_bound=-2.0,
            max_bound=2.0,
            prior_mean=None,
            prior_std=None,
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
            name="D0_ref",
            default=500.0,
            min_bound=100.0,
            max_bound=1e5,
            prior_mean=50050.0,
            prior_std=24975.0,
            log_space=True,
        )
        registry = _make_mock_registry({"D0_ref": info})

        priors = build_log_space_priors(["D0_ref"], registry=registry)
        assert "D0_ref" in priors
        assert isinstance(priors["D0_ref"], dist.LogNormal)

    @pytest.mark.unit
    def test_skips_non_log_space_params(self) -> None:
        info = MockParamInfo(
            name="alpha_ref",
            default=0.0,
            min_bound=-2.0,
            max_bound=2.0,
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
            name="v0",
            default=center,
            min_bound=1e-6,
            max_bound=1e4,
            prior_mean=center,
            prior_std=2500.0,
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
            name="D0_ref",
            default=center,
            min_bound=100.0,
            max_bound=1e5,
            prior_mean=center,
            prior_std=std,
            log_space=True,
        )
        registry = _make_mock_registry({"D0_ref": info})

        priors = build_log_space_priors(["D0_ref"], registry=registry)
        expected_sigma = math.sqrt(math.log1p((std / center) ** 2))
        assert float(priors["D0_ref"].scale) == pytest.approx(expected_sigma, rel=1e-6)


# ===========================================================================
# fit_cmc_sharded sigma tempering regression
# ===========================================================================


def test_fit_cmc_sharded_does_not_scale_sigma():
    """fit_cmc_sharded must NOT divide sigma by sqrt(K) — prior tempering is used instead.

    The new parallel path passes shards to MultiprocessingBackend.run_shards(); this
    test verifies the shard dicts carry the original unscaled sigma.
    """
    from unittest.mock import MagicMock, patch

    from heterodyne.optimization.cmc import CMCConfig
    from heterodyne.optimization.cmc.core import fit_cmc_sharded

    n = 40
    c2 = np.ones((n, n), dtype=np.float64) * 0.5
    sigma_val = 0.1
    num_shards = 4
    wrong_sigma = sigma_val / math.sqrt(num_shards)

    captured_shards: list[dict] = []

    def capture_run_shards(
        shards,
        config,
        initial_values=None,
        parameter_space=None,
        prior_width_multiplier=1.0,
        nlsq_uncertainties=None,
        nlsq_prior_width_factor=2.0,
        progress_bar=True,
    ):
        captured_shards.extend(shards)
        # Return one minimal success dict per shard so the combination doesn't crash.
        results = []
        for i, _ in enumerate(shards):
            results.append(
                {
                    "success": True,
                    "shard_idx": i,
                    "samples": {"D0_ref": np.ones(10)},
                    "param_names": ["D0_ref"],
                    "n_chains": 1,
                    "n_samples": 10,
                    "extra_fields": {},
                    "duration": 0.1,
                    "stats": {"num_divergent": 0, "n_warmup": 10, "n_samples": 10},
                }
            )
        return results

    mock_model = MagicMock()
    mock_model.t = np.linspace(0.001, 0.04, n)
    mock_model.q = 0.005
    mock_model.dt = 0.001
    mock_model.param_manager.space.varying_names = ["D0_ref"]
    mock_model.param_manager.varying_names = ["D0_ref"]
    mock_model.param_manager.space.priors = {}
    mock_model.scaling.get_for_angle.return_value = (1.0, 0.0)

    fake_priors = {"D0_ref": MagicMock()}

    with (
        patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend."
            "MultiprocessingBackend.run_shards",
            side_effect=capture_run_shards,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_nlsq_informed_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_default_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.temper_priors",
            return_value=fake_priors,
        ),
    ):
        try:
            fit_cmc_sharded(
                model=mock_model,
                c2_data=c2,
                sigma=sigma_val,
                num_shards=num_shards,
                sharding_strategy="contiguous",
                config=CMCConfig(num_warmup=10, num_samples=10),
            )
        except Exception:
            pass

    assert len(captured_shards) > 0, (
        "MultiprocessingBackend.run_shards was never called"
    )
    for shard in captured_shards:
        s = shard.get("sigma")
        if s is None:
            continue
        val = float(np.mean(np.asarray(s))) if hasattr(s, "__len__") else float(s)
        assert abs(val - wrong_sigma) > 1e-6, (
            f"sigma was scaled by sqrt(K): got {val:.6f}, "
            f"wrong value would be {wrong_sigma:.6f} (sigma/sqrt({num_shards}))"
        )


class TestEstimateContrastOffsetFromData:
    def _make_c2(
        self, contrast: float = 0.3, offset: float = 0.95, n: int = 2000, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0, 2, n)
        t2 = rng.uniform(0, 2, n)
        c2 = contrast * np.exp(-np.abs(t1 - t2) / 0.3) + offset + rng.normal(0, 0.02, n)
        return c2, t1, t2

    def test_returns_two_floats(self) -> None:
        c2, t1, t2 = self._make_c2()
        result = estimate_contrast_offset_from_data(c2, t1, t2)
        assert len(result) == 2

    def test_contrast_in_default_bounds(self) -> None:
        c2, t1, t2 = self._make_c2()
        contrast, _ = estimate_contrast_offset_from_data(c2, t1, t2)
        assert 0.0 <= contrast <= 1.0

    def test_offset_in_default_bounds(self) -> None:
        c2, t1, t2 = self._make_c2()
        _, offset = estimate_contrast_offset_from_data(c2, t1, t2)
        assert 0.5 <= offset <= 1.5

    def test_low_data_returns_midpoints(self) -> None:
        rng = np.random.default_rng(0)
        c2 = rng.normal(1.0, 0.05, 50)  # fewer than 100 points
        t1 = rng.uniform(0, 1, 50)
        t2 = rng.uniform(0, 1, 50)
        contrast, offset = estimate_contrast_offset_from_data(c2, t1, t2)
        assert contrast == pytest.approx(0.5)  # (0+1)/2
        assert offset == pytest.approx(1.0)  # (0.5+1.5)/2

    def test_custom_bounds_clipped(self) -> None:
        c2, t1, t2 = self._make_c2(contrast=0.05)
        contrast, _ = estimate_contrast_offset_from_data(
            c2, t1, t2, contrast_bounds=(0.0, 0.5)
        )
        assert 0.0 <= contrast <= 0.5

    def test_estimate_roughly_correct(self) -> None:
        c2, t1, t2 = self._make_c2(contrast=0.3, offset=0.95, n=5000)
        contrast, offset = estimate_contrast_offset_from_data(c2, t1, t2)
        assert abs(contrast - 0.3) < 0.15
        assert abs(offset - 0.95) < 0.15


# ===========================================================================
# fit_cmc_sharded NLSQ-informed-prior plumbing regression
# ===========================================================================


def test_fit_cmc_sharded_forwards_nlsq_uncertainties_to_workers():
    """fit_cmc_sharded must pass NLSQ point estimates AND uncertainties to workers.

    Without this, the sharded path silently falls back to default registry priors
    and the NLSQ posterior contraction is lost — defeating the purpose of warm-start.
    """
    from unittest.mock import MagicMock, patch

    from heterodyne.optimization.cmc import CMCConfig
    from heterodyne.optimization.cmc.core import fit_cmc_sharded

    n = 40
    c2 = np.ones((n, n), dtype=np.float64) * 0.5
    sigma_val = 0.1
    num_shards = 4

    captured: dict[str, object] = {}

    def capture_run_shards(
        shards,
        config,
        initial_values=None,
        parameter_space=None,
        prior_width_multiplier=1.0,
        nlsq_uncertainties=None,
        nlsq_prior_width_factor=2.0,
        progress_bar=True,
    ):
        captured["initial_values"] = initial_values
        captured["nlsq_uncertainties"] = nlsq_uncertainties
        captured["nlsq_prior_width_factor"] = nlsq_prior_width_factor
        # Return one minimal success dict per shard.
        return [
            {
                "success": True,
                "shard_idx": i,
                "samples": {"D0_ref": np.ones(10)},
                "param_names": ["D0_ref"],
                "n_chains": 1,
                "n_samples": 10,
                "extra_fields": {},
                "duration": 0.1,
                "stats": {"num_divergent": 0, "n_warmup": 10, "n_samples": 10},
            }
            for i in range(len(shards))
        ]

    mock_model = MagicMock()
    mock_model.t = np.linspace(0.001, 0.04, n)
    mock_model.q = 0.005
    mock_model.dt = 0.001
    mock_model.param_manager.space.varying_names = ["D0_ref"]
    mock_model.param_manager.varying_names = ["D0_ref"]
    mock_model.param_manager.space.priors = {}
    mock_model.scaling.get_for_angle.return_value = (1.0, 0.0)

    # Fake NLSQ result with uncertainties available
    mock_nlsq = MagicMock()
    mock_nlsq.success = True
    mock_nlsq.parameter_names = ["D0_ref"]
    mock_nlsq.get_param.return_value = 5e4
    mock_nlsq.get_uncertainty.return_value = 1.2e3

    fake_priors = {"D0_ref": MagicMock()}

    with (
        patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend."
            "MultiprocessingBackend.run_shards",
            side_effect=capture_run_shards,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_nlsq_informed_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_default_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.temper_priors",
            return_value=fake_priors,
        ),
    ):
        try:
            fit_cmc_sharded(
                model=mock_model,
                c2_data=c2,
                sigma=sigma_val,
                nlsq_result=mock_nlsq,
                num_shards=num_shards,
                sharding_strategy="contiguous",
                config=CMCConfig(
                    num_warmup=10,
                    num_samples=10,
                    use_nlsq_informed_priors=True,
                    nlsq_prior_width_factor=2.5,
                ),
            )
        except Exception:
            pass

    assert captured.get("initial_values") is not None, (
        "NLSQ point estimates not forwarded to workers"
    )
    assert captured.get("nlsq_uncertainties") is not None, (
        "NLSQ uncertainties not forwarded to workers — sharded path falls back to "
        "default priors and loses NLSQ contraction"
    )
    assert "D0_ref" in captured["nlsq_uncertainties"]  # type: ignore[operator]
    assert captured["nlsq_prior_width_factor"] == pytest.approx(2.5)


def test_fit_cmc_sharded_omits_nlsq_priors_when_config_disables_them():
    """When use_nlsq_informed_priors=False, NLSQ priors must NOT be forwarded
    even if an NLSQ result is available. Workers should fall back to defaults."""
    from unittest.mock import MagicMock, patch

    from heterodyne.optimization.cmc import CMCConfig
    from heterodyne.optimization.cmc.core import fit_cmc_sharded

    n = 40
    c2 = np.ones((n, n), dtype=np.float64) * 0.5

    captured: dict[str, object] = {}

    def capture_run_shards(
        shards,
        config,
        initial_values=None,
        parameter_space=None,
        prior_width_multiplier=1.0,
        nlsq_uncertainties=None,
        nlsq_prior_width_factor=2.0,
        progress_bar=True,
    ):
        captured["nlsq_uncertainties"] = nlsq_uncertainties
        return [
            {
                "success": True,
                "shard_idx": i,
                "samples": {"D0_ref": np.ones(10)},
                "param_names": ["D0_ref"],
                "n_chains": 1,
                "n_samples": 10,
                "extra_fields": {},
                "duration": 0.1,
                "stats": {"num_divergent": 0, "n_warmup": 10, "n_samples": 10},
            }
            for i in range(len(shards))
        ]

    mock_model = MagicMock()
    mock_model.t = np.linspace(0.001, 0.04, n)
    mock_model.q = 0.005
    mock_model.dt = 0.001
    mock_model.param_manager.space.varying_names = ["D0_ref"]
    mock_model.param_manager.varying_names = ["D0_ref"]
    mock_model.param_manager.space.priors = {}
    mock_model.scaling.get_for_angle.return_value = (1.0, 0.0)

    mock_nlsq = MagicMock()
    mock_nlsq.success = True
    mock_nlsq.parameter_names = ["D0_ref"]
    mock_nlsq.get_param.return_value = 5e4
    mock_nlsq.get_uncertainty.return_value = 1.2e3

    fake_priors = {"D0_ref": MagicMock()}

    with (
        patch(
            "heterodyne.optimization.cmc.backends.multiprocessing_backend."
            "MultiprocessingBackend.run_shards",
            side_effect=capture_run_shards,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_nlsq_informed_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.build_default_priors",
            return_value=fake_priors,
        ),
        patch(
            "heterodyne.optimization.cmc.core.temper_priors",
            return_value=fake_priors,
        ),
    ):
        try:
            fit_cmc_sharded(
                model=mock_model,
                c2_data=c2,
                sigma=0.1,
                nlsq_result=mock_nlsq,
                num_shards=4,
                sharding_strategy="contiguous",
                config=CMCConfig(
                    num_warmup=10,
                    num_samples=10,
                    use_nlsq_informed_priors=False,
                ),
            )
        except Exception:
            pass

    assert not captured.get("nlsq_uncertainties"), (
        "NLSQ uncertainties forwarded despite use_nlsq_informed_priors=False"
    )
