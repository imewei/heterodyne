"""Light integration tests for the CMC analysis pipeline.

These tests exercise the full data-flow through config creation,
prior construction, parameter-name resolution, and diagnostics
computation without performing any actual MCMC sampling.

They are intentionally cheap to run and require no external files.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: synthetic posterior samples
# ---------------------------------------------------------------------------


def _make_fake_posterior(
    param_names: list[str],
    n_samples: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate dict of fake posterior draws drawn from a Normal distribution.

    Means are taken from the DEFAULT_REGISTRY prior_mean (or default), and
    standard deviations are set to 10 % of the mean to give plausible spread.
    """
    from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

    rng = np.random.default_rng(seed)
    samples: dict[str, np.ndarray] = {}
    for name in param_names:
        info = DEFAULT_REGISTRY[name]
        center = float(info.prior_mean if info.prior_mean is not None else info.default)
        std = abs(center) * 0.10 if center != 0.0 else 0.5
        samples[name] = rng.normal(loc=center, scale=std, size=n_samples)
    return samples


# ---------------------------------------------------------------------------
# Test: CMCConfig creation
# ---------------------------------------------------------------------------


class TestCMCConfigCreation:
    """CMCConfig can be instantiated and exposes the expected attributes."""

    def test_cmc_config_creation(self) -> None:
        """Constructing CMCConfig with fast-test defaults succeeds."""
        from tests.factories.config_factory import make_cmc_config

        config = make_cmc_config()
        assert config.num_warmup == 50
        assert config.num_samples == 100
        assert config.num_chains == 2
        assert config.seed == 42

    def test_cmc_config_override(self) -> None:
        """Keyword overrides are applied correctly."""
        from tests.factories.config_factory import make_cmc_config

        config = make_cmc_config(num_chains=4, seed=99)
        assert config.num_chains == 4
        assert config.seed == 99

    def test_cmc_config_has_target_accept_prob(self) -> None:
        """target_accept_prob attribute exists and is in a valid range."""
        from tests.factories.config_factory import make_cmc_config

        config = make_cmc_config()
        assert 0.5 <= config.target_accept_prob <= 0.99

    def test_cmc_config_has_max_r_hat(self) -> None:
        """max_r_hat attribute exists and reflects the renamed field."""
        from tests.factories.config_factory import make_cmc_config

        config = make_cmc_config()
        assert hasattr(config, "max_r_hat")
        assert config.max_r_hat > 1.0  # always > 1 to allow some slack

    def test_cmc_config_has_nlsq_prior_width_factor(self) -> None:
        """nlsq_prior_width_factor attribute exists."""
        from tests.factories.config_factory import make_cmc_config

        config = make_cmc_config()
        assert hasattr(config, "nlsq_prior_width_factor")
        assert config.nlsq_prior_width_factor > 0.0


# ---------------------------------------------------------------------------
# Test: prior-to-init pipeline
# ---------------------------------------------------------------------------


class TestPriorToInitPipeline:
    """build_default_priors → get_param_names_in_order → build_init_values_dict."""

    def test_prior_to_init_pipeline(self) -> None:
        """The three-step prior→names→init pipeline produces a consistent result."""
        from heterodyne.config.parameter_space import ParameterSpace  # noqa: F401
        from heterodyne.optimization.cmc.priors import (
            build_default_priors,
            build_init_values_dict,
            get_param_names_in_order,
        )

        # Step 1: Build a ParameterSpace from an empty config (uses registry defaults)
        param_space = ParameterSpace.from_config({"parameters": {}})

        # Step 2: Build default priors
        priors = build_default_priors(param_space)
        assert len(priors) > 0

        # Step 3: Resolve ordered parameter names
        names = get_param_names_in_order()
        assert len(names) > 0

        # Step 4: Build initial values (no NLSQ — use fallback)
        init_values = build_init_values_dict()
        assert len(init_values) > 0

        # Consistency: every name in get_param_names_in_order that has a prior
        # should appear in init_values
        for name in names:
            if name in priors:
                assert name in init_values, f"'{name}' in priors but missing from init_values"

    def test_init_values_consistent_with_names(self) -> None:
        """init_values only contains names returned by get_param_names_in_order."""
        from heterodyne.optimization.cmc.priors import (
            build_init_values_dict,
            get_param_names_in_order,
        )

        names_set = set(get_param_names_in_order())
        init_values = build_init_values_dict()

        for name in init_values:
            assert name in names_set, f"Unexpected param in init_values: '{name}'"

    def test_init_values_with_nlsq_override(self) -> None:
        """NLSQ values for a subset of params are reflected in the init dict."""
        from heterodyne.optimization.cmc.priors import (
            build_init_values_dict,
            get_param_names_in_order,
        )

        names = get_param_names_in_order()
        # Override the first two varying parameters
        nlsq_override = {names[0]: 9999.0, names[1]: 0.123}

        # Clamp to valid bounds before testing
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        for n, v in list(nlsq_override.items()):
            info = DEFAULT_REGISTRY[n]
            nlsq_override[n] = float(max(info.min_bound, min(info.max_bound, v)))

        init_values = build_init_values_dict(nlsq_values=nlsq_override)
        for name, expected in nlsq_override.items():
            assert init_values[name] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test: diagnostics pipeline
# ---------------------------------------------------------------------------


class TestDiagnosticsPipeline:
    """compute_precision_analysis + compute_nlsq_comparison_metrics pipeline."""

    def test_diagnostics_pipeline(self) -> None:
        """Fake posterior samples flow through both diagnostic functions without error."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_nlsq_comparison_metrics,
            compute_precision_analysis,
        )
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        names = get_param_names_in_order()
        posterior = _make_fake_posterior(names, n_samples=300)

        # Precision analysis
        precision = compute_precision_analysis(posterior)
        assert len(precision) == len(names)
        for name, metrics in precision.items():
            assert "mean" in metrics
            assert "std" in metrics
            assert "cv" in metrics
            assert "hdi_width" in metrics

        # NLSQ comparison: use the posterior means as fake NLSQ estimates
        nlsq_estimates = {name: precision[name]["mean"] for name in precision}
        comparison = compute_nlsq_comparison_metrics(posterior, nlsq_estimates)
        assert len(comparison) == len(names)
        for name, metrics in comparison.items():
            # z_score should be near zero because nlsq_value == posterior_mean
            assert metrics["z_score"] == pytest.approx(0.0, abs=0.5)
            # nlsq_value at the mean is inside the HDI
            assert metrics["within_hdi"] == pytest.approx(1.0)

    def test_diagnostics_pipeline_partial_nlsq(self) -> None:
        """NLSQ values for a subset of parameters produce a partial comparison."""
        from heterodyne.optimization.cmc.diagnostics import (
            compute_nlsq_comparison_metrics,
            compute_precision_analysis,
        )
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        names = get_param_names_in_order()
        posterior = _make_fake_posterior(names, n_samples=200)
        precision = compute_precision_analysis(posterior)

        # Supply NLSQ estimates for only the first two parameters
        nlsq_partial = {names[0]: precision[names[0]]["mean"]}
        comparison = compute_nlsq_comparison_metrics(posterior, nlsq_partial)
        assert set(comparison.keys()) == {names[0]}

    def test_precision_all_hdi_widths_positive(self) -> None:
        """All HDI widths are positive for random Gaussian samples."""
        from heterodyne.optimization.cmc.diagnostics import compute_precision_analysis
        from heterodyne.optimization.cmc.priors import get_param_names_in_order

        names = get_param_names_in_order()
        posterior = _make_fake_posterior(names, n_samples=500)
        precision = compute_precision_analysis(posterior)
        for name, metrics in precision.items():
            assert metrics["hdi_width"] > 0.0, f"{name}: hdi_width is not positive"
