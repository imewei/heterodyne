"""Tests for CMC diagnostics additions in diagnostics.py.

Covers:
- :class:`BimodalResult` construction
- :func:`detect_bimodal` (requires scikit-learn)
- :func:`compute_nlsq_comparison_metrics`
- :func:`compute_precision_analysis`
- :func:`check_shard_bimodality` (requires scikit-learn)
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.cmc.diagnostics import (
    BimodalResult,
    compute_nlsq_comparison_metrics,
    compute_precision_analysis,
)

# ---------------------------------------------------------------------------
# BimodalResult construction
# ---------------------------------------------------------------------------


class TestBimodalResultCreation:
    """BimodalResult stores fields correctly."""

    def _unimodal(self) -> BimodalResult:
        return BimodalResult(
            param_name="D0_ref",
            is_bimodal=False,
            bic_unimodal=120.5,
            bic_bimodal=130.0,
            delta_bic=-9.5,
            means=None,
            weights=None,
        )

    def _bimodal(self) -> BimodalResult:
        return BimodalResult(
            param_name="v0",
            is_bimodal=True,
            bic_unimodal=200.0,
            bic_bimodal=150.0,
            delta_bic=50.0,
            means=(1e3, 5e3),
            weights=(0.6, 0.4),
        )

    def test_bimodal_result_unimodal_fields(self) -> None:
        """Unimodal result stores correct field values."""
        r = self._unimodal()
        assert r.param_name == "D0_ref"
        assert r.is_bimodal is False
        assert r.means is None
        assert r.weights is None
        assert pytest.approx(r.delta_bic) == -9.5

    def test_bimodal_result_bimodal_fields(self) -> None:
        """Bimodal result stores means and weights tuples."""
        r = self._bimodal()
        assert r.param_name == "v0"
        assert r.is_bimodal is True
        assert r.means is not None
        assert r.weights is not None
        assert len(r.means) == 2
        assert len(r.weights) == 2

    def test_bimodal_result_bic_values(self) -> None:
        """delta_bic equals bic_unimodal minus bic_bimodal."""
        r = self._bimodal()
        assert pytest.approx(r.delta_bic) == r.bic_unimodal - r.bic_bimodal


# ---------------------------------------------------------------------------
# detect_bimodal — sklearn-dependent
# ---------------------------------------------------------------------------


class TestDetectBimodal:
    """detect_bimodal identifies unimodal and bimodal distributions."""

    sklearn = pytest.importorskip("sklearn")

    def test_detect_bimodal_unimodal(self) -> None:
        """Samples from a single Normal distribution are detected as unimodal."""
        from heterodyne.optimization.cmc.diagnostics import detect_bimodal

        rng = np.random.default_rng(0)
        samples = rng.normal(loc=1e4, scale=500.0, size=500)
        result = detect_bimodal(samples, param_name="D0_ref")

        assert result.param_name == "D0_ref"
        assert result.is_bimodal is False

    def test_detect_bimodal_bimodal(self) -> None:
        """Samples from two well-separated Normals are detected as bimodal."""
        from heterodyne.optimization.cmc.diagnostics import detect_bimodal

        rng = np.random.default_rng(1)
        mode1 = rng.normal(loc=0.0, scale=0.1, size=500)
        mode2 = rng.normal(loc=10.0, scale=0.1, size=500)
        samples = np.concatenate([mode1, mode2])

        result = detect_bimodal(samples, param_name="v0")

        assert result.is_bimodal is True
        assert result.means is not None
        assert result.weights is not None
        # Means should be near 0 and 10
        sorted_means = sorted(result.means)
        assert sorted_means[0] == pytest.approx(0.0, abs=1.0)
        assert sorted_means[1] == pytest.approx(10.0, abs=1.0)

    def test_detect_bimodal_returns_bimodal_result(self) -> None:
        """Return type is always BimodalResult."""
        from heterodyne.optimization.cmc.diagnostics import detect_bimodal

        rng = np.random.default_rng(2)
        samples = rng.normal(size=200)
        result = detect_bimodal(samples, param_name="test")
        assert isinstance(result, BimodalResult)


# ---------------------------------------------------------------------------
# compute_nlsq_comparison_metrics
# ---------------------------------------------------------------------------


class TestComputeNlsqComparisonMetrics:
    """compute_nlsq_comparison_metrics produces the expected keys and values."""

    def _make_samples(
        self, mean: float = 1e4, std: float = 500.0, n: int = 400
    ) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(loc=mean, scale=std, size=n)

    def test_compute_nlsq_comparison_metrics_keys(self) -> None:
        """Result contains the five expected metrics for each matched parameter."""
        posterior = {"D0_ref": self._make_samples()}
        nlsq = {"D0_ref": 1e4}
        result = compute_nlsq_comparison_metrics(posterior, nlsq)

        assert "D0_ref" in result
        metrics = result["D0_ref"]
        for key in (
            "posterior_mean",
            "posterior_std",
            "nlsq_value",
            "z_score",
            "within_hdi",
        ):
            assert key in metrics, f"Missing metric key: {key}"

    def test_compute_nlsq_comparison_metrics_z_score(self) -> None:
        """z_score is near zero when nlsq_value equals the posterior mean."""
        mean = 1e4
        posterior = {"D0_ref": self._make_samples(mean=mean, std=500.0, n=10_000)}
        nlsq = {"D0_ref": mean}
        result = compute_nlsq_comparison_metrics(posterior, nlsq)
        assert result["D0_ref"]["z_score"] == pytest.approx(0.0, abs=0.5)

    def test_compute_nlsq_comparison_metrics_within_hdi(self) -> None:
        """nlsq_value equal to the posterior mean is inside the 95 % HDI."""
        mean = 5000.0
        posterior = {"D0_ref": self._make_samples(mean=mean, std=200.0, n=2000)}
        nlsq = {"D0_ref": mean}
        result = compute_nlsq_comparison_metrics(posterior, nlsq)
        assert result["D0_ref"]["within_hdi"] == pytest.approx(1.0)

    def test_compute_nlsq_comparison_metrics_outside_hdi(self) -> None:
        """nlsq_value far from the posterior mean is outside the 95 % HDI."""
        posterior = {"D0_ref": self._make_samples(mean=1e4, std=100.0, n=2000)}
        # nlsq value is 100 sigma away
        nlsq = {"D0_ref": 1e4 + 100 * 100.0}
        result = compute_nlsq_comparison_metrics(posterior, nlsq)
        assert result["D0_ref"]["within_hdi"] == pytest.approx(0.0)

    def test_compute_nlsq_comparison_metrics_unmatched_param_excluded(self) -> None:
        """Parameters present in only one input do not appear in the output."""
        posterior = {"D0_ref": self._make_samples()}
        nlsq = {"D0_sample": 1e4}  # no overlap
        result = compute_nlsq_comparison_metrics(posterior, nlsq)
        assert result == {}

    def test_compute_nlsq_comparison_metrics_multiple_params(self) -> None:
        """Multiple parameters in both inputs all produce metric entries."""
        rng = np.random.default_rng(0)
        posterior = {
            "D0_ref": rng.normal(1e4, 500, 300),
            "alpha_ref": rng.normal(0.0, 0.5, 300),
        }
        nlsq = {"D0_ref": 1e4, "alpha_ref": 0.0}
        result = compute_nlsq_comparison_metrics(posterior, nlsq)
        assert "D0_ref" in result
        assert "alpha_ref" in result


# ---------------------------------------------------------------------------
# compute_precision_analysis
# ---------------------------------------------------------------------------


class TestComputePrecisionAnalysis:
    """compute_precision_analysis returns the four expected keys per parameter."""

    def _samples(self, n: int = 500) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(7)
        return {
            "D0_ref": rng.normal(1e4, 300.0, n),
            "alpha_ref": rng.normal(0.0, 0.3, n),
        }

    def test_compute_precision_analysis_keys_present(self) -> None:
        """Each parameter entry contains mean, std, cv, and hdi_width."""
        result = compute_precision_analysis(self._samples())
        for param, metrics in result.items():
            for key in ("mean", "std", "cv", "hdi_width"):
                assert key in metrics, f"{param}: missing key '{key}'"

    def test_compute_precision_analysis_mean_reasonable(self) -> None:
        """Posterior mean matches the generating mean approximately."""
        result = compute_precision_analysis(self._samples(n=10_000))
        assert result["D0_ref"]["mean"] == pytest.approx(1e4, rel=0.05)
        assert result["alpha_ref"]["mean"] == pytest.approx(0.0, abs=0.05)

    def test_compute_precision_analysis_hdi_width_positive(self) -> None:
        """HDI width is strictly positive for non-degenerate samples."""
        result = compute_precision_analysis(self._samples())
        for metrics in result.values():
            assert metrics["hdi_width"] > 0.0

    def test_compute_precision_analysis_cv_positive(self) -> None:
        """Coefficient of variation is non-negative for positive-mean parameters."""
        result = compute_precision_analysis(
            {"D0_ref": np.abs(np.random.default_rng(0).normal(1e4, 200, 500))}
        )
        assert result["D0_ref"]["cv"] >= 0.0

    def test_compute_precision_analysis_empty_array_skipped(self) -> None:
        """A parameter with an empty sample array is omitted from the output."""
        result = compute_precision_analysis({"D0_ref": np.array([])})
        assert "D0_ref" not in result

    def test_compute_precision_analysis_all_same_samples(self) -> None:
        """Constant samples produce std=0 and hdi_width=0."""
        result = compute_precision_analysis({"D0_ref": np.full(100, 1e4)})
        assert result["D0_ref"]["std"] == pytest.approx(0.0)
        assert result["D0_ref"]["hdi_width"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_shard_bimodality — sklearn-dependent
# ---------------------------------------------------------------------------


class TestCheckShardBimodality:
    """check_shard_bimodality processes multi-shard sample dicts correctly."""

    sklearn = pytest.importorskip("sklearn")

    def test_check_shard_bimodality_structure(self) -> None:
        """Output is keyed by parameter name; each value is a list of BimodalResult."""
        from heterodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(10)
        shard_samples = {
            0: {"D0_ref": rng.normal(1e4, 300, 200)},
            1: {"D0_ref": rng.normal(1e4, 300, 200)},
        }
        result = check_shard_bimodality(shard_samples)

        assert "D0_ref" in result
        assert isinstance(result["D0_ref"], list)
        assert len(result["D0_ref"]) == 2
        for item in result["D0_ref"]:
            assert isinstance(item, BimodalResult)

    def test_check_shard_bimodality_unimodal_shards(self) -> None:
        """Unimodal shards produce BimodalResult with is_bimodal=False."""
        from heterodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(11)
        shard_samples = {
            0: {"alpha_ref": rng.normal(0.0, 0.2, 300)},
            1: {"alpha_ref": rng.normal(0.0, 0.2, 300)},
        }
        result = check_shard_bimodality(shard_samples)
        for br in result["alpha_ref"]:
            assert br.is_bimodal is False

    def test_check_shard_bimodality_missing_param_skipped(self) -> None:
        """A parameter absent from a shard does not raise; that shard is skipped."""
        from heterodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(12)
        shard_samples = {
            0: {"D0_ref": rng.normal(1e4, 300, 200)},
            1: {},  # no D0_ref in shard 1
        }
        result = check_shard_bimodality(shard_samples)
        # D0_ref appears only in shard 0 → exactly one BimodalResult
        assert len(result["D0_ref"]) == 1
