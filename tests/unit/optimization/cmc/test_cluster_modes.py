"""Tests for cluster_shard_modes, summarize_cross_shard_bimodality, and related dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.optimization.cmc.diagnostics import (
    BimodalConsensusResult,
    BimodalResult,
    ModeCluster,
    cluster_shard_modes,
    summarize_cross_shard_bimodality,
)

# ---------------------------------------------------------------------------
# ModeCluster dataclass
# ---------------------------------------------------------------------------


class TestModeCluster:
    def test_construction(self) -> None:
        mc = ModeCluster(
            mean={"a": 1.0, "b": 2.0},
            std={"a": 0.1, "b": 0.2},
            weight=0.6,
            n_shards=3,
        )
        assert mc.mean["a"] == 1.0
        assert mc.std["b"] == 0.2
        assert mc.weight == pytest.approx(0.6)
        assert mc.n_shards == 3

    def test_empty_mode(self) -> None:
        mc = ModeCluster(mean={}, std={}, weight=0.0, n_shards=0)
        assert mc.n_shards == 0
        assert mc.weight == 0.0


# ---------------------------------------------------------------------------
# BimodalConsensusResult dataclass
# ---------------------------------------------------------------------------


class TestBimodalConsensusResult:
    def test_construction(self) -> None:
        m0 = ModeCluster(mean={"x": 1.0}, std={"x": 0.1}, weight=0.5, n_shards=2)
        m1 = ModeCluster(mean={"x": 5.0}, std={"x": 0.2}, weight=0.5, n_shards=2)
        result = BimodalConsensusResult(
            modes=[m0, m1],
            modal_params=["x"],
            co_occurrence={"x": "both"},
        )
        assert len(result.modes) == 2
        assert result.modal_params == ["x"]
        assert "x" in result.co_occurrence

    def test_empty(self) -> None:
        result = BimodalConsensusResult(modes=[], modal_params=[], co_occurrence={})
        assert result.modes == []
        assert result.modal_params == []


# ---------------------------------------------------------------------------
# BimodalResult dataclass
# ---------------------------------------------------------------------------


class TestBimodalResult:
    def test_unimodal(self) -> None:
        r = BimodalResult(
            param_name="alpha",
            is_bimodal=False,
            bic_unimodal=100.0,
            bic_bimodal=105.0,
            delta_bic=-5.0,
            means=None,
            weights=None,
        )
        assert r.is_bimodal is False
        assert r.means is None

    def test_bimodal(self) -> None:
        r = BimodalResult(
            param_name="D0_ref",
            is_bimodal=True,
            bic_unimodal=200.0,
            bic_bimodal=150.0,
            delta_bic=50.0,
            means=(1.0, 5.0),
            weights=(0.4, 0.6),
        )
        assert r.is_bimodal is True
        assert r.means == (1.0, 5.0)
        assert r.weights[0] + r.weights[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Helpers for building test data
# ---------------------------------------------------------------------------


def _make_bimodal_result(
    param_name: str,
    *,
    bimodal: bool,
    means: tuple[float, float] | None = None,
    delta_bic: float = 20.0,
) -> BimodalResult:
    """Build a BimodalResult for testing."""
    return BimodalResult(
        param_name=param_name,
        is_bimodal=bimodal,
        bic_unimodal=100.0,
        bic_bimodal=100.0 - delta_bic if bimodal else 110.0,
        delta_bic=delta_bic if bimodal else -10.0,
        means=means if bimodal else None,
        weights=(0.5, 0.5) if bimodal else None,
    )


# ---------------------------------------------------------------------------
# cluster_shard_modes
# ---------------------------------------------------------------------------


class TestClusterShardModes:
    def test_two_clear_clusters(self) -> None:
        """Shards with clearly separated means should split into two clusters."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "D0_ref": [
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
            ],
        }
        # Shards 0,1 have low D0_ref; shards 2,3 have high D0_ref
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"D0_ref": np.array([1.0, 1.1, 0.9])},
            1: {"D0_ref": np.array([1.2, 0.8, 1.0])},
            2: {"D0_ref": np.array([10.0, 10.1, 9.9])},
            3: {"D0_ref": np.array([10.2, 9.8, 10.0])},
        }
        c0, c1 = cluster_shard_modes(bimodal_detections, shard_samples)
        # Cluster 0 should be the lower-mean cluster
        assert set(c0) == {0, 1}
        assert set(c1) == {2, 3}

    def test_no_modal_params(self) -> None:
        """When no parameter is bimodal, all shards go to cluster 0."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "D0_ref": [
                _make_bimodal_result("D0_ref", bimodal=False),
                _make_bimodal_result("D0_ref", bimodal=False),
            ],
        }
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"D0_ref": np.array([5.0, 5.1])},
            1: {"D0_ref": np.array([5.0, 4.9])},
        }
        c0, c1 = cluster_shard_modes(bimodal_detections, shard_samples)
        assert len(c0) == 2
        assert c1 == []

    def test_single_shard(self) -> None:
        """Single shard -> all in cluster 0."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "v0": [_make_bimodal_result("v0", bimodal=True, means=(1.0, 5.0))],
        }
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"v0": np.array([3.0, 3.1])},
        }
        c0, c1 = cluster_shard_modes(bimodal_detections, shard_samples)
        assert c0 == [0]
        assert c1 == []

    def test_empty_detections(self) -> None:
        """Empty detections -> all shards in cluster 0."""
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"x": np.array([1.0])},
            1: {"x": np.array([2.0])},
        }
        c0, c1 = cluster_shard_modes({}, shard_samples)
        assert set(c0) == {0, 1}
        assert c1 == []

    def test_with_param_bounds(self) -> None:
        """Custom param_bounds for normalization should not change cluster assignment."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "alpha": [
                _make_bimodal_result("alpha", bimodal=True, means=(0.1, 0.9)),
                _make_bimodal_result("alpha", bimodal=True, means=(0.1, 0.9)),
            ],
        }
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"alpha": np.array([0.1, 0.15])},
            1: {"alpha": np.array([0.9, 0.85])},
        }
        c0, c1 = cluster_shard_modes(
            bimodal_detections,
            shard_samples,
            param_bounds={"alpha": (0.0, 1.0)},
        )
        # Shard 0 lower, shard 1 upper
        assert c0 == [0]
        assert c1 == [1]

    def test_identical_shards_all_cluster_0(self) -> None:
        """All shards with identical means -> all in cluster 0."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [
                _make_bimodal_result("x", bimodal=True, means=(1.0, 5.0)),
                _make_bimodal_result("x", bimodal=True, means=(1.0, 5.0)),
            ],
        }
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"x": np.array([3.0, 3.0, 3.0])},
            1: {"x": np.array([3.0, 3.0, 3.0])},
        }
        c0, c1 = cluster_shard_modes(bimodal_detections, shard_samples)
        # Identical features -> idx_lo == idx_hi -> all in cluster 0
        assert len(c0) == 2
        assert c1 == []

    def test_missing_param_in_shard(self) -> None:
        """Shard missing a modal param should use 0.0 as fill."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "D0_ref": [
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
                _make_bimodal_result("D0_ref", bimodal=True, means=(1.0, 10.0)),
            ],
        }
        shard_samples: dict[int, dict[str, np.ndarray]] = {
            0: {"D0_ref": np.array([1.0])},
            1: {},  # missing D0_ref
        }
        # Should not raise
        c0, c1 = cluster_shard_modes(bimodal_detections, shard_samples)
        assert len(c0) + len(c1) == 2


# ---------------------------------------------------------------------------
# summarize_cross_shard_bimodality
# ---------------------------------------------------------------------------


class TestSummarizeCrossShardBimodality:
    def test_no_bimodal_detections(self) -> None:
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "D0_ref": [
                _make_bimodal_result("D0_ref", bimodal=False),
                _make_bimodal_result("D0_ref", bimodal=False),
            ],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=2)
        assert result["n_detections"] == 0
        pp = result["per_param"]["D0_ref"]
        assert pp["fraction_bimodal"] == 0.0
        assert pp["lower_mode_mean"] is None
        assert pp["upper_mode_mean"] is None
        assert pp["consensus_in_trough"] is False

    def test_all_bimodal(self) -> None:
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "v0": [
                _make_bimodal_result("v0", bimodal=True, means=(1.0, 5.0)),
                _make_bimodal_result("v0", bimodal=True, means=(1.5, 4.5)),
                _make_bimodal_result("v0", bimodal=True, means=(1.0, 5.0)),
            ],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=3)
        assert result["n_detections"] == 3
        pp = result["per_param"]["v0"]
        assert pp["fraction_bimodal"] == pytest.approx(1.0)
        # Lower mode mean ~ (1.0 + 1.5 + 1.0) / 3
        assert pp["lower_mode_mean"] == pytest.approx(
            np.mean([1.0, 1.5, 1.0]), rel=1e-6
        )
        # Upper mode mean ~ (5.0 + 4.5 + 5.0) / 3
        assert pp["upper_mode_mean"] == pytest.approx(
            np.mean([5.0, 4.5, 5.0]), rel=1e-6
        )
        assert pp["separation"] > 0

    def test_consensus_in_trough(self) -> None:
        """Consensus mean between modes should flag trough."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [
                _make_bimodal_result("x", bimodal=True, means=(0.0, 10.0)),
            ],
        }
        # Consensus at 5.0 (middle of gap 0..10)
        result = summarize_cross_shard_bimodality(
            bimodal_detections,
            n_shards=1,
            consensus_means={"x": 5.0},
        )
        pp = result["per_param"]["x"]
        assert pp["consensus_in_trough"] is True

    def test_consensus_not_in_trough(self) -> None:
        """Consensus near a mode should NOT flag trough."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [
                _make_bimodal_result("x", bimodal=True, means=(0.0, 10.0)),
            ],
        }
        # Consensus at 0.5 (near lower mode)
        result = summarize_cross_shard_bimodality(
            bimodal_detections,
            n_shards=1,
            consensus_means={"x": 0.5},
        )
        pp = result["per_param"]["x"]
        assert pp["consensus_in_trough"] is False

    def test_no_consensus_means(self) -> None:
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [_make_bimodal_result("x", bimodal=True, means=(1.0, 9.0))],
        }
        result = summarize_cross_shard_bimodality(
            bimodal_detections,
            n_shards=1,
            consensus_means=None,
        )
        pp = result["per_param"]["x"]
        assert pp["consensus_in_trough"] is False

    def test_mixed_bimodal_unimodal(self) -> None:
        """Some shards bimodal, some not."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "alpha": [
                _make_bimodal_result("alpha", bimodal=True, means=(0.5, 1.5)),
                _make_bimodal_result("alpha", bimodal=False),
                _make_bimodal_result("alpha", bimodal=True, means=(0.4, 1.6)),
            ],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=3)
        pp = result["per_param"]["alpha"]
        assert pp["fraction_bimodal"] == pytest.approx(2.0 / 3.0, rel=1e-6)
        assert result["n_detections"] == 2

    def test_significance_computation(self) -> None:
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [
                _make_bimodal_result("x", bimodal=True, means=(1.0, 5.0)),
            ],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=1)
        pp = result["per_param"]["x"]
        # With a single detection, std of lower/upper modes is 0 (or 1e-12 floor)
        # Separation = 4.0, pooled_std ~ 1e-12 -> significance >> 1
        assert pp["significance"] > 0

    def test_n_shards_in_output(self) -> None:
        result = summarize_cross_shard_bimodality({}, n_shards=7)
        assert result["n_shards"] == 7
        assert result["n_detections"] == 0

    def test_multiple_params(self) -> None:
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "a": [_make_bimodal_result("a", bimodal=True, means=(1.0, 9.0))],
            "b": [_make_bimodal_result("b", bimodal=False)],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=1)
        assert "a" in result["per_param"]
        assert "b" in result["per_param"]
        assert result["n_detections"] == 1

    def test_zero_shards(self) -> None:
        """n_shards=0 should not crash (edge case)."""
        bimodal_detections: dict[str, list[BimodalResult]] = {
            "x": [_make_bimodal_result("x", bimodal=True, means=(1.0, 5.0))],
        }
        result = summarize_cross_shard_bimodality(bimodal_detections, n_shards=0)
        # fraction_bimodal would be 1/0 -> guarded by n_shards > 0 check
        assert "per_param" in result
