"""Tests for angle-specific filtering utilities.

Covers filter_by_angle_range, select_angles, find_nearest_angle,
compute_angle_quality, and estimate_per_angle_scaling — all using
purely synthetic in-memory data.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.data.angle_filtering import (
    compute_angle_quality,
    filter_by_angle_range,
    find_nearest_angle,
    select_angles,
)
from heterodyne.data.types import AngleRange
from heterodyne.optimization.cmc.priors import estimate_per_angle_scaling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_c2_3d() -> np.ndarray:
    """3D correlation array: 4 angles, 5x5 time matrices."""
    rng = np.random.default_rng(0)
    return rng.random((4, 5, 5)).astype(np.float64)


@pytest.fixture()
def phi_4() -> np.ndarray:
    return np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# filter_by_angle_range
# ---------------------------------------------------------------------------


class TestFilterByAngleRange:
    def test_selects_angles_within_range(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        filtered_c2, filtered_phi = filter_by_angle_range(
            small_c2_3d, phi_4, AngleRange(0.0, 90.0)
        )
        assert filtered_c2.shape == (3, 5, 5)
        np.testing.assert_array_equal(filtered_phi, [0.0, 45.0, 90.0])

    def test_single_angle_range(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        filtered_c2, filtered_phi = filter_by_angle_range(
            small_c2_3d, phi_4, AngleRange(45.0, 45.0)
        )
        assert filtered_c2.shape == (1, 5, 5)
        assert filtered_phi[0] == pytest.approx(45.0)

    def test_all_angles_in_range(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        filtered_c2, filtered_phi = filter_by_angle_range(
            small_c2_3d, phi_4, AngleRange(-1.0, 200.0)
        )
        assert filtered_c2.shape == (4, 5, 5)

    def test_raises_when_no_angles_match(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="No angles"):
            filter_by_angle_range(small_c2_3d, phi_4, AngleRange(200.0, 300.0))

    def test_raises_on_non_3d_input(self, phi_4: np.ndarray) -> None:
        c2_2d = np.zeros((4, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="3D"):
            filter_by_angle_range(c2_2d, phi_4, AngleRange(0.0, 90.0))

    def test_raises_on_length_mismatch(
        self, small_c2_3d: np.ndarray
    ) -> None:
        phi_wrong = np.array([0.0, 45.0], dtype=np.float64)
        with pytest.raises(ValueError, match="phi_angles length"):
            filter_by_angle_range(small_c2_3d, phi_wrong, AngleRange(0.0, 90.0))

    def test_raises_when_phi_min_exceeds_phi_max(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="phi_min"):
            filter_by_angle_range(small_c2_3d, phi_4, AngleRange(90.0, 0.0))

    def test_output_data_matches_input_slices(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        filtered_c2, _ = filter_by_angle_range(
            small_c2_3d, phi_4, AngleRange(45.0, 135.0)
        )
        np.testing.assert_array_equal(filtered_c2[0], small_c2_3d[1])
        np.testing.assert_array_equal(filtered_c2[1], small_c2_3d[2])
        np.testing.assert_array_equal(filtered_c2[2], small_c2_3d[3])


# ---------------------------------------------------------------------------
# select_angles
# ---------------------------------------------------------------------------


class TestSelectAngles:
    def test_select_subset(self, phi_4: np.ndarray) -> None:
        result = select_angles(phi_4, [0, 2])
        np.testing.assert_array_equal(result, [0.0, 90.0])

    def test_select_all(self, phi_4: np.ndarray) -> None:
        result = select_angles(phi_4, list(range(4)))
        np.testing.assert_array_equal(result, phi_4)

    def test_select_single(self, phi_4: np.ndarray) -> None:
        result = select_angles(phi_4, [3])
        np.testing.assert_array_equal(result, [135.0])

    def test_select_empty_returns_empty(self, phi_4: np.ndarray) -> None:
        result = select_angles(phi_4, [])
        assert result.size == 0

    def test_raises_on_out_of_bounds_index(self, phi_4: np.ndarray) -> None:
        with pytest.raises(IndexError, match="out of bounds"):
            select_angles(phi_4, [10])

    def test_raises_on_negative_index(self, phi_4: np.ndarray) -> None:
        with pytest.raises(IndexError, match="out of bounds"):
            select_angles(phi_4, [-1])


# ---------------------------------------------------------------------------
# find_nearest_angle
# ---------------------------------------------------------------------------


class TestFindNearestAngle:
    def test_exact_match(self, phi_4: np.ndarray) -> None:
        idx = find_nearest_angle(phi_4, 90.0)
        assert idx == 2

    def test_nearest_when_not_exact(self, phi_4: np.ndarray) -> None:
        # 50 degrees is closer to 45 than to 90
        idx = find_nearest_angle(phi_4, 50.0)
        assert idx == 1

    def test_first_element(self, phi_4: np.ndarray) -> None:
        idx = find_nearest_angle(phi_4, -5.0)
        assert idx == 0

    def test_last_element(self, phi_4: np.ndarray) -> None:
        idx = find_nearest_angle(phi_4, 200.0)
        assert idx == 3

    def test_raises_on_empty_array(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            find_nearest_angle(np.array([]), 45.0)

    def test_returns_int(self, phi_4: np.ndarray) -> None:
        idx = find_nearest_angle(phi_4, 0.0)
        assert isinstance(idx, int)


# ---------------------------------------------------------------------------
# compute_angle_quality
# ---------------------------------------------------------------------------


class TestComputeAngleQuality:
    def test_returns_expected_keys(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        result = compute_angle_quality(small_c2_3d, phi_4)
        assert set(result.keys()) == {"phi_angles", "snr", "mean", "std"}

    def test_output_shapes(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        result = compute_angle_quality(small_c2_3d, phi_4)
        assert result["snr"].shape == (4,)
        assert result["mean"].shape == (4,)
        assert result["std"].shape == (4,)

    def test_phi_angles_preserved(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        result = compute_angle_quality(small_c2_3d, phi_4)
        np.testing.assert_array_equal(result["phi_angles"], phi_4)

    def test_snr_non_negative(
        self, small_c2_3d: np.ndarray, phi_4: np.ndarray
    ) -> None:
        result = compute_angle_quality(small_c2_3d, phi_4)
        assert np.all(result["snr"] >= 0.0)

    def test_all_nan_slice_gives_zero_snr(self) -> None:
        # Create a 1-angle 3D array with all-NaN off-diagonal elements
        c2 = np.full((1, 4, 4), np.nan)
        # Put finite values on diagonal to avoid empty off-diagonal case
        for i in range(4):
            c2[0, i, i] = 1.0
        phi = np.array([0.0])
        result = compute_angle_quality(c2, phi)
        assert result["snr"][0] == pytest.approx(0.0)

    def test_constant_slice_has_zero_std(self) -> None:
        # Off-diagonal all equal to 2.0 → std=0, snr=0
        c2 = np.full((1, 4, 4), 2.0)
        phi = np.array([0.0])
        result = compute_angle_quality(c2, phi)
        assert result["std"][0] == pytest.approx(0.0)
        assert result["mean"][0] == pytest.approx(2.0)

    def test_raises_on_non_3d_input(self, phi_4: np.ndarray) -> None:
        with pytest.raises(ValueError, match="3D"):
            compute_angle_quality(np.zeros((4, 5)), phi_4)

    def test_raises_on_phi_length_mismatch(
        self, small_c2_3d: np.ndarray
    ) -> None:
        phi_wrong = np.array([0.0, 45.0], dtype=np.float64)
        with pytest.raises(ValueError, match="phi_angles length"):
            compute_angle_quality(small_c2_3d, phi_wrong)


# ---------------------------------------------------------------------------
# estimate_per_angle_scaling
# ---------------------------------------------------------------------------


class TestEstimatePerAngleScaling:
    def _make_g2(self, n: int = 20, value: float = 1.0) -> np.ndarray:
        """Monotonically decaying g2-like array from value to 1."""
        return np.linspace(value, 1.0, n)

    def test_returns_dict_with_all_keys(self) -> None:
        data = {"phi_0": self._make_g2(20, value=1.5)}
        result = estimate_per_angle_scaling(data)
        assert "phi_0" in result
        contrast, offset = result["phi_0"]
        assert isinstance(contrast, float)
        assert isinstance(offset, float)

    def test_contrast_estimate_is_range(self) -> None:
        g2 = np.array([1.5, 1.3, 1.1, 1.0], dtype=np.float64)
        data = {"a": g2}
        result = estimate_per_angle_scaling(data)
        contrast, _ = result["a"]
        assert contrast == pytest.approx(0.5, abs=1e-10)

    def test_offset_clipped_to_unit_interval(self) -> None:
        # g2 values all above 1 — baseline mean > 1 should be clipped
        g2 = np.linspace(5.0, 3.0, 10)
        data = {"b": g2}
        result = estimate_per_angle_scaling(data)
        _, offset = result["b"]
        assert 0.0 <= offset <= 1.0

    def test_dict_value_with_g2_key(self) -> None:
        g2 = np.linspace(1.2, 1.0, 15)
        data = {"phi_30": {"g2": g2, "extra": "ignored"}}
        result = estimate_per_angle_scaling(data)
        assert "phi_30" in result

    def test_missing_g2_key_in_dict_skips_angle(self) -> None:
        data = {"phi_30": {"no_g2_here": np.ones(10)}}
        result = estimate_per_angle_scaling(data)
        assert "phi_30" not in result

    def test_angle_keys_subset(self) -> None:
        data = {
            "phi_0": self._make_g2(10, 1.4),
            "phi_45": self._make_g2(10, 1.2),
        }
        result = estimate_per_angle_scaling(data, angle_keys=["phi_0"])
        assert "phi_0" in result
        assert "phi_45" not in result

    def test_empty_array_skipped(self) -> None:
        data = {"phi_0": np.array([])}
        result = estimate_per_angle_scaling(data)
        assert "phi_0" not in result

    def test_none_value_skipped(self) -> None:
        data = {"phi_0": None}
        result = estimate_per_angle_scaling(data)
        assert "phi_0" not in result

    def test_multiple_angles_all_estimated(self) -> None:
        data = {
            "phi_0": self._make_g2(20, 1.5),
            "phi_45": self._make_g2(20, 1.3),
            "phi_90": self._make_g2(20, 1.1),
        }
        result = estimate_per_angle_scaling(data)
        assert len(result) == 3
        for key in data:
            assert key in result
