"""Tests for the per-angle quantile β,o estimator used by constant mode."""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.core.quantile_scaling import compute_per_angle_quantile_scaling


@pytest.mark.unit
class TestPerAngleQuantileScaling:
    def test_returns_per_angle_vectors_of_correct_shape(self) -> None:
        """Output vectors have shape (n_phi,), not scalars."""
        n_phi = 5
        n_pts_per_phi = 100
        # Synthetic g2: each angle has a clear baseline + contrast band.
        rng = np.random.default_rng(seed=0)
        phi_indices = np.repeat(np.arange(n_phi, dtype=np.int32), n_pts_per_phi)
        # Per-angle offset = 1.0 + 0.05·i ; contrast = 0.3 + 0.02·i
        offsets = 1.0 + 0.05 * np.arange(n_phi)
        contrasts = 0.3 + 0.02 * np.arange(n_phi)
        g2 = np.empty(n_phi * n_pts_per_phi, dtype=np.float64)
        for i in range(n_phi):
            sl = slice(i * n_pts_per_phi, (i + 1) * n_pts_per_phi)
            g2[sl] = offsets[i] + contrasts[i] * rng.uniform(0, 1, n_pts_per_phi)

        result = compute_per_angle_quantile_scaling(
            g2=g2,
            phi_indices=phi_indices,
            n_phi=n_phi,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        assert result.contrast_per_angle.shape == (n_phi,)
        assert result.offset_per_angle.shape == (n_phi,)
        # Quantile (q05..q95) covers the inner 90 % of a uniform band, so
        # the contrast estimate is biased low by ~10 %; tolerate that
        # systematic plus a small finite-n RNG term at n=100.
        np.testing.assert_allclose(result.contrast_per_angle, contrasts, atol=0.07)
        np.testing.assert_allclose(result.offset_per_angle, offsets, atol=0.05)

    def test_empty_angle_uses_default_within_bounds(self) -> None:
        """Angles with no data fall back to the midpoint of the bounds."""
        result = compute_per_angle_quantile_scaling(
            g2=np.array([1.0, 1.0]),
            phi_indices=np.array([0, 0], dtype=np.int32),
            n_phi=3,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )
        # Angle 0 has data; angles 1 and 2 have none.
        assert result.contrast_per_angle.shape == (3,)
        assert result.offset_per_angle.shape == (3,)
        # Empty-angle defaults: contrast=0.5 (midpoint), offset=1.0 (midpoint)
        assert result.contrast_per_angle[1] == pytest.approx(0.5)
        assert result.contrast_per_angle[2] == pytest.approx(0.5)
        assert result.offset_per_angle[1] == pytest.approx(1.0)
        assert result.offset_per_angle[2] == pytest.approx(1.0)

    def test_clips_outliers_to_bounds(self) -> None:
        """Quantile estimates outside the bounds are clipped, not raised."""
        # All g2 values are 10.0 → quantile says offset=10.0, contrast=0.0,
        # both outside the typical bounds.
        result = compute_per_angle_quantile_scaling(
            g2=np.full(50, 10.0),
            phi_indices=np.zeros(50, dtype=np.int32),
            n_phi=1,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )
        assert 0.0 <= result.contrast_per_angle[0] <= 1.0
        assert 0.5 <= result.offset_per_angle[0] <= 1.5
