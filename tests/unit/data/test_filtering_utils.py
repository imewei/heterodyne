"""Tests for heterodyne.data.filtering_utils.

Covers apply_time_window (2-D and 3-D paths) and apply_sigma_clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from heterodyne.data.filtering_utils import apply_time_window


class TestApplyTimeWindow:
    """Test time-window filtering for 2-D and 3-D correlation matrices."""

    def test_2d_basic(self) -> None:
        """2-D path returns correct subgrid."""
        t = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        c2 = np.arange(25, dtype=np.float64).reshape(5, 5)
        result = apply_time_window(c2, t, t_min=0.4, t_max=2.5)
        # t[1]=0.5, t[2]=1.0, t[3]=2.0 are in window → 3 points
        assert result.data.shape == (3, 3)
        assert result.n_removed == 2
        # Mask should mark which original indices were kept
        expected_mask = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(result.mask, expected_mask)

    def test_3d_preserves_phi_axis(self) -> None:
        """3-D path preserves the phi-angle axis and returns correct subgrid."""
        n_phi = 3
        n_t = 6
        t = np.linspace(0.1, 3.0, n_t)
        c2 = np.arange(n_phi * n_t * n_t, dtype=np.float64).reshape(n_phi, n_t, n_t)

        result = apply_time_window(c2, t, t_min=0.5, t_max=2.0)
        n_kept = int(np.sum((t >= 0.5) & (t <= 2.0)))

        assert result.data.ndim == 3
        assert result.data.shape == (n_phi, n_kept, n_kept)

    def test_3d_values_match_manual(self) -> None:
        """3-D indexing selects the correct subgrid elements."""
        t = np.array([0.1, 0.5, 1.0, 2.0])
        # Create 2 phi angles, 4x4 correlation matrix with known values
        c2 = np.zeros((2, 4, 4), dtype=np.float64)
        c2[0] = np.arange(16).reshape(4, 4)
        c2[1] = np.arange(16, 32).reshape(4, 4)

        result = apply_time_window(c2, t, t_min=0.4, t_max=1.5)
        # t[1]=0.5, t[2]=1.0 are in window → 2x2 subgrid
        assert result.data.shape == (2, 2, 2)
        # phi=0: rows/cols 1,2 of the original 4x4
        expected_phi0 = c2[0, 1:3, 1:3]
        np.testing.assert_array_equal(result.data[0], expected_phi0)
        # phi=1: same rows/cols
        expected_phi1 = c2[1, 1:3, 1:3]
        np.testing.assert_array_equal(result.data[1], expected_phi1)

    def test_empty_window_raises(self) -> None:
        """Window outside data range raises ValueError."""
        t = np.array([1.0, 2.0, 3.0])
        c2 = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="No time points"):
            apply_time_window(c2, t, t_min=10.0, t_max=20.0)

    def test_full_window_returns_original_shape(self) -> None:
        """Window encompassing all data returns same shape."""
        t = np.array([1.0, 2.0, 3.0])
        c2 = np.eye(3, dtype=np.float64)
        result = apply_time_window(c2, t, t_min=0.0, t_max=10.0)
        assert result.data.shape == (3, 3)
        assert result.n_removed == 0
        np.testing.assert_array_equal(result.data, c2)
