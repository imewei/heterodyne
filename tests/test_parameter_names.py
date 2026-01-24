"""Tests for config/parameter_names.py module.

Covers get_param_index and get_group_indices error paths for improved coverage.
"""

from __future__ import annotations

import pytest

from heterodyne.config.parameter_names import (
    ALL_PARAM_NAMES,
    PARAM_GROUPS,
    PARAM_INDICES,
    get_group_indices,
    get_param_index,
)


# ============================================================================
# Test Module Constants
# ============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_all_param_names_count(self) -> None:
        """ALL_PARAM_NAMES contains exactly 14 parameters."""
        assert len(ALL_PARAM_NAMES) == 14

    def test_param_indices_match_names(self) -> None:
        """PARAM_INDICES maps each name to correct index."""
        for i, name in enumerate(ALL_PARAM_NAMES):
            assert PARAM_INDICES[name] == i

    def test_param_groups_coverage(self) -> None:
        """All parameters belong to exactly one group."""
        all_in_groups = []
        for group_names in PARAM_GROUPS.values():
            all_in_groups.extend(group_names)
        assert sorted(all_in_groups) == sorted(ALL_PARAM_NAMES)

    def test_param_groups_keys(self) -> None:
        """PARAM_GROUPS has expected keys."""
        expected_groups = {"reference", "sample", "velocity", "fraction", "angle"}
        assert set(PARAM_GROUPS.keys()) == expected_groups


# ============================================================================
# Test get_param_index
# ============================================================================


class TestGetParamIndex:
    """Tests for get_param_index function."""

    @pytest.mark.parametrize(
        "name,expected_idx",
        [
            ("D0_ref", 0),
            ("alpha_ref", 1),
            ("D_offset_ref", 2),
            ("D0_sample", 3),
            ("alpha_sample", 4),
            ("D_offset_sample", 5),
            ("v0", 6),
            ("beta", 7),
            ("v_offset", 8),
            ("f0", 9),
            ("f1", 10),
            ("f2", 11),
            ("f3", 12),
            ("phi0", 13),
        ],
    )
    def test_get_param_index_valid_names(self, name: str, expected_idx: int) -> None:
        """get_param_index returns correct index for valid names."""
        assert get_param_index(name) == expected_idx

    def test_get_param_index_invalid_name_raises(self) -> None:
        """get_param_index raises KeyError for invalid parameter names."""
        with pytest.raises(KeyError, match="Unknown parameter 'invalid_param'"):
            get_param_index("invalid_param")

    def test_get_param_index_error_lists_valid_names(self) -> None:
        """get_param_index error message includes valid parameter names."""
        with pytest.raises(KeyError) as exc_info:
            get_param_index("not_a_param")

        error_msg = str(exc_info.value)
        # Check that some valid names are mentioned
        assert "D0_ref" in error_msg or "Valid names" in error_msg

    def test_get_param_index_case_sensitive(self) -> None:
        """get_param_index is case-sensitive."""
        with pytest.raises(KeyError):
            get_param_index("d0_ref")  # lowercase 'd'


# ============================================================================
# Test get_group_indices
# ============================================================================


class TestGetGroupIndices:
    """Tests for get_group_indices function."""

    def test_get_group_indices_reference(self) -> None:
        """get_group_indices returns correct indices for reference group."""
        indices = get_group_indices("reference")
        assert indices == (0, 1, 2)

    def test_get_group_indices_sample(self) -> None:
        """get_group_indices returns correct indices for sample group."""
        indices = get_group_indices("sample")
        assert indices == (3, 4, 5)

    def test_get_group_indices_velocity(self) -> None:
        """get_group_indices returns correct indices for velocity group."""
        indices = get_group_indices("velocity")
        assert indices == (6, 7, 8)

    def test_get_group_indices_fraction(self) -> None:
        """get_group_indices returns correct indices for fraction group."""
        indices = get_group_indices("fraction")
        assert indices == (9, 10, 11, 12)

    def test_get_group_indices_angle(self) -> None:
        """get_group_indices returns correct indices for angle group."""
        indices = get_group_indices("angle")
        assert indices == (13,)

    def test_get_group_indices_invalid_group_raises(self) -> None:
        """get_group_indices raises KeyError for invalid group names."""
        with pytest.raises(KeyError, match="Unknown group 'invalid_group'"):
            get_group_indices("invalid_group")

    def test_get_group_indices_error_lists_valid_groups(self) -> None:
        """get_group_indices error message includes valid group names."""
        with pytest.raises(KeyError) as exc_info:
            get_group_indices("not_a_group")

        error_msg = str(exc_info.value)
        assert "Valid groups" in error_msg

    def test_get_group_indices_returns_tuple(self) -> None:
        """get_group_indices returns a tuple."""
        indices = get_group_indices("reference")
        assert isinstance(indices, tuple)
