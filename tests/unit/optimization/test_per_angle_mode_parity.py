"""Homodyne parity tests for heterodyne NLSQ per-angle-mode dispatch.

Reference: https://homodyne.readthedocs.io/en/latest/theory/anti_degeneracy.html

Target mode table (heterodyne has 14 physics params; homodyne has 7):

  - constant:   14 physics, β(φ) and o(φ) frozen per-angle from quantile
  - auto:       14 + 2 averaged scaling = 16 params (when n_phi >= threshold)
  - fourier K=2: 14 + 2·(2K+1) = 24 params (K=2 -> 10 Fourier coefficients)
  - individual: 14 + 2·n_phi params
"""

from __future__ import annotations

import typing

import pytest

from heterodyne.optimization.nlsq.config import NLSQConfig


@pytest.mark.unit
class TestModeTaxonomy:
    """The four homodyne mode names are accepted by the public Literal."""

    def test_individual_mode_accepted(self) -> None:
        """`individual` is the canonical name (matches homodyne docs)."""
        cfg = NLSQConfig(per_angle_mode="individual")
        assert cfg.per_angle_mode == "individual"

    def test_constant_mode_accepted(self) -> None:
        """`constant` is a valid public configuration value."""
        cfg = NLSQConfig(per_angle_mode="constant")
        assert cfg.per_angle_mode == "constant"

    def test_literal_lists_exactly_four_modes(self) -> None:
        """`get_args` returns the four canonical homodyne mode names.

        The Literal may additionally include `"independent"` as a deprecation
        alias; we assert the canonical four are present and `"independent"`
        is the only allowed extra.
        """
        field_type = typing.get_type_hints(NLSQConfig)["per_angle_mode"]
        args = set(typing.get_args(field_type))
        canonical = {"individual", "constant", "fourier", "auto"}
        assert canonical.issubset(args), f"Missing canonical names: {canonical - args}"
        assert args - canonical <= {"independent"}, (
            f"Unexpected mode names in Literal: {args - canonical - {'independent'}}"
        )
