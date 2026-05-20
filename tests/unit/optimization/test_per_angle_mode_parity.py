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

    def test_literal_contains_four_canonical_modes(self) -> None:
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

    def test_validate_accepts_all_literal_modes(self) -> None:
        """`validate()` must accept every value the Literal allows.

        Catches the validate-vs-Literal drift that the code-quality review
        caught after Task A1 — the original A1 widened the Literal but
        forgot to update the validate() allowlist.
        """
        import warnings as _warnings

        for mode in ("individual", "constant", "fourier", "auto", "independent"):
            with _warnings.catch_warnings():
                # A2 alias: 'independent' emits DeprecationWarning at construction.
                _warnings.simplefilter("ignore", DeprecationWarning)
                cfg = NLSQConfig(per_angle_mode=mode)
            errors = cfg.validate()
            mode_errors = [e for e in errors if "per_angle_mode" in e]
            assert not mode_errors, (
                f"validate() rejected per_angle_mode={mode!r} "
                f"even though it is in the Literal: {mode_errors}"
            )

    def test_independent_deprecation_alias_normalises_to_individual(self) -> None:
        """`independent` maps to `individual` with a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="independent.*individual"):
            cfg = NLSQConfig(per_angle_mode="independent")
        assert cfg.per_angle_mode == "individual"

    def test_individual_does_not_warn(self) -> None:
        """`individual` is the canonical name and emits no DeprecationWarning."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            cfg = NLSQConfig(per_angle_mode="individual")
        assert cfg.per_angle_mode == "individual"
        deprecation_warnings = [
            r for r in record if issubclass(r.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, f"Unexpected: {deprecation_warnings}"

    def test_deprecation_warning_attributes_to_caller(self) -> None:
        """`stacklevel=3` makes the DeprecationWarning point to the user's
        call site, not the dataclass-generated __init__.

        Pins the empirically-verified stacklevel from the code-quality
        review fix of 45b5a1d.  With stacklevel=2 the warning reported
        '<string>:149' (the dataclass __init__); stacklevel=3 reports
        this test file at the line of the NLSQConfig() call.
        """
        import sys
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            expected_lineno = sys._getframe(0).f_lineno + 1
            NLSQConfig(per_angle_mode="independent")

        deprecation = [r for r in record if issubclass(r.category, DeprecationWarning)]
        assert len(deprecation) == 1, (
            f"Expected 1 DeprecationWarning, got {deprecation}"
        )
        w = deprecation[0]
        assert w.filename == __file__, (
            f"DeprecationWarning filename {w.filename!r} should point to the "
            f"test file {__file__!r}; stacklevel is probably wrong."
        )
        assert w.lineno == expected_lineno, (
            f"DeprecationWarning lineno {w.lineno} should match the caller "
            f"line {expected_lineno}; stacklevel is probably wrong."
        )
