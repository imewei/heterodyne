"""Regression pins for known-but-previously-untested production bugs.

Each test in this module pins a specific past incident so a future change
that reintroduces it fails CI loudly. Add new pins as RCAs land — the
filename intentionally groups them so newcomers can find all bug-pin tests
in one place.

Covered incidents:

* **het_457cc550** (2026-05-12) — fit_cmc_sharded crashed when called
  without an NLSQ warm-start because the per-pair time arrays
  ``t1`` / ``t2`` / ``time_grid`` were missing from ``_SHARD_ARRAY_KEYS``.
  Workers got ``None`` for the time axis and the element-wise CMC kernel
  blew up with ``TypeError: '>' NoneType vs float``.

* **NumPyro 0.21 init broadcast** (2026-05-12) — ``init_params`` for each
  varying parameter must have shape ``(num_chains,)``; passing a scalar
  raised ``IndexError: tuple index out of range`` deep in
  ``numpyro.infer.mcmc:683``. CMCConfig + samplers must broadcast
  consistently.

* **Alpha-bound widening** (2026-05-14, BFMI-gate-fix RCA) —
  ``alpha_ref`` and ``alpha_sample`` bounds were widened from ``[-2, 2]``
  to ``[-5, 5]`` after het_dd0f825b. Tightening either bound back below
  ``[-5, 5]`` would reintroduce the BFMI=0 failure mode at the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# het_457cc550 — no-warmstart shard array schema
# ---------------------------------------------------------------------------


class TestNoWarmstartShardArraySchema:
    """Pin the contract that ``_SHARD_ARRAY_KEYS`` carries the element-wise
    time fields. Without ``t1``/``t2``/``time_grid`` the worker receives
    ``None`` and the CMC element-wise kernel crashes."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_shard_array_specs_includes_element_wise_time_axes(self) -> None:
        """het_457cc550: t1, t2, time_grid MUST be in _SHARD_ARRAY_SPECS."""
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            _SHARD_ARRAY_SPECS,
        )

        for key in ("t1", "t2", "time_grid"):
            assert key in _SHARD_ARRAY_SPECS, (
                f"_SHARD_ARRAY_SPECS missing {key!r}; reintroducing het_457cc550 "
                f"(no-warmstart workers will get None and crash on element-wise "
                f"CMC kernel)."
            )
        # The element-wise time axes are optional (None when meshgrid path
        # is in use), so they must allow None.
        for key in ("t1", "t2", "time_grid"):
            assert _SHARD_ARRAY_SPECS[key].allow_none, (
                f"_SHARD_ARRAY_SPECS[{key!r}].allow_none must be True so "
                f"meshgrid-path workers don't reject the missing axes."
            )

    @pytest.mark.regression
    @pytest.mark.unit
    def test_shard_array_keys_alias_matches_specs(self) -> None:
        """_SHARD_ARRAY_KEYS is the iteration alias; it must enumerate the
        same keys as _SHARD_ARRAY_SPECS (no drift between schema and
        unpacker)."""
        from heterodyne.optimization.cmc.backends.multiprocessing import (
            _SHARD_ARRAY_KEYS,
            _SHARD_ARRAY_SPECS,
        )

        assert set(_SHARD_ARRAY_KEYS) == set(_SHARD_ARRAY_SPECS), (
            "Schema/key alias drift: _SHARD_ARRAY_KEYS does not enumerate "
            "the same fields as _SHARD_ARRAY_SPECS."
        )


# ---------------------------------------------------------------------------
# NumPyro 0.21 init_params broadcast
# ---------------------------------------------------------------------------


class TestNumpyroInitBroadcast:
    """Pin the contract that init_params has shape ``(num_chains,)``.

    NumPyro 0.21 reshaped its sampler init path so a scalar init triggers
    ``IndexError: tuple index out of range``. We assert here that the CMC
    init builder broadcasts every varying parameter to chain shape, for
    both single-chain (degenerate broadcast) and multi-chain configs.
    """

    @pytest.mark.regression
    @pytest.mark.unit
    @pytest.mark.parametrize("num_chains", [1, 2, 4])
    def test_init_params_have_chain_shape(self, num_chains: int) -> None:
        """For every num_chains, init_params for each varying param must
        be an array of shape (num_chains,) — never a scalar."""
        # Use the same fanout path the CMC sampler uses to seed init values.
        from heterodyne.optimization.cmc.sampler import _perturb_init_params

        # Synthetic 14-parameter init dict: one float per parameter.
        names = [
            "D0_ref",
            "alpha_ref",
            "D_offset_ref",
            "D0_sample",
            "alpha_sample",
            "D_offset_sample",
            "v0",
            "beta",
            "v_offset",
            "f0",
            "f1",
            "f2",
            "f3",
            "phi0",
        ]
        scalar_init = dict.fromkeys(names, 1.0)
        broadcast = _perturb_init_params(
            scalar_init,
            num_chains=num_chains,
            seed=0,
        )
        for name, value in broadcast.items():
            arr = np.asarray(value)
            assert arr.shape == (num_chains,), (
                f"init_params[{name!r}] has shape {arr.shape}, expected "
                f"({num_chains},). NumPyro 0.21 raises IndexError on "
                f"non-chain-shaped init."
            )
            assert np.all(np.isfinite(arr)), (
                f"init_params[{name!r}] contains non-finite values: {arr}. "
                f"NumPyro NUTS rejects NaN/Inf init."
            )


# ---------------------------------------------------------------------------
# Alpha-bound widening (BFMI gate)
# ---------------------------------------------------------------------------


class TestAlphaBoundsAtLeastFiveWide:
    """Pin the ``[-5, 5]`` widening of ``alpha_ref`` and ``alpha_sample``.

    Tightening either bound back below ``[-5, 5]`` would let NUTS hit the
    boundary on warm-start with a steep prior, producing BFMI=0 on all
    shards (het_dd0f825b). The 2026-05-14 RCA fixed this by widening both
    bounds; this test pins the widened envelope.
    """

    @pytest.mark.regression
    @pytest.mark.unit
    @pytest.mark.parametrize("param_name", ["alpha_ref", "alpha_sample"])
    def test_alpha_bounds_at_least_minus_five_to_plus_five(
        self, param_name: str
    ) -> None:
        """``alpha_*.min_bound <= -5`` and ``alpha_*.max_bound >= +5``."""
        from heterodyne.config.parameter_registry import DEFAULT_REGISTRY

        info = DEFAULT_REGISTRY[param_name]
        assert info.min_bound <= -5.0, (
            f"{param_name}.min_bound={info.min_bound} > -5.0; tightening below "
            f"-5 reintroduces the BFMI=0 failure mode from het_dd0f825b "
            f"(2026-05-14 RCA)."
        )
        assert info.max_bound >= 5.0, (
            f"{param_name}.max_bound={info.max_bound} < 5.0; tightening above "
            f"5 reintroduces the BFMI=0 failure mode from het_dd0f825b "
            f"(2026-05-14 RCA)."
        )
        # Sanity: the default must lie inside the widened envelope.
        assert info.min_bound <= info.default <= info.max_bound, (
            f"{param_name}.default={info.default} is outside the bound "
            f"envelope [{info.min_bound}, {info.max_bound}]."
        )
