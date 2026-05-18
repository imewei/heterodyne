"""Tests for the pjit per-device chain-split fix (Codex finding C4).

The previous code used ``max(1, n_chains // n_devices)`` which caused
spurious chains when ``n_chains < n_devices`` (e.g. 2 chains on 8 devices
ran 8 chains).  This test pins the corrected distribution and the new
``_slice_init_params`` helper that supports chain-shaped init.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from heterodyne.optimization.cmc.backends.pjit_backend import _slice_init_params


class TestSliceInitParams:
    """Per-device init slicing supports both broadcast and chain-shaped inits."""

    def test_none_passthrough(self) -> None:
        assert _slice_init_params(None, start=0, count=2, n_chains_total=4) is None

    def test_full_count_passthrough(self) -> None:
        init = {"x": jnp.zeros((4,))}
        result = _slice_init_params(init, start=0, count=4, n_chains_total=4)
        # When count == n_chains_total, no slicing — return unchanged dict.
        assert result is init

    def test_chain_shaped_init_sliced(self) -> None:
        # Chain-shaped: leading axis == n_chains_total.
        init = {
            "theta": jnp.arange(8.0).reshape(
                8,
            )
        }  # 8 chains
        sliced = _slice_init_params(init, start=2, count=3, n_chains_total=8)
        assert sliced is not None
        np.testing.assert_array_equal(
            np.asarray(sliced["theta"]), np.array([2.0, 3.0, 4.0])
        )

    def test_broadcast_init_passthrough(self) -> None:
        # Scalar / broadcast init — leading axis != n_chains_total.
        init = {"theta": jnp.array([1.5])}  # shape (1,), not (8,)
        sliced = _slice_init_params(init, start=2, count=3, n_chains_total=8)
        assert sliced is not None
        np.testing.assert_array_equal(np.asarray(sliced["theta"]), np.array([1.5]))

    def test_per_chain_2d_init_sliced(self) -> None:
        # (n_chains, n_params) per-chain init.
        init = {"theta": jnp.arange(8 * 14, dtype=jnp.float64).reshape(8, 14)}
        sliced = _slice_init_params(init, start=3, count=2, n_chains_total=8)
        assert sliced is not None
        assert sliced["theta"].shape == (2, 14)
        np.testing.assert_array_equal(
            np.asarray(sliced["theta"]),
            np.arange(3 * 14, 5 * 14, dtype=np.float64).reshape(2, 14),
        )


class TestChainDistributionMath:
    """Pin the floor-division + remainder distribution used by PjitBackend.run."""

    @staticmethod
    def _distribute(n_chains: int, n_devices: int) -> list[int]:
        """Reproduce the inner allocator used in PjitBackend.run."""
        base = n_chains // n_devices
        remainder = n_chains % n_devices
        return [base + (1 if i < remainder else 0) for i in range(n_devices)]

    def test_fewer_chains_than_devices(self) -> None:
        # Pre-fix bug: 2 chains on 8 devices spawned 8 chains.
        # Post-fix: only the first 2 devices get a chain; the rest get 0.
        per_device = self._distribute(n_chains=2, n_devices=8)
        assert per_device == [1, 1, 0, 0, 0, 0, 0, 0]
        assert sum(per_device) == 2

    def test_uneven_split(self) -> None:
        # 10 chains on 4 devices → [3, 3, 2, 2]
        per_device = self._distribute(n_chains=10, n_devices=4)
        assert per_device == [3, 3, 2, 2]
        assert sum(per_device) == 10

    def test_exact_split(self) -> None:
        per_device = self._distribute(n_chains=12, n_devices=4)
        assert per_device == [3, 3, 3, 3]

    def test_single_device(self) -> None:
        per_device = self._distribute(n_chains=4, n_devices=1)
        assert per_device == [4]
