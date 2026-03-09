"""Factory functions for configuration objects used in heterodyne tests.

All factories accept keyword overrides so individual tests can customise
only the fields they care about while keeping the rest at fast-test defaults.
"""

from __future__ import annotations

from typing import Any


def make_cmc_config(**overrides: Any):
    """Return a :class:`~heterodyne.optimization.cmc.config.CMCConfig` tuned for fast tests.

    Default counts are intentionally small (50 warmup, 100 samples, 2 chains)
    to keep unit-test runtime short.  Checkpoints and adaptive sampling are
    disabled by default so tests are fully deterministic.

    Args:
        **overrides: Any ``CMCConfig`` field can be overridden by name.

    Returns:
        A configured ``CMCConfig`` instance.
    """
    from heterodyne.optimization.cmc.config import CMCConfig

    defaults: dict[str, Any] = {
        "num_warmup": 50,
        "num_samples": 100,
        "num_chains": 2,
        "seed": 42,
        "enable_checkpoints": False,
        "adaptive_sampling": False,
        "chain_method": "sequential",
    }
    defaults.update(overrides)
    return CMCConfig(**defaults)


def make_nlsq_config(**overrides: Any) -> dict[str, Any]:
    """Return a dict of NLSQ configuration parameters suitable for fast tests.

    The returned dict can be passed directly to ``NLSQConfig(**make_nlsq_config())``.
    Defaults use a minimal iteration count and a loose tolerance so tests
    finish quickly without requiring full convergence.

    Args:
        **overrides: Any NLSQ config field can be overridden by name.

    Returns:
        Dictionary of NLSQ configuration values.
    """
    defaults: dict[str, Any] = {
        "max_iterations": 50,
        "tolerance": 1e-4,
        "method": "trf",
        "verbose": 0,
    }
    defaults.update(overrides)
    return defaults


def make_multistart_config(**overrides: Any):
    """Return a :class:`~heterodyne.optimization.nlsq.multistart.MultiStartConfig` for tests.

    Default counts are minimal (3 starts, 50-iteration cap) to keep tests
    fast.  Parallel execution is disabled to avoid JAX closure-serialisation
    issues across process boundaries.

    Args:
        **overrides: Any ``MultiStartConfig`` field can be overridden by name.

    Returns:
        A configured ``MultiStartConfig`` instance.
    """
    from heterodyne.optimization.nlsq.multistart import MultiStartConfig

    defaults: dict[str, Any] = {
        "n_starts": 3,
        "seed": 42,
        "parallel": False,
    }
    defaults.update(overrides)
    return MultiStartConfig(**defaults)
