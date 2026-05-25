"""Unit tests for CLI data loading pipeline parity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


def test_load_and_validate_data_passes_configured_q_selection(monkeypatch) -> None:
    """CLI loading passes wavevector_q into the loader like homodyne."""
    import heterodyne.cli.data_pipeline as pipeline
    from heterodyne.data.xpcs_loader import XPCSData

    calls: list[dict[str, object]] = []
    raw = XPCSData(
        c2=np.ones((2, 4, 4), dtype=np.float64),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        q_values=np.array([0.0049, 0.0051]),
        phi_angles=np.array([0.0, 90.0]),
    )

    def fake_load_xpcs_data(*args, **kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(pipeline, "load_xpcs_data", fake_load_xpcs_data)

    validation = MagicMock()
    validation.is_valid = True
    validation.errors = []
    validation.warnings = []
    monkeypatch.setattr(pipeline, "validate_xpcs_data", lambda data: validation)

    cfg = MagicMock()
    cfg.start_frame = 1
    cfg.end_frame = 10
    cfg.data_file_path = Path("/tmp/data.h5")
    cfg.cache_filename_template = "cached_q${wavevector_q}.npz"
    cfg.wavevector_q = 0.005
    cfg.cache_file_path = Path("/tmp/cache")
    cfg.cache_compression = True

    pipeline.load_and_validate_data(cfg)

    assert calls
    assert calls[0]["select_q"] == 0.005
    assert calls[0]["q_tolerance"] == 0.0005


def _make_loader_cfg(start_frame: int, end_frame: int) -> MagicMock:
    cfg = MagicMock()
    cfg.start_frame = start_frame
    cfg.end_frame = end_frame
    cfg.data_file_path = Path("/tmp/data.h5")
    cfg.cache_filename_template = None
    cfg.wavevector_q = 0.005
    cfg.cache_file_path = Path("/tmp/cache")
    cfg.cache_compression = True
    return cfg


def _patch_loader(monkeypatch, calls: list[dict[str, object]]) -> None:
    import heterodyne.cli.data_pipeline as pipeline
    from heterodyne.data.xpcs_loader import XPCSData

    raw = XPCSData(
        c2=np.ones((2, 4, 4), dtype=np.float64),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        q_values=np.array([0.0049, 0.0051]),
        phi_angles=np.array([0.0, 90.0]),
    )

    def fake_load_xpcs_data(*args, **kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(pipeline, "load_xpcs_data", fake_load_xpcs_data)

    validation = MagicMock()
    validation.is_valid = True
    validation.errors = []
    validation.warnings = []
    monkeypatch.setattr(pipeline, "validate_xpcs_data", lambda data: validation)


def test_large_end_frame_maps_to_load_all_sentinel(monkeypatch) -> None:
    """A legacy ``end_frame=100000`` load-all config must not request an
    out-of-range slice; it maps to the loader's ``-1`` ("to last") sentinel."""
    import heterodyne.cli.data_pipeline as pipeline

    calls: list[dict[str, object]] = []
    _patch_loader(monkeypatch, calls)

    pipeline.load_and_validate_data(_make_loader_cfg(1, 100_000))

    assert calls
    assert calls[0]["frame_range"] == (1, -1)


def test_explicit_frame_range_passes_through(monkeypatch) -> None:
    """An ordinary in-range request is forwarded verbatim (no sentinel)."""
    import heterodyne.cli.data_pipeline as pipeline

    calls: list[dict[str, object]] = []
    _patch_loader(monkeypatch, calls)

    pipeline.load_and_validate_data(_make_loader_cfg(5, 200))

    assert calls
    assert calls[0]["frame_range"] == (5, 200)
