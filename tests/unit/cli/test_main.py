"""Unit tests for the heterodyne CLI XLA bootstrap."""

from __future__ import annotations

import os


def test_threads_flag_overrides_inherited_omp_mkl(monkeypatch) -> None:
    """An explicit ``--threads`` must win over inherited OMP/MKL counts.

    Regression: the bootstrap previously used ``os.environ.setdefault`` for
    ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``, which left a high value set by a
    batch scheduler or login profile in place and oversubscribed BLAS/OpenMP
    relative to the XLA intra-op limit.
    """
    from heterodyne.cli.main import _bootstrap_xla_env

    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    monkeypatch.setenv("MKL_NUM_THREADS", "32")
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    _bootstrap_xla_env(["--threads", "4"])

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert "--intra_op_parallelism_threads=4" in os.environ["XLA_FLAGS"]


def test_threads_equals_form_overrides_inherited(monkeypatch) -> None:
    """The ``--threads=N`` form is parsed and overrides inherited counts too."""
    from heterodyne.cli.main import _bootstrap_xla_env

    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    _bootstrap_xla_env(["--threads=2"])

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_no_threads_leaves_inherited_omp_untouched(monkeypatch) -> None:
    """Without ``--threads`` the bootstrap must not touch OMP/MKL settings."""
    from heterodyne.cli.main import _bootstrap_xla_env

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

    _bootstrap_xla_env([])

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in os.environ
