"""Shell-injection hardening for the PBS backend (codex W5).

PBSConfig fields are interpolated into a generated PBS shell script.  An
attacker who controls a config value could inject commands via
``;``, ``$()``, backticks, or newlines.  The hardening:

1. ``PBSConfig.__post_init__`` validates every shell-bound field against
   a conservative allowlist; misconfigured construction raises
   :class:`ValueError` immediately.
2. ``_build_pbs_script`` re-validates the same fields and applies
   :func:`shlex.quote` to every interpolated value on shell-execution
   lines (PBS directive lines aren't quoted because qsub parses them
   directly).
3. ``extra_pbs_directives`` is a trusted-input field, but a defensive
   line-by-line metacharacter check still rejects obvious injection.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from heterodyne.optimization.cmc.backends.pbs import (
    PBSConfig,
    _build_pbs_script,
)


def _build_with(**overrides):
    base = {
        "queue": "batch",
        "walltime": "01:00:00",
        "memory": "4gb",
        "python_executable": "python",
        "nodes": 1,
        "ppn": 1,
    }
    base.update(overrides)
    return PBSConfig(**base)


class TestPBSConfigValidation:
    def test_queue_with_metacharacter_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(queue="my; rm -rf /")
        msg = str(exc.value)
        assert "queue" in msg
        assert "my; rm -rf /" in msg

    def test_python_executable_with_dollar_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(python_executable="/usr/bin/$(touch evil)/python")
        msg = str(exc.value)
        assert "python_executable" in msg

    def test_python_executable_with_space_raises(self) -> None:
        # Spaces in HPC python paths are vanishingly rare in practice.  If
        # your installation requires one, symlink to a no-space path.
        with pytest.raises(ValueError) as exc:
            _build_with(python_executable="/path with space/python")
        msg = str(exc.value)
        assert "python_executable" in msg

    def test_walltime_must_be_hh_mm_ss(self) -> None:
        with pytest.raises(ValueError):
            _build_with(walltime="not-a-walltime")

    def test_memory_must_match_pbs_unit(self) -> None:
        with pytest.raises(ValueError):
            _build_with(memory="4 gigs")

    def test_nodes_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _build_with(nodes=0)
        with pytest.raises(ValueError):
            _build_with(nodes=10_001)


class TestExtraDirectivesValidation:
    def test_extra_directives_with_semicolon_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["#PBS -l mem=4gb; rm -rf /"])
        msg = str(exc.value)
        assert "extra_pbs_directives" in msg
        assert "';'" in msg

    def test_extra_directives_with_backticks_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["#PBS -l mem=`whoami`gb"])
        assert "extra_pbs_directives" in str(exc.value)

    def test_extra_directives_with_command_substitution_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["#PBS -l mem=$(whoami)gb"])
        assert "$(" in str(exc.value) or "extra_pbs_directives" in str(exc.value)

    def test_extra_directives_with_newline_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["#PBS -l mem=4gb\nrm -rf /"])
        assert "extra_pbs_directives" in str(exc.value)

    def test_extra_directives_must_start_with_pbs_prefix(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["just a comment"])
        assert "#PBS" in str(exc.value)

    def test_extra_directives_oversize_raises(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build_with(extra_pbs_directives=["#PBS " + "x" * 300])
        assert "characters" in str(exc.value)


class TestScriptGeneration:
    def test_valid_inputs_produce_valid_script(self) -> None:
        cfg = _build_with()
        script = _build_pbs_script(
            job_name="test_job",
            pbs_cfg=cfg,
            python_exe=cfg.python_executable,
            worker_script_path=Path("/tmp/worker.py"),
            data_path=Path("/tmp/data.pkl"),
            result_path=Path("/tmp/result"),
            stdout_path=Path("/tmp/out"),
            stderr_path=Path("/tmp/err"),
        )
        # Shebang + standard PBS directives present
        assert script.startswith("#!/bin/bash\n")
        assert "#PBS -q batch\n" in script
        assert "#PBS -N test_job\n" in script
        # Shell-execution line uses shlex.quote on every interpolated value
        expected_token = shlex.quote("python")
        assert f"{expected_token} " in script.splitlines()[-1]
        # The worker / data / result paths are also quoted
        assert shlex.quote("/tmp/worker.py") in script
        assert shlex.quote("/tmp/data.pkl") in script
        assert shlex.quote("/tmp/result") in script

    def test_script_rejects_python_exe_with_metachars_at_build_time(self) -> None:
        # Even if PBSConfig somehow contained a bad python_executable
        # (e.g. via direct mutation), _build_pbs_script catches it on the
        # defense-in-depth pass.
        cfg = _build_with()
        with pytest.raises(ValueError) as exc:
            _build_pbs_script(
                job_name="test_job",
                pbs_cfg=cfg,
                python_exe="/usr/bin/python; touch /tmp/evil",
                worker_script_path=Path("/tmp/worker.py"),
                data_path=Path("/tmp/data.pkl"),
                result_path=Path("/tmp/result"),
                stdout_path=Path("/tmp/out"),
                stderr_path=Path("/tmp/err"),
            )
        assert "python_executable" in str(exc.value)
