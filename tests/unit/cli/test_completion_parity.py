"""Tests for shell completion system parity and correctness."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from heterodyne.post_install import install_zsh_completion


@pytest.mark.unit
class TestZshWrapper:
    """Zsh completion wrapper must use env vars, not hardcoded paths."""

    def test_zsh_wrapper_uses_env_var_not_hardcoded_path(
        self, tmp_path: Path
    ) -> None:
        """The generated zsh wrapper must reference the bash completion
        via an environment variable, not a hardcoded absolute path."""
        venv = tmp_path / "fakevenv"
        (venv / "etc" / "bash_completion.d").mkdir(parents=True)
        (venv / "etc" / "bash_completion.d" / "heterodyne").write_text("# stub")
        (venv / "etc" / "zsh").mkdir(parents=True)

        with patch(
            "heterodyne.post_install.get_completion_source_path",
            return_value=venv / "etc" / "bash_completion.d" / "heterodyne",
        ):
            ok = install_zsh_completion(venv, verbose=False)

        assert ok
        content = (venv / "etc" / "zsh" / "heterodyne-completion.zsh").read_text()
        # Must NOT contain the tmp_path literal
        assert str(tmp_path) not in content
        # Must use a variable-based path
        assert "${VIRTUAL_ENV:-${CONDA_PREFIX}}" in content


def _extract_help_flags(command: str) -> set[str]:
    """Run ``command --help`` and extract all ``--long-option`` flags."""
    result = subprocess.run(
        ["uv", "run", command, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Match --word patterns (ignore short flags like -v)
    return set(re.findall(r"--[\w][\w-]*", result.stdout))


def _extract_completion_opts(function_name: str) -> set[str]:
    """Extract all ``--flags`` from a completion function in completion.sh."""
    script = (
        Path(__file__).resolve().parents[3]
        / "heterodyne"
        / "runtime"
        / "shell"
        / "completion.sh"
    )
    content = script.read_text()

    # Find the function body
    pattern = rf"^{re.escape(function_name)}\(\)\s*\{{(.+?)\n\}}"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    assert match, f"Function {function_name} not found in completion.sh"
    body = match.group(1)

    # Extract all --flags from the function body (opts strings + case patterns)
    return set(re.findall(r"--[\w][\w-]*", body))


# Map: (CLI command, completion function name)
_PARITY_CASES = [
    ("heterodyne", "_heterodyne"),
    ("heterodyne-config", "_heterodyne_config"),
    ("heterodyne-config-xla", "_heterodyne_config_xla"),
    ("heterodyne-post-install", "_heterodyne_post_install"),
    ("heterodyne-cleanup", "_heterodyne_cleanup"),
    ("heterodyne-validate", "_heterodyne_validate"),
]

# Flags that argparse adds automatically — not needed in completion
_ARGPARSE_BUILTINS = {"--help", "--version"}


@pytest.mark.unit
class TestCompletionParity:
    """Completion script must offer every flag the CLI accepts."""

    @pytest.mark.parametrize(
        "command,function_name",
        _PARITY_CASES,
        ids=[c[0] for c in _PARITY_CASES],
    )
    def test_completion_covers_all_cli_flags(
        self, command: str, function_name: str
    ) -> None:
        """Every ``--flag`` in ``command --help`` must appear in the
        completion function's opts or case branches."""
        help_flags = _extract_help_flags(command) - _ARGPARSE_BUILTINS
        completion_flags = _extract_completion_opts(function_name)

        missing = help_flags - completion_flags
        assert not missing, (
            f"Completion function {function_name} is missing flags "
            f"from `{command} --help`: {sorted(missing)}"
        )
