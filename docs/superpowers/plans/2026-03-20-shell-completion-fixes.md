# Shell Completion System Fixes + Drift Detection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all stale/incorrect shell completions and add a CI-enforced test that prevents future drift between argparse definitions and completion scripts.

**Architecture:** Fix completion.sh to match actual CLI flags, fix post_install.py zsh wrapper to use env vars instead of hardcoded paths, remove redundant aliases, then add a pytest test that parses `--help` output from each entry point and asserts the completion script offers the same flags.

**Tech Stack:** Bash (completion.sh), Python (post_install.py, pytest)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `heterodyne/runtime/shell/completion.sh` | Modify | Fix stale flags, remove aliases |
| `heterodyne/post_install.py` | Modify | Fix zsh wrapper path hardcoding |
| `tests/unit/cli/test_completion_parity.py` | Create | Drift-detection test |

---

### Task 1: Fix `_heterodyne_config` completion (P1 — completely stale)

**Files:**
- Modify: `heterodyne/runtime/shell/completion.sh:119-140`

**Context:** The `_heterodyne_config` completion function offers flags that don't exist (`--template`, `--minimal`, `--verbose`) and is missing almost every real flag. The actual CLI (`heterodyne-config --help`) accepts: `--output/-o`, `--data/-d`, `--q`, `--dt`, `--time-length`, `--overwrite`, `--show-template`, `--interactive/-i`, `--validate/-V`, `--mode {full,minimal,nlsq_only,cmc_only}`.

- [ ] **Step 1: Replace `_heterodyne_config` function body**

Replace lines 119-140 of `completion.sh` with:

```bash
# heterodyne-config completion
_heterodyne_config() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--output --data --q --dt --time-length --overwrite --show-template --interactive --validate --mode --help"

    case "$prev" in
        --output|-o|--data|-d)
            _filedir
            return
            ;;
        --mode)
            mapfile -t COMPREPLY < <(compgen -W "full minimal nlsq_only cmc_only" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n heterodyne/runtime/shell/completion.sh`
Expected: no output, exit 0

- [ ] **Step 3: Verify function loads**

Run: `bash -c 'source heterodyne/runtime/shell/completion.sh && type _heterodyne_config'`
Expected: prints function definition

---

### Task 2: Add `--verbose` to `_heterodyne_cleanup` completion (P2)

**Files:**
- Modify: `heterodyne/runtime/shell/completion.sh:168-175` (the `_heterodyne_cleanup` function)

- [ ] **Step 1: Add `--verbose` to opts string**

Change:
```bash
    local opts="--dry-run --force --interactive --help"
```
To:
```bash
    local opts="--dry-run --force --interactive --verbose --help"
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n heterodyne/runtime/shell/completion.sh`
Expected: exit 0

---

### Task 3: Remove `hexp`/`hsim` aliases from completion.sh (P2)

**Files:**
- Modify: `heterodyne/runtime/shell/completion.sh:228-232` (end of file)

**Context:** `pyproject.toml` already defines `hexp` and `hsim` as proper entry points (`main_hexp`/`main_hsim`). The aliases in completion.sh shadow these entry points. The `complete -F` registrations for `hexp`/`hsim` should stay (they enable tab-completion for the entry point commands), but the `alias` lines must go.

- [ ] **Step 1: Remove the two alias lines**

Delete these two lines from completion.sh:
```bash
alias hexp='heterodyne --plot-experimental-data'
alias hsim='heterodyne --plot-simulated-data'
```

Keep the `complete -F _heterodyne hexp` and `complete -F _heterodyne hsim` lines.

- [ ] **Step 2: Verify syntax and that completions still register**

Run: `bash -c 'source heterodyne/runtime/shell/completion.sh && complete -p hexp && complete -p hsim'`
Expected: prints `complete -F _heterodyne hexp` and `complete -F _heterodyne hsim`

---

### Task 4: Fix zsh wrapper to use env var instead of hardcoded path (P3)

**Files:**
- Modify: `heterodyne/post_install.py:177-223` (the `install_zsh_completion` function)

**Context:** The zsh completion wrapper currently hardcodes the absolute filesystem path to the bash completion script (e.g., `source "/home/wei/.venv/etc/bash_completion.d/heterodyne"`). If the venv is relocated, this path breaks. Instead, use `${VIRTUAL_ENV}` or `${CONDA_PREFIX}` environment variables which resolve at shell-init time.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/cli/test_completion_parity.py` (we'll add the parity tests in Task 5, but start with the zsh wrapper test here):

```python
"""Tests for shell completion system parity and correctness."""

from __future__ import annotations

import textwrap
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_completion_parity.py::TestZshWrapper -v`
Expected: FAIL (current code hardcodes absolute path)

- [ ] **Step 3: Fix `install_zsh_completion` in post_install.py**

Replace the content template (around lines 208-215) from:

```python
        content = f"""# Zsh completion for heterodyne (generated)
# Source the bash completion in zsh-compatible mode

autoload -Uz bashcompinit
bashcompinit

source "{completion_path}"
"""
```

To:

```python
        content = """# Zsh completion for heterodyne (generated)
# Source the bash completion in zsh-compatible mode

autoload -Uz bashcompinit
bashcompinit

# Resolve venv root at shell-init time (works after venv relocation)
source "${VIRTUAL_ENV:-${CONDA_PREFIX}}/etc/bash_completion.d/heterodyne"
"""
```

Simplify the `installed_bash` / `completion_path` logic above it (lines 198-207): keep the `install_bash_completion()` call as a prerequisite (ensures the bash file exists when the user sources the zsh wrapper), but remove the path variable since the shell resolves it at init time via env var.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_completion_parity.py::TestZshWrapper -v`
Expected: PASS

- [ ] **Step 5: Commit Tasks 1-4**

```bash
git add heterodyne/runtime/shell/completion.sh heterodyne/post_install.py tests/unit/cli/test_completion_parity.py
git commit -m "fix(shell): sync completions with actual CLI flags, fix zsh hardcoded path

- _heterodyne_config: replace stale flags with actual CLI options
- _heterodyne_cleanup: add missing --verbose
- Remove hexp/hsim aliases (entry points handle this)
- Zsh wrapper uses env var instead of hardcoded absolute path"
```

---

### Task 5: Add completion-parity drift-detection test (core deliverable)

**Files:**
- Modify: `tests/unit/cli/test_completion_parity.py` (created in Task 4)

**Context:** This test parses `--help` output from each CLI entry point, extracts the set of `--long-flags`, and asserts they appear in the corresponding completion function's opts string in `completion.sh`. This catches future drift automatically in CI.

- [ ] **Step 1: Write the drift-detection test**

Append to `tests/unit/cli/test_completion_parity.py`:

```python
import re
import subprocess


def _extract_help_flags(command: str) -> set[str]:
    """Run `command --help` and extract all --long-option flags."""
    result = subprocess.run(
        ["uv", "run", command, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Match --word patterns (ignore short flags like -v)
    return set(re.findall(r"--[\w][\w-]*", result.stdout))


def _extract_completion_opts(function_name: str) -> set[str]:
    """Extract the opts string from a completion function in completion.sh."""
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
        """Every --flag in `command --help` must appear in the
        completion function's opts or case branches."""
        help_flags = _extract_help_flags(command) - _ARGPARSE_BUILTINS
        completion_flags = _extract_completion_opts(function_name)

        missing = help_flags - completion_flags
        assert not missing, (
            f"Completion function {function_name} is missing flags "
            f"from `{command} --help`: {sorted(missing)}"
        )
```

- [ ] **Step 2: Run the parity tests to verify they pass**

Run: `uv run pytest tests/unit/cli/test_completion_parity.py::TestCompletionParity -v`
Expected: All 6 parametrized cases PASS (since we fixed completions in Tasks 1-4)

If any fail, the output tells us exactly which flags are missing from which function — fix them in completion.sh.

- [ ] **Step 3: Run full test suite smoke check**

Run: `make test-smoke`
Expected: All pass, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/unit/cli/test_completion_parity.py
git commit -m "test(cli): add completion-parity drift-detection test

Parametrized test parses --help output for all 6 CLI entry points
and asserts every --flag appears in the corresponding completion.sh
function. Catches completion/CLI drift automatically in CI."
```

---

### Task 6: Update fish completion in post_install.py (P3 — staleness)

**Files:**
- Modify: `heterodyne/post_install.py:226-295` (the `install_fish_completion` function)

**Context:** The fish completion generated inline in `install_fish_completion()` has the same staleness issue as the bash completion — `heterodyne-config` section offers `--template` instead of `--mode`, etc. Since fish completions are generated as a Python string (not a separate file), they must be updated here.

- [ ] **Step 1: Update the fish completion string**

In `install_fish_completion()`, replace the `heterodyne-config` section (approx lines 256-261) with:

```python
# heterodyne-config
complete -c heterodyne-config -s o -l output -d 'Output file' -F
complete -c heterodyne-config -s d -l data -d 'Data file path' -F
complete -c heterodyne-config -l q -d 'Wavevector magnitude'
complete -c heterodyne-config -l dt -d 'Time step'
complete -c heterodyne-config -l time-length -d 'Number of time points'
complete -c heterodyne-config -l overwrite -d 'Overwrite existing file'
complete -c heterodyne-config -l show-template -d 'Print template path'
complete -c heterodyne-config -s i -l interactive -d 'Interactive config builder'
complete -c heterodyne-config -s V -l validate -d 'Validate config file'
complete -c heterodyne-config -l mode -d 'Config mode' -a 'full minimal nlsq_only cmc_only'
complete -c heterodyne-config -s h -l help -d 'Show help'
```

Note: the fish cleanup section already has `--verbose` and the post-install section already has `--xla-mode` — no changes needed there.

Remove the fish alias lines:
```python
alias hexp 'heterodyne --plot-experimental-data'
alias hsim 'heterodyne --plot-simulated-data'
```

- [ ] **Step 2: Verify syntax**

Run: `uv run python -c "from heterodyne.post_install import install_fish_completion; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add heterodyne/post_install.py
git commit -m "fix(shell): sync fish completion with actual CLI flags"
```

---

## Verification Checklist

After all tasks:

- [ ] `bash -n heterodyne/runtime/shell/completion.sh` — exit 0
- [ ] `bash -c 'source completion.sh && complete -p heterodyne'` — all 12 registrations load
- [ ] `uv run pytest tests/unit/cli/test_completion_parity.py -v` — all pass
- [ ] `make test-smoke` — no regressions
- [ ] No `alias hexp` or `alias hsim` in completion.sh
- [ ] Zsh wrapper contains `${VIRTUAL_ENV:-${CONDA_PREFIX}}`, not absolute path
