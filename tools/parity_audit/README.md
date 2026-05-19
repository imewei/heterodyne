# Parity Audit Tooling

Mechanical diff between `heterodyne/` and `homodyne/` for strict 1:1 mirror
enforcement. See the spec at
`docs/superpowers/specs/2026-05-18-heterodyne-homodyne-parity-design.md`.

## Usage

```bash
# From repo root, with both packages checked out side-by-side:
python -m tools.parity_audit run-all \
    --heterodyne /path/to/heterodyne \
    --homodyne /path/to/homodyne \
    --out docs/superpowers/audits/2026-05-18-parity-audit
```

Outputs:
- `extracts/heterodyne/*.json`, `extracts/homodyne/*.json` — raw per-extractor data
- `REPORT.md` — categorized human-readable diff (P0/P1/P2/P3 severity)
- `{homodyne,heterodyne}_sha.txt` — SHA pins

## Extractors

| File | Purpose |
|---|---|
| `extract_file_inventory.py` | File-structure diff |
| `extract_signatures.py` | Public function signatures |
| `extract_exports.py` | `__all__` lists per `__init__.py` |
| `extract_classes.py` | Public class shapes |
| `extract_configs.py` | Config keys |
| `extract_cli.py` | argparse flags + subcommands |
| `extract_logs_errors.py` | Log strings, exception stems, exit codes |
| `extract_docs.py` | Sphinx structure |

## Re-running after a PR

CI runs `python -m tools.parity_audit run-all` and posts the diff vs the pre-PR
baseline as a PR comment. A closed gap that reopens fails the build.
