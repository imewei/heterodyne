"""CLI entry point: `python -m tools.parity_audit`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.parity_audit import (
    diff_extracts,
    extract_classes,
    extract_configs,
    extract_docs,
    extract_exports,
    extract_file_inventory,
    extract_logs_errors,
    extract_signatures,
)
from tools.parity_audit import (
    extract_cli as extract_cli_mod,
)


def _cmd_extract(args: argparse.Namespace) -> int:
    package = Path(args.package).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    docs_root = (
        Path(args.docs).resolve() if args.docs else package.parent / "docs" / "source"
    )

    extract_signatures.write_json(package, out / "signatures.json")
    extract_exports.write_json(package, out / "exports.json")
    extract_classes.write_json(package, out / "classes.json")
    extract_configs.write_json(package, out / "configs.json")
    extract_cli_mod.write_json(package, out / "cli.json")
    extract_logs_errors.write_json(package, out / "logs_errors.json")
    extract_docs.write_json(docs_root, out / "docs.json")
    extract_file_inventory.write_json(
        package,
        out / "file_inventory.json",
        docs_root=docs_root if docs_root.exists() else None,
    )
    print(f"Wrote 8 extracts to {out}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    homo_dir = Path(args.homodyne).resolve()
    hetero_dir = Path(args.heterodyne).resolve()
    out = Path(args.out).resolve()
    all_gaps = diff_extracts.run_full_diff(
        homodyne_extracts=homo_dir, heterodyne_extracts=hetero_dir
    )
    body = diff_extracts.render_report(
        all_gaps, homodyne_sha=args.homodyne_sha, heterodyne_sha=args.heterodyne_sha
    )
    out.write_text(body)
    print(f"Wrote {len(all_gaps)} gaps to {out}")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    homo_pkg = Path(args.homodyne).resolve()
    hetero_pkg = Path(args.heterodyne).resolve()
    out = Path(args.out).resolve()
    homo_extracts = out / "extracts" / "homodyne"
    hetero_extracts = out / "extracts" / "heterodyne"
    homo_extracts.mkdir(parents=True, exist_ok=True)
    hetero_extracts.mkdir(parents=True, exist_ok=True)

    def _run_extract(pkg: Path, out_dir: Path) -> None:
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.parity_audit",
                "extract",
                "--package",
                str(pkg),
                "--out",
                str(out_dir),
            ],
            check=False,
        ).returncode
        if rc != 0:
            raise SystemExit(rc)

    _run_extract(homo_pkg, homo_extracts)
    _run_extract(hetero_pkg, hetero_extracts)

    homo_sha = subprocess.check_output(
        ["git", "-C", str(homo_pkg.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    hetero_sha = subprocess.check_output(
        ["git", "-C", str(hetero_pkg.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    (out / "homodyne_sha.txt").write_text(homo_sha + "\n")
    (out / "heterodyne_sha.txt").write_text(hetero_sha + "\n")

    report = out / "REPORT.md"
    all_gaps = diff_extracts.run_full_diff(
        homodyne_extracts=homo_extracts, heterodyne_extracts=hetero_extracts
    )
    body = diff_extracts.render_report(
        all_gaps, homodyne_sha=homo_sha, heterodyne_sha=hetero_sha
    )
    report.write_text(body)
    print(f"Run-all complete. {len(all_gaps)} gaps. Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.parity_audit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Run all extractors against one package")
    p_extract.add_argument("--package", required=True)
    p_extract.add_argument("--out", required=True)
    p_extract.add_argument(
        "--docs", help="Path to docs/source/ (defaults to ../docs/source)"
    )
    p_extract.set_defaults(func=_cmd_extract)

    p_diff = sub.add_parser("diff", help="Diff two extract dirs and emit REPORT.md")
    p_diff.add_argument("--homodyne", required=True)
    p_diff.add_argument("--heterodyne", required=True)
    p_diff.add_argument("--out", required=True)
    p_diff.add_argument("--homodyne-sha", default="unknown")
    p_diff.add_argument("--heterodyne-sha", default="unknown")
    p_diff.set_defaults(func=_cmd_diff)

    p_run = sub.add_parser("run-all", help="extract + diff for both packages, pin SHAs")
    p_run.add_argument("--homodyne", required=True)
    p_run.add_argument("--heterodyne", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
