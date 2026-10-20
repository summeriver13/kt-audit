from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .audit import audit_path
from .config import load_config
from .errors import AuditError
from .report import report_to_markdown, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kt-audit",
        description="Audit knowledge-tracing evaluation before you trust the AUC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="audit a CSV or JSONL file")
    run.add_argument("input", type=Path, help="input CSV or JSONL")
    run.add_argument(
        "--out",
        type=Path,
        default=Path("kt-audit-report"),
        help="output directory (default: kt-audit-report)",
    )
    run.add_argument("--config", type=Path, default=None, help="JSON config")
    run.add_argument("--no-temporal", action="store_true", help="skip temporal split check")
    run.add_argument("--no-multikc", action="store_true", help="skip multi-KC check")
    run.add_argument("--no-alignment", action="store_true", help="skip prediction alignment check")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config = replace(
            config,
            temporal=config.temporal and not args.no_temporal,
            multikc=config.multikc and not args.no_multikc,
            alignment=config.alignment and not args.no_alignment,
        )
        report = audit_path(args.input, config)
    except (AuditError, FileNotFoundError, ValueError) as exc:
        print(f"kt-audit: {exc}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(report, args.out / "report.json")
    (args.out / "report.md").write_text(report_to_markdown(report), encoding="utf-8")
    errors = sum(finding.severity == "error" for finding in report.findings)
    warnings = sum(finding.severity == "warn" for finding in report.findings)
    print(f"verdict: {report.verdict}")
    print(f"findings: {errors} error, {warnings} warn")
    print(f"report: {args.out / 'report.md'}")
    return 1 if report.verdict == "fail" else 0
