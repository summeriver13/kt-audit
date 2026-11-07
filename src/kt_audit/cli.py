from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

from . import __version__
from .audit import audit_path
from .compare import METRICS, ROUNDS, SEED, compare_paths, write_comparison
from .config import load_config
from .errors import AuditError
from .report import report_to_markdown, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kt-audit",
        description="Audit knowledge-tracing evaluation before you trust the AUC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run(subparsers)
    _add_compare(subparsers)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _add_run(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
    run.add_argument(
        "--no-alignment", action="store_true", help="skip prediction alignment findings"
    )
    run.add_argument("--no-baseline", action="store_true", help="skip cheap baseline check")
    run.add_argument(
        "--no-degenerate", action="store_true", help="skip degenerate prediction check"
    )
    run.add_argument("--no-floor", action="store_true", help="skip the majority-class floor check")
    run.add_argument("--no-coverage", action="store_true", help="skip the row coverage check")
    run.add_argument(
        "--expect",
        type=Path,
        default=None,
        help="CSV or JSONL of expected keys; every expected key must appear once",
    )
    run.add_argument(
        "--expect-rows",
        type=int,
        default=None,
        help="expected number of rows; a mismatch is a warning",
    )


def _add_compare(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    compare = subparsers.add_parser(
        "compare",
        help="compare two prediction files with a paired bootstrap",
    )
    compare.add_argument("a", type=Path, help="baseline predictions")
    compare.add_argument("b", type=Path, help="candidate predictions")
    compare.add_argument(
        "--out",
        type=Path,
        default=Path("kt-audit-comparison"),
        help="output directory (default: kt-audit-comparison)",
    )
    compare.add_argument("--config", type=Path, default=None, help="JSON config")
    compare.add_argument("--rounds", type=int, default=ROUNDS, help=f"default: {ROUNDS}")
    compare.add_argument("--seed", type=int, default=SEED, help=f"default: {SEED}")
    compare.add_argument(
        "--metric",
        choices=METRICS,
        default="auc",
        help="metric to compare (default: auc)",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return _compare(args)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        config = replace(
            config,
            temporal=config.temporal and not args.no_temporal,
            multikc=config.multikc and not args.no_multikc,
            alignment=config.alignment and not args.no_alignment,
            cheap_baseline=config.cheap_baseline and not args.no_baseline,
            degenerate=config.degenerate and not args.no_degenerate,
            metric_floor=config.metric_floor and not args.no_floor,
            coverage=config.coverage and not args.no_coverage,
            expect_rows=args.expect_rows if args.expect_rows is not None else config.expect_rows,
        )
        report = audit_path(args.input, config, expect=args.expect)
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
    for fold, row in sorted(report.baselines.items()):
        model = row.get("model_auc")
        if isinstance(model, float):
            baseline = float(cast(float, row["baseline_auc"]))
            print(f"baseline[{fold}]: {baseline:.4f} vs model {model:.4f}")
    print(f"report: {args.out / 'report.md'}")
    return 1 if report.verdict == "fail" else 0


def _compare(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        comparison = compare_paths(
            args.a,
            args.b,
            config,
            rounds=args.rounds,
            seed=args.seed,
            metric=args.metric,
        )
    except (AuditError, FileNotFoundError, ValueError) as exc:
        print(f"kt-audit: {exc}", file=sys.stderr)
        return 2
    _, markdown_path = write_comparison(comparison, args.out)
    print(f"verdict: {comparison.verdict}")
    print(
        f"delta: {comparison.delta:+.4f} "
        f"(95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}])"
    )
    print(f"report: {markdown_path}")
    return 0
