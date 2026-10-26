from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from ._version import __version__
from .checks.alignment import check_alignment
from .checks.baseline import check_cheap_baseline
from .checks.degenerate import check_degenerate
from .checks.multikc import check_multikc
from .checks.temporal import check_temporal
from .config import AuditConfig
from .errors import AuditError
from .schema import Finding, Report, Verdict, prepare_frame

SCHEMA_VERSION = "0.2"
_SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


def audit_frame(
    df: pd.DataFrame,
    config: AuditConfig | None = None,
    *,
    digest: str | None = None,
) -> Report:
    config = config or AuditConfig()
    prepared = prepare_frame(df, config)
    findings: list[Finding] = []
    if config.temporal:
        findings.extend(check_temporal(prepared, config))
    if config.multikc:
        findings.extend(check_multikc(prepared, config))
    if config.degenerate:
        findings.extend(check_degenerate(prepared, config))
    metrics: dict[str, dict[str, object]] = {}
    if config.alignment:
        alignment_findings, metrics = check_alignment(prepared, config)
        findings.extend(alignment_findings)
    baselines: dict[str, dict[str, object]] = {}
    if config.cheap_baseline:
        baseline_findings, baselines = check_cheap_baseline(prepared, config)
        findings.extend(baseline_findings)
    findings.sort(key=_finding_sort_key)
    return Report(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        input=_input_summary(prepared, digest),
        findings=tuple(findings),
        metrics=metrics,
        verdict=_verdict(findings),
        tool_version=__version__,
        baselines=baselines,
    )


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise AuditError(f"input not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise AuditError(f"unsupported input format: {suffix}; use .csv or .jsonl")


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_path(path: str | Path, config: AuditConfig | None = None) -> Report:
    resolved = Path(path)
    return audit_frame(read_table(resolved), config, digest=file_digest(resolved))


def _finding_sort_key(finding: Finding) -> tuple[int, str, str]:
    return (_SEVERITY_ORDER[finding.severity], finding.check, finding.message)


def _verdict(findings: list[Finding]) -> Verdict:
    severities = {finding.severity for finding in findings}
    if "error" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"


def _input_summary(df: pd.DataFrame, digest: str | None) -> dict[str, object]:
    splits = {
        str(key): int(value) for key, value in df["split"].value_counts().sort_index().items()
    }
    has_items = bool(df["item_id"].notna().any())
    return {
        "rows": len(df),
        "users": int(cast(int, df["user_id"].nunique())),
        "items": int(cast(int, df["item_id"].nunique())) if has_items else None,
        "folds": int(cast(int, df["_fold"].nunique())),
        "splits": splits,
        "sha256": digest,
    }
