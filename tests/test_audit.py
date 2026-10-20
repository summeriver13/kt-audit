from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kt_audit import AuditConfig, AuditError, audit_frame, audit_path
from kt_audit.schema import prepare_frame


def test_audit_frame_clean_rows_pass(make_frame, clean_rows) -> None:
    report = audit_frame(make_frame(clean_rows))
    assert report.verdict == "pass"
    assert report.findings == ()
    assert report.input["rows"] == 4


def test_audit_frame_leaky_rows_fail() -> None:
    path = Path(__file__).parents[1] / "examples" / "leaky.csv"
    report = audit_path(path, AuditConfig(reported_auc=0.5))
    assert report.verdict == "fail"
    checks = {finding.check for finding in report.findings}
    assert {"temporal_split", "multi_kc", "alignment"}.issubset(checks)


def test_audit_path_reads_jsonl(tmp_path: Path, clean_rows) -> None:
    path = tmp_path / "events.jsonl"
    pd.DataFrame(clean_rows).to_json(path, orient="records", lines=True)
    report = audit_path(path)
    assert report.verdict == "pass"


def test_audit_path_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="input not found"):
        audit_path(tmp_path / "missing.csv")


def test_audit_path_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "events.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(AuditError, match="unsupported input format"):
        audit_path(path)


def test_audit_config_can_disable_all_checks(make_frame, clean_rows) -> None:
    rows = clean_rows
    rows[0]["timestamp"] = "2024-01-02T10:00:00"
    report = audit_frame(
        make_frame(rows),
        AuditConfig(temporal=False, multikc=False, alignment=False),
    )
    assert report.verdict == "pass"
    assert report.findings == ()
    assert report.metrics == {}


def test_audit_summary_without_items(make_frame, clean_rows) -> None:
    df = make_frame(clean_rows).drop(columns=["item_id"])
    report = audit_frame(df)
    assert report.input["items"] is None


def test_audit_custom_kc_separator(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "item_id": "q1",
                "kc_ids": "kc_a;kc_b",
                "timestamp": "2024-01-01",
                "split": "train",
            }
        ]
    )
    prepared = prepare_frame(df, AuditConfig(kc_separator=";"))
    assert prepared.loc[0, "kc_ids"] == ("kc_a", "kc_b")


def test_audit_frame_warn_only(make_frame, clean_rows) -> None:
    rows = clean_rows
    for row in rows:
        row["y_pred"] = 0.5
    report = audit_frame(make_frame(rows))
    assert report.verdict == "warn"
    assert report.findings
