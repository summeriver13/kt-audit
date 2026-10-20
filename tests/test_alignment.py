from __future__ import annotations

import pandas as pd

from kt_audit import AuditConfig
from kt_audit.checks.alignment import check_alignment
from kt_audit.schema import prepare_frame


def _prepare(make_frame, rows: list[dict[str, object]], config: AuditConfig | None = None):
    config = config or AuditConfig()
    return check_alignment(prepare_frame(make_frame(rows), config), config)


def test_alignment_clean_rows(make_frame, clean_rows) -> None:
    findings, metrics = _prepare(make_frame, clean_rows)
    assert findings == []
    assert metrics["train"]["auc"] == 1.0
    assert metrics["test"]["acc"] == 1.0
    assert metrics["test"]["n"] == 2


def test_alignment_rejects_out_of_range_and_nan(make_frame, clean_rows) -> None:
    rows = clean_rows
    rows[0]["y_pred"] = 1.2
    rows[1]["y_pred"] = float("nan")
    findings, _ = _prepare(make_frame, rows)
    invalid = [finding for finding in findings if "y_pred" in finding.message]
    assert invalid[0].count == 2


def test_alignment_rejects_non_binary_true(make_frame, clean_rows) -> None:
    rows = clean_rows
    rows[0]["y_true"] = 2
    findings, _ = _prepare(make_frame, rows)
    assert any(finding.message == "y_true is not binary" for finding in findings)


def test_alignment_detects_duplicates(make_frame, clean_rows) -> None:
    rows = [*clean_rows, dict(clean_rows[0])]
    findings, _ = _prepare(make_frame, rows)
    duplicate = [finding for finding in findings if finding.message == "duplicate interaction keys"]
    assert duplicate[0].count == 1
    assert len(duplicate[0].examples) == 2


def test_alignment_warns_constant_predictions(make_frame, clean_rows) -> None:
    rows = clean_rows
    for row in rows:
        row["y_pred"] = 0.5
    findings, _ = _prepare(make_frame, rows)
    assert any(
        finding.message == "predictions are constant within this fold" for finding in findings
    )


def test_alignment_reported_auc_mismatch(make_frame, clean_rows) -> None:
    findings, _ = _prepare(make_frame, clean_rows, AuditConfig(reported_auc=0.5))
    assert any(
        finding.message == "reported test AUC differs from recomputed AUC" for finding in findings
    )


def test_alignment_skips_without_predictions(make_frame, clean_rows) -> None:
    df = make_frame(clean_rows).drop(columns=["y_pred"])
    findings, metrics = check_alignment(prepare_frame(df, AuditConfig()), AuditConfig())
    assert findings[0].severity == "info"
    assert metrics == {}


def test_alignment_single_class_has_no_auc(make_frame, clean_rows) -> None:
    rows = [row for row in clean_rows if row["split"] == "test"]
    for row in rows:
        row["y_true"] = 1
    _, metrics = _prepare(make_frame, rows)
    assert metrics["test"]["auc"] is None


def test_alignment_key_fallback_without_item_id(make_frame, clean_rows) -> None:
    df = make_frame(clean_rows).drop(columns=["item_id"])
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    findings, _ = check_alignment(prepare_frame(duplicated, AuditConfig()), AuditConfig())
    assert any(finding.message == "duplicate interaction keys" for finding in findings)


def test_alignment_reported_auc_without_test_split(make_frame, clean_rows) -> None:
    rows = [row for row in clean_rows if row["split"] == "train"]
    findings, _ = _prepare(make_frame, rows, AuditConfig(reported_auc=0.5))
    assert any("cannot be computed" in finding.message for finding in findings)
