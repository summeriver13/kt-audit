from __future__ import annotations

from kt_audit import AuditConfig
from kt_audit.checks.temporal import check_temporal
from kt_audit.schema import prepare_frame


def _rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": "u1",
            "item_id": "q1",
            "kc_ids": "kc_a",
            "timestamp": "2024-01-01T10:00:00",
            "split": "train",
        },
        {
            "user_id": "u2",
            "item_id": "q2",
            "kc_ids": "kc_b",
            "timestamp": "2024-01-01T10:05:00",
            "split": "test",
        },
    ]


def _check(make_frame, rows: list[dict[str, object]], config: AuditConfig | None = None):
    config = config or AuditConfig()
    return check_temporal(prepare_frame(make_frame(rows), config), config)


def test_temporal_detects_per_user_violation(make_frame) -> None:
    rows = _rows()
    rows[1]["user_id"] = "u1"
    rows[0]["timestamp"] = "2024-01-02T10:00:00"
    findings = _check(make_frame, rows)
    assert any(finding.severity == "error" for finding in findings)
    assert any("user" in finding.message for finding in findings)


def test_temporal_warns_when_user_overlap_is_ordered(make_frame) -> None:
    rows = _rows()
    rows[1]["user_id"] = "u1"
    findings = _check(make_frame, rows)
    assert not any(finding.severity == "error" for finding in findings)
    assert any(finding.severity == "warn" for finding in findings)


def test_temporal_skips_without_evaluation_split(make_frame) -> None:
    rows = _rows()[:1]
    assert _check(make_frame, rows) == []


def test_temporal_reports_per_fold(make_frame) -> None:
    rows = _rows()
    rows[0]["fold"] = 1
    rows[1]["fold"] = 1
    rows[1]["timestamp"] = "2023-12-31T10:00:00"
    findings = _check(make_frame, rows)
    examples = [example for finding in findings for example in finding.examples]
    assert any(example.get("fold") == "1" for example in examples)
