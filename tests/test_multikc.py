from __future__ import annotations

from kt_audit import AuditConfig
from kt_audit.checks.multikc import check_multikc
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
            "item_id": "q1",
            "kc_ids": "kc_b",
            "timestamp": "2024-01-02T10:00:00",
            "split": "test",
        },
    ]


def _check(make_frame, rows: list[dict[str, object]], config: AuditConfig | None = None):
    config = config or AuditConfig()
    return check_multikc(prepare_frame(make_frame(rows), config), config)


def test_multikc_detects_cross_split_kcs(make_frame) -> None:
    findings = _check(make_frame, _rows())
    assert any(finding.severity == "error" for finding in findings)
    assert any(finding.count == 1 for finding in findings)


def test_multikc_warns_item_overlap_without_cross(make_frame) -> None:
    rows = _rows()
    rows[1]["kc_ids"] = "kc_a"
    findings = _check(make_frame, rows)
    assert not any(finding.severity == "error" for finding in findings)
    assert any(finding.severity == "warn" for finding in findings)


def test_multikc_skips_without_item_id(make_frame) -> None:
    rows = _rows()
    for row in rows:
        row.pop("item_id")
    findings = _check(make_frame, rows)
    assert findings[0].severity == "info"
    assert "skipped" in findings[0].message


def test_multikc_warns_when_some_rows_lack_item_id(make_frame) -> None:
    rows = _rows()
    rows[1]["item_id"] = None
    findings = _check(make_frame, rows)
    assert any(finding.message == "rows with missing item_id were skipped" for finding in findings)


def test_multikc_clean_rows_have_no_findings(make_frame, clean_rows) -> None:
    assert _check(make_frame, clean_rows) == []
