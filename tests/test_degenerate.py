from __future__ import annotations

from kt_audit import AuditConfig
from kt_audit.checks.degenerate import check_degenerate
from kt_audit.schema import prepare_frame


def _rows(
    count: int,
    prediction,
    label=None,
    split: str = "test",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "user_id": f"u{index % 20}",
                "item_id": f"q{index % 10}",
                "kc_ids": "kc_a",
                "timestamp": f"2024-01-{1 + index // 20:02d}T{index % 20:02d}:00:00",
                "split": split,
                "y_true": index % 2 if label is None else label,
                "y_pred": prediction(index),
            }
        )
    return rows


def _check(make_frame, rows: list[dict[str, object]]):
    config = AuditConfig()
    return check_degenerate(prepare_frame(make_frame(rows), config), config)


def test_flags_few_distinct_values(make_frame) -> None:
    findings = _check(make_frame, _rows(60, lambda index: 0.4 if index % 2 else 0.6))
    assert any("fewer than 10 distinct values" in finding.message for finding in findings)


def test_flags_narrow_band(make_frame) -> None:
    findings = _check(make_frame, _rows(60, lambda index: 0.5 + index * 1e-5))
    assert any("narrow band" in finding.message for finding in findings)


def test_small_sample_is_info(make_frame) -> None:
    findings = _check(make_frame, _rows(10, lambda index: 0.1 + index * 0.05))
    small = [finding for finding in findings if "fewer than 30 predictions" in finding.message]
    assert small[0].severity == "info"


def test_flags_rare_minority(make_frame) -> None:
    rows = _rows(200, lambda index: 0.1 + (index % 20) * 0.04)
    for row in rows:
        row["y_true"] = 0
    rows[0]["y_true"] = 1
    findings = _check(make_frame, rows)
    assert any("minority class" in finding.message for finding in findings)


def test_skips_without_predictions(make_frame) -> None:
    rows = _rows(60, lambda index: 0.5)
    for row in rows:
        row.pop("y_pred")
    findings = _check(make_frame, rows)
    assert findings[0].severity == "info"
    assert "y_pred is missing" in findings[0].message


def test_clean_predictions_have_no_findings(make_frame) -> None:
    findings = _check(make_frame, _rows(200, lambda index: 0.1 + (index % 20) * 0.04))
    assert findings == []


def test_only_evaluation_splits_are_checked(make_frame) -> None:
    findings = _check(make_frame, _rows(60, lambda index: 0.5, split="train"))
    assert findings == []


def test_single_class_skips_minority_check(make_frame) -> None:
    rows = _rows(200, lambda index: 0.1 + (index % 20) * 0.04, label=1)
    findings = _check(make_frame, rows)
    assert not any("minority class" in finding.message for finding in findings)
