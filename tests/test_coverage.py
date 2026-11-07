from __future__ import annotations

import pandas as pd

from kt_audit import AuditConfig
from kt_audit.checks.coverage import check_coverage
from kt_audit.schema import prepare_frame

_ROWS = 40


def _rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": f"u{index:02d}",
            "item_id": f"q{index % 5:02d}",
            "kc_ids": f"kc_{index % 5:02d}",
            "timestamp": f"2024-03-{1 + index % 20:02d}T{index % 24:02d}:00:00",
            "split": "test",
            "y_true": "Recall Failure",
            "y_pred_label": "Recall Failure",
        }
        for index in range(_ROWS)
    ]


def _prepared(rows: list[dict[str, object]], config: AuditConfig | None = None) -> pd.DataFrame:
    return prepare_frame(pd.DataFrame(rows), config or AuditConfig())


def _check(rows: list[dict[str, object]], config: AuditConfig | None = None, expect=None):
    config = config or AuditConfig()
    return check_coverage(_prepared(rows, config), config, expect)


def test_every_expected_key_present_is_silent() -> None:
    rows = _rows()
    assert _check(rows, expect=pd.DataFrame(rows)) == []


def test_missing_expected_keys_are_errors() -> None:
    rows = _rows()
    expect = pd.DataFrame(rows)
    findings = _check(rows[:-3], expect=expect)
    assert findings[0].severity == "error"
    assert findings[0].count == 3
    assert len(findings[0].examples) == 3


def test_extra_input_keys_warn() -> None:
    rows = _rows()
    findings = _check(rows, expect=pd.DataFrame(rows[:30]))
    assert findings[0].severity == "warn"
    assert findings[0].count == 10


def test_expected_set_without_shared_keys_warns() -> None:
    findings = _check(_rows(), expect=pd.DataFrame([{"other": "x"}]))
    assert findings[0].severity == "warn"
    assert "shares no key column" in findings[0].message


def test_expected_set_with_fewer_key_columns_warns() -> None:
    rows = _rows()
    expect = pd.DataFrame({"user_id": [row["user_id"] for row in rows]})
    findings = _check(rows, expect=expect)
    partial = [finding for finding in findings if "comparison is partial" in finding.message]
    assert partial[0].severity == "warn"
    assert partial[0].count == 2
    assert partial[0].examples[0]["missing_key_columns"] == "item_id, timestamp"


def test_reported_row_count_mismatch_is_an_error() -> None:
    findings = _check(_rows(), AuditConfig(reported_n=1000))
    assert findings[0].severity == "error"
    assert findings[0].count == 1000 - _ROWS


def test_expected_row_count_mismatch_is_a_warning() -> None:
    findings = _check(_rows(), AuditConfig(expect_rows=41))
    assert findings[0].severity == "warn"
    assert findings[0].count == 1


def test_blank_label_is_an_error() -> None:
    rows = _rows()
    rows[0]["y_pred_label"] = ""
    findings = _check(rows)
    assert findings[0].severity == "error"
    assert findings[0].message == "the predicted label is missing"


def test_error_placeholder_label_is_an_error() -> None:
    rows = _rows()
    rows[0]["y_pred_label"] = "ERROR: not_a_dict"
    findings = _check(rows)
    assert findings[0].severity == "error"
    assert findings[0].message == "the predicted label holds an error placeholder"
    assert findings[0].examples[0]["user_id"] == "u00"


def test_ranking_file_is_silent() -> None:
    rows = [
        {
            "user_id": f"u{index:02d}",
            "item_id": f"q{index:02d}",
            "kc_ids": "kc_a",
            "timestamp": f"2024-03-01T{index % 24:02d}:00:00",
            "split": "test",
            "y_true": index % 2,
            "y_pred": 0.5,
        }
        for index in range(4)
    ]
    assert _check(rows) == []


def test_cli_expect_flag(tmp_path) -> None:
    from kt_audit.cli import main

    out = tmp_path / "report"
    code = main(
        [
            "run",
            "examples/dropped_rows.csv",
            "--expect",
            "examples/multiclass.csv",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert (out / "report.json").exists()
