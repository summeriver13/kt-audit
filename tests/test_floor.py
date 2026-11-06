from __future__ import annotations

import pandas as pd

from kt_audit import AuditConfig
from kt_audit.checks.floor import check_metric_floor
from kt_audit.schema import prepare_frame

_MAJORITY = ["PE"] * 40 + ["CG"] * 20


def _frame(labels: list[str], predictions: list[str], split: str = "test") -> pd.DataFrame:
    rows = [
        {
            "user_id": f"u{index}",
            "item_id": f"q{index}",
            "kc_ids": "kc_a",
            "timestamp": f"2024-01-01T{index % 24:02d}:00:00",
            "split": split,
            "y_true": label,
            "y_pred_label": prediction,
        }
        for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True))
    ]
    return pd.DataFrame(rows)


def _check(df: pd.DataFrame, config: AuditConfig | None = None):
    config = config or AuditConfig()
    return check_metric_floor(prepare_frame(df, config), config)


def test_accuracy_below_the_floor_is_an_error() -> None:
    findings = _check(_frame(_MAJORITY, ["CG"] * 60))
    assert findings[0].severity == "error"
    assert "does not beat always answering PE" in findings[0].message
    assert findings[0].count == 60


def test_accuracy_equal_to_the_floor_warns() -> None:
    findings = _check(_frame(_MAJORITY, ["PE"] * 60))
    assert findings[0].severity == "warn"
    assert "within" in findings[0].message


def test_beating_the_floor_is_silent() -> None:
    findings = _check(_frame(_MAJORITY, _MAJORITY))
    assert findings == []


def test_margin_is_configurable() -> None:
    config = AuditConfig(floor_warn_margin=0.0)
    assert _check(_frame(_MAJORITY, ["PE"] * 60), config) == []


def test_reported_accuracy_below_the_floor() -> None:
    findings = _check(_frame(_MAJORITY, _MAJORITY), AuditConfig(reported_acc=0.3))
    assert findings[0].severity == "error"
    assert "reported test accuracy" in findings[0].message


def test_skips_without_labels() -> None:
    df = _frame(_MAJORITY, ["PE"] * 60).drop(columns=["y_true"])
    findings = check_metric_floor(prepare_frame(df, AuditConfig()), AuditConfig())
    assert findings[0].severity == "info"


def test_train_split_is_not_judged() -> None:
    findings = _check(_frame(_MAJORITY, ["CG"] * 60, split="train"))
    assert findings == []
