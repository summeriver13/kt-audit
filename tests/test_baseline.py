from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from kt_audit import AuditConfig
from kt_audit.checks.baseline import _features, check_cheap_baseline
from kt_audit.schema import prepare_frame


def _dataset(
    predict: Callable[[int, int], float],
    users: int = 40,
    events: int = 20,
    train_events: int = 15,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(5)
    ability = rng.normal(0.0, 1.0, size=users)
    difficulty = rng.normal(0.0, 1.0, size=10)
    rows: list[dict[str, object]] = []
    for user in range(users):
        for index in range(events):
            item = index % 10
            probability = 1.0 / (1.0 + np.exp(-1.5 * (ability[user] - difficulty[item])))
            label = int(rng.random() < probability)
            rows.append(
                {
                    "user_id": f"u{user:02d}",
                    "item_id": f"q{item:02d}",
                    "kc_ids": f"kc_{item % 3}",
                    "timestamp": f"2024-01-{1 + index // 8:02d}T{index % 24:02d}:00:00",
                    "split": "train" if index < train_events else "test",
                    "y_true": label,
                    "y_pred": predict(label, index),
                }
            )
    return rows


def _metric(baselines: dict[str, dict[str, object]], key: str) -> float:
    return float(cast(float, baselines["__all__"][key]))


def _check(make_frame, rows: list[dict[str, object]], config: AuditConfig):
    config = config or AuditConfig()
    return check_cheap_baseline(prepare_frame(make_frame(rows), config), config)


def test_flags_predictions_that_a_cheap_baseline_beats(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5 + (index % 5) * 0.001)
    findings, baselines = _check(make_frame, rows, AuditConfig())
    assert any(
        finding.message == "reported predictions do not beat a cheap history baseline"
        for finding in findings
    )
    assert _metric(baselines, "baseline_auc") > _metric(baselines, "model_auc")


def test_passes_when_predictions_are_strong(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.1 + 0.8 * label)
    findings, baselines = _check(make_frame, rows, AuditConfig())
    assert findings == []
    assert _metric(baselines, "delta") > 0.1


def test_skips_when_training_split_is_small(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5, users=6, events=8, train_events=5)
    findings, baselines = _check(make_frame, rows, AuditConfig())
    assert findings[0].severity == "info"
    assert "training split is too small" in findings[0].message
    assert baselines == {}


def test_falls_back_to_reported_auc(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5 + (index % 5) * 0.001)
    frame = make_frame(rows).drop(columns=["y_pred"])
    config = AuditConfig(reported_auc=0.5)
    findings, baselines = check_cheap_baseline(prepare_frame(frame, config), config)
    assert baselines["__all__"]["model_auc"] == 0.5
    assert any(finding.severity == "error" for finding in findings)


def test_skips_without_labels(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5)
    frame = make_frame(rows).drop(columns=["y_true", "y_pred"])
    findings, baselines = _check(make_frame, frame.to_dict("records"), AuditConfig())
    assert findings[0].message == "y_true is missing; baseline check skipped"
    assert baselines == {}


def test_features_use_only_past_labels(make_frame) -> None:
    rows = [
        {
            "user_id": "u1",
            "item_id": f"q{index}",
            "kc_ids": "kc_a",
            "timestamp": f"2024-01-0{index + 1}T00:00:00",
            "split": "train",
            "y_true": [1, 0, 0, 1, 1][index],
        }
        for index in range(5)
    ]
    features = _features(prepare_frame(make_frame(rows), AuditConfig()))
    assert features["prior_n"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert features["prior_acc"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1 / 3, 0.5])


def test_later_labels_do_not_change_earlier_features(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5)
    before = _features(prepare_frame(make_frame(rows), AuditConfig()))
    rows[-1]["y_true"] = 1 - int(cast(int, rows[-1]["y_true"]))
    after = _features(prepare_frame(make_frame(rows), AuditConfig()))
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_runs_per_fold(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5 + (index % 5) * 0.001)
    frame = make_frame(rows)
    frame["fold"] = frame["user_id"].str[-1].astype(int) % 2
    config = AuditConfig()
    _, baselines = check_cheap_baseline(prepare_frame(frame, config), config)
    assert set(baselines) == {"0", "1"}


def test_runs_without_item_id(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.5 + (index % 5) * 0.001)
    frame = make_frame(rows).drop(columns=["item_id"])
    config = AuditConfig()
    _, baselines = check_cheap_baseline(prepare_frame(frame, config), config)
    payload = baselines["__all__"]
    assert 0.0 <= _metric(baselines, "baseline_auc") <= 1.0
    assert payload["n_train"] and payload["n_evaluation"]


def test_skips_oversized_input(make_frame, monkeypatch: pytest.MonkeyPatch) -> None:
    from kt_audit.checks import baseline

    monkeypatch.setattr(baseline, "MAX_ROWS", 10)
    rows = _dataset(lambda label, index: 0.5)
    findings, baselines = _check(make_frame, rows, AuditConfig())
    assert "more than 10 rows" in findings[0].message
    assert baselines == {}


def test_reported_auc_without_predictions_labels_frame(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.1 + 0.8 * label)
    frame = make_frame(rows).drop(columns=["y_pred"])
    config = AuditConfig(reported_auc=0.9)
    findings, baselines = check_cheap_baseline(prepare_frame(frame, config), config)
    assert findings == []
    assert baselines["__all__"]["model_auc"] == 0.9


def test_skips_when_training_split_has_one_class(make_frame) -> None:
    rows = _dataset(lambda label, index: 0.1 + 0.8 * label)
    for row in rows:
        if row["split"] == "train":
            row["y_true"] = 0
    findings, baselines = _check(make_frame, rows, AuditConfig())
    assert findings[0].severity == "info"
    assert "training split has a single class" in findings[0].message
    assert baselines == {}
