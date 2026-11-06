from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest

import kt_audit.metrics as metrics_module
from kt_audit.metrics import MIN_ROWS, bootstrap_intervals, evaluate, label_stats, split_metrics
from kt_audit.task import PREDICTED_COLUMN


def _frame(
    labels: Sequence[object], predictions: Sequence[object], split: str = "test"
) -> pd.DataFrame:
    rows = [
        {
            "user_id": f"u{index}",
            "item_id": f"q{index}",
            "kc_ids": "kc_a",
            "timestamp": f"2024-01-01T{index % 24:02d}:00:00",
            "split": split,
            "y_true": label,
            PREDICTED_COLUMN: prediction,
        }
        for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True))
    ]
    return pd.DataFrame(rows)


def test_evaluate_binary_scores() -> None:
    scores = evaluate(pd.Series([1.0, 0.0, 1.0, 0.0]), pd.Series([0.9, 0.1, 0.2, 0.8]), "ranking")
    assert scores["n"] == 4
    assert scores["acc"] == 0.5
    assert scores["auc"] == 0.75


def test_evaluate_multi_class_scores() -> None:
    truth = pd.Series(["A", "A", "B", "B"])
    predicted = pd.Series(["A", "B", "B", "B"])
    scores = evaluate(truth, predicted, "classification")
    assert scores["acc"] == 0.75
    assert 0.0 < float(cast(float, scores["macro_f1"])) < 1.0
    assert scores["auc"] is None


def test_evaluate_drops_rows_without_prediction() -> None:
    scores = evaluate(pd.Series(["A", "B"]), pd.Series(["A", None]), "classification")
    assert scores["n"] == 1


def test_label_stats_reports_the_majority() -> None:
    stats = label_stats(pd.Series(["PE", "PE", "CG", "RF", "PE"]))
    assert stats["majority_class"] == "PE"
    assert stats["majority_share"] == 0.6


def test_label_stats_without_labels() -> None:
    stats = label_stats(pd.Series([None, None], dtype="object"))
    assert stats["majority_class"] is None


def test_split_metrics_adds_intervals_above_min_rows() -> None:
    rng = np.random.default_rng(0)
    labels = ["A"] * 60 + ["B"] * 60
    predictions = [rng.choice(["A", "B"]) for _ in range(120)]
    metrics = split_metrics(_frame(labels, predictions), "classification", rounds=200)
    intervals = cast(dict[str, list[float]], metrics["test"]["intervals"])
    assert set(intervals) == {"acc", "macro_f1", "kappa"}
    low, high = intervals["acc"]
    assert low <= float(cast(float, metrics["test"]["acc"])) <= high


def test_split_metrics_skips_intervals_below_min_rows() -> None:
    metrics = split_metrics(_frame(["A", "B"], ["A", "B"]), "classification", rounds=50)
    assert metrics["test"]["intervals"] == {}
    assert metrics["test"]["n"] == 2
    assert int(cast(int, metrics["test"]["n"])) < MIN_ROWS


def test_intervals_are_deterministic() -> None:
    rng = np.random.default_rng(1)
    labels = ["A"] * 50 + ["B"] * 50
    predictions = [rng.choice(["A", "B"]) for _ in range(100)]
    frame = _frame(labels, predictions)
    first = bootstrap_intervals(frame, "classification", rounds=100, seed=7)
    second = bootstrap_intervals(frame, "classification", rounds=100, seed=7)
    assert first == second


def test_ranking_metrics_keep_auc() -> None:
    labels = [1.0, 0.0] * 60
    predictions = [0.9 if label else 0.1 for label in labels]
    metrics = split_metrics(_frame(labels, predictions), "ranking", rounds=200)
    assert metrics["test"]["auc"] == 1.0
    assert metrics["test"]["acc"] == 1.0
    intervals = cast(dict[str, list[float]], metrics["test"]["intervals"])
    assert "auc" in intervals


def test_evaluate_reads_each_label_once(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    original = metrics_module.label_text

    def spy(value: object) -> str:
        seen.append(value)
        return original(value)

    monkeypatch.setattr(metrics_module, "label_text", spy)
    truth = pd.Series(["A", "B"] * 50)
    predicted = pd.Series(["A", "C"] * 50)
    evaluate(truth, predicted, "classification")
    assert len(seen) == 200


def test_bootstrap_scores_without_reading_the_labels_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    original = metrics_module.label_text

    def spy(value: object) -> str:
        seen.append(value)
        return original(value)

    monkeypatch.setattr(metrics_module, "label_text", spy)
    labels = ["A", "B"] * 60
    predictions = ["A", "C"] * 60
    bootstrap_intervals(_frame(labels, predictions), "classification", rounds=50)
    assert len(seen) == 240


def test_ranking_leaves_auc_empty_for_labels_outside_zero_and_one() -> None:
    truth = pd.Series([0.0, 1.0, 2.0, 0.0])
    scores = evaluate(truth, pd.Series([0.2, 0.8, 0.5, 0.1]), "ranking")
    assert scores["auc"] is None
    assert scores["acc"] == 0.75
