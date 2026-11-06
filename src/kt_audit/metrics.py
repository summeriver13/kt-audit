from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from .schema import SPLITS
from .task import PREDICTED_COLUMN, Task, label_text

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0
PERCENTILES = (2.5, 97.5)
MIN_ROWS = 30
MIN_ROUNDS = 50
BINARY_THRESHOLD = 0.5
METRIC_NAMES = ("auc", "acc", "macro_f1", "kappa")


def evaluate(truth: pd.Series, predicted: pd.Series, task: Task) -> dict[str, object]:
    mask = cast(pd.Series, truth.notna() & predicted.notna())
    values = _arrays(cast(pd.Series, truth[mask]), cast(pd.Series, predicted[mask]), task)
    scores = score_rows(values[0], values[1], task)
    return {"n": int(values[0].size), **scores}


def score_rows(truth: np.ndarray, predicted: np.ndarray, task: Task) -> dict[str, float | None]:
    scores: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)
    if truth.size == 0:
        return scores
    if task == "ranking":
        labels = truth.astype("float64")
        probs = predicted.astype("float64")
        answers = (probs >= BINARY_THRESHOLD).astype("float64")
        if np.unique(labels).size > 1:
            scores["auc"] = float(roc_auc_score(labels, probs))
        truth_text = np.array([label_text(value) for value in labels])
        pred_text = np.array([label_text(value) for value in answers])
    else:
        truth_text = np.array([label_text(value) for value in truth])
        pred_text = np.array([label_text(value) for value in predicted])
    scores["acc"] = float(accuracy_score(truth_text, pred_text))
    if np.unique(truth_text).size > 1:
        scores["macro_f1"] = float(
            f1_score(truth_text, pred_text, average="macro", zero_division=cast(str, 0))
        )
        scores["kappa"] = float(cohen_kappa_score(truth_text, pred_text))
    return scores


def label_stats(truth: pd.Series) -> dict[str, object]:
    labels = cast(pd.Series, truth.dropna().map(label_text))
    if labels.empty:
        return {"majority_class": None, "majority_share": None, "majority_acc": None}
    shares = labels.value_counts(normalize=True)
    share = float(shares.iloc[0])
    return {
        "majority_class": str(shares.index[0]),
        "majority_share": share,
        "majority_acc": share,
    }


def split_metrics(
    df: pd.DataFrame,
    task: Task,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        part = cast(pd.DataFrame, df[df["split"] == split])
        if part.empty:
            continue
        scores = evaluate(
            cast(pd.Series, part["y_true"]),
            cast(pd.Series, part[PREDICTED_COLUMN]),
            task,
        )
        stats = label_stats(cast(pd.Series, part["y_true"]))
        intervals = (
            bootstrap_intervals(part, task, rounds=rounds, seed=seed)
            if int(cast(int, scores["n"])) >= MIN_ROWS
            else {}
        )
        result[split] = {**scores, **stats, "intervals": intervals}
    return result


def bootstrap_intervals(
    df: pd.DataFrame,
    task: Task,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    truth_column = cast(pd.Series, df["y_true"])
    pred_column = cast(pd.Series, df[PREDICTED_COLUMN])
    mask = cast(pd.Series, truth_column.notna() & pred_column.notna())
    truth, predicted = _arrays(
        cast(pd.Series, truth_column[mask]),
        cast(pd.Series, pred_column[mask]),
        task,
    )
    if truth.size == 0 or rounds < 1:
        return {}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(rounds):
        index = rng.integers(0, truth.size, truth.size)
        for name, value in score_rows(truth[index], predicted[index], task).items():
            if value is not None:
                collected[name].append(value)
    intervals: dict[str, list[float]] = {}
    for name, values in collected.items():
        if len(values) < MIN_ROUNDS:
            continue
        low, high = np.percentile(values, PERCENTILES)
        intervals[name] = [float(low), float(high)]
    return intervals


def _arrays(
    truth: pd.Series,
    predicted: pd.Series,
    task: Task,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "ranking":
        labels = truth.astype("float64").to_numpy()
        probs = predicted.astype("float64").to_numpy()
        finite = np.isfinite(labels) & np.isfinite(probs)
        return labels[finite], probs[finite]
    labels = np.array([label_text(value) for value in truth])
    preds = np.array([label_text(value) for value in predicted])
    return labels, preds
