from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
_BINARY_LABELS = (0.0, 1.0)


@dataclass(frozen=True)
class Sample:
    """Rows encoded once, so a resample scores without reading any label twice."""

    truth_codes: np.ndarray
    answer_codes: np.ndarray
    truth_labels: np.ndarray | None = None
    scores: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.truth_codes.size)

    def take(self, index: np.ndarray) -> Sample:
        return Sample(
            truth_codes=self.truth_codes[index],
            answer_codes=self.answer_codes[index],
            truth_labels=None if self.truth_labels is None else self.truth_labels[index],
            scores=None if self.scores is None else self.scores[index],
        )

    def score(self, task: Task) -> dict[str, float | None]:
        result: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)
        if self.size == 0:
            return result
        result["acc"] = float(accuracy_score(self.truth_codes, self.answer_codes))
        if np.unique(self.truth_codes).size > 1:
            result["macro_f1"] = float(
                f1_score(
                    self.truth_codes,
                    self.answer_codes,
                    average="macro",
                    zero_division=cast(str, 0),
                )
            )
            result["kappa"] = float(cohen_kappa_score(self.truth_codes, self.answer_codes))
        result["auc"] = self._auc(task)
        return result

    def _auc(self, task: Task) -> float | None:
        if task != "ranking" or self.truth_labels is None or self.scores is None:
            return None
        if not _auc_ready(self.truth_labels):
            return None
        return float(roc_auc_score(self.truth_labels, self.scores))


def encode_labels(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Map several label arrays onto one integer code space."""
    sizes = [np.asarray(array).size for array in arrays]
    if not sizes or sum(sizes) == 0:
        return [np.zeros(0, dtype="int64") for _ in sizes]
    joined = np.concatenate([np.asarray(array) for array in arrays])
    codes, _ = pd.factorize(joined)
    bounds = np.cumsum([0, *sizes])
    return [codes[bounds[index] : bounds[index + 1]] for index in range(len(sizes))]


def sample_from_arrays(truth: np.ndarray, predicted: np.ndarray, task: Task) -> Sample:
    if task == "ranking":
        return _ranking_sample(truth, predicted)
    return _label_sample(truth, predicted)


def sample_from_series(truth: pd.Series, predicted: pd.Series, task: Task) -> Sample:
    mask = cast(pd.Series, truth.notna() & predicted.notna())
    return sample_from_arrays(
        cast(pd.Series, truth[mask]).to_numpy(),
        cast(pd.Series, predicted[mask]).to_numpy(),
        task,
    )


def sample_pair(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    task: Task,
) -> tuple[Sample, Sample]:
    """Two prediction sets of the same rows, so one resample scores both."""
    if task == "ranking":
        return _ranking_sample(truth, left), _ranking_sample(truth, right)
    truth_codes, left_codes, right_codes = encode_labels([truth, left, right])
    return Sample(truth_codes, left_codes), Sample(truth_codes, right_codes)


def evaluate(truth: pd.Series, predicted: pd.Series, task: Task) -> dict[str, object]:
    sample = sample_from_series(truth, predicted, task)
    return {"n": sample.size, **sample.score(task)}


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
    sample = sample_from_series(
        cast(pd.Series, df["y_true"]),
        cast(pd.Series, df[PREDICTED_COLUMN]),
        task,
    )
    if sample.size == 0 or rounds < 1:
        return {}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(rounds):
        index = rng.integers(0, sample.size, sample.size)
        for name, value in sample.take(index).score(task).items():
            if value is not None:
                collected[name].append(value)
    intervals: dict[str, list[float]] = {}
    for name, values in collected.items():
        if len(values) < MIN_ROUNDS:
            continue
        low, high = np.percentile(values, PERCENTILES)
        intervals[name] = [float(low), float(high)]
    return intervals


def _label_sample(truth: np.ndarray, predicted: np.ndarray) -> Sample:
    truth_text = np.array([label_text(value) for value in truth], dtype=object)
    answer_text = np.array([label_text(value) for value in predicted], dtype=object)
    truth_codes, answer_codes = encode_labels([truth_text, answer_text])
    return Sample(truth_codes, answer_codes)


def _ranking_sample(truth: np.ndarray, predicted: np.ndarray) -> Sample:
    labels = np.asarray(truth, dtype="float64")
    scores = np.asarray(predicted, dtype="float64")
    finite = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    answers = np.where(scores >= BINARY_THRESHOLD, 1.0, 0.0)
    truth_codes, answer_codes = encode_labels([labels, answers])
    return Sample(truth_codes, answer_codes, labels, scores)


def _auc_ready(labels: np.ndarray) -> bool:
    if labels.size == 0 or np.unique(labels).size < 2:
        return False
    return bool(np.isin(labels, _BINARY_LABELS).all())
