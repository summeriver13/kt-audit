from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from .audit import read_table
from .checks._common import key_columns
from .config import AuditConfig
from .errors import AuditError
from .metrics import score_rows
from .schema import prepare_frame
from .task import LABEL_COLUMN, PREDICTED_COLUMN, TASK_COLUMN, Task, label_text

ROUNDS = 2000
SEED = 0
PERCENTILES = (2.5, 97.5)
METRICS = ("auc", "acc", "macro_f1", "kappa")


@dataclass(frozen=True)
class Comparison:
    n: int
    metric: str
    value_a: float
    value_b: float
    delta: float
    ci_low: float
    ci_high: float
    rounds: int
    seed: int
    verdict: str


def compare_frames(
    a: pd.DataFrame,
    b: pd.DataFrame,
    config: AuditConfig | None = None,
    *,
    rounds: int = ROUNDS,
    seed: int = SEED,
    metric: str = "auc",
) -> Comparison:
    if rounds < 1:
        raise AuditError("rounds must be >= 1")
    if metric not in METRICS:
        raise AuditError(f"unknown metric: {metric}; choose from {', '.join(METRICS)}")
    config = config or AuditConfig()
    left, right = _aligned(a, b, config)
    task = _task(left, right)
    if metric == "auc" and task != "ranking":
        raise AuditError("auc needs numeric labels; use --metric acc for a label file")
    truth, pred_a, pred_b = _filtered(left, right, task)
    value_a = _value(truth, pred_a, task, metric)
    value_b = _value(truth, pred_b, task, metric)
    deltas = _bootstrap(truth, pred_a, pred_b, task, metric, rounds, seed)
    ci_low = float(np.percentile(deltas, PERCENTILES[0]))
    ci_high = float(np.percentile(deltas, PERCENTILES[1]))
    verdict = "tie"
    if ci_low > 0:
        verdict = "b_better"
    elif ci_high < 0:
        verdict = "a_better"
    return Comparison(
        n=int(truth.size),
        metric=metric,
        value_a=value_a,
        value_b=value_b,
        delta=value_b - value_a,
        ci_low=ci_low,
        ci_high=ci_high,
        rounds=rounds,
        seed=seed,
        verdict=verdict,
    )


def compare_paths(
    a: str | Path,
    b: str | Path,
    config: AuditConfig | None = None,
    *,
    rounds: int = ROUNDS,
    seed: int = SEED,
    metric: str = "auc",
) -> Comparison:
    return compare_frames(
        read_table(a),
        read_table(b),
        config,
        rounds=rounds,
        seed=seed,
        metric=metric,
    )


def comparison_to_dict(comparison: Comparison) -> dict[str, object]:
    return {
        "n": comparison.n,
        "metric": comparison.metric,
        "value_a": comparison.value_a,
        "value_b": comparison.value_b,
        "delta": comparison.delta,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "rounds": comparison.rounds,
        "seed": comparison.seed,
        "verdict": comparison.verdict,
    }


def comparison_to_markdown(comparison: Comparison) -> str:
    metric = comparison.metric
    lines = [
        "# kt-audit comparison",
        "",
        f"**Verdict: {comparison.verdict.upper()}**",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| metric | {metric} |",
        f"| rows | {comparison.n} |",
        f"| {metric} a | {comparison.value_a:.4f} |",
        f"| {metric} b | {comparison.value_b:.4f} |",
        f"| delta (b - a) | {comparison.delta:+.4f} |",
        f"| 95% CI | [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] |",
        f"| bootstrap rounds | {comparison.rounds} |",
        f"| seed | {comparison.seed} |",
        "",
        "## Reading",
        "",
        "- A CI that contains 0 means the two inputs are not distinguishable on these rows.",
        "- The interval is paired: both inputs are scored on the same resampled rows.",
        "- A tie does not mean the models are equal; it means this sample cannot tell them apart.",
    ]
    return "\n".join(lines) + "\n"


def write_comparison(comparison: Comparison, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "comparison.json"
    markdown_path = directory / "comparison.md"
    payload = json.dumps(comparison_to_dict(comparison), indent=2, ensure_ascii=False)
    json_path.write_text(payload + "\n", encoding="utf-8")
    markdown_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def _task(left: pd.DataFrame, right: pd.DataFrame) -> Task:
    left_task = str(left[TASK_COLUMN].iloc[0])
    right_task = str(right[TASK_COLUMN].iloc[0])
    if left_task != right_task:
        raise AuditError("inputs disagree on the task; one is ranking, the other classification")
    return cast(Task, left_task)


def _aligned(
    a: pd.DataFrame,
    b: pd.DataFrame,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = prepare_frame(a, config)
    right = prepare_frame(b, config)
    for frame, name in ((left, "a"), (right, "b")):
        if "y_true" not in frame.columns:
            raise AuditError(f"input {name} needs y_true")
        if "y_pred" not in frame.columns and LABEL_COLUMN not in frame.columns:
            raise AuditError(f"input {name} needs y_pred or y_pred_label")
    keys = key_columns(left)
    if key_columns(right) != keys:
        raise AuditError("inputs do not share the same key columns")
    indexed_left = _indexed(left, keys)
    indexed_right = _indexed(right, keys)
    if set(indexed_left.index) != set(indexed_right.index):
        raise AuditError("inputs have different interaction keys")
    indexed_right = indexed_right.reindex(indexed_left.index)
    labels_left = _truth_text(indexed_left)
    labels_right = _truth_text(indexed_right)
    if bool(labels_left.isna().any()) or bool(labels_right.isna().any()):
        raise AuditError("y_true has missing values; compare needs labels on every row")
    if not labels_left.equals(labels_right):
        raise AuditError("inputs disagree on y_true for the same interactions")
    return indexed_left, indexed_right


def _truth_text(frame: pd.DataFrame) -> pd.Series:
    truth = cast(pd.Series, frame["y_true"])
    return cast(pd.Series, truth.map(_label_or_missing))


def _label_or_missing(value: object) -> object:
    if value is None or bool(cast(bool, pd.isna(value))):
        return pd.NA
    return label_text(value)


def _indexed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    values = df[keys].astype("string").agg("||".join, axis=1)
    index = pd.Index(values.to_numpy(), name="key")
    if bool(index.duplicated().any()):
        raise AuditError("duplicate interaction keys in input")
    return df.set_index(index)


def _filtered(
    left: pd.DataFrame,
    right: pd.DataFrame,
    task: Task,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_series = cast(pd.Series, left["y_true"])
    pred_a = cast(pd.Series, left[PREDICTED_COLUMN])
    pred_b = cast(pd.Series, right[PREDICTED_COLUMN])
    mask = cast(pd.Series, truth_series.notna() & pred_a.notna() & pred_b.notna())
    truth = _array(cast(pd.Series, truth_series[mask]), task)
    array_a = _array(cast(pd.Series, pred_a[mask]), task)
    array_b = _array(cast(pd.Series, pred_b[mask]), task)
    if truth.size == 0:
        raise AuditError("inputs have no rows with both predictions")
    if task == "ranking" and not (np.isfinite(truth).all() and np.isfinite(array_a).all()):
        raise AuditError("y_pred has missing values; compare needs a prediction on every row")
    return truth, array_a, array_b


def _array(series: pd.Series, task: Task) -> np.ndarray:
    if task == "ranking":
        return series.to_numpy(dtype="float64")
    return np.array([label_text(value) for value in series])


def _value(truth: np.ndarray, predicted: np.ndarray, task: Task, metric: str) -> float:
    scores = score_rows(truth, predicted, task)
    value = scores.get(metric)
    if value is None:
        if metric == "auc" and np.unique(truth).size < 2:
            raise AuditError("y_true has a single class; AUC is undefined")
        raise AuditError(f"{metric} is undefined on these rows")
    return float(value)


def _bootstrap(
    truth: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    task: Task,
    metric: str,
    rounds: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deltas = np.empty(rounds, dtype="float64")
    size = truth.size
    for index in range(rounds):
        sample = rng.integers(0, size, size)
        labels = truth[sample]
        first = score_rows(labels, pred_a[sample], task).get(metric)
        second = score_rows(labels, pred_b[sample], task).get(metric)
        deltas[index] = np.nan if first is None or second is None else second - first
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        raise AuditError("bootstrap produced no finite deltas")
    return finite
