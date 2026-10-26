from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .audit import read_table
from .checks._common import key_columns
from .config import AuditConfig
from .errors import AuditError
from .schema import prepare_frame

ROUNDS = 2000
SEED = 0
PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Comparison:
    n: int
    auc_a: float
    auc_b: float
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
) -> Comparison:
    if rounds < 1:
        raise AuditError("rounds must be >= 1")
    config = config or AuditConfig()
    left, right = _aligned(a, b, config)
    y = left["y_true"].to_numpy(dtype=float)
    pred_a = left["y_pred"].to_numpy(dtype=float)
    pred_b = right["y_pred"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y, pred_a, pred_b = y[mask], pred_a[mask], pred_b[mask]
    if y.size == 0:
        raise AuditError("inputs have no rows with both predictions")
    if np.unique(y).size < 2:
        raise AuditError("y_true has a single class; AUC is undefined")
    auc_a = float(roc_auc_score(y, pred_a))
    auc_b = float(roc_auc_score(y, pred_b))
    deltas = _bootstrap(y, pred_a, pred_b, rounds, seed)
    ci_low = float(np.percentile(deltas, PERCENTILES[0]))
    ci_high = float(np.percentile(deltas, PERCENTILES[1]))
    verdict = "tie"
    if ci_low > 0:
        verdict = "b_better"
    elif ci_high < 0:
        verdict = "a_better"
    return Comparison(
        n=int(y.size),
        auc_a=auc_a,
        auc_b=auc_b,
        delta=auc_b - auc_a,
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
) -> Comparison:
    return compare_frames(read_table(a), read_table(b), config, rounds=rounds, seed=seed)


def comparison_to_dict(comparison: Comparison) -> dict[str, object]:
    return {
        "n": comparison.n,
        "auc_a": comparison.auc_a,
        "auc_b": comparison.auc_b,
        "delta": comparison.delta,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "rounds": comparison.rounds,
        "seed": comparison.seed,
        "verdict": comparison.verdict,
    }


def comparison_to_markdown(comparison: Comparison) -> str:
    lines = [
        "# kt-audit comparison",
        "",
        f"**Verdict: {comparison.verdict.upper()}**",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| rows | {comparison.n} |",
        f"| AUC a | {comparison.auc_a:.4f} |",
        f"| AUC b | {comparison.auc_b:.4f} |",
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


def _aligned(
    a: pd.DataFrame,
    b: pd.DataFrame,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = prepare_frame(a, config)
    right = prepare_frame(b, config)
    for frame, name in ((left, "a"), (right, "b")):
        if "y_true" not in frame.columns or "y_pred" not in frame.columns:
            raise AuditError(f"input {name} needs y_true and y_pred")
    keys = key_columns(left)
    if key_columns(right) != keys:
        raise AuditError("inputs do not share the same key columns")
    indexed_left = _indexed(left, keys)
    indexed_right = _indexed(right, keys)
    if set(indexed_left.index) != set(indexed_right.index):
        raise AuditError("inputs have different interaction keys")
    indexed_right = indexed_right.reindex(indexed_left.index)
    labels_left = indexed_left["y_true"].to_numpy(dtype=float)
    labels_right = indexed_right["y_true"].to_numpy(dtype=float)
    if not np.isfinite(labels_left).all() or not np.isfinite(labels_right).all():
        raise AuditError("y_true has missing values; compare needs labels on every row")
    if not np.array_equal(labels_left, labels_right):
        raise AuditError("inputs disagree on y_true for the same interactions")
    return indexed_left, indexed_right


def _indexed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    values = df[keys].astype("string").agg("||".join, axis=1)
    index = pd.Index(values.to_numpy(), name="key")
    if bool(index.duplicated().any()):
        raise AuditError("duplicate interaction keys in input")
    return df.set_index(index)


def _bootstrap(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    rounds: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deltas = np.empty(rounds, dtype="float64")
    size = y.size
    for index in range(rounds):
        sample = rng.integers(0, size, size)
        labels = y[sample]
        if np.unique(labels).size < 2:
            deltas[index] = np.nan
            continue
        deltas[index] = roc_auc_score(labels, pred_b[sample]) - roc_auc_score(
            labels, pred_a[sample]
        )
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        raise AuditError("bootstrap produced no finite deltas")
    return finite
