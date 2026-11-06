from __future__ import annotations

from typing import cast

import pandas as pd

from ..config import AuditConfig
from ..metrics import evaluate, label_stats
from ..schema import Finding
from ..task import PREDICTED_COLUMN, TASK_COLUMN, Task
from ._common import fold_example, iter_folds

_EVALUATION_SPLITS = ("valid", "test")


def check_metric_floor(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    if "y_true" not in df.columns:
        return [
            Finding(
                "metric_floor",
                "info",
                "y_true is missing; floor check skipped",
                0,
                (),
            )
        ]
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        for split in _EVALUATION_SPLITS:
            findings.extend(_split_findings(group, split, fold, config))
    findings.extend(_reported_findings(df, config))
    return findings


def _split_findings(
    df: pd.DataFrame,
    split: str,
    fold: str,
    config: AuditConfig,
) -> list[Finding]:
    rows = cast(pd.DataFrame, df[df["split"] == split])
    if rows.empty:
        return []
    task = cast(Task, str(df[TASK_COLUMN].iloc[0]))
    scores = evaluate(
        cast(pd.Series, rows["y_true"]),
        cast(pd.Series, rows[PREDICTED_COLUMN]),
        task,
    )
    stats = label_stats(cast(pd.Series, rows["y_true"]))
    share = stats["majority_share"]
    acc = scores["acc"]
    if share is None or not isinstance(acc, float):
        return []
    floor = float(cast(float, share))
    margin = acc - floor
    example = fold_example(
        fold,
        {
            "split": split,
            "accuracy": round(acc, 4),
            "majority_class": stats["majority_class"],
            "majority_share": round(floor, 4),
            "n": scores["n"],
        },
    )
    if margin < 0:
        return [
            Finding(
                "metric_floor",
                "error",
                f"{split} accuracy {acc:.3f} does not beat always answering "
                f"{stats['majority_class']} ({floor:.3f})",
                int(cast(int, scores["n"])),
                (example,),
            )
        ]
    if margin < config.floor_warn_margin:
        return [
            Finding(
                "metric_floor",
                "warn",
                f"{split} accuracy {acc:.3f} is within {config.floor_warn_margin:.2f} of "
                f"always answering {stats['majority_class']} ({floor:.3f})",
                int(cast(int, scores["n"])),
                (example,),
            )
        ]
    return []


def _reported_findings(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    reported = config.reported_acc
    if reported is None:
        return []
    test_rows = cast(pd.DataFrame, df[df["split"] == "test"])
    stats = label_stats(cast(pd.Series, test_rows["y_true"]))
    share = stats["majority_share"]
    if share is None or reported >= float(cast(float, share)):
        return []
    return [
        Finding(
            "metric_floor",
            "error",
            f"reported test accuracy {reported:.3f} does not beat always answering "
            f"{stats['majority_class']} ({float(cast(float, share)):.3f})",
            1,
            (
                {
                    "reported_acc": reported,
                    "majority_class": stats["majority_class"],
                    "majority_share": float(cast(float, share)),
                },
            ),
        )
    ]
