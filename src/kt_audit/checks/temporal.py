from __future__ import annotations

from typing import Any, cast

import pandas as pd

from ..config import AuditConfig
from ..schema import Finding
from ._common import examples_from_frame, fold_example, iter_folds


def check_temporal(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        findings.extend(_check_fold(fold, group, config))
    return findings


def _check_fold(fold: str, group: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    train = cast(pd.DataFrame, group[group["split"] == "train"])
    evaluation = cast(pd.DataFrame, group[group["split"].isin(["valid", "test"])])
    if train.empty or evaluation.empty:
        return []
    findings = _global_findings(fold, train, evaluation)
    findings.extend(_overlap_findings(fold, group, config))
    findings.extend(_violation_findings(fold, train, evaluation, config))
    return findings


def _global_findings(
    fold: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> list[Finding]:
    train_max: Any = train["timestamp"].max()
    eval_min: Any = evaluation["timestamp"].min()
    if train_max < eval_min:
        return []
    example = fold_example(fold, {"train_max": train_max, "eval_min": eval_min})
    return [
        Finding(
            "temporal_split",
            "error",
            "training events are not strictly before evaluation events",
            1,
            (example,),
        )
    ]


def _overlap_findings(
    fold: str,
    group: pd.DataFrame,
    config: AuditConfig,
) -> list[Finding]:
    split_counts = group.groupby("user_id")["split"].nunique()
    multi = cast(pd.Series, split_counts[split_counts > 1])
    if multi.empty:
        return []
    users = (
        group[group["user_id"].isin(multi.index.tolist())]
        .groupby("user_id")["split"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    examples = tuple(
        fold_example(fold, {"user_id": user, "splits": splits})
        for user, splits in users.head(config.max_examples).items()
    )
    return [
        Finding(
            "temporal_split",
            "warn",
            "users appear in more than one split",
            len(multi),
            examples,
        )
    ]


def _violation_findings(
    fold: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: AuditConfig,
) -> list[Finding]:
    train_max_by_user = train.groupby("user_id")["timestamp"].max()
    eval_min_by_user = evaluation.groupby("user_id")["timestamp"].min()
    merged = pd.DataFrame({"train_max": train_max_by_user, "eval_min": eval_min_by_user}).dropna()
    violations = cast(pd.DataFrame, merged[merged["train_max"] >= merged["eval_min"]])
    if violations.empty:
        return []
    examples = examples_from_frame(
        violations.reset_index(),
        ("user_id", "train_max", "eval_min"),
        config.max_examples,
        fold,
    )
    return [
        Finding(
            "temporal_split",
            "error",
            "a user's training events overlap or follow evaluation events",
            len(violations),
            examples,
        )
    ]
