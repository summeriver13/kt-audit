from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ..config import AuditConfig
from ..metrics import split_metrics
from ..schema import Finding
from ..task import LABEL_COLUMN, PREDICTED_COLUMN, TASK_COLUMN, Task
from ._common import examples_with_fold, fold_example, iter_folds, key_columns


def check_alignment(
    df: pd.DataFrame,
    config: AuditConfig,
) -> tuple[list[Finding], dict[str, dict[str, object]]]:
    if "y_pred" not in df.columns and LABEL_COLUMN not in df.columns:
        return (
            [
                Finding(
                    "alignment",
                    "info",
                    "y_pred is missing; metric checks skipped",
                    0,
                    (),
                )
            ],
            {},
        )

    task = _task(df)
    findings: list[Finding] = []
    findings.extend(_invalid_findings(df, config, task))
    findings.extend(_duplicate_findings(df, config))
    findings.extend(_constant_findings(df))
    metrics = split_metrics(_metric_rows(df, task), task)
    findings.extend(_reported_findings(metrics, config))
    return findings, metrics


def _task(df: pd.DataFrame) -> Task:
    return cast(Task, str(df[TASK_COLUMN].iloc[0]))


def _metric_rows(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    if task != "ranking" or "y_pred" not in df.columns:
        return df
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    pred = valid["y_pred"].astype("float64")
    true = valid["y_true"].astype("float64")
    keep = np.isfinite(pred) & pred.between(0, 1) & true.isin([0.0, 1.0])
    return cast(pd.DataFrame, valid.loc[keep])


def _invalid_findings(
    df: pd.DataFrame,
    config: AuditConfig,
    task: Task,
) -> list[Finding]:
    if task == "classification":
        return []
    pred = df["y_pred"].astype("float64")
    true = df["y_true"].astype("float64")
    invalid_pred = ~np.isfinite(pred) | (pred < 0) | (pred > 1)
    invalid_true = ~true.isin([0.0, 1.0])
    findings: list[Finding] = []
    if bool(invalid_pred.any()):
        examples = examples_with_fold(
            cast(pd.DataFrame, df.loc[invalid_pred]),
            ("_row", "user_id", "item_id", "y_pred"),
            config.max_examples,
        )
        findings.append(
            Finding(
                "alignment",
                "error",
                "y_pred has missing, non-finite, or out-of-range values",
                int(np.count_nonzero(invalid_pred)),
                examples,
            )
        )
    if bool(invalid_true.any()):
        examples = examples_with_fold(
            cast(pd.DataFrame, df.loc[invalid_true]),
            ("_row", "user_id", "item_id", "y_true"),
            config.max_examples,
        )
        findings.append(
            Finding(
                "alignment",
                "error",
                "y_true is not binary",
                int(np.count_nonzero(invalid_true.to_numpy())),
                examples,
            )
        )
    return findings


def _duplicate_findings(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    keys = key_columns(df)
    extra = df.duplicated(subset=keys, keep="first")
    if not bool(extra.any()):
        return []
    examples = examples_with_fold(
        cast(pd.DataFrame, df.loc[df.duplicated(subset=keys, keep=False)]),
        ("_row", "user_id", "item_id", "timestamp"),
        config.max_examples,
    )
    return [
        Finding(
            "alignment",
            "error",
            "duplicate interaction keys",
            int(np.count_nonzero(extra.to_numpy())),
            examples,
        )
    ]


def _constant_findings(df: pd.DataFrame) -> list[Finding]:
    column = "y_pred" if "y_pred" in df.columns else PREDICTED_COLUMN
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        if cast(pd.Series, group[column]).nunique(dropna=True) <= 1:
            findings.append(
                Finding(
                    "alignment",
                    "warn",
                    "predictions are constant within this fold",
                    1,
                    (fold_example(fold, {"n": len(group)}),),
                )
            )
    return findings


def _reported_findings(
    metrics: dict[str, dict[str, object]],
    config: AuditConfig,
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_reported_auc_findings(metrics, config))
    findings.extend(_reported_acc_findings(metrics, config))
    return findings


def _reported_auc_findings(
    metrics: dict[str, dict[str, object]],
    config: AuditConfig,
) -> list[Finding]:
    reported = config.reported_auc
    if reported is None:
        return []
    test_auc = metrics.get("test", {}).get("auc")
    if not isinstance(test_auc, (int, float)):
        return [
            Finding(
                "alignment",
                "warn",
                "reported_auc is set but test AUC cannot be computed",
                0,
                (),
            )
        ]
    delta = abs(float(test_auc) - reported)
    if delta <= config.auc_tolerance:
        return []
    return [
        Finding(
            "alignment",
            "error",
            "reported test AUC differs from recomputed AUC",
            1,
            (
                {
                    "reported_auc": reported,
                    "recomputed_auc": float(test_auc),
                    "delta": delta,
                },
            ),
        )
    ]


def _reported_acc_findings(
    metrics: dict[str, dict[str, object]],
    config: AuditConfig,
) -> list[Finding]:
    reported = config.reported_acc
    if reported is None:
        return []
    test_acc = metrics.get("test", {}).get("acc")
    if not isinstance(test_acc, (int, float)):
        return [
            Finding(
                "alignment",
                "warn",
                "reported_acc is set but test accuracy cannot be computed",
                0,
                (),
            )
        ]
    delta = abs(float(test_acc) - reported)
    if delta <= config.auc_tolerance:
        return []
    return [
        Finding(
            "alignment",
            "error",
            "reported test accuracy differs from recomputed accuracy",
            1,
            (
                {
                    "reported_acc": reported,
                    "recomputed_acc": float(test_acc),
                    "delta": delta,
                },
            ),
        )
    ]
