from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from ..config import AuditConfig
from ..schema import SPLITS, Finding
from ._common import examples_with_fold, fold_example, iter_folds


def check_alignment(
    df: pd.DataFrame,
    config: AuditConfig,
) -> tuple[list[Finding], dict[str, dict[str, object]]]:
    if "y_pred" not in df.columns:
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

    findings: list[Finding] = []
    findings.extend(_invalid_findings(df, config))
    findings.extend(_duplicate_findings(df, config))
    findings.extend(_constant_findings(df))
    metrics = _metrics(df)
    findings.extend(_reported_findings(metrics, config))
    return findings, metrics


def _invalid_findings(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
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
    keys = _key_columns(df)
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
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        if group["y_pred"].nunique(dropna=True) <= 1:
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


def _key_columns(df: pd.DataFrame) -> list[str]:
    if bool(df["item_id"].notna().any()):
        return ["_fold", "user_id", "item_id", "timestamp"]
    if {"sequence_id", "position"}.issubset(df.columns):
        return ["_fold", "user_id", "sequence_id", "position"]
    return ["_fold", "user_id", "timestamp"]


def _metrics(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    pred = valid["y_pred"].astype("float64")
    true = valid["y_true"].astype("float64")
    mask = np.isfinite(pred) & pred.between(0, 1) & true.isin([0.0, 1.0])
    valid = cast(pd.DataFrame, valid.loc[mask])
    metrics: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        part = cast(pd.DataFrame, valid[valid["split"] == split])
        if part.empty:
            continue
        y = cast(pd.Series, part["y_true"]).astype(int)
        p = cast(pd.Series, part["y_pred"]).astype(float)
        auc = float(roc_auc_score(y, p)) if y.nunique() == 2 else None
        acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
        metrics[split] = {"auc": auc, "acc": acc, "n": len(part)}
    return metrics


def _reported_findings(
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
