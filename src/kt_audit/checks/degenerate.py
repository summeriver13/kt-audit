from __future__ import annotations

from typing import cast

import pandas as pd

from ..config import AuditConfig
from ..schema import Finding
from ._common import examples_from_frame, fold_example, iter_folds

MIN_ROWS = 30
MIN_DISTINCT = 10
NARROW_IQR = 0.02
MIN_MINORITY_SHARE = 0.01
_EVALUATION_SPLITS = ("valid", "test")


def check_degenerate(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    if "y_pred" not in df.columns or not pd.api.types.is_numeric_dtype(df["y_pred"]):
        return [
            Finding(
                "degenerate",
                "info",
                "no numeric predictions; distribution checks skipped",
                0,
                (),
            )
        ]
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        findings.extend(_fold_findings(group, fold, config))
    return findings


def _fold_findings(df: pd.DataFrame, fold: str, config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []
    for split in _EVALUATION_SPLITS:
        rows = cast(pd.DataFrame, df[df["split"] == split])
        rows = cast(pd.DataFrame, rows[rows["y_pred"].notna()])
        if rows.empty:
            continue
        findings.extend(_distribution_findings(rows, split, fold, config))
    return findings


def _distribution_findings(
    rows: pd.DataFrame,
    split: str,
    fold: str,
    config: AuditConfig,
) -> list[Finding]:
    pred = cast(pd.Series, rows["y_pred"]).astype("float64")
    count = len(pred)
    distinct = int(pred.nunique())
    findings: list[Finding] = []
    if count < MIN_ROWS:
        findings.append(
            Finding(
                "degenerate",
                "info",
                f"{split} split has fewer than {MIN_ROWS} predictions; metrics are unstable",
                count,
                _examples(rows, fold, config),
            )
        )
    if count >= MIN_ROWS and 1 < distinct < MIN_DISTINCT:
        findings.append(
            Finding(
                "degenerate",
                "warn",
                f"{split} predictions have fewer than {MIN_DISTINCT} distinct values",
                distinct,
                _examples(rows, fold, config),
            )
        )
    spread = float(pred.quantile(0.75) - pred.quantile(0.25))
    if count >= MIN_ROWS and distinct > 1 and spread < NARROW_IQR:
        findings.append(
            Finding(
                "degenerate",
                "warn",
                f"{split} predictions are concentrated in a narrow band",
                count,
                _examples(rows, fold, config),
            )
        )
    findings.extend(_minority_findings(rows, split, fold, config))
    return findings


def _minority_findings(
    rows: pd.DataFrame,
    split: str,
    fold: str,
    config: AuditConfig,
) -> list[Finding]:
    if "y_true" not in rows.columns:
        return []
    true = cast(pd.Series, rows["y_true"]).dropna()
    if true.empty or true.nunique() < 2:
        return []
    share = float(true.value_counts(normalize=True).min())
    if share >= MIN_MINORITY_SHARE:
        return []
    return [
        Finding(
            "degenerate",
            "warn",
            f"{split} minority class is rarer than {MIN_MINORITY_SHARE:.0%}",
            int(true.value_counts().min()),
            (fold_example(fold, {"split": split, "minority_share": share}),),
        )
    ]


def _examples(
    rows: pd.DataFrame,
    fold: str,
    config: AuditConfig,
) -> tuple[dict[str, object], ...]:
    return examples_from_frame(
        rows,
        ("user_id", "item_id", "y_true", "y_pred"),
        config.max_examples,
        fold,
    )
