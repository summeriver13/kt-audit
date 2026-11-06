from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from .config import AuditConfig

Task = Literal["ranking", "classification"]

LABEL_COLUMN = "y_pred_label"
TASK_COLUMN = "_task"
PREDICTED_COLUMN = "_pred"
POSITIVE_THRESHOLD = 0.5
_BINARY_LABELS = frozenset({"0", "1"})


def label_text(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def resolve_task(df: pd.DataFrame, config: AuditConfig) -> Task:
    if config.task != "auto":
        return cast(Task, config.task)
    if LABEL_COLUMN in df.columns:
        return "classification"
    for column in ("y_true", "y_pred"):
        if column in df.columns and not pd.api.types.is_numeric_dtype(cast(pd.Series, df[column])):
            return "classification"
    return "ranking"


def predicted_labels(df: pd.DataFrame, task: Task) -> pd.Series:
    empty = pd.Series(pd.NA, index=df.index, dtype="object")
    if LABEL_COLUMN in df.columns:
        return cast(pd.Series, cast(pd.Series, df[LABEL_COLUMN]).map(_as_label))
    if "y_pred" not in df.columns:
        return empty
    pred = cast(pd.Series, df["y_pred"])
    if not pd.api.types.is_numeric_dtype(pred):
        return cast(pd.Series, pred.map(_as_label))
    if task == "ranking":
        return pred.astype("float64")
    if _binary_labels(df) <= _BINARY_LABELS and _binary_labels(df):
        return cast(pd.Series, pred.map(_threshold_or_missing))
    return empty


def _binary_labels(df: pd.DataFrame) -> set[str]:
    if "y_true" not in df.columns:
        return set()
    truth = cast(pd.Series, df["y_true"]).dropna()
    return {label_text(value) for value in truth}


def _as_label(value: object) -> object:
    if _is_missing(value):
        return pd.NA
    return label_text(value)


def _threshold_or_missing(value: object) -> object:
    if _is_missing(value):
        return pd.NA
    return "1" if float(cast(float, value)) >= POSITIVE_THRESHOLD else "0"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return bool(cast(bool, pd.isna(value)))
