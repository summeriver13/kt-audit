from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

from .config import AuditConfig
from .errors import AuditError

Severity = Literal["error", "warn", "info"]
Verdict = Literal["pass", "warn", "fail"]
SPLITS: tuple[str, ...] = ("train", "valid", "test")
_SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    message: str
    count: int = 0
    examples: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class Report:
    schema_version: str
    generated_at: str
    input: dict[str, object]
    findings: tuple[Finding, ...]
    metrics: dict[str, dict[str, object]]
    verdict: Verdict


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_split(value: object) -> str:
    if _is_missing(value):
        raise AuditError("split is missing")
    text = str(value).strip().lower()
    if text not in _SPLIT_ALIASES:
        raise AuditError(f"invalid split: {value!r}")
    return _SPLIT_ALIASES[text]


def parse_kc_ids(value: object, separator: str) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid kc_ids JSON: {value!r}") from exc
            if not isinstance(parsed, list):
                raise AuditError(f"kc_ids JSON must be a list: {value!r}")
            values = parsed
        else:
            values = text.split(separator)
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif _is_missing(value):
        raise AuditError("kc_ids is missing")
    else:
        raise AuditError(f"kc_ids must be a string or list: {value!r}")
    cleaned = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not cleaned:
        raise AuditError(f"kc_ids is empty: {value!r}")
    return cleaned


def jsonable(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return cast(pd.Timestamp, value).isoformat()
    if isinstance(value, np.datetime64):
        return cast(pd.Timestamp, pd.Timestamp(value)).isoformat()
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if _is_missing(value):
        return None
    return value


def example_from_row(row: pd.Series, columns: tuple[str, ...]) -> dict[str, object]:
    return {column: jsonable(row[column]) for column in columns if column in row.index}


def prepare_frame(df: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    if df.empty:
        raise AuditError("input has no rows")
    missing = [column for column in ("user_id", "timestamp", "split", "kc_ids") if column not in df]
    if missing:
        raise AuditError(f"missing required columns: {', '.join(missing)}")

    out = df.copy().reset_index(drop=True)
    out["_row"] = np.arange(1, len(out) + 1)
    out["user_id"] = _to_string(_series(out, "user_id"), "user_id", required=True)
    out["split"] = _map_rows(_series(out, "split"), normalize_split, "split")
    out["kc_ids"] = _map_rows(
        _series(out, "kc_ids"),
        lambda value: parse_kc_ids(value, config.kc_separator),
        "kc_ids",
    )
    out["timestamp"] = _parse_timestamps(_series(out, "timestamp"))

    if "item_id" in out.columns:
        out["item_id"] = _to_string(_series(out, "item_id"), "item_id", required=False)
    else:
        out["item_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "fold" in out.columns:
        out["_fold"] = _to_string(_series(out, "fold"), "fold", required=False).fillna("__all__")
    else:
        out["_fold"] = "__all__"

    if "sequence_id" in out.columns:
        out["sequence_id"] = _to_string(
            _series(out, "sequence_id"),
            "sequence_id",
            required=False,
        )
    if "position" in out.columns:
        out["position"] = pd.to_numeric(_series(out, "position"), errors="coerce")

    has_pred = "y_pred" in out.columns
    has_true = "y_true" in out.columns
    if has_pred and not has_true:
        raise AuditError("y_pred requires y_true")
    if has_true:
        out["y_true"] = pd.to_numeric(_series(out, "y_true"), errors="coerce")
    if has_pred:
        out["y_pred"] = pd.to_numeric(_series(out, "y_pred"), errors="coerce")
    return out


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


def _to_string(series: pd.Series, name: str, *, required: bool) -> pd.Series:
    if required and bool(series.isna().any()):
        raise AuditError(f"{name} contains missing values")
    return series.astype("string")


def _map_rows(
    series: pd.Series,
    func: Callable[[object], object],
    name: str,
) -> pd.Series:
    values: list[object] = []
    for row_number, value in enumerate(series, start=1):
        try:
            values.append(func(value))
        except AuditError as exc:
            raise AuditError(f"row {row_number} column {name}: {exc}") from exc
    return pd.Series(values, index=series.index, dtype="object")


def _parse_timestamps(series: pd.Series) -> pd.Series:
    if bool(series.isna().any()):
        raise AuditError("timestamp contains missing values")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    try:
        parsed = pd.to_datetime(series, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid timestamp: {exc}") from exc
    if bool(parsed.isna().any()):
        raise AuditError("timestamp contains unparseable values")
    return parsed
