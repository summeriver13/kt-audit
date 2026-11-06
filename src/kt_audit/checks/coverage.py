from __future__ import annotations

from typing import cast

import pandas as pd

from ..config import AuditConfig
from ..schema import Finding
from ..task import PREDICTED_COLUMN, label_text

KEY_COLUMNS = ("user_id", "item_id", "timestamp")
ERROR_PREFIX = "ERROR"


def check_coverage(
    df: pd.DataFrame,
    config: AuditConfig,
    expect: pd.DataFrame | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_count_findings(len(df), config))
    if expect is not None:
        findings.extend(_expect_findings(df, expect, config))
    findings.extend(_label_findings(df, config))
    return findings


def _count_findings(rows: int, config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []
    if config.reported_n is not None and rows != config.reported_n:
        findings.append(
            Finding(
                "coverage",
                "error",
                "the input holds a different number of rows than the report claims",
                abs(rows - config.reported_n),
                ({"reported_n": config.reported_n, "input_rows": rows},),
            )
        )
    if config.expect_rows is not None and rows != config.expect_rows:
        findings.append(
            Finding(
                "coverage",
                "warn",
                "the input holds a different number of rows than expected",
                abs(rows - config.expect_rows),
                ({"expected_rows": config.expect_rows, "input_rows": rows},),
            )
        )
    return findings


def _expect_findings(
    df: pd.DataFrame,
    expect: pd.DataFrame,
    config: AuditConfig,
) -> list[Finding]:
    columns = [column for column in KEY_COLUMNS if column in df.columns and column in expect]
    if "user_id" not in columns:
        return [
            Finding(
                "coverage",
                "warn",
                "the expected set shares no key column with the input",
                0,
                (),
            )
        ]
    actual = _keys(df, columns)
    wanted = _keys(expect, columns)
    findings: list[Finding] = []
    missing = sorted(wanted - actual)
    if missing:
        findings.append(
            Finding(
                "coverage",
                "error",
                "expected keys are missing from the input",
                len(missing),
                tuple(_example(columns, key) for key in missing[: config.max_examples]),
            )
        )
    extra = sorted(actual - wanted)
    if extra:
        findings.append(
            Finding(
                "coverage",
                "warn",
                "the input holds keys that the expected set does not",
                len(extra),
                tuple(_example(columns, key) for key in extra[: config.max_examples]),
            )
        )
    return findings


def _label_findings(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    if PREDICTED_COLUMN not in df.columns:
        return []
    predicted = cast(pd.Series, df[PREDICTED_COLUMN])
    if predicted.empty or pd.api.types.is_numeric_dtype(predicted):
        return []
    texts = cast(pd.Series, predicted.map(_text))
    findings: list[Finding] = []
    blank = cast(pd.Series, texts.isna())
    if bool(blank.any()):
        findings.append(
            Finding(
                "coverage",
                "error",
                "the predicted label is missing",
                int(blank.sum()),
                _row_examples(df, blank, config),
            )
        )
    failed = cast(pd.Series, texts.fillna("").str.upper().str.startswith(ERROR_PREFIX))
    if bool(failed.any()):
        findings.append(
            Finding(
                "coverage",
                "error",
                "the predicted label holds an error placeholder",
                int(failed.sum()),
                _row_examples(df, failed, config),
            )
        )
    return findings


def _row_examples(
    df: pd.DataFrame,
    mask: pd.Series,
    config: AuditConfig,
) -> tuple[dict[str, object], ...]:
    rows = cast(pd.DataFrame, df.loc[mask])
    return tuple(
        {"user_id": str(row["user_id"]), "item_id": str(row["item_id"])}
        for _, row in rows.head(config.max_examples).iterrows()
    )


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    parts = []
    for column in columns:
        series = cast(pd.Series, frame[column])
        parts.append(_timestamp_text(series) if column == "timestamp" else series.astype("string"))
    table = pd.concat(parts, axis=1)
    table.columns = columns
    return {tuple(str(part) for part in row) for row in table.itertuples(index=False, name=None)}


def _timestamp_text(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return cast(pd.Series, series.map(_number_text))
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return cast(pd.Series, parsed.map(_iso_text))


def _number_text(value: object) -> str:
    return f"{float(cast(float, value)):g}"


def _iso_text(value: object) -> str:
    if value is None or bool(cast(bool, pd.isna(value))):
        return ""
    return cast(pd.Timestamp, value).isoformat()


def _text(value: object) -> object:
    if value is None or bool(cast(bool, pd.isna(value))):
        return pd.NA
    text = label_text(value)
    return text if text else pd.NA


def _example(columns: list[str], key: tuple[str, ...]) -> dict[str, object]:
    return {column: value for column, value in zip(columns, key, strict=False)}
