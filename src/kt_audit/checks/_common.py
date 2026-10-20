from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from ..schema import jsonable


def iter_folds(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    for fold, group in df.groupby("_fold", sort=True, dropna=False):
        yield str(fold), group


def fold_example(fold: str, example: dict[str, Any]) -> dict[str, object]:
    values = {key: jsonable(value) for key, value in example.items()}
    if fold != "__all__":
        return {"fold": fold, **values}
    return values


def examples_with_fold(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    max_examples: int,
) -> tuple[dict[str, object], ...]:
    examples = []
    for _, row in frame.head(max_examples).iterrows():
        values = {column: row[column] for column in columns if column in row.index}
        examples.append(fold_example(str(row["_fold"]), values))
    return tuple(examples)


def examples_from_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    max_examples: int,
    fold: str,
) -> tuple[dict[str, object], ...]:
    examples = []
    for _, row in frame.head(max_examples).iterrows():
        values = {column: row[column] for column in columns if column in row.index}
        examples.append(fold_example(fold, values))
    return tuple(examples)
