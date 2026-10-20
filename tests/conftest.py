from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest


@pytest.fixture
def make_frame() -> Callable[[list[dict[str, object]]], pd.DataFrame]:
    def _make(rows: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def clean_rows() -> list[dict[str, object]]:
    return [
        {
            "user_id": "u1",
            "item_id": "q1",
            "kc_ids": "kc_a",
            "timestamp": "2024-01-01T10:00:00",
            "split": "train",
            "y_true": 1,
            "y_pred": 0.8,
            "fold": 0,
        },
        {
            "user_id": "u2",
            "item_id": "q2",
            "kc_ids": "kc_b",
            "timestamp": "2024-01-01T10:05:00",
            "split": "train",
            "y_true": 0,
            "y_pred": 0.2,
            "fold": 0,
        },
        {
            "user_id": "u3",
            "item_id": "q3",
            "kc_ids": "kc_c",
            "timestamp": "2024-01-02T09:00:00",
            "split": "test",
            "y_true": 1,
            "y_pred": 0.7,
            "fold": 0,
        },
        {
            "user_id": "u4",
            "item_id": "q4",
            "kc_ids": "kc_d",
            "timestamp": "2024-01-02T09:05:00",
            "split": "test",
            "y_true": 0,
            "y_pred": 0.3,
            "fold": 0,
        },
    ]
