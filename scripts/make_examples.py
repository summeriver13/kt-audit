from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
USERS = 40
EVENTS = 30
TRAIN_EVENTS = 24
ITEMS = 20
DEADLINE = "2024-01-01"


def _abilities(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=USERS)


def _difficulties(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=ITEMS)


def _write(name: str, rows: list[dict[str, object]]) -> None:
    path = EXAMPLES / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def _row(
    user: int,
    index: int,
    item: int,
    label: int,
    prediction: float,
    force_split: str | None = None,
) -> dict[str, object]:
    return {
        "user_id": f"u{user:02d}",
        "item_id": f"q{item:02d}",
        "kc_ids": f"kc_{item % 5}",
        "timestamp": f"2024-01-{1 + index // 4:02d}T{index % 24:02d}:00:00",
        "split": force_split or ("train" if index < TRAIN_EVENTS else "test"),
        "y_true": label,
        "y_pred": round(float(prediction), 6),
    }


def floor_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(7)
    ability = _abilities(rng)
    difficulty = _difficulties(rng)
    rows: list[dict[str, object]] = []
    for user in range(USERS):
        for index in range(EVENTS):
            item = int(rng.integers(0, ITEMS))
            probability = 1.0 / (1.0 + np.exp(-1.5 * (ability[user] - difficulty[item])))
            label = int(rng.random() < probability)
            prediction = 0.5 + rng.normal(0.0, 0.02)
            rows.append(_row(user, index, item, label, prediction))
    return rows


def degenerate_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for index in range(120):
        label = int(index % 2)
        prediction = 0.5 + rng.normal(0.0, 0.002)
        rows.append(_row(index % 30, index, index % ITEMS, label, prediction, "test"))
    return rows


def compare_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(13)
    rows: list[dict[str, object]] = []
    for index in range(100):
        label = int(index % 2)
        weak = 0.5 + rng.normal(0.0, 0.05)
        strong = 0.2 + 0.6 * label + rng.normal(0.0, 0.01)
        row = _row(index % 30, index, index % ITEMS, label, weak)
        row["_strong"] = round(float(strong), 6)
        rows.append(row)
    return rows


def main() -> None:
    _write("floor.csv", floor_rows())
    _write("degenerate.csv", degenerate_rows())
    comparisons = compare_rows()
    _write(
        "compare_a.csv",
        [{key: value for key, value in row.items() if key != "_strong"} for row in comparisons],
    )
    _write(
        "compare_b.csv",
        [
            {key: value for key, value in row.items() if key != "_strong"}
            | {"y_pred": row["_strong"]}
            for row in comparisons
        ],
    )


if __name__ == "__main__":
    main()
