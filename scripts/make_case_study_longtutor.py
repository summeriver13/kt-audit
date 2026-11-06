"""Convert LongTutor gold labels and one saved annotation file into a kt-audit input."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path


def load(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(path: Path) -> dict[str, str]:
    labels = {}
    for row in load(path):
        label = row.get("diagnosis")
        if label:
            labels[str(row["_key"])] = str(label)
    return labels


def build_rows(gold: Path, predictions: Path) -> list[dict[str, object]]:
    labels = load_predictions(predictions)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(load(gold), start=1):
        key = str(row["_key"])
        user, _, digest = key.partition("||")
        rows.append(
            {
                "user_id": user,
                "item_id": digest[:12],
                "kc_ids": "diagnosis",
                "timestamp": index,
                "split": "test",
                "y_true": row["diagnosis"],
                "y_pred_label": labels.get(key, ""),
            }
        )
    return rows


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scored = sum(1 for row in rows if row["y_pred_label"])
    print(f"{path}: {len(rows)} rows, {scored} scored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="gold label JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="annotation JSONL")
    parser.add_argument("--out", type=Path, required=True, help="output CSV")
    args = parser.parse_args()
    write(args.out, build_rows(args.gold, args.predictions))


if __name__ == "__main__":
    main()
