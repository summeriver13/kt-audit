from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = {
    "user_id": "user",
    "item_id": "problem",
    "kc_id": "skill",
    "timestamp": "order",
    "correct": "correct",
    "position": "position",
}
MIN_INTERACTIONS = 5


def _load(path: Path, max_users: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "order_id",
            "user_id",
            "problem_id",
            "skill_id",
            "correct",
            "original",
            "position",
        ],
        low_memory=False,
        encoding="latin-1",
    )
    frame = frame.dropna(subset=["user_id", "problem_id", "skill_id", "correct"])
    frame = frame[frame["original"] == 1]
    frame = frame[frame["skill_id"].astype("string").str.strip() != ""]
    users = sorted(frame["user_id"].unique())[:max_users]
    return frame[frame["user_id"].isin(users)].copy()


def _split(frame: pd.DataFrame, test_fraction: float) -> pd.DataFrame:
    ordered = frame.sort_values(["user_id", "order_id"])
    sizes = ordered.groupby("user_id")["order_id"].transform("size")
    ranks = ordered.groupby("user_id").cumcount()
    cutoff = (sizes * (1.0 - test_fraction)).astype(int)
    ordered["split"] = (ranks >= cutoff).map({True: "test", False: "train"})
    ordered["size"] = sizes
    return ordered[ordered["size"] >= MIN_INTERACTIONS].drop(columns=["size"])


def convert(path: Path, out: Path, max_users: int, test_fraction: float) -> pd.DataFrame:
    frame = _split(_load(path, max_users), test_fraction)
    converted = pd.DataFrame(
        {
            "user_id": "u" + frame["user_id"].astype("int64").astype(str),
            "item_id": "p" + frame["problem_id"].astype("int64").astype(str),
            "kc_ids": "k" + frame["skill_id"].fillna(0).astype("int64").astype(str),
            "timestamp": frame["order_id"].astype("int64"),
            "split": frame["split"],
            "y_true": frame["correct"].astype("int64"),
            "position": frame["position"],
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(out, index=False)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ASSISTments 2009 to the kt-audit contract."
    )
    parser.add_argument("data", type=Path, help="skill_builder_data_corrected.csv")
    parser.add_argument("--out", type=Path, default=Path("assist09.csv"))
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    converted = convert(args.data, args.out, args.max_users, args.test_fraction)
    print(f"rows: {len(converted)}")
    print(f"users: {converted['user_id'].nunique()}")
    print(f"splits: {converted['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
