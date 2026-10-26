from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from ..config import AuditConfig
from ..schema import Finding
from ._common import fold_example, iter_folds

MIN_TRAIN_ROWS = 50
MIN_EVALUATION_ROWS = 30
TREE_DEPTH = 4
RANDOM_STATE = 0
MAX_ROWS = 200_000
SECONDS_PER_HOUR = 3600.0
MISSING_GAP = -1.0
NEUTRAL_ACC = 0.5
FEATURE_NAMES = (
    "prior_n",
    "prior_acc",
    "prior_gap",
    "item_prior_n",
    "item_prior_acc",
    "item_popularity",
    "kc_prior_acc",
    "position",
)


def check_cheap_baseline(
    df: pd.DataFrame,
    config: AuditConfig,
) -> tuple[list[Finding], dict[str, dict[str, object]]]:
    if "y_true" not in df.columns:
        return (
            [
                Finding(
                    "cheap_baseline",
                    "info",
                    "y_true is missing; baseline check skipped",
                    0,
                    (),
                )
            ],
            {},
        )
    if len(df) > MAX_ROWS:
        return (
            [
                Finding(
                    "cheap_baseline",
                    "info",
                    f"input has more than {MAX_ROWS} rows; baseline check skipped",
                    len(df),
                    (),
                )
            ],
            {},
        )
    findings: list[Finding] = []
    baselines: dict[str, dict[str, object]] = {}
    for fold, group in iter_folds(df):
        fold_findings, payload = _fold(group, fold, config)
        findings.extend(fold_findings)
        if payload is not None:
            baselines[fold] = payload
    return findings, baselines


def _fold(
    df: pd.DataFrame,
    fold: str,
    config: AuditConfig,
) -> tuple[list[Finding], dict[str, object] | None]:
    labelled = cast(pd.DataFrame, df[df["y_true"].notna()])
    train = cast(pd.DataFrame, labelled[labelled["split"] == "train"])
    evaluation = _evaluation_rows(labelled)
    if len(train) < MIN_TRAIN_ROWS:
        return ([_skip(fold, "the training split is too small")], None)
    if len(evaluation) < MIN_EVALUATION_ROWS:
        return ([_skip(fold, "the evaluation split is too small")], None)
    if evaluation["y_true"].nunique() < 2:
        return ([_skip(fold, "the evaluation split has a single class")], None)
    features = _features(df)
    tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=RANDOM_STATE)
    labels = cast(pd.Series, train["y_true"]).astype(int)
    tree.fit(features.loc[train.index], labels)
    probabilities = np.asarray(tree.predict_proba(features.loc[evaluation.index]))[:, 1]
    truth = cast(pd.Series, evaluation["y_true"]).astype(int)
    baseline_auc = float(roc_auc_score(truth, probabilities))
    model_auc = _model_auc(evaluation, config)
    if model_auc is None:
        payload: dict[str, object] = {
            "baseline_auc": baseline_auc,
            "n_train": len(train),
            "n_evaluation": len(evaluation),
        }
        return (
            [
                Finding(
                    "cheap_baseline",
                    "info",
                    "no predictions or reported AUC to compare the baseline with",
                    len(evaluation),
                    (fold_example(fold, payload),),
                )
            ],
            payload,
        )
    payload = {
        "baseline_auc": baseline_auc,
        "model_auc": model_auc,
        "delta": model_auc - baseline_auc,
        "n_train": len(train),
        "n_evaluation": len(evaluation),
    }
    return (_delta_findings(fold, payload, config), payload)


def _evaluation_rows(labelled: pd.DataFrame) -> pd.DataFrame:
    test = cast(pd.DataFrame, labelled[labelled["split"] == "test"])
    if not test.empty:
        return test
    return cast(pd.DataFrame, labelled[labelled["split"] == "valid"])


def _delta_findings(
    fold: str,
    payload: dict[str, object],
    config: AuditConfig,
) -> list[Finding]:
    delta = float(cast(float, payload["delta"]))
    if delta <= 0:
        return [
            Finding(
                "cheap_baseline",
                "error",
                "reported predictions do not beat a cheap history baseline",
                int(cast(int, payload["n_evaluation"])),
                (fold_example(fold, payload),),
            )
        ]
    if delta < config.baseline_warn_margin:
        return [
            Finding(
                "cheap_baseline",
                "warn",
                "reported predictions barely beat a cheap history baseline",
                int(cast(int, payload["n_evaluation"])),
                (fold_example(fold, payload),),
            )
        ]
    return []


def _skip(fold: str, reason: str) -> Finding:
    return Finding(
        "cheap_baseline", "info", f"baseline skipped: {reason}", 0, (fold_example(fold, {}),)
    )


def _model_auc(rows: pd.DataFrame, config: AuditConfig) -> float | None:
    if "y_pred" in rows.columns:
        pred = cast(pd.Series, rows["y_pred"]).astype("float64")
        mask = np.isfinite(pred.to_numpy()) & pred.between(0.0, 1.0).to_numpy()
        valid = cast(pd.DataFrame, rows.loc[mask])
        if len(valid) >= MIN_EVALUATION_ROWS and valid["y_true"].nunique() == 2:
            scores = roc_auc_score(valid["y_true"].astype(int), valid["y_pred"].astype("float64"))
            return float(scores)
    return config.reported_auc


def _features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values("timestamp", kind="stable")
    y = cast(pd.Series, work["y_true"]).astype("float64")
    user = cast(pd.Series, work["user_id"])
    history = _user_history(user, y)
    item = _item_history(user, y, cast(pd.Series, work["item_id"]).astype("string"))
    features = pd.DataFrame(
        {
            "prior_n": history["n"],
            "prior_acc": history["acc"],
            "prior_gap": _gap(user, cast(pd.Series, work["timestamp"])),
            "item_prior_n": item["n"],
            "item_prior_acc": item["acc"],
            "item_popularity": item["popularity"],
            "kc_prior_acc": _kc_acc(work, y),
            "position": _position(work),
        },
        index=work.index,
    )
    return features.reindex(df.index).fillna(MISSING_GAP)


def _user_history(user: pd.Series, y: pd.Series) -> dict[str, pd.Series]:
    frame = pd.DataFrame({"u": user, "y": y.fillna(0.0), "l": y.notna().astype("float64")})
    grouped = frame.groupby("u", sort=False)
    return {
        "n": grouped.cumcount().astype("float64"),
        "acc": _running_acc(frame, ["u"]),
    }


def _item_history(
    user: pd.Series,
    y: pd.Series,
    item: pd.Series,
) -> dict[str, pd.Series]:
    frame = pd.DataFrame(
        {
            "u": user,
            "i": item.fillna("__missing__"),
            "y": y.fillna(0.0),
            "l": y.notna().astype("float64"),
        }
    )
    grouped = frame.groupby(["u", "i"], sort=False)
    return {
        "n": grouped.cumcount().astype("float64"),
        "acc": _running_acc(frame, ["u", "i"]),
        "popularity": frame.groupby("i", sort=False).cumcount().astype("float64"),
    }


def _running_acc(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    grouped = frame.groupby(keys, sort=False)
    prior_sum = cast(pd.Series, grouped["y"].cumsum()) - frame["y"]
    prior_count = cast(pd.Series, grouped["l"].cumsum()) - frame["l"]
    ratio = prior_sum / cast(pd.Series, prior_count).where(prior_count > 0)
    return cast(pd.Series, ratio).fillna(NEUTRAL_ACC)


def _gap(user: pd.Series, timestamp: pd.Series) -> pd.Series:
    numeric = _numeric_time(timestamp)
    gap = numeric.groupby(user, sort=False).diff()
    return gap.fillna(MISSING_GAP)


def _numeric_time(timestamp: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(timestamp):
        seconds = timestamp.astype("int64") / 1e9
        return seconds / SECONDS_PER_HOUR
    return timestamp.astype("float64")


def _kc_acc(work: pd.DataFrame, y: pd.Series) -> pd.Series:
    lengths = cast(pd.Series, work["kc_ids"]).map(len).to_numpy()
    rows = np.repeat(np.arange(len(work)), lengths)
    kcs = [kc for values in work["kc_ids"] for kc in values]
    values = y.to_numpy()[rows]
    frame = pd.DataFrame({"kc": kcs, "y": np.nan_to_num(values), "l": ~np.isnan(values)})
    frame["l"] = frame["l"].astype("float64")
    acc = cast(pd.Series, _running_acc(frame, ["kc"]))
    per_row = cast(pd.Series, acc.groupby(pd.Series(rows, index=frame.index)).mean())
    restored = cast(pd.Series, per_row.reindex(range(len(work)))).fillna(NEUTRAL_ACC)
    return cast(pd.Series, restored).set_axis(work.index)


def _position(work: pd.DataFrame) -> pd.Series:
    if "position" not in work.columns:
        return pd.Series(MISSING_GAP, index=work.index, dtype="float64")
    values = pd.to_numeric(cast(pd.Series, work["position"]), errors="coerce")
    return cast(pd.Series, values).fillna(MISSING_GAP).astype("float64")
