from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ..config import AuditConfig
from ..schema import Finding
from ._common import fold_example, iter_folds


def check_multikc(df: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []
    for fold, group in iter_folds(df):
        findings.extend(_check_fold(fold, group, config))
    return findings


def _check_fold(fold: str, group: pd.DataFrame, config: AuditConfig) -> list[Finding]:
    if bool(group["item_id"].isna().all()):
        return [
            Finding(
                "multi_kc",
                "info",
                "item_id is missing; multi-KC leakage check skipped",
                0,
                (),
            )
        ]
    working = group.dropna(subset=["item_id"])
    findings = _missing_findings(group)
    if working.empty:
        return findings
    item_splits, item_kcs, kc_splits = _aggregate(working)
    overlap_items, cross_items = _leaks(item_splits, item_kcs, kc_splits)
    findings.extend(
        _leak_findings(
            overlap_items,
            cross_items,
            item_kcs,
            item_splits,
            kc_splits,
            fold,
            config,
        )
    )
    return findings


def _missing_findings(group: pd.DataFrame) -> list[Finding]:
    missing = int(np.count_nonzero(group["item_id"].isna().to_numpy()))
    if not missing:
        return []
    return [
        Finding(
            "multi_kc",
            "warn",
            "rows with missing item_id were skipped",
            missing,
            (),
        )
    ]


def _aggregate(working: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    item_splits = cast(
        pd.Series,
        working.groupby("item_id")["split"].agg(lambda values: tuple(sorted(set(values)))),
    )
    item_kcs = cast(pd.Series, working.groupby("item_id")["kc_ids"].agg(_union_kcs))
    exploded = working.explode("kc_ids")
    kc_splits = cast(
        pd.Series,
        exploded.groupby("kc_ids")["split"].agg(lambda values: frozenset(values)),
    )
    return item_splits, item_kcs, kc_splits


def _leaks(
    item_splits: pd.Series,
    item_kcs: pd.Series,
    kc_splits: pd.Series,
) -> tuple[list[str], list[str]]:
    overlap_items: list[str] = []
    cross_items: list[str] = []
    for item_id, kcs in item_kcs.items():
        splits = set(item_splits[item_id])
        if not {"train", "test"}.issubset(splits):
            continue
        overlap_items.append(str(item_id))
        train_only = [kc for kc in kcs if set(kc_splits[kc]) == {"train"}]
        test_only = [kc for kc in kcs if set(kc_splits[kc]) == {"test"}]
        if train_only and test_only:
            cross_items.append(str(item_id))
    return overlap_items, cross_items


def _leak_findings(
    overlap_items: list[str],
    cross_items: list[str],
    item_kcs: pd.Series,
    item_splits: pd.Series,
    kc_splits: pd.Series,
    fold: str,
    config: AuditConfig,
) -> list[Finding]:
    findings: list[Finding] = []
    if overlap_items:
        examples = tuple(
            _example(item, item_kcs, item_splits, kc_splits, fold)
            for item in overlap_items[: config.max_examples]
        )
        findings.append(
            Finding(
                "multi_kc",
                "warn",
                "the same item appears in train and test",
                len(overlap_items),
                examples,
            )
        )
    if cross_items:
        examples = tuple(
            _example(item, item_kcs, item_splits, kc_splits, fold)
            for item in cross_items[: config.max_examples]
        )
        findings.append(
            Finding(
                "multi_kc",
                "error",
                "an item has train-only and test-only knowledge components",
                len(cross_items),
                examples,
            )
        )
    return findings


def _union_kcs(values: pd.Series) -> tuple[str, ...]:
    return tuple(sorted({kc for value in values for kc in value}))


def _example(
    item_id: str,
    item_kcs: pd.Series,
    item_splits: pd.Series,
    kc_splits: pd.Series,
    fold: str,
) -> dict[str, object]:
    kcs = item_kcs[item_id]
    return fold_example(
        fold,
        {
            "item_id": item_id,
            "splits": list(item_splits[item_id]),
            "kc_ids": list(kcs),
            "kc_splits": {kc: sorted(kc_splits[kc]) for kc in kcs},
        },
    )
