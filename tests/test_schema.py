from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kt_audit import AuditConfig, AuditError
from kt_audit.schema import jsonable, parse_kc_ids, prepare_frame


def test_prepare_frame_normalizes_split_and_kcs(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "item_id": "q1",
                "kc_ids": '["kc_a", "kc_b"]',
                "timestamp": "2024-01-01",
                "split": "validation",
                "y_true": 1,
                "y_pred": 0.5,
            }
        ]
    )
    prepared = prepare_frame(df, AuditConfig())
    assert prepared.loc[0, "split"] == "valid"
    assert prepared.loc[0, "kc_ids"] == ("kc_a", "kc_b")
    assert prepared.loc[0, "_fold"] == "__all__"


def test_prepare_frame_requires_columns(make_frame) -> None:
    df = make_frame([{"user_id": "u1", "timestamp": "2024-01-01", "split": "train"}])
    with pytest.raises(AuditError, match="kc_ids"):
        prepare_frame(df, AuditConfig())


def test_prepare_frame_rejects_missing_user(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": None,
                "kc_ids": "kc_a",
                "timestamp": "2024-01-01",
                "split": "train",
            }
        ]
    )
    with pytest.raises(AuditError, match="user_id"):
        prepare_frame(df, AuditConfig())


def test_prepare_frame_rejects_invalid_split(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "kc_ids": "kc_a",
                "timestamp": "2024-01-01",
                "split": "dev",
            }
        ]
    )
    with pytest.raises(AuditError, match="row 1 column split"):
        prepare_frame(df, AuditConfig())


def test_prepare_frame_rejects_empty_kcs(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "kc_ids": "|",
                "timestamp": "2024-01-01",
                "split": "train",
            }
        ]
    )
    with pytest.raises(AuditError, match="kc_ids is empty"):
        prepare_frame(df, AuditConfig())


def test_prepare_frame_requires_y_true_with_y_pred(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "kc_ids": "kc_a",
                "timestamp": "2024-01-01",
                "split": "train",
                "y_pred": 0.5,
            }
        ]
    )
    with pytest.raises(AuditError, match="y_pred requires y_true"):
        prepare_frame(df, AuditConfig())


def test_prepare_frame_adds_item_id(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "kc_ids": "kc_a",
                "timestamp": "2024-01-01",
                "split": "train",
            }
        ]
    )
    prepared = prepare_frame(df, AuditConfig())
    assert bool(prepared["item_id"].isna().all())


def test_prepare_frame_keeps_numeric_timestamp(make_frame) -> None:
    df = make_frame(
        [
            {
                "user_id": "u1",
                "kc_ids": "kc_a",
                "timestamp": 100,
                "split": "train",
            }
        ]
    )
    prepared = prepare_frame(df, AuditConfig())
    assert prepared.loc[0, "timestamp"] == 100.0


def test_parse_kc_ids_dedupes() -> None:
    assert parse_kc_ids("kc_a|kc_a|kc_b", "|") == ("kc_a", "kc_b")


def test_parse_kc_ids_rejects_bad_json() -> None:
    with pytest.raises(AuditError, match="invalid kc_ids JSON"):
        parse_kc_ids("[kc_a", "|")


def test_parse_kc_ids_rejects_non_list_json() -> None:
    with pytest.raises(AuditError, match="must be a list"):
        parse_kc_ids('{"kc": "a"}', "|")


def test_jsonable_handles_numpy_and_timestamp() -> None:
    import numpy as np

    assert jsonable(np.int64(2)) == 2
    assert jsonable(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"


def test_load_config(tmp_path: Path) -> None:
    from kt_audit import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reported_auc": 0.8, "max_examples": 1}), encoding="utf-8")
    config = load_config(path)
    assert config.reported_auc == 0.8
    assert config.max_examples == 1


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    from kt_audit import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"unknown": True}), encoding="utf-8")
    with pytest.raises(AuditError, match="unknown config keys"):
        load_config(path)


def test_load_config_rejects_bad_json(tmp_path: Path) -> None:
    from kt_audit import load_config

    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(AuditError, match="invalid JSON"):
        load_config(path)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="kc_separator"):
        AuditConfig(kc_separator="")
    with pytest.raises(ValueError, match="max_examples"):
        AuditConfig(max_examples=0)
    with pytest.raises(ValueError, match="auc_tolerance"):
        AuditConfig(auc_tolerance=-0.1)
    with pytest.raises(ValueError, match="reported_auc"):
        AuditConfig(reported_auc=1.1)


def test_load_config_defaults() -> None:
    from kt_audit import load_config

    assert load_config(None) == AuditConfig()


def test_load_config_rejects_invalid_value(tmp_path: Path) -> None:
    from kt_audit import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reported_auc": 2}), encoding="utf-8")
    with pytest.raises(AuditError, match="invalid config"):
        load_config(path)
