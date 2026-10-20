from __future__ import annotations

import dataclasses
import inspect
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import kt_audit
from kt_audit import AuditConfig, Finding, Report, audit_frame, audit_path
from kt_audit.report import report_to_dict

PUBLIC_NAMES = {
    "AuditConfig",
    "AuditError",
    "Finding",
    "Report",
    "audit_frame",
    "audit_path",
    "load_config",
}


def test_public_names() -> None:
    assert set(kt_audit.__all__) == PUBLIC_NAMES


def test_version_matches_metadata() -> None:
    assert kt_audit.__version__ == version("kt-audit")


def test_audit_config_fields() -> None:
    assert [field.name for field in dataclasses.fields(AuditConfig)] == [
        "kc_separator",
        "temporal",
        "multikc",
        "alignment",
        "max_examples",
        "reported_auc",
        "auc_tolerance",
    ]


def test_finding_fields() -> None:
    assert [field.name for field in dataclasses.fields(Finding)] == [
        "check",
        "severity",
        "message",
        "count",
        "examples",
    ]


def test_report_fields() -> None:
    assert [field.name for field in dataclasses.fields(Report)] == [
        "schema_version",
        "generated_at",
        "input",
        "findings",
        "metrics",
        "verdict",
    ]


def test_audit_frame_signature() -> None:
    parameters = inspect.signature(audit_frame).parameters
    assert list(parameters) == ["df", "config"]
    assert parameters["config"].default is None


def test_audit_path_signature() -> None:
    parameters = inspect.signature(audit_path).parameters
    assert list(parameters) == ["path", "config"]
    assert parameters["config"].default is None


def test_audit_path_accepts_string(tmp_path: Path, clean_rows) -> None:
    path = tmp_path / "events.csv"
    pd.DataFrame(clean_rows).to_csv(path, index=False)
    assert audit_path(str(path)).verdict == "pass"


def test_load_config_accepts_string(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    assert kt_audit.load_config(str(path)) == AuditConfig()


def test_report_json_keys(clean_rows) -> None:
    payload = report_to_dict(audit_frame(pd.DataFrame(clean_rows)))
    assert set(payload) == {
        "schema_version",
        "generated_at",
        "verdict",
        "input",
        "metrics",
        "findings",
    }
