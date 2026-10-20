from __future__ import annotations

import json
from pathlib import Path

from kt_audit import AuditConfig, audit_frame
from kt_audit.report import finding_to_dict, report_to_dict, report_to_markdown, write_json


def test_report_dict_is_json_serializable(tmp_path: Path, make_frame, clean_rows) -> None:
    rows = clean_rows
    rows[0]["y_pred"] = 1.2
    report = audit_frame(make_frame(rows))
    payload = report_to_dict(report)
    encoded = json.dumps(payload)
    assert "alignment" in encoded
    assert finding_to_dict(report.findings[0])["severity"] == "error"


def test_report_markdown_contains_sections(make_frame, clean_rows) -> None:
    rows = clean_rows
    rows[0]["y_pred"] = 1.2
    report = audit_frame(make_frame(rows), AuditConfig(reported_auc=0.5))
    markdown = report_to_markdown(report)
    assert "**Verdict: FAIL**" in markdown
    assert "## Findings" in markdown
    assert "## Metrics" in markdown
    assert "## Limitations" in markdown
    assert "```json" in markdown


def test_write_json(tmp_path: Path, make_frame, clean_rows) -> None:
    report = audit_frame(make_frame(clean_rows))
    path = tmp_path / "report.json"
    write_json(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "pass"
    assert payload["schema_version"] == "0.1"


def test_report_markdown_without_metrics(make_frame, clean_rows) -> None:
    report = audit_frame(make_frame(clean_rows), AuditConfig(alignment=False))
    markdown = report_to_markdown(report)
    assert "## Metrics" not in markdown
    assert "No findings" in markdown
