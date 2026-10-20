from __future__ import annotations

import json
from pathlib import Path

from .schema import Finding, Report, jsonable


def finding_to_dict(finding: Finding) -> dict[str, object]:
    return {
        "check": finding.check,
        "severity": finding.severity,
        "message": finding.message,
        "count": finding.count,
        "examples": jsonable(finding.examples),
    }


def report_to_dict(report: Report) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "generated_at": report.generated_at,
        "verdict": report.verdict,
        "input": jsonable(report.input),
        "metrics": jsonable(report.metrics),
        "findings": [finding_to_dict(finding) for finding in report.findings],
    }


def write_json(report: Report, path: Path) -> None:
    payload = json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")


def report_to_markdown(report: Report) -> str:
    lines = _header_lines(report)
    lines.extend(["", "## Findings"])
    if not report.findings:
        lines.extend(["", "No findings."])
    for finding in report.findings:
        lines.extend(_finding_lines(finding))
    lines.extend(_metrics_lines(report.metrics))
    lines.extend(_limitation_lines())
    return "\n".join(lines) + "\n"


def _header_lines(report: Report) -> list[str]:
    lines = [
        "# kt-audit report",
        "",
        f"**Verdict: {report.verdict.upper()}**",
        "",
        f"- generated: {report.generated_at}",
        f"- schema: {report.schema_version}",
        f"- rows: {report.input.get('rows')}",
        f"- users: {report.input.get('users')}",
        f"- items: {report.input.get('items')}",
    ]
    splits = report.input.get("splits")
    if isinstance(splits, dict):
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(splits.items()))
        lines.append(f"- splits: {rendered}")
    return lines


def _finding_lines(finding: Finding) -> list[str]:
    lines = [
        "",
        f"### {finding.severity.upper()} · {finding.check}",
        "",
        f"{finding.message} (count: {finding.count})",
    ]
    if finding.examples:
        payload = json.dumps(jsonable(finding.examples), indent=2, ensure_ascii=False)
        lines.extend(["", "```json", payload, "```"])
    return lines


def _metrics_lines(metrics: dict[str, dict[str, object]]) -> list[str]:
    if not metrics:
        return []
    lines = [
        "",
        "## Metrics",
        "",
        "| Split | n | AUC | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in sorted(metrics):
        metric = metrics[split]
        lines.append(
            f"| {split} | {metric.get('n')} | {_format_metric(metric.get('auc'))} "
            f"| {_format_metric(metric.get('acc'))} |"
        )
    return lines


def _limitation_lines() -> list[str]:
    return [
        "",
        "## Limitations",
        "",
        "- Findings are heuristics, not causal proof.",
        "- The multi-KC check needs `item_id`; it is skipped when missing.",
        "- The temporal check assumes timestamps are comparable within a user.",
        "- A passing report does not prove that a model is valid.",
    ]


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
