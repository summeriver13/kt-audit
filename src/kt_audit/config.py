from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from .errors import AuditError

_TASKS = ("auto", "ranking", "classification")


@dataclass(frozen=True)
class AuditConfig:
    kc_separator: str = "|"
    temporal: bool = True
    multikc: bool = True
    alignment: bool = True
    max_examples: int = 3
    reported_auc: float | None = None
    auc_tolerance: float = 0.01
    acc_tolerance: float = 0.01
    cheap_baseline: bool = True
    degenerate: bool = True
    baseline_warn_margin: float = 0.02
    task: str = "auto"
    metric_floor: bool = True
    floor_warn_margin: float = 0.02
    coverage: bool = True
    expect_keys: str | None = None
    expect_rows: int | None = None
    reported_n: int | None = None
    reported_acc: float | None = None

    def __post_init__(self) -> None:
        if not self.kc_separator:
            raise ValueError("kc_separator must not be empty")
        if self.max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        if self.auc_tolerance < 0:
            raise ValueError("auc_tolerance must be >= 0")
        if self.acc_tolerance < 0:
            raise ValueError("acc_tolerance must be >= 0")
        if self.reported_auc is not None and not 0 <= self.reported_auc <= 1:
            raise ValueError("reported_auc must be in [0, 1]")
        if self.baseline_warn_margin < 0:
            raise ValueError("baseline_warn_margin must be >= 0")
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {', '.join(_TASKS)}")
        if self.floor_warn_margin < 0:
            raise ValueError("floor_warn_margin must be >= 0")
        for name in ("expect_rows", "reported_n"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.reported_acc is not None and not 0 <= self.reported_acc <= 1:
            raise ValueError("reported_acc must be in [0, 1]")


def load_config(path: str | Path | None) -> AuditConfig:
    if path is None:
        return AuditConfig()
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AuditError("config must be a JSON object")
    allowed = {item.name for item in fields(AuditConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise AuditError(f"unknown config keys: {', '.join(unknown)}")
    try:
        return AuditConfig(**cast(dict[str, Any], raw))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid config: {exc}") from exc
