from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from .errors import AuditError


@dataclass(frozen=True)
class AuditConfig:
    kc_separator: str = "|"
    temporal: bool = True
    multikc: bool = True
    alignment: bool = True
    max_examples: int = 3
    reported_auc: float | None = None
    auc_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if not self.kc_separator:
            raise ValueError("kc_separator must not be empty")
        if self.max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        if self.auc_tolerance < 0:
            raise ValueError("auc_tolerance must be >= 0")
        if self.reported_auc is not None and not 0 <= self.reported_auc <= 1:
            raise ValueError("reported_auc must be in [0, 1]")


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
