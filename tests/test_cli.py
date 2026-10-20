from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kt_audit import __version__
from kt_audit.cli import main


def _leaky_path() -> Path:
    return Path(__file__).parents[1] / "examples" / "leaky.csv"


def test_cli_leaky_returns_1(tmp_path: Path) -> None:
    out = tmp_path / "report"
    code = main(
        [
            "run",
            str(_leaky_path()),
            "--config",
            str(Path(__file__).parents[1] / "examples" / "audit.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 1
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "fail"
    assert (out / "report.md").exists()


def test_cli_clean_returns_0(tmp_path: Path, clean_rows) -> None:
    csv = tmp_path / "clean.csv"
    pd.DataFrame(clean_rows).to_csv(csv, index=False)
    out = tmp_path / "report"
    assert main(["run", str(csv), "--out", str(out)]) == 0
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "pass"


def test_cli_missing_input_returns_2(tmp_path: Path) -> None:
    assert main(["run", str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2


def test_cli_missing_config_returns_2(tmp_path: Path) -> None:
    assert (
        main(
            [
                "run",
                str(_leaky_path()),
                "--config",
                str(tmp_path / "missing.json"),
                "--out",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"kt-audit {__version__}"
