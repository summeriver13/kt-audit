from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kt_audit import AuditError
from kt_audit.cli import main
from kt_audit.compare import (
    compare_frames,
    comparison_to_dict,
    comparison_to_markdown,
)

ROWS = 120


def _rows(prediction, label_offset: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(ROWS):
        rows.append(
            {
                "user_id": f"u{index % 20}",
                "item_id": f"q{index % 10}",
                "kc_ids": "kc_a",
                "timestamp": f"2024-01-{1 + index // 20:02d}T{index % 20:02d}:00:00",
                "split": "test",
                "y_true": (index + label_offset) % 2,
                "y_pred": prediction(index),
            }
        )
    return rows


def _weak(index: int) -> float:
    return 0.45 + (index % 7) * 0.01


def _label(prediction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(ROWS):
        label = "A" if index < 80 else "B"
        rows.append(
            {
                "user_id": f"u{index % 20}",
                "item_id": f"q{index % 10}",
                "kc_ids": "kc_a",
                "timestamp": f"2024-01-{1 + index // 20:02d}T{index % 20:02d}:00:00",
                "split": "test",
                "y_true": label,
                "y_pred_label": label if prediction == "truth" else prediction,
            }
        )
    return rows


def _strong(index: int) -> float:
    return 0.1 + 0.8 * (index % 2)


def test_identical_inputs_tie(make_frame) -> None:
    frame = make_frame(_rows(_weak))
    comparison = compare_frames(frame, frame, rounds=200)
    assert comparison.n == ROWS
    assert comparison.delta == pytest.approx(0.0)
    assert comparison.verdict == "tie"


def test_detects_better_candidate(make_frame) -> None:
    a = make_frame(_rows(_weak))
    b = make_frame(_rows(_strong))
    comparison = compare_frames(a, b, rounds=200)
    assert comparison.verdict == "b_better"
    assert comparison.ci_low > 0
    assert comparison.delta > 0


def test_detects_worse_candidate(make_frame) -> None:
    a = make_frame(_rows(_strong))
    b = make_frame(_rows(_weak))
    comparison = compare_frames(a, b, rounds=200)
    assert comparison.verdict == "a_better"
    assert comparison.ci_high < 0


def test_is_deterministic(make_frame) -> None:
    a = make_frame(_rows(_weak))
    b = make_frame(_rows(_strong))
    first = compare_frames(a, b, rounds=200, seed=3)
    second = compare_frames(a, b, rounds=200, seed=3)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_rejects_different_keys(make_frame) -> None:
    a = make_frame(_rows(_weak))
    b = make_frame(_rows(_weak)[:-1])
    with pytest.raises(AuditError, match="different interaction keys"):
        compare_frames(a, b, rounds=50)


def test_rejects_label_mismatch(make_frame) -> None:
    a = make_frame(_rows(_weak))
    b = make_frame(_rows(_strong, label_offset=1))
    with pytest.raises(AuditError, match="disagree on y_true"):
        compare_frames(a, b, rounds=50)


def test_rejects_missing_labels(make_frame) -> None:
    rows = _rows(_weak)
    rows[0]["y_true"] = None
    with pytest.raises(AuditError, match="missing values"):
        compare_frames(make_frame(rows), make_frame(rows), rounds=50)


def test_rejects_non_finite_prediction_in_candidate(make_frame) -> None:
    rows = _rows(_weak)
    rows[0]["y_pred"] = float("inf")
    with pytest.raises(AuditError, match="prediction on every row"):
        compare_frames(make_frame(_rows(_weak)), make_frame(rows), rounds=50)


def test_rejects_non_finite_prediction_in_baseline(make_frame) -> None:
    rows = _rows(_weak)
    rows[0]["y_pred"] = float("inf")
    with pytest.raises(AuditError, match="prediction on every row"):
        compare_frames(make_frame(rows), make_frame(_rows(_weak)), rounds=50)


def test_rejects_zero_rounds(make_frame) -> None:
    frame = make_frame(_rows(_weak))
    with pytest.raises(AuditError, match="rounds"):
        compare_frames(frame, frame, rounds=0)


def test_rejects_single_class(make_frame) -> None:
    rows = _rows(_weak)
    for row in rows:
        row["y_true"] = 1
    with pytest.raises(AuditError, match="single class"):
        compare_frames(make_frame(rows), make_frame(rows), rounds=50)


def test_markdown_has_verdict_and_interval(make_frame) -> None:
    comparison = compare_frames(make_frame(_rows(_weak)), make_frame(_rows(_strong)), rounds=200)
    markdown = comparison_to_markdown(comparison)
    assert "**Verdict: B_BETTER**" in markdown
    assert "95% CI" in markdown


def test_cli_compare_writes_files(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame(_rows(_weak)).to_csv(a, index=False)
    pd.DataFrame(_rows(_strong)).to_csv(b, index=False)
    out = tmp_path / "cmp"
    assert main(["compare", str(a), str(b), "--rounds", "100", "--out", str(out)]) == 0
    assert (out / "comparison.json").exists()
    assert (out / "comparison.md").exists()


def test_compare_accuracy_on_label_files(make_frame) -> None:
    a = make_frame(_label("A"))
    b = make_frame(_label("truth"))
    comparison = compare_frames(a, b, rounds=200, metric="acc")
    assert comparison.metric == "acc"
    assert comparison.value_a == pytest.approx(80 / 120)
    assert comparison.value_b == pytest.approx(1.0)
    assert comparison.verdict == "b_better"


def test_compare_kappa_on_label_files(make_frame) -> None:
    comparison = compare_frames(
        make_frame(_label("A")),
        make_frame(_label("truth")),
        rounds=200,
        metric="kappa",
    )
    assert comparison.value_a == pytest.approx(0.0)
    assert comparison.value_b == pytest.approx(1.0)


def test_compare_rejects_auc_on_label_files(make_frame) -> None:
    with pytest.raises(AuditError, match="needs numeric labels"):
        compare_frames(make_frame(_label("A")), make_frame(_label("truth")), rounds=50)


def test_compare_rejects_unknown_metric(make_frame) -> None:
    frame = make_frame(_rows(_weak))
    with pytest.raises(AuditError, match="unknown metric"):
        compare_frames(frame, frame, rounds=50, metric="f1")


def test_compare_rejects_mixed_tasks(make_frame) -> None:
    rows = _rows(_weak)
    labelled = [dict(row) | {"y_pred_label": str(row["y_true"])} for row in rows]
    with pytest.raises(AuditError, match="disagree on the task"):
        compare_frames(make_frame(rows), make_frame(labelled), rounds=50)


def test_comparison_dict_names_the_metric(make_frame) -> None:
    comparison = compare_frames(
        make_frame(_rows(_weak)),
        make_frame(_rows(_strong)),
        rounds=200,
        metric="acc",
    )
    payload = comparison_to_dict(comparison)
    assert payload["metric"] == "acc"
    assert set(payload) >= {"value_a", "value_b", "delta", "verdict"}


def test_cli_compare_metric_flag(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame(_label("A")).to_csv(a, index=False)
    pd.DataFrame(_label("truth")).to_csv(b, index=False)
    out = tmp_path / "cmp"
    assert (
        main(["compare", str(a), str(b), "--metric", "acc", "--rounds", "100", "--out", str(out)])
        == 0
    )
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["metric"] == "acc"


def test_cli_compare_missing_input(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    pd.DataFrame(_rows(_weak)).to_csv(a, index=False)
    assert main(["compare", str(a), str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2
