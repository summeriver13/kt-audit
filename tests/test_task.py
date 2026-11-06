from __future__ import annotations

from kt_audit import AuditConfig
from kt_audit.schema import prepare_frame
from kt_audit.task import PREDICTED_COLUMN, TASK_COLUMN, label_text


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "user_id": "u1",
        "item_id": "q1",
        "kc_ids": "kc_a",
        "timestamp": "2024-01-01T10:00:00",
        "split": "test",
        "y_true": 1,
        "y_pred": 0.8,
    }
    row.update(overrides)
    return row


def test_binary_numeric_is_ranking(make_frame) -> None:
    prepared = prepare_frame(make_frame([_row()]), AuditConfig())
    assert prepared.loc[0, TASK_COLUMN] == "ranking"


def test_label_column_marks_classification(make_frame) -> None:
    rows = [
        _row(y_true="Recall Failure", y_pred_label="Recall Failure"),
        _row(user_id="u2", y_true="Conceptual Gap", y_pred_label="Recall Failure"),
    ]
    prepared = prepare_frame(make_frame(rows), AuditConfig())
    assert prepared.loc[0, TASK_COLUMN] == "classification"
    assert prepared.loc[1, PREDICTED_COLUMN] == "Recall Failure"


def test_string_labels_mark_classification(make_frame) -> None:
    rows = [_row(y_true="Recall Failure"), _row(user_id="u2", y_true="Conceptual Gap")]
    prepared = prepare_frame(make_frame(rows), AuditConfig())
    assert prepared.loc[0, TASK_COLUMN] == "classification"


def test_stray_numeric_label_stays_ranking(make_frame) -> None:
    rows = [_row(y_true=1), _row(user_id="u2", y_true=2)]
    prepared = prepare_frame(make_frame(rows), AuditConfig())
    assert prepared.loc[0, TASK_COLUMN] == "ranking"


def test_config_can_force_the_task(make_frame) -> None:
    prepared = prepare_frame(make_frame([_row()]), AuditConfig(task="classification"))
    assert prepared.loc[0, TASK_COLUMN] == "classification"


def test_binary_classification_thresholds_scores(make_frame) -> None:
    rows = [
        _row(y_true=1, y_pred=0.9),
        _row(user_id="u2", y_true=0, y_pred=0.1),
    ]
    prepared = prepare_frame(make_frame(rows), AuditConfig(task="classification"))
    assert list(prepared[PREDICTED_COLUMN]) == ["1", "0"]


def test_unscored_multi_class_gets_no_label(make_frame) -> None:
    rows = [
        _row(y_true="A", y_pred=0.4),
        _row(user_id="u2", y_true="B", y_pred=0.6),
    ]
    prepared = prepare_frame(make_frame(rows), AuditConfig())
    assert bool(prepared[PREDICTED_COLUMN].isna().all())


def test_label_text_normalizes_numbers() -> None:
    assert label_text(1.0) == "1"
    assert label_text(True) == "1"
    assert label_text(" Recall Failure ") == "Recall Failure"
