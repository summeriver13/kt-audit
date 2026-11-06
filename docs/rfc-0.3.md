# RFC: kt-audit 0.3 — widen the metrics, keep the audience

> Status: draft, 2026-11-07. Author: Xia Jiang. Reviewer: pending.
> Scope chosen from the 2026-11 upgrade note: option A (widen the metrics inside the KT scope).

## Background

0.2 shipped three of the four failure modes of [rfc-0.2](rfc-0.2.md). The fourth, silent
degradation, has no check. The tool also judges one metric (AUC) over one label type (binary), so
four measurements of the last two months cannot be expressed at all:

1. **Silent row loss.** With `max_tokens = 3000`, 29 of 200 rows fail to parse; the output file is
   still written and the score drops 7 points. The upstream CLI additionally defaults to
   `--limit 500`, so two files in one table can have different n. A reader of a `kt-audit` report
   cannot see either fact: the tool receives one file and has no expected row set.
2. **Below the majority baseline.** The 13 models of the audited paper report accuracy between 26
   and 42 against a 50.6% majority class. Our strongest model reaches 0.282 against 0.522. A
   constant answer beats the whole table, and no check says so.
3. **Point estimates only.** `compare` bootstraps a delta; `run` prints one number per metric. Our
   own paired result −0.04 with CI [−0.10, +0.03] is a tie, while 19 / 73 / 30 wins invites the
   opposite reading.
4. **One label type.** Diagnosis audits compare four labels (Recall Failure, Conceptual Gap,
   Procedural Error, Transfer Deficit). `prepare_frame` rejects them: `y_true` must be 0 or 1.

## Scope

| Item | Answers | Default |
| --- | --- | --- |
| task inference | binary ranking or multi-class classification | derived |
| `metrics` intervals | how wide is this number at this n | info |
| `metric_floor` | does the answer beat always saying the majority class | error / warn |
| `coverage` | is every expected row here, once, under the claimed n | error / warn |
| `compare --metric` | is the delta of acc / macro-F1 / kappa larger than the noise | subcommand |

### Task inference

`AuditConfig.task` is `auto` (default), `ranking` or `classification`.

- `auto`: a `y_pred_label` column, or a non-numeric `y_true` or `y_pred`, means
  `classification`. Everything else is `ranking`, including numeric labels with a stray value
  outside {0, 1}: that file keeps the 0.2 "y_true is not binary" error, because a single [0, 1]
  score cannot express a multi-class prediction.
- `ranking` keeps today's contract: numeric `y_true` in {0, 1}, numeric `y_pred` in [0, 1], AUC.
- `classification`: `y_true` is any scalar label (string or number). A predicted label comes from
  `y_pred_label` when present, otherwise from `y_pred` if it is a label type, otherwise from
  `y_pred >= 0.5` in the binary case. AUC is reported only in `ranking`; accuracy, macro-F1 and
  kappa are reported in both.

`prepare_frame` stops rejecting non-binary `y_true`: the column stays numeric when every value
is numeric, and stays a string label otherwise. The {0, 1} check moves into
`checks/alignment._invalid_findings` and runs only for `ranking`.

How the existing checks behave in `classification`:

| Check | In `classification` |
| --- | --- |
| `temporal`, `multikc` | unchanged — they read split, time and KCs only |
| `alignment` | reports acc, macro-F1 and kappa on labels; no AUC, no [0, 1] range check on `y_pred` |
| `degenerate` | runs on `y_pred` only when it is numeric; skipped with `info` otherwise |
| `cheap_baseline` | skipped with `info`: its claim is about ranking, and a multi-class tree would need an AUC-like metric it does not have |
| `compare` | `--metric auc` requires `ranking`; the other three work in both tasks |

`report.py` gains a `## Metrics` table with n, metric, interval and floor per split.

### Metrics and intervals

`check_alignment` returns, per split:

```json
{"n": 1000, "acc": 0.282, "macro_f1": 0.284, "kappa": 0.071,
 "auc": null, "majority_class": "Procedural Error", "majority_share": 0.506,
 "majority_acc": 0.506,
 "intervals": {"acc": [0.254, 0.310], "macro_f1": [0.258, 0.310], "kappa": [0.0, 0.14]}}
```

- `intervals` is a 2000-round percentile bootstrap, seed 0, on the evaluation rows of the split.
  The constants live in `checks/alignment.py` as `BOOTSTRAP_ROUNDS` and `BOOTSTRAP_SEED`; the
  interval is not configurable in 0.3, for the same reason as the degenerate thresholds.
- A metric whose rows are fewer than `MIN_ROWS` (30) gets `null` intervals and an `info` finding.
- `kappa` is Cohen's kappa between `y_true` and the predicted label.

### `metric_floor`

New check, on by default, `AuditConfig.metric_floor` and `floor_warn_margin` (0.02).

- The floor is `majority_share` of the same rows: the accuracy of always answering the most common
  label. It is computed from the evaluated rows, not from train, because a reader compares the
  reported test number with what the test labels allow.
- Recomputed accuracy < floor: `error`. Reported `reported_acc` < floor: `error`.
- 0 ≤ accuracy − floor < `floor_warn_margin`: `warn`. Equality ties the constant answer but does
  not lose to it, and the examples of 0.2 keep their verdict.
- No `y_true`: skipped with `info`.
- The finding text names the majority label and the share, for example
  "accuracy 0.282 does not beat always answering Procedural Error (0.506)".

### `coverage`

New check, on by default, driven by new inputs:

- `--expect <file>`: a CSV or JSONL of expected keys (`user_id`, `item_id`, `timestamp` — the same
  key columns as `compare`).
- `--expect-rows <n>` and `reported_n` in the config: the n the report claims.

Findings:

| Condition | Severity |
| --- | --- |
| a key of `--expect` is absent from the input | error |
| `len(input) != reported_n` | error |
| the input holds duplicate keys | error (already today) |
| the input holds keys that `--expect` does not | warn |
| a predicted label is empty or starts with `ERROR` | error |
| `len(input) != expect-rows` | warn |

The empty-label rule is what makes a truncated LLM run visible: our failure rows wrote an empty
`diagnosis` and `reason: "ERROR: ...not_a_dict"`.

### `compare --metric`

`compare` gains `--metric {auc,acc,macro_f1,kappa}`, default `auc`. `acc`, `macro_f1` and
`kappa` need a predicted label on both sides and do not need binary labels; `auc` keeps the
binary requirement. The output field `metric` is added to `comparison.json`; `delta`, CI and
verdict keep their meaning.

## Interface changes

- `AuditConfig` gains `task`, `metric_floor`, `floor_warn_margin`, `coverage`, `expect_keys`
  (path), `expect_rows`, `reported_n`, `reported_acc`.
- `Report` gains nothing at the top level: intervals live inside `metrics[split]`.
- `schema_version` goes from `0.2` to `0.3`. Fields are added; `metrics[split]` gains keys and
  keeps `auc`, `acc`, `n`.
- CLI: `run --expect FILE`, `run --expect-rows N`, `run --no-floor`, `run --no-coverage`,
  `compare --metric NAME`.
- Exit codes do not change.

## Non-goals

- No `label_rule` check (a depth-4 tree over the file's own columns). It is the strongest finding
  of the reproduction but needs a margin calibrated on three datasets. 0.4 candidate.
- No `judge-audit`. It needs a multi-file input bundle. 0.4 candidate.
- No HTML report, no pyKT adapter (unchanged from the 0.3 roadmap).
- No change to `cheap_baseline`: it stays AUC-only, because its claim is about ranking.

## Acceptance criteria

- `pytest` green, coverage not below 0.2.0.
- `uv run kt-audit run examples/multiclass.csv` reports acc / macro-F1 / kappa with intervals and
  a `metric_floor` finding.
- `uv run kt-audit run examples/dropped_rows.csv --expect examples/multiclass.csv` triggers a
  `coverage` error for the missing keys, and a second run on the same file with `--expect-rows`
  1,001 triggers the n error.
- `uv run kt-audit compare a.csv b.csv --metric acc` prints a delta and a CI.
- Each new check has a positive and a negative test.
- README, data-contract, CHANGELOG updated; `schema_version` `0.3`.
- One case study run on the LongTutor gold labels (1,000 rows, four classes) recorded in
  `docs/case-study-longtutor.md`.

## Risks

- Multi-class opens the door to every metric. 0.3 reports accuracy, macro-F1 and kappa only.
- The floor can be high on an imbalanced set and thereby harsh. The finding text must name the
  majority label and the share, so the reader sees the arithmetic.
- `coverage` trusts `--expect`. A wrong expect file produces false errors; the finding prints the
  expect digest, not just the count.
- Bootstrap on 1,000 multi-class rows costs about a second per metric. Accept it; do not add a
  parallelism knob.
